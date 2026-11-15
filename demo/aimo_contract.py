"""The Valio Aimo API contract in one place.

Everything the agent assumes about /api/availability and /api/price lives here, with where the
assumption comes from: the API docs (`data/openapi.json`, the product-data schema) or what we
observed on the mock. `assess()` reads one product's price and availability and returns either a
rejection code or the facts the engine uses, plus `anomalies`: every place where the response
breaks the contract. Any anomaly on a line fails the guardrail "live data consistent with the API
contract" (engine.guardrails), so a person approves. On an API that keeps the contract no anomaly
fires and nothing else changes.

Delivery date: the API reports only the earliest delivery date (the mock returns the same date,
cut-off and price for any requestedDate). The agent orders for that date and says how many days
before the deadline it arrives (`days_early`); it does not try to schedule a later delivery.

`python3 probe_contract.py` checks the live API against this contract (a few calls, writes nothing).
"""
from __future__ import annotations

import datetime as dt

AMPLE_STOCK = 1e7            # observed: 9999999999 base units (divided by the pack size in sales units) = ample
TERMINAL_UNKNOWN = 999999    # doc: Terminal items always report 999999 (quantity unknown to Valio Aimo)
LOGISTIC_MODELS = ("Stock", "Terminal")
CONTINUOUS_SALES_UNITS = ("KG",)  # sold by weight: ordered in steps of allowedLotSize, not in whole units

# id, source, the rule, the anomaly code that reports a break (None = not a response check), what then happens
CONTRACT = [
    {"id": "ONE_SCHEDULE_LINE", "source": "doc",
     "rule": "An available product has exactly one schedule line; an empty list means it cannot be delivered.",
     "anomaly": "MULTIPLE_SCHEDULE_LINES",
     "if_broken": "the amounts of all lines up to the deadline are added up, the goods count as complete on the "
                  "last of those dates, and a person approves"},
    {"id": "EARLIEST_DATE", "source": "doc + observed",
     "rule": "availableDate is the earliest delivery date and can differ from requestedDate. Observed on "
             "2026-11-14: the mock returns the same date, cut-off and price for any requestedDate.",
     "anomaly": None,
     "if_broken": "nothing to break: the agent orders for the date the API offers and states how many days "
                  "before the deadline the goods arrive"},
    {"id": "CUTOFF_FOR_AVAILABLE_DATE", "source": "doc + observed",
     "rule": "closingTime is the order cut-off for availableDate. Observed: Terminal items report a cut-off "
             "that has passed (Thu 15:00) and still the same availableDate.",
     "anomaly": "STALE_CUTOFF",
     "if_broken": "needed on that date: rejected (CUTOFF_PASSED). Needed later: kept, and a person confirms "
                  "the delivery date"},
    {"id": "TERMINAL_AMOUNT", "source": "doc",
     "rule": "Terminal (supplier) items always report availableAmount 999999: the quantity is unknown.",
     "anomaly": "UNEXPECTED_AMOUNT",
     "if_broken": "the reported amount is used as a limit, counts as unconfirmed, and a person approves"},
    {"id": "AMPLE_SENTINEL", "source": "observed",
     "rule": "9999999999 base units, scaled down to sales units, means ample stock. A Stock item reporting "
             "999999 (the Terminal value) is not expected.",
     "anomaly": "UNEXPECTED_AMOUNT",
     "if_broken": "the amount is used as a limit, counts as unconfirmed, and a person approves"},
    {"id": "LOGISTIC_MODEL", "source": "doc",
     "rule": "logisticModel is Stock or Terminal for a deliverable product.",
     "anomaly": "UNKNOWN_LOGISTIC_MODEL",
     "if_broken": "the quantity counts as unconfirmed and a person approves"},
    {"id": "DATE_ECHO", "source": "doc",
     "rule": "Responses echo the date that was asked: requestedDate (availability) and pricingDate (price).",
     "anomaly": "DATE_ECHO_MISMATCH",
     "if_broken": "every product of that response is flagged and a person approves"},
    {"id": "PRICE_FOR_ONE_UNIT", "source": "doc",
     "rule": "A price is for quantity 1.0 sales unit, so a line total is unit price x quantity (no quantity breaks).",
     "anomaly": "PRICE_QUANTITY",
     "if_broken": "the line total may be wrong, so a person approves"},
    {"id": "LOT_SIZE_IN_SALES_UNITS", "source": "doc (product schema)",
     "rule": "allowedLotSize is the order step in sales units. Count units (ST, KI, ...) are ordered in whole "
             "units, weight units (KG) in steps of the lot size. Observed: fractional lots occur only on KG.",
     "anomaly": None,
     "if_broken": "nothing to break: a rounding rule (engine._line)"},
    {"id": "CUSTOMER_INDEPENDENT", "source": "doc",
     "rule": "In the challenge the customer id does not change prices or availability.",
     "anomaly": None,
     "if_broken": "nothing to break: the customer id from policy.json is always sent, as production would need"},
]
ANOMALY_CODES = sorted({c["anomaly"] for c in CONTRACT if c["anomaly"]})


def anomaly(code, text):
    return {"code": code, "text": text}


def _ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")


def _date(line):
    return dt.date.fromisoformat(line["availableDate"][:10]) if line.get("availableDate") else None


def is_continuous(sales_unit):
    """True for sales units ordered in lot-size steps rather than whole units (sold by weight)."""
    return (sales_unit or "") in CONTINUOUS_SALES_UNITS


# --- Response level --------------------------------------------------------------------------
def check_echo(resp, endpoint, requested):
    """DATE_ECHO: anomalies for a whole response whose echoed date is not the date that was asked.
    AimoClient stores them in resp["contract_anomalies"]; absent echo fields are not checked."""
    field = {"/api/availability": "requestedDate", "/api/price": "pricingDate"}.get(endpoint)
    echo = (resp or {}).get(field) if field else None
    if echo and requested and str(echo)[:10] != str(requested)[:10]:
        return [anomaly("DATE_ECHO_MISMATCH", f"{endpoint} answered for {str(echo)[:10]}, not the requested {requested}")]
    return []


def response_anomalies(*responses):
    """Anomalies that apply to every product of these responses (recorded by AimoClient)."""
    return [a for r in responses for a in (r or {}).get("contract_anomalies") or []]


# --- Product level ---------------------------------------------------------------------------
def _schedule(lines, logistic, deadline, anomalies):
    """(line, amount): the schedule line the decision rests on and the amount available by then.
    One line is the contract. Several lines: add up the lines due by the deadline (all of them
    without a deadline); the goods are complete on the last of those dates."""
    if len(lines) == 1:
        return lines[0], lines[0].get("availableAmount") or 0
    dated = sorted(lines, key=lambda ln: ln.get("availableDate") or "9999")
    counted = [ln for ln in dated if _date(ln) and (deadline is None or _date(ln) <= deadline)] or dated[:1]
    amounts = [ln.get("availableAmount") or 0 for ln in counted]
    if any(a >= AMPLE_STOCK for a in amounts):
        amount = max(amounts)
    elif logistic == "Terminal" and all(a == TERMINAL_UNKNOWN for a in amounts):
        amount = TERMINAL_UNKNOWN
    else:
        amount = sum(amounts)
    anomalies.append(anomaly("MULTIPLE_SCHEDULE_LINES",
                             f"{len(lines)} schedule lines (" + ", ".join(
                                 f"{ln.get('availableAmount') or 0:g} on {ln.get('availableDate')}" for ln in dated)
                             + f"); the contract says one. Counted {len(counted)}, complete on "
                             + f"{counted[-1].get('availableDate')}"))
    return counted[-1], amount


def qty_status(amount, logistic, sales_unit):
    if amount >= AMPLE_STOCK:
        return "ample stock"
    if logistic == "Terminal" and amount == TERMINAL_UNKNOWN:
        return "supplier stock (unconfirmed)"
    return f"limited: {amount:g} {sales_unit}"


def assess(pr, av, now, deadline):
    """Troubleshooting rules for one product, shared by orders, proposals and the approval re-check.

    Returns (code, reason, st). code is a rejection code (NO_PRICE, NO_AVAILABILITY, SALES_BLOCK,
    NOT_DELIVERABLE, OUT_OF_STOCK, TOO_LATE, CUTOFF_PASSED) or None when the product can be ordered
    in time; st (None if the checks stopped before the schedule) holds the schedule line, amount,
    earliest date, cut-off, logistic model, days_early (days between arrival and the deadline) and
    the contract anomalies."""
    if not pr or pr.get("error") or not pr.get("catalogPrice"):
        return "NO_PRICE", "no price returned by /api/price", None
    if not av:
        return "NO_AVAILABILITY", "no availability returned", None
    lines = av.get("scheduleLines") or []
    if av.get("salesBlock") or any(ln.get("salesBlock") for ln in lines):
        return "SALES_BLOCK", "sales block: product cannot be ordered currently", None
    if not lines:
        return "NOT_DELIVERABLE", "no schedule line: cannot be delivered on any date", None
    anomalies = []
    logistic = av.get("logisticModel") or "?"
    ln, amount = _schedule(lines, logistic, deadline, anomalies)
    adate = _date(ln)
    ct = _ts(av["closingTime"]) if av.get("closingTime") else None
    if logistic not in LOGISTIC_MODELS:
        anomalies.append(anomaly("UNKNOWN_LOGISTIC_MODEL", f"logistic model {logistic!r}, expected Stock or Terminal"))
    if logistic == "Terminal" and 0 < amount < AMPLE_STOCK and amount != TERMINAL_UNKNOWN:
        anomalies.append(anomaly("UNEXPECTED_AMOUNT", f"Terminal item reports {amount:g}; the contract says {TERMINAL_UNKNOWN}"))
    if logistic == "Stock" and amount == TERMINAL_UNKNOWN:
        anomalies.append(anomaly("UNEXPECTED_AMOUNT", f"Stock item reports {TERMINAL_UNKNOWN}, the Terminal 'unknown' value"))
    qty = pr.get("quantity")
    if qty is not None and abs(float(qty) - 1.0) > 1e-9:
        anomalies.append(anomaly("PRICE_QUANTITY", f"price is for quantity {qty:g}; the contract says 1.0"))
    days_early = (deadline - adate).days if deadline and adate and adate < deadline else 0
    st = {"line": ln, "amount": amount, "adate": adate, "ct": ct, "logistic": logistic,
          "days_early": days_early, "anomalies": anomalies}
    if amount <= 0:
        return "OUT_OF_STOCK", "available amount is 0", st
    if deadline and (adate is None or adate > deadline):
        return "TOO_LATE", f"earliest delivery {adate} is after the deadline {deadline}", st
    if ct and ct < now:
        if deadline and adate and deadline > adate:
            # The cut-off for the earliest date has passed although the API still reports that date. The
            # buyer needs the goods later, so a later delivery is likely, but the API does not confirm it.
            anomalies.append(anomaly("STALE_CUTOFF",
                                     f"cut-off {ct:%a %d.%m %H:%M} for the earliest date {adate} has passed, yet the "
                                     f"API still reports that date; delivery by {deadline} is likely but not "
                                     "confirmed by the API"))
            return None, None, st
        return ("CUTOFF_PASSED",
                f"order cut-off {ct:%a %d.%m %H:%M} already passed (server time {now:%a %d.%m %H:%M}); "
                f"API still says {adate} but that delivery cannot be confirmed", st)
    return None, None, st


def days_early_text(days, arrives, deadline):
    """'arrives 4 days before it is needed' sentence, or '' when it arrives on the deadline."""
    if not days:
        return ""
    return (f"Arrives {days} day{'s' if days != 1 else ''} earlier than needed ({arrives} instead of {deadline}): "
            "Valio Aimo's API offers only its earliest delivery date, so the order is placed for that date.")
