"""Live checks (troubleshooting rules), order plans, multi-objective ranking,
guardrail decision, explanations and preference learning.

Amounts are in the order's measure: L, KG or PCS (the price API compares prices per L or
per KG). A plan carries its measure; texts for milk read exactly as before ("1000 L")."""
from __future__ import annotations

import datetime as dt
import math

import aimo_contract

STOCK_AMPLE = aimo_contract.AMPLE_STOCK            # what the API's quantity values mean: see aimo_contract.CONTRACT
TERMINAL_UNKNOWN = aimo_contract.TERMINAL_UNKNOWN
UNKNOWN_RELIABILITY = 0.90   # prior when no delivery-reliability history exists
UNCONFIRMED_QTY_FACTOR = 0.8
MEASURE_WORD = {"L": "litres", "KG": "kilos", "PCS": "pieces"}
OBJECTIVES = ["cost", "availability", "preference", "evidence", "handling", "carbon"]


def _ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")


# --- 1. Live checks on API responses ------------------------------------------------
_offer_status = aimo_contract.assess   # troubleshooting rules + contract anomalies, one place
_qty_status = aimo_contract.qty_status
QTY_ANOMALIES = ("UNEXPECTED_AMOUNT", "UNKNOWN_LOGISTIC_MODEL")  # the reported quantity cannot be trusted


def _anomaly_flags(anomalies):
    return [f"{a['code']}: {a['text']}" for a in anomalies]


def live_check(cands, prices, avail, query, reliability, carbon):
    """Apply troubleshooting rules to price/availability. Returns (offers, rejected, server_now)."""
    now = _ts(avail["timeNow"])
    pm = {x["productId"]: x for x in prices["products"]}
    am = {x["productId"]: x for x in avail["products"]}
    deadline = dt.date.fromisoformat(query["deliver_by"]) if query.get("deliver_by") else None
    offers, rejected = [], []
    shared = aimo_contract.response_anomalies(prices, avail)  # e.g. a date echo that does not match the request

    for c in cands:
        pid, pr, av, flags = c["id"], pm.get(c["id"]), am.get(c["id"]), []
        code, reason, st = _offer_status(pr, av, now, deadline)
        if code:
            rejected.append({"id": pid, "name": c["name"], "stage": "live", "code": code, "reason": reason})
            continue
        ln, amount, adate, ct = st["line"], st["amount"], st["adate"], st["ct"]

        cp = pr["catalogPrice"]
        measure = c.get("measure", "L")
        per = c["per_unit"]
        cmp_ = pr.get("comparisonPrice")
        if cmp_ and pr.get("comparisonUnit") == measure and cmp_.get("value"):
            implied = cp["value"] / cmp_["value"]
            if per is None:
                per = implied
                flags.append(f"{measure}/unit derived from comparison price")
            elif abs(implied - per) / per > 0.05:
                if measure != "L" and (c.get("per_unit_method") or "").startswith("pack weight from name"):
                    # e.g. portion butter '6 g' sold by the case: the name gives one portion, not the case
                    flags.append(f"UNIT_MISMATCH: name suggests {per:g} {measure}/unit, price API {implied:.2f} "
                                 f"{measure}/unit; using the price API")
                    per = implied
                else:
                    flags.append(f"UNIT_MISMATCH: catalog {per:g} {measure}/unit vs price API {implied:.2f} {measure}/unit")
        if not per:
            rejected.append({"id": pid, "name": c["name"], "stage": "live", "code": "NO_UNIT_CONVERSION",
                             "reason": f"cannot convert the sales unit to {measure} (no unit table, name or comparison price)"})
            continue
        if c.get("variant_note"):
            flags.append(c["variant_note"])
        anomalies = shared + st["anomalies"]
        flags += _anomaly_flags(anomalies)

        logistic = st["logistic"]
        qty_status = _qty_status(amount, logistic, ln.get("salesUnit"))
        avail_amt = amount * per if qty_status.startswith("limited") else None
        vt = cp.get("validTo")
        if vt and deadline and _ts(vt).date() < deadline:
            flags.append("price validity ends before delivery date")
        rel = reliability.get(pid)
        if rel is None:
            flags.append(f"no delivery-reliability history (assumed {UNKNOWN_RELIABILITY:.0%})")

        offers.append(dict(
            c, logistic=logistic, sales_unit=ln.get("salesUnit"), per_unit=per,
            unit_price=cp["value"], unit_price_tax=cp["valueWithTax"], list_price=(pr.get("listPrice") or {}).get("value"),
            net_price=pr.get("netPriceFound"), eur_per_unit=cp["value"] / per, available_amt=avail_amt,
            qty_status=qty_status,
            qty_confirmed=not qty_status.startswith("supplier") and not any(a["code"] in QTY_ANOMALIES for a in anomalies),
            available_date=adate.isoformat() if adate else None, days_early=st["days_early"],
            closing_time=ct.isoformat() if ct else None, reliability=rel,
            delivery_confirmed=not any(a["code"] == "STALE_CUTOFF" for a in anomalies),
            co2_per_kg=carbon.get(pid), flags=flags, anomalies=anomalies,
        ))
    return offers, rejected, now


def quote_products(ids, prices, avail, deadline, reliability):
    """Live quote per product for a proposal (any unit of measure, no order plan).
    Returns ({product_id: quote}, server_now). Same troubleshooting rules as live_check."""
    now = _ts(avail["timeNow"])
    pm = {x["productId"]: x for x in prices["products"]}
    am = {x["productId"]: x for x in avail["products"]}
    out = {}
    shared = aimo_contract.response_anomalies(prices, avail)
    for pid in ids:
        pr, av = pm.get(pid), am.get(pid)
        code, reason, st = _offer_status(pr, av, now, deadline)
        anomalies = shared + ((st or {}).get("anomalies") or [])
        stale = next((a for a in anomalies if a["code"] == "STALE_CUTOFF"), None)
        q = {"product_id": pid, "orderable": code is None, "code": code, "reason": reason,
             "reliability": reliability.get(pid), "delivery_note": f"STALE_CUTOFF: {stale['text']}" if stale else None,
             "anomalies": [a["code"] for a in anomalies], "days_early": (st or {}).get("days_early") or 0}
        cp = (pr or {}).get("catalogPrice")
        if cp:
            q.update(unit_price=cp["value"], unit_price_tax=cp["valueWithTax"])
            cmp_ = pr.get("comparisonPrice") or {}
            if cmp_.get("value"):
                q.update(comparison_price=cmp_["value"], comparison_unit=pr.get("comparisonUnit"))
        if st:
            ln = st["line"]
            q.update(sales_unit=ln.get("salesUnit"), logistic=st["logistic"],
                     available_date=st["adate"].isoformat() if st["adate"] else None,
                     order_cutoff=st["ct"].isoformat() if st["ct"] else None,
                     stock=_qty_status(st["amount"], st["logistic"], ln.get("salesUnit")) if st["amount"] > 0 else "none")
        out[pid] = q
    return out, now


def revalidate(lines, prices, avail, deadline):
    """Commit-time check of an option a person approved (public plan lines) against a fresh
    quote. Returns (changes, fresh, server_now): `changes` are the reasons the quote no longer
    holds (empty = still valid: commit), `fresh` the current price and delivery per line.
    A lower price is not a change; the order is committed at the fresh price."""
    now = _ts(avail["timeNow"])
    pm = {x["productId"]: x for x in prices["products"]}
    am = {x["productId"]: x for x in avail["products"]}
    shared = aimo_contract.response_anomalies(prices, avail)
    changes, fresh = [], []
    for ln in lines:
        pid, label = ln["productId"], f"{ln['productId']} {ln['name'][:40]}"
        code, reason, st = _offer_status(pm.get(pid), am.get(pid), now, deadline)
        if code:
            changes.append(f"{label}: {reason}")
            fresh.append({"productId": pid, "code": code, "reason": reason})
            continue
        price, unit = pm[pid]["catalogPrice"]["value"], ln.get("salesUnit") or "unit"
        if price > ln["unitPrice"] + 0.005:
            changes.append(f"{label}: price rose from EUR {ln['unitPrice']:.2f} to EUR {price:.2f} per {unit}")
        qty_status = _qty_status(st["amount"], st["logistic"], st["line"].get("salesUnit"))
        if qty_status.startswith("limited") and st["amount"] < ln["quantity"]:
            changes.append(f"{label}: stock fell to {st['amount']:g} {unit}; the order needs {ln['quantity']:g}")
        # a contract anomaly the person did not see when deciding is a change (orders quoted before
        # aimo_contract.py carry the stale cut-off only as a DELIVERY_UNCONFIRMED flag)
        quoted = set(ln.get("anomalies") or [])
        if any(str(f).startswith("DELIVERY_UNCONFIRMED") for f in ln.get("flags") or []):
            quoted.add("STALE_CUTOFF")
        for a in shared + st["anomalies"]:
            if a["code"] in quoted:
                continue
            quoted.add(a["code"])
            changes.append(f"{label}: delivery by {deadline} is no longer confirmed (the order cut-off passed)"
                           if a["code"] == "STALE_CUTOFF" else f"{label}: {a['code']}: {a['text']}")
        fresh.append({"productId": pid, "code": None, "unitPrice": price, "quoted_unitPrice": ln["unitPrice"],
                      "quantity": ln["quantity"], "lineTotal": round(price * ln["quantity"], 2),
                      "availableDate": st["adate"].isoformat() if st["adate"] else None,
                      "daysEarly": st["days_early"],
                      "orderCutOff": st["ct"].isoformat() if st["ct"] else None, "stock": qty_status})
    return changes, fresh, now


# --- 2. Order plans -------------------------------------------------------------
def _line(o, amount):
    """One order line for `amount` (in the offer's measure). allowedLotSize is the order step in sales
    units (product schema): count units (ST, KI, ...) are rounded up to whole units first, units sold by
    weight (KG) only to the next lot step, so 2.35 kg with a 0.01 step stays 2.35 kg."""
    units = amount / o["per_unit"]
    if not aimo_contract.is_continuous(o.get("sales_unit") or o.get("salesUnit")):
        units = math.ceil(units - 1e-9)
    lot = o.get("allowedLotSize") or 1
    units = round(math.ceil(units / lot - 1e-9) * lot, 6)  # respect allowedLotSize
    return {
        "id": o["id"], "name": o["name"], "units": round(units, 2), "sales_unit": o["sales_unit"],
        "amount": round(units * o["per_unit"], 2), "unit_price": o["unit_price"],
        "total": round(units * o["unit_price"], 2), "total_tax": round(units * o["unit_price_tax"], 2),
        "_o": o,
    }


def _plan(lines, need, kind):
    amount = sum(ln["amount"] for ln in lines)
    os_ = [ln["_o"] for ln in lines]
    rel = [(o["reliability"] if o["reliability"] is not None else UNKNOWN_RELIABILITY)
           * (1 if o["qty_confirmed"] else UNCONFIRMED_QTY_FACTOR) for o in os_]
    co2 = [ln["amount"] * ln["_o"]["kg_per_measure"] * ln["_o"]["co2_per_kg"] for ln in lines
           if ln["_o"]["co2_per_kg"] and ln["_o"].get("kg_per_measure")]
    total = round(sum(ln["total"] for ln in lines), 2)
    return {
        "kind": kind, "lines": lines, "measure": os_[0].get("measure", "L"), "amount": round(amount, 2), "need": need,
        "shortfall": round(max(0.0, need - amount), 2), "coverage": min(1.0, amount / need),
        "total": total, "total_tax": round(sum(ln["total_tax"] for ln in lines), 2),
        "eur_per_unit": total / amount if amount else float("inf"),
        "availability": min(rel),
        "preference": sum(ln["amount"] * ln["_o"]["pref_match"] for ln in lines) / amount,
        "evidence": min(o["evidence_strength"] for o in os_),
        "packs": sum(ln["units"] * ln["_o"]["packs_per_unit"] for ln in lines),
        "co2_kg": round(sum(co2), 1) if len(co2) == len(lines) else None,
    }


def build_plans(offers, query):
    need = float(query["quantity"])
    plans = [_plan([_line(o, need)], need, "single")
             for o in offers if o["available_amt"] is None or o["available_amt"] >= need]
    # Split plan: when nothing single-sourced covers the need, or none matches stated preferences.
    if offers and (not plans or all(p["preference"] < 0.999 for p in plans)):
        lines, remaining = [], need
        for o in sorted(offers, key=lambda o: (-o["pref_match"], o["eur_per_unit"])):
            if remaining <= 1e-9:
                break
            take = remaining if o["available_amt"] is None else min(remaining, o["available_amt"])
            if take > 0:
                ln = _line(o, take)
                if o["available_amt"] is not None and ln["amount"] > o["available_amt"] + 1e-6:
                    ln = _line(o, o["available_amt"] - o["per_unit"] + 1e-6)  # never exceed stock
                lines.append(ln)
                remaining -= ln["amount"]
        if len(lines) > 1 or (lines and remaining > 1e-6):
            plans.append(_plan(lines, need, "split" if remaining <= 1e-6 else "partial"))
    return plans


# --- 3. Multi-objective ranking -------------------------------------------------
def effective_weights(prefs, query):
    w = dict(prefs["weights"])
    for k in query.get("optimize_for") or []:  # objectives named in the request get a boost
        if k in w:
            w[k] *= 1.5
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def utilities(p, plans):
    def ratio_min(key):
        vals = [q[key] for q in plans if q[key]]
        return (min(vals) / p[key]) if p[key] and vals else 1.0
    # Extra pieces in a larger pack should not make buying more look cheaper.
    cost = lambda q: q["total"] / min(q["amount"], q["need"])
    handling = lambda q: q["packs"] / min(q["amount"], q["need"])
    return {
        "cost": min(map(cost, plans)) / cost(p) if p["total"] else 1.0,
        "availability": p["availability"],
        "preference": p["preference"],
        "evidence": p["evidence"],
        "handling": min(map(handling, plans)) / handling(p),
        "carbon": ratio_min("co2_kg") if p["co2_kg"] else 1.0,
    }


def rank(plans, weights):
    for p in plans:
        p["u"] = utilities(p, plans)
        p["score"] = round(p["coverage"] * sum(weights[k] * p["u"][k] for k in OBJECTIVES), 4)
    plans.sort(key=lambda p: -p["score"])
    keys = ["cost", "availability", "preference", "handling"]
    for p in plans:
        p["pareto"] = not any(
            all(q["u"][k] >= p["u"][k] for k in keys) and q["coverage"] >= p["coverage"]
            and (any(q["u"][k] > p["u"][k] for k in keys) or q["coverage"] > p["coverage"])
            for q in plans if q is not p)
        p["tags"] = []
    def tag(name, key, fn):
        vals = [key(p) for p in plans]
        best = fn(vals)
        if len(plans) > 1 and sum(abs(v - best) < 1e-9 for v in vals) == 1:  # only a unique best earns a tag
            fn(plans, key=key)["tags"].append(name)
    if plans:
        tag("lowest unit price", lambda p: p["eur_per_unit"], min)
        tag("most reliable", lambda p: p["availability"], max)
        tag("fewest packs", lambda p: p["packs"] / min(p["amount"], p["need"]), min)
        if any(p["preference"] < 1 for p in plans):
            tag("best preference match", lambda p: p["preference"], max)
    return plans


# --- 4. Guardrails --------------------------------------------------------------
def price_ceiling(policy, product_type):
    """Per-product-type price ceiling from policy.json (None if the buyer set none)."""
    ceilings = policy.get("max_price_per_unit_eur")
    if ceilings is None and "max_price_per_litre_eur" in policy:  # policy files from before 2026-11-14.2
        ceilings = {"milk": policy["max_price_per_litre_eur"]}
    return (ceilings or {}).get(product_type)


def guardrails(plan, query, policy, interp_issues, unreviewed_rules=(), unreviewed_type=None, duplicates=None):
    """Every check is explicit so the audit trail shows exactly why the agent may / may not act.
    unreviewed_rules: diet codes in the query whose rule no person has reviewed yet
    (rules/diet_overrides.json); unreviewed_type: the product, if it is not a reviewed type
    in rules/product_types.json. Relying on either always needs a human. duplicates: open
    orders for the same product and delivery date (None = not checked, e.g. a dry run)."""
    if plan is None:
        return "ESCALATE_NO_OPTION", [(False, "at least one feasible option exists")]
    lines = plan["lines"]
    m = plan["measure"]
    ceiling = price_ceiling(policy, query.get("product_type"))
    prefs_stated = any((query.get("preferences") or {}).get(k) for k in ("brand", "fat_level", "organic"))
    min_rel = min((ln["_o"]["reliability"] if ln["_o"]["reliability"] is not None else -1) for ln in lines)
    checks = [
        (plan["shortfall"] == 0, f"full quantity covered ({plan['amount']:g} {m} of {plan['need']:g} {m})"),
        (plan["total"] <= policy["auto_approve_max_total_eur"],
         f"spend EUR {plan['total']:,.2f} within autonomous limit EUR {policy['auto_approve_max_total_eur']:,.0f}"),
        ((ceiling is not None and plan["eur_per_unit"] <= ceiling),
         f"price EUR {plan['eur_per_unit']:.2f}/{m} within ceiling EUR {ceiling:.2f}/{m}" if ceiling is not None
         else f"price ceiling set in policy.json for this product (none; price EUR {plan['eur_per_unit']:.2f}/{m})"),
        (len(lines) == 1 or policy["allow_split_orders"], f"single product line ({len(lines)} line(s))"),
        (not prefs_stated or plan["preference"] >= 0.999,
         (f"matches your stated preferences ({_stated(query)}): "
          f"{plan['preference']:.0%} of {MEASURE_WORD.get(m, m)} do") if prefs_stated else "no preferences stated"),
        (plan["evidence"] >= 1.0 or not policy["require_structured_diet_evidence"],
         "dietary claim verified by structured GS1 data"),
        (all(ln["_o"]["qty_confirmed"] for ln in lines) or policy["allow_unconfirmed_supplier_stock"],
         "quantity confirmed by Valio Aimo stock"),
        (min_rel >= policy["min_delivery_reliability"],
         f"delivery reliability >= {policy['min_delivery_reliability']:.0%} "
         f"(lowest: {'unknown' if min_rel < 0 else format(min_rel, '.0%')})"),
        (not any(f.startswith("UNIT_MISMATCH") for ln in lines for f in ln["_o"]["flags"]),
         "unit conversion cross-checked against price API"),
        (not interp_issues, "request interpretation unambiguous"
         + ("" if not interp_issues else ": " + "; ".join(interp_issues))),
    ]
    if unreviewed_rules:  # only shown when it matters, so existing audit trails stay unchanged
        checks.append((False, "diet rule reviewed by a person (not yet: " + ", ".join(unreviewed_rules) + ")"))
    if unreviewed_type:
        checks.append((False, f"product type reviewed by a person (not yet: '{unreviewed_type}'; "
                              "identified by Claude from catalogue search)"))
    broken = [f"{ln['_o']['name'][:40]}: {a['code']}: {a['text']}" for ln in lines for a in ln["_o"].get("anomalies") or []]
    if broken:  # aimo_contract.py: the response breaks the documented API contract (e.g. a stale cut-off)
        checks.append((False, "live data consistent with the API contract (not: " + "; ".join(broken) + ")"))
    variants = [ln["_o"]["name"] for ln in lines if ln["_o"].get("variant_note")]
    if variants:
        checks.append((False, "every line is the requested product itself (variant: " + "; ".join(variants) + ")"))
    if duplicates is not None:
        hours = policy.get("duplicate_order_window_hours")
        checks.append((not duplicates, f"no other open order for the same product and delivery date within {hours:g} h"
                       + (" (possible duplicate: " + "; ".join(duplicates) + ")" if duplicates else "")))
    return ("AUTO_ORDER" if all(ok for ok, _ in checks) else "NEEDS_HUMAN_APPROVAL"), checks


def _stated(query):
    p = query.get("preferences") or {}
    parts = [f"brand {p['brand']}" if p.get("brand") else "", p.get("fat_level") or "",
             "organic" if p.get("organic") else ""]
    return ", ".join(x for x in parts if x)


def select_for_display(plans, n=5):
    """Top-ranked plans, but always include Pareto-optimal and tagged ones so the
    human sees every genuine trade-off, not just near-duplicates of the winner."""
    keep = [p for p in plans if p["pareto"] or p["tags"]]
    for p in plans:
        if len(keep) >= n:
            break
        if p not in keep:
            keep.append(p)
    return sorted(keep, key=lambda p: -p["score"])


# --- 5. Explanation -------------------------------------------------------------
DIET_LABEL = {"FREE_FROM_LACTOSE": "Lactose-free", "LOW_LACTOSE": "Low-lactose"}


def _short(p):
    name = p["lines"][0]["name"]
    return f"'{name}'" + (f" + {len(p['lines']) - 1} more" if len(p["lines"]) > 1 else "")


def explain(plan, plans, n_offers, query, labels=None):
    """labels: optional {diet code: label} (from catalog.DIET_RULES)."""
    if plan is None:
        return ["No option satisfies the hard requirements; see what was ruled out and the suggested next step."]
    o = plan["lines"][0]["_o"]
    out = []
    for d in query.get("diet") or []:
        name = (labels or {}).get(d, "").capitalize() or DIET_LABEL.get(d, d)
        ev = "; ".join(sorted({ln["_o"]["evidence"] for ln in plan["lines"]}))
        if "CONFLICT" in ev:
            out.append(f"{name}: the product data contradicts itself ({ev}), so a person must confirm.")
        else:
            out.append(f"{name}, verified by {ev}.")
    cut = (o["closing_time"] or "")[:16].replace("T", " ")
    m = plan["measure"]
    cut_text = ("" if not cut else f"; order cut-off {cut} is still open." if o.get("delivery_confirmed", True)
                else f"; the cut-off {cut} for that date has passed, so the date is not confirmed.")
    out.append(f"Covers {plan['amount']:,.6g} {m} of the {plan['need']:,.6g} {m} needed and arrives {o['available_date']}"
               + (f" (deadline {query['deliver_by']})" if query.get("deliver_by") else "")
               + (cut_text or "."))
    others = [p for p in plans if p is not plan]
    if others:
        nxt = min(others, key=lambda p: p["eur_per_unit"])
        lo, hi = min(p["eur_per_unit"] for p in plans), max(p["eur_per_unit"] for p in plans)
        rank = "the cheapest" if plan["eur_per_unit"] <= nxt["eur_per_unit"] else "not the cheapest"
        out.append(f"EUR {plan['eur_per_unit']:.2f}/{m} is {rank} of {len(plans)} feasible options "
                   f"(EUR {lo:.2f}-{hi:.2f}/{m}). Next cheapest: {_short(nxt)} at EUR {nxt['eur_per_unit']:.2f}/{m} "
                   f"({nxt['total'] - plan['total']:+,.2f} EUR in total).")
    else:
        out.append(f"EUR {plan['eur_per_unit']:.2f}/{m}; this is the only feasible option.")
    early = sorted({ln["_o"].get("days_early") or 0 for ln in plan["lines"]})
    if early[-1] and len(early) == 1:
        out.append(aimo_contract.days_early_text(early[0], o["available_date"], query.get("deliver_by")))
    elif early[-1]:
        out.append(f"The lines arrive {early[0]}-{early[-1]} days earlier than needed (deadline {query.get('deliver_by')}): "
                   "Valio Aimo's API offers only its earliest delivery date, so each line is ordered for that date.")
    for ln in plan["lines"]:
        lo_ = ln["_o"]
        where = "In Valio Aimo stock" if lo_["logistic"] == "Stock" else "Delivered from the supplier's warehouse"
        rel = "no delivery history" if lo_["reliability"] is None else f"{lo_['reliability']:.0%} delivery reliability"
        out.append(f"{where} ({lo_['qty_status']}), {rel}"
                   + (f" [{lo_['name']}]." if len(plan["lines"]) > 1 else "."))
    alts = sorted(others, key=lambda p: (not (p["pareto"] or p["tags"]), -p["score"]))
    names = {"availability": "availability", "preference": "your stated preferences", "handling": "handling (fewer packs)"}
    for p in alts[:3]:
        better = [names[k] for k in names if p["u"][k] > plan["u"][k] + 1e-6]
        if better:
            out.append(f"Trade-off: {_short(p)} is better on {' and '.join(better)} "
                       f"but costs {p['total'] - plan['total']:+,.2f} EUR.")
    for ln in plan["lines"]:
        if ln["_o"].get("variant_note"):
            out.append(f"Check: '{ln['name']}' is a {ln['_o']['variant_note']}.")
    if plan["co2_kg"] is not None:
        out.append(f"About {plan['co2_kg']:,.0f} kg CO2e (category average, so it does not separate the options).")
    return out


# --- 6. Preference learning -----------------------------------------------------
def learn_from_choice(prefs, u_recommended, u_chosen, lr=0.5):
    """Human picked another plan -> move weights toward the objectives where the chosen
    plan is better (perceptron-style pairwise update), then renormalise."""
    old = dict(prefs["weights"])
    w = {k: max(0.01, old[k] + lr * (u_chosen[k] - u_recommended[k])) for k in old}
    s = sum(w.values())
    prefs["weights"] = {k: round(v / s, 4) for k, v in w.items()}
    return old, prefs["weights"]
