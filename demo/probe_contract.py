"""Check the live Valio Aimo API against the contract in aimo_contract.py.

    python3 probe_contract.py                  # default sample, 3 requested dates: 6 API calls, writes nothing
    python3 probe_contract.py 464306 468201    # your own product IDs
    python3 probe_contract.py --save tests/fixtures/mock_api_YYYY-MM-DD.json   # also record the raw responses
    python3 probe_contract.py --strict         # exit code 1 when any contract item is broken

The report lists every contract item with what the probe saw: kept, broken (with the anomaly code
and the products) or not exercised by the sample, and whether requestedDate changes the answer
(on the mock, 2026-11-14, it does not). Run it before a demo and whenever the API changes. The API
docs forbid bulk queries, so keep the sample small.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import sys

import aimo_contract as ac

# Stock with ample stock, Stock with a real limit (sold by KG), two Terminal items, one not deliverable
DEFAULT_SAMPLE = ["464306", "461302", "468201", "474421", "460721"]


def record(client, ids, dates):
    """Raw price + availability responses per requested date (2 calls per date)."""
    return {"recorded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "products": list(ids),
            "dates": {d: {"availability": client.availability(ids, d), "price": client.price(ids, d)} for d in dates}}


def analyse(rec):
    """Pure function over recorded responses (no network): {contract id: (status, detail)} plus the
    per-date answers. status is 'kept', 'broken', 'not exercised' or 'info'."""
    broken = collections.defaultdict(set)    # anomaly code -> product ids ('*' = whole response)
    answers = collections.defaultdict(dict)  # product -> date -> (availableDate, closingTime, price, lines)
    stale, lines_seen, terminal_seen, logistic_seen = set(), 0, 0, 0
    for date, r in sorted(rec["dates"].items()):
        av, pr = r["availability"], r["price"]
        for a in ac.check_echo(av, "/api/availability", date) + ac.check_echo(pr, "/api/price", date):
            broken[a["code"]].add("*")
        now = ac._ts(av["timeNow"])
        deadline = dt.date.fromisoformat(date)
        pm = {p["productId"]: p for p in pr.get("products") or []}
        for p in av.get("products") or []:
            pid, sl = p["productId"], p.get("scheduleLines") or []
            _, _, st = ac.assess(pm.get(pid), p, now, deadline)
            for a in (st or {}).get("anomalies") or []:
                broken[a["code"]].add(pid)
            if sl:
                lines_seen += 1
                logistic_seen += 1
                terminal_seen += p.get("logisticModel") == "Terminal"
                if p.get("closingTime") and ac._ts(p["closingTime"]) < now:  # independent of the deadline
                    stale.add(pid)
            price = ((pm.get(pid) or {}).get("catalogPrice") or {}).get("value")
            answers[pid][date] = ([x.get("availableDate") for x in sl], p.get("closingTime"), price, len(sl))
    broken["STALE_CUTOFF"] |= stale
    same = {pid: len({json.dumps(v) for v in by_date.values()}) == 1 for pid, by_date in answers.items()}
    exercised = {"ONE_SCHEDULE_LINE": lines_seen, "CUTOFF_FOR_AVAILABLE_DATE": lines_seen, "TERMINAL_AMOUNT": terminal_seen,
                 "AMPLE_SENTINEL": lines_seen, "LOGISTIC_MODEL": logistic_seen, "DATE_ECHO": len(rec["dates"]),
                 "PRICE_FOR_ONE_UNIT": len(answers)}
    report = {}
    for c in ac.CONTRACT:
        cid, code = c["id"], c["anomaly"]
        if cid == "EARLIEST_DATE":
            ignored = [pid for pid, s in same.items() if s]
            report[cid] = ("info", f"requestedDate changed nothing for {len(ignored)} of {len(same)} products "
                                   f"over {len(rec['dates'])} dates" if len(rec["dates"]) > 1 else "only one date probed")
        elif code is None:
            report[cid] = ("info", "not a response check")
        elif broken.get(code) and (code != "UNEXPECTED_AMOUNT" or _amount_item(cid, broken[code], rec)):
            report[cid] = ("broken", f"{code}: " + ", ".join(sorted(broken[code])))
        elif not exercised.get(cid):
            report[cid] = ("not exercised", "the sample has no product this applies to")
        else:
            report[cid] = ("kept", "")
    return {"report": report, "answers": answers, "requested_date_ignored": all(same.values()) if same else None}


def _amount_item(cid, pids, rec):
    """UNEXPECTED_AMOUNT belongs to TERMINAL_AMOUNT (Terminal items) or AMPLE_SENTINEL (Stock items)."""
    models = {p["productId"]: p.get("logisticModel") for r in rec["dates"].values()
              for p in r["availability"].get("products") or []}
    want = "Terminal" if cid == "TERMINAL_AMOUNT" else "Stock"
    return any(models.get(pid) == want for pid in pids)


def print_report(result, rec):
    print(f"API contract probe · {len(rec['products'])} products · requested dates {', '.join(sorted(rec['dates']))}\n")
    mark = {"kept": "✓", "broken": "✗", "not exercised": "–", "info": "·"}
    for c in ac.CONTRACT:
        status, detail = result["report"][c["id"]]
        print(f"{mark[status]} {c['id']:<26} {status:<14} {detail}")
        if status == "broken":
            print(f"    {'':<26} the agent then: {c['if_broken']}")
    print("\nAnswers per product (availableDate, closingTime, price) by requested date:")
    for pid, by_date in sorted(result["answers"].items()):
        vals = {json.dumps(v[:3]) for v in by_date.values()}
        first = next(iter(by_date.values()))
        print(f"  {pid}: " + ("same for every date: " if len(vals) == 1 else "differs by date: ")
              + (f"{first[0]} {first[1]} EUR {first[2]}" if len(vals) == 1
                 else "; ".join(f"{d} -> {v[0]} {v[1]} EUR {v[2]}" for d, v in sorted(by_date.items()))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help=f"product IDs (default: {' '.join(DEFAULT_SAMPLE)})")
    ap.add_argument("--dates", nargs="+", help="requested dates YYYY-MM-DD (default: in 2, 4 and 6 days)")
    ap.add_argument("--save", metavar="PATH", help="write the raw responses as a test fixture")
    ap.add_argument("--strict", action="store_true", help="exit 1 when any contract item is broken")
    args = ap.parse_args()

    import agent  # loads .env
    from aimo_api import AimoClient
    agent.load_dotenv()
    today = dt.date.today()
    dates = args.dates or [(today + dt.timedelta(days=n)).isoformat() for n in (2, 4, 6)]
    rec = record(AimoClient(), args.ids or DEFAULT_SAMPLE, dates)
    result = analyse(rec)
    print_report(result, rec)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print(f"\nraw responses saved to {args.save}")
    if args.strict and any(s == "broken" for s, _ in result["report"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
