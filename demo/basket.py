"""Multi-line orders for named dishes ("source the ingredients for ..."), intent `source`.

  recipe (Claude, one per named dish, quantities per portion)
    -> one line per ingredient, split per diet group where the needs differ (34 regular + 6 allergy)
    -> catalogue search, Claude's identity verdicts, allergen verdicts per group (rules)
    -> live price and availability for the candidates only
    -> a traffic light per line: green = the usual product (order guide) can be ordered in time
    -> red lines get alternatives in tiers, never relaxing a hard requirement:
         best    same product, another SKU (pack size and unit price as usual): nothing relaxed
         better  same product, but pack size, unit price or form differs: a preference is relaxed
         good    another ingredient with the same function in the dish (Claude suggests, rules check)
         later   the safe product, delivered later for this group (a date is relaxed)
         warn    allergen evidence is missing: offered only with an acknowledgement and a label check
       a known violation (CONTAINS milk for a milk allergy) is never offered, only listed as ruled out
    -> one preference question (price / delivery / stock / fit) re-ranks within the tiers only
    -> one confirmation for the whole order, re-validated live, every line's tier and gate audited.

The LLM writes recipes, judges identity and suggests substitutes; prices, stock, allergen
verdicts, tiers, gates and ordering are deterministic rules on catalogue and live data.
"""
from __future__ import annotations

import collections
import copy
import datetime as dt
import json
import math
import os
import re
import secrets

import catalog
import engine
import nlu
from aimo_api import AimoApiError, AimoClient

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO_DIR = os.path.join(HERE, "scenarios")
SEARCH_LIMIT = 12          # catalogue candidates per ingredient
SUB_SEARCH_LIMIT = 8       # catalogue candidates per substitute idea
MAX_PER_TIER = 3
PACK_TOLERANCE = 0.25      # best: pack size within +-25 % of the usual product
PRICE_TOLERANCE = 0.05     # best: unit price at most 5 % above the usual product
PREFERENCE_BOOST = 2.0     # the buyer's one answer doubles that objective's weight (ranking only)
STOCK_WEIGHT = 0.10

TIERS = ["best", "better", "good", "later", "warn"]
TIER_INFO = {
    "as_requested": {"label": "as requested", "gate": "none", "learnable": None,
                     "text": "the usual product, within the agent's authority"},
    "best": {"label": "Best", "gate": "none", "learnable": True,
             "text": "same product, another SKU: nothing relaxed"},
    "better": {"label": "Better", "gate": "one_click", "learnable": True,
               "text": "same product, a preference relaxed (pack size, unit price or form)"},
    "good": {"label": "Good", "gate": "tradeoff", "learnable": True,
             "text": "another ingredient with the same function in the dish"},
    "later": {"label": "Later", "gate": "one_click", "learnable": True,
              "text": "the safe product, delivered later for this group"},
    "warn": {"label": "Incomplete evidence", "gate": "acknowledge", "learnable": False,
             "text": "allergen data missing: you confirm and check the label on arrival"},
}
PREFERENCE_OPTIONS = {"price": "cost", "delivery": "availability", "stock": "stock", "fit": "preference"}
PREFERENCE_QUESTION = "One question for the replacements: what matters most to you?"
STATUS_TEXT = {
    "AWAITING_CONFIRMATION": "quoted, waiting for one confirmation of the whole order",
    "CONFIRMED_BY_HUMAN": "confirmed by a person",
    "STALE_QUOTE": "the quote changed before confirmation, nothing was ordered",
    "CANCELLED_BY_HUMAN": "cancelled by a person",
}


def _load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return copy.deepcopy(default)


ORDER_GUIDE = _load_json(os.path.join(catalog.RULES_DIR, "order_guide.json"), {"items": {}}).get("items", {})


def _norm_ingredient(name):
    """'Fresh carrots (washed), sliced' -> 'carrots': no brackets, no text after a comma, no 'fresh'."""
    n = re.sub(r"\(.*?\)", " ", (name or "").lower()).split(",")[0]
    n = re.sub(r"\b(fresh|raw|whole)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def usual_product(ingredient):
    """(product id, note) from the buyer's order guide for an ingredient name, or (None, None).
    The name must equal the key or an alias (after _norm_ingredient): 'sweet potato' is not 'potato'."""
    name = _norm_ingredient(ingredient)
    for key, item in ORDER_GUIDE.items():
        if name in {_norm_ingredient(a) for a in [key] + list(item.get("aliases") or [])}:
            return item["product_id"], item.get("note")
    return None, None


def load_simulation(name):
    """A fault-injection fixture from scenarios/<name>.json (or None)."""
    if not name:
        return None
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"unknown simulation {name!r}")
    path = os.path.join(SCENARIO_DIR, name + ".json")
    if not os.path.exists(path):
        raise ValueError(f"unknown simulation {name!r}")
    sim = _load_json(path, {})
    sim["name"] = name
    return sim


def apply_simulation(avail, sim):
    """Shift availableDate for the fixture's products, after the real response was received and
    hashed. Returns the product IDs that were changed."""
    if not sim:
        return []
    ids, days, changed = set(sim.get("products") or []), int(sim.get("delay_days") or 0), []
    for p in avail.get("products") or []:
        if p.get("productId") in ids:
            for ln in p.get("scheduleLines") or []:
                if ln.get("availableDate"):
                    ln["availableDate"] = (dt.date.fromisoformat(ln["availableDate"])
                                           + dt.timedelta(days=days)).isoformat()
                    changed.append(p["productId"])
    return sorted(set(changed))


# --- Groups, quantities, measures -------------------------------------------------------
def build_groups(portions, diet_groups, everyone_diet):
    """[{key, name, portions, diet, allergy}]: the regular group first, then each diet sub-group."""
    labels = lambda codes: ", ".join(catalog.DIET_RULES[d]["label"] for d in codes)
    total = portions or 0
    subs = [g for g in diet_groups or [] if g.get("diet")]
    base = max(0.0, total - sum(g["portions"] for g in subs)) if total else None
    groups = [{"key": "all", "name": "regular" + (f" ({labels(everyone_diet)})" if everyone_diet else ""),
               "portions": base, "diet": list(everyone_diet), "allergy": False}]
    for i, g in enumerate(subs, 1):
        diet = sorted(set(everyone_diet) | set(g["diet"]))
        groups.append({"key": f"g{i}", "name": f"{g['portions']:g} portions {labels(diet)}"
                       + (" (allergy)" if g.get("allergy") else ""),
                       "portions": g["portions"], "diet": diet, "allergy": bool(g.get("allergy")),
                       "quote": g.get("quote"), "read_by": g.get("read_by")})
    return groups


def _measure_for(p, preferred):
    """The measure to order a product in: the ingredient's own (KG/L) when the product can be
    converted to it, else the other comparison unit (1 L counted as 1 kg, flagged)."""
    order = [preferred] + [m for m in ("KG", "L") if m != preferred]
    for m in order:
        if m == "KG" or catalog.amount_per_sales_unit(p, m)[0]:
            return m
    return preferred


def _need(portions, ing):
    if not portions or not ing.get("qty_per_portion") or ing.get("unit") not in ("KG", "L", "PCS"):
        return None
    return round(portions * ing["qty_per_portion"], 3)


# --- Options (one product for one line) ---------------------------------------------------
ALLERGEN_WORDS = {"AM": "milk protein", "ML": "lactose", "AW": "gluten", "NR": "rye", "GB": "barley", "UW": "wheat",
                  "AE": "egg", "AY": "soy", "AN": "tree nuts", "AP": "peanuts", "AF": "fish", "AC": "crustaceans",
                  "UM": "molluscs", "AS": "sesame", "BC": "celery", "BM": "mustard", "AU": "sulphites", "NL": "lupin"}
LOOKALIKE_CLAIMS = {"FREE_FROM_LACTOSE": "lactose-free", "LOW_LACTOSE": "low-lactose", "VEGAN": "vegan",
                    "FREE_FROM_GLUTEN": "labelled gluten-free", "FREE_FROM_MILK_PROTEIN": "milk-protein-free"}


def plain_reason(p, reasons):
    """'contains milk protein, although lactose-free': why a product was ruled out, in plain words.
    look-alike = it carries a claim that makes it look suitable."""
    text = " ".join(r["evidence"] for r in reasons if r["verdict"] == "reject")
    found = []
    for verb, code in re.findall(r"(MAY CONTAIN|CONTAINS) ([A-Z]{2})", text):
        w = ("may contain " if verb == "MAY CONTAIN" else "contains ") + ALLERGEN_WORDS.get(code, code)
        if w not in found:
            found.append(w)
    looks = [LOOKALIKE_CLAIMS[c] for c in p.get("claims") or [] if c in LOOKALIKE_CLAIMS]
    plain = (", ".join(found) or text) + (f", although {' and '.join(looks)}" if looks and found else "")
    return plain, bool(looks and found)


def _option(o, need, tier, ref, relaxed, verdict, reasons, kind="same", substitute=None):
    """A priced option for a line: the order line (units rounded to sales units and lot size)
    plus the raw metrics the ranking uses."""
    ln = engine._line(o, need)
    per = o["per_unit"]
    fit = 1.0
    if ref and ref.get("per_unit"):
        fit = 1.0 / (1.0 + abs(math.log(per / ref["per_unit"])))
    if kind != "same":
        fit *= 0.6
    stock = 1.0 if o["available_amt"] is None and o["qty_confirmed"] else (
        0.6 if not o["qty_confirmed"] else min(1.0, o["available_amt"] / max(need, 1e-9)) * 0.8)
    rel = (o["reliability"] if o["reliability"] is not None else engine.UNKNOWN_RELIABILITY) * (
        1 if o["qty_confirmed"] else engine.UNCONFIRMED_QTY_FACTOR)
    return {
        "id": f"{tier}:{o['id']}", "tier": tier, "kind": kind, "productId": o["id"], "name": o["name"],
        "name_en": o.get("name_en"), "quantity": ln["units"], "salesUnit": ln["sales_unit"], "amount": ln["amount"],
        "measure": o.get("measure"), "unitPrice": ln["unit_price"], "lineTotal": ln["total"],
        "lineTotalTax": ln["total_tax"], "eur_per_unit": round(o["eur_per_unit"], 3),
        "availableDate": o["available_date"], "orderCutOff": o["closing_time"], "stock": o["qty_status"],
        "daysEarly": o.get("days_early") or 0,  # ordered for the API's earliest date: days before the deadline
        "anomalies": [a["code"] for a in o.get("anomalies") or []],
        "deliveryReliability": o["reliability"], "flags": list(o["flags"]), "relaxed": relaxed,
        "allergen_verdict": verdict, "allergen_reasons": reasons, "substitute": substitute,
        "summary": "",  # one short phrase: what changes for the buyer (set by the tier builders)
        "metrics": {"eur": o["eur_per_unit"], "total": ln["total"], "availability": rel, "stock": stock, "fit": fit,
                    "packs": ln["units"] * o["packs_per_unit"] / max(ln["amount"], 1e-9),
                    "evidence": 1.0 if verdict == "pass" else 0.5},
    }


def _day(iso):
    return dt.date.fromisoformat(iso[:10]).strftime("%a %d %b") if iso else "?"


def _early_text(opt):
    n = opt.get("daysEarly") or 0
    return f"arrives {_day(opt['availableDate'])}, {n} day{'s' if n != 1 else ''} early" if n else ""


def _later_text(o, deadline):
    if not o.get("delivery_confirmed", True):
        return f"the order cut-off for {o['available_date']} has passed: arrives after {deadline}, date to be confirmed"
    return f"delivered {o['available_date']}, after the deadline {deadline}"


def _weights(prefs, answer):
    w = dict(prefs["weights"])
    w["stock"] = STOCK_WEIGHT
    key = PREFERENCE_OPTIONS.get(answer)
    if key:
        w[key] = w.get(key, 0) * PREFERENCE_BOOST if key != "stock" else STOCK_WEIGHT * PREFERENCE_BOOST * 3
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def utilities(opt, peers):
    """0-1 utilities of an option relative to the other options of the same line (best = 1)."""
    m = opt["metrics"]
    total = lambda x: max(x["metrics"].get("total", x["metrics"]["eur"]), 0.01)  # what is paid, packs rounded up
    min_total = min(total(p) for p in peers)
    min_packs = min(p["metrics"]["packs"] for p in peers)
    return {"cost": min_total / total(opt), "availability": m["availability"],
            "preference": m["fit"], "evidence": m["evidence"],
            "handling": min_packs / m["packs"] if m["packs"] else 1.0, "carbon": 1.0, "stock": m["stock"]}


def rank_line(line, weights):
    """Rank inside every tier (the tiers themselves keep their order) and pick the default:
    the top option of the first non-empty tier among best, better, good. `later` and `warn`
    are never chosen for the buyer."""
    peers = [o for t in TIERS for o in line["tiers"].get(t, [])]
    seen = set()
    for t in TIERS:
        opts = [o for o in line["tiers"].get(t) or [] if not (o["productId"] in seen or seen.add(o["productId"]))]
        for o in opts:
            o["u"] = {k: round(v, 4) for k, v in utilities(o, peers).items()}
            o["score"] = round(sum(weights.get(k, 0) * o["u"][k] for k in o["u"]), 4)
        opts.sort(key=lambda o: (-o["score"], o["productId"]))
        line["tiers"][t] = opts[:MAX_PER_TIER]
    line["default_option"] = next((line["tiers"][t][0]["id"] for t in ("best", "better", "good")
                                   if line["tiers"].get(t)), None)


# --- The basket ---------------------------------------------------------------------------
def run_basket(text, query, trace, policy, client, today, prefs, persist_fn=None, simulate=None, recipe=None):
    """Build the quoted basket for intent `source` (see the module docstring)."""
    calls = trace["llm_calls"]
    sim = load_simulation(simulate)
    everyone = [d for d in trace.get("proposal_diet") or [] if d in catalog.DIET_RULES]
    groups = build_groups(trace.get("portions"), trace.get("diet_groups"), everyone)
    portions = trace.get("portions")
    deliver_by = query.get("deliver_by")
    deadline = dt.date.fromisoformat(deliver_by) if deliver_by else today + dt.timedelta(days=1)
    later = deadline + dt.timedelta(days=1)
    dishes = trace.get("dishes") or []
    rec = recipe or nlu.recipe_for(text, dishes, portions, today, calls,
                         [catalog.DIET_RULES[d]["label"] for d in everyone],
                         [g["name"] for g in groups if g["key"] != "all"])
    idx = catalog.load_index()
    by_id = {p["id"]: p for p in idx}

    # 1. ingredients (merged over the dishes), pantry items aside
    ings, pantry = collections.OrderedDict(), []
    for r in rec["recipes"]:
        for ing in r["ingredients"]:
            if ing["role"] == "pantry":
                pantry.append({"ingredient": ing["name_en"], "dish": r["name"]})
                continue
            key = ing["name_en"].strip().lower()
            if key in ings:
                ings[key]["used_in"].append(r["name"])
                if ings[key]["qty_per_portion"] and ing.get("qty_per_portion") and ing.get("unit") == ings[key]["unit"]:
                    ings[key]["qty_per_portion"] += ing["qty_per_portion"]
                continue
            ings[key] = dict(ing, key=f"i{len(ings)}", used_in=[r["name"]])

    # 2. catalogue search per ingredient (+ the usual product) and identity verdicts in one call
    for ing in ings.values():
        found = catalog.search_products(ing["search_terms_fi"], ing["search_terms_en"], idx, limit=SEARCH_LIMIT,
                                        terms_sv=ing.get("search_terms_sv"))
        usual, note = usual_product(ing["name_en"])
        if usual and usual in by_id and usual not in [c["id"] for c in found]:
            p = by_id[usual]
            found.insert(0, {"id": p["id"], "name_fi": p["name_fi"], "name_en": p["name_en"],
                             "name_sv": p.get("name_sv", ""), "salesUnit": p["salesUnit"], "category": p["category"]})
        ing.update(candidates=found, usual=usual if usual in by_id else None, usual_note=note)
    try:
        verdicts = nlu.classify_groups([{"key": i["key"], "ingredient": i["name_en"], "form": i["form"],
                                         "candidates": i["candidates"]} for i in ings.values()], calls)
    except Exception as e:  # without identity nothing is matched automatically: the usual product only
        trace["notes"].append(f"identity check by Claude unavailable ({type(e).__name__}); only order-guide "
                              "products are matched")
        verdicts = {i["key"]: {c["id"]: {"verdict": "is_the_product" if c["id"] == i["usual"] else "different_product",
                                         "reason": "order guide" if c["id"] == i["usual"] else "not checked"}
                               for c in i["candidates"]} for i in ings.values()}
    for ing in ings.values():
        v = verdicts.get(ing["key"]) or {}
        if ing["usual"] and ing["usual"] in v:
            v[ing["usual"]] = {"verdict": "is_the_product", "reason": "the buyer's usual product (order guide)"}
        ing["verdicts"] = v

    # 3. live price and availability for every identified candidate (one call pair)
    cand_ids = sorted({c["id"] for i in ings.values() for c in i["candidates"]
                       if (i["verdicts"].get(c["id"]) or {}).get("verdict") in ("is_the_product", "variant")})
    live = _live(client, cand_ids, deadline, sim)

    # 4. lines: shared when the usual product suits every group, else one line per group
    lines = []
    for ing in ings.values():
        lines += _ingredient_lines(ing, groups, by_id, live, deadline, later)

    # 5. alternatives for red lines: function substitutes (one suggest + one pick + one live call)
    red = [ln for ln in lines if ln["light"] == "red"]
    if red:
        _add_substitutes(red, groups, by_id, client, deadline, later, sim, calls, trace, live)
    weights = _weights(prefs, None)
    for ln in lines:
        if ln["light"] == "red":
            rank_line(ln, weights)
    for ln in lines:  # every option is ordered for the API's earliest date: say when that is before the deadline
        for o in ([ln["chosen"]] if ln.get("chosen") else []) + [o for t in TIERS for o in ln["tiers"].get(t) or []]:
            if o.get("daysEarly"):
                o["summary"] = ((o["summary"] + ", ") if o["summary"] else "") + _early_text(o)

    order_id = f"BSK-{dt.datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    basket = {
        "order_id": order_id, "kind": "basket", "status": "AWAITING_CONFIRMATION",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "request": text, "query": query, "interpretation": trace, "policy": policy,
        "dishes": dishes, "portions": portions, "groups": groups, "deliver_by": deadline.isoformat(),
        "diet_labels": {d: catalog.DIET_RULES[d]["label"] for g in groups for d in g["diet"]},
        "recipes": [{k: r[k] for k in ("name", "description", "course", "served")}
                    | {"ingredients": r["ingredients"]} for r in rec["recipes"]],
        "assumptions": rec.get("assumptions") or [], "pantry": pantry,
        "lines": lines,
        "preference": {"question": PREFERENCE_QUESTION, "options": list(PREFERENCE_OPTIONS),
                       "answer": None, "asked": bool(red), "note": "Only changes the ranking inside each tier. "
                                                                    "It never relaxes a rule."},
        "weights": weights,
        "simulation": ({"name": sim["name"], "label": sim.get("label"), "delay_days": sim.get("delay_days"),
                        "changed_products": sorted(set(live["simulated"]))} if sim else None),
        "api_evidence": client.calls,
        "retrieval_stats": {"catalog_size": len(idx), "ingredients": len(ings), "pantry": len(pantry),
                            "candidates": sum(len(i["candidates"]) for i in ings.values()),
                            "live_checked": len(cand_ids), "server_time": live["now"].isoformat() if live["now"] else None},
    }
    _refresh(basket)
    if persist_fn:
        persist_fn(basket, write_audit)
    return basket


def _live(client, ids, deadline, sim):
    """Price and availability for `ids`, with the fault-injection overlay applied after the real
    response was hashed. Returns {"prices", "avail", "now", "simulated"}."""
    if not ids:
        return {"prices": {"products": []}, "avail": {"products": [], "timeNow": None}, "now": None, "simulated": []}
    prices = client.price(ids, deadline.isoformat())
    avail = client.availability(ids, deadline.isoformat())
    simulated = apply_simulation(avail, sim)
    return {"prices": prices, "avail": avail, "now": engine._ts(avail["timeNow"]), "simulated": simulated}


def _merge_live(a, b):
    a["prices"]["products"] += b["prices"]["products"]
    a["avail"]["products"] += b["avail"]["products"]
    a["simulated"] += b["simulated"]
    if a["now"] is None:
        a["now"], a["avail"]["timeNow"] = b["now"], b["avail"]["timeNow"]


def _offers(cands, live, deadline):
    """(offers by id, rejected by id) for slim candidates (catalog._check output)."""
    if not cands or not live["avail"].get("timeNow"):
        return {}, {}
    offers, rejected, _ = engine.live_check(cands, live["prices"], live["avail"], {"deliver_by": deadline.isoformat()},
                                            catalog.load_reliability(), catalog.load_carbon())
    return {o["id"]: o for o in offers}, {r["id"]: r for r in rejected}


def _checked(p, measure):
    """catalog._check without diet rules (allergens are judged per group by allergen_verdict)."""
    rej = []
    c = catalog._check(p, {"diet": []}, measure, lambda p, code, reason, constraint=None: rej.append((code, reason)))
    return c, rej


def _reference(usual_c, live):
    """Pack size and unit price of the usual product, even when it cannot be ordered (the price
    API still returns its price)."""
    if not usual_c:
        return None
    pr = next((x for x in live["prices"]["products"] if x["productId"] == usual_c["id"]), None) or {}
    cp, cmp_ = (pr.get("catalogPrice") or {}).get("value"), pr.get("comparisonPrice") or {}
    per = usual_c.get("per_unit")
    if not per and cp and cmp_.get("value") and pr.get("comparisonUnit") == usual_c.get("measure"):
        per = cp / cmp_["value"]
    return {"per_unit": per, "eur_per_unit": (cp / per) if cp and per else None}


def _ingredient_lines(ing, groups, by_id, live, deadline, later):
    """One line for every group (or one shared line), each with its light and its tiers."""
    measure = ing.get("unit") if ing.get("unit") in ("KG", "L", "PCS") else "KG"
    same, variants = [], []
    for c in ing["candidates"]:
        v = (ing["verdicts"].get(c["id"]) or {}).get("verdict")
        if v in ("is_the_product", "variant"):
            chk, _ = _checked(by_id[c["id"]], measure)
            if chk:
                (same if v == "is_the_product" else variants).append(chk)
    offers, rejected = _offers(same + variants, live, deadline)
    offers_later, _ = _offers([c for c in same + variants if c["id"] in rejected], live, later)
    usual_c = next((c for c in same if c["id"] == ing["usual"]), None)
    ref = _reference(usual_c, live)
    base = {"ingredient": ing["name_en"], "ingredient_key": ing["key"], "used_in": ing["used_in"],
            "role": ing["role"], "form": ing["form"], "qty_per_portion": ing.get("qty_per_portion"),
            "unit": ing.get("unit"), "measure": measure, "usual": ing["usual"], "usual_note": ing.get("usual_note"),
            "candidates_searched": len(ing["candidates"]), "identified": len(same), "variants": len(variants)}

    def verdict(c, g):
        return catalog.allergen_verdict(by_id[c["id"]], g["diet"], g["allergy"]) if g["diet"] else ("pass", [])

    def first_choice(g):
        """The usual product; else the best-ranked orderable same product that passes the group."""
        if usual_c:
            return usual_c
        ok = [offers[c["id"]] for c in same if c["id"] in offers and verdict(c, g)[0] == "pass"]
        need_g = _need(g["portions"], ing) or 1.0
        ok.sort(key=lambda o: (engine._line(o, need_g)["total"], -(o["reliability"] or 0)))  # cheapest for this need
        return ok[0] if ok else (same[0] if same else None)

    real_groups = [g for g in groups if g["portions"] is None or g["portions"] > 0]
    if not real_groups:
        real_groups = groups[:1]
    total_portions = sum(g["portions"] or 0 for g in real_groups) or None
    shared = first_choice(real_groups[0])
    if shared and all(verdict(shared, g)[0] == "pass" for g in real_groups):
        plan = [(dict(real_groups[0], key="all", name="all portions",
                      portions=total_portions, diet=sorted({d for g in real_groups for d in g["diet"]}),
                      allergy=any(g["allergy"] for g in real_groups)), shared)]
    else:
        plan = []
        for g in real_groups:
            fc = shared if (shared and verdict(shared, g)[0] == "pass") else None
            if fc is None and g["diet"]:
                cands_ok = [offers[c["id"]] for c in same if c["id"] in offers and verdict(c, g)[0] == "pass"]
                need_g = _need(g["portions"], ing) or 1.0
                cands_ok.sort(key=lambda o: engine._line(o, need_g)["total"])
                fc = cands_ok[0] if cands_ok else None
            plan.append((g, fc))

    out = []
    for g, fc in plan:
        need = _need(g["portions"], ing)
        line = dict(base, key=f"{ing['key']}:{g['key']}", group=g["key"], group_name=g["name"],
                    group_diet=g["diet"], allergy=g["allergy"], portions=g["portions"],
                    need={"qty": need, "unit": ing.get("unit"),
                          "basis": (f"{g['portions']:g} portions x {ing['qty_per_portion']:g} {ing['unit']}"
                                    if need is not None else "quantity per portion unknown")},
                    tiers={t: [] for t in TIERS}, excluded=[], chosen=None, default_option=None,
                    deadline=deadline.isoformat())
        if need is None:
            line.update(light="red", status="QTY_UNKNOWN", why="the recipe gives no quantity per portion")
            out.append(line)
            continue
        line["first_choice"] = ({"productId": fc["id"], "name": fc["name"], "name_en": fc.get("name_en"),
                                 "source": "order guide" if fc["id"] == ing["usual"] else "agent's best match"}
                                if fc else None)
        o = offers.get(fc["id"]) if fc else None
        if fc and o and (o["available_amt"] is None or o["available_amt"] >= need):
            v, reasons = verdict(fc, g)
            opt = _option(o, need, "as_requested", ref, [], v, reasons)
            opt["summary"] = "your usual product" if fc["id"] == ing["usual"] else "best match, in time"
            line.update(light="green", status="OK", chosen=opt, why="the usual product can be ordered in time"
                        if fc["id"] == ing["usual"] else "orderable in time")
            out.append(line)
            continue
        if not fc:
            why = (f"no catalogue product meets {', '.join(catalog.DIET_RULES[d]['label'] for d in g['diet'])}"
                   if g["diet"] and same else "not found in the catalogue")
        elif not o:
            r = rejected.get(fc["id"]) or {}
            why = f"#{fc['id']}: {r.get('reason') or 'not orderable'}"
        else:
            why = f"#{fc['id']}: only {o['qty_status']}, the order needs {need:g} {ing.get('unit')}"
        line.update(light="red", status="RED", why=why)
        _tiers_same_product(line, same, variants, offers, offers_later, rejected, ref, fc, g, need, verdict)
        out.append(line)
    return out


def _tiers_same_product(line, same, variants, offers, offers_later, rejected, ref, fc, g, need, verdict):
    """best / better / later / warn from SKUs of the same ingredient; known violations excluded."""
    for kind, pool in (("same", same), ("variant", variants)):
        for c in pool:
            if fc and c["id"] == fc["id"]:
                continue
            v, reasons = verdict(c, g)
            if v == "reject":
                plain, look = plain_reason(c, reasons)
                line["excluded"].append({"productId": c["id"], "name": c["name"], "name_en": c.get("name_en"),
                                         "plain": plain, "lookalike": look, "allergen_reasons": reasons,
                                         "reason": "; ".join(r["evidence"] for r in reasons if r["verdict"] == "reject")})
                continue
            o = offers.get(c["id"])
            if o and o["available_amt"] is not None and o["available_amt"] < need:
                o = None
            if not o:
                ol = offers_later.get(c["id"])
                if ol and v == "pass" and (ol["available_amt"] is None or ol["available_amt"] >= need):
                    opt = _option(ol, need, "later", ref, [_later_text(ol, line["deadline"])], v, reasons, kind)
                    opt["summary"] = _summary_later(ol)
                    line["tiers"]["later"].append(opt)
                continue
            relaxed = []
            if kind == "variant":
                relaxed.append("another form or cut of the ingredient")
            if ref and ref.get("per_unit") and o["per_unit"]:
                ratio = o["per_unit"] / ref["per_unit"]
                if abs(ratio - 1) > PACK_TOLERANCE:
                    relaxed.append(f"pack size {o['per_unit']:g} {o['measure']} instead of {ref['per_unit']:g}")
            if ref and ref.get("eur_per_unit") and o["eur_per_unit"] > ref["eur_per_unit"] * (1 + PRICE_TOLERANCE):
                relaxed.append(f"unit price EUR {o['eur_per_unit']:.2f}/{o['measure']} instead of "
                               f"EUR {ref['eur_per_unit']:.2f}")
            tier = "warn" if v == "warn" else ("best" if not relaxed else "better")
            opt = _option(o, need, tier, ref, relaxed, v, reasons, kind)
            opt["summary"] = _summary_same(tier, kind, o, ref)
            line["tiers"][tier].append(opt)


def _summary_same(tier, kind, o, ref):
    if tier == "best":
        return "same product, nothing changes"
    parts = []
    if kind == "variant":
        parts.append("another form")
    if ref and ref.get("per_unit") and o["per_unit"] and abs(o["per_unit"] / ref["per_unit"] - 1) > PACK_TOLERANCE:
        parts.append(("bigger" if o["per_unit"] > ref["per_unit"] else "smaller") + f" pack ({o['per_unit']:g} {o['measure']})")
    if ref and ref.get("eur_per_unit") and o["eur_per_unit"] > ref["eur_per_unit"] * (1 + PRICE_TOLERANCE):
        parts.append(f"+{(o['eur_per_unit'] / ref['eur_per_unit'] - 1) * 100:.0f} % unit price")
    if tier == "warn":
        parts.append("no allergen data, you check the label")
    return ", ".join(parts) or "similar product"


def _summary_later(o):
    if not o.get("delivery_confirmed", True):
        return "safe, but its cut-off has passed: arrives later, date not confirmed"
    return "safe, but one day late"


def _idea(alt):
    """'oat cooking cream (gluten-free certified)' -> 'oat cooking cream'."""
    return re.sub(r"\s*\(.*?\)", "", alt["name_en"]).strip()


def _add_substitutes(red, groups, by_id, client, deadline, later, sim, calls, trace, live):
    """Good / warn / later options from ingredients with the same function: Claude suggests, the
    catalogue is searched, allergen verdicts screen each group, Claude picks one product per
    (idea, verdict) from real IDs, and the picks are quoted live."""
    items = []
    for ln in red:
        if ln["status"] == "QTY_UNKNOWN":
            continue
        labels = [catalog.DIET_RULES[d]["label"] for d in ln["group_diet"]]
        items.append({"key": ln["key"], "name_en": ln["ingredient"], "form": ln["form"], "role_in_dish": ln["role"],
                      "used_in": ln["used_in"][:3], "why_missing": ln["why"], "dietary_requirements": labels,
                      "allergy": ln["allergy"], "whole_food_alternative": bool(ln["allergy"])})
    if not items:
        return
    try:
        ideas = nlu.suggest_substitutes(items, calls)
    except Exception as e:
        trace["notes"].append(f"substitute suggestions unavailable ({type(e).__name__}); same-product options only")
        return
    idx = catalog.load_index()
    by_line = {ln["key"]: ln for ln in red}
    pick_groups, meta = [], {}
    for key, alts in ideas.items():
        ln = by_line.get(key)
        if not ln:
            continue
        g = next(x for x in groups if x["key"] == ln["group"]) if ln["group"] != "all" else {
            "diet": ln["group_diet"], "allergy": ln["allergy"]}
        for j, alt in enumerate(alts):
            found = catalog.search_products(alt["search_terms_fi"], alt["search_terms_en"], idx,
                                            limit=SUB_SEARCH_LIMIT, terms_sv=alt.get("search_terms_sv"))
            usual, _ = usual_product(alt["name_en"])  # the kitchen's usual product for this idea is always checked
            if usual and usual in by_id and usual not in {c["id"] for c in found}:
                p = by_id[usual]
                found.insert(0, {"id": p["id"], "name_fi": p["name_fi"], "name_en": p["name_en"],
                                 "name_sv": p.get("name_sv", ""), "salesUnit": p["salesUnit"], "category": p["category"]})
            bucket = {"pass": [], "warn": []}
            for c in found:
                if c["id"] == (ln.get("first_choice") or {}).get("productId"):
                    continue
                v, reasons = (catalog.allergen_verdict(by_id[c["id"]], g["diet"], g["allergy"])
                              if g["diet"] else ("pass", []))
                if v == "reject":
                    if c["id"] not in {x["productId"] for x in ln["excluded"]}:
                        plain, look = plain_reason(by_id[c["id"]], reasons)
                        ln["excluded"].append({"productId": c["id"], "name": c["name_fi"], "via": alt["name_en"],
                                               "name_en": c.get("name_en"), "plain": plain, "lookalike": look,
                                               "allergen_reasons": reasons,
                                               "usual": c["id"] == usual,
                                               "reason": "; ".join(r["evidence"] for r in reasons if r["verdict"] == "reject")})
                    continue
                bucket[v].append(dict(c, _reasons=reasons))
            for v, cands in bucket.items():
                if cands:
                    gk = f"{key}|{j}|{v}"
                    pick_groups.append({"key": gk, "ingredient": alt["name_en"], "form": ln["form"],
                                        "need": f"{ln['need']['qty']:g} {ln['need']['unit']}",
                                        "used_in": [f"instead of {ln['ingredient']} in {ln['used_in'][0]}"],
                                        "candidates": [{k: c[k] for k in ("id", "name_fi", "name_en", "name_sv", "salesUnit")}
                                                       for c in cands]})
                    meta[gk] = (ln, alt, v, {c["id"]: c["_reasons"] for c in cands})
    if not pick_groups:
        return
    try:
        picks = nlu.pick_products(pick_groups, calls)
    except Exception as e:
        trace["notes"].append(f"substitute pick unavailable ({type(e).__name__})")
        return
    chosen = {gk: p["product_id"] for gk, p in picks.items() if p["product_id"] != "NONE"}
    runner = {gk: p["runner_up"] for gk, p in picks.items() if p.get("runner_up") not in (None, "NONE")}
    for g in pick_groups:  # an idea that is in the buyer's order guide uses the usual product, if it qualifies
        usual, _ = usual_product(meta[g["key"]][1]["name_en"])
        if usual and usual in {c["id"] for c in g["candidates"]}:
            chosen[g["key"]] = usual
            picks[g["key"]] = {"product_id": usual, "runner_up": "NONE", "reason": "the buyer's usual product (order guide)"}
    ids = sorted(set(chosen.values()) | set(runner.values()))
    if not ids:
        return
    sub_live = _live(client, ids, deadline, sim)
    _merge_live(live, sub_live)
    pairs = list(chosen.items()) + [(gk, pid) for gk, pid in runner.items() if pid != chosen.get(gk)]
    for gk, pid in pairs:
        ln, alt, v, reasons_by_id = meta[gk]
        measure = _measure_for(by_id[pid], ln["unit"] if ln["unit"] in ("KG", "L") else "KG")
        c, _ = _checked(by_id[pid], measure)
        if not c:
            continue
        need = ln["need"]["qty"]
        relaxed = [f"{alt['name_en']} instead of {ln['ingredient']}: {alt['why_similar'].rstrip('.')}"]
        if measure != ln["unit"]:
            relaxed.append(f"quantity assumed 1 {ln['unit']} = 1 {measure}; adjust if needed")
        offers, _ = _offers([c], sub_live, deadline)
        o = offers.get(pid)
        sub = {"name_en": alt["name_en"], "why_similar": alt["why_similar"], "pick_reason": picks[gk]["reason"]}
        if o and (o["available_amt"] is None or o["available_amt"] >= need):
            tier = "good" if v == "pass" else "warn"
            opt = _option(o, need, tier, None, relaxed, v, reasons_by_id[pid], "substitute", sub)
            opt["summary"] = (f"{_idea(alt)} instead of {ln['ingredient']}" if tier == "good"
                              else f"{_idea(alt)}: no allergen data, you check the label")
            ln["tiers"][tier].append(opt)
            continue
        ol = _offers([c], sub_live, later)[0].get(pid)
        if ol and v == "pass" and (ol["available_amt"] is None or ol["available_amt"] >= need):
            opt = _option(ol, need, "later", None, relaxed + [_later_text(ol, deadline)], v, reasons_by_id[pid],
                          "substitute", sub)
            opt["summary"] = f"{_idea(alt)}: " + _summary_later(ol)
            ln["tiers"]["later"].append(opt)


# --- Choices, totals, confirmation --------------------------------------------------------
def _find_option(line, option_id):
    if line.get("chosen") and line["chosen"]["id"] == option_id:
        return line["chosen"]
    return next((o for t in TIERS for o in line["tiers"].get(t) or [] if o["id"] == option_id), None)


def _refresh(b):
    """Totals, counts, quote validity and next steps from the current choices."""
    sel = [(ln, _selected(ln)) for ln in b["lines"]]
    picked = [o for _, o in sel if o]
    b["totals"] = {"eur": round(sum(o["lineTotal"] for o in picked), 2),
                   "eur_incl_vat": round(sum(o["lineTotalTax"] for o in picked), 2),
                   "lines": len(b["lines"]), "green": sum(ln["light"] == "green" for ln in b["lines"]),
                   "red": sum(ln["light"] == "red" for ln in b["lines"]),
                   "unresolved": [ln["key"] for ln, o in sel if not o]}
    cutoffs = [o["orderCutOff"] for o in picked if o.get("orderCutOff")]
    b["quote"] = {"captured_at": b["created_at"], "valid_until": min(cutoffs) if cutoffs else None,
                  "note": "The mock has no quote endpoint: the agent holds the quoted prices and re-checks price and "
                          "stock live at confirmation. If anything changed, nothing is ordered."}
    steps = []
    if b["status"] == "AWAITING_CONFIRMATION":
        if b["totals"]["red"] and not b["preference"]["answer"]:
            steps.append(PREFERENCE_QUESTION + " (price, delivery, stock or fit)")
        if b["totals"]["unresolved"]:
            steps.append("Choose an option for: " + ", ".join(
                f"{ln['ingredient']} ({ln['group_name']})" for ln, o in sel if not o) + ".")
        if any(o and o["tier"] == "warn" for _, o in sel):
            steps.append("A line with incomplete allergen evidence is selected: confirm that you will check the "
                         "package label on arrival.")
        steps.append("Confirm the whole order once; every line is re-checked live first.")
        if b.get("pantry"):
            steps.append("Pantry items are not searched or checked against the diet groups: "
                         + ", ".join(sorted({p["ingredient"] for p in b["pantry"]})) + ".")
    allergy = [g for g in b["groups"] if g.get("allergy")]
    b["pantry_warning"] = (f"Pantry items come from your own kitchen and were not checked: "
                           f"{', '.join(sorted({p['ingredient'] for p in b['pantry']}))}. Check them for "
                           + "; ".join(g["name"] for g in allergy) + " before use.") \
        if allergy and b.get("pantry") else None
    b["next_steps"] = steps


def _selected(line):
    if line["light"] == "green":
        return line["chosen"]
    if line.get("chosen"):
        return line["chosen"]
    return _find_option(line, line["default_option"]) if line.get("default_option") else None


def set_preference(b, answer, prefs):
    """Re-rank within each tier. None restores the default weights."""
    if b["status"] != "AWAITING_CONFIRMATION":
        raise ValueError(f"basket {b['order_id']} is {b['status']}")
    if answer is not None and answer not in PREFERENCE_OPTIONS:
        raise ValueError(f"answer must be one of {', '.join(PREFERENCE_OPTIONS)}")
    b["preference"]["answer"] = answer
    b["weights"] = _weights(prefs, answer)
    for ln in b["lines"]:
        if ln["light"] == "red":
            rank_line(ln, b["weights"])
    _refresh(b)
    return b


def with_quantity(ln, opt, units):
    """The option re-priced for `units` sales units chosen by a person (validated): positive, whole
    units unless sold by the kilogram, never more than limited stock. Returns a new option dict."""
    try:
        units = float(units)
    except (TypeError, ValueError):
        raise ValueError(f"{ln['ingredient']}: quantity {units!r} is not a number")
    whole = (opt.get("salesUnit") or "").upper() not in ("KG",)
    if units <= 0 or (whole and abs(units - round(units)) > 1e-9):
        raise ValueError(f"{ln['ingredient']}: order a positive {'whole ' if whole else ''}number of "
                         f"{opt.get('salesUnit') or 'units'}")
    units = round(units) if whole else round(units, 2)
    m = re.match(r"limited: ([\d.]+)", opt.get("stock") or "")
    if m and units > float(m.group(1)) + 1e-9:
        raise ValueError(f"{ln['ingredient']}: only {float(m.group(1)):g} {opt.get('salesUnit')} in stock")
    if abs(units - opt["quantity"]) < 1e-9:
        return opt
    per = opt["amount"] / opt["quantity"] if opt["quantity"] else 1.0
    tax = opt["lineTotalTax"] / opt["lineTotal"] if opt.get("lineTotal") else 1.0
    out = dict(opt, quantity=units, amount=round(units * per, 3), lineTotal=round(units * opt["unitPrice"], 2))
    out["lineTotalTax"] = round(out["lineTotal"] * tax, 2)
    out["quantity_change"] = {"proposed": opt["quantity"], "chosen": units, "need": (ln.get("need") or {}).get("qty"),
                              "unit": opt.get("measure")}
    return out


def confirm(b, choices, acks, by, client, prefs, now, quantities=None):
    """One confirmation for the whole basket. choices: {line key: option id} (missing = default);
    acks: line keys whose incomplete-evidence warning the person accepted; quantities: {line key:
    sales units} a person changed (re-priced and re-checked like the rest). Returns
    (basket, preference_updates). Raises ValueError for a missing choice, acknowledgement or a bad quantity."""
    if b["status"] != "AWAITING_CONFIRMATION":
        raise ValueError(f"basket {b['order_id']} is {b['status']}, nothing to confirm")
    choices, acks, quantities = choices or {}, set(acks or []), quantities or {}
    unknown = sorted(set(quantities) - {ln["key"] for ln in b["lines"]})
    if unknown:
        raise ValueError(f"unknown line(s) for a quantity: {', '.join(unknown)}")
    picked = []
    for ln in b["lines"]:
        if ln["light"] == "green":
            opt = ln["chosen"]
            picked.append((ln, with_quantity(ln, opt, quantities[ln["key"]]) if ln["key"] in quantities else opt))
            continue
        oid = choices.get(ln["key"]) or ln.get("default_option")
        opt = _find_option(ln, oid) if oid else None
        if not opt:
            raise ValueError(f"choose an option for {ln['ingredient']} ({ln['group_name']})")
        if TIER_INFO[opt["tier"]]["gate"] == "acknowledge" and ln["key"] not in acks:
            raise ValueError(f"{ln['ingredient']} ({ln['group_name']}): {opt['name']} has incomplete allergen "
                             "evidence; confirm that you will check the package label on arrival")
        picked.append((ln, with_quantity(ln, opt, quantities[ln["key"]]) if ln["key"] in quantities else opt))
    # commit-time check against a fresh live quote (engine.revalidate), per delivery deadline
    sim = load_simulation((b.get("simulation") or {}).get("name"))
    deadline = dt.date.fromisoformat(b["deliver_by"])
    ids = sorted({o["productId"] for _, o in picked})
    live = _live(client, ids, deadline, sim)
    changes, fresh = [], []
    for tier_deadline in sorted({deadline + dt.timedelta(days=1 if o["tier"] == "later" else 0) for _, o in picked}):
        lines = [o for _, o in picked if (o["tier"] == "later") == (tier_deadline != deadline)]
        ch, fr, _ = engine.revalidate(lines, live["prices"], live["avail"], tier_deadline)
        changes += ch
        fresh += fr
    b["revalidation"] = {"at": now, "server_time": live["now"].isoformat() if live["now"] else None,
                         "ok": not changes, "changes": changes, "lines": fresh, "api_evidence": client.calls}
    if changes:
        b["status"] = "STALE_QUOTE"
        b["human_decision"] = {"by": by, "at": now, "action": "confirm"}
        _refresh(b)
        return b, []
    updates, tasks = [], []
    for ln, opt in picked:
        ln["chosen"], info = opt, TIER_INFO[opt["tier"]]
        ln["decision"] = {"tier": opt["tier"], "tier_label": info["label"], "gate": info["gate"],
                          "acknowledged": ln["key"] in acks if info["gate"] == "acknowledge" else None,
                          "learnable": info["learnable"], "relaxed": opt.get("relaxed") or [],
                          "quantity_change": opt.get("quantity_change")}
        if opt["tier"] == "warn":
            tasks.append({"line": ln["key"], "productId": opt["productId"], "name": opt["name"],
                          "due": opt["availableDate"], "accepted_by": by,
                          "check": "On arrival, check the package label: no "
                                   + " and no ".join(r["label"].replace("-free", "") for r in opt["allergen_reasons"])
                                   + f" ingredients, before it is used for {ln['group_name']}."})
            ln["decision"]["learning"] = "not learned: incomplete evidence is warned about again every time"
        elif info["learnable"] and ln["light"] == "red" and ln.get("default_option") and opt["id"] != ln["default_option"]:
            rec = _find_option(ln, ln["default_option"])
            old, new = engine.learn_from_choice(prefs, {k: rec["u"][k] for k in prefs["weights"]},
                                                {k: opt["u"][k] for k in prefs["weights"]})
            updates.append({"line": ln["key"], "chose": opt["id"], "instead_of": rec["id"], "before": old, "after": new})
            ln["decision"]["learning"] = "weights updated from this choice (ranking only)"
    b["status"] = "CONFIRMED_BY_HUMAN"
    b["tasks"] = tasks
    b["human_decision"] = {"by": by, "at": now, "action": "confirm", "acknowledged": sorted(acks),
                           "preference_updates": updates,
                           "committed_total_eur": round(sum(x.get("lineTotal") or 0 for x in fresh), 2)}
    _refresh(b)
    return b, updates


# --- Audit -------------------------------------------------------------------------------
def _opt_text(o):
    early = f" = {o['daysEarly']} days before needed" if o.get("daysEarly") else ""
    return (f"{o['productId']} {o['name'][:40]} x{o['quantity']:g} {o['salesUnit']} = EUR {o['lineTotal']:,.2f} "
            f"(EUR {o['eur_per_unit']:.2f}/{o['measure']}, arrives {o['availableDate']}{early}, {o['stock']})")


def _smt_audit_lines(option):
    lines = []
    for reason in option.get("allergen_reasons", []):
        if reason.get("smt"):
            results = ", ".join(f"{query} {result.upper()}" for query, result in reason["smt"].items())
            lines.append(f"  - SKU {option['productId']} · Z3 {reason['label']}: {results}. {reason['evidence']}")
    return lines


def write_audit(b, out_dir):
    os.makedirs(os.path.join(out_dir, "audit"), exist_ok=True)
    path = os.path.join(out_dir, "audit", b["order_id"] + ".md")
    it = b["interpretation"]
    L = [f"# Audit trail {b['order_id']} (multi-line order)", "",
         f"- **Created:** {b['created_at']}  ", f"- **Buyer / policy:** {b['policy']['buyer']} / "
         f"{b['policy']['policy_version']}  ", f"- **Status:** `{b['status']}` ({STATUS_TEXT[b['status']]})", ""]
    if b.get("simulation"):
        s = b["simulation"]
        L += [f"> **{s['label']}** (fixture `scenarios/{s['name']}.json`): availableDate shifted by {s['delay_days']} "
              f"day(s) after the real response was received, for {len(s['changed_products'])} products: "
              + ", ".join(s["changed_products"]) + ". The SHA-256 values below hash the real responses.", ""]
    L += ["## 1. Request", "", f"> {b['request']}", "", "## 2. Interpretation", "",
          f"Parser: {it['parser']}. Intent: `{it['intent']}`. Dishes: {', '.join(b['dishes']) or '-'}. "
          f"Portions: {b['portions'] if b['portions'] is not None else 'not stated'}. Deliver by: {b['deliver_by']}.", ""]
    L += [f"- group `{g['key']}`: {g['name']}, {g['portions'] if g['portions'] is not None else '?'} portions, "
          f"diet {g['diet'] or 'none'}" + (f", read by {g.get('read_by')} from \"{g.get('quote')}\"" if g.get("quote") else "")
          for g in b["groups"]]
    L += [f"- note: {n}" for n in it["notes"]] + [f"- **issue:** {n}" for n in it["issues"]]
    L += [f"- LLM call `{c['stage']}`: {c['model']}, effort {c['effort']}, {c['input_tokens']} in / "
          f"{c['output_tokens']} out tokens, {c['seconds']} s" for c in it.get("llm_calls") or []]
    L += ["", "## 3. Lines", "", "| line | group | need | light | first choice | chosen | tier |", "|---|---|---|---|---|---|---|"]
    for ln in b["lines"]:
        sel = _selected(ln)
        fc = ln.get("first_choice") or {}
        L.append(f"| {ln['ingredient']} | {ln['group_name']} | {ln['need']['basis']} = {ln['need']['qty'] or '?'} "
                 f"{ln['need']['unit'] or ''} | {ln['light']} | {fc.get('productId', '-')} ({fc.get('source', '-')}) | "
                 f"{_opt_text(sel) if sel else '-'} | {(sel or {}).get('tier', '-')} |")
    for ln in b["lines"]:
        if ln["light"] != "red":
            results = _smt_audit_lines(_selected(ln) or {})
            if results:
                L += [""] + results
            continue
        L += ["", f"### {ln['ingredient']} ({ln['group_name']}): red", "", f"Why: {ln['why']}", ""]
        for t in TIERS:
            for o in ln["tiers"].get(t) or []:
                L.append(f"- {TIER_INFO[t]['label']}: {_opt_text(o)}; relaxed: {'; '.join(o['relaxed']) or 'nothing'}; "
                         f"allergen verdict {o['allergen_verdict']}; score {o.get('score')}")
                L += _smt_audit_lines(o)
        for x in ln["excluded"]:
            L.append(f"- ruled out: {x['productId']} {x['name'][:40]}: {x['reason']}")
            L += _smt_audit_lines(x)
    L += ["", "## 4. Live data evidence", "", "| endpoint | at | requestedDate | products | response sha256 |",
          "|---|---|---|---|---|"]
    L += [f"| {c['endpoint']} | {c['at']} | {c['requestedDate']} | {c['products']} | `{c['response_sha256'][:16]}...` |"
          for c in b["api_evidence"]]
    L += ["", "## 5. Decision", "", f"Quote held until {b['quote']['valid_until']}; {b['quote']['note']}", "",
          f"Total EUR {b['totals']['eur']:,.2f} excl. VAT. Preference question asked: {b['preference']['asked']}."]
    L += [f"- next: {x}" for x in b["next_steps"]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


def append_audit(b, path, title, extra=()):
    L = ["", f"## {title}", ""] + list(extra)
    if b.get("human_decision") and title.startswith("Confirmation"):
        d = b["human_decision"]
        L.append(f"- {d['action'].upper()} by {d['by']} at {d['at']}")
        rv = b.get("revalidation") or {}
        if rv.get("ok"):
            L.append(f"- re-checked live (server time {rv['server_time']}): every line still holds; committed "
                     f"EUR {d.get('committed_total_eur', 0):,.2f}")
        elif rv:
            L += ["- re-checked live: the quote no longer holds, nothing was ordered:"] + [f"  - {c}" for c in rv["changes"]]
        L += [f"- evidence: {c['endpoint']} at {c['at']}, response sha256 `{c['response_sha256'][:16]}...`"
              for c in rv.get("api_evidence") or []]
        for ln in b["lines"]:
            dd = ln.get("decision")
            if dd:
                L.append(f"- line {ln['ingredient']} ({ln['group_name']}): tier **{dd['tier_label']}**, gate {dd['gate']}"
                         + (f", acknowledged: {dd['acknowledged']}" if dd["acknowledged"] is not None else "")
                         + (f", relaxed: {'; '.join(dd['relaxed'])}" if dd["relaxed"] else "")
                         + (f", quantity set by a person: {dd['quantity_change']['chosen']:g} {ln['chosen'].get('salesUnit')} "
                            f"(proposed {dd['quantity_change']['proposed']:g}; need {dd['quantity_change']['need']:g} "
                            f"{dd['quantity_change']['unit']})" if dd.get("quantity_change") else "")
                         + (f"; {dd['learning']}" if dd.get("learning") else ""))
        for t in b.get("tasks") or []:
            L.append(f"- **arrival check task** ({t['due']}): {t['productId']} {t['name']}: {t['check']} "
                     f"Accepted with incomplete allergen evidence by {t['accepted_by']}.")
        for u in d.get("preference_updates") or []:
            L.append(f"- preference weights learned on {u['line']}: " + ", ".join(
                f"{k} {u['before'][k]:.2f}->{v:.2f}" for k, v in u["after"].items()))
    L.append(f"- status: `{b['status']}`")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
