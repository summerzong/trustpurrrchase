#!/usr/bin/env python3
"""Trusted Trade Agent - minimal demo.

  python3 agent.py "I need 1,000 litres of lactose-free milk by Monday. ..."   # request -> lookup -> order/approval
  python3 agent.py "..." --json                                                # machine-readable response only
  python3 agent.py approve ORD-... [--choose N] [--by NAME]                    # human sign-off (+ preference learning)
  python3 agent.py reject  ORD-... [--by NAME]
  python3 agent.py relax   ORD-... R1 [--by NAME]                                # accept a relaxation, re-run
  python3 agent.py cancel  ORD-...|BSK-...                                       # undo before the order cut-off
  python3 agent.py confirm BSK-... [--prefer price] [--pick LINE=OPTION] [--ack LINE] [--qty LINE=UNITS]  # one confirmation
  AGENT_TODAY=2026-11-15 pins 'today' for relative dates; --simulate hospital_delay (fault injection, labelled)
  python3 agent.py "..." --new                                                 # same goods again (skips the replay)

Env: AIMO_API_KEY (required), ANTHROPIC_API_KEY (optional, enables the LLM parser).
Both are read from ./.env if not already set in the shell.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import sys

import basket
import catalog
import engine
import nlu
from aimo_api import AimoApiError, AimoClient

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ORDERS = os.path.join(OUT, "orders.json")
PREFS = os.path.join(OUT, "prefs.json")
DEFAULT_PREFS = {"weights": {"cost": 0.5, "availability": 0.2, "preference": 0.15,
                             "evidence": 0.05, "handling": 0.05, "carbon": 0.05}, "history": []}
OPEN_STATUSES = ("PLACED_AUTONOMOUSLY", "PENDING_APPROVAL", "APPROVED_BY_HUMAN")  # committed or waiting for a person
CLIENT_KEY_TTL_HOURS = 24  # how long a caller's Idempotency-Key is honoured


def today():
    """The agent's 'today' for reading relative dates ('tomorrow', 'by Monday'). AGENT_TODAY=YYYY-MM-DD
    pins it for a recorded demo; prices, stock and cut-offs always use the live server time."""
    pinned = os.environ.get("AGENT_TODAY")
    return dt.date.fromisoformat(pinned) if pinned else dt.date.today()


def clock_info():
    return {"today": today().isoformat(), "pinned": bool(os.environ.get("AGENT_TODAY"))}


def load_dotenv(path=os.path.join(HERE, ".env")):
    """Minimal .env reader (KEY=VALUE lines, # comments). Shell variables win over the file."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip().strip("'\"")
            if key and value:
                os.environ.setdefault(key, value)


def _load(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(default))


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _public_plan(p, i):
    """Plan without internal references - this is what the buyer's agent receives.
    Amounts are in `measure` (L, KG or PCS); for L the earlier litre keys are kept too."""
    m = p["measure"]
    out = {
        "option": i + 1, "kind": p["kind"], "score": p["score"], "tags": p["tags"], "pareto_optimal": p["pareto"],
        "total_eur": p["total"], "total_eur_incl_vat": p["total_tax"], "measure": m,
        "eur_per_unit": round(p["eur_per_unit"], 3), "amount": p["amount"], "shortfall": p["shortfall"],
        "consumer_packs": p["packs"], "co2e_kg": p["co2_kg"], "utilities": {k: round(v, 3) for k, v in p["u"].items()},
        "lines": [{
            "productId": ln["id"], "name": ln["name"], "quantity": ln["units"], "salesUnit": ln["sales_unit"],
            "amount": ln["amount"], "unitPrice": ln["unit_price"], "lineTotal": ln["total"],
            "logisticModel": ln["_o"]["logistic"], "availability": ln["_o"]["qty_status"],
            "availableDate": ln["_o"]["available_date"], "orderCutOff": ln["_o"]["closing_time"],
            # the order is placed for availableDate (the API's earliest date): this many days before the deadline
            "daysEarly": ln["_o"].get("days_early") or 0,
            "deliveryReliability": ln["_o"]["reliability"], "dietEvidence": ln["_o"]["evidence"],
            "flags": ln["_o"]["flags"], "anomalies": [a["code"] for a in ln["_o"].get("anomalies") or []],
        } for ln in p["lines"]],
    }
    if m == "L":
        out.update(eur_per_litre=out["eur_per_unit"], litres=p["amount"], shortfall_litres=p["shortfall"])
        for ln in out["lines"]:
            ln["litres"] = ln["amount"]
    return out


def run_request(text, policy, use_llm=True, persist=True, idempotency_key=None, allow_duplicate=False,
                simulate=None):
    """Request -> order decision (AUTO / approval / escalation) or, for an abstract
    request such as a menu idea, a proposal that never orders.

    A repeated request is answered with the earlier order instead of a new one: the caller's
    `idempotency_key` (honoured CLIENT_KEY_TTL_HOURS), or the same normalised request
    (request_hash) within policy `idempotency_window_minutes` while that order is placed,
    approved or waiting. allow_duplicate skips the replay when a person wants the same goods
    again; the duplicate-order guardrail then asks for approval."""
    today_ = today()
    replay = persist and not allow_duplicate
    if replay and idempotency_key:
        prior = _find_replay(policy, client_key=idempotency_key)
        if prior:
            if prior["request"] != text:
                raise ValueError(f"idempotency key {idempotency_key!r} was already used for a different request "
                                 f"({prior['order_id']})")
            return _replay(prior, "same idempotency key")
    query, trace = nlu.interpret(text, today_, use_llm=use_llm)
    client = None if trace.get("past_delivery_date") else AimoClient(customer=policy["customer_id"])
    if trace.get("intent") == "order" and trace.get("diet_groups") and not trace["blocking"]:
        # one product, but some portions have their own diet: split it per group instead of ordering
        # the same product for everyone (the whole-order diet would ignore the sub-group)
        if trace.get("portions") and query.get("quantity") and query.get("unit"):
            return run_basket(text, query, trace, policy, client, persist, simulate, recipe=_split_recipe(query, trace))
        msg = "some portions have their own diet: how many portions are there in total?"
        trace["issues"].append(msg)
        trace["blocking"].append(msg)
    if trace.get("intent") == "source":
        try:
            return run_basket(text, query, trace, policy, client, persist, simulate)
        except AimoApiError:
            raise
        except ValueError:
            raise
        except Exception as e:  # recipe step failed -> fall back to the (blocked) order path: ask the buyer
            trace["notes"].append(f"recipe unavailable ({type(e).__name__}: {str(e)[:120]})")
            trace["intent"] = "order"
            trace["blocking"] = nlu._validate(query)
            trace["issues"] += trace["blocking"]
    if trace.get("intent") == "recommend":
        try:
            return run_proposal(text, query, trace, policy, client, persist)
        except AimoApiError:
            raise
        except Exception as e:  # recipe step failed -> fall back to the (blocked) order path: ask the buyer
            trace["notes"].append(f"recipe suggestions unavailable ({type(e).__name__}: {str(e)[:120]})")
            trace["intent"] = "order"
            trace["blocking"] = nlu._validate(query)
            trace["issues"] += trace["blocking"]
    if replay and not trace["blocking"]:
        prior = _find_replay(policy, request_hash(query, policy.get("customer_id")))
        if prior:
            return _replay(prior, f"same request within {policy['idempotency_window_minutes']:g} minutes")
    return _run_order(text, query, trace, policy, client, persist, client_key=idempotency_key)


def _run_order(text, query, trace, policy, client, persist=True, relaxation=None, relaxed_from=None,
               stale=None, client_key=None):
    """Order path for an interpreted query: retrieval, live checks, ranking, guardrails.
    Re-run with a changed query when a person accepts a relaxation (relax_order), and with
    the same query when a quote changed before approval (decide_order, `stale`)."""
    today_ = today()
    if not trace.get("past_delivery_date"):
        nlu.block_past_delivery(trace, today_, query.get("deliver_by"))
    ptype = catalog.PRODUCT_TYPES.get(query.get("product_type"))
    generic = query.get("product_type") == "other"
    if ptype and query.get("unit") and query["unit"] != ptype["measure"]:
        msg = f"unit {query['unit']} does not fit {query['product_type']} (sold by {ptype['measure']})"
        trace["issues"].append(msg)
        trace["blocking"].append(msg)
    offers, rej_live, server_now = [], [], None
    if trace["blocking"]:  # cannot search safely -> ask the buyer instead of guessing
        cands, rej_static, stats = [], [], {"catalog_size": None, "error": "request needs clarification"}
    elif generic:
        cands, rej_static, stats = _generic_retrieve(query, trace)
    else:
        cands, rej_static, stats = catalog.retrieve(query, catalog.load_index())
    if cands:
        date = query.get("deliver_by") or (today_ + dt.timedelta(days=1)).isoformat()
        ids = [c["id"] for c in cands]
        prices, avail = client.price(ids, date), client.availability(ids, date)
        offers, rej_live, server_now = engine.live_check(
            cands, prices, avail, query, catalog.load_reliability(), catalog.load_carbon())

    prefs = _load(PREFS, DEFAULT_PREFS)
    weights = engine.effective_weights(prefs, query)
    plans = engine.rank(engine.build_plans(offers, query), weights) if offers else []
    best = plans[0] if plans else None
    diet = [d for d in query.get("diet") or [] if d in catalog.DIET_RULES]
    diet_labels = {d: catalog.DIET_RULES[d]["label"] for d in diet}
    unreviewed = [d for d in diet if not catalog.DIET_RULES[d]["reviewed"]]
    product_label = (ptype or {}).get("label") or (query.get("product") or {}).get("name_en") or query.get("product_type")
    type_unreviewed = product_label if (generic or (ptype and not ptype.get("reviewed"))) else None
    duplicates = _possible_duplicates(query, policy) if persist else None
    decision, checks = engine.guardrails(best, query, policy, trace["issues"], unreviewed, type_unreviewed, duplicates)
    if relaxation and decision != "ESCALATE_NO_OPTION":
        checks.append((True, f"relaxation accepted by {relaxation['accepted_by']}: {relaxation['text']}"))
    explanation = engine.explain(best, plans, len(offers), query, diet_labels)

    order_id = f"ORD-{dt.datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    status = {"AUTO_ORDER": "PLACED_AUTONOMOUSLY", "NEEDS_HUMAN_APPROVAL": "PENDING_APPROVAL",
              "ESCALATE_NO_OPTION": "ESCALATED_NO_OPTION"}[decision]
    # policy.json `require_human_confirmation` (default on): the agent never places an order on its own.
    # Even when every purchasing rule passes, a person confirms the order, which is then re-checked live.
    human_gate = status == "PLACED_AUTONOMOUSLY" and policy.get("require_human_confirmation", True)
    if human_gate:
        status = "PENDING_APPROVAL"
    rejected = rej_static + rej_live
    relaxations = (_relaxations(query, trace, stats, rejected, client, product_label, diet_labels, generic)
                   if status == "ESCALATED_NO_OPTION" else [])
    response = {
        "order_id": order_id, "status": status,
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "request": text, "query": query, "interpretation": trace, "diet_labels": diet_labels,
        "product": {"label": product_label, "type": query.get("product_type"),
                    "reviewed_type": bool(ptype and ptype.get("reviewed")),
                    "measure": (ptype or {}).get("measure") or query.get("unit")},
        "rule_reviews": _rule_reviews(diet, ptype),
        "constraints": _constraints(query, diet_labels, product_label),
        "relaxations": relaxations, "relaxed_from": relaxed_from, "relaxation": relaxation, "stale_quote": stale,
        "idempotency": {"request_hash": request_hash(query, policy.get("customer_id")), "client_key": client_key},
        "policy": policy, "weights": weights,
        "recommended_option": 1 if best else None,
        "options": [_public_plan(p, i) for i, p in enumerate(engine.select_for_display(plans))],
        "guardrail_checks": [dict(zip(("group", "ref"), check_source(label)), ok=ok, check=label)
                             for ok, label in checks],
        "explanation": explanation,
        "rejected": rejected,
        "retrieval_stats": dict(stats, live_checked=len(cands), feasible_offers=len(offers),
                                server_time=server_now.isoformat() if server_now else None),
        "api_evidence": client.calls if client is not None else [],
        "next_steps": _next_steps(status, rejected, best, query, stats, trace, diet_labels, product_label, relaxations),
        "clock": clock_info(),
    }
    if human_gate:
        response["requires_confirmation"] = "policy.json · require_human_confirmation"
        response["next_steps"].insert(0, "Every purchasing rule passes. The agent never orders on its own: "
                                         "a person confirms, and the quote is re-checked live first.")
    if status == "PLACED_AUTONOMOUSLY":
        response["cancel_until"] = _cancel_until(response)
        if response["cancel_until"]:
            response["next_steps"].append(f"You can cancel or change this order until the order cut-off "
                                          f"{response['cancel_until'][:16].replace('T', ' ')}.")
    _persist(response, write_audit, persist)
    return response


# --- Constraint strength, 'why no result', relaxations with consent -----------------
STRENGTH_WHY = {
    "hard": "diet and allergy needs are never relaxed",
    "negotiable": "applied as a requirement; relaxed only with your OK",
    "preference": "ranks the options; an option outside it needs your OK",
}


def _constraints(query, diet_labels, product_label):
    """Every requirement in the request with its strength: hard (never relaxed), negotiable
    (relaxed only with a person's OK) or preference (ranks options)."""
    out = []

    def add(name, strength):
        out.append({"name": name, "strength": strength, "why": STRENGTH_WHY[strength]})

    if product_label:
        add(f"product: {product_label}", "negotiable")
    if query.get("quantity"):
        add(f"quantity: {query['quantity']:g} {query.get('unit') or ''}".strip(), "negotiable")
    if query.get("deliver_by"):
        add(f"deliver by {query['deliver_by']}", "negotiable")
    for d in query.get("diet") or []:
        add(diet_labels.get(d, d), "hard")
    for b in query.get("avoid_brands") or []:
        add(f"not {b}", "negotiable")
    prefs, must = query.get("preferences") or {}, set(query.get("must_have") or [])
    for k, name in (("brand", "brand"), ("fat_level", "fat level"), ("organic", "organic")):
        v = prefs.get(k)
        if v:
            add(f"{name}: {v if v is not True else 'yes'}" + (" (only)" if k in must else ""),
                "negotiable" if k in must else "preference")
    return out


def _next_steps(status, rejected, best, query=None, stats=None, trace=None, labels=None, product_label=None,
                relaxations=()):
    """What to do next. For 'nothing could be ordered', say which requirement blocked it:
    hard ones are explained and never relaxed; negotiable ones come with relaxations."""
    if status == "PLACED_AUTONOMOUSLY":
        return ["Order recorded in the order book (the mock exposes no order endpoint)."]
    if status == "PENDING_APPROVAL":
        return ["A buyer must run: approve <order_id> [--choose N]  or  reject <order_id>"]
    query, stats, trace, labels = query or {}, stats or {}, trace or {}, labels or {}
    if trace.get("past_delivery_date"):
        return [nlu.PAST_DELIVERY_MESSAGE]
    what = product_label or "product"
    tips = []
    if trace.get("blocking"):
        tips.append("Clarify the request first: " + "; ".join(trace["blocking"]) + ".")
    elif stats.get("error"):
        tips.append(stats["error"][0].upper() + stats["error"][1:] + ".")
    codes = collections.Counter(r["code"] for r in rejected)
    diet_out = collections.Counter(r.get("constraint") for r in rejected if r["code"] in ("DIET_FAIL", "DIET_NEAR_MISS"))
    reached = sum(diet_out.values()) + (stats.get("candidates") or 0)
    if stats.get("candidates"):  # products passed the diet check: diet is not why nothing could be ordered
        diet_out = {}
    for d, n in diet_out.items():
        lab = labels.get(d, d)
        mine = [r for r in rejected if r.get("constraint") == d]
        near = sum(1 for r in mine if r["code"] == "DIET_NEAR_MISS")
        contra = sum(1 for r in mine if r["reason"].startswith(f"not {lab}:"))
        parts = [f"{contra} contradicted by the allergen data" if contra else "", f"{near} look-alikes" if near else "",
                 f"{n - contra - near} without evidence" if n - contra - near else ""]
        tips.append(f"Hard requirement '{lab}': all {n} of {reached} {what} products were ruled out ("
                    + ", ".join(x for x in parts if x) + "). Diet and allergy requirements are never relaxed.")
    for code, text in (("AVOIDED_BRAND", "were excluded by brand"), ("MUST_HAVE", "did not meet an 'only' requirement")):
        if codes.get(code):
            tips.append(f"{codes[code]} {what} product{'s' if codes[code] != 1 else ''} {text}.")
    late = [m.group(1) for r in rejected if r["code"] == "TOO_LATE"
            for m in [re.search(r"earliest delivery (\d{4}-\d{2}-\d{2})", r["reason"])] if m]
    if late:
        first = min(late)
        tips.append(f"Earliest possible delivery is {first} ({late.count(first)} products).")
    if codes.get("CUTOFF_PASSED"):
        tips.append(f"{codes['CUTOFF_PASSED']} supplier items missed the order cut-off.")
    gone = sum(codes.get(c, 0) for c in ("OUT_OF_STOCK", "NOT_DELIVERABLE", "SALES_BLOCK", "NO_PRICE"))
    if gone:
        tips.append(f"{gone} product{'s' if gone != 1 else ''} cannot be ordered at the moment "
                    "(out of stock, not deliverable or blocked).")
    if relaxations:
        tips.append("Options that need your OK are listed below; accepting one re-runs the request with that change.")
    elif not tips:
        tips.append("Ask the buyer to clarify the request (see interpretation issues).")
    else:
        tips.append("No relaxation is possible without changing a hard requirement: escalate to a Valio Aimo sales rep.")
    return tips


SUBSTITUTE_ITEMS = 6  # at most this many missing items get substitutes (each costs search + pick + quote)


def _diet_screen(cands, diet, labels, by_id):
    """Split candidates by the hard diet requirements: (pass, ruled out, unverified).
    Ruled out = a look-alike or the data contradicts the need (e.g. CONTAINS milk);
    unverified = no evidence either way (fresh potatoes carry no vegan claim)."""
    kept, removed, unverified = [], [], []
    for c in cands:
        verdict, note = "pass", None
        for d in diet:
            status, ev, strength = catalog.diet_evidence(by_id[c["id"]], d)
            if status in ("near_miss", "contradicted"):
                verdict, note = "removed", f"{labels[d]}: {ev}"
                break
            if status == "fail":
                verdict, note = "unverified", f"{labels[d]}: {ev}"
            elif strength < 1.0 and verdict == "pass":
                note = f"{labels[d]}: {ev}; a person should check"
        c = dict(c, diet_note=note) if note else c
        {"pass": kept, "removed": removed, "unverified": unverified}[verdict].append(dict(c, reason=note) if verdict == "removed" else c)
    return kept, removed, unverified


def _substitute_options(items, trace, client, deliver_by, diet_labels, by_id=None):
    """Claude suggests substitutes for missing items; each is searched in the catalogue, screened
    by the same hard diet rules, the best real product is picked from its candidate IDs, and it
    is quoted live. Returns {key: [{"alt", "product", "quote"}]}, orderable ones first."""
    if not trace.get("parser", "").startswith("llm"):
        return {}
    labels = dict(diet_labels)
    try:
        subs = nlu.suggest_substitutes(items, trace["llm_calls"], list(labels.values()))
    except Exception as e:
        trace["notes"].append(f"substitute suggestions unavailable ({type(e).__name__})")
        return {}
    idx = catalog.load_index()
    by_id = by_id or {p["id"]: p for p in idx}
    groups = []
    for key, alts in subs.items():
        for alt in alts:
            cands = catalog.search_products(alt["search_terms_fi"], alt["search_terms_en"], idx, limit=8,
                                            terms_sv=alt.get("search_terms_sv"))
            cands = _diet_screen(cands, list(labels), labels, by_id)[0]  # substitutes obey the hard diet rules
            if cands:
                groups.append({"key": f"{key}_{len(groups)}", "item": key, "alt": alt, "ingredient": alt["name_en"],
                               "form": "any", "used_in": [f"substitute for {key}"], "candidates": cands})
    if not groups:
        return {}
    try:
        picks = nlu.pick_products(groups, trace["llm_calls"])
    except Exception as e:
        trace["notes"].append(f"substitute pick unavailable ({type(e).__name__})")
        return {}
    ids = sorted({p["product_id"] for p in picks.values() if p["product_id"] != "NONE"})
    if not ids:
        return {}
    date = deliver_by or (today() + dt.timedelta(days=1)).isoformat()
    deadline = dt.date.fromisoformat(deliver_by) if deliver_by else None
    quotes, _ = engine.quote_products(ids, client.price(ids, date), client.availability(ids, date),
                                      deadline, catalog.load_reliability())
    out = {}
    for g in groups:
        pid = picks.get(g["key"], {}).get("product_id", "NONE")
        if pid != "NONE":
            prod = next(c for c in g["candidates"] if c["id"] == pid)
            out.setdefault(g["item"], []).append({"alt": g["alt"], "product": prod, "quote": quotes.get(pid) or {}})
    for key in out:
        out[key].sort(key=lambda o: not o["quote"].get("orderable"))
    return out


def _relaxations(query, trace, stats, rejected, client, product_label, diet_labels, generic):
    """Changes a person may accept when nothing could be ordered. Hard requirements (diet,
    allergy) are never offered; each relaxation re-runs the request when accepted."""
    if trace.get("blocking"):
        return []
    rels = []

    def add(kind, text, evidence, change):
        rels.append({"id": f"R{len(rels) + 1}", "kind": kind, "text": text, "evidence": evidence, "change": change})

    late = [m.group(1) for r in rejected if r["code"] == "TOO_LATE"
            for m in [re.search(r"earliest delivery (\d{4}-\d{2}-\d{2})", r["reason"])] if m]
    if late:
        first = min(late)
        add("date", f"Deliver on {first} instead of {query.get('deliver_by')}",
            f"{late.count(first)} products can arrive on {first}", {"deliver_by": first})
    codes = collections.Counter(r["code"] for r in rejected)
    for b in query.get("avoid_brands") or []:
        n = sum(1 for r in rejected if r["code"] == "AVOIDED_BRAND" and r.get("constraint") == f"not {b}")
        if n:
            add("brand", f"Allow {b} again", f"{n} products were set aside only because of the brand",
                {"avoid_brands": [x for x in query["avoid_brands"] if x != b]})
    for must in query.get("must_have") or []:
        n = sum(1 for r in rejected if r["code"] == "MUST_HAVE" and r.get("constraint", "").startswith("only"))
        if n:
            name = {"brand": f"brands other than {(query.get('preferences') or {}).get('brand')}",
                    "organic": "non-organic products", "fat_level": "other fat levels"}[must]
            add("must_have", f"Accept {name}", f"{n} products were set aside by this 'only' requirement",
                {"must_have": [m for m in query["must_have"] if m != must]})
    # the product itself: missing, not identified, blocked by a hard diet need, or not orderable
    diet_blocked = any(r["code"] in ("DIET_FAIL", "DIET_NEAR_MISS") for r in rejected) and not stats.get("candidates")
    missing = stats.get("error") or (generic and not stats.get("identified")) or diet_blocked or (
        not late and any(codes.get(c) for c in ("OUT_OF_STOCK", "NOT_DELIVERABLE", "SALES_BLOCK", "CUTOFF_PASSED")))
    if missing and product_label:
        why = ("not in the catalogue" if stats.get("error") else
               f"no product meets {', '.join(diet_labels.values())}" if diet_blocked else "not orderable now")
        options = _substitute_options([{"key": "product", "name_en": product_label, "form": "any", "why_missing": why}],
                                      trace, client, query.get("deliver_by"), diet_labels)
        for o in [o for o in options.get("product", []) if o["quote"].get("orderable")]:
            alt, prod, q = o["alt"], o["product"], o["quote"]
            add("substitute", f"Use {alt['name_en']} instead of {product_label}: {prod['name_en'] or prod['name_fi']} "
                f"(#{prod['id']}), EUR {q['unit_price']:.2f}/{q.get('sales_unit') or 'unit'}, arrives {q.get('available_date')}",
                f"Claude: {alt['why_similar']} Diet requirements still apply.",
                {"product_type": "other", "product": {k: alt[k] for k in ("name_en", "search_terms_fi", "search_terms_en",
                                                                         "search_terms_sv")}})
    return rels


def relax_order(order_id, relaxation_id, by):
    """A person accepts relaxation `relaxation_id` of an order that could not be placed:
    the request is re-run with that change (no new interpretation), linked to the original."""
    orders = _load(ORDERS, {})
    o = orders.get(order_id)
    if not o:
        raise ValueError(f"unknown order {order_id}")
    if o.get("superseded_by"):
        if (o.get("relaxation_accepted") or {}).get("id") == relaxation_id and o["superseded_by"] in orders:
            return _replay(orders[o["superseded_by"]], f"relaxation {relaxation_id} was already accepted", "relaxation")
        raise ValueError(f"order {order_id} was already relaxed into {o['superseded_by']}")
    rel = next((r for r in o.get("relaxations") or [] if r["id"] == relaxation_id), None)
    if not rel:
        raise ValueError(f"order {order_id} has no relaxation {relaxation_id}")
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    accepted = dict(rel, accepted_by=by, accepted_at=now)
    query = json.loads(json.dumps(o["query"]))
    query.update(json.loads(json.dumps(rel["change"])))
    trace = json.loads(json.dumps(o["interpretation"]))
    trace["notes"].append(f"relaxed with the OK of {by}: {rel['text']} (from {order_id})")
    trace["blocking"] = nlu._validate(query)
    client = AimoClient(customer=o["policy"]["customer_id"])
    r = _run_order(o["request"], query, trace, o["policy"], client, True, accepted, order_id)
    orders = _load(ORDERS, {})
    o["superseded_by"], o["relaxation_accepted"] = r["order_id"], accepted
    orders[order_id] = o
    _save(ORDERS, orders)
    path = os.path.join(OUT, "audit", order_id + ".md")
    if os.path.exists(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## 7. Relaxation accepted\n\n- {rel['text']} ({rel['evidence']})\n"
                    f"- accepted by {by} at {now}\n- re-run as `{r['order_id']}`: `{r['status']}`\n")
    return r


# Where each guardrail check comes from (shown next to it in the UI): (label prefix, group, reference)
CHECK_SOURCES = [
    ("at least one feasible option", "Retrieval", "steps 3-4: nothing passed the product and live checks"),
    ("full quantity covered", "Your request", "the quantity you asked for"),
    ("spend EUR", "Buyer policy", "policy.json · auto_approve_max_total_eur"),
    ("price EUR", "Buyer policy", "policy.json · max_price_per_unit_eur"),
    ("price ceiling set", "Buyer policy", "policy.json · max_price_per_unit_eur"),
    ("single product line", "Buyer policy", "policy.json · allow_split_orders"),
    ("matches your stated preferences", "Your request", "the preferences you stated"),
    ("no preferences stated", "Your request", "the preferences you stated"),
    ("dietary claim verified", "Buyer policy", "policy.json · require_structured_diet_evidence"),
    ("quantity confirmed by Valio Aimo stock", "Buyer policy", "policy.json · allow_unconfirmed_supplier_stock"),
    ("delivery reliability", "Buyer policy", "policy.json · min_delivery_reliability"),
    ("unit conversion cross-checked", "Data check", "catalogue unit table vs price API comparison price"),
    ("request interpretation unambiguous", "Understanding", "step 2: rule parser and Claude cross-checked"),
    ("diet rule reviewed", "Reviewed rules", "rules/diet_overrides.json · reviewed"),
    ("product type reviewed", "Reviewed rules", "rules/product_types.json · reviewed"),
    ("every line is the requested product", "Claude's check", "step 3: Claude judged a line a variant"),
    ("relaxation accepted", "Your consent", "you accepted this change to the request"),
    ("live data consistent with the API contract", "Data check",
     "aimo_contract.py: the price / availability response breaks the documented API contract"),
    ("no other open order", "Buyer policy", "policy.json · duplicate_order_window_hours"),
]


def check_source(label):
    """(group, reference) of a guardrail check, from its label."""
    return next(((g, r) for prefix, g, r in CHECK_SOURCES if label.startswith(prefix)), ("Other", ""))


GENERIC_CANDIDATES = 40  # search results Claude judges for a product outside the reviewed types


def _generic_retrieve(query, trace):
    """Any product: word search over the whole catalogue, then Claude's verdict on every
    result (is the product / variant / contains it / different), then the same deterministic
    checks as typed retrieval. The guardrails send the order to a person afterwards."""
    idx = catalog.load_index()
    prod = query.get("product") or {}
    found = catalog.search_products(prod.get("search_terms_fi"), prod.get("search_terms_en"), idx,
                                    limit=GENERIC_CANDIDATES, terms_sv=prod.get("search_terms_sv"))
    stats = {"catalog_size": len(idx), "mode": "generic search + Claude verdict", "in_category": len(found)}
    if not found:
        return [], [], dict(stats, error=f"no catalogue product matches '{prod.get('name_en')}'")
    try:
        verdicts = nlu.classify_candidates(prod, found, trace["llm_calls"])
    except Exception as e:  # without a verdict nothing can be identified safely
        trace["issues"].append(f"product check by Claude unavailable ({type(e).__name__})")
        return [], [], dict(stats, error="products could not be identified (Claude unavailable)")
    by_id = {p["id"]: p for p in idx}
    return catalog.check_candidates([by_id[c["id"]] for c in found], verdicts, query, query.get("unit"), stats)


def _rule_reviews(diet, ptype):
    """Which human-reviewed rules this decision relied on (shown in the audit trail)."""
    out = [{"rule": f"diet {d}", "reviewed": catalog.DIET_RULES[d]["reviewed"],
            "by": catalog.DIET_RULES[d].get("reviewed_by"), "at": catalog.DIET_RULES[d].get("reviewed_at"),
            "sheet": catalog.DIET_RULES[d].get("review_sheet_sha256")} for d in diet]
    if ptype:
        out.append({"rule": f"product type {ptype.get('label')}", "reviewed": bool(ptype.get("reviewed")),
                    "by": ptype.get("reviewed_by"), "at": ptype.get("reviewed_at"),
                    "sheet": ptype.get("review_sheet_sha256")})
    return out


def _persist(response, audit_writer, persist):
    response["audit_file"] = audit_writer(response) if persist else None
    if persist:
        orders = _load(ORDERS, {})
        orders[response["order_id"]] = response
        _save(ORDERS, orders)


# --- Idempotency and duplicate orders -------------------------------------------------
def request_hash(query, customer):
    """Normalised identity of an order request: what is bought, how much, by when and under
    which requirements. Wording, quotes and ranking words ("cheapest") are not part of it, so
    "1000 l" and "1,000 litres" are the same request."""
    p = query.get("preferences") or {}
    other = query.get("product_type") == "other"
    key = {"customer": customer, "product_type": query.get("product_type"),
           "product": ((query.get("product") or {}).get("name_en") or "").strip().lower() if other else None,
           "quantity": float(query["quantity"]) if query.get("quantity") is not None else None,
           "unit": query.get("unit"), "deliver_by": query.get("deliver_by"),
           "diet": sorted(query.get("diet") or []), "must_have": sorted(query.get("must_have") or []),
           "avoid_brands": sorted(str(b).lower() for b in query.get("avoid_brands") or []),
           "preferences": {k: (str(p[k]).lower() if p.get(k) else None) for k in ("brand", "fat_level", "organic")}}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def _age_minutes(o, now):
    try:
        return (now - dt.datetime.fromisoformat(o["created_at"])).total_seconds() / 60
    except (KeyError, TypeError, ValueError):
        return float("inf")


def duplicate_checks_on(policy):
    """policy.json `check_duplicate_orders` (default on): the same-request replay within
    idempotency_window_minutes and the duplicate-order guardrail. A caller's own
    Idempotency-Key is honoured either way."""
    return bool(policy.get("check_duplicate_orders", True))


def _find_replay(policy, req_hash=None, client_key=None):
    """The latest earlier order this request repeats, or None (see run_request)."""
    window = policy.get("idempotency_window_minutes") or 0
    if not client_key and (not window or not duplicate_checks_on(policy)):
        return None
    now, best = dt.datetime.now().astimezone(), None
    for o in _load(ORDERS, {}).values():
        idem, age = o.get("idempotency") or {}, _age_minutes(o, now)
        if client_key:
            hit = idem.get("client_key") == client_key and age <= CLIENT_KEY_TTL_HOURS * 60
        else:
            hit = idem.get("request_hash") == req_hash and o.get("status") in OPEN_STATUSES and age <= window
        if hit and (best is None or o["created_at"] > best["created_at"]):
            best = o
    return best


def _replay(prior, why, kind="request"):
    """Answer a repeated request or decision with the stored result; nothing new happens."""
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    r = json.loads(json.dumps(prior))
    r["idempotent_replay"] = {"of": prior["order_id"], "at": now, "why": why, "kind": kind}
    note = f"Repeated {kind} ({why}): this is {prior['order_id']} again; nothing new was ordered."
    if kind == "request":
        note += " To order the same goods again, use --new (CLI) or 'Order again anyway' (UI); a person then approves it."
    r["next_steps"] = [note] + list(prior.get("next_steps") or [])
    path = os.path.join(OUT, "audit", prior["order_id"] + ".md")
    if os.path.exists(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n- {now}: repeated {kind} ({why}) answered with this order; nothing new was ordered\n")
    return r


def _possible_duplicates(query, policy):
    """Open orders from the last duplicate_order_window_hours for the same product and
    delivery date (None if the policy sets no window). Ordering the same goods twice can be
    intended, so this is a guardrail check for a person, not a refusal."""
    hours = policy.get("duplicate_order_window_hours")
    if not hours or not duplicate_checks_on(policy):
        return None
    def ident(q):
        name = ((q.get("product") or {}).get("name_en") or "").strip().lower() if q.get("product_type") == "other" else ""
        return q.get("product_type"), q.get("deliver_by"), name
    now, out = dt.datetime.now().astimezone(), []
    for o in _load(ORDERS, {}).values():
        q = o.get("query") or {}
        if o.get("status") in OPEN_STATUSES and _age_minutes(o, now) <= hours * 60 and ident(q) == ident(query):
            out.append(f"{o['order_id']} {o['status']}, {q.get('quantity') or 0:g} {q.get('unit') or ''}".strip())
    return out


# --- Proposal path (abstract requests: menu ideas, themes, events) ------------------
def _estimate(ing, quote, portions):
    """Rough cost from the API's comparison price (per KG or L), only when units line up."""
    if not portions or not ing.get("qty_per_portion") or not quote.get("comparison_price"):
        return None
    if ing.get("unit") not in ("KG", "L") or ing["unit"] != quote.get("comparison_unit"):
        return None
    qty = portions * ing["qty_per_portion"]
    return {"quantity": round(qty, 2), "unit": ing["unit"], "approx_eur": round(qty * quote["comparison_price"], 2),
            "basis": f"{portions:g} portions x {ing['qty_per_portion']:g} {ing['unit']} at "
                     f"EUR {quote['comparison_price']:.2f}/{quote['comparison_unit']} (comparison price)"}


def _product(c):
    return {"productId": c["id"], "name": c["name_fi"], "name_en": c["name_en"], "salesUnit": c["salesUnit"]}


def run_proposal(text, query, trace, policy, client, persist=True):
    """Recipes from Claude -> static catalogue search per ingredient -> Claude picks a real
    product ID per ingredient -> live price and availability. Status is always
    PROPOSAL_FOR_REVIEW: nothing is ordered, and the guardrails are not involved."""
    today_ = today()
    diet = [d for d in trace.get("proposal_diet") or query.get("diet") or [] if d in catalog.DIET_RULES]
    diet_labels = {d: catalog.DIET_RULES[d]["label"] for d in diet}
    rec = nlu.recommend(text, trace.get("culinary_goal") or text, today_, trace["llm_calls"], list(diet_labels.values()))
    idx = catalog.load_index()
    by_id = {p["id"]: p for p in idx}

    groups, by_name = [], {}  # one catalogue search per distinct ingredient
    for r in rec["recipes"]:
        for ing in r["ingredients"]:
            if ing["role"] == "pantry":
                continue
            key = ing["name_en"].strip().lower()
            if key not in by_name:
                by_name[key] = {"key": f"i{len(groups)}", "ingredient": ing["name_en"], "form": ing["form"],
                                "used_in": [], "roles": set(), "candidates": catalog.search_products(
                                    ing["search_terms_fi"], ing["search_terms_en"], idx,
                                    terms_sv=ing.get("search_terms_sv"))}
                groups.append(by_name[key])
            by_name[key]["used_in"].append(r["name"])
            by_name[key]["roles"].add(ing["role"])
    for g in groups:  # hard diet requirements apply to every ingredient, before Claude picks
        g["searched"] = len(g["candidates"])
        g["candidates"], g["diet_removed"], unverified = _diet_screen(g["candidates"], diet, diet_labels, by_id)
        g["diet_unverified"] = False
        if not g["candidates"] and unverified:  # products exist, the data just cannot prove the diet
            g["candidates"], g["diet_unverified"] = unverified, True
    try:
        picks = nlu.pick_products(groups, trace["llm_calls"])
    except Exception as e:  # degrade visibly: nothing is picked automatically; a person chooses
        trace["notes"].append(f"product pick by Claude unavailable ({type(e).__name__}); nothing picked automatically")
        picks = {}
        for g in groups:
            g["not_picked"] = bool(g["candidates"])
            picks[g["key"]] = {"product_id": "NONE", "runner_up": "NONE", "reason": (
                f"not picked (Claude unavailable): {len(g['candidates'])} candidates, e.g. "
                + "; ".join(c["name_en"][:40] for c in g["candidates"][:2])) if g["candidates"] else ""}

    deliver_by = query.get("deliver_by")
    date = deliver_by or (today_ + dt.timedelta(days=1)).isoformat()
    deadline = dt.date.fromisoformat(deliver_by) if deliver_by else None
    ids = sorted({x for p in picks.values() for x in (p["product_id"], p["runner_up"]) if x != "NONE"})
    quotes, now = {}, None
    if ids:  # only the picked products (and their runners-up) are queried live
        quotes, now = engine.quote_products(ids, client.price(ids, date), client.availability(ids, date),
                                            deadline, catalog.load_reliability())

    for g in groups:
        cand = {c["id"]: c for c in g["candidates"]}
        p = picks.get(g["key"]) or {"product_id": "NONE", "runner_up": "NONE", "reason": "no catalogue candidates"}
        first, ru = p["product_id"], p["runner_up"]
        q1 = quotes.get(first, {})
        if first == "NONE" and g.get("not_picked"):
            g["match"] = {"status": "NOT_PICKED", "note": p["reason"] + ". A person chooses."}
        elif first == "NONE":
            note = (p["reason"] or "no suitable product in the catalogue").rstrip(".")
            if not g["searched"]:
                note = f"'{g['ingredient']}' is not in the catalogue"
            elif g["diet_removed"] and not g["candidates"]:
                note = (f"{g['searched']} found, none meets {', '.join(diet_labels.values())} "
                        f"(e.g. {g['diet_removed'][0]['name_en'][:40]}: {g['diet_removed'][0]['reason']})")
            g["match"] = {"status": "NO_MATCH", "note": note}
        elif q1.get("orderable"):
            g["match"] = {"status": "OK", "product": cand[first], "quote": q1,
                          "note": p["reason"] + (" Delivery date not confirmed by the API." if q1.get("delivery_note") else "")
                          + (f" Arrives {q1['days_early']} day{'s' if q1['days_early'] != 1 else ''} before it is needed "
                             "(the API offers only its earliest date)." if q1.get("days_early") else "")}
        elif ru != "NONE" and quotes.get(ru, {}).get("orderable"):
            g["match"] = {"status": "SUBSTITUTE", "product": cand[ru], "quote": quotes[ru],
                          "first_choice": cand[first], "first_choice_quote": q1,
                          "note": f"first choice #{first} is not orderable ({q1.get('reason')}); runner-up shown"}
        else:
            g["match"] = {"status": "UNAVAILABLE", "product": cand[first], "quote": q1,
                          "note": q1.get("reason") or "not orderable"}

    for g in groups:  # a product whose diet the data cannot prove is shown, but a person must confirm it
        if g["diet_unverified"] and g["match"].get("product"):
            g["match"] = dict(g["match"], status="DIET_UNVERIFIED",
                              note=f"no catalogue {g['ingredient']} has {', '.join(diet_labels.values())} evidence; "
                                   f"a person must confirm this one")
    # ingredients the catalogue lacks, or key ones that cannot be ordered: Claude suggests substitutes,
    # which pass the same diet screen, are picked from real IDs and are quoted live
    missing = [g for g in groups if g["match"]["status"] == "NO_MATCH"
               or (g["match"]["status"] == "UNAVAILABLE" and "key" in g["roles"])][:SUBSTITUTE_ITEMS]
    subs = _substitute_options([{"key": g["key"], "name_en": g["ingredient"], "form": g["form"],
                                 "why_missing": g["match"]["note"]} for g in missing],
                               trace, client, deliver_by, diet_labels, by_id) if missing else {}
    for g in missing:
        opts = subs.get(g["key"]) or []
        ok = [o for o in opts if o["quote"].get("orderable")]
        if ok:
            o = ok[0]
            g["match"] = dict(g["match"], status="SUBSTITUTE_SUGGESTED", substitute=o["product"], quote=o["quote"],
                              product=o["product"],
                              note=f"{g['match']['note']}. Suggested instead: {o['alt']['name_en']} "
                                   f"({o['alt']['why_similar'].rstrip('.')}). Needs your OK.")
        elif opts:
            tried = "; ".join(f"{o['alt']['name_en']} (#{o['product']['id']}: {o['quote'].get('reason') or 'not orderable'})"
                              for o in opts[:2])
            g["match"]["note"] = f"{g['match']['note']}. Substitutes found but not orderable now: {tried}"
    portions = trace.get("portions")
    recipes = []
    for r in rec["recipes"]:
        lines = []
        for ing in r["ingredients"]:
            line = {"ingredient": ing["name_en"], "role": ing["role"], "form": ing["form"],
                    "qty_per_portion": ing["qty_per_portion"], "unit": ing["unit"]}
            if ing["role"] == "pantry":
                line.update(status="PANTRY", note="pantry item (not searched)" + (
                    f"; not checked against {', '.join(diet_labels.values())}" if diet_labels else ""))
            else:
                g = by_name[ing["name_en"].strip().lower()]
                m = g["match"]
                line.update(status=m["status"], note=m.get("note"), candidates_considered=len(g["candidates"]))
                if m.get("product"):
                    line.update(product=_product(m["product"]), quote=m["quote"])
                    if m["product"].get("diet_note"):
                        line["note"] = (line.get("note") or "") + f" Diet: {m['product']['diet_note']}."
                    est = _estimate(ing, m["quote"], portions)
                    if est:
                        line["estimate"] = est
                if m.get("first_choice"):
                    line["first_choice"] = _product(m["first_choice"])
            lines.append(line)
        recipes.append(dict({k: r[k] for k in ("name", "description", "why_it_fits", "course", "served")},
                            lines=lines))

    questions = []
    if not portions:
        questions.append("How many portions of each dish do you plan to serve?")
    if not deliver_by:
        questions.append(f"Which delivery date do you need? Prices and availability above assume {date}.")
    questions += [x for x in trace["clarifying_questions"]
                  if not re.search(r"portion|serving|guest|cover|people|date|when|deliver", x, re.I)]
    questions.append("Which dishes should I turn into an order?")

    response = {
        "order_id": f"PRP-{dt.datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}",
        "status": "PROPOSAL_FOR_REVIEW",
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "request": text, "query": query, "interpretation": trace, "policy": policy,
        "recipes": recipes, "avoid": rec.get("avoid") or [],
        "assumptions": (rec.get("assumptions") or []) + [f"Prices and availability checked for delivery on {date}."],
        "clarifying_questions": questions,
        "unsupported_requirements": trace["unsupported"],
        "diet_labels": diet_labels,
        "diet_removed": sum(len(g["diet_removed"]) for g in groups),
        "line_status": dict(collections.Counter(g["match"]["status"] for g in groups)),
        "retrieval_stats": {"catalog_size": len(idx), "ingredients_searched": len(groups),
                            "candidates": sum(len(g["candidates"]) for g in groups), "quoted_products": len(ids),
                            "server_time": now.isoformat() if now else None},
        "api_evidence": client.calls,
        "options": [], "guardrail_checks": [], "rejected": [], "explanation": [],
        "next_steps": ["Nothing was ordered: this is a proposal for a person to review.",
                       "Reply with the dishes, portions and delivery date. Milk lines can then go through the normal "
                       "order flow autonomously; other products can be ordered with a person's approval."],
    }
    _persist(response, write_proposal_audit, persist)
    return response


# --- Multi-line orders for named dishes (intent source, basket.py) -----------------------
def _basket_audit(b):
    return os.path.relpath(basket.write_audit(b, OUT), HERE)


def _split_recipe(query, trace):
    """A one-product order split by diet group, as a one-ingredient 'recipe': quantity per portion
    = the ordered quantity / all portions (4 L for 40 portions = 0.1 L each)."""
    ptype = catalog.PRODUCT_TYPES.get(query.get("product_type")) or {}
    prod = dict(query.get("product") or {})
    req = ptype.get("request") or {}
    name = prod.get("name_en") or ptype.get("label") or "product"
    portions, qty, unit = float(trace["portions"]), float(query["quantity"]), query["unit"]
    ing = {"name_en": name, "form": "any", "role": "key", "qty_per_portion": qty / portions, "unit": unit,
           **{f"search_terms_{lang}": list(prod.get(f"search_terms_{lang}") or req.get(lang) or []) for lang in ("fi", "en", "sv")}}
    return {"recipes": [{"name": name, "description": f"{qty:g} {unit} for {portions:g} portions", "why_it_fits": "",
                         "course": "other", "served": "cold", "ingredients": [ing]}], "avoid": [],
            "assumptions": [f"{qty:g} {unit} for {portions:g} portions = {qty / portions:g} {unit} per portion, "
                            "split by diet group"]}


def run_basket(text, query, trace, policy, client, persist=True, simulate=None, recipe=None):
    """Recipe (or one product split by diet group) -> lines per ingredient and diet group -> live
    quote -> tiers for red lines. Status AWAITING_CONFIRMATION: nothing is ordered until one
    confirmation of the whole order."""
    b = basket.run_basket(text, query, trace, policy, client, today(), _load(PREFS, DEFAULT_PREFS),
                          (lambda r, _w: _persist(r, _basket_audit, persist)), simulate, recipe)
    b["clock"] = clock_info()
    if persist:
        _store(b)
    return b


def _store(o):
    orders = _load(ORDERS, {})
    orders[o["order_id"]] = o
    _save(ORDERS, orders)


def _get(order_id):
    o = _load(ORDERS, {}).get(order_id)
    if not o:
        raise ValueError(f"unknown order {order_id}")
    return o


def _audit_path(order_id):
    return os.path.join(OUT, "audit", order_id + ".md")


def basket_preference(order_id, answer, by="buyer"):
    """Re-rank the options for a new preference or restore the default."""
    b = _get(order_id)
    if b.get("kind") != "basket":
        raise ValueError(f"{order_id} is not a multi-line order")
    basket.set_preference(b, answer, _load(PREFS, DEFAULT_PREFS))
    _store(b)
    basket.append_audit(b, _audit_path(order_id), "Preference answer",
                        [f"- {by} selected {answer or 'balanced'} sorting",
                         "- options re-ranked inside their tiers; no rule or tier changed"])
    return b


def basket_confirm(order_id, choices, acks, by, quantities=None):
    """One confirmation for the whole multi-line order, re-validated live (see basket.confirm)."""
    b = _get(order_id)
    if b.get("kind") != "basket":
        raise ValueError(f"{order_id} is not a multi-line order")
    if b["status"] == "CONFIRMED_BY_HUMAN":
        return _replay(b, "this order was already confirmed", "decision")
    if nlu.is_past_delivery_date(b.get("deliver_by"), today()):
        raise ValueError(nlu.PAST_DELIVERY_MESSAGE)
    prefs = _load(PREFS, DEFAULT_PREFS)
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    client = AimoClient(customer=b["policy"]["customer_id"])
    try:
        b, updates = basket.confirm(b, choices, acks, by, client, prefs, now, quantities)
    except AimoApiError as e:  # fail closed: never commit a quote that could not be re-checked
        raise AimoApiError(f"confirmation not recorded: the live quote could not be re-checked ({e})")
    if updates:
        for u in updates:
            prefs["history"].append({"order_id": order_id, "at": now, "line": u["line"], "chose": u["chose"],
                                     "weights_before": u["before"], "weights_after": u["after"]})
        _save(PREFS, prefs)
    if b["status"] == "CONFIRMED_BY_HUMAN":
        b["cancel_until"] = _cancel_until(b)
    _store(b)
    basket.append_audit(b, _audit_path(order_id), "Confirmation (whole order)")
    return b


# --- Cancel or change an order before its cut-off -------------------------------------------
CANCELLABLE = ("PLACED_AUTONOMOUSLY", "APPROVED_BY_HUMAN", "CONFIRMED_BY_HUMAN")


def _committed_lines(o):
    """Public lines of what an order committed to (single-product orders and baskets)."""
    if o.get("kind") == "basket":
        return [ln["chosen"] for ln in o["lines"] if ln.get("chosen")]
    choose = (o.get("human_decision") or {}).get("option") or 1
    return o["options"][choose - 1]["lines"] if o.get("options") else []


def _cancel_until(o):
    """The earliest order cut-off of the committed lines: until then the order can be cancelled or changed."""
    cut = [ln.get("orderCutOff") for ln in _committed_lines(o) if ln.get("orderCutOff")]
    return min(cut) if cut else None


def cancel_order(order_id, by, reason="cancelled by the buyer"):
    """Undo an order (placed by the agent, or approved / confirmed by a person) before the
    earliest order cut-off of its lines. After the cut-off the wholesaler may already be
    picking it, so the agent refuses and says who to contact."""
    o = _get(order_id)
    if o["status"] == "CANCELLED_BY_HUMAN":
        return _replay(o, "this order was already cancelled", "decision")
    if o["status"] not in CANCELLABLE:
        raise ValueError(f"order {order_id} is {o['status']}: there is nothing to cancel")
    until = _cancel_until(o)
    now = dt.datetime.now().astimezone()
    if until and now > dt.datetime.fromisoformat(until):
        raise ValueError(f"the order cut-off {until[:16].replace('T', ' ')} has passed: the order can no longer be "
                         "cancelled here; contact Valio Aimo customer service")
    at = now.isoformat(timespec="seconds")
    o["cancellation"] = {"by": by, "at": at, "reason": reason, "previous_status": o["status"], "cancel_until": until}
    o["status"] = "CANCELLED_BY_HUMAN"
    _store(o)
    path = _audit_path(order_id)
    if os.path.exists(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Cancellation\n\n- CANCELLED by {by} at {at} ({reason}), before the order cut-off {until}\n"
                    f"- previous status: `{o['cancellation']['previous_status']}`\n- final status: `{o['status']}`\n")
    return o


def change_order(order_id, text, policy, by, use_llm=True):
    """Change an order before its cut-off: cancel it, then run the edited request as a new order
    (linked both ways). If the new request fails, the original order is restored."""
    original = _get(order_id)
    cancel_order(order_id, by, reason="replaced by a changed request")
    try:
        r = run_request(text, policy, use_llm=use_llm, allow_duplicate=True)
        if (r.get("interpretation") or {}).get("past_delivery_date"):
            raise ValueError(nlu.PAST_DELIVERY_MESSAGE)
    except Exception:
        orders = _load(ORDERS, {})
        o = orders[order_id]
        o["status"] = o["cancellation"]["previous_status"]
        o.pop("cancellation", None)
        _save(ORDERS, orders)
        raise
    orders = _load(ORDERS, {})
    orders[order_id]["replaced_by"] = r["order_id"]
    if r["order_id"] in orders:
        orders[r["order_id"]]["replaces"] = order_id
    _save(ORDERS, orders)
    r["replaces"] = order_id
    r["replaced_request"] = original["request"]
    return r


# --- Human approval + preference learning -------------------------------------------
def decide_order(order_id, approve, choose, by):
    """A person approves option `choose` of a pending order, or rejects it. Approval commits,
    so the chosen option is first re-checked against a fresh live quote (engine.revalidate).
    If the price rose, the stock fell or the cut-off passed, nothing is placed: the order
    becomes STALE_QUOTE and the same query is re-planned with live data for a new decision
    (returned instead of the old order). If the quote cannot be re-checked, the approval is
    refused. Repeating a recorded decision returns the stored result (idempotent)."""
    orders = _load(ORDERS, {})
    o = orders.get(order_id)
    if not o:
        raise ValueError(f"unknown order {order_id}")
    if o["status"] != "PENDING_APPROVAL":
        hd = o.get("human_decision") or {}
        if hd.get("action") == ("approve" if approve else "reject") and (not approve or hd.get("option") == choose):
            target = orders.get(o.get("superseded_by")) if o["status"] == "STALE_QUOTE" else o
            if target:
                return _replay(target, "this decision was already recorded", "decision")
        raise ValueError(f"order {order_id} is {o['status']}, nothing to decide")
    if approve and not 1 <= choose <= len(o["options"]):
        raise ValueError(f"option {choose} does not exist (1-{len(o['options'])})")
    if approve and nlu.is_past_delivery_date((o.get("query") or {}).get("deliver_by"), today()):
        raise ValueError(nlu.PAST_DELIVERY_MESSAGE)
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    if not approve:
        o["status"] = "REJECTED_BY_HUMAN"
        o["human_decision"] = {"by": by, "at": now, "action": "reject"}
    else:
        chosen = o["options"][choose - 1]
        o["human_decision"] = {"by": by, "at": now, "action": "approve", "option": choose}
        o["revalidation"] = rv = _revalidate(o, chosen, now)
        if rv["changes"]:
            return _replan_stale(orders, o, by, now)
        o["status"] = "APPROVED_BY_HUMAN"
        o["cancel_until"] = _cancel_until(o)
        if choose != 1:
            prefs = _load(PREFS, DEFAULT_PREFS)
            old, new = engine.learn_from_choice(prefs, o["options"][0]["utilities"], chosen["utilities"])
            prefs["history"].append({"order_id": order_id, "at": now, "chose_option": choose,
                                     "weights_before": old, "weights_after": new})
            _save(PREFS, prefs)
            o["human_decision"]["preference_update"] = {"before": old, "after": new}
    orders[order_id] = o
    _save(ORDERS, orders)
    append_audit(o)
    return o


def _revalidate(o, chosen, now):
    """Fresh price and availability for the chosen option only (commit-time check)."""
    q = o["query"]
    deadline = dt.date.fromisoformat(q["deliver_by"]) if q.get("deliver_by") else None
    date = q.get("deliver_by") or chosen["lines"][0]["availableDate"]
    ids = [ln["productId"] for ln in chosen["lines"]]
    client = AimoClient(customer=o["policy"]["customer_id"])
    try:
        changes, fresh, server_now = engine.revalidate(chosen["lines"], client.price(ids, date),
                                                       client.availability(ids, date), deadline)
    except AimoApiError as e:  # fail closed: never commit a quote that could not be re-checked
        raise AimoApiError(f"approval not recorded: the live quote could not be re-checked ({e})")
    rv = {"at": now, "server_time": server_now.isoformat(), "option": chosen["option"], "ok": not changes,
          "changes": changes, "lines": fresh, "api_evidence": client.calls}
    if not changes:
        rv["committed_total_eur"] = round(sum(x["lineTotal"] for x in fresh), 2)
        rv["quoted_total_eur"] = chosen["total_eur"]
    return rv


def _replan_stale(orders, o, by, now):
    """The approved quote no longer holds: mark the order STALE_QUOTE and re-plan the same
    query with live data. The new order (returned) needs a new decision or orders within
    the agent's authority, like any other run."""
    order_id = o["order_id"]
    o["status"] = "STALE_QUOTE"
    orders[order_id] = o
    _save(ORDERS, orders)  # before the re-plan, so the duplicate check does not count it
    trace = json.loads(json.dumps(o["interpretation"]))
    trace["notes"].append(f"re-planned with live data: the quote of {order_id} changed before approval")
    stale = {"order_id": order_id, "changes": o["revalidation"]["changes"], "attempted_by": by, "at": now,
             "option": o["human_decision"]["option"]}
    r = _run_order(o["request"], json.loads(json.dumps(o["query"])), trace, o["policy"],
                   AimoClient(customer=o["policy"]["customer_id"]), True, stale=stale)
    orders = _load(ORDERS, {})
    o["superseded_by"] = r["order_id"]
    orders[order_id] = o
    _save(ORDERS, orders)
    append_audit(o)
    return r


# --- Audit trail ---------------------------------------------------------------
def write_audit(r):
    os.makedirs(os.path.join(OUT, "audit"), exist_ok=True)
    path = os.path.join(OUT, "audit", r["order_id"] + ".md")
    q = r["query"]
    L = [f"# Audit trail {r['order_id']}", "",
         f"- **Created:** {r['created_at']}  ",
         f"- **Buyer / policy:** {r['policy']['buyer']} / {r['policy']['policy_version']}  ",
         f"- **Status:** `{r['status']}`", "",
         "## 1. Request", "", f"> {r['request']}", "",
         "## 2. Interpretation", "", f"Parser: {r['interpretation']['parser']}", "",
         "```json", json.dumps(q, ensure_ascii=False, indent=2), "```", ""]
    L += [f"- note: {n}" for n in r["interpretation"]["notes"]]
    L += [f"- **issue:** {n}" for n in r["interpretation"]["issues"]]
    L += _llm_call_lines(r)
    s = r["retrieval_stats"]
    generic = s.get("mode", "").startswith("generic")
    L += ["", "## 3. Retrieval (" + (s.get("mode") or "static catalog rules") + ")", "",
          f"{s.get('catalog_size')} products -> {s.get('in_category', 0)} {'search results' if generic else 'in category'} -> "
          + (f"{s.get('identified', 0)} identified by Claude -> " if generic else "")
          + f"{s.get('candidates', 0)} candidates -> {s.get('feasible_offers', 0)} feasible after live checks.", "",
          "| stage | product | rule | reason |", "|---|---|---|---|"]
    L += [f"| {x['stage']} | {x['id']} {x['name'][:50]} | {x['code']} | {x['reason']} |" for x in r["rejected"]]
    L += ["", "## 4. Live data evidence", "", "| endpoint | at | requestedDate | products | response sha256 |",
          "|---|---|---|---|---|"]
    L += [f"| {c['endpoint']} | {c['at']} | {c['requestedDate']} | {c['products']} | `{c['response_sha256'][:16]}...` |"
          for c in r["api_evidence"]]
    L += [f"- contract anomaly in `{c['endpoint']}`: {a['code']}: {a['text']}"
          for c in r["api_evidence"] for a in c.get("contract_anomalies") or []]
    L += ["", "## 5. Options", "", "Weights: " + ", ".join(f"{k} {v:.2f}" for k, v in r["weights"].items()), "",
          f"| # | products | EUR/{(r['options'][0]['measure'] if r['options'] else 'unit')} | total EUR | score | tags |",
          "|---|---|---|---|---|---|"]
    for p in r["options"]:
        prods = " + ".join("{} x{:g}".format(ln["productId"], ln["quantity"]) for ln in p["lines"])
        L.append(f"| {p['option']} | {prods} | {p['eur_per_unit']:.2f} | {p['total_eur']:,.2f} | "
                 f"{p['score']:.3f} | {', '.join(p['tags'])} |")
    L += ["", "**Why option 1:**", ""] + [f"- {e}" for e in r["explanation"]]
    L += ["", "## 6. Guardrails", ""] + [f"- {'PASS' if c['ok'] else 'FAIL'}: {c['check']}" for c in r["guardrail_checks"]]
    L += [f"- rule used: {x['rule']}: " + (f"reviewed by {x['by']} on {x['at']}" if x["reviewed"] else "NOT reviewed")
          + (f" (signed review sheet sha256 `{x['sheet'][:16]}...`)" if x.get("sheet") else "")
          for x in r.get("rule_reviews") or []]
    L += ["", f"**Decision:** `{r['status']}`", ""] + [f"- next: {x}" for x in r["next_steps"]]
    L += [f"- requirement: {c['name']} ({c['strength']})" for c in r.get("constraints") or []]
    L += [f"- relaxation {x['id']} (needs a person's OK): {x['text']} ({x['evidence']})" for x in r.get("relaxations") or []]
    if r.get("relaxed_from"):
        x = r["relaxation"]
        L += [f"- relaxed from `{r['relaxed_from']}`: {x['text']}, accepted by {x['accepted_by']} at {x['accepted_at']}"]
    if r.get("stale_quote"):
        x = r["stale_quote"]
        L += [f"- re-planned from `{x['order_id']}`: the approval by {x['attempted_by']} at {x['at']} was not placed "
              "because the quote changed: " + "; ".join(x["changes"])]
    idem = r.get("idempotency") or {}
    L += [f"- idempotency: request hash `{idem.get('request_hash')}`"
          + (f", caller key `{idem['client_key']}`" if idem.get("client_key") else "")]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return os.path.relpath(path, HERE)


def _llm_call_lines(r):
    return [f"- LLM call `{c['stage']}`: {c['model']}, effort {c['effort']}, stop {c['stop_reason']}, "
            f"{c['input_tokens']} in / {c['output_tokens']} out tokens, {c['seconds']} s"
            for c in r["interpretation"].get("llm_calls") or []]


def _quote_text(qt):
    if not qt or qt.get("unit_price") is None:
        return "no price"
    s = f"EUR {qt['unit_price']:.2f}/{qt.get('sales_unit') or 'unit'}"
    if qt.get("comparison_price"):
        s += f" (EUR {qt['comparison_price']:.2f}/{qt['comparison_unit']})"
    return s


def write_proposal_audit(r):
    os.makedirs(os.path.join(OUT, "audit"), exist_ok=True)
    path = os.path.join(OUT, "audit", r["order_id"] + ".md")
    it, s = r["interpretation"], r["retrieval_stats"]
    L = [f"# Audit trail {r['order_id']}", "",
         f"- **Created:** {r['created_at']}  ",
         f"- **Buyer / policy:** {r['policy']['buyer']} / {r['policy']['policy_version']}  ",
         f"- **Status:** `{r['status']}` (nothing ordered)", "",
         "## 1. Request", "", f"> {r['request']}", "",
         "## 2. Interpretation", "", f"Parser: {it['parser']}. Intent: `{it['intent']}`.", "",
         f"- goal: {it.get('culinary_goal')}", f"- portions: {it.get('portions')}"]
    L += [f"- note: {n}" for n in it["notes"]] + [f"- **issue:** {n}" for n in it["issues"]]
    L += _llm_call_lines(r)
    L += ["", "## 3. Recipes and catalogue matches", "",
          f"{s['ingredients_searched']} ingredients searched in {s['catalog_size']} products, "
          f"{s['candidates']} candidates, {s['quoted_products']} products quoted live. "
          "Products were picked by Claude from per-ingredient candidate ID lists (constrained output).", ""]
    for rc in r["recipes"]:
        L += [f"### {rc['name']} ({rc['course']}, {rc['served']})", "", rc["description"], "",
              "| ingredient | role | product | price | delivery | status | note |", "|---|---|---|---|---|---|---|"]
        for ln in rc["lines"]:
            pr, qt = ln.get("product"), ln.get("quote") or {}
            prod = f"{pr['productId']} {pr['name'][:45]}" if pr else "-"
            L.append(f"| {ln['ingredient']} | {ln['role']} | {prod} | {_quote_text(qt) if pr else '-'} | "
                     f"{qt.get('available_date') or '-'} | {ln['status']} | {(ln.get('note') or '')[:90]} |")
        L.append("")
    L += ["## 4. Live data evidence", "", "| endpoint | at | requestedDate | products | response sha256 |",
          "|---|---|---|---|---|"]
    L += [f"| {c['endpoint']} | {c['at']} | {c['requestedDate']} | {c['products']} | `{c['response_sha256'][:16]}...` |"
          for c in r["api_evidence"]]
    L += ["", "## 5. Decision", "", f"**Status:** `{r['status']}`. No guardrail ran because nothing was ordered.", ""]
    L += [f"- question: {q}" for q in r["clarifying_questions"]] + [f"- next: {x}" for x in r["next_steps"]]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return os.path.relpath(path, HERE)


def append_audit(o):
    path = os.path.join(OUT, "audit", o["order_id"] + ".md")
    d = o["human_decision"]
    L = ["", "## 7. Human decision", "", f"- {d['action'].upper()} by {d['by']} at {d['at']}"]
    if d.get("option"):
        L.append(f"- chosen option: {d['option']}")
    if d.get("preference_update"):
        L.append("- preference weights learned from this choice: " + ", ".join(
            f"{k} {d['preference_update']['before'][k]:.2f}->{v:.2f}" for k, v in d["preference_update"]["after"].items()))
    rv = o.get("revalidation")
    if rv and rv["ok"]:
        L.append(f"- re-checked live at approval (server time {rv['server_time']}): price and availability still hold; "
                 f"committed EUR {rv['committed_total_eur']:,.2f} (quoted EUR {rv['quoted_total_eur']:,.2f})")
    elif rv:
        L += ["- re-checked live at approval: the quote no longer holds, so nothing was placed:"]
        L += [f"  - {c}" for c in rv["changes"]]
    L += [f"- evidence: {c['endpoint']} at {c['at']}, response sha256 `{c['response_sha256'][:16]}...`"
          for c in (rv or {}).get("api_evidence") or []]
    if o.get("superseded_by"):
        L.append(f"- re-planned as `{o['superseded_by']}`")
    L.append(f"- final status: `{o['status']}`")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --- Terminal rendering ----------------------------------------------------------
def render_proposal(r):
    it, s = r["interpretation"], r["retrieval_stats"]
    print(f"\n━━ Trusted Trade Agent ━━  {r['order_id']}\n\"{r['request']}\"\n")
    print(f"① Interpretation  ({it['parser']})  intent=recommend")
    print(f"   goal: {it.get('culinary_goal')}")
    for n in it["notes"] + it["issues"]:
        print(f"   · {n}")
    print(f"\n② Proposal  {len(r['recipes'])} dishes · {s['ingredients_searched']} ingredients searched "
          f"→ {s['candidates']} candidates → {s['quoted_products']} quoted live (server time {str(s.get('server_time'))[:16]})")
    if r["avoid"]:
        print("   already planned (not repeated): " + ", ".join(r["avoid"]))
    for rc in r["recipes"]:
        print(f"\n   ▸ {rc['name']}  ({rc['course']}, {rc['served']})\n     {rc['description']}")
        for ln in rc["lines"]:
            pr, qt = ln.get("product"), ln.get("quote") or {}
            prod = f"{pr['productId']} {pr['name'][:34]}" if pr else "-"
            per = f"{ln['qty_per_portion']:g} {ln['unit']}/portion" if ln.get("qty_per_portion") and ln.get("unit") else ""
            when = qt.get("available_date") or ""
            print(f"     {ln['ingredient'][:22]:<22} {prod:<41} {(_quote_text(qt) if pr else ''):<30} "
                  f"{when:<10} {ln['status']:<11} {per}")
    print("\n③ Questions for the buyer")
    for q in r["clarifying_questions"]:
        print(f"   ? {q}")
    print(f"\n→ {r['status']}  (nothing ordered)")
    for n in r["next_steps"]:
        print(f"   {n}")
    print(f"   audit: {r['audit_file']}   order book: out/orders.json\n")


def render_basket(r):
    print(f"\n━━ Trusted Trade Agent ━━  {r['order_id']}  (multi-line order)\n\"{r['request']}\"\n")
    if r.get("simulation"):
        print(f"   !! {r['simulation']['label']}: {', '.join(r['simulation']['changed_products'])}\n")
    print("① Groups  " + " + ".join((f"{g['portions']:g} portions" if g["portions"] is not None else "? portions")
                                   + f" {'regular' if g['key'] == 'all' else g['name'].split(' portions ', 1)[-1]}"
                                   for g in r["groups"]))
    print(f"② Lines   (deliver by {r['deliver_by']}, prices held until {str(r['quote']['valid_until'])[:16]})")
    for ln in r["lines"]:
        o = basket._selected(ln)
        mark = "●" if ln["light"] == "green" else "○"
        print(f"   {mark} {ln['light']:<5} {ln['ingredient'][:24]:<24} {ln['group_name'][:28]:<28} {ln['need']['basis']:<26} "
              + (f"{o['tier']:<12} {basket._opt_text(o)}" if o else f"choose: {ln.get('why')}"))
        if ln["light"] == "red":
            for t in basket.TIERS:
                for x in ln["tiers"].get(t) or []:
                    print(f"        {x['id']:<22} {basket._opt_text(x)[:90]}  relaxed: {'; '.join(x['relaxed'])[:60] or 'nothing'}")
            if ln["excluded"]:
                print(f"        ruled out: {len(ln['excluded'])} ({'; '.join(sorted({x['reason'][:40] for x in ln['excluded']}))})")
    print(f"\n→ {r['status']}  total EUR {r['totals']['eur']:,.2f}")
    for n in r["next_steps"]:
        print(f"   {n}")
    for t in r.get("tasks") or []:
        print(f"   arrival check: {t['check']}")
    print(f"   audit: {r.get('audit_file')}   confirm in the web UI, or: python3 agent.py confirm {r['order_id']} "
          "[--pick LINE=OPTION] [--ack LINE]\n")


def render(r):
    if r.get("kind") == "basket":
        return render_basket(r)
    if r["status"] == "PROPOSAL_FOR_REVIEW":
        return render_proposal(r)
    q, s = r["query"], r["retrieval_stats"]
    ok = lambda b: "✓" if b else "✗"
    print(f"\n━━ Trusted Trade Agent ━━  {r['order_id']}\n\"{r['request']}\"\n")
    print(f"① Interpretation  ({r['interpretation']['parser']})")
    print(f"   product={q.get('product_type')}  diet={q.get('diet')}  qty={q.get('quantity') or 0:g} {q.get('unit')}  "
          f"deliver_by={q.get('deliver_by')}  optimize={q.get('optimize_for')}  prefs={q.get('preferences')}")
    for n in r["interpretation"]["notes"] + r["interpretation"]["issues"]:
        print(f"   · {n}")
    if s.get("error"):
        print(f"\n② Retrieval  skipped: {s['error']}")
    else:
        generic = s.get("mode", "").startswith("generic")
        print(f"\n② Retrieval  {s.get('catalog_size')} products → {s.get('in_category', 0)} "
              + ("search results → " + str(s.get("identified", 0)) + " identified by Claude " if generic else "in category ")
              + f"→ {s.get('candidates', 0)} candidates → {s.get('feasible_offers', 0)} feasible "
              f"(server time {str(s.get('server_time'))[:16]})")
    by = collections.defaultdict(list)
    for x in r["rejected"]:
        by[(x["stage"], x["code"], x["reason"] if x["code"] != "CUTOFF_PASSED" else x["reason"][:38])].append(x)
    for (stage, code, reason), xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if code == "NOT_PRODUCT_TYPE":
            continue
        print(f"   ✗ {len(xs):>2} {stage:<7} {code:<16} {reason[:70]}  e.g. {xs[0]['id']} {xs[0]['name'][:28]}")
    print("\n③ Options  (weights: " + ", ".join(f"{k} {v:.2f}" for k, v in r["weights"].items()) + ")")
    m = r["options"][0]["measure"] if r["options"] else "L"
    print(f"   {'#':<2} {'products':<52} {'EUR/' + m:>6} {'total EUR':>10} {'score':>6}  tags")
    for p in r["options"]:
        names = " + ".join(f"{ln['productId']} {ln['name'][:28]} x{ln['quantity']:g}" for ln in p["lines"])
        star = "★ " if p["option"] == 1 else ""
        print(f"   {p['option']:<2} {names[:52]:<52} {p['eur_per_unit']:>6.2f} {p['total_eur']:>10,.2f} "
              f"{p['score']:>6.3f}  {star}{', '.join(p['tags'])}{'' if p['pareto_optimal'] else ' (dominated)'}")
    print("\n   Why option 1:")
    for e in r["explanation"]:
        print(f"   · {e}")
    print("\n④ Guardrails  (policy " + r["policy"]["policy_version"] + ")")
    for c in r["guardrail_checks"]:
        print(f"   {ok(c['ok'])} {c['check']}")
    print(f"\n→ {r['status']}" + (f"  (relaxed from {r['relaxed_from']}: {r['relaxation']['text']})" if r.get("relaxed_from") else ""))
    if r.get("stale_quote"):
        x = r["stale_quote"]
        print(f"   re-planned: the approval of {x['order_id']} was not placed because the quote changed:")
        for c in x["changes"]:
            print(f"     · {c}")
    for x in r.get("relaxations") or []:
        print(f"   option {x['id']} (needs your OK): {x['text']}  [{x['evidence']}]")
    if r.get("relaxations"):
        print(f"   accept one with: python3 agent.py relax {r['order_id']} R1")
    for n in r["next_steps"]:
        print(f"   {n}")
    print(f"   audit: {r['audit_file']}   order book: out/orders.json\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="+", help="request text, or: approve|reject ORDER_ID")
    ap.add_argument("--json", action="store_true", help="print the structured response only")
    ap.add_argument("--no-llm", action="store_true", help="use the deterministic parser only")
    ap.add_argument("--policy", default=os.path.join(HERE, "policy.json"))
    ap.add_argument("--choose", type=int, default=1, help="option number to approve")
    ap.add_argument("--by", default="buyer@hotel (demo)")
    ap.add_argument("--new", action="store_true", help="order again even if the same request was just ordered")
    ap.add_argument("--idempotency-key", help="caller's key: a retry with the same key returns the same order")
    ap.add_argument("--pick", action="append", default=[], help="confirm: LINE=OPTION for a red line")
    ap.add_argument("--ack", action="append", default=[], help="confirm: LINE whose incomplete-evidence warning you accept")
    ap.add_argument("--qty", action="append", default=[], help="confirm: LINE=UNITS sales units for a line")
    ap.add_argument("--prefer", choices=list(basket.PREFERENCE_OPTIONS), help="confirm: answer the preference question first")
    ap.add_argument("--simulate", help="fault-injection fixture from scenarios/ (e.g. hospital_delay)")
    a = ap.parse_args()
    load_dotenv()

    if a.words[0] == "cancel" and len(a.words) == 2:
        try:
            o = cancel_order(a.words[1], a.by)
        except ValueError as e:
            sys.exit(str(e))
        print(f"{o['order_id']}: {o['status']} (until the cut-off {(o.get('cancellation') or {}).get('cancel_until')})")
        return
    if a.words[0] == "confirm" and len(a.words) == 2:
        try:
            if a.prefer:
                basket_preference(a.words[1], a.prefer, a.by)
            r = basket_confirm(a.words[1], dict(x.split("=", 1) for x in a.pick), a.ack, a.by,
                               {k: float(v) for k, v in (x.split("=", 1) for x in a.qty)})
        except (ValueError, AimoApiError) as e:
            sys.exit(str(e))
        render(r)
        return

    if a.words[0] == "relax" and len(a.words) == 3:
        try:
            r = relax_order(a.words[1], a.words[2], a.by)
        except (ValueError, AimoApiError) as e:
            sys.exit(str(e))
        render(r)
        return
    if a.words[0] in ("approve", "reject") and len(a.words) == 2:
        try:
            o = decide_order(a.words[1], a.words[0] == "approve", a.choose, a.by)
        except (ValueError, AimoApiError) as e:
            sys.exit(str(e))
        if o.get("stale_quote") and not o.get("idempotent_replay"):
            print(f"{a.words[1]}: approval NOT placed, the quote changed. Fresh plan:")
            render(o)
            return
        if o.get("idempotent_replay"):
            print("(this decision was already recorded; nothing changed)")
        print(f"{o['order_id']}: {o['status']} by {(o.get('human_decision') or {}).get('by') or a.by}")
        rv = o.get("revalidation")
        if rv and rv["ok"]:
            print(f"re-checked live at approval: price and stock still hold; committed EUR {rv['committed_total_eur']:,.2f}")
        upd = o["human_decision"].get("preference_update")
        if upd:
            print("learned weights: " + ", ".join(f"{k} {upd['before'][k]:.2f}->{v:.2f}" for k, v in upd["after"].items()))
        return
    with open(a.policy, encoding="utf-8") as f:
        policy = json.load(f)
    try:
        r = run_request(" ".join(a.words), policy, use_llm=not a.no_llm, idempotency_key=a.idempotency_key,
                        allow_duplicate=a.new, simulate=a.simulate or os.environ.get("AGENT_SIMULATE"))
    except AimoApiError as e:
        sys.exit(f"Valio Aimo API error: {e}")
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        render(r)


if __name__ == "__main__":
    main()
