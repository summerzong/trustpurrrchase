"""Static catalog: slim index over the Valio Aimo product file + rule-based retrieval.

Only static data is used here. Live price/availability is queried afterwards for
the narrowed-down set only (as the API docs require).
"""
from __future__ import annotations

import collections
import csv
import difflib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AIMO_DATA_DIR", os.path.join(HERE, "..", "data"))
PRODUCT_FILE = os.path.join(DATA_DIR, "valio_aimo_junction26_product_data.json")
RELIABILITY_FILE = os.path.join(DATA_DIR, "valio_aimo_junction26_delivery_reliability.csv")
CARBON_FILE = os.path.join(DATA_DIR, "valio_aimo_junction26_carbon_footprint.csv")
CACHE_FILE = os.path.join(HERE, ".cache", "catalog_index.json")
INDEX_VERSION = 3  # bump whenever _slim() changes, so a stale cache is rebuilt
RULES_DIR = os.path.join(HERE, "rules")
MEASURES = ("L", "KG", "PCS")


def _read_rules_file(name, default):
    path = os.path.join(RULES_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Product types (explicit, auditable, easy to extend) ----------------------
# Reviewed definitions live in rules/product_types.json (milk today). Anything else is
# retrieved generically (search_products + Claude's per-candidate verdict) and an order
# for it always needs a person.
PRODUCT_TYPES = _read_rules_file("product_types.json", {"types": {}}).get("types", {})

# Single source of truth for dietary requirements, built from three files in rules/:
#   diet_vocab.json      GENERATED from the data by build_diet_vocab.py: every GS1 claim code,
#                        its type/element grammar, derived relations, suggested contradictions
#   diet_terms.json      synonyms: words buyers use (-> nlu.DIET_TERMS) and words in product names
#   diet_overrides.json  human policy: thresholds, contradictions, composite concepts, reviewed flag
# nlu builds both the LLM schema enum and the rule parser's DIET_TERMS from DIET_RULES, so
# the two parsers cannot drift apart. Evidence sources, strongest first: GS1 claims,
# nonAllergen FREE_FROM entries and nutrient values, then product names.
# See diet_evidence() for how conflicting sources are resolved.
NUTRIENT_LABEL = {"LACS": "lactose", "FAT": "fat", "SALTEQ": "salt", "FIBTG": "fibre", "PRO-": "protein",
                  "SUGAR-": "sugars"}


def build_diet_rules(vocab, terms, overrides):
    """Merge generated vocabulary, synonyms and policy into one rule per diet code.

    Every claim code in the data gets a rule. Relations come from the claim grammar
    (FREE_FROM_X satisfies LOW_X; LOW_X is a look-alike of FREE_FROM_X) plus
    `also_satisfied_by` in the overrides. Codes without reviewed=true are still
    verified, but agent/engine send any order relying on them to a person.
    """
    vc = vocab.get("codes", {})
    oc = overrides.get("codes", {})
    tc = {k: v for k, v in terms.items() if not k.startswith("_")}
    unknown = sorted(set(tc) - set(vc) - set(oc))
    if vc and unknown:  # (without a vocabulary yet, i.e. before the first build_diet_vocab.py run, skip)
        raise ValueError(f"rules/diet_terms.json has codes without a rule (typo?): {unknown}")

    def label(code):
        return (oc.get(code) or {}).get("label") or (vc.get(code) or {}).get("label") or code.lower().replace("_", " ")

    def own_names(code):
        return list((tc.get(code) or {}).get("product_name") or [])

    rules = {}
    for code in sorted(set(vc) | set(oc)):
        v, o, t = vc.get(code) or {}, oc.get(code) or {}, tc.get(code) or {}
        satisfied_by = [c for c in v.get("satisfied_by", []) + o.get("also_satisfied_by", []) if c != code]
        near = [c for c in v.get("near_miss", []) if c != code]
        nid = o.get("nutrient") or v.get("nutrient")
        if nid and o.get("nutrient_max") is not None:
            nutrient = (nid, "max", o["nutrient_max"])
        elif nid and o.get("nutrient_min") is not None:
            nutrient = (nid, "min", o["nutrient_min"])
        else:
            nutrient = None
        lab, request = label(code), list(t.get("request") or [])
        desc = o.get("description")
        if not desc:
            syn = [x for x in request if x != lab][:4]
            desc = lab + (f" ({', '.join(syn)})" if syn else "") + "."
            if satisfied_by:
                desc += " " + " and ".join(label(c) for c in satisfied_by).capitalize() + " products also satisfy it."
            if near:
                desc += " NOT the same as " + " / ".join(label(c) for c in near) + "."
        name_terms = own_names(code)
        for c in satisfied_by:
            name_terms += [x for x in own_names(c) if x not in name_terms]
        near_terms = []
        for c in near:
            near_terms += [x for x in own_names(c) if x not in near_terms and x not in name_terms]
        rules[code] = {
            "label": lab,
            "description": desc,
            "pos_claims": ([code] if code in vc else []) + satisfied_by,
            "pos_free_from": list(o.get("free_from", [])),
            "inherent_categories": list(o.get("inherent_categories", [])),  # whole produce that meets it naturally
            "nutrient": nutrient,                                   # (id, "max"|"min", g per 100 g) or None
            "nutrient_contradicts": o.get("nutrient_contradicts", True),
            "contradict_allergens": list(o.get("contradict_allergens", [])),
            "contradict_may_contain": bool(o.get("contradict_may_contain", False)),
            "name_terms": name_terms,
            # look-alikes that must NOT pass as this requirement
            "near_miss_label": " / ".join(label(c) for c in near),
            "near_miss_claims": near,
            "near_miss_terms": near_terms,
            "request_terms": request,                               # -> nlu.DIET_TERMS
            "reviewed": bool(o.get("reviewed", False)),
            "reviewed_by": o.get("reviewed_by"),
            "reviewed_at": o.get("reviewed_at"),
            "review_sheet_sha256": o.get("review_sheet_sha256"),   # the review sheet the reviewer signed
            "products": v.get("products", 0),
        }
    return rules


DIET_RULES = build_diet_rules(_read_rules_file("diet_vocab.json", {"codes": {}}),
                              _read_rules_file("diet_terms.json", {}),
                              _read_rules_file("diet_overrides.json", {"codes": {}}))

VOL_RE = re.compile(r"(?:(\d+)\s*x\s*)?(\d+(?:[.,]\d+)?)\s*(l|dl|cl|ml)\b", re.I)
VOL_FACTOR = {"l": 1.0, "dl": 0.1, "cl": 0.01, "ml": 0.001}


def _text(items, lang):
    for it in items or []:
        if it.get("language") == lang:
            return it.get("value") or ""
    return ""


def _slim(p):
    s = p.get("synkkaData") or {}
    claims, labels, free_from, nutrients = set(), set(), set(), {}
    by_unit = {"CONTAINS": set(), "MAY_CONTAIN": set()}
    for c in s.get("classifications", []):
        for v in c.get("values", []):
            if c["name"] == "nutritionalClaim":
                claims.add(v["id"])
            elif c["name"] == "allergen" and v.get("unit") in by_unit:
                by_unit[v["unit"]].add(v["id"])
            elif c["name"] == "nonAllergen" and v.get("unit") == "FREE_FROM":
                free_from.add(v["id"])
            elif c["name"] == "packagingMarkedLabel":
                labels.add(v["id"])
            elif c["name"] == "nutritionFact" and v.get("value") is not None:
                nutrients[v["id"]] = v["value"]
    return {
        "id": p["id"],
        "name_fi": _text(s.get("names"), "fi"),
        "name_en": _text(s.get("names"), "en"),
        "name_sv": _text(s.get("names"), "sv"),
        "category": p.get("category"),
        "salesUnit": p.get("salesUnit"),
        "baseUnit": p.get("baseUnit"),
        "allowedLotSize": p.get("allowedLotSize"),
        "units": {u["unitId"]: u["sizeInBaseUnits"] for u in p.get("units", [])},
        "vendor": p.get("vendorName", ""),
        "brand": s.get("brand", ""),
        "country": p.get("countryOfOrigin", ""),
        "claims": sorted(claims),
        "allergens": sorted(by_unit["CONTAINS"]),
        "allergens_may_contain": sorted(by_unit["MAY_CONTAIN"]),
        "free_from": sorted(free_from),
        "labels": sorted(labels),
        "nutrients": nutrients,          # per 100 g, e.g. {"LACS": 0.0, "FAT": 1.5}
        "lacs": nutrients.get("LACS"),
    }


_INDEX = None  # in-memory copy for long-running use (web server)


def load_index():
    """Slim index of the 87 MB product file, cached on disk after the first run."""
    global _INDEX
    if _INDEX is None:
        _INDEX = _load_index()
    return _INDEX


def _load_index():
    if not os.path.exists(PRODUCT_FILE):
        raise FileNotFoundError(f"{os.path.normpath(PRODUCT_FILE)} is missing: run `python3 fetch_data.py` in demo/")
    if os.path.exists(CACHE_FILE) and os.path.getmtime(CACHE_FILE) >= os.path.getmtime(PRODUCT_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cached = json.load(f)
        if isinstance(cached, dict) and cached.get("version") == INDEX_VERSION:
            return cached["index"]
    with open(PRODUCT_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    idx = [_slim(p) for p in raw if not p.get("deleted")]
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"version": INDEX_VERSION, "index": idx}, f, ensure_ascii=False)
    return idx


def _load_csv(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2 and row[1]:
                    out[row[0]] = float(row[1])
    return out


def load_reliability():
    return _load_csv(RELIABILITY_FILE)


def load_carbon():
    return _load_csv(CARBON_FILE)


# --- Derived attributes ---------------------------------------------------------
def parse_volume_l(name):
    m = VOL_RE.search(name or "")
    if not m:
        return None
    mult = float(m.group(1)) if m.group(1) else 1.0
    return mult * float(m.group(2).replace(",", ".")) * VOL_FACTOR[m.group(3).lower()]


def litres_per_sales_unit(p):
    """Returns (litres, method). ZML is the litre unit in this dataset."""
    units = p["units"]
    s = units.get(p["salesUnit"])
    if not s:
        return None, None
    if units.get("ZML"):
        return s / units["ZML"], "unit table (ZML = litre)"
    v = parse_volume_l(p["name_fi"]) or parse_volume_l(p["name_en"])
    if v:
        return v * s / (units.get(p["baseUnit"]) or 1.0), "pack volume from name x pack size"
    return None, None


WEIGHT_RE = re.compile(r"(?:(\d+)\s*x\s*)?(\d+(?:[.,]\d+)?)\s*(kg|g)\b", re.I)
WHOLE_EGG_CATEGORY = "10590"
EGG_PACK_COUNT_RE = re.compile(r"\b(?:[sml](\d+)|(\d+)\s*(?:kpl|pcs|st))\b", re.I)


def amount_per_sales_unit(p, measure):
    """Returns (amount of `measure` per sales unit, method), or (None, None) if unknown.

    L uses litres_per_sales_unit. KG uses the unit table when it has a KG unit; otherwise a
    weight in the name, but only when the name clearly describes one base unit (no 'N x'
    and sold per base unit, or a single pack weight). Ambiguous cases return None, and the
    live check derives the amount from the price API's comparison price instead.
    PCS counts base units per sales unit, except for whole eggs, where the name says
    how many individual eggs each base unit contains (e.g. M10 is a carton of 10)."""
    if measure == "L":
        return litres_per_sales_unit(p)
    units = p["units"]
    s = units.get(p["salesUnit"])
    if not s:
        return None, None
    base = units.get(p["baseUnit"]) or 1.0
    if measure == "KG":
        if units.get("KG"):
            return s / units["KG"], "unit table (KG)"
        m = WEIGHT_RE.search(p["name_fi"]) or WEIGHT_RE.search(p["name_en"])
        if m and not m.group(1):
            kg = float(m.group(2).replace(",", ".")) * (1.0 if m.group(3).lower() == "kg" else 0.001)
            return kg * s / base, "pack weight from name x pack size"
        return None, None
    if measure == "PCS":
        if p.get("category") == WHOLE_EGG_CATEGORY:
            m = EGG_PACK_COUNT_RE.search(p["name_fi"]) or EGG_PACK_COUNT_RE.search(p["name_en"])
            if not m:  # never count an unknown number of eggs as one egg
                return None, None
            return int(m.group(1) or m.group(2)) * s / base, "egg count from name x pack size"
        return s / base, "consumer packs per sales unit"
    return None, None


# --- Brands (index generated from the data) -------------------------------------
_BRANDS = {}  # id(index) -> brand index
_SUFFIXES = r"(?:a|ä|n|ta|tä|lta|ltä|lle|sta|stä|ssa|ssä|ksi|na|nä|in)?"  # Finnish case endings ("Arlaa", "Valion")
COMMON_NAME_MENTIONS = 20  # a one-word brand mentioned in this many names of other brands is a common word


def norm_brand(s):
    return re.sub(r"\s+", " ", re.sub(r"[®™©]", "", s or "")).strip().lower()


def _same_family(a, b):
    return a == b or a.startswith(b + " ") or b.startswith(a + " ")


def brand_index(index=None):
    """{normalised brand: {"display", "products", "ambiguous", "why"}} for every brand in the data.

    A brand is ambiguous as a request word when it is 1-3 letters long ('hp', 'sun') or is a
    common word that appears in many product names of *other* brand families ('fresh',
    'classic', 'juusto'). Ambiguous brands are never detected by the rule parser; Claude may
    still name them, and they are then snapped and confirmed by a person."""
    index = index if index is not None else load_index()
    key = id(index)
    if key in _BRANDS:
        return _BRANDS[key]
    spell = collections.defaultdict(collections.Counter)
    for p in index:
        if p["brand"].strip():
            spell[norm_brand(p["brand"])][p["brand"].strip()] += 1
    mentions = collections.defaultdict(collections.Counter)  # name word -> brand -> products
    for p, (tf, te, _ts) in zip(index, _tokens(index)):
        for w in set(tf) | set(te):
            mentions[w][norm_brand(p["brand"])] += 1
    out = {}
    for b, sp in spell.items():
        own = sum(sp.values())
        other = sum(n for ob, n in mentions.get(b, {}).items() if not _same_family(ob, b)) if " " not in b else 0
        # 'fresh' (120 own, 123 other) is a common word; 'Fazer' (349 own, 44 in sub-brand names) is not
        common = other >= COMMON_NAME_MENTIONS and other >= 0.5 * own
        why = "1-3 letters" if len(b) <= 3 else (f"common word: in {other} names of other brands" if common else "")
        out[b] = {"display": sp.most_common(1)[0][0], "products": own, "ambiguous": bool(why), "why": why}
    _BRANDS.clear()
    _BRANDS[key] = out
    return out


_BRAND_RE = {}


def find_brands(text, index=None):
    """Unambiguous brands mentioned in `text`, longest first, as [(normalised brand, start, end)]."""
    brands = brand_index(index)
    key = id(brands)
    if key not in _BRAND_RE:
        names = sorted((b for b, v in brands.items() if not v["ambiguous"]), key=len, reverse=True)
        _BRAND_RE.clear()
        _BRAND_RE[key] = re.compile(r"(?<![a-zåäö0-9])(" + "|".join(re.escape(b) for b in names) + r")"
                                    + _SUFFIXES + r"(?![a-zåäö0-9])")
    t = re.sub(r"[®™©]", "", text.lower())
    return [(m.group(1), m.start(), m.end()) for m in _BRAND_RE[key].finditer(t)]


def resolve_brand(name, index=None):
    """Snap a brand name (for example from Claude) to the data: returns (display or None, status).
    status: exact | family | fuzzy | ambiguous | unknown."""
    brands = brand_index(index)
    n = norm_brand(name)
    if not n:
        return None, "unknown"
    if n in brands:
        return brands[n]["display"], "ambiguous" if brands[n]["ambiguous"] else "exact"
    family = [b for b in brands if b.startswith(n + " ")]
    if family:
        return name.strip(), "family"
    close = difflib.get_close_matches(n, list(brands), n=1, cutoff=0.8)
    if close:
        return brands[close[0]]["display"], "fuzzy"
    return None, "unknown"


def brand_matches(p, brand):
    """True if product p belongs to the brand family (brand field), or its vendor/name carries it."""
    b = norm_brand(brand)
    return bool(b) and (_same_family(norm_brand(p["brand"]), b)
                        or b in (p["vendor"] + " " + p["name_fi"]).lower())


def fat_level(p):
    n = (p["name_fi"] + " " + p["name_en"]).lower()
    if "rasvaton" in n or "fat free" in n or "fatfree" in n:
        return "skimmed"
    if "kevyt" in n or "semi" in n or "low fat" in n:
        return "semi-skimmed"
    if "täys" in n or "whole" in n or "3 %" in n:
        return "whole"
    return None


ORGANIC_LABELS = {"EU_ORGANIC_FARMING", "KRAV_MARK", "SOIL_ASSOCIATION_ORGANIC_SYMBOL"}


def is_organic(p):
    """Organic certification label first; the word in the name as a fallback."""
    return bool(ORGANIC_LABELS & set(p.get("labels") or [])) or "luomu" in p["name_fi"].lower() \
        or "organic" in p["name_en"].lower()


def preference_match(p, prefs):
    """Fraction of stated soft preferences this product satisfies (1.0 if none stated)."""
    checks = []
    if prefs.get("brand"):
        b = prefs["brand"].lower()
        checks.append(b in (p["brand"] + " " + p["vendor"] + " " + p["name_fi"]).lower())
    if prefs.get("fat_level"):
        checks.append(fat_level(p) == prefs["fat_level"])
    if prefs.get("organic"):
        checks.append(is_organic(p))
    return (sum(checks) / len(checks)) if checks else 1.0


# --- Rule-based retrieval -------------------------------------------------------
def _catalog_diet_evidence(p, rule, claims):
    hits = [c for c in rule["pos_claims"] if c in claims]
    evidence = ["GS1 claim " + "/".join(hits)] if hits else []
    free_from = rule.get("pos_free_from") or []
    if free_from and all(a in (p.get("free_from") or []) for a in free_from):
        evidence.append("allergen data: free from " + "/".join(free_from))
    if p.get("category") in (rule.get("inherent_categories") or []):
        evidence.append(f"category rule: whole unprocessed produce (category {p['category']}) is naturally {rule['label']}")
    return evidence, hits


def diet_evidence(p, diet, rules=None):
    """Returns (status, evidence_text, strength 0..1) for one dietary requirement.

    status is "pass", "near_miss", "contradicted" or "fail". All signals are collected first
    and then resolved, so a source that contradicts another is never silently ignored:

      structured positive, no contradiction   -> pass, 1.0
      structured positive + contradiction     -> pass, 0.5 ("CONFLICT: ...", a human decides)
      no structured positive, look-alike      -> near_miss (rejected)
      no structured positive, contradiction   -> contradicted (rejected: e.g. CONTAINS milk for dairy-free)
      product name only                       -> pass, 0.5 (a human decides)
      nothing                                 -> fail: no evidence either way (unverified)
    """
    rule = (rules if rules is not None else DIET_RULES).get(diet)
    if rule is None:
        return "fail", f"no rule for {diet}", 0.0
    name = (p["name_fi"] + " " + p["name_en"]).lower()
    claims = set(p["claims"])
    pos, hit = _catalog_diet_evidence(p, rule, claims)
    contra = []
    nid, op, lim = rule.get("nutrient") or (None, None, None)
    val = (p.get("nutrients") or {}).get(nid) if nid else None
    if val is not None:
        label = NUTRIENT_LABEL.get(nid, nid)
        if (val <= lim) if op == "max" else (val >= lim):
            pos.append(f"nutrition: {label} {val:g} g")
        elif rule.get("nutrient_contradicts", True):
            contra.append(f"nutrition: {label} {val:g} g/100 g ({'limit' if op == 'max' else 'minimum'} {lim:g} g)")
    for a in rule["contradict_allergens"]:
        if a in p["allergens"]:
            contra.append(f"allergen list says it CONTAINS {a}")
        elif rule.get("contradict_may_contain") and a in (p.get("allergens_may_contain") or []):
            contra.append(f"allergen list says it MAY CONTAIN {a}")
    near_claims = [c for c in rule["near_miss_claims"] if c in claims]
    if near_claims and not hit:  # the manufacturer itself only claims the weaker property
        contra.append("GS1 claim " + "/".join(near_claims) + " only")

    if pos:
        if not contra:
            return "pass", " + ".join(pos), 1.0
        return "pass", "CONFLICT: " + " + ".join(pos) + " BUT " + "; ".join(contra), 0.5
    near = [t for t in rule["near_miss_terms"] if t in name] + near_claims
    if near:
        return "near_miss", f"'{near[0]}' = {rule['near_miss_label']}, which is NOT {rule['label']}", 0.0
    if contra:
        return "contradicted", f"not {rule['label']}: {contra[0]}", 0.0
    if any(t in name for t in rule["name_terms"]):
        return "pass", "product name only (no structured claim)", 0.5
    label = rule["label"]
    return "fail", (f"no evidence for '{label}'" if label.startswith("no ") else f"no {label} evidence"), 0.0


ALLERGEN_VERDICTS = ("pass", "warn", "reject")


def _smt_allergen_verdict(p, codes):
    """Check what the catalog evidence entails for the hospital's two allergen requirements."""
    try:
        import z3
    except ImportError as exc:
        raise ValueError("The milk and gluten allergy check needs z3-solver. Install demo/requirements.txt.") from exc

    ctx = z3.Context()  # each request thread uses its own Z3 context
    solver = z3.Solver(ctx=ctx)
    solver.set(timeout=100)
    reasons = []
    for code in codes:
        rule = DIET_RULES[code]
        label = rule["label"]
        positive, _ = _catalog_diet_evidence(p, rule, set(p["claims"]))
        contains = [a for a in rule["contradict_allergens"] if a in p["allergens"]]
        may_contain = [a for a in rule["contradict_allergens"]
                       if rule["contradict_may_contain"] and a in (p.get("allergens_may_contain") or [])]
        # Preserve the demo policy: confirmed presence wins, and positive evidence
        # alongside MAY_CONTAIN needs a label check.
        blocks = bool(contains or (may_contain and not positive))
        meets = z3.Bool(code, ctx=ctx)
        if blocks:
            solver.add(z3.Not(meets))
        elif positive and not may_contain:
            solver.add(meets)
        # An unconstrained variable represents missing or inconclusive evidence.
        violation = solver.check(z3.Not(meets))
        smt = {"violation": str(violation)}
        if violation == z3.sat:
            smt["compliance"] = str(solver.check(meets))
        if violation == z3.unsat:
            verdict = "pass"
        elif smt.get("compliance") == "unsat" or blocks:
            verdict = "reject"
        else:
            verdict = "warn"
        evidence = positive + [f"allergen list says it CONTAINS {a}" for a in contains]
        evidence += [f"allergen list says it MAY CONTAIN {a}" for a in may_contain]
        if "unknown" in smt.values():
            evidence.append("Z3 could not complete the check")
        reasons.append({"code": code, "label": label, "verdict": verdict, "smt": smt,
                        "evidence": "; ".join(evidence) or f"no data that shows it is {label}"})
    return max((r["verdict"] for r in reasons), key=ALLERGEN_VERDICTS.index), reasons


def allergen_verdict(p, codes, allergy=True, rules=None):
    """Two-strength guardrail for one product and a group's dietary needs (allergy or diet).

    Returns (verdict, reasons) with verdict:
      reject  a known violation: the data contradicts the need (CONTAINS milk for dairy-free), a
              look-alike (low-lactose for lactose-free), or, for an allergy, a claim contradicted by
              the product's own CONTAINS list. Never offered.
      warn    evidence is missing or weak: no data at all, the product name only, or a claim that
              only a MAY_CONTAIN entry contradicts. Missing data is not safety: offered only with an
              explicit acknowledgement and a label check on arrival.
      pass    structured evidence (GS1 claim, allergen data, nutrient value, reviewed produce rule).
    """
    if allergy and rules is None:
        codes = tuple(codes)
        if set(codes) == {"DAIRY_FREE", "FREE_FROM_GLUTEN"}:
            return _smt_allergen_verdict(p, codes)
    verdict, reasons = "pass", []
    for d in codes:
        rule = (rules if rules is not None else DIET_RULES).get(d) or {}
        label = rule.get("label", d)
        status, ev, strength = diet_evidence(p, d, rules)
        if status in ("near_miss", "contradicted"):
            v = "reject"
        elif status == "fail":
            v = "warn"
            ev = f"no data that shows it is {label}" if not label.startswith("no ") else ev
        elif strength >= 1.0:
            v = "pass"
        elif "CONFLICT" in ev and allergy and "CONTAINS" in ev:
            v, ev = "reject", f"not {label}: {ev}"
        else:
            v = "warn"
        reasons.append({"code": d, "label": label, "verdict": v, "evidence": ev})
        if ALLERGEN_VERDICTS.index(v) > ALLERGEN_VERDICTS.index(verdict):
            verdict = v
    return verdict, reasons


# --- Generic lexical search (recommendation path and generic order path) --------
_WORD_RE = re.compile(r"[a-zåäöéü]+")
_TOKENS = {}  # id(index) -> per-product token lists (fi, en, sv), built once per index


def _tokens(index):
    key = id(index)
    if key not in _TOKENS:
        _TOKENS.clear()
        _TOKENS[key] = [(_WORD_RE.findall(p["name_fi"].lower()), _WORD_RE.findall(p["name_en"].lower()),
                         _WORD_RE.findall((p.get("name_sv") or "").lower())) for p in index]
    return _TOKENS[key]


def _en_word_match(term_word, tokens):
    """Whole-word match with simple English plurals ('rhubarb' ~ 'rhubarbs', 'berry' ~ 'berries')."""
    forms = {term_word, term_word + "s", term_word + "es"}
    if term_word.endswith("y"):
        forms.add(term_word[:-1] + "ies")
    return any(t in forms for t in tokens)


def _prefix_score(terms, tokens):
    """3 if every word of a term is a whole word in the name, 2 if every word starts a word, else 0."""
    best = 0
    for words in terms:
        if all(any(t == w for t in tokens) for w in words):
            best = max(best, 3)
        elif all(any(t.startswith(w) for t in tokens) for w in words):
            best = max(best, 2)
    return best


def search_products(terms_fi, terms_en, index=None, limit=8, terms_sv=None):
    """Static catalogue search for one ingredient or product. Returns up to `limit` slim candidates.

    English terms must match whole words in the English name, so 'pork' cannot hit
    'PORKKANA' (carrot). Finnish and Swedish terms may match the start of a word, because
    compounds ('raparperikakku' = rhubarb cake, 'rabarberpaj') and inflections ('porsaan')
    are common. Ranking is deliberately simple; Claude judges the candidates afterwards.
    """
    index = index if index is not None else load_index()
    split = lambda terms: [w for t in terms or [] for w in [_WORD_RE.findall(t.lower())] if w]
    fi, en, sv = split(terms_fi), split(terms_en), split(terms_sv)
    scored = []
    for p, (tf, te, ts) in zip(index, _tokens(index)):
        s_en = 3 if any(all(_en_word_match(w, te) for w in words) for words in en) else 0
        score = _prefix_score(fi, tf) + s_en + _prefix_score(sv, ts)
        if score:
            scored.append((-score, len(p["name_fi"]), p))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [{"id": p["id"], "name_fi": p["name_fi"], "name_en": p["name_en"], "name_sv": p.get("name_sv", ""),
             "salesUnit": p["salesUnit"], "category": p["category"]} for _, _, p in scored[:limit]]


# --- Retrieval: shared deterministic checks -------------------------------------
def _check(p, query, measure, reject, density=1.0):
    """Checks every candidate must pass, whichever way it was found: excluded brands,
    diet evidence and conversion to the order's measure. Returns a candidate dict or None."""
    avoid = [b for b in query.get("avoid_brands") or [] if brand_matches(p, b)]
    if avoid:
        reject(p, "AVOIDED_BRAND", f"brand {avoid[0]} is excluded by the buyer", constraint=f"not {avoid[0]}")
        return None
    prefs = query.get("preferences") or {}
    for must in query.get("must_have") or []:  # a preference the buyer made mandatory ('only Arla')
        if must == "brand" and prefs.get("brand") and not brand_matches(p, prefs["brand"]):
            reject(p, "MUST_HAVE", f"not {prefs['brand']}, and you asked for {prefs['brand']} only",
                   constraint=f"only {prefs['brand']}")
            return None
        if must == "organic" and not is_organic(p):
            reject(p, "MUST_HAVE", "not organic, and you asked for organic only", constraint="only organic")
            return None
        if must == "fat_level" and prefs.get("fat_level") and fat_level(p) != prefs["fat_level"]:
            reject(p, "MUST_HAVE", f"not {prefs['fat_level']}, and you asked for {prefs['fat_level']} only",
                   constraint=f"only {prefs['fat_level']}")
            return None
    evidence, strength = [], 1.0
    for d in query.get("diet") or []:
        status, ev, st = diet_evidence(p, d)
        if status != "pass":
            reject(p, "DIET_NEAR_MISS" if status == "near_miss" else "DIET_FAIL", ev, constraint=d)
            return None
        evidence.append(ev)
        strength = min(strength, st)
    per, method = amount_per_sales_unit(p, measure)
    if measure == "L" and not per:  # KG / PCS may still be derived from the price API's comparison price
        reject(p, "NO_UNIT_CONVERSION", "cannot convert sales unit to litres")
        return None
    if measure == "PCS" and p.get("category") == WHOLE_EGG_CATEGORY and not per:
        reject(p, "NO_UNIT_CONVERSION", "catalogue does not state how many eggs are in the pack")
        return None
    packs = (p["units"].get(p["salesUnit"]) or 1.0) / (p["units"].get(p["baseUnit"]) or 1.0)
    return dict(
        p,
        name=p["name_fi"],
        measure=measure,
        per_unit=per,
        per_unit_method=method,
        packs_per_unit=packs,
        kg_per_measure={"KG": 1.0, "L": density}.get(measure),  # for CO2e (data is per kg)
        fat=fat_level(p),
        evidence="; ".join(evidence),
        evidence_strength=strength,
        pref_match=preference_match(p, query.get("preferences") or {}),
        variant_note=None,
    )


def _rejector(rejected, stage):
    def reject(p, code, reason, constraint=None):
        x = {"id": p["id"], "name": p["name_fi"], "stage": stage, "code": code, "reason": reason}
        if constraint:
            x["constraint"] = constraint  # which requirement ruled it out (for 'why no result')
        rejected.append(x)
    return reject


def retrieve(query, index):
    """Rule-based retrieval for a product type defined in rules/product_types.json.
    Returns (candidates, rejected, stats)."""
    ptype = PRODUCT_TYPES.get(query.get("product_type"))
    stats = {"catalog_size": len(index), "mode": "product type rules"}
    if ptype is None:
        return [], [], dict(stats, error=f"product type '{query.get('product_type')}' not in ontology")
    in_cat = [p for p in index if p["category"] in ptype["categories"]]
    stats["in_category"] = len(in_cat)
    cands, rejected = [], []
    reject = _rejector(rejected, "catalog")
    for p in in_cat:
        name = (p["name_fi"] + " " + p["name_en"]).lower()
        if not any(t in name for t in ptype["include"]):
            reject(p, "NOT_PRODUCT_TYPE", f"not a {query['product_type']} product")
            continue
        bad = next((why for t, why in ptype["exclude"].items() if t in name), None)
        if bad:
            reject(p, "WRONG_SUBTYPE", bad)
            continue
        c = _check(p, query, ptype["measure"], reject, ptype.get("density_kg_per_l", 1.0))
        if c:
            cands.append(c)
    stats["candidates"] = len(cands)
    return cands, rejected, stats


VERDICT_TEXT = {"variant": "a variant, not the requested product",
                "contains_it": "contains or is made with it", "different_product": "a different product"}


def check_candidates(products, verdicts, query, measure, stats=None):
    """Generic retrieval, second half: `products` came from search_products and `verdicts`
    from Claude (nlu.classify_candidates). Products that are not the requested product are
    rejected with Claude's reason; the rest pass the same deterministic checks as typed
    retrieval. Returns (candidates, rejected, stats)."""
    stats = dict(stats or {}, mode="generic search + Claude verdict")
    cands, rejected = [], []
    reject_llm = _rejector(rejected, "llm")
    reject = _rejector(rejected, "catalog")
    kept = 0
    for p in products:
        v = verdicts.get(p["id"]) or {"verdict": "different_product", "reason": "no verdict"}
        if v["verdict"] != "is_the_product":
            reject_llm(p, "NOT_THE_PRODUCT", f"Claude: {VERDICT_TEXT.get(v['verdict'], v['verdict'])} ({v['reason']})")
            continue
        kept += 1
        c = _check(p, query, measure, reject)
        if c:
            cands.append(c)
    stats.update(identified=kept, candidates=len(cands))
    return cands, rejected, stats
