#!/usr/bin/env python3
"""Write a review sheet for a product type, with every product behind every number.

    python3 review_product_type.py milk                     # -> rules/review/type_milk.md
    python3 review_product_type.py --sign milk --by "Name"  # after reading the sheet

A product type in rules/product_types.json may order autonomously only after a person has
checked what its rules accept, what they exclude and what they might miss, and signed the
sheet (see review_diet_rule.py for how signing works: the sheet hash is stored with the
reviewer's name, and the offline tests fail when the sheet changes later).
"""
from __future__ import annotations

import collections
import os
import sys

import catalog
from review_diet_rule import REVIEW_DIR, sign

PATH = os.path.join(catalog.RULES_DIR, "product_types.json")
EXAMPLES = 5  # per category in the "possible misses" table


def _name(p):
    return (p["name_fi"] + " " + p["name_en"]).lower()


def _row(p, *cols):
    return "| " + " | ".join([p["id"], (p["name_en"] or p["name_fi"])[:60]] + [str(c) for c in cols]) + " |"


def sheet(type_key, idx):
    t = catalog.PRODUCT_TYPES[type_key]
    measure = t["measure"]
    in_cat = [p for p in idx if p["category"] in t["categories"]]
    unmatched, excluded, accepted = [], [], []
    for p in in_cat:
        name = _name(p)
        if not any(term in name for term in t["include"]):
            unmatched.append(p)
            continue
        hit = next(((term, why) for term, why in t["exclude"].items() if term in name), None)
        if hit:
            excluded.append((p, hit))
        else:
            accepted.append(p)
    outside = [p for p in idx if p["category"] not in t["categories"] and any(term in _name(p) for term in t["include"])]
    by_cat = collections.Counter(p["category"] for p in outside)
    status = (f"reviewed by {t['reviewed_by']} on {t['reviewed_at']}" if t.get("reviewed")
              else "**NOT reviewed**: orders for this product type always need a person")
    L = [f"# Review sheet: product type `{type_key}` ({t['label']})", "", f"Status: {status}.", "",
         "## 1. The rule as it runs now", "",
         f"- Categories: {', '.join(t['categories'])}. {t.get('note', '')}",
         f"- A product must contain one of these name terms: {', '.join(t['include'])}",
         "- Excluded subtypes: " + "; ".join(f"`{k}` ({v})" for k, v in t["exclude"].items()),
         f"- Measure: {measure}. Buyer words: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in t["request"].items()),
         "", "## 2. Results on the whole catalogue", "", "| step | products |", "|---|---:|",
         f"| in the categories | {len(in_cat)} |", f"| no name term: not this type | {len(unmatched)} |",
         f"| excluded subtype | {len(excluded)} |", f"| **accepted as {t['label']}** | {len(accepted)} |",
         f"| outside the categories but with a name term (possible misses) | {len(outside)} |", "",
         f"## 3. Accepted ({len(accepted)}): each must be {t['label']}, and the {measure} per sales unit must be right", "",
         f"| product | name | sales unit | {measure} per sales unit | how the amount was found |", "|---|---|---|---:|---|"]
    for p in sorted(accepted, key=lambda p: p["name_en"] or p["name_fi"]):
        amount, how = catalog.amount_per_sales_unit(p, measure)
        L.append(_row(p, p["salesUnit"], f"{amount:g}" if amount else "**unknown**",
                      how or "price API comparison price at order time"))
    L += ["", f"## 4. Excluded subtypes ({len(excluded)}): each must really not be {t['label']}", "",
          "| product | name | term | reason |", "|---|---|---|---|"]
    L += [_row(p, f"`{term}`", why) for p, (term, why) in sorted(excluded, key=lambda x: x[1][0])]
    L += ["", f"## 5. In the categories without a name term ({len(unmatched)}): none of these may be {t['label']}", "",
          "| product | name |", "|---|---|"]
    L += [_row(p) for p in sorted(unmatched, key=lambda p: p["name_en"] or p["name_fi"])]
    L += ["", "## 6. Possible misses: outside the categories, with a name term", "",
          f"A real {t['label']} product here would never be offered. Most are other products that only contain the "
          "word (milk chocolate, coffee drinks).", "", "| category | products | examples |", "|---|---:|---|"]
    for cat, n in by_cat.most_common():
        ex = [f"{p['id']} {(p['name_en'] or p['name_fi'])[:38]}" for p in outside if p["category"] == cat][:EXAMPLES]
        L.append(f"| {cat} | {n} | {'; '.join(ex)} |")
    L += ["", "## 7. Checklist", "",
          f"- [ ] Every product in section 3 is {t['label']} a buyer could order under this name.",
          f"- [ ] The {measure} per sales unit in section 3 is right for every product (open a few in the catalogue).",
          "- [ ] Every exclusion in section 4 is correct, and no excluded term removes a real product.",
          f"- [ ] Nothing in section 5 is {t['label']}.",
          f"- [ ] No category in section 6 holds {t['label']} that buyers expect (otherwise add it to `categories`).",
          "- [ ] The buyer words are right in every language (native speaker).",
          f"- [ ] `policy.json` has a price ceiling for `{type_key}` (`max_price_per_unit_eur`).",
          f"- [ ] Then sign it: `python3 review_product_type.py --sign {type_key} --by \"<your name>\"`, and run "
          "`python3 -m unittest discover -s tests`.", ""]
    return "\n".join(L)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        sys.exit(__doc__)
    if args[0] == "--sign":
        if len(args) != 4 or args[2] != "--by" or args[1] not in catalog.PRODUCT_TYPES:
            sys.exit(f'usage: python3 review_product_type.py --sign TYPE --by "Name"   (TYPE: {", ".join(catalog.PRODUCT_TYPES)})')
        key = args[1]
        sign(PATH, "types", key, args[3], sheet(key, catalog.load_index()), os.path.join(REVIEW_DIR, f"type_{key}.md"),
             f"python3 review_product_type.py {key}")
        return
    unknown = [k for k in args if k not in catalog.PRODUCT_TYPES]
    if unknown:
        sys.exit(f"unknown product type(s): {unknown}. Defined: {', '.join(catalog.PRODUCT_TYPES)}")
    idx = catalog.load_index()
    os.makedirs(REVIEW_DIR, exist_ok=True)
    for key in args:
        path = os.path.join(REVIEW_DIR, f"type_{key}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sheet(key, idx))
        print("wrote", os.path.relpath(path, catalog.HERE))


if __name__ == "__main__":
    main()
