#!/usr/bin/env python3
"""Write a review sheet for a diet rule, with the actual products behind every number.

    python3 review_diet_rule.py FREE_FROM_GLUTEN     # -> rules/review/FREE_FROM_GLUTEN.md
    python3 review_diet_rule.py --all                # one sheet per rule
    python3 review_diet_rule.py --list               # codes, labels and review status
    python3 review_diet_rule.py --sign FREE_FROM_GLUTEN --by "Name"   # after reading the sheet

A rule may order autonomously only after a person has worked through its sheet and signed
it. --sign sets "reviewed": true, "reviewed_by", "reviewed_at" and "review_sheet_sha256" (the
hash of the sheet the person read, without its status line) in rules/diet_overrides.json,
leaving the rest of the file as it is. When the data or the rule changes later, the sheet
changes, --list reports it and the offline tests fail until someone reviews it again.
Generating a sheet changes no rule; it only shows what the rule does to the catalogue.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys

import catalog

REVIEW_DIR = os.path.join(catalog.RULES_DIR, "review")
SAMPLE = 15       # strong-evidence products shown (random, fixed seed so sheets are reproducible)
MAX_ROWS = 100    # per bucket


def sheet_hash(text):
    """Hash of a review sheet without its status line (which changes when it is signed)."""
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("Status:"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _block_end(text, i):
    """Index of the brace closing the JSON object whose body starts at text[i]."""
    depth, in_str, esc = 1, False, False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError("unbalanced JSON")


REVIEW_FIELDS = ("reviewed", "reviewed_by", "reviewed_at", "review_sheet_sha256")


def set_review_fields(text, container, code, fields):
    """Set the review fields of text[container][code] in a hand-formatted JSON file without
    reformatting the rest; the result is verified by parsing it."""
    expected = json.loads(text)
    expected.setdefault(container, {}).setdefault(code, {}).update(fields)
    new_fields = ", ".join(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in fields.items())
    m = re.search(r'"%s"\s*:\s*\{' % re.escape(code), text)
    if m:
        end = _block_end(text, m.end())
        block = text[m.end():end]
        ws = re.match(r"\s*", block).group(0)
        rest = re.sub(r'\s*"(%s)"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null)\s*,?' % "|".join(REVIEW_FIELDS), "", block)
        if rest.strip():  # other fields follow: keep their layout
            joined = ws + new_fields + ("," if rest[:1].isspace() else ", ") + rest
        else:             # the entry held only review fields
            joined = ws + new_fields + rest
        out = text[:m.end()] + joined + text[end:]
    else:
        m = re.search(r'"%s"\s*:\s*\{' % re.escape(container), text)
        out = text[:m.end()] + f"\n    {json.dumps(code)}: {{{new_fields}}}," + text[m.end():]
    if json.loads(out) != expected:
        raise ValueError(f"could not update {code} safely; edit the file by hand")
    return out


def sign(path, container, code, by, fresh_sheet, sheet_path, how_to_generate):
    """Record that `by` reviewed `code`, bound to the sheet they read (see the module doc)."""
    if not by or not by.strip():
        sys.exit("--by NAME is required: the reviewer's name goes into the audit trail of every order.")
    if not os.path.exists(sheet_path):
        sys.exit(f"No review sheet yet. Generate and read it first: {how_to_generate}")
    with open(sheet_path, encoding="utf-8") as f:
        on_disk = f.read()
    h = sheet_hash(fresh_sheet)
    if sheet_hash(on_disk) != h:
        sys.exit(f"The data or the rule changed since {os.path.relpath(sheet_path, catalog.HERE)} was written. "
                 f"Regenerate and read it again: {how_to_generate}")
    fields = {"reviewed": True, "reviewed_by": by.strip(), "reviewed_at": dt.date.today().isoformat(),
              "review_sheet_sha256": h}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(set_review_fields(text, container, code, fields))
    status = f"Status: reviewed by {fields['reviewed_by']} on {fields['reviewed_at']} (sheet sha256 {h[:16]}...)."
    with open(sheet_path, "w", encoding="utf-8") as f:
        f.write("\n".join(status if ln.startswith("Status:") else ln for ln in fresh_sheet.splitlines()) + "\n")
    print(f"signed {code}: reviewed by {fields['reviewed_by']} on {fields['reviewed_at']}, sheet sha256 {h[:16]}... "
          f"-> {os.path.relpath(path, catalog.HERE)}. Run python3 -m unittest discover -s tests.")


def _buckets(rule_code, idx):
    out = collections.defaultdict(list)
    for p in idx:
        status, ev, strength = catalog.diet_evidence(p, rule_code)
        if status == "pass":
            key = "conflict" if ev.startswith("CONFLICT") else "strong" if strength >= 1.0 else "name only"
        elif status in ("near_miss", "contradicted"):
            key = "look-alike"
        else:
            continue
        out[key].append((p, ev))
    return out


def _table(rows):
    L = ["| product | name | evidence |", "|---|---|---|"]
    L += [f"| {p['id']} | {(p['name_en'] or p['name_fi'])[:60]} | {ev[:140]} |" for p, ev in rows[:MAX_ROWS]]
    if len(rows) > MAX_ROWS:
        L.append(f"| … | {len(rows) - MAX_ROWS} more | |")
    return L


def sheet(code, idx):
    r = catalog.DIET_RULES[code]
    vocab = catalog._read_rules_file("diet_vocab.json", {"codes": {}})["codes"].get(code, {})
    b = _buckets(code, idx)
    nut = r["nutrient"]
    nut_text = "–"
    if nut:
        nid, op, lim = nut
        nut_text = f"{nid} {'≤' if op == 'max' else '≥'} {lim:g} g/100 g"
        if not r.get("nutrient_contradicts", True):
            nut_text += " (a value on the wrong side does not contradict)"
    status = (f"reviewed by {r['reviewed_by']} on {r['reviewed_at']}" if r["reviewed"]
              else "**NOT reviewed**: orders relying on this rule always need a person")
    if r["reviewed"] and r.get("review_sheet_sha256"):
        status += f" (sheet sha256 {r['review_sheet_sha256'][:16]}...)"
    L = [f"# Review sheet: `{code}` ({r['label']})", "", f"Status: {status}.", "",
         "## 1. The rule as it runs now", "",
         f"- Positive evidence: GS1 claims {r['pos_claims'] or '–'}; nonAllergen FREE_FROM {r['pos_free_from'] or '–'}; "
         f"nutrient {nut_text}",
         f"- Contradictions: allergens {r['contradict_allergens'] or '–'}"
         + (" (CONTAINS and MAY_CONTAIN)" if r["contradict_may_contain"] else " (CONTAINS only)"),
         f"- Look-alikes (rejected): claims {r['near_miss_claims'] or '–'}; name terms {r['near_miss_terms'] or '–'}",
         f"- Product-name words (weak evidence, needs a person): {r['name_terms'] or '–'}",
         f"- Buyer words for the rule parser (`diet_terms.json`): {r['request_terms'] or '– (only Claude can read this need)'}",
         "", "## 2. What the data suggests (`diet_vocab.json`)", ""]
    sugg = vocab.get("suggested_contradictions")
    L.append("- Suggested contradicting allergens (ratio = how often under the claim vs the whole catalogue): "
             + (", ".join(f"{s['allergen']} {s['ratio']:g}" for s in sugg) if sugg
                else vocab.get("suggested_contradictions_note", "none")))
    L.append(f"- In the policy now: {r['contradict_allergens'] or 'none'}. Differences are deliberate decisions or "
             "something to fix; write the reason as a `note` in `diet_overrides.json`.")
    L += ["", "## 3. Results on the whole catalogue", "", "| bucket | products | what happens |", "|---|---:|---|",
          f"| strong | {len(b['strong'])} | passes; can order autonomously once the rule is reviewed |",
          f"| conflict | {len(b['conflict'])} | stays a candidate at strength 0.5, a person decides |",
          f"| name only | {len(b['name only'])} | stays a candidate at strength 0.5, a person decides |",
          f"| look-alike | {len(b['look-alike'])} | rejected: a look-alike, or the data contradicts the need |", ""]
    for key, title in [("conflict", "4. Every conflict"), ("name only", "5. Every name-only match"),
                       ("look-alike", "6. Every look-alike")]:
        L += [f"## {title} ({len(b[key])})", ""] + (_table(b[key]) if b[key] else ["None."]) + [""]
    strong = list(b["strong"])
    random.Random(code).shuffle(strong)
    L += [f"## 7. Strong evidence: random sample of {min(SAMPLE, len(strong))} of {len(strong)}", ""]
    L += (_table(strong[:SAMPLE]) if strong else ["None."]) + [""]
    L += ["## 8. Checklist", "",
          "- [ ] The positive evidence sources are right for this need.",
          "- [ ] Each conflict in section 4 is better decided by a person. If one pattern repeats, change the "
          "contradiction list instead.",
          "- [ ] The name-only matches in section 5 are plausible. Add missing name words, or remove misleading ones, "
          "in `diet_terms.json`.",
          "- [ ] The look-alikes in section 6 really do not meet the need.",
          "- [ ] The sample in section 7 contains no product that clearly fails the need.",
          "- [ ] A native speaker checked the Finnish and Swedish words in `diet_terms.json`.",
          f"- [ ] Then sign it: `python3 review_diet_rule.py --sign {code} --by \"<your name>\"`, and run "
          "`python3 -m unittest discover -s tests`.", ""]
    return "\n".join(L)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        sys.exit(__doc__)
    if args[0] == "--list":
        idx = None
        for code, r in sorted(catalog.DIET_RULES.items(), key=lambda kv: -kv[1]["products"]):
            state = "not reviewed"
            if r["reviewed"]:
                state = f"reviewed by {r['reviewed_by']} on {r['reviewed_at']}"
                if not r.get("review_sheet_sha256"):
                    state += " (no signed sheet)"
                else:
                    idx = idx or catalog.load_index()
                    same = sheet_hash(sheet(code, idx)) == r["review_sheet_sha256"]
                    state += " (signed sheet unchanged)" if same else " (CHANGED since the review: review again)"
            print(f"{code:<34} {r['label']:<32} {state}")
        return
    if args[0] == "--sign":
        if len(args) != 4 or args[2] != "--by" or args[1] not in catalog.DIET_RULES:
            sys.exit('usage: python3 review_diet_rule.py --sign CODE --by "Name"   (CODE from --list)')
        code = args[1]
        sign(os.path.join(catalog.RULES_DIR, "diet_overrides.json"), "codes", code, args[3],
             sheet(code, catalog.load_index()), os.path.join(REVIEW_DIR, code + ".md"),
             f"python3 review_diet_rule.py {code}")
        return
    codes = sorted(catalog.DIET_RULES) if args[0] == "--all" else args
    unknown = [c for c in codes if c not in catalog.DIET_RULES]
    if unknown:
        sys.exit(f"unknown diet code(s): {unknown}. Try --list.")
    idx = catalog.load_index()
    os.makedirs(REVIEW_DIR, exist_ok=True)
    for code in codes:
        path = os.path.join(REVIEW_DIR, code + ".md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sheet(code, idx))
        print("wrote", os.path.relpath(path, catalog.HERE))


if __name__ == "__main__":
    main()
