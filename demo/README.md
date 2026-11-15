# Trusted Trade Agent: minimal demo

The demo takes a buyer's natural-language request, looks up products in the Valio Aimo catalog and the live mock APIs, and then either places the order within the agent's authority or sends it to a human for sign-off. Abstract needs (a menu idea) become a quoted proposal instead of an order; named dishes ("source the ingredients for salmon and carrot soup for 100 guests") become a multi-line order that a person confirms once. When nothing can be ordered, the agent says why and offers relaxations that only a person can accept; diet and allergy requirements are never relaxed. Every step is explained and audited. Setup, the request flow and the diet-rule files are described in the [root README](../README.md); this file covers the design, the rules and the demo script.

```
"I need 1,000 litres of lactose-free milk by Monday..."
   │ ① interpret      nlu.py      rules parser (+ optional Claude, cross-checked, every LLM-only value quoted)
   ▼
 structured query  {product: milk, diet: FREE_FROM_LACTOSE, qty: 1000 L, deliver_by: 2026-11-16}
   │ ② retrieve       catalog.py  static rules on 17,546 products → ~27 candidates
   │ ③ live check     engine.py   POST /api/price + /api/availability for candidates only
   │ ④ rank           engine.py   multi-objective score + Pareto front + explanation
   │ ⑤ guardrails     engine.py   policy.json → AUTO_ORDER | NEEDS_HUMAN_APPROVAL | ESCALATE
   │ ⑥ if escalated   agent.py    why nothing could be ordered + relaxations R1, R2, … (never diet or allergy)
   ▼
 order book (out/orders.json) + human-readable audit (out/audit/<order>.md)
   │ a person accepts R1   → the same pipeline re-runs with that one change, linked to the old order
   │ a person approves     → fresh live quote for the chosen option; changed → STALE_QUOTE + re-plan
```

A repeated request returns the earlier order instead of a new one (idempotency), and a second open order for the same product and date always needs a person.

Abstract requests ("hot rhubarb main courses in a Scandinavian style?") take a second path, and **never order**:

```
 ① interpret (Claude, intent = recommend)
   │ ② recommend      nlu.py      Claude: 3–5 dishes, ingredients with Finnish + English search terms
   │ ③ search         catalog.py  static word search per ingredient → ≤ 8 real candidates
   │ ④ pick           nlu.py      Claude picks one product per ingredient from an enum of those candidate IDs
   │ ⑤ diet screen    agent.py    same diet evidence as orders: contradicted → removed, no evidence → a person confirms
   │ ⑥ live quote     engine.py   price + availability for the picked products only (same troubleshooting rules)
   │ ⑦ substitutes    agent.py    missing or undeliverable key ingredient → Claude suggests ≤ 3, same search/screen/quote
   ▼
 PROPOSAL_FOR_REVIEW: dishes, matched products, live status, questions (portions? date? which dishes?)
```

Named dishes take a third path (`basket.py`, intent `source`), which **orders after one confirmation**:

```
 ① interpret (Claude + rules)  dishes, portions, diet sub-groups ("6 of them are allergic to milk protein and gluten")
   │ ② recipe         nlu.py      Claude: the standard recipe per named dish, quantity per portion
   │ ③ lines          basket.py   portions × quantity → need per ingredient; one line per diet group where needs differ
   │ ④ identity       nlu.py      catalogue search per ingredient, Claude's verdicts in one call (real IDs only)
   │ ⑤ allergens      catalog.py  allergen_verdict per group: reject (never offered) / warn (missing data) / pass
   │ ⑥ live quote     engine.py   price + availability; green = the usual product (rules/order_guide.json) in time
   │ ⑦ red lines      basket.py   Best (other SKU) / Better (pack, price, form) / Good (same function) / Later / warn
   │ ⑧ one question   basket.py   price | delivery | stock | fit → re-ranks inside the tiers, never relaxes a rule
   ▼
 AWAITING_CONFIRMATION → one confirmation, re-checked live (engine.revalidate) → CONFIRMED_BY_HUMAN, tier per line in the audit
```

The LLM understands and suggests; the catalogue and the live API decide what exists, what it costs and whether it can be delivered. Because the pick enum is built from real candidates, constrained decoding makes an invented product impossible.

## Run

Requires Python 3.9 or later. Install `requirements.txt` to run the hospital scenario with Z3 or use Claude. First-time setup (`.env`, downloading the data with `fetch_data.py`) is described in the [root README](../README.md).

Put the API key in `.env` (copy `.env.example`). The file is git-ignored, and a variable already set in the shell takes precedence over it.

```bash
cp .env.example .env                    # then set AIMO_API_KEY=<key from the Junction challenge page>
python3 fetch_data.py                   # downloads the git-ignored data files into ../data/
python3 agent.py "I need 1,000 litres of lactose-free milk by Monday. Find an option that meets our requirements based on price and availability."
python3 agent.py "..." --json          # the structured response the buyer's agent receives
```

**Web UI**

```bash
python3 server.py                       # then open http://127.0.0.1:8000 (classic UI) or http://127.0.0.1:8000/trustpurrrchase (TrustPurrrchase chat UI)
```

Type a request (or click an example: "Basics" shows the core outcomes, "Hard rules" diet and allergy rules that are never relaxed, "Soft & relax" requirements that can change with your OK; ✦ needs Claude) and press **Find & order**. An escalated order shows **"Why nothing could be ordered"** and **"Options that need your OK"** with an Accept button per relaxation; accepting re-runs the request, and the new result is labelled "Re-run of ORD-…". The page shows the decision and the options compared, plus **"How the agent decided"**, one step per stage, each labelled with who acted (⚙ rules, ✦ Claude, ◉ live data, ⚖ policy):

1. **Read the request**: your words, marked by who read them (the rule parser, a verbatim quote from Claude, or both).
2. **Understand**: every requirement with its strength (hard, negotiable, preference), each Claude call with its constrained decoding (the enums the schema allowed, built from the rule files), the reasoning summary the API returns, and a field-by-field table of the rule parser's reading, Claude's reading, Claude's quote and the result (agree, Claude only → confirm, ignored, …).
3. **Find products**: the retrieval funnel, each stage with its rule source (`rules/product_types.json`, `rules/diet_*.json`, brand exclusion, or Claude's verdicts for any other product) and the products it ruled out.
4. **Check live price & stock**: live rejections by reason, with the SHA-256 of every API response.
5. **Compare options** and 6. **Decide**: every purchasing-rule check, grouped by where it comes from (a `policy.json` key, your request, your consent to a relaxation, a review flag, the understanding step).

Proposals show the same steps for the menu path: the hard diet requirements applied to every picked product, and the substitute call.

Claude's tokenizer and its raw chain of thought are internal to the model and are not shown; the reasoning summary is the one the API provides (`thinking.display = "summarized"`), and simple low-effort calls often return none. Pending orders can be approved (any option) or rejected right there. The server binds to localhost only.

The first run builds `.cache/catalog_index.json` from the 87 MB product file, which takes about 2 seconds.

There is an optional LLM parser (the SDK call was tested with `anthropic` 0.125.0 on Python 3.9; the current interpret / recommend / pick / classify steps are covered by `eval_llm.py`: 85 of 85 checks passed on 2026-11-14, 26 Claude calls, about 137k input and 17k output tokens; that run predates the `must_have`, diet-quote and substitute changes, so re-run it before the demo). Run `pip install anthropic` and set `ANTHROPIC_API_KEY` in `.env`, and `claude-opus-5-5` will parse the request using JSON-schema-constrained output. The rules parser still runs alongside it, and if the two disagree on a critical field, the order goes to human approval instead of being auto-resolved. A value only Claude read must quote the request: quoted values go to a human, unquoted ones are discarded. Without the SDK, the deterministic parser is used, and menu ideas fall back to "please clarify".

**Tests.** `python3 -m unittest discover -s tests` runs offline (no network, no LLM, no files written). `python3 eval_llm.py` checks the Claude steps end to end; it **costs money** (about 20 Claude calls) and writes nothing to `out/`.

## Demo scenarios (the video)

The three scenarios of `demo_scenarios.md`, one engine. Each has a chip in the UI row "Demo".

| # | Request | What it shows |
|---|---|---|
| ① | `I need 1,000 litres of lactose-free milk by Monday. Find an option …` | Funnel 17,546 → 91 → 27 → 11, the low-lactose trap rejected, live price and stock with SHA-256, every guardrail passes → **ready to order, a person confirms** (464306, €1.01/L, €1,010; re-checked live on confirmation). **Cancel** or **Change…** stay available until the order cut-off (Sun 15.11 17:00); a change cancels the old order and links both |
| ② | `We're catering a wedding for 100 guests on Saturday and want to serve salmon and carrot soup. Can you source the ingredients?` | **Partial availability, one decision.** Claude writes the recipe and the quantities are scaled × 100. Five lines are ready (folded into one row). **One red line:** the buyer's usual carrot 460721 is really not deliverable. Three choices, one per tier: **Best** carrot 1 kg bag (same product, nothing changes), **Better** peeled 5 kg (another form, bigger pack), **Good** swede or parsnip (another root, same job). Best is preselected; "Rank by" re-orders inside the tiers only. **One confirmation**, re-checked live |
| ③ | `Hospital kitchen: we need 4 litres of cooking cream for tomorrow's lunch soup for 40 patients. 6 of them are allergic to milk protein and gluten.` | **Allergen safety under pressure, one decision.** One product, split by group: 34 portions get the usual cream (3.4 L, ready). For the 6 allergy portions (0.6 L), the card first shows what is **ruled out and never offered**, e.g. the lactose-free creams ("contains milk protein, although lactose-free"; oat products such as Oatly show "contains gluten, although vegan" when the search finds them). Then three choices: **Good** puréed potato (1 kg, safe now), **Later** a safe oat base one day late, **Unverified** AROY-D coconut cream (in time, no allergen data: confirming needs "I will check the label", creates an arrival check task, never learned) |

Recording notes:

- **Clock.** "Tomorrow" in ③ must be Monday 16.11, the earliest delivery date in the mock. Record on Sunday before 17:00, or start the server with `AGENT_TODAY=2026-11-15 python3 server.py` (the UI then shows "Demo clock: today = Sun 15 Nov"). Prices, stock and cut-offs always use the live server time.
- **Time pressure in ③ is simulated.** In the live data several fully evidenced bases (Alpro 464745, Delibon 464924, Fazer 473956) arrive on Monday, so without help the agent offers one of them as a safe choice next to the potato (only Elovena 473496 really misses its cut-off). Tick **Simulate a supplier delay** (or `--simulate hospital_delay`): `scenarios/hospital_delay.json` shifts their delivery by one day after the real response is hashed; the result and the audit say SIMULATED and list the products.
- **The hospital allergen check uses Z3.** For the milk-and-gluten allergy group, Z3 checks whether a violation is possible under the demo rules and the catalog evidence for each requirement. A declared violation is excluded, and incomplete evidence requires acknowledging a label check. The audit records the SKU, evidence and solver results. The two demo allergen rules are still pending human review.
- **Demo-specific data, on purpose:** `rules/order_guide.json` makes carrot 460721 the usual SKU (really undeliverable, so ② has its red line). The other entries are ordinary usual products.
- ② and ③ need Claude: ② about 50 s (interpret, recipe, identity, substitutes, pick), ③ about 25 s (no recipe step). Without the SDK they fall back to "please clarify".

Display rule for a red line: at most three choices, the top option of each of the first three non-empty tiers (Best, Better, Good, Later, Unverified); everything else is folded under "All N options", and the agent's internals under "How the agent decided". A one-product request with a diet sub-group is split per group (never ordered for everyone); without the total number of portions the agent asks.

CLI: `python3 agent.py "<request>" [--simulate hospital_delay]`, then `python3 agent.py confirm BSK-… [--prefer price] [--pick LINE=OPTION] [--ack LINE]` and `python3 agent.py cancel ORD-…|BSK-…`.

## Demo script

| # | Request | What it shows |
|---|---|---|
| 1 | `I need 1,000 litres of lactose-free milk by Monday...` | Happy path. Valio 464306 at €1.01/L, €1,010 total. All guardrails pass, so the agent **recommends it and a person confirms** (`require_human_confirmation`, on by default) |
| 2 | `We need 1,000 litres of lactose-free Arla milk by Monday, cheapest available.` | **Substitution outside stated preference.** The cheapest option is Valio, so the order goes to approval. The all-Arla alternative (a split of 3 stock items, €2,580) is shown with its trade-off |
| 3 | `I need 3000 l lactose-free milk by Monday` | **Spend over the autonomous limit** (€3,030 > €1,500), so the order goes to approval |
| 4 | `I need 1,000 litres of lactose-free milk by tomorrow` (UI chip "Too soon → later date") | **No feasible option, then consent.** The agent escalates with the reason ("Earliest possible delivery is 2026-11-16 (26 products)") and the relaxation R1 "Deliver on 2026-11-16 instead of 2026-11-15". Accepting it re-runs the request, which then **orders autonomously** with the check "relaxation accepted by …" |
| 4h | `1000 l vegan milk by Monday` | **Hard requirement.** "Hard requirement 'vegan': all 75 of 75 milk products were ruled out (75 contradicted by the allergen data). Diet and allergy requirements are never relaxed." No Accept button. With Claude, the request is read as a plant drink and goes to approval |
| 4a | `1000 l lactose-free milk for a guest with a fish allergy, by Monday` | **Allergy rule** `FISH_FREE` (GS1 allergen data). Only products declared fish-free remain; the rule is not reviewed yet, so the order needs a person |
| 4s | `We'd like to serve durian pancakes as a dessert special on Friday. What should we order?` (✦) | **Substitute for a missing ingredient.** "'durian pulp' is not in the catalogue. Suggested instead: frozen mango pulp (Findus 460040). Needs your OK." Undeliverable key ingredients get a substitute the same way. Nothing is ordered |
| 4b | `We need 50 kg of butter by Monday` (UI chip "Any product") | **Any product** (needs Claude). Catalogue search, Claude's verdict on each of 40 results (ice cream and butter fillings are ruled out with a reason), prices per KG, then **approval**, because butter is not a reviewed product type and has no price ceiling yet |
| 5 | `We've already planned rhubarb pudding, bread and pancakes ... hot rhubarb-themed main courses ... Scandinavian` (UI chip "Menu idea") | **Inferring needs from an abstract request** (needs Claude). Dishes, each ingredient matched to a real product (fresh rhubarb 460519 / 460404, not jam or cake), live price and availability, out-of-season or past-cut-off items flagged, then questions. **Nothing is ordered** |
| – | `We need gluten-free, lactose-free milk for Monday.` | No quantity, so the agent asks instead of guessing. Gluten-free is understood and verified (generated rule), but the rule is not yet reviewed, so an order would need a person |
| – | `1000 l kosher lactose-free milk by Monday` | The data cannot verify "kosher", so it is reported as an unsupported requirement instead of being dropped |
| – | `1000 l low-lactose milk by Monday` | 29 candidates (2 low-lactose Arla milks + 27 lactose-free ones, which also qualify) |
| – | `1000 l lactose-free milk, no Valio, by Monday` | The exclusion is executed (`AVOIDED_BRAND`); the remaining Arla split needs approval |
| – | `1000 l lactose-free milk, only Arla, by Monday` | "Only" makes the brand a negotiable requirement (`MUST_HAVE`) instead of a preference: non-Arla products are filtered out, and the all-Arla split (€2,580) needs approval. If no Arla product were left, the agent would offer "drop the 'only Arla' requirement" for your OK |
| – | `Vegan brunch for 40 guests next Friday, any ideas for warm dishes?` (✦) | Diet in a proposal: produce without a vegan claim is shown as `DIET_UNVERIFIED` for a person to confirm; contradicted products are removed |
| – | `1000 l lactose-free milk and 50 kg butter by Monday` | Two products in one request: the agent asks for one per request instead of silently dropping the butter |
| – | Scenario 1 again, as `1000 l laktoositon maito by Monday, cheapest` | **Idempotency.** Same goods, quantity, date and requirements within 10 minutes, so the earlier order is returned and nothing new is ordered. "Order again anyway" (`--new`) runs it, and the duplicate guardrail sends it to approval |
| – | Approve any pending order | **Re-check at approval.** The chosen option is priced again live before it commits ("price and stock still hold", with response hashes). If the quote changed, nothing is placed: `STALE_QUOTE`, with a fresh plan to decide on |

Human sign-off and preference learning:

```bash
python3 agent.py approve ORD-... --choose 5 --by "Head chef"   # pick the all-Arla option instead of #1
python3 agent.py "We need 1,000 litres of lactose-free Arla milk by Monday..."   # now ranked #1
python3 agent.py reject  ORD-...
```

Approval re-checks the chosen option against a fresh live quote before it commits (see the root README, "Commit controls"); a changed quote is never placed. When the human picks a non-recommended option, a pairwise update shifts the objective weights toward the objectives where the chosen option is better. The weights are stored in `out/prefs.json` along with their history. **Learning only re-ranks options. It never relaxes guardrails**, which stay owned by `policy.json`. To reset, delete `out/`.

## Rules

**Static retrieval** (`catalog.py`) works on catalog data only, as the API docs require.

| Rule | Example |
|---|---|
| Reviewed product types | `rules/product_types.json`: milk → category `10400`, include and exclude terms, measure L. Any other product is searched generically and identified by Claude per candidate, and the order always needs a person |
| Exclude wrong subtypes | piimä (sour milk), coffee/latte drinks, cocoa, flavoured milk |
| Buyer filters | "no Valio" → `AVOIDED_BRAND`; "only Arla", "vain luomu" → `MUST_HAVE` (brand, fat level or organic). Both are negotiable: a person can relax them |
| Diet evidence, strongest first | GS1 claim `FREE_FROM_LACTOSE` and nutrient `LACS ≤ 0.01 g/100 g` give strong evidence. A name match alone is weak evidence and needs human approval. `LOW_LACTOSE` accepts its own claim, `LACS ≤ 1.0 g`, or the stricter lactose-free claim |
| Conflicting sources | All signals are collected before deciding. If the claim is contradicted by the nutrient value, the allergen list (CONTAINS, or MAY_CONTAIN where the rule says so) or a weaker own claim, the product stays a candidate but at strength 0.5, marked `CONFLICT`, so a human decides. For lactose-free this affects 49 products catalogue-wide, none of them milk; `rules/diet_coverage.md` has the count for every rule |
| Other evidence sources | Besides GS1 claims and nutrient thresholds (`≤` or `≥`), the nonAllergen FREE_FROM entries count as positive evidence. That is the only evidence for `DAIRY_FREE`, `NUT_FREE` and the nine other EU major allergens (fish, crustaceans, molluscs, peanuts, sesame, celery, mustard, sulphites, lupin), which have no GS1 claim; CONTAINS or MAY_CONTAIN contradicts them. Buyers' allergy phrasings ("fish allergy", "allergic to sesame") map to these rules |
| Contradicted vs no evidence | A product the data contradicts (milk for vegan: allergen AM) is `contradicted`; one with no evidence either way is a plain fail. The explanation counts each, and the proposal path keeps "no evidence" products for a person (`DIET_UNVERIFIED`) |
| Near-miss trap | `vähälaktoosinen` / HYLA / "low lactose" is **not** lactose-free, so it is rejected with that reason |
| Single vocabulary, generated | `catalog.DIET_RULES` is the only list of diet codes, built at import from `rules/`: all 21 GS1 claim codes in the data (`diet_vocab.json`, generated by `build_diet_vocab.py`), synonyms (`diet_terms.json`) and human policy (`diet_overrides.json`). The LLM schema enum and the rule parser's `DIET_TERMS` are both built from it, so the two parsers cannot drift |
| Relations from the claim grammar | A GS1 claim code is a type plus an element (`FREE_FROM` + `LACTOSE`). So for every element, `FREE_FROM_X` satisfies `LOW_X`, and `LOW_X` is a look-alike of `FREE_FROM_X`; nothing is hand-written per pair |
| Contradictions from co-occurrence, decided by people | The generator suggests allergens that almost never appear under a claim (gluten-free → AW 0.02, NR 0.03; egg-free → AE 0). Small samples give nonsense (sweetener-free → ML), so suggestions only become rules in `diet_overrides.json`. See `rules/diet_coverage.md` |
| Reviewed rules only act alone | 32 rules exist (21 claims + 11 from nonAllergen data); only the two lactose rules are reviewed, by the placeholder "demo team". Any order relying on an unreviewed rule fails the guardrail "diet rule reviewed by a person". Flip `reviewed` after checking the coverage report |
| Unit conversion | Per sales unit in the order's measure. L: the `ZML` unit, or the volume in the name × pack size. KG: the `KG` unit, or a single pack weight in the name; otherwise derived from the price API's comparison price. PCS: consumer packs |
| Brands | Every brand in the data (1,418), matched by family; 67 that are common words ("Fresh", "Classic") or 1–3 letters are never detected by the rule parser. Exclusions are hard filters |
| Dates and quantities | See the root README. What the rule parser cannot read is reported, never guessed |

**Live troubleshooting** (`engine.live_check`, reading every response through `aimo_contract.assess`; the contract table is in the root README) rejects or flags each offer with a reason code:

| Code / flag | Rule |
|---|---|
| `CUTOFF_PASSED` | `closingTime` is earlier than the server `timeNow` and the deadline is the reported earliest date. The mock still reports Monday for supplier items whose Thursday 15:00 cut-off has passed, so the cut-off is treated as authoritative |
| `STALE_CUTOFF` (anomaly) | The same past cut-off, but the deadline is later than the earliest date: the item is kept (a later delivery is plausible), and the guardrail "live data consistent with the API contract" sends the order to a person |
| other anomalies | `MULTIPLE_SCHEDULE_LINES`, `UNEXPECTED_AMOUNT`, `UNKNOWN_LOGISTIC_MODEL`, `DATE_ECHO_MISMATCH`, `PRICE_QUANTITY`: the response breaks the documented contract, so the same guardrail fails. None occurs on the mock today (`python3 probe_contract.py`) |
| `TOO_LATE` | `availableDate` is after the deadline |
| `daysEarly` | Every line is ordered for `availableDate`, the earliest date the API offers. When that is before the deadline, the line, the explanation, the UI and the audit say how many days early it arrives |
| `SALES_BLOCK`, `NOT_DELIVERABLE`, `OUT_OF_STOCK`, `NO_PRICE` | Taken directly from the API fields |
| ample / unconfirmed / limited | `9999999999` (scaled to sales units) means ample stock. A Terminal item reporting `999999` has quantity unknown to Valio Aimo. Otherwise the stock is a real limit, and partial stock can be combined in a split plan |
| `UNIT_MISMATCH` | Litres derived from the catalog are cross-checked against `catalogPrice / comparisonPrice` from the price API |
| no reliability history | A prior of 0.90 is assumed and the offer is flagged |

## Ranking (multi-objective)

Utilities are scaled to the 0–1 range, with best = 1: **cost** (€/L), **availability** (delivery reliability × whether the quantity is confirmed), **preference match** (brand, fat level, organic), **evidence** strength, **handling** (consumer packs per litre) and **carbon** (category-level CO2e, which currently does not separate milk options). The score is the weighted sum × coverage. Objectives named in the request ("price", "availability") get a 1.5× weight boost. The agent marks Pareto-optimal options, and always shows tagged and Pareto options next to the top-ranked ones, so the human sees the real trade-offs.

## Guardrails (`policy.json`)

The agent never places an order on its own: `require_human_confirmation` (default `true`) sets every order to `PENDING_APPROVAL`, and a person confirms it, re-checked live. With the switch off, an order is placed autonomously only if **all** of these checks pass; otherwise the failed checks are listed:

full quantity covered · spend ≤ €1,500 · price ≤ the ceiling for the product type (milk €2.50/L; no ceiling = a person decides) · single product line · no substitution outside stated preferences · dietary claim backed by structured data · quantity confirmed by Valio Aimo stock · delivery reliability ≥ 95% · unit conversion cross-checked · unambiguous interpretation · and, only when it applies: every diet rule involved is reviewed, the product type is reviewed, no line is a variant, the live data is consistent with the API contract (no anomaly, e.g. no stale cut-off); in a real run (not a dry run): no other open order for the same product and delivery date within 24 h. A re-run after a relaxation also lists the passing check "relaxation accepted by <name>", so the audit shows who changed the request.

"Unambiguous interpretation" fails on any parser issue: the two parsers disagree, a field was read only by Claude (with its quote), a requirement cannot be expressed or verified (`unsupported_requirements`, or the rules-side detector for `X-free`, `without X`, `no <brand>`, vegan, ...), or Claude's intent contradicts a complete rules reading.

## Assumptions (visible on purpose)

- Category `10400` = milk & milk drinks, and unit `ZML` = litre. Both were inferred from the data, and the litre assumption was verified against the price API's per-litre comparison price.
- The mock has no order endpoint, so "placing" an order writes it to `out/orders.json` with its status.
- Orders are rounded up to whole sales units for count units (ST, KI, …) and to the next `allowedLotSize` step for products sold by weight (KG). The customer id does not affect prices in the mock.
- Every line is ordered for the earliest delivery date the API offers (the mock ignores `requestedDate`). When that is before the deadline, the order says how many days early the goods arrive; it does not schedule a later delivery. For the Saturday wedding (scenario ②), for example, the options read "arrives Mon 16 Nov, 5 days early".
- Only `milk` is a reviewed product type. Any other product can be ordered through the generic path with a person's approval; menu proposals are quoted only.
- One product per request. A package count without a size ("10 cases") is asked about, because pack sizes differ per product.
- Diet and allergy rules are hard; product, quantity, date, excluded brands and "only …" are negotiable; brand, fat level and organic without "only" are preferences. "Peanut allergy" also applies `NUT_FREE` (the phrase contains "nut allergy"), which is stricter than needed but never unsafe.
- In proposals, a product without diet evidence either way is shown for a person to confirm rather than dropped, because fresh produce rarely carries a diet claim.
- Proposal estimates (≈ €) are only shown when the buyer states portions and the ingredient unit matches the API's comparison unit (KG or L).

## Agent interface contract

A buyer's agent talks to the same functions the UI uses, over local HTTP (`server.py`). Responses are described by JSON Schemas: `schema/order_response.schema.json` (one product) and `schema/basket.schema.json` (named dishes).

| Endpoint | Body | Returns |
|---|---|---|
| `POST /api/request` | `{text, use_llm, idempotency_key?, allow_duplicate?, simulate?}` (or header `Idempotency-Key`) | an order decision, a proposal, or a basket (`kind: "basket"`) |
| `POST /api/decide` | `{order_id, action: approve\|reject, choose}` | the order, re-checked live at approval |
| `POST /api/relax` | `{order_id, relaxation_id}` | a re-run with one change a person accepted |
| `POST /api/cancel` | `{order_id}` | `CANCELLED_BY_HUMAN`, only before the order cut-off (`cancel_until`) |
| `POST /api/change` | `{order_id, text}` | the old order cancelled, the changed request run and linked (`replaces`) |
| `POST /api/basket/preference` | `{order_id, answer: price\|delivery\|stock\|fit}` | the basket, re-ranked inside the tiers |
| `POST /api/basket/confirm` | `{order_id, choices: {line: option}, acks: [line], quantities?: {line: sales units}}` (a changed quantity is re-priced and re-checked; never above limited stock) | `CONFIRMED_BY_HUMAN` (re-checked live) or `STALE_QUOTE`; 400 if a red line has no choice or a warning is not acknowledged |

Every response carries `api_evidence` (SHA-256 of each Valio Aimo response) and the path of its audit trail.

## Next steps

### Known gaps in this demo

This is an honest list. Each gap maps to a step below.

- **Per-order spend limit only.** An agent could split one large purchase into several orders, each under the €1,500 limit.
- **Earliest-date delivery only.** The mock answers only for its earliest date, so goods needed on Friday are ordered for Monday and the order says "4 days early". Perishables may not keep that long; a production API that answers for `requestedDate` removes the gap without code changes (`daysEarly` becomes 0).
- **Slow proposals.** A menu or substitute request makes four sequential Claude calls (interpret, recommend, pick, substitute: about 55 s together, recommend alone about 30 s) plus the live API calls, and took 100–115 s end to end in our runs. The UI shows a spinner but not which stage is running. Streaming stage progress, running independent calls in parallel and prompt caching would cut the wait.
- **Placeholder reviewers.** The lactose rules and the milk product type are marked reviewed by "demo team", without a signed sheet. A named person should work through `rules/review/FREE_FROM_LACTOSE.md`, `LOW_LACTOSE.md` and `type_milk.md` and sign them (`--sign … --by`). The nine allergen rules are unreviewed drafts, and their GS1 allergen codes still need checking against the code list.
- **Re-check covers price, stock and date only.** The approval re-check does not re-run diet evidence or the guardrails (catalogue data does not change between quote and approval in the mock), and a split option is committed or refused as a whole.
- **Pack-size wishes are detected, not handled.** "In 20 L boxes" is reported (human approval) rather than applied to the search. Brand exclusions are applied.
- **Generic products always need a person** until their type is added to `rules/product_types.json` and reviewed, and Claude's verdicts have not been measured on many products yet (`eval_llm.py` has one product).
- **Menu ideas stop at a quote.** Named dishes become multi-line orders (`basket.py`), but a menu proposal (intent `recommend`) is not yet turned into one; picking dishes from a proposal should hand them to the basket path.
- **Baskets lean on Claude and on synthetic buyer data.** The recipe quantities per portion come from Claude (shown, not verified), and `rules/order_guide.json` is a synthetic order guide. Without Claude a named dish falls back to "please clarify".
- **New allergen policy is unreviewed.** For the hospital scenario `DAIRY_FREE` now also accepts the GS1 `VEGAN` claim, and `DAIRY_FREE` / `FREE_FROM_GLUTEN` accept whole unprocessed produce by category (`inherent_categories`, checked on 2026-11-14: no milk or gluten allergen in those categories; an allergen entry still contradicts). Both rules stay `reviewed: false` until a person signs them.
- **Valio's substitution table is unused.** The catalogue's `substitutions` field covers 189 products. For example, our recommended product 464306 lists 465039, 464398 and 469363 as substitutes.
- **Coarse preference learning.** It is global rather than per buyer or category, and fast: one human choice can flip the ranking.

### Before judging: most value per hour

The steps are ordered by judging weight and effort (S is under 2 hours, M is half a day).

| # | Step | Criterion | Effort | What it changes |
|---|---|---|---|---|
| 1 | ~~Re-validate at approval~~ **Done** (2026-11-14): a fresh quote for the chosen option; a rise in price, a fall in stock or a passed cut-off gives `STALE_QUOTE` and a re-plan; fail closed when the API is down | Autonomy & controls (20) | – | A human can never approve a stale quote |
| 2 | ~~"Accept earliest date" button~~ **Done** (2026-11-14), generalised to relaxations: later date, excluded brand, "only …" requirement, substitute product; never diet | End-to-end (30), exceptions (15) | – | The "too soon" scenario ends in an order instead of a dead end, after the Sunday 17:00 cut-off too |
| 3 | **Good-better-best from Valio's own substitution table.** Listed substitutes that pass the hard rules rank first and are labelled "Valio-approved substitute". They still need a human if they fall outside the buyer's stated preferences. The rules matter here: the table also maps coffee milk to a *vanilla* coffee drink | Exceptions (15), need (20) | M | Stock-out handling is backed by the wholesaler's data, not only by our heuristics |
| 4 | **Cumulative budget guard.** Add a daily or weekly spend cap per buyer to `policy.json` and check it against `orders.json` | Controls (20) | S | Closes the order-splitting loophole |
| 5 | ~~Idempotency~~ **Done** (2026-11-14): caller keys plus the normalised request (10 min), a 24 h duplicate-order guardrail, idempotent approve and relax | Controls (20) | – | Retries and double clicks cannot create duplicate orders |
| 6 | **Pack sizes.** Add `pack_size` to the query schema and rules (brand exclusions are done; pack-size wishes are reported and sent to a human) | Need (20) | S | "20 L boxes" is handled exactly instead of escalated |
| 7 | **MCP server** wrapping the existing functions as tools: `search_products`, `quote`, `place_order`, `request_approval`, `get_order`. The JSON Schemas of the responses are **done** (`schema/`, see "Agent interface contract"); the official Python SDK needs Python ≥ 3.10, so on 3.9 this is a small stdlib JSON-RPC server | Openness (15) | M | In the live demo, any MCP-capable agent can act as the hotel's purchasing agent |
| 8 | **Peppol BIS Ordering 3.0 (UBL 2.1) document** for each placed order, with GTINs and standard unit codes | Openness (15) | M | The commitment step uses an open European e-procurement format instead of our own JSON |
| 9 | **Demo safety.** Rehearse against the real clock and record a fallback video of the five scenarios | End-to-end (30) | S | A live-API outage or the date cannot sink the pitch |
| 10 | **Review the generated diet rules** (gluten-free, vegan and the allergen rules first; replace the "demo team" placeholder on the lactose rules and milk) with `review_diet_rule.py` / `review_product_type.py` and `--sign`, and have a native speaker check `rules/diet_terms.json`; then set `reviewed: true` | Need (20), controls (20) | S | More diet requests can be ordered autonomously, with a documented human decision behind each rule |
| 11 | **Faster proposals**: stream progress to the UI, run the per-dish pick and substitute calls in parallel, cache the long system prompts | End-to-end (30) | S | The menu demo stops being a 100-second wait |

### Path to production (what Valio Aimo would build)

**Phase 1: a read-only agent interface.**
- Publish catalogue search and quotes as MCP tools and as an OpenAPI API.
- Authenticate each customer with OAuth 2.0 client credentials and scopes such as `catalog:read` and `quote:read`. A customer's agents see only that customer's prices.
- Enforce data minimisation. Agents get narrowed results, never the full price list. Publish explicit per-agent rate limits; this turns the mock's "don't bulk-query" rule into policy.
- Use standard identifiers. GS1 GTINs identify products. Units use UN/ECE Recommendation 20 codes (`LTR` instead of the internal `ZML`). Diet claims come from GS1 Synkka.

**Phase 2: commitment.**
- Introduce quote objects with an ID, a fixed price, a validity window and the order cut-off.
- The order API accepts a quote ID plus an idempotency key, re-validates at commit, and returns a confirmation.
- Status updates arrive by webhook: accepted, picked, delivered, short-shipped.
- Order documents use Peppol BIS Ordering (UBL).

**Phase 3: delegated authority, with guardrails on both sides.**
- The buyer's policy becomes a machine-readable, signed mandate issued by the customer. It states who may order what, the limits per order and per period, and which substitutions are allowed.
- The agent presents the mandate with every order, and **Valio Aimo enforces it server-side as well**. The wholesaler cannot assume an external agent applies its own guardrails correctly.
- Approvals happen where people already work (webshop, email, Teams). Each approval expires at the order cut-off, and roles carry different limits, for example chef versus purchasing manager.

**Phase 4: a trust layer.**
- Sign offers, stock claims and order confirmations, as JWS or W3C Verifiable Credentials over the response hashes the demo already logs. The buyer's agent can then verify price, availability, origin and certifications without a manual check.
- Keep one tamper-evident audit log that both parties can read.

**Phase 5: learning and proactive features, under the same guardrails.**
- Learn preferences per buyer and per category.
- Detect drift in standing orders, such as portion counts or cost creeping up, before auto-renewing them.
- Let Valio Aimo's selling agent propose seasonal or new products to the buyer's agent. Every suggestion passes the same rules and is logged.

**Cross-cutting.**
- **Evals.** `eval_llm.py` (15 parser cases, 3 menu cases, 1 generic-product case, 1 injection case) and `tests/` (offline rules regression) are the start. Grow the parser set with real buyer phrasings (English, Finnish and Swedish; units; relative dates; pack counts) and run it before every prompt or model change.
- **Security.** Treat catalogue text as data, never as instructions. The interpret and recommend steps see only the buyer's request. The pick step also sees candidate product names, passed as JSON data in the user turn; its output is constrained to those candidate IDs and re-checked in code, and `eval_llm.py` includes a prompt-injection case. Prices, stock and guardrails never come from the model.
- **Observability.** Trace every request. Track the autonomy rate, approval turnaround time, substitution acceptance rate and parser-mismatch rate.
