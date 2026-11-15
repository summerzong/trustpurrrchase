# TrustPurrrchase: Junction 2026 (Valio Aimo challenge)

[Open the interactive demo](https://summerzong--demo.us-east.modal.run/) · Team Crux Cats

This is the onboarding guide for coding agents and new teammates. Read it before changing anything. Design rationale, the demo script, known gaps and the roadmap are in [`demo/README.md`](demo/README.md) (see "Next steps").

## What this is

This is our entry for the Junction 2026 challenge "Trusted Trade Agents" by Valio Aimo, a Finnish food-service wholesaler. An external purchasing agent takes a buyer's natural-language request and handles three kinds of requests.

**A concrete order**, for example *"I need 1,000 litres of lactose-free milk by Monday. Find an option that meets our requirements based on price and availability."* or *"We need 50 kg of butter by Monday"*. The agent:

1. turns it into a structured query (rule parser for dates, quantities, units, diet words and brands; cross-checked by Claude when available),
2. finds matching products in Valio Aimo's synthetic catalogue: with reviewed rules for reviewed product types (milk today), or, for any other product, with a catalogue search plus Claude's verdict on every result,
3. checks live price and availability on Valio Aimo's mock API,
4. ranks the options with multiple objectives and explains the choice,
5. **asks the buyer to confirm the order**. The selected quote is re-checked live before the simulated order is recorded,
6. when nothing can be ordered, **explains why** and offers relaxations that need a person's OK (a later date, another brand, a substitute). Diet and allergy requirements are never relaxed,
7. writes an audit trail. When a human picks a non-recommended option, it learns preference weights.

**An abstract need**, for example *"hot rhubarb-themed main courses in a Scandinavian style?"* Claude suggests dishes and ingredients, the agent matches each ingredient to a real catalogue product (Claude can only pick from real candidate IDs), screens it against the buyer's diet, quotes it live, suggests a substitute for anything missing or unavailable, and returns a **proposal with questions. It never orders.**

**Named dishes, or one product for portions with different diets**, for example *"We're catering a wedding for 100 guests on Saturday and want to serve salmon and carrot soup. Can you source the ingredients?"* or *"we need 4 litres of cooking cream … for 40 patients. 6 of them are allergic to milk protein and gluten."* The agent writes the recipe (or takes the one product), scales it to the portions, splits lines per diet group where the needs differ, quotes every line live, gives a red line Best / Better / Good alternatives (never relaxing a hard rule; missing allergen data only with an explicit acknowledgement), and **orders after one confirmation of the whole order** (`basket.py`).

The design rule behind all three: **the LLM understands and suggests; the catalogue, the live API and explicit rules decide what exists, what it costs and what the agent may do alone.**

The judges score: an end-to-end demo (30%), matching the buyer's need (20%), autonomy and guardrails (20%), robustness to exceptions (15%), and open standards with a path to production (15%).

## Setup (about 2 minutes)

You need Python **3.9 or newer**. Run the setup commands from `demo/`.

```bash
cd trusted-trade-agents/demo

# 1. Secrets: create your own .env (it is git-ignored, never commit it)
cp .env.example .env
#    then set AIMO_API_KEY=...  (Junction platform -> challenge "Trusted Trade Agents" -> Resources;
#    also page 4 of the challenge PDF; or ask a teammate). Set GEMINI_API_KEY for the language model.

# 2. Data: download the git-ignored challenge files into ../data/ (~87 MB, ~10 s)
python3 fetch_data.py

# 3. Enable Claude (needs ANTHROPIC_API_KEY in .env) and the hospital allergen solver
python3 -m pip install -r requirements.txt
```

`fetch_data.py` reads `AIMO_API_KEY` from `.env` and downloads three files through the API's `/api/*/download` endpoints: the product data JSON, the delivery-reliability CSV and the carbon-footprint CSV. It skips files that already exist, and `--force` re-downloads them. The product-data JSON Schema and the API docs are on the resource portal (`https://junction26.valioaimo.fi/`, event password on the challenge page). The code does not need them to run.

## Verify the setup

```bash
python3 -m unittest discover -s tests      # offline: no network, no LLM, no files written (85 tests)
python3 agent.py "I need 1,000 litres of lactose-free milk by Monday. Find an option that meets our requirements based on price and availability." --no-llm
```

When the setup works, the output shows sections ①–④ and ends with a status line:

- The first run builds `demo/.cache/catalog_index.json` (about 2 s). Later runs are faster.
- The `② Retrieval` line reads `17546 products → 91 in category → … candidates → … feasible`.
- The final line is one of `→ PENDING_APPROVAL` (a person confirms every order; `require_human_confirmation` in `policy.json`), `ESCALATED_NO_OPTION`, or `PLACED_AUTONOMOUSLY` (only when that switch is off).
- **Earlier orders count.** A repeat of the same request within 10 minutes returns the earlier order instead of a new one, and any open order for the same product and delivery date within 24 hours sends a new one to approval ("possible duplicate"). To rehearse the demo from a clean state, delete `demo/out/`.
- **The outcome depends on the current date and time.** "By Monday" means the next Monday. Stock items have an order cut-off, which was Sun 17:00 Helsinki time for a Monday delivery. Before that cut-off, the example ordered Valio product 464306 at €1.01/L. After it, expect `ESCALATED_NO_OPTION` with the reason and a relaxation "Deliver on <date>"; accepting it (UI button, or `agent.py relax ORD-… R1`) re-runs the request with that date. **That result is correct behaviour, not a bug.**

Common failures:

| Symptom | Cause / fix |
|---|---|
| `Set AIMO_API_KEY` or HTTP 401 | `.env` is missing or the key is wrong |
| `… product_data.json is missing` | Run `python3 fetch_data.py` |
| HTTP 403 `error code: 1010` | Cloudflare is blocking the request's User-Agent. Our client sets its own UA, so keep it if you write new HTTP code |
| `LLM parser skipped` / `LLM parser unavailable (…)` in the output | The `anthropic` SDK or `ANTHROPIC_API_KEY` is missing. The rule parser is used instead, which is fine for orders; menu ideas then fall back to "please clarify" |
| `ValueError: rules/diet_terms.json has codes without a rule` | A code in `diet_terms.json` is misspelled, or is neither in the data nor in `diet_overrides.json` |

## Online demo (deploy)

The submission demo runs on Modal with the entry point in [deploy/modal/app.py](deploy/modal/app.py). Create a Modal secret named `trustpurrrchase-api` containing `AIMO_API_KEY` and `GEMINI_API_KEY`, then run `modal deploy deploy/modal/app.py` from the repository root. The first start downloads the challenge catalogue into a Modal Volume.

The repository runs as one Docker image on any Docker host (Hugging Face Spaces, Render, Railway, Fly.io).
Set the secret `AIMO_API_KEY` (required) and `GEMINI_API_KEY` (optional). The challenge data is not in the
repository; `demo/start.sh` downloads it on first start, then serves the TrustPurrrchase chat UI at `/`
(classic UI at `/classic`) on port 7860.

```bash
docker build -t trustpurrrchase . && docker run -p 7860:7860 -e AIMO_API_KEY=… -e GEMINI_API_KEY=… trustpurrrchase
```

Hugging Face Space: create a Docker Space, upload `deploy/huggingface/Dockerfile` and `deploy/huggingface/README.md`
(the Space builds this public repository; set `BRANCH` if it is not `main`), and add the two secrets in the Space settings.
The agent never orders on its own (`require_human_confirmation` in `policy.json`), and orders exist only in the demo's
own order book: the mock API has no order endpoint.

## Run

```bash
python3 server.py                     # web UI at http://127.0.0.1:8000 and the TrustPurrrchase chat UI at /trustpurrrchase (localhost only; PORT env to change)
AGENT_TODAY=2026-11-15 python3 server.py   # pin 'today' for relative dates ("tomorrow"), e.g. to record the demo before Sunday
python3 agent.py "<request>"          # CLI, human-readable output
python3 agent.py "<request>" --json   # the structured response (the agent-facing contract)
python3 agent.py "<request>" --no-llm # deterministic parser only, no Claude call
python3 agent.py "<request>" --new    # the same goods again: skips the replay (only matters when check_duplicate_orders is on)
python3 agent.py "<request>" --simulate hospital_delay   # labelled fault injection from scenarios/ (demo scenario 3)
python3 agent.py "<request>" --idempotency-key K   # a caller's retry key: the same key returns the same order
python3 agent.py approve ORD-… [--choose N] [--by NAME]   # human sign-off; N != 1 triggers preference learning
python3 agent.py reject  ORD-…
python3 agent.py relax   ORD-… R1 [--by NAME]   # accept relaxation R1 of an escalated order; re-runs the request
python3 agent.py confirm BSK-… [--prefer price] [--pick LINE=OPTION] [--ack LINE] [--qty LINE=UNITS]   # one confirmation of a multi-line order
python3 agent.py cancel  ORD-…|BSK-…            # undo a placed or confirmed order before its cut-off

python3 -m unittest discover -s tests # offline regression tests
python3 eval_llm.py                   # end-to-end checks of the Claude steps: COSTS MONEY (~20 calls), writes nothing
python3 build_diet_vocab.py           # only after the product data changes: regenerates rules/diet_vocab.json + diet_coverage.md
python3 review_diet_rule.py FREE_FROM_GLUTEN   # review sheet with the actual products (rules/review/); --all, --list
python3 review_diet_rule.py --sign FREE_FROM_GLUTEN --by "Name"   # sign it after reading (stores the sheet hash)
python3 review_product_type.py milk            # review sheet for a product type; --sign milk --by "Name"
```

The three video scenarios (① milk, every rule passes and a person confirms; ② wedding soup, one red line with Best / Better / Good; ③ hospital, 34 + 6 allergy portions) and their recording notes are in `demo/README.md` ("Demo scenarios"). The other demo scenarios (happy path, over budget, brand preference, hard diet and allergy rules, too soon → later date, "only Arla", substitutes, any product, menu idea) are in `demo/README.md` and are also example buttons in the UI. To reset demo state (orders, proposals, learned weights), delete `demo/out/`.

## Layout

```
trusted-trade-agents/
├── README.md            ← this file (setup + orientation)
├── .gitignore           ← ignores .env, data/, *.pdf, .cache/, out/, demo/rules/review/ (generated sheets)
├── data/                ← git-ignored; created by demo/fetch_data.py
└── demo/
    ├── README.md        design, rules, guardrails, assumptions, demo script
    ├── agent.py         CLI + orchestration (run_request, run_proposal), requirement strengths, "why nothing
    │                       could be ordered", relaxations (relax_order), order book, audit, approve/reject
    ├── nlu.py           ① request → query or menu goal: rule parser + optional Claude (JSON-schema output),
    │                       cross-checked and quote-grounded; also the recommend and pick steps for menu ideas
    ├── catalog.py       ② static retrieval: slim catalogue index, product ontology, diet-evidence rules,
    │                       allergen_verdict (reject / warn / pass), generic word search for proposals
    ├── basket.py        named dishes → multi-line order: recipe lines per diet group, traffic lights, Best / Better /
    │                       Good / Later / warn tiers, one preference question, one live-checked confirmation, audit
    ├── aimo_api.py      ③ Valio Aimo API client (price, availability); logs SHA-256 of every response
    ├── aimo_contract.py ③ the API contract in one place: what each response field means, its source (doc / observed),
    │                       troubleshooting rules (assess) and contract anomalies; see "The API contract"
    ├── probe_contract.py   checks the live API against the contract (6 calls, writes nothing; --save records a fixture)
    ├── engine.py        ③④⑤ live troubleshooting checks, order plans, ranking/Pareto, guardrails, explanations, learning
    ├── server.py        web server (stdlib http.server): GET /, POST /api/request (Idempotency-Key header), /api/decide,
    │                       /api/relax, /api/cancel, /api/change, /api/basket/preference, /api/basket/confirm
    ├── schema/          JSON Schemas of the responses (order_response, basket): the agent interface contract
    ├── scenarios/       labelled fault-injection fixtures for the demo (hospital_delay.json), off by default
    ├── web/index.html   single-file UI (inline CSS/JS, no external deps)
    ├── web/trustpurrrchase.html   the TrustPurrrchase chat UI with Maito (served at /trustpurrrchase), wired to the same /api/* endpoints
    ├── policy.json      buyer guardrails (spend limit, price ceiling, split orders, reliability, …)
    ├── rules/           product_types.json (reviewed product types), order_guide.json (the buyer's usual SKUs), diet_vocab.json (generated),
    │                       diet_terms.json (synonyms), diet_overrides.json (policy),
    │                       diet_coverage.md + review/<code>.md (generated reports; review/ is git-ignored)
    ├── build_diet_vocab.py  regenerates rules/diet_vocab.json + diet_coverage.md from the data
    ├── review_diet_rule.py  writes a review sheet per diet rule, with the products behind every number; --sign
    ├── review_product_type.py  the same for a product type (accepted, excluded, possible misses); --sign
    ├── fetch_data.py    downloads the data files
    ├── tests/           offline unit tests (python3 -m unittest discover -s tests); fixtures/ = recorded API responses
    ├── eval_llm.py      end-to-end checks of the Claude steps (costs money; writes nothing)
    ├── requirements.txt Claude integration and Z3 for the hospital scenario
    ├── .env.example     template for .env
    ├── .cache/          generated: catalog_index.json
    └── out/             generated: orders.json, prefs.json (learned weights), audit/<order>.md
```

## How a request flows

`agent.run_request(text, policy)` first interprets the request, then takes one of three paths.

**Interpretation** (`nlu.interpret`) returns the `query` and a `trace`:

- The rule parser always runs (`nlu.parse_rules_detailed`). It reads requirements such as "only Arla" or "vain luomu" (`must_have`: brand, fat level or organic), allergy phrasings ("fish allergy", "allergic to sesame", "coeliac", mapped to the matching free-from rule), dates (`16.11.`, `16/11`, `20 November`, `18. marraskuuta`, tomorrow, in 3 days, weekdays in en/fi/sv), quantities with units (`1,000 litres`, `2 dl`, `500 g`, `50 x 20 l`, `50 boxes of 20 l`, `a thousand liters`, `tuhat litraa`), diet words, and any brand in the data that is not a common word (`catalog.brand_index`), including exclusions such as "no Arla", "ei Arlaa" and "utan Arla". What it cannot read is reported, not dropped: a second quantity or a package count without a size blocks the search; a vague or unreadable date, a vague quantity or a pack-size wish becomes an issue.
- When Claude is available, it returns an intent (`order`, `recommend`, `source`, `clarify`, `out_of_scope`) and a structured order through a JSON-schema-constrained call. Portions and diet sub-groups ("6 of them are allergic to milk protein and gluten") are read by both parsers (`nlu.parse_portions`, `nlu.parse_diet_groups`, merged in `nlu._merge_groups`); a sub-group's words are never the whole order's product or diet. Brands it names are snapped to real brands in the data; unknown or fuzzy matches become issues.
- For each critical field (product, quantity, unit, date, diet, excluded brands, `must_have`), the two readings are compared. A disagreement, a value only Claude read (it must quote the request; an unquoted value is discarded), or a requirement nothing can express (`unsupported_requirements`, plus a rules-side detector for `X-free`, `without X`, `no <brand>`, …) goes into `trace.issues`, which forces human approval.
- Missing fields go into `trace.blocking`, which skips the search and asks the buyer to clarify. A quantity is never guessed.

**Order path** (intent `order`, the default):

1. Retrieval records a reason code for every product it rejects.
   - **Reviewed product type** (`rules/product_types.json`, milk today): `catalog.retrieve` applies its category codes, include terms and excluded subtypes. 17,546 products become about 27 candidates.
   - **Any other product** (`product_type: "other"`): `catalog.search_products` (Finnish, English and Swedish names) returns up to 40 results, `nlu.classify_candidates` has Claude give every result a verdict (`is_the_product`, `variant`, `contains_it`, `different_product`). Each verdict's `id` is an enum of the candidate IDs, so Claude cannot invent a product, and a result it skips counts as "not the product". (One required property per candidate would also force completeness, but the API rejects that schema at 40 candidates: "compiled grammar is too large".), and `catalog.check_candidates` keeps exact matches for standalone orders; variants are not ranked as matches. Rejected results carry Claude's reason (`NOT_THE_PRODUCT`).
   - Both paths then apply the same deterministic checks: excluded brands (`AVOIDED_BRAND`), the buyer's "only …" requirement (`MUST_HAVE`), diet evidence (`catalog.diet_evidence`, see "Diet rules" below) and conversion to the order's measure (L, KG or PCS; `catalog.amount_per_sales_unit`). Each rejection records the requirement it enforced, so the explanation can say which one emptied the list.
2. `AimoClient.price` and `.availability` are called **for those candidates only**.
3. `engine.live_check` turns them into feasible offers and records more rejects (`CUTOFF_PASSED`, `TOO_LATE`, …). Every response is read through `aimo_contract.assess` (see "The API contract" below): where it breaks the documented contract, the offer carries an anomaly, for example `STALE_CUTOFF` for an item whose order cut-off has passed but whose deadline is later than its earliest date, and the guardrail "live data consistent with the API contract" sends it to a person. Each line is ordered for the earliest date the API offers, with `daysEarly` saying how long before the deadline that is.
4. `engine.build_plans` builds single-source plans, plus a split plan when needed. `engine.rank` scores them.
5. `engine.guardrails(best, …)` returns `AUTO_ORDER`, `NEEDS_HUMAN_APPROVAL` or `ESCALATE_NO_OPTION`. Failing checks are added when the decision relies on something no person has reviewed: an unreviewed diet rule, a product outside the reviewed types, a line Claude judged a variant, or a product without a price ceiling in `policy.json` (`max_price_per_unit_eur`, per product type). Name-derived weights that disagree with the price API's comparison price are replaced by the API value and flagged `UNIT_MISMATCH`, which also fails a check.
6. If no option is left (`ESCALATED_NO_OPTION`), `agent._next_steps` explains why, from the recorded rejections (for vegan milk: "Hard requirement 'vegan': all 75 of 75 milk products were ruled out (75 contradicted by the allergen data)"), and `agent._relaxations` lists options `R1, R2, …` that a person may accept: the earliest delivery date, dropping an excluded brand or an "only …" requirement, or a substitute product. Accepting one (`agent.relax_order`, `POST /api/relax`, CLI `relax`) re-runs the whole pipeline with that single change; the new order links back (`relaxed_from`, and `superseded_by` on the old one) and carries the passing check "relaxation accepted by <name>".
7. The response dict (including `constraints` and `relaxations`) is written to `out/orders.json` and `out/audit/<id>.md`. The CLI renders it, or the UI receives it as JSON.

**Commit controls** (what makes ordering safe to retry and safe to delay):

- **Switch:** `check_duplicate_orders` in `policy.json` turns the same-request replay and the duplicate guardrail below on or off (default on when the key is missing; **off for the demo**, so scenarios can be rehearsed again and again). A caller's own `Idempotency-Key` is honoured either way.
- **Idempotent replay** (`agent.run_request`, `request_hash`). Before anything is searched, a request is matched against earlier orders: the caller's `idempotency_key` (body field or `Idempotency-Key` header, honoured 24 h; reusing a key for a different request is an error), or the same normalised request within `idempotency_window_minutes` (10) while the earlier order is placed, approved or waiting. The hash covers what is bought, how much, by when and under which requirements, not the wording: "1,000 litres of lactose-free milk" and "1000 l laktoositon maito, cheapest" are the same request. A match returns the earlier order with `idempotent_replay` and orders nothing. `allow_duplicate` (`--new`, UI "Order again anyway") skips the replay.
- **Duplicate guardrail** (`agent._possible_duplicates`). Any other open order for the same product and delivery date within `duplicate_order_window_hours` (24) fails the check "no other open order for the same product and delivery date", so a second order for the same goods always needs a person.
- **Re-check at approval** (`agent.decide_order`, `engine.revalidate`). Approving commits, so the chosen option's products are priced and checked again live first. If a price rose, the stock fell below the order, the cut-off passed or the date moved past the deadline, nothing is placed: the order becomes `STALE_QUOTE` and the same query is re-planned with live data (`stale_quote` on the new order lists what changed). A lower price is committed at the fresh price. If the API cannot be reached, the approval is refused (fail closed). The audit records the fresh quote with its response hashes.
- **Repeated decisions** return the stored result: approving the same option twice, or accepting the same relaxation twice, places nothing new.

**Proposal path** (intent `recommend`, `agent.run_proposal`):

1. `nlu.recommend`: Claude suggests 3–5 dishes, each with ingredients, a form (fresh, frozen, …), Finnish and English search terms, and a quantity per portion.
2. `catalog.search_products`: a static word search per ingredient, at most 8 candidates each. English terms match whole words, so "pork" cannot hit "porkkana" (carrot).
3. `nlu.pick_products`: Claude picks one product (and a runner-up) per ingredient. Its schema is built per request, with an enum of that ingredient's candidate IDs plus `NONE`, so it cannot invent a product.
4. Diet screen (`agent._diet_screen`): when the buyer stated a diet, every picked product is checked with the same `catalog.diet_evidence` as the order path. A contradicted product is removed; a product without evidence either way (fresh produce has no vegan claim) is kept as `DIET_UNVERIFIED` for a person to confirm. Pantry items (salt, flour) are not searched, and the proposal says they were not checked.
5. `engine.quote_products`: live price and availability for the picked products only, with the same troubleshooting rules as the order path. If the first choice cannot be ordered, the runner-up is shown as a substitute.
6. Substitutes (`agent._substitute_options`, `nlu.suggest_substitutes`): for an ingredient that is not in the catalogue, or a key ingredient that cannot be delivered, Claude suggests up to 3 alternatives, which go through the same search, diet screen and live quote. The line becomes `SUBSTITUTE_SUGGESTED` ("'durian pulp' is not in the catalogue. Suggested instead: frozen mango pulp (Findus 460040). Needs your OK."). If the pick call fails, the line is `NOT_PICKED`; the agent never falls back to the first search hit.
7. Status `PROPOSAL_FOR_REVIEW`, with questions (portions, date, which dishes). Nothing is ordered and no guardrail runs.

**Multi-line path** (intent `source`, or an order with a diet sub-group; `agent.run_basket` → `basket.run_basket`):

1. Recipe: `nlu.recipe_for` writes the standard recipe per named dish with a quantity per portion (not adapted to the sub-groups). A one-product order becomes a one-line recipe (`agent._split_recipe`: 4 L for 40 portions = 0.1 L each).
2. Groups and lines: the regular group plus each diet sub-group (`basket.build_groups`). A line is shared when the first choice is safe for every group, otherwise split per group.
3. First choice: the buyer's usual product (`rules/order_guide.json`), else the cheapest match for the quantity. Candidates come from `catalog.search_products`; Claude's identity verdicts for all ingredients come in one call (`nlu.classify_groups`).
4. Allergens: `catalog.allergen_verdict` per group: `reject` (the data contradicts the need, never offered), `warn` (no data: offered only with an acknowledgement and a label check), `pass` (structured evidence or a reviewed rule).
5. Live quote: price and availability for the identified candidates; green = the first choice can be ordered in time.
6. Red lines get tiers (`basket._tiers_same_product`, `basket._add_substitutes`): Best (same product, pack ±25 %, price ≤ +5 %), Better (pack, price or form relaxed), Good (another ingredient with the same function), Later (safe, after the deadline), warn (missing data). One preference question (price, delivery, stock, fit) re-ranks inside the tiers only.
7. `AWAITING_CONFIRMATION` → `basket.confirm`: one confirmation for the whole order, with optional per-line quantities; every line is re-checked live (`engine.revalidate`); a changed quote gives `STALE_QUOTE`. The audit names each line's tier and gate; a warned line creates an arrival check task and is never learned.

## Requirement strength and relaxations

Every requirement in the query gets a strength (`agent._constraints`, shown in step 2 of the UI and in the audit):

| Strength | Requirements | What the agent may do |
|---|---|---|
| hard | diet and allergy rules | Never relaxed. If nothing meets them, the agent explains why and stops; there is no relaxation button |
| negotiable | product, quantity, delivery date, excluded brands, "only …" (`must_have`) | Enforced as filters. When they empty the list, the agent proposes a relaxation; only a person can accept it, and the re-run order records who did |
| preference | brand, fat level, organic | Only rank the options. Choosing outside them is allowed but fails the guardrail "matches your stated preferences", so a person approves |

A relaxation changes exactly one requirement and re-runs the pipeline, so every other rule and guardrail still applies to the new result.

## Diet rules

`catalog.DIET_RULES` is built at import from `demo/rules/`. Both the LLM schema enum and the rule parser's `nlu.DIET_TERMS` come from it, so the two parsers always cover the same codes.

| File | What it holds | Who maintains it |
|---|---|---|
| `diet_vocab.json` | What the data says: all 21 GS1 claim codes with product counts, relations derived from the claim grammar (`FREE_FROM_X` satisfies `LOW_X`; `LOW_X` is a look-alike of `FREE_FROM_X`), and *suggested* contradicting allergens from co-occurrence statistics | Generated by `build_diet_vocab.py`; never edit by hand |
| `diet_terms.json` | Words per code: what buyers say (→ `nlu.DIET_TERMS`) and what appears in product names (weak evidence) | People. Non-lactose terms are drafts that need a native-speaker review |
| `diet_overrides.json` | What we decide to rely on: nutrient thresholds, the contradicting allergens actually used, whether MAY_CONTAIN counts, composite concepts with no GS1 claim (`DAIRY_FREE`, `NUT_FREE`), and `reviewed` | People. The only diet file that needs careful review |
| `diet_coverage.md` | Report: each rule run on every product (strong / conflict / name only / look-alike), suggestions next to policy | Generated; not read at runtime |

Every code found in the data gets a rule automatically and can be searched and verified. `diet_overrides.json` adds composite rules with no GS1 claim: `DAIRY_FREE`, `NUT_FREE` and the other EU major allergens (`FISH_FREE`, `CRUSTACEAN_FREE`, `MOLLUSC_FREE`, `PEANUT_FREE`, `SESAME_FREE`, `CELERY_FREE`, `MUSTARD_FREE`, `SULPHITE_FREE`, `LUPIN_FREE`), each backed by the nonAllergen FREE_FROM entry and contradicted by CONTAINS or MAY_CONTAIN; 32 rules in total. Allergy phrasings map to them; "peanut allergy" also matches "nut allergy", so it applies `NUT_FREE` too, which is stricter but never unsafe. **Only rules with `reviewed: true` (today: the two lactose rules, reviewer still the placeholder "demo team") can be part of an autonomous order.**

Evidence per product is one of: pass (strong, or weak / conflict at strength 0.5 for a person), `near_miss` (a look-alike such as low-lactose for lactose-free), `contradicted` (the data says the product does not meet the need, e.g. milk for vegan) or fail (no evidence either way). The explanation for an empty result names which of these removed the candidates.

**How to review a diet rule.** Run `python3 review_diet_rule.py <CODE>` and work through `rules/review/<CODE>.md`: the rule as it runs, the data's suggestions next to the policy, every conflict, every name-only match, every look-alike, a random sample of strong matches, and a checklist. Fix `diet_terms.json` / `diet_overrides.json` where needed (then regenerate the sheet), and sign: `python3 review_diet_rule.py --sign <CODE> --by "<name>"`. Signing refuses a sheet that no longer matches the data, then sets `reviewed`, `reviewed_by`, `reviewed_at` and `review_sheet_sha256` without reformatting the file. The audit trail of every order names the reviewed rules it relied on, with the sheet hash. If the data or the rule changes later, `--list` shows "CHANGED since the review" and the offline tests fail until someone reviews it again. No Python changes are needed.

## Product types

`rules/product_types.json` holds the product types a person has reviewed (milk today): category codes, include terms, excluded subtypes with reasons, the measure (L, KG or PCS), buyer words per language and `reviewed_by` / `reviewed_at`. `python3 review_product_type.py milk` writes `rules/review/type_milk.md`: every accepted product with its litres per sales unit and how they were found, every excluded subtype, products in the category the name terms miss, and products outside the category whose name contains a milk word (possible misses). Sign it with `--sign milk --by "<name>"`, as for diet rules. The lactose rules and milk still carry the placeholder reviewer "demo team" and no signed sheet. A request for any other product still works through the generic path above, but always ends in human approval. To make a new type autonomous, add it here with its rules, set a price ceiling for it in `policy.json`, and mark it reviewed.

## Where to change things

| Goal | Edit |
|---|---|
| Make a product type autonomous (cream, butter, …) | Add it to `rules/product_types.json` (categories, include / exclude terms, measure, buyer words, reviewer) and a price ceiling to `policy.json` (`max_price_per_unit_eur`). The rule parser's synonyms and the LLM enum follow automatically |
| Brands | Generated from the data (`catalog.brand_index`): family matching ("Arla" covers "Arla Pro"), common-word brands flagged so the rule parser ignores them, `difflib` snapping for Claude's spelling |
| Dates, quantities, units | Vocabulary at the top of `nlu.py` (`UNIT_WORDS`, `PACKAGE_WORDS`, `NUMBER_WORDS`, `MONTHS`, `WEEKDAYS`, ...); logic in `nlu.parse_date` and `nlu.parse_quantity` |
| Dietary rules (gluten-free, vegan, …) | No Python. Every GS1 claim code in the data already has a rule, built at import from `demo/rules/`: `diet_vocab.json` (generated by `python3 build_diet_vocab.py`, never edited), `diet_terms.json` (buyer words → `nlu.DIET_TERMS`, product-name words), `diet_overrides.json` (policy: thresholds, contradicting allergens, MAY_CONTAIN, composite concepts such as `DAIRY_FREE`, and `reviewed`). Read `rules/diet_coverage.md`, then set `reviewed: true` to let a rule order autonomously. Bump `catalog.INDEX_VERSION` only if `_slim` needs a new field |
| Menu proposals (recipes, product pick) | `nlu.RECOMMEND_SYSTEM` / `RECOMMEND_SCHEMA` (search terms in fi, en and sv), `nlu.PICK_SYSTEM` / `build_pick_schema`, `catalog.search_products`, `agent.run_proposal` |
| Requirement strengths, explanations, relaxations | `agent._constraints`, `agent._next_steps`, `agent._relaxations`, `agent.relax_order`; the substitute step in proposals: `agent._substitute_options` (`SUBSTITUTE_ITEMS`), `nlu.SUBSTITUTE_SCHEMA`, `nlu.SUBSTITUTE_MAX` |
| Idempotency, duplicate orders, re-check at approval | `policy.json` (`check_duplicate_orders`, `idempotency_window_minutes`, `duplicate_order_window_hours`), `agent.request_hash` (which query fields make two requests the same), `agent._find_replay`, `agent._possible_duplicates`, `engine.revalidate` (what counts as a changed quote), `agent.decide_order` / `_replan_stale` |
| Generic product identification | `nlu.CLASSIFY_SYSTEM` / `build_classify_schema` / `VERDICTS`, `agent._generic_retrieve` (`GENERIC_CANDIDATES` = 40), `catalog.check_candidates` |
| Guardrail thresholds | `policy.json`. A new kind of check goes in `engine.guardrails` |
| What the API's fields mean (cut-off, sentinels, schedule lines, a new API version) | `aimo_contract.py` (`CONTRACT`, `assess`), then run `python3 probe_contract.py` against the live API and re-record the fixture with `--save tests/fixtures/…`. The README table above must list every `CONTRACT` id (a test checks it) |
| Ranking objectives and default weights | `engine.utilities` / `engine.OBJECTIVES`, `agent.DEFAULT_PREFS` |
| Claude prompt or model | `nlu.MODEL` (`claude-opus-5-5`) and `nlu._call_structured` (server-side refusal fallback; adaptive thinking with `display: "summarized"`; every call logged in `trace.llm_calls` with tokens, its `constraint` and the `reasoning_summary`); prompts in `nlu._interpret_system`, `RECOMMEND_SYSTEM`, `PICK_SYSTEM`, `CLASSIFY_SYSTEM`; efforts: interpret low, recommend medium, pick low, classify low |
| "How the agent decided" view | `web/index.html` (`stepRead` … `stepDecide`, `renderSteps`). Data: `trace.spans` (`nlu.rule_spans`, `nlu.quote_spans`), `trace.field_checks` (`nlu._apply_llm`), `guardrail_checks[].group/ref` (`agent.CHECK_SOURCES`) |
| UI | `web/index.html`. It is served from disk, so a browser refresh is enough. Python changes need a `server.py` restart |

## The API contract (`aimo_contract.py`)

Everything the agent assumes about `/api/availability` and `/api/price` lives in `demo/aimo_contract.py` (`CONTRACT`), with its source: the API docs (`data/openapi.json`, the product-data schema) or what we observed on the mock. `aimo_contract.assess` reads every response through it. A response that breaks the contract gets an **anomaly** on the offer, and any anomaly fails the guardrail "live data consistent with the API contract", so a person approves. On an API that keeps the contract no anomaly fires and nothing else changes. `python3 probe_contract.py` checks the live API (6 calls, writes nothing); `tests/fixtures/mock_api_2026-11-14.json` is its recording of the mock, and the offline tests replay it.

| Contract item | Source | Rule | Anomaly when broken | Mock on 2026-11-14 |
|---|---|---|---|---|
| `ONE_SCHEDULE_LINE` | doc | An available product has exactly one schedule line; empty = not deliverable | `MULTIPLE_SCHEDULE_LINES`: lines up to the deadline are added up, a person approves | kept |
| `EARLIEST_DATE` | doc + observed | `availableDate` is the earliest delivery date and can differ from `requestedDate` | none: the agent orders for that date and says how many days before the deadline it arrives (`daysEarly`) | the same date, cut-off and price for any `requestedDate` |
| `CUTOFF_FOR_AVAILABLE_DATE` | doc + observed | `closingTime` is the order cut-off for `availableDate` | `STALE_CUTOFF`: needed on that date → rejected (`CUTOFF_PASSED`); needed later → kept, a person confirms | **broken** on Terminal items: cut-off Thu 15:00 passed, still Monday |
| `TERMINAL_AMOUNT` | doc | Terminal items always report `999999` (quantity unknown) | `UNEXPECTED_AMOUNT`: the amount is a limit, unconfirmed | kept |
| `AMPLE_SENTINEL` | observed | `9999999999` base units, scaled down to sales units, = ample stock; a Stock item with `999999` is unexpected | `UNEXPECTED_AMOUNT` | kept |
| `LOGISTIC_MODEL` | doc | `logisticModel` is `Stock` or `Terminal` | `UNKNOWN_LOGISTIC_MODEL`: quantity unconfirmed | kept |
| `DATE_ECHO` | doc | Responses echo `requestedDate` / `pricingDate` | `DATE_ECHO_MISMATCH` on every product of the response | kept |
| `PRICE_FOR_ONE_UNIT` | doc | A price is for quantity 1.0, so line total = unit price × quantity | `PRICE_QUANTITY` | kept |
| `LOT_SIZE_IN_SALES_UNITS` | product schema | `allowedLotSize` is the order step in sales units: count units (ST, KI, …) in whole units, KG in lot steps (fractional lots occur only on KG) | none (rounding rule in `engine._line`) | – |
| `CUSTOMER_INDEPENDENT` | doc | The customer id does not change prices or availability in the challenge | none: the customer id from `policy.json` is always sent | – |

**Delivery date (a decision, kept simple):** the mock only offers its earliest date, so every line is ordered for that date. When it is before the deadline, the response says so (`daysEarly` on each line, the explanation "Arrives 4 days earlier than needed (2026-11-16 instead of 2026-11-20) …", the UI next to the delivery date, the audit). The agent does not try to schedule a later delivery.

Other facts about the mock API (verified, non-obvious):

- **Base URL:** `https://junction26.valioaimo.fi`. Authenticate with the `X-API-Key` header. `POST /api/price` and `POST /api/availability` take `{customer, requestedDate, products[≤1000]}`.
- **Rate limits:** the API docs **forbid bulk-querying all products**, and the limits are undisclosed. Always narrow the list with static data first.
- **No order endpoint.** "Placing" an order means writing it to `out/orders.json`.
- **Units:** `ZML` is litre and `KG` is kilogram in the unit table (`{KG: 1, KI: 2}` = 2 kg per box). Category `10400` is milk and milk drinks (category names are not in the data). The price API's `comparisonPrice` (only per L or per KG) cross-checks our conversion. Portion packs show why: "Valio butter 6 g" is sold by the case, so the name's 6 g is one portion, and the API value is used instead.
- **Names:** 17,546 products have a Finnish name, 17,544 an English one and 16,100 a Swedish one.
- **Low-lactose trap:** "vähälaktoosinen", HYLA and "low lactose" are **not** lactose-free. Lactose-free evidence is the GS1 claim `FREE_FROM_LACTOSE` or the nutrient `LACS ≤ 0.01 g/100 g`.
- **Manufacturer data contradicts itself sometimes:** for example 460104 (Fria cinnamon buns) claims gluten-free but lists allergen AW (cereals containing gluten). Such products are kept but marked `CONFLICT`, so a person decides. `rules/diet_coverage.md` counts them per rule.

## Conventions for contributors (human or agent)

- Keep the core **standard-library only** and **Python 3.9 compatible**: no `match`, no `X | Y` types at runtime, and no nested same-type quotes in f-strings. Every module starts with `from __future__ import annotations`.
- **The LLM understands, suggests and picks among given options; it never decides facts.** Retrieval, pricing, availability, ranking, guardrails and ordering stay deterministic and auditable, and every LLM choice of a product is constrained to real candidate IDs. Closed vocabularies always have a way out (null, `other`, `unsupported_requirements`). This separation is part of the pitch.
- Every rejection or escalation must carry a human-readable reason, because it appears in the UI and in the audit trail.
- Diet knowledge lives in `demo/rules/*.json`, not in Python. Never hand-edit `diet_vocab.json`. A new rule starts with `reviewed: false`.
- **Claude structured-output limits** (found the hard way): a schema may have at most 16 nullable or union parameters, and a schema with one property per candidate fails at about 40 candidates with "compiled grammar is too large". Over the first limit, *every* interpret call fails and the agent silently falls back to the rule parser, so `tests/` checks the count. Use plain strings with `""` for "not stated", and arrays of `{id: enum, …}` items checked for completeness in code.
- Run `python3 -m unittest discover -s tests` before you share a change. If you change a prompt or schema, also run `eval_llm.py`, which costs money, so ask first.
- Never commit `.env`, `data/`, the challenge PDF, `.cache/` or `out/`. The root `.gitignore` already covers them.
