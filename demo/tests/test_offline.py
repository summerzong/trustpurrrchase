"""Offline regression tests: no network, no LLM, no files written.

    cd demo && python3 -m unittest discover -s tests -v

Catalogue tests need the data files (python3 fetch_data.py) and are skipped without them.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent  # noqa: E402
import aimo_api  # noqa: E402
import aimo_contract  # noqa: E402
import basket  # noqa: E402
import catalog  # noqa: E402
import engine  # noqa: E402
import nlu  # noqa: E402
import probe_contract  # noqa: E402

TODAY = dt.date(2026, 11, 14)  # a Saturday: "by Monday" = 2026-11-16
EXACT = ("I need 1,000 litres of lactose-free milk by Monday. "
         "Find an option that meets our requirements based on price and availability.")
RHUBARB = ("We've already planned to make rhubarb pudding, rhubarb bread, and rhubarb pancakes, but we'd like to make "
           "some hot rhubarb-themed main courses. Could you recommend some recipes? Our restaurant specializes in "
           "Scandinavian cuisine, so we need something in that style.")
HAS_DATA = os.path.exists(catalog.PRODUCT_FILE)


def _plan(measure="L", eur=1.0):
    return {"lines": [{"_o": {"reliability": 1.0, "qty_confirmed": True, "flags": [], "name": "x"}}], "shortfall": 0,
            "measure": measure, "amount": 10, "need": 10, "total": 10, "eur_per_unit": eur, "preference": 1.0,
            "evidence": 1.0}


def _policy():
    with open(os.path.join(os.path.dirname(catalog.__file__), "policy.json"), encoding="utf-8") as f:
        return json.load(f)


def _objects(schema, path="$"):
    """Yield (path, schema) for every object schema nested in `schema`."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield path, schema
        for k, v in schema.items():
            yield from _objects(v, f"{path}.{k}")
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            yield from _objects(v, f"{path}[{i}]")


def _envelope(intent="order", order=None, **kw):
    env = {"intent": intent, "order": order, "culinary_goal": None, "portions": None, "portions_quote": None,
           "unsupported_requirements": [], "clarifying_questions": []}
    env.update(kw)
    return env


def _order(quantity, unit="L", deliver_by="2026-11-16", diet=("FREE_FROM_LACTOSE",), evidence=None):
    ev = {"product_type": "milk", "quantity": None, "unit": None, "deliver_by": "by Monday", "diet": "lactose-free"}
    ev.update(evidence or {})
    return {"product_type": "milk", "search_terms": ["milk"], "quantity": quantity, "unit": unit,
            "deliver_by": deliver_by, "diet": list(diet),
            "preferences": {"brand": None, "fat_level": None, "organic": False}, "optimize_for": [], "evidence": ev}


class Vocabulary(unittest.TestCase):
    def test_diet_codes_have_a_single_source(self):
        self.assertLessEqual(set(nlu.DIET_TERMS), set(catalog.DIET_RULES))
        self.assertEqual(nlu.ORDER_SCHEMA["properties"]["diet"]["items"]["enum"], sorted(catalog.DIET_RULES))
        self.assertEqual(set(nlu.DIET_TERMS), {c for c, r in catalog.DIET_RULES.items() if r["request_terms"]})

    def test_every_claim_code_in_the_data_has_a_rule(self):
        vocab = catalog._read_rules_file("diet_vocab.json", {"codes": {}})
        self.assertEqual(len(vocab["codes"]), 21)
        self.assertLessEqual(set(vocab["codes"]), set(catalog.DIET_RULES))
        self.assertTrue({"DAIRY_FREE", "NUT_FREE"} <= set(catalog.DIET_RULES))  # composite, from the overrides

    def test_only_reviewed_rules_may_act_alone(self):
        self.assertEqual(sorted(c for c, r in catalog.DIET_RULES.items() if r["reviewed"]),
                         ["FREE_FROM_LACTOSE", "LOW_LACTOSE"])

    def test_generated_lactose_rules_match_the_reviewed_hand_written_ones(self):
        r = catalog.DIET_RULES
        self.assertEqual(r["FREE_FROM_LACTOSE"]["pos_claims"], ["FREE_FROM_LACTOSE"])
        self.assertEqual(tuple(r["FREE_FROM_LACTOSE"]["nutrient"]), ("LACS", "max", 0.01))
        self.assertEqual(r["FREE_FROM_LACTOSE"]["contradict_allergens"], ["ML"])
        self.assertEqual(r["FREE_FROM_LACTOSE"]["near_miss_claims"], ["LOW_LACTOSE"])
        self.assertTrue({"vähälaktoosinen", "low lactose", "hyla"} <= set(r["FREE_FROM_LACTOSE"]["near_miss_terms"]))
        self.assertEqual(r["LOW_LACTOSE"]["pos_claims"], ["LOW_LACTOSE", "FREE_FROM_LACTOSE"])  # from the claim grammar
        self.assertEqual(tuple(r["LOW_LACTOSE"]["nutrient"]), ("LACS", "max", 1.0))

    def test_relations_from_the_claim_grammar(self):
        r = catalog.DIET_RULES
        self.assertEqual(r["LOW_FAT"]["pos_claims"], ["LOW_FAT", "FREE_FROM_FAT"])
        self.assertEqual(r["FREE_FROM_SODIUM_SALT"]["near_miss_claims"], ["LOW_SODIUM_SALT"])
        self.assertEqual(r["VEGETARIAN"]["pos_claims"], ["VEGETARIAN", "VEGAN"])  # from the overrides

    def test_unreviewed_rule_needs_a_human(self):
        plan = _plan()
        policy = _policy()
        q = {"product_type": "milk", "preferences": {}, "diet": ["FREE_FROM_GLUTEN"]}
        self.assertEqual(engine.guardrails(plan, q, policy, [])[0], "AUTO_ORDER")
        decision, checks = engine.guardrails(plan, q, policy, [], ["FREE_FROM_GLUTEN"])
        self.assertEqual(decision, "NEEDS_HUMAN_APPROVAL")
        self.assertIn("diet rule reviewed by a person", checks[-1][1])

    def test_schemas_are_strict(self):
        groups = [{"key": "i0", "candidates": [{"id": "460519"}, {"id": "460404"}]}]
        for name, schema in [("request", nlu.REQUEST_SCHEMA), ("recommend", nlu.RECOMMEND_SCHEMA),
                             ("pick", nlu.build_pick_schema(groups))]:
            for path, obj in _objects(schema):
                with self.subTest(schema=name, path=path):
                    self.assertIs(obj.get("additionalProperties"), False)
                    self.assertEqual(sorted(obj["required"]), sorted(obj["properties"]))

    def test_schemas_stay_within_the_api_limits(self):
        # The API rejects a schema with more than 16 union-typed (nullable / anyOf) parameters. When that
        # happened, every interpret call failed and the agent silently fell back to the rule parser.
        def unions(schema):
            if isinstance(schema, dict):
                return int("anyOf" in schema or isinstance(schema.get("type"), list)) + sum(unions(v) for v in schema.values())
            if isinstance(schema, list):
                return sum(unions(v) for v in schema)
            return 0
        for name, schema in [("request", nlu.REQUEST_SCHEMA), ("recommend", nlu.RECOMMEND_SCHEMA),
                             ("substitute", nlu.SUBSTITUTE_SCHEMA)]:
            with self.subTest(schema=name):
                self.assertLessEqual(unions(schema), 16)

    def test_pick_schema_only_allows_real_candidates(self):
        s = nlu.build_pick_schema([{"key": "i0", "candidates": [{"id": "460519"}, {"id": "460404"}]}])
        item = s["properties"]["picks"]["items"]["properties"]
        self.assertEqual(item["product_id"]["enum"], ["460404", "460519", "NONE"])
        self.assertEqual(item["key"]["enum"], ["i0"])

    def test_pick_must_come_from_its_own_ingredient(self):
        groups = [{"key": "i0", "ingredient": "rhubarb", "form": "fresh", "used_in": [], "candidates": [
                       {"id": "460519", "name_en": "RHUBARBS", "name_fi": "RAPARPERI", "salesUnit": "KG"}]},
                  {"key": "i1", "ingredient": "pork", "form": "fresh", "used_in": [], "candidates": [
                       {"id": "462131", "name_en": "PORK ROAST", "name_fi": "PORSAAN PAISTI", "salesUnit": "KG"}]}]
        out = {"picks": [{"key": "i0", "product_id": "462131", "runner_up": "NONE", "reason": "wrong group"}]}
        with mock.patch.object(nlu, "_call_structured", return_value=out):
            picks = nlu.pick_products(groups, [])
        self.assertEqual(picks["i0"]["product_id"], "NONE")   # a candidate of another ingredient is refused
        self.assertEqual(picks["i1"]["product_id"], "NONE")   # a skipped ingredient gets no product


class RulesParser(unittest.TestCase):
    def test_exact_request_is_unchanged(self):
        q, tr = nlu.interpret(EXACT, TODAY, use_llm=False)
        self.assertEqual((q["product_type"], q["quantity"], q["unit"], q["deliver_by"], q["diet"]),
                         ("milk", 1000.0, "L", "2026-11-16", ["FREE_FROM_LACTOSE"]))
        self.assertEqual(tr["issues"], [])

    def test_requirement_hints(self):
        cases = {
            EXACT: [],
            "We need 1,000 litres of lactose-free Arla milk by Monday, cheapest available.": [],
            # diet codes generated from the data are understood, so they are not "unsupported"
            "We need gluten-free, lactose-free milk for Monday.": [],
            "gluteenitonta ja laktoositonta maitoa 1000 l": [],
            "vegan milk 100 l": [],
            "1000 l milk without lactose by Monday": [],   # negation word inside a known term
            "no added sugar oat drink 100 l": [],
            "20 cartons of fat-free milk": [],
            # exclusions and needs the data cannot verify are still reported
            "1000 l lactose-free milk, no Arla please, no later than Monday": [],  # executed as avoid_brands
            "NOT lactose-free milk please, 500 l": ["not lactose-free milk"],
            "1000 l kosher lactose-free milk by Monday": ["kosher"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(nlu.requirement_hints(text), expected)

    def test_unsupported_requirement_needs_a_human(self):
        _, tr = nlu.interpret("1000 l kosher lactose-free milk by Monday", TODAY, use_llm=False)
        self.assertTrue(any("cannot verify requirement: kosher" in i for i in tr["issues"]))

    def test_generated_diet_terms(self):
        q = nlu.parse_rules("1000 l gluteenitonta ja laktoositonta maitoa maanantaiksi", TODAY)
        self.assertEqual(q["diet"], ["FREE_FROM_GLUTEN", "FREE_FROM_LACTOSE"])
        # the stricter of two related codes wins, generalised from the lactose special case
        self.assertEqual(nlu.parse_rules("low lactose or lactose-free milk", TODAY)["diet"], ["FREE_FROM_LACTOSE"])
        self.assertEqual(nlu.parse_rules("vegan and vegetarian dishes", TODAY)["diet"], ["VEGAN"])

    def test_grounded(self):
        self.assertTrue(nlu.grounded("A thousand  liters.", "we need a thousand liters of milk"))
        self.assertFalse(nlu.grounded("2000 l", "need 1000 l"))
        self.assertFalse(nlu.grounded(None, "anything"))


class LlmMerge(unittest.TestCase):
    """interpret() with a canned Claude envelope: checks the cross-check and grounding rules."""

    def run_with(self, text, env):
        with mock.patch.object(nlu, "parse_llm", return_value=env):
            return nlu.interpret(text, TODAY, use_llm=True)

    def test_invented_quantity_is_discarded(self):
        q, tr = self.run_with("We need lactose-free milk for Monday.", _envelope(order=_order(1)))
        self.assertIsNone(q["quantity"])
        self.assertIn("quantity missing", tr["blocking"])

    def test_llm_only_value_with_quote_needs_confirmation(self):
        text = "We need 1k liters of lactose-free milk by Monday"  # the rule parser cannot read '1k'
        q, tr = self.run_with(text, _envelope(order=_order(1000, evidence={"quantity": "1k liters",
                                                                            "unit": "liters"})))
        self.assertEqual(q["quantity"], 1000)
        self.assertTrue(any("read only by Claude" in i and "quantity" in i for i in tr["issues"]))

    def test_counted_product_name_can_supply_missing_quote(self):
        order = _order(18, "PCS", None, (),
                       {"product_type": "", "quantity": "18", "unit": "chicken eggs"})
        order.update(product_type="other", product={"name_en": "chicken eggs"})
        text = "Please find 18 chicken eggs."
        q, tr = self.run_with(text, _envelope(order=order))
        self.assertEqual((q["product_type"], q["quantity"], q["unit"]), ("other", 18, "PCS"))
        self.assertFalse(tr["blocking"])
        order["product"]["name_en"] = "quail eggs"
        q, tr = self.run_with(text, _envelope(order=order))
        self.assertIsNone(q["product_type"])
        self.assertTrue(tr["blocking"])

    def test_disagreement_is_not_resolved_silently(self):
        _, tr = self.run_with("1000 l lactose-free milk by Monday", _envelope(order=_order(100)))
        self.assertTrue(any("interpretation mismatch on 'quantity'" in i for i in tr["issues"]))

    def test_headcount_estimates_an_open_quantity_for_a_named_product(self):
        order = _order(None, None, None, (), {"product_type": "apples"})
        order.update(product_type="other", product={"name_en": "apples"})
        env = _envelope(order=order, portions=2, portions_quote="2 people",
                        serving_estimate={"per_person": 0.15, "unit": "KG"})
        text = "I want to order some apples for this weekend for 2 people"
        q, tr = self.run_with(text, env)
        self.assertEqual((q["quantity"], q["unit"], q["deliver_by"]), (0.3, "KG", "2026-11-14"))
        self.assertFalse(tr["blocking"])
        self.assertEqual(tr["estimated_quantity"], {"portions": 2, "per_person": 0.15})
        self.assertEqual(engine.guardrails(_plan("KG"), q, _policy(), tr["issues"])[0], "NEEDS_HUMAN_APPROVAL")

        q, tr = self.run_with(text.replace("2 people", "two people"), dict(env, portions_quote="two people"))
        self.assertEqual(q["quantity"], 0.3)
        q, tr = self.run_with("I need apples for the weekend", env)
        self.assertIn("quantity missing", tr["blocking"])
        q, tr = self.run_with("I need 2 cartons of apples for 2 people", env)
        self.assertIn("without a pack size", tr["blocking"][0])
        q, tr = self.run_with("I need 1 kg of apples for 2 people",
                              dict(env, order=dict(order, quantity=1, unit="KG")))
        self.assertEqual((q["quantity"], q["unit"]), (1, "KG"))
        self.assertNotIn("estimated_quantity", tr)

    def test_recommend_intent(self):
        _, tr = self.run_with(RHUBARB, _envelope("recommend", culinary_goal="hot rhubarb mains"))
        self.assertEqual((tr["intent"], tr["blocking"]), ("recommend", []))

    def test_recommend_intent_conflicting_with_a_complete_order(self):
        _, tr = self.run_with(EXACT, _envelope("recommend", culinary_goal="x"))
        self.assertEqual(tr["intent"], "order")
        self.assertTrue(any("intent mismatch" in i for i in tr["issues"]))


@unittest.skipUnless(HAS_DATA, "catalogue data missing: run python3 fetch_data.py")
class Catalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = catalog.load_index()
        cls.by_id = {p["id"]: p for p in cls.idx}

    def test_diet_evidence_on_real_products(self):
        cases = [  # (product, diet, status, strength, text fragment)
            ("464306", "FREE_FROM_LACTOSE", "pass", 1.0, "GS1 claim FREE_FROM_LACTOSE"),
            ("460050", "FREE_FROM_LACTOSE", "pass", 0.5, "CONFLICT"),         # claim vs LACS 1.0 g
            ("465346", "FREE_FROM_LACTOSE", "pass", 0.5, "CONTAINS ML"),      # claim vs allergen list
            ("461675", "FREE_FROM_LACTOSE", "pass", 0.5, "LOW_LACTOSE only"),  # LACS 0 vs low-lactose claim
            ("463047", "FREE_FROM_LACTOSE", "near_miss", 0.0, "NOT lactose-free"),
            ("475534", "FREE_FROM_LACTOSE", "near_miss", 0.0, "hyla"),
            ("463047", "LOW_LACTOSE", "pass", 1.0, "LOW_LACTOSE"),
            ("464306", "LOW_LACTOSE", "pass", 1.0, "FREE_FROM_LACTOSE"),      # stricter claim satisfies it
            # generated rules (unreviewed), with the policy proposals in diet_overrides.json
            ("460000", "FREE_FROM_GLUTEN", "pass", 1.0, "free from AW"),      # claim + nonAllergen entry
            ("460104", "FREE_FROM_GLUTEN", "pass", 0.5, "CONTAINS AW"),       # gluten-free buns listing AW
            ("460218", "FREE_FROM_GLUTEN", "pass", 0.5, "MAY CONTAIN AW"),    # MAY_CONTAIN counts for gluten
            ("460000", "DAIRY_FREE", "pass", 1.0, "free from AM"),            # no claim exists: nonAllergen only
            ("464306", "DAIRY_FREE", "contradicted", 0.0, "CONTAINS AM"),  # lactose-free milk is still milk
            ("460001", "VEGETARIAN", "pass", 1.0, "GS1 claim VEGAN"),         # vegan satisfies vegetarian
            ("460095", "HIGH_FIBRE", "pass", 1.0, "GS1 claim HIGH_FIBRE"),    # low value does not contradict
        ]
        for pid, diet, status, strength, frag in cases:
            with self.subTest(product=pid, diet=diet):
                st, ev, s = catalog.diet_evidence(self.by_id[pid], diet)
                self.assertEqual((st, s), (status, strength))
                self.assertIn(frag, ev)

    def test_retrieval_counts(self):
        q = nlu.parse_rules("1000 l lactose-free milk by Monday", TODAY)
        self.assertEqual(len(catalog.retrieve(q, self.idx)[0]), 27)
        q = nlu.parse_rules("1000 l low lactose milk by Monday", TODAY)
        cands = [c["id"] for c in catalog.retrieve(q, self.idx)[0]]
        self.assertEqual(len(cands), 29)
        self.assertTrue({"463047", "469541"} <= set(cands))

    def test_search_products(self):
        ids = [c["id"] for c in catalog.search_products(["raparperi"], ["rhubarb"], self.idx)]
        self.assertEqual(ids[:2], ["460519", "460404"])  # the two fresh rhubarbs rank first
        pork = [c["id"] for c in catalog.search_products([], ["pork"], self.idx, limit=50)]
        self.assertNotIn("460060", pork)  # PORKKANA (carrot) must not match "pork"


class LiveRules(unittest.TestCase):
    """engine troubleshooting rules on canned API responses."""
    NOW = "2026-11-14T15:00:00+0200"

    def quote(self, sched, closing="2026-11-15T17:00:00+0200", logistic="Stock"):
        prices = {"products": [{"productId": "1", "catalogPrice": {"value": 3.63, "valueWithTax": 4.17},
                                "comparisonPrice": {"value": 3.63}, "comparisonUnit": "KG"}]}
        avail = {"timeNow": self.NOW, "products": [{"productId": "1", "salesBlock": False, "closingTime": closing,
                                                    "logisticModel": logistic, "scheduleLines": sched}]}
        return engine.quote_products(["1"], prices, avail, None, {})[0]["1"]

    def test_orderable(self):
        q = self.quote([{"availableAmount": 9999999999, "availableDate": "2026-11-16", "salesUnit": "KG"}])
        self.assertEqual((q["orderable"], q["stock"], q["comparison_unit"]), (True, "ample stock", "KG"))

    def test_cutoff_passed(self):
        q = self.quote([{"availableAmount": 999999, "availableDate": "2026-11-16", "salesUnit": "KG"}],
                       closing="2026-11-12T14:00:00+0200", logistic="Terminal")
        self.assertEqual(q["code"], "CUTOFF_PASSED")

    def test_not_deliverable(self):
        self.assertEqual(self.quote([])["code"], "NOT_DELIVERABLE")


class Dates(unittest.TestCase):
    def test_weekend_on_sunday_means_upcoming_saturday(self):
        sunday = dt.date(2026, 11, 15)
        self.assertEqual(nlu.parse_date("for this weekend", sunday), ("2026-11-21", []))
        self.assertEqual(nlu.parse_date("for next weekend", sunday), ("2026-11-21", []))
        monday = sunday + dt.timedelta(days=1)
        self.assertEqual(nlu.parse_date("for next weekend", monday), ("2026-11-28", []))

    def test_past_dates_stop_orders_and_menu_requests_before_quoting(self):
        with mock.patch.object(agent, "today", return_value=TODAY), \
                mock.patch.object(agent, "AimoClient") as client:
            order = agent.run_request("I need 10 litres of milk by yesterday", _policy(),
                                      use_llm=False, persist=False)
            with mock.patch.object(nlu, "parse_llm", return_value=_envelope("recommend")):
                menu = agent.run_request("Can you suggest a menu for 2 people by 10 November 2026?",
                                         _policy(), persist=False)
        client.assert_not_called()
        for result, date in ((order, "2026-11-13"), (menu, "2026-11-10")):
            self.assertEqual(result["status"], "ESCALATED_NO_OPTION")
            self.assertEqual(result["interpretation"]["past_delivery_date"], date)
            self.assertEqual(result["next_steps"], [nlu.PAST_DELIVERY_MESSAGE])
            self.assertEqual(result["options"], [])

    def test_dates(self):
        cases = {  # request -> (ISO date or None, fragment of an issue or None)
            "by Monday": ("2026-11-16", None),
            "by last Monday": ("2026-11-09", None),
            "by tomorrow": ("2026-11-15", None),
            "for this weekend": ("2026-11-14", None),
            "for last weekend": ("2026-11-07", None),
            "the day after tomorrow": ("2026-11-16", None),
            "in 3 days": ("2026-11-17", None),
            "by 16.11.": ("2026-11-16", None),
            "by 16.11.2026": ("2026-11-16", None),
            "by 16/11": ("2026-11-16", None),
            "by 20 November": ("2026-11-20", None),
            "November 20th, 2026": ("2026-11-20", None),
            "18. marraskuuta": ("2026-11-18", None),
            "på fredag": ("2026-11-20", None),
            "by 5.1.": ("2027-01-05", None),          # January is next year, not the past
            "by 10.11.": ("2026-11-10", "in the past"),
            "next week": (None, "not a single date"),
            "16.11 mennessä": (None, "could not be read"),
            "by 16.11. or 18.11.": (None, "several delivery dates"),
            "1.5 l bottles": (None, None),            # a volume, not a date
        }
        for text, (iso, issue) in cases.items():
            with self.subTest(text=text):
                got, issues = nlu.parse_date(text, TODAY)
                self.assertEqual(got, iso)
                if issue:
                    self.assertTrue(any(issue in i for i in issues), issues)
                else:
                    self.assertEqual(issues, [])


class Quantities(unittest.TestCase):
    def test_quantities(self):
        cases = {  # request -> (quantity, unit)
            "I need 1,000 litres of milk": (1000.0, "L"), "3000 l": (3000.0, "L"), "1 000 l": (1000.0, "L"),
            "2 dl cream": (0.2, "L"), "500 g butter": (0.5, "KG"), "50 kg": (50.0, "KG"), "20 kpl": (20.0, "PCS"),
            "50 x 20 l": (1000.0, "L"), "a thousand liters": (1000.0, "L"),
            "tuhat litraa": (1000.0, "L"), "1000 litraa": (1000.0, "L"), "two thousand five hundred litres": (2500.0, "L"),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                q, u, issues, blocking = nlu.parse_quantity(text)
                self.assertEqual((q, u), expected)
                self.assertEqual((issues, blocking), ([], []))

    def test_what_the_rule_parser_will_not_guess(self):
        self.assertIn("more than one quantity", nlu.parse_quantity("1000 l milk and 50 kg butter")[3][0])
        self.assertIn("without a pack size", nlu.parse_quantity("20 cartons of milk")[3][0])
        self.assertIn("pack-size wish", nlu.parse_quantity("1000 l milk in 20 l boxes")[2][0])
        q, u, issues, _ = nlu.parse_quantity("50 boxes of 20 L")  # the amount is clear, the box size is a wish
        self.assertEqual((q, u), (1000.0, "L"))
        self.assertIn("pack-size wish", issues[0])
        self.assertIn("vague quantity", nlu.parse_quantity("a few hundred liters")[2][0])

    def test_interpret_blocks_a_second_item(self):
        _, tr = nlu.interpret("1000 l lactose-free milk and 50 kg butter by Monday", TODAY, use_llm=False)
        self.assertTrue(any("more than one quantity" in b for b in tr["blocking"]))


@unittest.skipUnless(HAS_DATA, "catalogue data missing: run python3 fetch_data.py")
class Brands(unittest.TestCase):
    def test_index_flags_common_words(self):
        b = catalog.brand_index()
        self.assertGreater(len(b), 1400)
        self.assertFalse(any(b[k]["ambiguous"] for k in ("valio", "arla", "fazer", "myllyn paras")))
        self.assertTrue(all(b[k]["ambiguous"] for k in ("fresh", "classic", "juusto", "hp")))

    def test_parse_brands(self):
        cases = {
            "lactose-free Arla milk": ("Arla", []),
            "Myllyn Paras flour": ("Myllyn Paras", []),
            "fresh rhubarb": (None, []),                 # 'Fresh' is a brand, but here it is a word
            "milk, no Arla please": (None, ["Arla"]),
            "maitoa, ei Arlaa": (None, ["Arla"]),        # Finnish partitive
            "mjölk utan Arla": (None, ["Arla"]),
        }
        for text, (prefer, avoid) in cases.items():
            with self.subTest(text=text):
                p, a, _ = nlu.parse_brands(text)
                self.assertEqual((p, a), (prefer, avoid))

    def test_resolve_brand(self):
        self.assertEqual(catalog.resolve_brand("arla"), ("Arla", "exact"))
        self.assertEqual(catalog.resolve_brand("valoi"), ("Valio", "fuzzy"))
        self.assertEqual(catalog.resolve_brand("No Such Brand Oy")[1], "unknown")

    def test_avoided_brand_is_excluded(self):
        q = nlu.parse_rules("1000 l lactose-free milk, no Valio, by Monday", TODAY)
        cands, rejected, _ = catalog.retrieve(q, catalog.load_index())
        self.assertFalse(any(c["brand"].lower().startswith("valio") for c in cands))
        self.assertTrue(any(x["code"] == "AVOIDED_BRAND" for x in rejected))


@unittest.skipUnless(HAS_DATA, "catalogue data missing: run python3 fetch_data.py")
class GenericProducts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.by_id = {p["id"]: p for p in catalog.load_index()}

    def test_measures(self):
        p = self.by_id
        self.assertEqual(catalog.amount_per_sales_unit(p["464411"], "KG")[0], 0.5)   # 'butter 500 g', sold per piece
        self.assertEqual(catalog.amount_per_sales_unit(p["460007"], "KG")[0], 2.0)   # unit table: 2 kg per box
        self.assertIsNone(catalog.amount_per_sales_unit(p["464205"], "KG")[0])       # '4x5 kg': left to the price API
        self.assertEqual(catalog.amount_per_sales_unit(p["464306"], "L")[0], 1.0)

    def test_count_whole_eggs_not_cartons(self):
        carton, case = self.by_id["467676"], self.by_id["468445"]
        self.assertEqual(catalog.amount_per_sales_unit(carton, "PCS")[0], 10)
        self.assertEqual(catalog.amount_per_sales_unit(case, "PCS")[0], 120)
        unknown = dict(carton, name_en="Barn eggs", name_fi="Kananmunat")
        self.assertIsNone(catalog.amount_per_sales_unit(unknown, "PCS")[0])

    def test_generic_quote_excludes_variants(self):
        prods = [self.by_id[i] for i in ("464411", "472127", "469665")]
        verdicts = {"464411": {"verdict": "is_the_product", "reason": "butter"},
                    "472127": {"verdict": "variant", "reason": "clarified butter"},
                    "469665": {"verdict": "different_product", "reason": "ice cream"}}
        q = {"product_type": "other", "diet": [], "avoid_brands": [], "preferences": {}}
        cands, rejected, stats = catalog.check_candidates(prods, verdicts, q, "KG")
        self.assertEqual([c["id"] for c in cands], ["464411"])
        self.assertEqual([(x["id"], x["code"], x["stage"]) for x in rejected],
                         [("472127", "NOT_THE_PRODUCT", "llm"), ("469665", "NOT_THE_PRODUCT", "llm")])
        self.assertEqual(stats["identified"], 1)

    def test_classify_schema_only_allows_real_candidates(self):
        s = nlu.build_classify_schema([{"id": "464411"}, {"id": "469665"}])
        item = s["properties"]["verdicts"]["items"]["properties"]
        self.assertEqual(item["id"]["enum"], ["464411", "469665"])
        self.assertEqual(item["verdict"]["enum"], nlu.VERDICTS)

    def test_skipped_candidate_counts_as_not_identified(self):
        cands = [{"id": i, "name_en": "x", "name_fi": "x", "salesUnit": "ST"} for i in ("464411", "469665")]
        out = {"verdicts": [{"id": "464411", "verdict": "is_the_product", "reason": "butter"}]}
        with mock.patch.object(nlu, "_call_structured", return_value=out):
            v = nlu.classify_candidates({"name_en": "butter"}, cands, [])
        self.assertEqual(v["464411"]["verdict"], "is_the_product")
        self.assertEqual(v["469665"]["verdict"], "different_product")


class GenericGuardrails(unittest.TestCase):
    def test_bulk_discount_does_not_reward_overbuying(self):
        small = {"amount": 20, "need": 18, "packs": 2, "total": 4.02, "eur_per_unit": 4.02 / 20}
        bulk = {"amount": 90, "need": 18, "packs": 1, "total": 11.87, "eur_per_unit": 11.87 / 90}
        for p in (small, bulk):
            p.update(availability=0.99, preference=1, evidence=1, coverage=1, co2_kg=None)
        weights = engine.effective_weights(agent.DEFAULT_PREFS, {})
        self.assertIs(engine.rank([bulk, small], weights)[0], small)

    def test_unreviewed_product_type_and_missing_ceiling_need_a_human(self):
        q = {"product_type": "other", "preferences": {}, "diet": []}
        decision, checks = engine.guardrails(_plan("KG", 6.6), q, _policy(), [], (), "butter")
        failed = [c for ok, c in checks if not ok]
        self.assertEqual(decision, "NEEDS_HUMAN_APPROVAL")
        self.assertTrue(any("price ceiling" in c for c in failed))
        self.assertTrue(any("product type reviewed by a person" in c for c in failed))

    def test_milk_texts_unchanged(self):
        decision, checks = engine.guardrails(_plan(), {"product_type": "milk", "preferences": {}, "diet": []}, _policy(), [])
        self.assertEqual(decision, "AUTO_ORDER")
        self.assertIn("price EUR 1.00/L within ceiling EUR 2.50/L", [c for _, c in checks])

    def test_name_weight_corrected_by_price_api(self):
        cand = {"id": "1", "name": "butter 6 g", "measure": "KG", "per_unit": 0.006,
                "per_unit_method": "pack weight from name x pack size", "kg_per_measure": 1.0, "pref_match": 1.0}
        prices = {"products": [{"productId": "1", "catalogPrice": {"value": 28.56, "valueWithTax": 32.0},
                                "comparisonPrice": {"value": 11.9}, "comparisonUnit": "KG"}]}
        avail = {"timeNow": "2026-11-14T15:00:00+0200", "products": [{"productId": "1", "salesBlock": False,
                 "closingTime": "2026-11-15T17:00:00+0200", "logisticModel": "Stock",
                 "scheduleLines": [{"availableAmount": 9999999999, "availableDate": "2026-11-16", "salesUnit": "KI"}]}]}
        offers, _, _ = engine.live_check([cand], prices, avail, {"deliver_by": None}, {}, {})
        self.assertAlmostEqual(offers[0]["per_unit"], 2.4)  # 28.56 / 11.90, not 0.006
        self.assertTrue(offers[0]["flags"][0].startswith("UNIT_MISMATCH"))


class Reviews(unittest.TestCase):
    def test_reviewed_rules_name_a_reviewer_and_date(self):
        for code, r in catalog.DIET_RULES.items():
            if r["reviewed"]:
                with self.subTest(rule=code):
                    self.assertTrue(r["reviewed_by"] and r["reviewed_at"])
        for name, t in catalog.PRODUCT_TYPES.items():
            if t.get("reviewed"):
                with self.subTest(product_type=name):
                    self.assertTrue(t.get("reviewed_by") and t.get("reviewed_at"))

    @unittest.skipUnless(HAS_DATA, "catalogue data missing")
    def test_review_sheet_lists_the_products(self):
        import review_diet_rule
        text = review_diet_rule.sheet("FREE_FROM_GLUTEN", catalog.load_index())
        self.assertIn("NOT reviewed", text)
        self.assertIn("460104", text)  # the gluten-free buns that list AW appear among the conflicts
        self.assertIn("## 8. Checklist", text)

    @unittest.skipUnless(HAS_DATA, "catalogue data missing")
    def test_signed_reviews_still_match_the_data(self):
        """A signature covers the sheet the reviewer read. If the data or the rule changed since,
        the sheet changes and the rule must be reviewed again (review_*.py --sign)."""
        import review_diet_rule
        import review_product_type
        idx = catalog.load_index()
        for code, r in catalog.DIET_RULES.items():
            if r.get("review_sheet_sha256"):
                with self.subTest(rule=code):
                    self.assertEqual(review_diet_rule.sheet_hash(review_diet_rule.sheet(code, idx)),
                                     r["review_sheet_sha256"], f"{code} changed since its review: review it again")
        for key, t in catalog.PRODUCT_TYPES.items():
            if t.get("review_sheet_sha256"):
                with self.subTest(product_type=key):
                    self.assertEqual(review_diet_rule.sheet_hash(review_product_type.sheet(key, idx)),
                                     t["review_sheet_sha256"], f"{key} changed since its review: review it again")

    @unittest.skipUnless(HAS_DATA, "catalogue data missing")
    def test_product_type_sheet_shows_conversions_exclusions_and_misses(self):
        import review_product_type
        text = review_product_type.sheet("milk", catalog.load_index())
        self.assertIn("| **accepted as milk** | 75 |", text)
        self.assertIn("469209", text)        # name says 20 l, unit table 10 l: the reviewer must see the conversion
        self.assertIn("`piimä`", text)       # sour milk excluded with its reason
        self.assertIn("## 6. Possible misses", text)

    def test_signing_edits_only_the_review_fields_and_needs_the_current_sheet(self):
        import tempfile
        import review_diet_rule as R
        tmp = tempfile.mkdtemp()
        src = os.path.join(catalog.RULES_DIR, "diet_overrides.json")
        with open(src, encoding="utf-8") as f:
            original = f.read()
        path, sheet_path = os.path.join(tmp, "overrides.json"), os.path.join(tmp, "sheet.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(original)
        text = "# sheet\n\nStatus: **NOT reviewed**.\n\n| 1 | product |\n"
        with open(sheet_path, "w", encoding="utf-8") as f:
            f.write(text)
        with self.assertRaises(SystemExit):  # the reviewer read an older sheet
            R.sign(path, "codes", "FREE_FROM_FAT", "Test Person", text + "| 2 | new product |\n", sheet_path, "cmd")
        with mock.patch("builtins.print"):
            R.sign(path, "codes", "FREE_FROM_FAT", "Test Person", text, sheet_path, "cmd")
        with open(path, encoding="utf-8") as f:
            signed = f.read()
        after, before = json.loads(signed), json.loads(original)
        entry = after["codes"].pop("FREE_FROM_FAT")
        self.assertEqual((entry["reviewed"], entry["reviewed_by"], entry["review_sheet_sha256"]),
                         (True, "Test Person", R.sheet_hash(text)))
        before["codes"].pop("FREE_FROM_FAT")
        self.assertEqual(after, before)                                   # nothing else changed
        self.assertEqual(len(signed.splitlines()), len(original.splitlines()))  # and the layout is kept
        with open(sheet_path, encoding="utf-8") as f:
            self.assertIn("Status: reviewed by Test Person", f.read())


class HardSoftConstraints(unittest.TestCase):
    """Scenario 2 (hard constraints are explained, never relaxed) and 3 (soft ones relaxed with consent)."""

    def test_must_have_turns_a_preference_into_a_requirement(self):
        self.assertEqual(nlu.parse_rules("1000 l lactose-free milk, only Arla, by Monday", TODAY)["must_have"], ["brand"])
        self.assertEqual(nlu.parse_rules("1000 l lactose-free Arla milk by Monday", TODAY)["must_have"], [])
        self.assertEqual(nlu.parse_rules("500 l organic milk only", TODAY)["must_have"], ["organic"])
        self.assertEqual(nlu.parse_rules("vain luomu maitoa 100 l", TODAY)["must_have"], ["organic"])

    def test_allergy_phrasing_maps_to_allergen_rules(self):
        self.assertIn("FISH_FREE", nlu.parse_rules("milk for a guest with a fish allergy", TODAY)["diet"])
        self.assertIn("SESAME_FREE", nlu.parse_rules("seesamiton maito 100 l", TODAY)["diet"])
        self.assertEqual(nlu.requirement_hints("1000 l milk, kiwi allergy"), ["kiwi allergy"])  # no rule: reported

    def test_constraint_strengths(self):
        q = nlu.parse_rules("1000 l lactose-free milk, only Arla, no Valio, by Monday", TODAY)
        got = {c["name"]: c["strength"] for c in agent._constraints(q, {"FREE_FROM_LACTOSE": "lactose-free"}, "milk")}
        self.assertEqual(got["lactose-free"], "hard")
        self.assertEqual(got["brand: Arla (only)"], "negotiable")
        self.assertEqual(got["not Valio"], "negotiable")
        self.assertEqual(got["deliver by 2026-11-16"], "negotiable")

    def test_relaxations_offer_soft_changes_only(self):
        q = {"deliver_by": "2026-11-15", "avoid_brands": ["Valio"], "diet": ["FREE_FROM_LACTOSE"], "preferences": {}}
        rejected = [{"code": "TOO_LATE", "reason": "earliest delivery 2026-11-16 is after the deadline 2026-11-15"},
                    {"code": "AVOIDED_BRAND", "reason": "brand Valio is excluded", "constraint": "not Valio"},
                    {"code": "DIET_FAIL", "reason": "no lactose-free evidence", "constraint": "FREE_FROM_LACTOSE"}]
        trace = {"parser": "rules", "blocking": []}
        rels = agent._relaxations(q, trace, {"candidates": 3}, rejected, None, "milk", {}, False)
        self.assertEqual([(r["kind"], r["change"]) for r in rels],
                         [("date", {"deliver_by": "2026-11-16"}), ("brand", {"avoid_brands": []})])
        self.assertFalse(any("diet" in r["change"] for r in rels))  # hard requirements are never offered

    def test_relax_order_reruns_with_the_change_and_links_both(self):
        import tempfile
        tmp = tempfile.mkdtemp()
        order = {"order_id": "ORD-1", "status": "ESCALATED_NO_OPTION", "request": "x", "policy": _policy(),
                 "query": {"deliver_by": "2026-11-15"}, "interpretation": {"notes": [], "blocking": []},
                 "relaxations": [{"id": "R1", "kind": "date", "text": "Deliver on 2026-11-16", "evidence": "26 products",
                                  "change": {"deliver_by": "2026-11-16"}}]}
        seen = {}

        def fake_run(text, query, trace, policy, client, persist, relaxation, relaxed_from):
            seen.update(query=query, relaxation=relaxation, relaxed_from=relaxed_from, runs=seen.get("runs", 0) + 1)
            r = {"order_id": "ORD-2", "status": "PLACED_AUTONOMOUSLY"}
            agent._save(agent.ORDERS, dict(agent._load(agent.ORDERS, {}), **{"ORD-2": r}))
            return r

        with mock.patch.object(agent, "OUT", tmp), mock.patch.object(agent, "ORDERS", os.path.join(tmp, "orders.json")), \
                mock.patch.object(agent, "_run_order", side_effect=fake_run), \
                mock.patch.dict(os.environ, {"AIMO_API_KEY": "test"}):
            agent._save(agent.ORDERS, {"ORD-1": order})
            r = agent.relax_order("ORD-1", "R1", "Head chef")
            parent = agent._load(agent.ORDERS, {})["ORD-1"]
            again = agent.relax_order("ORD-1", "R1", "Head chef")  # double click: the same re-run, not a second one
            with self.assertRaises(ValueError):
                agent.relax_order("ORD-1", "R2", "Head chef")      # a different change after the first: refused
        self.assertEqual((again["order_id"], again["idempotent_replay"]["kind"], seen["runs"]), ("ORD-2", "relaxation", 1))
        self.assertEqual(r["order_id"], "ORD-2")
        self.assertEqual(seen["query"]["deliver_by"], "2026-11-16")
        self.assertEqual((seen["relaxed_from"], seen["relaxation"]["accepted_by"]), ("ORD-1", "Head chef"))
        self.assertEqual(parent["superseded_by"], "ORD-2")

    def test_later_deadline_keeps_unconfirmed_supplier_items_for_a_person(self):
        cand = {"id": "1", "name": "x", "measure": "L", "per_unit": 1.0, "kg_per_measure": 1.0, "pref_match": 1.0}
        prices = {"products": [{"productId": "1", "catalogPrice": {"value": 1.0, "valueWithTax": 1.2}}]}
        avail = {"timeNow": "2026-11-14T15:00:00+0200", "products": [{"productId": "1", "salesBlock": False,
                 "closingTime": "2026-11-12T14:00:00+0200", "logisticModel": "Terminal",
                 "scheduleLines": [{"availableAmount": 999999, "availableDate": "2026-11-16", "salesUnit": "ST"}]}]}
        offers, rej, _ = engine.live_check([cand], prices, avail, {"deliver_by": "2026-11-16"}, {}, {})
        self.assertEqual((len(offers), rej[0]["code"]), (0, "CUTOFF_PASSED"))   # needed on the earliest date: rejected
        offers, rej, _ = engine.live_check([cand], prices, avail, {"deliver_by": "2026-11-20"}, {}, {})
        self.assertFalse(offers[0]["delivery_confirmed"])                       # needed later: kept, flagged
        self.assertEqual([a["code"] for a in offers[0]["anomalies"]], ["STALE_CUTOFF"])
        line = {"_o": dict(offers[0], reliability=1.0, qty_confirmed=True), "name": "x"}
        plan = dict(_plan(), lines=[line])
        _, checks = engine.guardrails(plan, {"product_type": "milk", "preferences": {}, "diet": []}, _policy(), [])
        self.assertTrue(any(not ok and "live data consistent with the API contract" in c and "STALE_CUTOFF" in c
                            for ok, c in checks))


@unittest.skipUnless(HAS_DATA, "catalogue data missing: run python3 fetch_data.py")
class HardConstraintsOnData(unittest.TestCase):
    def test_vegan_milk_is_explained_not_relaxed(self):
        with mock.patch.dict(os.environ, {"AIMO_API_KEY": "test"}):  # nothing reaches the live API
            r = agent.run_request("1000 l vegan milk by Monday", _policy(), use_llm=False, persist=False)
        self.assertEqual(r["status"], "ESCALATED_NO_OPTION")
        self.assertIn("Hard requirement 'vegan'", r["next_steps"][0])
        self.assertIn("contradicted by the allergen data", r["next_steps"][0])
        self.assertEqual(r["relaxations"], [])

    def test_diet_screen_separates_contradicted_from_unverified(self):
        by_id = {p["id"]: p for p in catalog.load_index()}
        potato = catalog.search_products(["peruna"], ["potato"], limit=1)[0]
        cands = [{"id": "460001", "name_en": "macaroni"}, {"id": "464306", "name_en": "milk"},
                 {"id": potato["id"], "name_en": potato["name_en"]}]
        kept, removed, unverified = agent._diet_screen(cands, ["VEGAN"], {"VEGAN": "vegan"}, by_id)
        self.assertEqual([c["id"] for c in kept], ["460001"])          # GS1 vegan claim
        self.assertEqual([c["id"] for c in removed], ["464306"])       # milk: CONTAINS AM contradicts vegan
        self.assertEqual([c["id"] for c in unverified], [potato["id"]])  # no evidence either way: a person checks



@unittest.skipUnless(HAS_DATA, "catalogue data missing: run python3 fetch_data.py")
class HumanConfirmation(unittest.TestCase):
    """policy.json require_human_confirmation (default on): the agent never places an order on its own."""
    def _run(self, policy):
        empty = {"products": [], "timeNow": "2026-11-15T05:00:00+0200"}
        with mock.patch.dict(os.environ, {"AIMO_API_KEY": "test"}), \
             mock.patch.object(aimo_api.AimoClient, "price", lambda self, ids, d: empty), \
             mock.patch.object(aimo_api.AimoClient, "availability", lambda self, ids, d: empty), \
             mock.patch.object(engine, "guardrails", lambda *a, **k: ("AUTO_ORDER", [(True, "every rule passes")])):
            return agent.run_request("I need 1,000 litres of lactose-free milk by Monday", policy, use_llm=False, persist=False)

    def test_every_rule_passing_still_waits_for_a_person(self):
        r = self._run(_policy())
        self.assertEqual(r["status"], "PENDING_APPROVAL")
        self.assertEqual(r["requires_confirmation"], "policy.json · require_human_confirmation")
        policy = {k: v for k, v in _policy().items() if k != "require_human_confirmation"}
        self.assertEqual(self._run(policy)["status"], "PENDING_APPROVAL")   # absent = on

    def test_autonomy_only_when_switched_off(self):
        self.assertEqual(self._run(dict(_policy(), require_human_confirmation=False))["status"], "PLACED_AUTONOMOUSLY")

if __name__ == "__main__":
    unittest.main()


class _FakeAimo:
    """Price + availability responses for one product, as the mock API returns them."""

    def __init__(self, price=1.01, amount=9999999999, closing="2026-11-15T17:00:00+0200", adate="2026-11-16",
                 now="2026-11-14T15:00:00+0200", fail=False):
        self.calls, self.kw = [], dict(price=price, amount=amount, closing=closing, adate=adate, now=now)
        self.fail = fail

    def price(self, ids, date):
        if self.fail:
            raise agent.AimoApiError("/api/price unreachable")
        self.calls.append({"endpoint": "POST /api/price", "at": "t", "requestedDate": date, "products": len(ids),
                           "response_sha256": "0" * 64})
        return {"products": [{"productId": i, "catalogPrice": {"value": self.kw["price"],
                                                               "valueWithTax": self.kw["price"] * 1.14}} for i in ids]}

    def availability(self, ids, date):
        k = self.kw
        return {"timeNow": k["now"], "products": [{"productId": i, "salesBlock": False, "closingTime": k["closing"],
                "logisticModel": "Stock", "scheduleLines": [{"availableAmount": k["amount"], "availableDate": k["adate"],
                                                             "salesUnit": "ST"}]} for i in ids]}


def _pending(order_id="ORD-P"):
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return {"order_id": order_id, "status": "PENDING_APPROVAL", "request": "x", "created_at": now, "policy": _policy(),
            "query": {"product_type": "milk", "quantity": 1000, "unit": "L", "deliver_by": "2026-11-16"},
            "interpretation": {"notes": [], "blocking": [], "issues": []},
            "options": [{"option": 1, "total_eur": 1010.0, "utilities": {}, "lines": [
                {"productId": "464306", "name": "Valio milk", "quantity": 1000, "salesUnit": "ST", "unitPrice": 1.01,
                 "availableDate": "2026-11-16", "flags": []}]}]}


class CommitControls(unittest.TestCase):
    """P3: approval re-checks the quote; repeated requests and decisions never order twice."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "audit"))
        self.patches = [mock.patch.object(agent, "OUT", self.tmp),
                        mock.patch.object(agent, "ORDERS", os.path.join(self.tmp, "orders.json")),
                        mock.patch.object(agent, "PREFS", os.path.join(self.tmp, "prefs.json")),
                        mock.patch.object(agent, "today", return_value=TODAY),
                        mock.patch.dict(os.environ, {"AIMO_API_KEY": "test"})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_revalidate_finds_what_changed(self):
        line = _pending()["options"][0]["lines"][0]
        deadline = dt.date(2026, 11, 16)
        ok = _FakeAimo()
        self.assertEqual(engine.revalidate([line], ok.price(["464306"], "d"), ok.availability(["464306"], "d"),
                                           deadline)[0], [])
        for fake, expect in [(_FakeAimo(price=1.09), "price rose from EUR 1.01 to EUR 1.09"),
                             (_FakeAimo(amount=400), "stock fell to 400"),
                             (_FakeAimo(closing="2026-11-14T14:00:00+0200"), "cut-off"),
                             (_FakeAimo(adate="2026-11-17"), "after the deadline")]:
            changes, _, _ = engine.revalidate([line], fake.price(["464306"], "d"), fake.availability(["464306"], "d"),
                                              deadline)
            self.assertTrue(changes and expect in changes[0], (expect, changes))
        cheaper = _FakeAimo(price=0.95)  # a lower price is not a change; it is committed at the fresh price
        changes, fresh, _ = engine.revalidate([line], cheaper.price(["464306"], "d"),
                                              cheaper.availability(["464306"], "d"), deadline)
        self.assertEqual((changes, fresh[0]["lineTotal"]), ([], 950.0))

    def test_approval_commits_only_after_a_fresh_check_and_is_idempotent(self):
        agent._save(agent.ORDERS, {"ORD-P": _pending()})
        fake = _FakeAimo()
        with mock.patch.object(agent, "AimoClient", lambda customer: fake):
            o = agent.decide_order("ORD-P", True, 1, "Head chef")
            again = agent.decide_order("ORD-P", True, 1, "Head chef")  # double click
            with self.assertRaises(ValueError):
                agent.decide_order("ORD-P", False, 1, "Head chef")     # a different decision afterwards
        self.assertEqual((o["status"], o["revalidation"]["ok"], o["revalidation"]["committed_total_eur"]),
                         ("APPROVED_BY_HUMAN", True, 1010.0))
        self.assertEqual((again["status"], again["idempotent_replay"]["kind"]), ("APPROVED_BY_HUMAN", "decision"))
        self.assertEqual(len(fake.calls), 1)  # the repeat did not query the API again

    def test_stale_quote_is_not_placed_and_the_request_is_replanned(self):
        agent._save(agent.ORDERS, {"ORD-P": _pending()})
        seen = {}

        def fake_run(text, query, trace, policy, client, persist, stale=None, **kw):
            seen.update(stale=stale, status_during=agent._load(agent.ORDERS, {})["ORD-P"]["status"])
            return {"order_id": "ORD-N", "status": "PENDING_APPROVAL", "stale_quote": stale}

        with mock.patch.object(agent, "AimoClient", lambda customer: _FakeAimo(price=1.20)), \
                mock.patch.object(agent, "_run_order", side_effect=fake_run):
            r = agent.decide_order("ORD-P", True, 1, "Head chef")
        old = agent._load(agent.ORDERS, {})["ORD-P"]
        self.assertEqual((r["order_id"], old["status"], old["superseded_by"]), ("ORD-N", "STALE_QUOTE", "ORD-N"))
        self.assertEqual(seen["status_during"], "STALE_QUOTE")  # not counted as an open duplicate by the re-plan
        self.assertIn("price rose", seen["stale"]["changes"][0])

    def test_approval_fails_closed_when_the_quote_cannot_be_rechecked(self):
        agent._save(agent.ORDERS, {"ORD-P": _pending()})
        with mock.patch.object(agent, "AimoClient", lambda customer: _FakeAimo(fail=True)):
            with self.assertRaises(agent.AimoApiError):
                agent.decide_order("ORD-P", True, 1, "Head chef")
        self.assertEqual(agent._load(agent.ORDERS, {})["ORD-P"]["status"], "PENDING_APPROVAL")

    def test_approval_refuses_an_expired_delivery_date(self):
        agent._save(agent.ORDERS, {"ORD-P": _pending()})
        with mock.patch.object(agent, "today", return_value=dt.date(2026, 11, 17)), \
                mock.patch.object(agent, "AimoClient") as client:
            with self.assertRaisesRegex(ValueError, "delivery date has already passed"):
                agent.decide_order("ORD-P", True, 1, "Head chef")
        client.assert_not_called()

    def test_request_hash_ignores_wording_but_not_the_goods(self):
        h = lambda t: agent.request_hash(nlu.interpret(t, TODAY, use_llm=False)[0], "C")
        a = h("I need 1,000 litres of lactose-free milk by Monday")
        self.assertEqual(a, h("1000 l laktoositon maito by Monday, cheapest please"))
        self.assertNotEqual(a, h("I need 1,200 litres of lactose-free milk by Monday"))
        self.assertNotEqual(a, h("I need 1,000 litres of lactose-free milk by Tuesday"))

    def test_repeated_request_is_answered_with_the_earlier_order(self):
        text = "I need 1,000 litres of lactose-free milk by Monday"
        policy = _policy()
        q = nlu.interpret(text, dt.date.today(), use_llm=False)[0]
        policy = dict(policy, check_duplicate_orders=True)
        prior = dict(_pending("ORD-A"), status="PLACED_AUTONOMOUSLY", request=text, next_steps=[],
                     idempotency={"request_hash": agent.request_hash(q, policy["customer_id"]), "client_key": "k1"})
        agent._save(agent.ORDERS, {"ORD-A": prior})
        with mock.patch.object(agent, "_run_order", return_value={"order_id": "ORD-B"}) as run:
            r = agent.run_request(text + ".", policy, use_llm=False)
            self.assertEqual((r["order_id"], r["idempotent_replay"]["kind"], run.call_count), ("ORD-A", "request", 0))
            self.assertEqual(agent.run_request(text, policy, use_llm=False, idempotency_key="k1")["order_id"], "ORD-A")
            with self.assertRaises(ValueError):  # the same caller key for a different request
                agent.run_request("500 l lactose-free milk by Monday", policy, use_llm=False, idempotency_key="k1")
            self.assertEqual(agent.run_request(text, policy, use_llm=False, allow_duplicate=True)["order_id"], "ORD-B")
            self.assertEqual(run.call_count, 1)

    def test_duplicate_checks_can_be_switched_off(self):
        text = "I need 1,000 litres of lactose-free milk by Monday"
        off = dict(_policy(), check_duplicate_orders=False)
        q = nlu.interpret(text, dt.date.today(), use_llm=False)[0]
        prior = dict(_pending("ORD-A"), status="PLACED_AUTONOMOUSLY", request=text, next_steps=[],
                     query={"product_type": "milk", "quantity": 1000, "unit": "L", "deliver_by": q["deliver_by"]},
                     idempotency={"request_hash": agent.request_hash(q, off["customer_id"]), "client_key": "k1"})
        agent._save(agent.ORDERS, {"ORD-A": prior})
        self.assertIsNone(agent._possible_duplicates(prior["query"], off))           # no duplicate guardrail
        with mock.patch.object(agent, "_run_order", return_value={"order_id": "ORD-B"}) as run:
            self.assertEqual(agent.run_request(text, off, use_llm=False)["order_id"], "ORD-B")  # no replay
            self.assertEqual(agent.run_request(text, off, use_llm=False, idempotency_key="k1")["order_id"], "ORD-A")
        self.assertEqual(run.call_args.kwargs.get("client_key"), None)
        self.assertFalse(agent.duplicate_checks_on(off))
        self.assertTrue(agent.duplicate_checks_on({}))                                 # default: on

    def test_same_goods_on_the_same_date_need_a_person(self):
        q = {"product_type": "milk", "quantity": 1000, "unit": "L", "deliver_by": "2026-11-16"}
        agent._save(agent.ORDERS, {"ORD-A": dict(_pending("ORD-A"), status="PLACED_AUTONOMOUSLY"),
                                   "ORD-R": dict(_pending("ORD-R"), status="REJECTED_BY_HUMAN")})
        on = dict(_policy(), check_duplicate_orders=True)
        dups = agent._possible_duplicates(q, on)
        self.assertEqual(len(dups), 1)
        self.assertTrue(dups[0].startswith("ORD-A"))
        self.assertEqual(agent._possible_duplicates(dict(q, deliver_by="2026-11-17"), on), [])
        decision, checks = engine.guardrails(_plan(), {"product_type": "milk", "preferences": {}}, on, [],
                                             duplicates=dups)
        self.assertEqual(decision, "NEEDS_HUMAN_APPROVAL")
        self.assertTrue(any(not ok and "possible duplicate: ORD-A" in c for ok, c in checks))


# --- Demo scenarios 2 and 3: multi-line orders, tiers, allergen verdicts, undo -----------------------
HOSPITAL = ("Hospital kitchen: we need 4 litres of cooking cream for tomorrow's lunch soup for 40 patients. "
            "6 of them are allergic to milk protein and gluten.")
WEDDING = ("We're catering a wedding for 100 guests on Saturday and want to serve salmon and carrot soup. "
           "Can you source the ingredients?")
# Live price / availability on 2026-11-14 19:42 (server time) for the products the tests use:
# id: (EUR per sales unit, comparison EUR per KG/L, unit, logistic model, amount, closing time); None = not deliverable
LIVE_1114 = {
    "460721": (1.82, 1.82, "KG", None, None, None),
    "460811": (1.26, 1.26, "KG", "Stock", 190, "2026-11-15T17:00:00+0200"),
    "460708": (11.18, 1.12, "KG", "Stock", 20, "2026-11-15T17:00:00+0200"),
    "462929": (5.23, 1.05, "KG", "Stock", 9999999999, "2026-11-15T17:00:00+0200"),
    "460110": (2.93, 2.93, "KG", "Stock", 9999999999, "2026-11-15T17:00:00+0200"),
    "470961": (9.78, 0.65, "KG", "Stock", 9999999999, "2026-11-15T17:00:00+0200"),
    "464249": (2.63, 2.63, "L", "Stock", 9999999999, "2026-11-15T17:00:00+0200"),
    "463797": (3.31, 3.31, "L", "Stock", 114, "2026-11-15T17:00:00+0200"),
    "466510": (4.23, 4.23, "L", "Stock", 2019, "2026-11-15T17:00:00+0200"),
    "464924": (6.59, 6.59, "L", "Stock", 271, "2026-11-15T17:00:00+0200"),
    "460697": (2.48, 2.48, "L", "Stock", 9999999999, "2026-11-15T17:00:00+0200"),
    "471970": (0.76, 0.76, "KG", "Stock", 9999999999, "2026-11-15T17:00:00+0200"),
    "473956": (2.65, 2.65, "L", "Stock", 9999999999, "2026-11-15T17:00:00+0200"),
}


class _FakeLive:
    """Price + availability for many products (LIVE_1114), recording calls like AimoClient."""

    def __init__(self, data=None, price_factor=1.0):
        self.data, self.factor, self.calls = data or LIVE_1114, price_factor, []

    def _log(self, endpoint, ids, date):
        self.calls.append({"endpoint": endpoint, "at": "2026-11-14T19:42:32+02:00", "requestedDate": date,
                           "products": len(ids), "response_sha256": "0" * 64})

    def price(self, ids, date):
        self._log("POST /api/price", ids, date)
        out = []
        for i in ids:
            if i in self.data:
                p, cmp_, unit = self.data[i][:3]
                out.append({"productId": i, "catalogPrice": {"value": round(p * self.factor, 2),
                                                             "valueWithTax": round(p * self.factor * 1.14, 2)},
                            "comparisonPrice": {"value": cmp_ * self.factor}, "comparisonUnit": unit})
        return {"products": out}

    def availability(self, ids, date):
        self._log("POST /api/availability", ids, date)
        out = []
        for i in ids:
            if i not in self.data:
                continue
            _, _, unit, logistic, amount, closing = self.data[i]
            lines = [] if logistic is None else [{"availableDate": "2026-11-16", "availableAmount": amount,
                                                  "salesUnit": "ST"}]
            out.append({"productId": i, "logisticModel": logistic, "closingTime": closing, "salesBlock": False,
                        "scheduleLines": lines})
        return {"timeNow": "2026-11-14T19:42:32+0200", "products": out}


def _ing(name, fi, en, qty, unit, role="supporting"):
    return {"name_en": name, "search_terms_fi": fi, "search_terms_en": en, "search_terms_sv": [], "form": "fresh",
            "role": role, "qty_per_portion": qty, "unit": unit}


def _fake_classify(groups, calls):
    """Identity by name, as Claude would judge these well-known products."""
    return {g["key"]: {c["id"]: {"verdict": "is_the_product" if c["id"] in LIVE_1114 and g["ingredient"].split()[-1][:5].lower()
                                 in (c["name_en"] + c["name_fi"]).lower() else "different_product", "reason": "test"}
                       for c in g["candidates"]} for g in groups}


@unittest.skipUnless(HAS_DATA, "needs the product data (python3 fetch_data.py)")
class Scenarios(unittest.TestCase):
    """demo_scenarios.md: ② partial availability with Best / Better / Good, ③ allergen safety under pressure."""

    @classmethod
    def setUpClass(cls):
        cls.by_id = {p["id"]: p for p in catalog.load_index()}

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "audit"))
        self.patches = [mock.patch.object(agent, "OUT", self.tmp),
                        mock.patch.object(agent, "ORDERS", os.path.join(self.tmp, "orders.json")),
                        mock.patch.object(agent, "PREFS", os.path.join(self.tmp, "prefs.json")),
                        mock.patch.dict(os.environ, {"AIMO_API_KEY": "test"}),
                        mock.patch.object(nlu, "classify_groups", _fake_classify)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    # rules
    def test_allergen_verdicts_on_real_products(self):
        cases = {"463797": "reject",   # Arla lactose-free cream: lactose-free is not milk-protein-free (AM)
                 "460697": "reject",   # Oatly iMat: oat is not gluten-free (AW)
                 "471881": "reject",   # claims gluten-free, but its own allergen list CONTAINS AW
                 "466510": "warn",     # AROY-D coconut cream: no allergen data at all
                 "473956": "pass",     # Fazer oat base: FREE_FROM_GLUTEN + VEGAN (vegan = no milk ingredient)
                 "473496": "pass",     # Elovena oat base: FREE_FROM_GLUTEN + VEGAN
                 "464745": "pass",     # Alpro soy cooking: free from AM, gluten-free claim
                 "470961": "pass",     # washed potatoes: reviewed rule for whole unprocessed produce
                 "460916": "reject"}   # cheddar filed under root vegetables: the allergen list still wins
        for pid, expect in cases.items():
            with self.subTest(pid=pid):
                verdict, reasons = catalog.allergen_verdict(self.by_id[pid], ["DAIRY_FREE", "FREE_FROM_GLUTEN"])
                self.assertEqual(verdict, expect)
                for reason in reasons:
                    self.assertEqual(reason["smt"]["violation"], "unsat" if reason["verdict"] == "pass" else "sat")
                    if reason["verdict"] != "pass":
                        expected = "unsat" if reason["verdict"] == "reject" else "sat"
                        self.assertEqual(reason["smt"]["compliance"], expected)

    def test_hospital_solver_cannot_clear_uncertain_evidence(self):
        import z3

        codes = ["DAIRY_FREE", "FREE_FROM_GLUTEN"]
        alpro = self.by_id["464745"]
        caution = dict(alpro, allergens_may_contain=["AM"])
        self.assertEqual(catalog.allergen_verdict(caution, codes)[0], "warn")
        with mock.patch.object(z3.Solver, "check", return_value=z3.unknown):
            self.assertEqual(catalog.allergen_verdict(alpro, codes)[0], "warn")
            self.assertEqual(catalog.allergen_verdict(self.by_id["463797"], codes)[0], "reject")

    def test_diet_groups_and_portions(self):
        groups = nlu.parse_diet_groups(HOSPITAL)
        self.assertEqual([(g["portions"], g["diet"], g["unmapped"]) for g in groups],
                         [(6.0, ["DAIRY_FREE", "FREE_FROM_GLUTEN"], [])])
        self.assertEqual(nlu.parse_portions(HOSPITAL)[0], 40.0)
        self.assertEqual(nlu.parse_portions(WEDDING)[0], 100.0)
        self.assertEqual(nlu.requirement_hints(HOSPITAL), [])          # 'and gluten' is understood, not unverifiable
        kiwi = nlu.parse_diet_groups("3 of the guests are allergic to kiwi")
        self.assertEqual((kiwi[0]["diet"], kiwi[0]["unmapped"]), ([], ["kiwi"]))  # reported, never dropped

    def test_group_disagreement_keeps_the_stricter_reading(self):
        trace = {"portions": 40.0, "issues": [], "notes": [], "unsupported": [], "spans": {"rules": []},
                 "llm_output": {"diet": [], "diet_groups": [{"portions": 6, "diet": ["DAIRY_FREE"], "allergy": True,
                                                            "quote": "6 of them are allergic to milk protein"}]}}
        nlu._merge_groups(HOSPITAL, trace, {"diet": ["DAIRY_FREE"]})
        self.assertEqual(trace["diet_groups"][0]["diet"], ["DAIRY_FREE", "FREE_FROM_GLUTEN"])
        self.assertTrue(any("mismatch" in i for i in trace["issues"]))
        self.assertEqual(trace["proposal_diet"], [])                   # the allergy is not everyone's diet

    def test_usual_product_needs_the_exact_ingredient(self):
        self.assertEqual(basket.usual_product("Carrots")[0], "460721")
        self.assertEqual(basket.usual_product("salmon fillet, skinless")[0], "462150")
        self.assertIsNone(basket.usual_product("sweet potato")[0])     # not 'potato'

    # baskets
    def _basket(self, text, recipe, portions, groups=(), subs=None, simulate=None, deliver_by="2026-11-21", given=None):
        trace = {"parser": "fixture", "llm_calls": [], "notes": [], "issues": [], "intent": "source", "dishes": ["soup"],
                 "portions": portions, "diet_groups": list(groups), "proposal_diet": []}
        rec = {"recipes": [{"name": "Soup", "description": "", "why_it_fits": "", "course": "main", "served": "hot",
                            "ingredients": recipe}], "avoid": [], "assumptions": []}
        def pick(gs, calls):  # Claude's pick, as it would choose among these well-known products
            out = {}
            for g in gs:
                ids = [c["id"] for c in g["candidates"]]
                live = [i for i in ids if i in LIVE_1114] or ids
                out[g["key"]] = {"product_id": live[0], "runner_up": live[1] if len(live) > 1 else "NONE", "reason": "t"}
            return out
        with mock.patch.object(nlu, "recipe_for", return_value=given or rec), \
                mock.patch.object(nlu, "suggest_substitutes", subs or (lambda items, calls: {})), \
                mock.patch.object(nlu, "pick_products", pick):
            return basket.run_basket(text, {"deliver_by": deliver_by}, trace, _policy(), _FakeLive(), TODAY,
                                     json.loads(json.dumps(agent.DEFAULT_PREFS)), None, simulate)

    def test_wedding_red_line_gets_best_better_good(self):
        parsnip = lambda items, calls: {it["key"]: [{"key": it["key"], "name_en": "parsnip", "search_terms_fi": ["palsternakka"],
                                                     "search_terms_en": ["parsnip"], "search_terms_sv": [],
                                                     "why_similar": "sweet root"}] for it in items}
        b = self._basket(WEDDING, [_ing("carrot", ["porkkana"], ["carrot"], 0.15, "KG", "key"),
                                   _ing("cooking cream", ["ruokakerma"], ["cooking cream"], 0.05, "L")], 100, subs=parsnip)
        carrot, cream = b["lines"]
        self.assertEqual((cream["light"], cream["chosen"]["productId"]), ("green", "464249"))
        self.assertEqual((carrot["light"], carrot["first_choice"]["productId"]), ("red", "460721"))  # usual: not deliverable
        best = carrot["tiers"]["best"][0]
        self.assertEqual((best["productId"], best["quantity"], best["relaxed"]), ("460811", 15, []))  # 100 x 0.15 kg = 15 bags
        self.assertEqual({o["productId"] for o in carrot["tiers"]["better"]}, {"460708", "462929"})
        self.assertTrue(all(any("pack size" in r for r in o["relaxed"]) for o in carrot["tiers"]["better"]))
        self.assertEqual([o["productId"] for o in carrot["tiers"]["good"]], ["460110"])
        self.assertEqual(carrot["default_option"], best["id"])           # Best is chosen for the buyer
        self.assertTrue(b["preference"]["asked"])
        before = [o["id"] for o in carrot["tiers"]["better"]]
        prefs = json.loads(json.dumps(agent.DEFAULT_PREFS))
        basket.set_preference(b, "stock", prefs)
        self.assertEqual(carrot["tiers"]["better"][0]["productId"], "462929")  # ample stock first; tiers unchanged
        self.assertEqual(sorted(before), sorted(o["id"] for o in carrot["tiers"]["better"]))
        basket.set_preference(b, "price", prefs)
        self.assertEqual(b["preference"]["answer"], "price")
        basket.set_preference(b, None, prefs)
        self.assertIsNone(b["preference"]["answer"])
        self.assertEqual([o["id"] for o in carrot["tiers"]["better"]], before)

    def _hospital(self, simulate=None):
        cream = lambda items, calls: {it["key"]: [
            {"key": it["key"], "name_en": "coconut cream", "search_terms_fi": ["kookoskerma"], "search_terms_en": ["coconut cream"],
             "search_terms_sv": [], "why_similar": "creamy"},
            {"key": it["key"], "name_en": "oat cooking base", "search_terms_fi": ["kauraruoka"],
             "search_terms_en": ["imat"], "search_terms_sv": [], "why_similar": "creamy"},
            {"key": it["key"], "name_en": "potato", "search_terms_fi": ["peruna"], "search_terms_en": ["potato"],
             "search_terms_sv": [], "why_similar": "puréed potato thickens"}] for it in items}
        group = {"portions": 6.0, "diet": ["DAIRY_FREE", "FREE_FROM_GLUTEN"], "allergy": True, "quote": "6 of them",
                 "read_by": "rules"}
        query = {"product_type": "other", "quantity": 4, "unit": "L", "deliver_by": "2026-11-16",
                 "product": {"name_en": "cooking cream", "search_terms_fi": ["ruokakerma"], "search_terms_en": ["cooking cream"],
                             "search_terms_sv": []}}
        recipe = agent._split_recipe(query, {"portions": 40.0})
        return self._basket(HOSPITAL, [], 40, [group], cream, simulate, deliver_by="2026-11-16", given=recipe)

    def test_hospital_splits_34_plus_6_and_never_offers_a_violation(self):
        b = self._hospital()
        self.assertEqual(len(b["lines"]), 2)                               # one product, two groups: one decision
        regular, allergy = b["lines"]
        self.assertEqual((regular["portions"], regular["need"]["qty"], regular["light"], regular["chosen"]["productId"]),
                         (34.0, 3.4, "green", "464249"))                   # 4 L / 40 portions x 34
        self.assertEqual((allergy["portions"], allergy["light"]), (6.0, "red"))
        offered = {o["productId"] for t in basket.TIERS for o in allergy["tiers"][t]}
        excluded = {x["productId"]: x for x in allergy["excluded"]}
        self.assertTrue({"464249", "463797", "460697"} <= set(excluded))  # milk (AM) and oat with gluten (AW)
        self.assertFalse(offered & set(excluded))
        self.assertEqual(excluded["463797"]["plain"], "contains milk protein, although lactose-free")
        self.assertEqual(excluded["460697"]["plain"], "contains gluten, although vegan")  # found by 'imat' here
        self.assertIn("466510", [o["productId"] for o in allergy["tiers"]["warn"]])   # no data: a warning
        safe_now = allergy["tiers"]["good"]
        self.assertTrue(safe_now and all(o["allergen_verdict"] == "pass" for o in safe_now))
        self.assertEqual(allergy["default_option"], safe_now[0]["id"])    # never 'later' or 'warn' for the buyer

    def test_simulated_delay_moves_safe_bases_to_later_and_is_labelled(self):
        b = self._hospital(simulate="hospital_delay")
        allergy = next(ln for ln in b["lines"] if ln["group"] == "g1" and ln["ingredient"] == "cooking cream")
        later = [o["productId"] for o in allergy["tiers"]["later"]]
        self.assertTrue({"464924", "473956"} & set(later))                 # safe bases, one day later (simulated)
        self.assertFalse(set(later) & {o["productId"] for o in allergy["tiers"]["good"]})
        self.assertEqual([o["productId"] for o in allergy["tiers"]["good"]], ["471970"])  # only the potato is safe now
        self.assertIn("464924", b["simulation"]["changed_products"])
        self.assertTrue(b["simulation"]["label"].startswith("SIMULATED"))

    def test_incomplete_evidence_needs_an_acknowledgement_and_is_never_learned(self):
        b = self._hospital()
        allergy = next(ln for ln in b["lines"] if ln["group"] == "g1" and ln["ingredient"] == "cooking cream")
        warn = next(o for o in allergy["tiers"]["warn"] if o["productId"] == "466510")
        self.assertTrue(all(r["smt"] == {"violation": "sat", "compliance": "sat"} for r in warn["allergen_reasons"]))
        with open(basket.write_audit(b, self.tmp), encoding="utf-8") as audit:
            self.assertIn("Z3 dairy-free: violation SAT, compliance SAT", audit.read())
        prefs = json.loads(json.dumps(agent.DEFAULT_PREFS))
        with self.assertRaises(ValueError):
            basket.confirm(json.loads(json.dumps(b)), {allergy["key"]: warn["id"]}, [], "chef", _FakeLive(), prefs, "t")
        done, updates = basket.confirm(b, {allergy["key"]: warn["id"]}, [allergy["key"]], "chef", _FakeLive(), prefs, "t")
        self.assertEqual(done["status"], "CONFIRMED_BY_HUMAN")
        self.assertEqual((updates, prefs["weights"]), ([], agent.DEFAULT_PREFS["weights"]))  # not learned
        self.assertEqual([t["productId"] for t in done["tasks"]], ["466510"])               # arrival label check
        d = next(ln for ln in done["lines"] if ln["key"] == allergy["key"])["decision"]
        self.assertEqual((d["gate"], d["acknowledged"], d["learnable"]), ("acknowledge", True, False))

    def test_a_person_can_change_the_quantity_of_a_line(self):
        b = self._hospital()
        regular = b["lines"][0]                                           # 34 regular portions: 4 x 1 L cream
        prefs = json.loads(json.dumps(agent.DEFAULT_PREFS))
        for bad in (0, -1, 2.5, "x"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                basket.confirm(json.loads(json.dumps(b)), {}, [], "chef", _FakeLive(), prefs, "t", {regular["key"]: bad})
        with self.assertRaises(ValueError):                              # more than the limited stock (114 pieces)
            basket.with_quantity(regular, dict(regular["chosen"], stock="limited: 114 ST"), 115)
        done, _ = basket.confirm(b, {}, [], "chef", _FakeLive(), prefs, "t", {regular["key"]: 6})
        line = done["lines"][0]
        self.assertEqual((line["chosen"]["quantity"], line["chosen"]["lineTotal"]), (6, 15.78))   # 6 x EUR 2.63
        self.assertEqual(line["decision"]["quantity_change"]["proposed"], 4)
        self.assertEqual(done["human_decision"]["committed_total_eur"],
                         round(sum(ln["chosen"]["lineTotal"] for ln in done["lines"]), 2))   # what is shown is what is committed

    def test_a_price_rise_before_confirmation_orders_nothing(self):
        b = self._hospital()
        done, _ = basket.confirm(b, {}, [], "chef", _FakeLive(price_factor=1.2), json.loads(json.dumps(agent.DEFAULT_PREFS)), "t")
        self.assertEqual(done["status"], "STALE_QUOTE")
        self.assertTrue(any("price rose" in c for c in done["revalidation"]["changes"]))

    def test_a_substitute_is_always_paid_for_even_if_another_line_buys_it(self):
        potato = lambda items, calls: {it["key"]: [{"key": it["key"], "name_en": "potato", "search_terms_fi": ["peruna"],
                                                    "search_terms_en": ["potato"], "search_terms_sv": [],
                                                    "why_similar": "root vegetable"}] for it in items}
        b = self._basket(WEDDING, [_ing("carrot", ["porkkana"], ["carrot"], 0.15, "KG", "key"),
                                   _ing("potato", ["peruna"], ["potato"], 0.08, "KG")], 100, subs=potato)
        carrot, spud = b["lines"]
        self.assertEqual(spud["light"], "green")
        sub = next(o for o in carrot["tiers"]["good"] if o["productId"] == spud["chosen"]["productId"])
        self.assertEqual(sub["quantity"], 15)                            # 15 kg of its own, not the potato line's spare
        self.assertGreater(sub["lineTotal"], 0)
        self.assertNotIn("no extra cost", sub["summary"])

    def test_one_product_with_a_diet_group_is_split_not_ordered_for_everyone(self):
        q = {"product_type": "other", "quantity": 4.0, "unit": "L", "deliver_by": "2026-11-16", "diet": [],
             "product": {"name_en": "cooking cream", "search_terms_fi": ["ruokakerma"], "search_terms_en": ["cooking cream"]}}
        group = {"portions": 6.0, "diet": ["DAIRY_FREE"], "allergy": True, "quote": "6 of them", "read_by": "rules"}
        trace = {"intent": "order", "blocking": [], "issues": [], "notes": [], "llm_calls": [], "diet_groups": [group],
                 "portions": 40.0, "proposal_diet": []}
        with mock.patch.object(nlu, "interpret", return_value=(q, trace)), \
                mock.patch.object(agent, "run_basket", return_value={"kind": "basket"}) as rb, \
                mock.patch.object(agent, "_run_order") as ro:
            agent.run_request(HOSPITAL, _policy(), use_llm=False, persist=False)
        self.assertEqual((rb.call_count, ro.call_count), (1, 0))
        self.assertAlmostEqual(rb.call_args.kwargs["recipe"]["recipes"][0]["ingredients"][0]["qty_per_portion"], 0.1)
        no_total = dict(trace, portions=None, blocking=[], issues=[])
        with mock.patch.object(nlu, "interpret", return_value=(q, no_total)), \
                mock.patch.object(agent, "_run_order", return_value={"order_id": "x"}) as ro:
            agent.run_request(HOSPITAL, _policy(), use_llm=False, persist=False)
        self.assertTrue(any("portions are there in total" in b for b in ro.call_args.args[2]["blocking"]))

    def test_diet_group_words_are_not_the_product_or_the_whole_order_diet(self):
        q = nlu.parse_rules(HOSPITAL, dt.date(2026, 11, 15))
        self.assertEqual((q["product_type"], q["diet"], q["quantity"], q["unit"]), (None, [], 4.0, "L"))

    # scenario 1: undo window
    def test_cancel_before_the_cut_off_only(self):
        o = dict(_pending("ORD-C"), status="PLACED_AUTONOMOUSLY")
        o["options"][0]["lines"][0]["orderCutOff"] = "2999-11-15T17:00:00+02:00"
        agent._save(agent.ORDERS, {"ORD-C": o})
        self.assertEqual(agent.cancel_order("ORD-C", "chef")["status"], "CANCELLED_BY_HUMAN")
        self.assertEqual(agent.cancel_order("ORD-C", "chef")["idempotent_replay"]["kind"], "decision")
        late = dict(_pending("ORD-L"), status="PLACED_AUTONOMOUSLY")
        late["options"][0]["lines"][0]["orderCutOff"] = "2020-01-01T17:00:00+02:00"
        agent._save(agent.ORDERS, {"ORD-L": late})
        with self.assertRaises(ValueError):
            agent.cancel_order("ORD-L", "chef")
        with self.assertRaises(ValueError):
            agent.cancel_order("ORD-L-unknown", "chef")

    def test_demo_clock(self):
        with mock.patch.dict(os.environ, {"AGENT_TODAY": "2026-11-15"}):
            self.assertEqual(agent.today(), dt.date(2026, 11, 15))
            self.assertEqual(nlu.parse_date("lunch tomorrow", agent.today())[0], "2026-11-16")


# --- API contract (aimo_contract.py): recorded mock responses and an API that keeps the contract ---------
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "mock_api_2026-11-14.json")
NOW_1114 = "2026-11-14T22:35:25+0200"


def _live(sched, closing="2026-11-15T17:00:00+0200", logistic="Stock", deliver_by="2026-11-20", price=None,
          avail_extra=None, unit="ST"):
    """live_check for one candidate with the given schedule lines (and price / response fields)."""
    cand = {"id": "1", "name": "x", "measure": "L", "per_unit": 1.0, "kg_per_measure": 1.0, "pref_match": 1.0,
            "salesUnit": unit, "evidence_strength": 1.0, "evidence": "", "packs_per_unit": 1.0}
    pr = dict({"productId": "1", "catalogPrice": {"value": 1.0, "valueWithTax": 1.14}}, **(price or {}))
    avail = dict({"timeNow": NOW_1114, "products": [{"productId": "1", "salesBlock": False, "closingTime": closing,
                                                     "logisticModel": logistic, "scheduleLines": sched}]},
                 **(avail_extra or {}))
    return engine.live_check([cand], {"products": [pr]}, avail, {"deliver_by": deliver_by}, {}, {})


def _codes(offer):
    return [a["code"] for a in offer["anomalies"]]


def _sched(*lines):
    return [{"availableDate": d, "availableAmount": a, "salesUnit": "ST"} for d, a in lines]


class ApiContract(unittest.TestCase):
    def test_recorded_mock_responses_break_only_the_cut_off_rule(self):
        with open(FIXTURE, encoding="utf-8") as f:
            rec = json.load(f)
        result = probe_contract.analyse(rec)
        self.assertTrue(result["requested_date_ignored"])  # Mon, Wed and Fri asked: the same answer
        status, detail = result["report"]["CUTOFF_FOR_AVAILABLE_DATE"]
        self.assertEqual((status, "468201" in detail, "474421" in detail, "464306" in detail), ("broken", True, True, False))
        for cid in ("ONE_SCHEDULE_LINE", "TERMINAL_AMOUNT", "AMPLE_SENTINEL", "LOGISTIC_MODEL", "DATE_ECHO",
                    "PRICE_FOR_ONE_UNIT"):
            self.assertEqual(result["report"][cid][0], "kept", cid)

    def test_recorded_mock_responses_through_the_engine(self):
        with open(FIXTURE, encoding="utf-8") as f:
            rec = json.load(f)
        ids = rec["products"]
        for date, early, terminal in (("2026-11-16", 0, "CUTOFF_PASSED"), ("2026-11-20", 4, None)):
            r = rec["dates"][date]
            q, _ = engine.quote_products(ids, r["price"], r["availability"], dt.date.fromisoformat(date), {})
            with self.subTest(deadline=date):
                self.assertEqual((q["464306"]["orderable"], q["464306"]["days_early"], q["464306"]["anomalies"]),
                                 (True, early, []))
                self.assertEqual(q["468201"]["code"], terminal)     # needed Monday: its cut-off has passed
                self.assertEqual(q["460721"]["code"], "NOT_DELIVERABLE")
        self.assertEqual(q["468201"]["anomalies"], ["STALE_CUTOFF"])  # needed Friday: kept, a person confirms
        self.assertTrue(q["468201"]["delivery_note"].startswith("STALE_CUTOFF"))

    def test_an_api_that_keeps_the_contract_raises_no_anomaly(self):
        # requestedDate respected, cut-off still open, dates echoed, price for one unit
        offers, rej, _ = _live(_sched(("2026-11-20", 9999999999)), closing="2026-11-19T17:00:00+0200",
                               price={"quantity": 1.0, "pricingDate": "2026-11-20T00:00:00+0200"},
                               avail_extra={"requestedDate": "2026-11-20T00:00:00+0200"})
        o = offers[0]
        self.assertEqual((_codes(o), o["days_early"], o["delivery_confirmed"], o["qty_confirmed"]), ([], 0, True, True))
        plan = dict(_plan(), lines=[{"_o": dict(o, reliability=1.0), "name": "x"}])
        decision, checks = engine.guardrails(plan, {"product_type": "milk", "preferences": {}, "diet": []}, _policy(), [])
        self.assertFalse(any("API contract" in c for _, c in checks))
        self.assertEqual(decision, "AUTO_ORDER")

    def test_several_schedule_lines_are_added_up_and_need_a_person(self):
        offers, _, _ = _live(_sched(("2026-11-18", 600), ("2026-11-16", 400), ("2026-11-21", 500)))
        o = offers[0]
        self.assertEqual((o["available_amt"], o["available_date"], _codes(o)), (1000, "2026-11-18", ["MULTIPLE_SCHEDULE_LINES"]))
        plan = dict(_plan(), lines=[{"_o": dict(o, reliability=1.0), "name": "x"}])
        decision, checks = engine.guardrails(plan, {"product_type": "milk", "preferences": {}, "diet": []}, _policy(), [])
        self.assertEqual(decision, "NEEDS_HUMAN_APPROVAL")
        self.assertTrue(any(not ok and "MULTIPLE_SCHEDULE_LINES" in c for ok, c in checks))

    def test_contract_breaks_become_anomalies(self):
        o = _live(_sched(("2026-11-16", 50)), logistic="Terminal", closing="2026-11-15T17:00:00+0200")[0][0]
        self.assertEqual((_codes(o), o["available_amt"], o["qty_confirmed"]), (["UNEXPECTED_AMOUNT"], 50, False))
        o = _live(_sched(("2026-11-16", 999999)))[0][0]                      # the Terminal value on a Stock item
        self.assertEqual((_codes(o), o["qty_confirmed"]), (["UNEXPECTED_AMOUNT"], False))
        o = _live(_sched(("2026-11-16", 10)), logistic="Dropship")[0][0]
        self.assertEqual((_codes(o), o["qty_confirmed"]), (["UNKNOWN_LOGISTIC_MODEL"], False))
        o = _live(_sched(("2026-11-16", 9999999999)), price={"quantity": 10.0})[0][0]
        self.assertEqual(_codes(o), ["PRICE_QUANTITY"])
        echo = aimo_contract.check_echo({"requestedDate": "2026-11-16T00:00:00+0200"}, "/api/availability", "2026-11-20")
        self.assertEqual([a["code"] for a in echo], ["DATE_ECHO_MISMATCH"])
        self.assertEqual(aimo_contract.check_echo({"requestedDate": "2026-11-20T00:00:00+0200"}, "/api/availability",
                                                  "2026-11-20"), [])
        o = _live(_sched(("2026-11-16", 9999999999)), avail_extra={"contract_anomalies": echo})[0][0]
        self.assertEqual(_codes(o), ["DATE_ECHO_MISMATCH"])                # applies to every product of the response
        self.assertEqual(sorted({c["anomaly"] for c in aimo_contract.CONTRACT if c["anomaly"]}), aimo_contract.ANOMALY_CODES)

    def test_approval_recheck_reports_only_new_anomalies(self):
        stale = _FakeAimo(closing="2026-11-12T14:00:00+0200")  # cut-off passed, deadline later than the earliest date
        price, avail = stale.price(["464306"], "d"), stale.availability(["464306"], "d")
        line = _pending()["options"][0]["lines"][0]
        deadline = dt.date(2026, 11, 20)
        self.assertIn("no longer confirmed", engine.revalidate([line], price, avail, deadline)[0][0])
        for seen in (dict(line, anomalies=["STALE_CUTOFF"]), dict(line, flags=["DELIVERY_UNCONFIRMED: older order"])):
            self.assertEqual(engine.revalidate([seen], price, avail, deadline)[0], [])  # the person saw it already

    def test_lot_size_rounding(self):
        def units(sales_unit, lot, need):
            o = {"id": "1", "name": "x", "per_unit": 1.0, "allowedLotSize": lot, "sales_unit": sales_unit,
                 "unit_price": 2.0, "unit_price_tax": 2.3}
            return engine._line(o, need)["units"]
        self.assertEqual(units("KG", 0.01, 2.35), 2.35)   # sold by weight: the lot step only
        self.assertEqual(units("KG", 1.5, 2.0), 3.0)
        self.assertEqual(units("ST", 0.01, 2.35), 3)      # count units: whole units first
        self.assertEqual(units("KI", 2.0, 3.0), 4)

    def test_explanation_says_how_many_days_early(self):
        offers, _, _ = _live(_sched(("2026-11-16", 9999999999)))
        query = {"quantity": 10, "deliver_by": "2026-11-20", "diet": []}
        plans = engine.rank(engine.build_plans([dict(offers[0], reliability=1.0)], query),
                            engine.effective_weights(agent.DEFAULT_PREFS, query))
        text = " ".join(engine.explain(plans[0], plans, 1, query))
        self.assertIn("Arrives 4 days earlier than needed (2026-11-16 instead of 2026-11-20)", text)
        self.assertEqual(agent._public_plan(plans[0], 0)["lines"][0]["daysEarly"], 4)

    def test_readme_lists_every_contract_item(self):
        with open(os.path.join(os.path.dirname(catalog.HERE), "README.md"), encoding="utf-8") as f:
            readme = f.read()
        for c in aimo_contract.CONTRACT:
            self.assertIn(f"`{c['id']}`", readme, c["id"])
