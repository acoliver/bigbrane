"""Regression tests from the live Phase 4 calibration of 2026-10-13 (runs/calibration-math-20261013T082935Z), which
stopped after k1-run1, k2-run1 and part of k3-run1 (results/phase4-20261013T073748Z/FIX-1.md).

The fixtures in fixtures/calibration_live_20261013/ are the recorded model replies and records, with call and
message ids replaced:

- k1_prover_truncated.txt: Claude Opus's prover reply that the gateway returned with finish_reason=length after 64000
  output tokens. The old extractor skipped the unclosed outer object and returned the first attempt inside it, so the
  error said the root had unexpected keys ``notes``, ``proof``, ``source`` and ``statement_id``, the retry told the
  model the same, and the second reply was cut off again: a stage error ended the card-run.
- k2_prover_declined.txt and k2_prover_preamble.txt: Claude Sonnet's replies (the fallback after Opus returned empty
  content at the cap) that left every construction and learner null, one of them after tool-call debris.
- k2_frozen_statements.json: the statements the fixer froze for k2 (12 prover statements in one call before).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hybrid.llm import LLMClient, SchemaValidationError
from hybrid.llm import jsonmode
from hybrid.llm.backends import BackendReply
from hybrid.math import cost, profile
from hybrid.research import calibration
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.tests.llm.fakes import FakeBackend, ListSink, write_config
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_math_stages import Harness

LIVE = Path(__file__).parent / "fixtures" / "calibration_live_20261013"


def live_text(name: str) -> str:
    return (LIVE / name).read_text(encoding="utf-8")


def live_json(name: str) -> dict:
    return json.loads((LIVE / name).read_text(encoding="utf-8"))


# ----- defect 1: a reply cut off at the output limit -------------------------------------------------------------

def test_the_cut_off_k1_reply_is_reported_as_unclosed_not_as_inner_keys_at_the_root():
    meta = live_json("k1_prover_truncated.json")
    text = live_text("k1_prover_truncated.txt")
    assert meta["finish_reason"] == "length" and meta["usage"]["output"] == 64000
    assert "Additional properties" in meta["validation_error_then"]
    parsed, error = jsonmode.parse_and_validate(text, meta["json_schema"])
    assert parsed is None
    assert "never closed" in error and "ends inside it" in error
    assert "Additional properties" not in error and "statement_id" not in error
    with pytest.raises(jsonmode.UnclosedJSON):
        jsonmode.extract_first_json_object(text)


def test_an_inner_object_of_a_malformed_outer_object_is_never_returned():
    text = '{"attempts": [{"statement_id": "O1.a", "proof": null}, oops]} {"attempts": []}'
    assert jsonmode.extract_first_json_object(text) == {"attempts": []}


def test_the_k2_reply_after_tool_call_debris_still_parses():
    text = live_text("k2_prover_preamble.txt")
    assert text.lstrip().startswith("<invoke")
    parsed = jsonmode.extract_first_json_object(text)
    assert [a["statement_id"] for a in parsed["attempts"]][:4] == ["O1.a", "O2.a", "O3.a", "O4.a"]


def test_a_bare_attempt_object_names_every_wrong_key_and_the_missing_wrapper():
    schema = live_json("k1_prover_truncated.json")["json_schema"]
    bare = json.dumps({"statement_id": "O4.a", "proof": None, "source": "def init(sizes, ops): pass", "notes": "x"})
    parsed, error = jsonmode.parse_and_validate(bare, schema)
    assert parsed is None
    assert "'attempts' is a required property" in error
    for key in ("notes", "proof", "source", "statement_id"):
        assert repr(key) in error


class CutOffBackend(FakeBackend):
    """Replies like the gateway did on k1: the text so far, finish_reason=length, the whole cap used."""

    async def complete(self, request):
        self.requests.append(request)
        return BackendReply(text=self.script.pop(0), model="claude-opus-5-5",
                            usage={"input": 16012, "output": 64000, "cached": 0},
                            detail={"finish_reason": "length"})


def test_the_client_says_the_reply_was_cut_off_and_asks_for_a_shorter_one(tmp_path):
    meta = live_json("k1_prover_truncated.json")
    text = live_text("k1_prover_truncated.txt")
    backend = CutOffBackend(script=[text, text])
    client = LLMClient(write_config(tmp_path), str(tmp_path / "run"), ListSink(), backends={"a": backend})
    with pytest.raises(SchemaValidationError) as raised:
        asyncio.run(client.complete("solo", "s", "u", json_schema=meta["json_schema"]))
    assert "cut off at the output limit (finish_reason=length, 64000 output tokens)" in str(raised.value)
    assert raised.value.call_id and raised.value.call_id.startswith("call-")
    retry = backend.requests[1].user
    assert "cut off by the output limit" in retry and "never closed" in retry
    assert "Additional properties" not in retry


# ----- defect 1: a schema slip is a failed attempt, not the end of the card-run -----------------------------------

def test_an_unusable_prover_reply_costs_one_attempt_and_the_rounds_go_on(tmp_path):
    text = live_text("k1_prover_truncated.txt")
    _, error = jsonmode.parse_and_validate(text, live_json("k1_prover_truncated.json")["json_schema"])
    good = mf.prover_output(mf.GOOD_PROOF)
    lean_calls = []

    def prover(index, user):
        if '"form": "lean"' in user:
            lean_calls.append(index)
        if len(lean_calls) == 1 and '"form": "lean"' in user:
            raise SchemaValidationError("output invalid after one retry: the reply was cut off at the output "
                                        f"limit (finish_reason=length, 64000 output tokens); {error}")
        return good(index, user)

    h = Harness(tmp_path, dict(mf.math_script(), prover=prover))
    assert h.m() == "C"
    attempts = [(e["data"]["statement_id"], e["data"]["attempt"], e["data"]["status"])
                for e in h.events("proof_attempt")]
    assert ("O5.a", 1, "invalid_reply") in attempts and ("O5.a", 2, "pass") in attempts
    assert ("O6.a", 1, "discharged") in attempts
    retry = [c["user"] for c in h.llm.calls if c["role"] == "prover" and '"statement_id": "O5.a"' in c["user"]][1]
    assert "previous_attempt_failed" in retry and "could not be used" in retry and "never closed" in retry
    invalid = [e["data"] for e in h.events("math_role_call") if e["data"].get("invalid_reply")]
    assert len(invalid) == 1 and "cut off" in invalid[0]["invalid_reply"]
    assert not h.events("stage_error")


def test_an_unusable_reply_on_every_attempt_leaves_the_claim_unproven_without_a_stage_error(tmp_path):
    def prover(index, user):
        if '"form": "lean"' in user:
            raise SchemaValidationError("output invalid after one retry: no JSON object found in model output")
        return mf.prover_output(mf.GOOD_PROOF)(index, user)

    h = Harness(tmp_path, dict(mf.math_script(), prover=prover))
    assert h.m() is None
    assert h.verdict("M")["kill_codes"] == ["unproven:capability.1"]
    statuses = [e["data"]["status"] for e in h.events("proof_attempt") if e["data"]["statement_id"] == "O5.a"]
    assert statuses == ["invalid_reply", "invalid_reply"]


# ----- defect 2: one call for every statement -------------------------------------------------------------------

def test_the_k2_statements_go_to_one_call_per_construction_and_small_groups_by_form():
    record = live_json("k2_frozen_statements.json")
    todo = [s for s in m_math.all_statements(record) if s["form"] in ("lean", "construction", "cost", "learnability")]
    assert len(todo) == 12
    batches = [[s["id"] for s in batch] for batch in m_math.prover_batches(record, todo)]
    assert batches == [["O1.a", "O2.a", "O3.a", "O4.a"], ["O5.a"], ["O6.a"], ["O7.a"], ["O8.a", "O8.b"], ["O9.a"],
                       ["O10.a", "O11.a"]]
    assert all(len(batch) <= m_math.PROVER_BATCH for batch in batches)


def test_a_construction_call_shows_only_its_statements_with_the_profile_and_sandbox_notes():
    record = live_json("k2_frozen_statements.json")
    card = calibration.load_cards(("k2",))["k2"]
    batch = [s for s in m_math.all_statements(record) if s["id"] in ("O8.a", "O8.b")]
    user = m_math.prover_user(card, record, batch, {})
    shown = json.JSONDecoder().raw_decode(user.split("FROZEN STATEMENTS:\n", 1)[1])[0]
    assert [item["statement_id"] for item in shown] == ["O8.a", "O8.b"]
    assert "PROFILE OBLIGATIONS (Phase 4)" in user and "SANDBOX." in user
    lean_only = [s for s in m_math.all_statements(record) if s["id"] == "O9.a"]
    assert "SANDBOX." not in m_math.prover_user(card, record, lean_only, {})


def test_the_sandbox_note_lists_the_checkers_builtins():
    note = m_math.sandbox_note()
    for name in cost.SAFE_BUILTINS:
        if not name.startswith("_"):
            assert name in note
    assert "__import__" not in note and "isinstance" in note and "ops.np.normal" in note


def test_the_prover_prompt_requires_the_wrapper_and_an_attempt_for_every_statement():
    from hybrid.research.stages.common import load_prompt
    prompt = load_prompt("prover")
    assert '{"attempts": [...]}' in prompt and "bare attempt object" in prompt
    assert "An attempt with both null counts as a failed attempt" in prompt


def test_the_declined_k2_reply_is_recorded_as_no_text_for_each_null_statement(tmp_path):
    declined = json.loads(live_text("k2_prover_declined.txt"))
    nulls = [a["statement_id"] for a in declined["attempts"] if not a["proof"] and not a["source"]]
    assert nulls == ["O5.a", "O6.a", "O7.a", "O8.a", "O8.b", "O10.a", "O11.a"]
    schema = mc.prover_schema([a["statement_id"] for a in declined["attempts"]])
    assert jsonmode.parse_and_validate(live_text("k2_prover_declined.txt"), schema)[0] == declined


# ----- defect 2: the fixer had no example counts, the matcher compared parts with whole slots ---------------------

def test_every_calibration_card_states_m_for_each_p4_task_and_its_information_account():
    for key, card in calibration.load_cards().items():
        text = card["claims"]["sample_efficiency"]["mechanism"]
        for task in ("copying at L", "recall at K", "bracket nesting at depth 2", "two-hop lookup at table size"):
            assert task in text, (key, task)
        # FIX-5: a defined class per task replaced the uniform eight-bit account (test_fix5_p5_cards.py).
        assert "Class definition:" in text and "Prior bits: 0" in text and "Bits per example" in text


def test_the_fixer_is_told_which_task_parameters_a_statement_takes():
    note = m_math.params_note()
    assert "copy: D" in note and "recall: V, n_keys" in note and "binding: values" in note
    assert "definition" in note


def test_the_matcher_is_told_which_part_of_a_profile_slot_each_statement_covers():
    record = live_json("k2_frozen_statements.json")
    user = c_challenge.match_user(record, {}, [], [], card=calibration.load_cards(("k2",))["k2"])
    items = json.loads(user.split("STATEMENTS TO COMPARE:\n", 1)[1].split("\n\nDEFINITIONS:", 1)[0])
    by_id = {item["statement_id"]: item for item in items}
    assert by_id["O8.a"]["profile_slot"] == {"slot": "P2.composition", "statements": ["O8.a", "O8.b"],
                                             "part": "composition"}
    assert by_id["O8.b"]["profile_slot"]["part"] == "binding"
    assert by_id["O4.a"]["profile_slot"] == {"slot": "P1", "statements": ["O4.a"], "part": None}
    assert "profile_slot" not in by_id["O1.a"]
    assert "PROFILE SLOTS (Phase 4)" in user and "symbolic formula P1 asks for" in user


def test_a_card_without_the_profile_gets_no_profile_match_section():
    record = {"obligations": mf.fixer_output()["obligations"], "claims": [], "definitions": [], "premises": []}
    for obligation in record["obligations"]:
        for statement in obligation["statements"]:
            statement["id"] = f"{obligation['id']}.a"
            statement["spec"] = statement[statement["form"]]
    user = c_challenge.match_user(record, {}, [], [], card=calibration.load_cards(("k1",))["k1"])
    assert "PROFILE SLOTS" not in user and not any(profile.is_slot(o["claim_id"]) for o in record["obligations"])
