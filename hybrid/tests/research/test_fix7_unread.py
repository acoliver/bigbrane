"""FIX-7 item 6: a statement-match reply that fails its schema after the retry is no reading, never a stage error.

Recorded data: fixtures/calibration_math_20261014/k2_run1_backtranslate.json, the failed back-translation of
runs/calibration-math-20261014T053341Z/k2-run1 (call-212c9bc7358c, extracted by
results/phase4-20261013T073748Z/fix7-logs/make_fix7_k2_fixture.py). The local matcher was shown sixteen statements, two
of them identical (O4.a, the learning claim, and O11.a, P4.copy). Both replies finished normally (finish_reason stop,
about 11.8k of 16000 output tokens) with fifteen translations: the duplicate was translated once, the ids of the four
statements after it moved up by one and O15.a was left out. The retry was told only that the whole array, quoted back
in ten thousand characters, "is too short". Stage C raised SchemaValidationError and the card-run stopped at C.

The stage C cases replay the recorded k3-run1 stage M (test_fix7_match.Replay) with the failing role scripted to raise.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from hybrid.llm import LLMClient, SchemaValidationError, TransportError
from hybrid.llm import jsonmode
from hybrid.research import calibration
from hybrid.research.stages import c_challenge
from hybrid.research.stages import match_adjudication as ma
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import StageError
from hybrid.tests.llm.fakes import FakeBackend, ListSink, write_config
from hybrid.tests.research.test_fix7_match import CARD_ID, M_STATUSES, Replay, recorded, with_statement_objection

FIXTURE = Path(__file__).parent / "fixtures" / "calibration_math_20261014" / "k2_run1_backtranslate.json"
K2 = json.loads(FIXTURE.read_text(encoding="utf-8"))
SHOWN = json.loads(K2["user"][len("FORMAL STATEMENTS:\n"):K2["user"].index("\n\nDEFINITIONS:")])
IDS = [s["statement_id"] for s in SHOWN]


def reply_object(raw: str) -> dict:
    return jsonmode.extract_first_json_object(raw)


def recorded_failure() -> SchemaValidationError:
    exc = SchemaValidationError(K2["error"].split(": ", 1)[1])
    exc.call_id = K2["call_id"]
    return exc


# ----- what the matcher produced ------------------------------------------------------------------------------------

def test_the_recorded_replies_finished_normally_and_shifted_the_ids_after_a_duplicate():
    assert K2["role"] == "statement_match" and K2["error"].startswith("SchemaValidationError: output invalid after")
    assert [a["kind"] for a in K2["attempts"]] == ["request", "schema_retry"]
    for attempt in K2["attempts"]:
        # Not a truncation: the replies stopped on their own, well inside the role's output limit.
        assert attempt["finish_reason"] == "stop"
        assert attempt["max_output_tokens"] == 16000 and attempt["usage"]["output"] < 12000
        translations = reply_object(attempt["raw_output"])["translations"]
        assert [t["statement_id"] for t in translations] == IDS[:-1]
        by_id = {t["statement_id"]: t["plain"] for t in translations}
        # O11.a is the copy learnability statement, but the reading labelled O11.a is recall's (O12.a), and the
        # conservation statement O15.a was read under O14.a's id.
        assert "recall" in by_id["O11.a"] and "conservation" in by_id["O14.a"]
    by_id = {s["statement_id"]: s for s in SHOWN}
    assert {k: v for k, v in by_id["O4.a"].items() if k != "statement_id"} == \
        {k: v for k, v in by_id["O11.a"].items() if k != "statement_id"}
    # The schema itself is what it should be: one translation per statement.
    assert K2["json_schema"]["properties"]["translations"]["minItems"] == len(IDS) == 16
    # The retry was shown the whole array back and "is too short", nothing that names the missing statement.
    first = K2["attempts"][0]["validation_error"]
    assert len(first) > 10000 and first.endswith("is too short") and "O15.a" not in first.split("]")[-1]


def test_identical_statements_are_translated_once():
    distinct, same_as = c_challenge.distinct_statements(SHOWN)
    assert len(distinct) == 15 and "O11.a" not in [s["statement_id"] for s in distinct]
    assert same_as["O11.a"] == "O4.a"
    assert all(same_as[sid] == sid for sid in IDS if sid != "O11.a")
    schema = mc.backtranslate_schema([s["statement_id"] for s in distinct])
    assert schema["properties"]["translations"]["minItems"] == 15


def test_a_short_array_is_described_by_its_count_and_the_ids_it_lacks():
    parsed, error = jsonmode.parse_and_validate(K2["attempts"][0]["raw_output"], K2["json_schema"])
    assert parsed is None
    assert "at translations: has 15 items; the schema requires exactly 16; no item has statement_id O15.a" in error
    assert len(error) < 1000 and "is too short" not in error
    repeated = {"translations": [{"statement_id": "O1.a", "plain": "x"}] * 17}
    _, error = jsonmode.parse_and_validate(json.dumps(repeated), K2["json_schema"])
    assert "has 17 items; the schema requires exactly 16" in error
    assert "no item has statement_id O2.a, O3.a" in error and "more than one item has statement_id O1.a" in error


def test_the_retry_of_the_recorded_call_names_the_missing_statement(tmp_path, monkeypatch):
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    backend = FakeBackend(script=[a["raw_output"] for a in K2["attempts"]])
    client = LLMClient(write_config(tmp_path), str(tmp_path / "run"), ListSink(), backends={"a": backend,
                                                                                         "b": FakeBackend()})
    with pytest.raises(SchemaValidationError, match="no item has statement_id O15.a"):
        asyncio.run(client.complete("r", "sys", K2["user"], json_schema=K2["json_schema"]))
    assert len(backend.requests) == 2
    retry = backend.requests[1].user
    assert "It was rejected: schema validation failed at translations: has 15 items; the schema requires exactly " \
           "16; no item has statement_id O15.a" in retry


# ----- stage C reads a failed reply as no reading ---------------------------------------------------------------------

def unread_replay(tmp_path, prompt: str, exc: Exception, compare: dict | None = None) -> Replay:
    replay = Replay(tmp_path, compare or recorded())
    replay.llm.fail = {prompt: exc}
    return replay


def test_a_back_translation_that_fails_its_schema_leaves_every_statement_unresolved(tmp_path):
    replay = unread_replay(tmp_path, "statement_backtranslate", recorded_failure())
    replay.run()                                     # no StageError: the card-run reaches its C verdict
    assert "statement_match" not in replay.llm.prompts()
    challenge = replay.challenge()
    assert [(u["role"], u["prompt"], u["call_id"]) for u in challenge["matcher_unread"]] == [
        ("statement_match", "statement_backtranslate", K2["call_id"])]
    assert challenge["statement_match"] and all(
        i["match"] == ma.UNRESOLVED and not i["mismatch"] for i in challenge["statement_match"])
    assert "no reading settles this match" in challenge["statement_match"][0]["disputed"][0]
    # Every claim keeps the status its checks gave, and every formalized one is unresolved.
    assert replay.statuses() == M_STATUSES
    claims = replay.claims()
    formalized = {cid for cid, c in claims.items() if c["status"] != "not_formalized"}
    assert len(formalized) == 17 and all(claims[cid]["match"] == "unresolved" for cid in formalized)
    verdict = replay.verdict()
    assert verdict["verdict"] == "fail" and "statement_mismatch" not in verdict["kill_codes"]
    assert {f"match_unresolved:{cid}" for cid in formalized} <= set(verdict["kill_codes"])
    assert K2["call_id"] in replay.store.get_card(CARD_ID)["status"]["verdicts"]["C"]["by_calls"]
    # Calibration scores the cells from their checks (outcome_record) and lists every one as unresolved.
    assert replay.cells() == {"P1": "pass", "P2.copy": "fail", "P2.recall": "fail", "P2.nesting": "fail",
                              "P2.composition": "fail", "P3": "pass", "P5": "fail"}
    assert all(claims[cell]["match"] == "unresolved" for cell in calibration.CELLS)


def test_a_comparison_that_fails_its_schema_leaves_every_statement_unresolved(tmp_path):
    replay = unread_replay(tmp_path, "statement_match", recorded_failure())
    replay.run()
    assert replay.llm.prompts().count("statement_backtranslate") == 1
    challenge = replay.challenge()
    assert [u["prompt"] for u in challenge["matcher_unread"]] == ["statement_match"]
    assert all(i["match"] == ma.UNRESOLVED for i in challenge["statement_match"])
    # The translations were read and are kept with the statements.
    assert all(i["back_translation"] for i in challenge["statement_match"])
    assert challenge["circularity"] == [] and challenge["definitions"] == []
    assert replay.statuses() == M_STATUSES
    assert all(c["match"] == "unresolved" for c in replay.claims().values() if c["status"] != "not_formalized")


def test_a_cross_check_that_fails_its_schema_leaves_its_objection_unresolved(tmp_path):
    objection = {"item": "exponents", "claim_quote": "Theta(k V)", "statement_quote": "c*(k*v) ≤ tokenCost k v",
                 "difference": "the claim is linear in k V; suppose the statement bounded k*v*v"}
    replay = unread_replay(tmp_path, "statement_crosscheck", recorded_failure(),
                           with_statement_objection("O1.a", objection))
    replay.run()
    claims = replay.claims()
    assert (claims["complexity.time_per_token"]["status"], claims["complexity.time_per_token"]["match"]) == \
        ("proven", "unresolved")
    challenge = replay.challenge()
    assert challenge["crosscheck"]["verdicts"] is None and challenge["crosscheck"]["call_id"] == K2["call_id"]
    o1 = next(i for i in challenge["statement_match"] if i["statement_id"] == "O1.a")
    assert o1["match"] == ma.UNRESOLVED and o1["objections"][0]["adjudication"] == ma.UNREAD
    # Statements without objections are still matched.
    assert all(i["match"] == ma.MATCHED for i in challenge["statement_match"] if i["statement_id"] != "O1.a")
    assert "statement_mismatch" not in replay.verdict()["kill_codes"]


def test_other_matcher_failures_still_stop_the_stage(tmp_path):
    replay = unread_replay(tmp_path, "statement_backtranslate", TransportError("all backend entries failed"))
    with pytest.raises(StageError, match="TransportError"):
        replay.run()


def test_the_recorded_k3_back_translation_request_shows_its_duplicate_once(tmp_path):
    replay = Replay(tmp_path, recorded())
    replay.run()
    shown = json.loads(re.search(r"FORMAL STATEMENTS:\n(.*?)\n\nDEFINITIONS:", replay.user("statement_backtranslate"),
                                 re.S).group(1))
    ids = [s["statement_id"] for s in shown]
    assert "O4.a" in ids and "O13.a" not in ids
    # O13.a (P4.copy) still gets the reading of its identical statement, and is compared with its own claim.
    item = next(i for i in replay.challenge()["statement_match"] if i["statement_id"] == "O13.a")
    o4 = next(i for i in replay.challenge()["statement_match"] if i["statement_id"] == "O4.a")
    assert item["back_translation"] == o4["back_translation"] != ""
    assert item["match"] == ma.MATCHED
