"""KD2 regressions: a materiality replay that stops on the card's own stated table size is reported as that limit, never
as an open premise (decision KD2; results/kd2-20261017/REPORT.md).

Fixture fixtures/calibration_math_20261017T233208Z/k1_run1_table_limit.json, written read-only by
results/kd2-20261017/make_fixture.py from runs/calibration-math-20261017T233208Z/k1-run1, with the sha256 of every file
it read: the card, the frozen record, the stage M checks, the recorded challenger finding C-585bf6c5 ("out-of-table
position generation") and the recorded stage C outcome (P1 conditional on C-585bf6c5 after an IndexError replay).

Stage C runs for real on the recorded stage M, with the cost replay on the host executor and no live model call.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from hybrid.math import cost
from hybrid.math.runner import MathRunner
from hybrid.research.ledger import Ledger
from hybrid.research.stages import c_challenge, m_math, table_limit
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import Budget, StageContext
from hybrid.research.store import Store
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, orch_config

FIX = json.loads((Path(__file__).parent / "fixtures" / "calibration_math_20261017T233208Z"
                  / "k1_run1_table_limit.json").read_text(encoding="utf-8"))
CARD_ID = FIX["card"]["id"]
FINDING = FIX["finding"]
FINDING_ID = FIX["recorded_c"]["finding"]["id"]
HOST_KINDS = ("audited_construction", "conservation", "magic_x", "finding", "cost")
CRASH_EVIDENCE = FIX["recorded_c"]["replay_check"]["evidence"]
LIMITS = FIX["record"]["card_limits"]
SWEEP = tuple(cost.DEFAULT_NS)
SIZES = {8, 32}   # O1.a sweeps V and d over these
# The earlier calibration (20261017T030854Z k1-run1, premise C-a14f39aa) raised on a line that reads two parameters.
EARLIER_EVIDENCE = (CRASH_EVIDENCE.split("; raised at")[0] + "; raised at line 36 in update(): "
                    "h = np.add(params['E'][token], params['P'][t]) of the submitted source")


class Replay:
    """Stage C on the recorded k1-run1 stage M: the frozen record and the M verdict are stored as stage M stored them;
    the challenger gives ``findings`` in its first batch and nothing in the others; every statement matches."""

    def __init__(self, tmp_path, findings_=(), *, rec=None):
        self.run_dir = (tmp_path / "run").resolve()
        self.ledger = Ledger(self.run_dir)
        self.store = Store(self.run_dir, self.ledger)
        self.store.put_card(dict(copy.deepcopy(FIX["card"]), version=1))
        self.executor = mf.MathExecutor(tmp_path / "exec", host_kinds=HOST_KINDS)
        batches = len(c_challenge.check_batches(FIX["m_checks"]))
        replies = [mf.challenger_output(findings=list(findings_))] + [mf.challenger_output()] * (batches - 1)
        script = {"challenger": replies, "statement_backtranslate": mf.backtranslate_output,
                  "statement_match": mf.match_output()}
        self.llm = FakeLLM(self.run_dir, self.ledger, script,
                           models={"challenger": "gpt-6.1-sol", "statement_match": "qwen3.6-35b-a3b-mlx"})
        self.ctx = StageContext(run_id=self.run_dir.name, run_dir=self.run_dir, ledger=self.ledger, store=self.store,
                                llm=self.llm, runner=None, bench=None, cfg=orch_config(),
                                budget=Budget(1, 500, 10**9, 3600.0), lenses={},
                                math=MathRunner(self.executor, mf.IMAGE, self.run_dir))
        rec = copy.deepcopy(rec or FIX["record"])
        for statement in m_math.all_statements(rec):
            statement["sha256"] = m_math.statement_digest(statement)
        for definition in rec["definitions"]:
            definition["sha256"] = mc.definition_hash(definition)
        stored = self.ctx.put_json_artifact("math_statements", self.ctx.artifact_name(CARD_ID), rec, CARD_ID)
        hashes = {s["id"]: s["sha256"] for s in m_math.all_statements(rec)}
        hashes.update({d["id"]: d["sha256"] for d in rec["definitions"]})
        self.ctx.emit("math_statements_frozen", {"card_id": CARD_ID, "path": stored["path"].relative_to(
            self.run_dir).as_posix(), "sha256": stored["sha256"], "hashes": hashes})
        verdict = {"card_id": CARD_ID, "stage": "M", "verdict": "fail", "kill_codes": [],
                   "claims": FIX["m_claims"], "models": FIX["m_models"], "checks": copy.deepcopy(FIX["m_checks"])}
        self.ctx.put_json_artifact("math_verdict", self.ctx.artifact_name(CARD_ID), verdict, CARD_ID)

    def run(self):
        return asyncio.run(c_challenge.run(self.ctx, CARD_ID))

    def verdict(self) -> dict:
        return [v for v in c_challenge._verdicts(self.ctx, CARD_ID) if v["stage"] == "C"][-1]

    def statuses(self) -> dict:
        return {c["claim_id"]: c["status"] for c in self.verdict()["claims"]}

    def challenge(self) -> dict:
        return [json.loads((self.run_dir / e["data"]["path"]).read_text()) for e in self.ledger.iter_events("artifact")
                if e["data"]["kind"] == "challenge"][-1]

    def findings(self) -> dict:
        return {f["id"]: f for f in self.challenge()["findings"] if f.get("parent")}

    def check_artifact(self, name: str) -> dict:
        return json.loads((self.run_dir / "artifacts" / "math_check" / name).read_text())

    def cost_requests(self) -> list[dict]:
        return [r for r in self.executor.requests() if r["kind"] == "cost" and r["source"] != FIX["m_checks"]["O1.a"]["text"]]


def finding_with_edit(replacement: str) -> dict:
    """The recorded finding with its single edit's replacement changed (the host's strict lookup is the recorded one)."""
    changed = copy.deepcopy(FINDING)
    changed["targets"][0]["edits"][0]["replacement"] = replacement
    return changed


def piece(replay: Replay) -> dict:
    (item,) = replay.findings().values()
    return item


# ----- 1. the recorded run and its old outcome -------------------------------------------------------------------------

def test_the_fixture_is_the_recorded_run_and_its_old_outcome():
    recorded = FIX["recorded_c"]
    assert recorded["finding"]["id"] == FINDING_ID == "C-585bf6c5"
    assert recorded["finding"]["materiality"]["outcome"] == "unresolved"
    assert recorded["claims"]["P1"] == "conditional" and "unproven:C-585bf6c5" in recorded["kill_codes"]
    assert recorded["replay_check"]["status"] == "unknown" and not recorded["replay_check"]["decided"]
    assert "index 256 is out of bounds for axis 0 with size 256" in CRASH_EVIDENCE
    assert "256" in FIX["m_checks"]["O1.a"]["text"] and "params['P'][t]" in CRASH_EVIDENCE
    # The frozen statement does not pin n: the sweep is the host default, which passes the 256 rows.
    assert "ns" not in FIX["record"]["obligations"][0]["statements"][0]["spec"] and max(SWEEP) == 4096
    assert any("table P" in limit["resources"] for limit in LIMITS)


# ----- 2. the detector ---------------------------------------------------------------------------------------------

def crashed(evidence: str = CRASH_EVIDENCE, **changes) -> dict:
    return dict({"decided": False, "status": "unknown", "evidence": evidence}, **changes)


def test_the_recorded_crash_is_the_cards_stated_table_limit():
    expected = {"table": "P", "rows": 256, "index": 256, "inside_ns": [64, 128, 256],
                "beyond_ns": [512, 1024, 2048, 4096]}
    assert table_limit.table_overrun(crashed(), LIMITS, SWEEP, SIZES) == expected


def test_the_earlier_calibrations_crash_on_a_line_that_reads_two_tables_is_the_same_limit():
    assert table_limit.table_overrun(crashed(EARLIER_EVIDENCE), LIMITS, SWEEP, SIZES)["table"] == "P"
    # E has V rows: a failure with a size of V or d is not a table of fixed size, even on the same line.
    sized = EARLIER_EVIDENCE.replace("index 256", "index 8").replace("size 256", "size 8")
    assert table_limit.table_overrun(crashed(sized), LIMITS, SWEEP, SIZES) is None


@pytest.mark.parametrize("why, check, limits, ns", [
    ("a table whose rows are a size of the statement", crashed(CRASH_EVIDENCE.replace("256", "32")), LIMITS, SWEEP),
    ("a sweep inside the table cannot overrun it", crashed(), LIMITS, (64, 128, 256)),
    ("a table the card's limits do not name", crashed(), [dict(LIMITS[0], resources="a position store with N rows")],
     SWEEP),
    ("no card limits at all", crashed(), [], SWEEP),
    ("a different table is not the declared one", crashed(CRASH_EVIDENCE.replace("params['P']", "params['E']")),
     LIMITS, SWEEP),
    ("an index into a row is not a row of the table",
     crashed(CRASH_EVIDENCE.replace("params['P'][t]", "params['P'][t][d]")), LIMITS, SWEEP),
    ("a second axis is not a row", crashed(CRASH_EVIDENCE.replace("axis 0", "axis 1")), LIMITS, SWEEP),
    ("an index inside the rows is some other bug", crashed(CRASH_EVIDENCE.replace("index 256", "index 255")),
     LIMITS, SWEEP),
    ("another exception", crashed("KeyError: 'P'; raised at line 33 in position_row(): return params['P'][t]"),
     LIMITS, SWEEP),
    ("a decided check", crashed(decided=True, status="violated"), LIMITS, SWEEP),
    ("a check that passed", crashed(decided=True, status="discharged"), LIMITS, SWEEP),
])
def test_any_other_crash_is_not_a_table_limit(why, check, limits, ns):
    assert table_limit.table_overrun(check, limits, ns, SIZES) is None, why


# ----- 3. stage C on the recorded run ----------------------------------------------------------------------------------

def test_the_replay_that_hit_the_table_limit_is_reported_as_the_limit_and_p1_is_no_longer_conditional(tmp_path):
    replay = Replay(tmp_path, [FINDING])
    replay.run()
    assert replay.statuses()["P1"] == "observed"
    assert not any(code.startswith("unproven:C-") for code in replay.verdict()["kill_codes"])
    item = piece(replay)
    assert item["disposition"] == "immaterial" and item["used_by"] == ["O1"]
    record = item["materiality"]
    assert record["outcome"] == "immaterial" and record["original_sha256"] == FIX["provenance"]["source_sha256"]["O1.a"]
    assert record["variant_sha256"] == FIX["provenance"]["variant_sha256"]
    limit = record["domain_limit"]
    assert (limit["table"], limit["rows"], limit["index"]) == ("P", 256, 256)
    assert limit["inside_ns"] == [64, 128, 256] and limit["beyond_ns"] == [512, 1024, 2048, 4096]
    # The outcome names the limit and the sizes, in the record and in the verdict.
    for text in (record["evidence"], item["reason"]):
        assert "table P to 256 rows" in text or "stated table limit" in text
    assert "undefined at n = 512, 1024, 2048, 4096" in record["evidence"] and "n = 64, 128, 256" in record["evidence"]
    summary = {f["id"]: f for f in replay.verdict()["findings"]}[item["id"]]
    assert summary["materiality"]["outcome"] == "immaterial" and summary["materiality"]["domain_limit"] == limit
    # The crash is kept as evidence; the decided replay is a second check of the same variant inside the table.
    crash = replay.check_artifact(limit["crash_artifact"])
    assert crash["status"] == "unknown" and not crash["decided"] and limit["crash_evidence"] == crash["evidence"]
    decided = replay.check_artifact(record["check"]["artifact"])
    assert decided["decided"] and decided["status"] == "discharged" and decided["text"] == record["variant_source"]
    assert decided["purpose"] == "materiality"
    # Both replays run the frozen statement; only the sweep differs, and only in the second.
    first, *inside = replay.cost_requests()
    assert "ns" not in first and len(inside) == 2   # a decided cost check is two agreeing runs
    for second in inside:
        assert second["ns"] == [64, 128, 256]
        assert {k: v for k, v in second.items() if k != "ns"} == first
    assert first["source"] == record["variant_source"]
    # Stage M evidence is untouched.
    assert replay.verdict()["checks"]["O1.a"]["text"] == FIX["m_checks"]["O1.a"]["text"]


@pytest.mark.parametrize("lookup, error", [
    ("params['scale'][t]", "TypeError"),
    # An index error on the width of a row, not on the rows of the table.
    ("params['P'][t][d]", "IndexError: index"),
])
def test_a_replay_crash_of_another_kind_stays_unresolved(tmp_path, lookup, error):
    replay = Replay(tmp_path, [finding_with_edit(f"def position_row(params, t, d, ops):\n    return {lookup}")])
    replay.run()
    item = piece(replay)
    assert item["disposition"] == "unresolved" and "domain_limit" not in item["materiality"]
    assert error in item["materiality"]["evidence"]
    assert replay.statuses()["P1"] == "conditional"
    assert f"unproven:{item['id']}" in replay.verdict()["kill_codes"]
    assert len(replay.cost_requests()) == 1


def test_an_out_of_bounds_index_on_a_table_the_card_does_not_state_stays_unresolved(tmp_path):
    rec = copy.deepcopy(FIX["record"])
    for limit in rec["card_limits"]:
        limit["resources"] = limit["resources"].replace("table P", "store P")
    replay = Replay(tmp_path, [FINDING], rec=rec)
    replay.run()
    item = piece(replay)
    assert item["disposition"] == "unresolved" and "domain_limit" not in item["materiality"]
    assert "IndexError: index 256 is out of bounds for axis 0 with size 256" in item["materiality"]["evidence"]
    assert replay.statuses()["P1"] == "conditional"
    assert len(replay.cost_requests()) == 1
