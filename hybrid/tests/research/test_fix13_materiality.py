"""FIX-13 regressions: a witness violation decides a claim only through a host replay of a conforming replacement, and
a convention chosen for P1's exact count binds only the cost reference (results/phase4-20261013T073748Z/k1-maxsub-diag/
REPORT.md and FIX-13.md).

Fixture fixtures/calibration_math_20261014T204729Z/k1_run1_maxsub.json, written read-only by
results/phase4-20261013T073748Z/fix13-staging/make_fixture.py from runs/calibration-math-20261014T204729Z/k1-run1 and
k1-maxsub-diag, with the sha256 of every file it read: the card, the frozen record, the stage M checks (the recorded
O6.a copy and O7.a recall witnesses among them), the recorded challenger finding C-01fd5cf1 ("witnesses use max
subtraction, D2 forbids it"), the recorded stage C outcome (P2.copy and P2.recall failed on it), and the report's
no-max sources.

Stage C runs for real on the recorded stage M, with the recorded finding restated with FIX-13 targets; every finding
verification and every audited construction runs on the host executor.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid.math import constructions, findings, spec_checks
from hybrid.math.runner import MathRunner
from hybrid.research.ledger import Ledger
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import Budget, StageContext, load_prompt, load_section
from hybrid.research.store import Store
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, orch_config

FIX = json.loads((Path(__file__).parent / "fixtures" / "calibration_math_20261014T204729Z" / "k1_run1_maxsub.json")
                 .read_text(encoding="utf-8"))
CARD_ID = FIX["card"]["id"]
FINDING = FIX["finding"]
SUBJECT = FINDING["subject"]
COPY, RECALL = FIX["m_checks"]["O6.a"]["text"], FIX["m_checks"]["O7.a"]["text"]
# The two edits of the report's replay (k1-maxsub-diag/replay.py): only the max subtraction goes.
COPY_TARGET = mf.target("O6.a", "sc - sc.max()", [("np.exp(sc - sc.max())", "np.exp(sc)")])
RECALL_TARGET = mf.target("O7.a", "s = s - s.max()", [("    s = s - s.max()\n", "")])
# sha256 of the recorded sources, from k1-maxsub-diag/provenance.json.
COPY_SHA = "e093078e5bb4b006d904709ad7fa602a8ba6e15802df033d496f71afa8673b97"
RECALL_SHA = "7fdc58731f895812842288f628daa2730c3b2ab132367495cfd18c623c6c21a7"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finding(*targets, used_by=("O6", "O7"), **changes) -> dict:
    """The recorded finding as the challenger wrote it, with FIX-13 targets."""
    return dict(copy.deepcopy(FINDING), used_by=list(used_by), targets=[copy.deepcopy(t) for t in targets],
                **changes)


def piece_id(sid: str, oid: str, rec: dict | None = None) -> str:
    return c_challenge.finding_id(rec or FIX["record"], f"{SUBJECT} {sid}", [oid])


class FailingExecutor(mf.MathExecutor):
    """The host executor, except that a job whose source is ``source`` stops with a timeout (an executor stop, not
    a result)."""

    def __init__(self, tmp_path, source: str):
        super().__init__(tmp_path)
        self.source = source

    def run(self, job):
        staged = Path(job.input_dir) / "request.json"
        request = json.loads(staged.read_text()) if staged.is_file() else {}
        if request.get("source") == self.source:
            self.jobs.append(job)
            return SimpleNamespace(status="timeout", detail="wall clock cap reached", exit_code=None, run_s=600.0,
                                   stdout_path=str(self._empty(job, "stdout")),
                                   stderr_path=str(self._empty(job, "stderr")))
        return super().run(job)

    def _empty(self, job, stream: str) -> Path:
        path = self.tmp_path / f"{job.job_id}.{stream}"
        path.write_text("")
        return path


class Replay:
    """Stage C on the recorded k1-run1 stage M: the frozen record and the M verdict are stored as stage M stored them;
    the challenger gives ``findings`` in its first batch and nothing in the others; every statement matches."""

    def __init__(self, tmp_path, findings_=(), *, rec=None, executor=None):
        self.run_dir = (tmp_path / "run").resolve()
        self.ledger = Ledger(self.run_dir)
        self.store = Store(self.run_dir, self.ledger)
        # The store numbers versions from 1; the recorded card is version 2 (stage M bumped it).
        self.store.put_card(dict(copy.deepcopy(FIX["card"]), version=1))
        self.executor = executor or mf.MathExecutor(tmp_path / "exec")
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

    def construction_requests(self) -> list[dict]:
        return [r for r in self.executor.requests() if r["kind"] == "audited_construction"]

    def ledger_entries(self) -> dict:
        name = self.verdict()["assumption_ledger"]
        data = json.loads((self.run_dir / "artifacts" / "assumption_ledger" / name).read_text())
        return {p["id"]: p for p in data["premises"]}


# ----- 1. the recorded run and the report's replay -------------------------------------------------------------------

def test_the_fixture_is_the_recorded_run_and_its_old_outcome():
    assert sha(COPY) == COPY_SHA == FIX["provenance"]["source_sha256"]["O6.a"]
    assert sha(RECALL) == RECALL_SHA == FIX["provenance"]["source_sha256"]["O7.a"]
    assert FIX["requests"]["O6.a"]["source"] == COPY and FIX["requests"]["O7.a"]["source"] == RECALL
    # D2 is the P1 counting convention both P2 obligations cite.
    d2 = next(d for d in FIX["record"]["definitions"] if d["id"] == "D2")
    assert "without max subtraction" in d2["text"] and d2["kind"] == "accounting" and "scope" not in d2
    cites = {o["id"]: o["definitions"] for o in FIX["record"]["obligations"]}
    assert "D2" in cites["O6"] and "D2" in cites["O7"] and "D2" in cites["O1"]
    # Before FIX-13 the existence of the deviation failed both claims: the verification checks two substrings.
    recorded = FIX["recorded_c"]
    assert recorded["finding"]["id"] == "C-01fd5cf1" and recorded["finding"]["disposition"] == "violated"
    assert recorded["finding"]["used_by"] == ["O6", "O7"]
    assert recorded["claims"]["P2.copy"] == recorded["claims"]["P2.recall"] == "failed"
    assert {"obligation_failed:P2.copy", "obligation_failed:P2.recall"} <= set(recorded["kill_codes"])
    assert "s = s - s.max()' in recall" in FINDING["code"] and "accuracy" not in FINDING["code"]
    # The report's replay: copy passes with and without the subtraction, recall only with it.
    statuses = {(r["statement"], r["variant"]): r["statuses"] for r in FIX["replay"]}
    assert statuses[("O6.a", "no_max")] == ["discharged"] * 3
    assert statuses[("O7.a", "original")] == ["discharged"] * 3 and statuses[("O7.a", "no_max")] == ["violated"] * 3


def test_the_host_builds_the_reports_conforming_variants():
    for source, target in ((COPY, COPY_TARGET), (RECALL, RECALL_TARGET)):
        variant, problems = findings.conforming_variant(source, target["deviation"], target["edits"])
        sid = target["statement_id"]
        assert problems == [] and variant == FIX["no_max"][sid]
        assert sha(variant) == FIX["provenance"]["no_max_sha256"][sid]
        assert "max()" not in variant


def test_the_recall_witness_overflows_without_max_subtraction_and_copy_does_not():
    """The frozen precision is float64 and D2 excludes the shift, so real-arithmetic shift invariance cannot excuse
    the overflow: the recall variant is decided violated at every size, from its first token."""
    contract = FIX["requests"]["O7.a"]["architecture"]
    assert contract == {"hidden": "d", "precision_bits": 64}
    spec = dict(FIX["requests"]["O7.a"]["spec"], count=40)
    with np.errstate(over="ignore", invalid="ignore"):
        recall = spec_checks.audited_construction(spec, FIX["no_max"]["O7.a"], contract)
        original = spec_checks.audited_construction(spec, RECALL, contract)
    assert original["status"] == "discharged"
    assert recall["status"] == "violated"
    for row in recall["results"]:
        assert not row["passed"] and row["accuracy"] == 0.0
        assert any("non-finite carried state" in v and "instance 0, after token 0" in v
                   for v in row["resource_violations"])
    copy_spec = dict(FIX["requests"]["O6.a"]["spec"], count=40)
    result = spec_checks.audited_construction(copy_spec, FIX["no_max"]["O6.a"], FIX["requests"]["O6.a"]["architecture"])
    assert result["status"] == "discharged" and all(r["accuracy"] == 1.0 for r in result["results"])


# ----- 2. stage C on the recorded run ---------------------------------------------------------------------------------

def test_the_recorded_finding_splits_copy_is_immaterial_and_recall_is_material(tmp_path):
    replay = Replay(tmp_path, [finding(COPY_TARGET, RECALL_TARGET)])
    replay.run()
    statuses = replay.statuses()
    assert statuses["P2.copy"] == "observed"
    assert statuses["P2.recall"] == "failed"
    codes = replay.verdict()["kill_codes"]
    assert "obligation_failed:P2.recall" in codes and "obligation_failed:P2.copy" not in codes
    pieces = replay.findings()
    copy_id, recall_id = piece_id("O6.a", "O6"), piece_id("O7.a", "O7")
    assert set(pieces) == {copy_id, recall_id} and copy_id != recall_id
    copy_item, recall_item = pieces[copy_id], pieces[recall_id]
    assert copy_item["disposition"] == "immaterial" and copy_item["used_by"] == ["O6"]
    assert recall_item["disposition"] == "violated" and recall_item["used_by"] == ["O7"]
    assert copy_item["verification"]["status"] == recall_item["verification"]["status"] == "confirmed"
    # The archive: original and variant hashes, the edit, the variant source and the replay's check artifact.
    for item, original, sid in ((copy_item, COPY_SHA, "O6.a"), (recall_item, RECALL_SHA, "O7.a")):
        record = item["materiality"]
        assert record["original_sha256"] == original
        assert record["variant_sha256"] == FIX["provenance"]["no_max_sha256"][sid]
        assert record["variant_source"] == FIX["no_max"][sid]
        assert record["edits"] == (COPY_TARGET if sid == "O6.a" else RECALL_TARGET)["edits"]
        check = replay.check_artifact(record["check"]["artifact"])
        assert check["stage"] == "C" and check["purpose"] == "materiality" and check["statement_id"] == sid
        assert check["text"] == FIX["no_max"][sid] and check["decided"]
    assert copy_item["materiality"]["outcome"] == "immaterial"
    assert "L=64 instance accuracy 1.0000" in copy_item["materiality"]["evidence"]
    assert recall_item["materiality"]["outcome"] == "material"
    assert "non-finite carried state" in recall_item["materiality"]["evidence"]
    # The replays ran under the frozen statements and contract, unchanged.
    frozen = {s["id"]: s["spec"] for s in m_math.all_statements(FIX["record"])}
    replays = [r for r in replay.construction_requests() if r["source"] in (FIX["no_max"]["O6.a"],
                                                                             FIX["no_max"]["O7.a"])]
    assert len(replays) == 4   # each decided run is replayed once
    for request in replays:
        sid = "O6.a" if request["source"] == FIX["no_max"]["O6.a"] else "O7.a"
        assert request["spec"] == frozen[sid] and request["architecture"] == FIX["requests"][sid]["architecture"]
    # The verdict keeps a summary per statement, the stage M evidence is untouched.
    summary = {f["id"]: f for f in replay.verdict()["findings"]}
    assert summary[copy_id]["statement_id"] == "O6.a" and summary[copy_id]["materiality"]["outcome"] == "immaterial"
    assert summary[recall_id]["materiality"]["status"] == "violated"
    assert replay.verdict()["checks"]["O6.a"]["text"] == COPY


def test_a_mixed_finding_with_one_named_witness_settles_each_statement_on_its_own(tmp_path):
    replay = Replay(tmp_path, [finding(COPY_TARGET)])
    replay.run()
    pieces = replay.findings()
    copy_id, recall_id = piece_id("O6.a", "O6"), piece_id("O7", "O7")
    assert pieces[copy_id]["disposition"] == "immaterial"
    # Nothing identifies the recall witness to replay: unresolved, which blocks without a measured failure.
    assert pieces[recall_id]["disposition"] == "unresolved" and pieces[recall_id]["materiality"] is None
    statuses = replay.statuses()
    assert statuses["P2.copy"] == "observed" and statuses["P2.recall"] == "conditional"
    # The card still fails on P4.composition (recorded at M); the recall gap is an open premise, not a failure.
    assert replay.verdict()["kill_codes"] == ["obligation_failed:P4.composition"]
    entry = replay.ledger_entries()[recall_id]
    assert entry["status"] == "open" and entry["used_by"] == ["O7"]
    assert "without asserting a measured failure" in entry["basis"]
    assert not any(r["source"] == FIX["no_max"]["O7.a"] for r in replay.construction_requests())


RETUNED = mf.target("O7.a", "S1 = 1.0e7 * sq", [("    S1 = 1.0e7 * sq\n", "    S1 = 1.0e2 * sq\n")])


@pytest.mark.parametrize("deviation, edits, problem", [
    # Retuning a weight: a number the anchor does not have.
    ("S1 = 1.0e7 * sq", [("    S1 = 1.0e7 * sq\n", "    S1 = 1.0e2 * sq\n")], "adds numbers [100.0]"),
    # More precision: a name the source does not use, and a precision name.
    ("s = s - s.max()", [("    s = s - s.max()\n", "    s = s.astype(np.longdouble)\n")],
     "names ['astype', 'longdouble']"),
    # An anchor that reaches a line without the deviation.
    ("s = s - s.max()", [("    s = s - s.max()\n    w = np.exp(s)\n", "    w = np.exp(s)\n")],
     "hold no part of the deviation"),
    # An anchor that is not unique.
    ("s = s - s.max()", [("s", "t")], "times, not once"),
    # An edit that leaves the deviation in place.
    ("s.max()", [("    s = s - s.max()\n", "    s = s - s.max() \n")], "still contains the deviation"),
    # No edit at all.
    ("s = s - s.max()", [], "no edit is given"),
])
def test_an_edit_that_is_not_a_narrow_conforming_replacement_is_refused(deviation, edits, problem):
    variant, problems = findings.conforming_variant(RECALL, deviation, [{"anchor": a, "replacement": r}
                                                                        for a, r in edits])
    assert variant is None and any(problem in p for p in problems), problems


def test_a_refused_edit_leaves_the_witness_unresolved_and_runs_nothing(tmp_path):
    replay = Replay(tmp_path, [finding(RETUNED, used_by=("O7",))])
    replay.run()
    recall = replay.findings()[piece_id("O7.a", "O7")]
    assert recall["disposition"] == "unresolved"
    assert recall["materiality"]["outcome"] == "unresolved" and recall["materiality"]["check"] is None
    assert any("adds numbers" in p for p in recall["materiality"]["problems"])
    assert replay.statuses()["P2.recall"] == "conditional"
    assert not any("1.0e2" in r["source"] for r in replay.construction_requests())


def test_an_executor_failure_on_the_replay_is_unresolved_not_a_measured_failure(tmp_path):
    executor = FailingExecutor(tmp_path / "exec", FIX["no_max"]["O7.a"])
    replay = Replay(tmp_path, [finding(RECALL_TARGET, used_by=("O7",))], executor=executor)
    replay.run()
    recall = replay.findings()[piece_id("O7.a", "O7")]
    assert recall["disposition"] == "unresolved"
    assert recall["materiality"]["check"]["decided"] is False
    assert recall["materiality"]["check"]["status"] == "unknown"
    assert any(r.get("source") == FIX["no_max"]["O7.a"] for r in executor.requests())
    statuses = replay.statuses()
    assert statuses["P2.recall"] == "conditional" and statuses["P2.copy"] == "observed"
    assert "obligation_failed:P2.recall" not in replay.verdict()["kill_codes"]


def test_a_deviation_that_is_not_in_the_named_witness_is_rejected_for_it(tmp_path):
    wrong = mf.target("O6.a", "s = s - s.max()", [("np.exp(sc - sc.max())", "np.exp(sc)")])
    replay = Replay(tmp_path, [finding(wrong, RECALL_TARGET)])
    replay.run()
    pieces = replay.findings()
    assert pieces[piece_id("O6.a", "O6")]["disposition"] == "rejected"
    assert pieces[piece_id("O7.a", "O7")]["disposition"] == "violated"
    assert replay.statuses()["P2.copy"] == "observed"


# ----- 3. a P1 cost-reference convention binds only the cost reference ------------------------------------------------

def scoped_record() -> dict:
    """The recorded record with D2 frozen as a P1 cost-reference convention, as the FIX-13 fixer writes it."""
    rec = copy.deepcopy(FIX["record"])
    next(d for d in rec["definitions"] if d["id"] == "D2")["scope"] = "cost_reference"
    return rec


def test_a_finding_anchored_in_a_cost_reference_convention_does_not_bind_a_witness(tmp_path):
    rec = scoped_record()
    replay = Replay(tmp_path, [finding(COPY_TARGET, RECALL_TARGET)], rec=rec)
    replay.run()
    pieces = replay.findings()
    for sid, oid in (("O6.a", "O6"), ("O7.a", "O7")):
        item = pieces[piece_id(sid, oid, rec)]
        assert item["disposition"] == "out_of_scope" and "scoped to the P1 cost reference" in item["reason"]
    statuses = replay.statuses()
    assert statuses["P2.copy"] == statuses["P2.recall"] == "observed"
    assert not any(r["source"] in FIX["no_max"].values() for r in replay.construction_requests())
    # The challenger is shown the scope.
    assert '"scope": "cost_reference"' in next(c["user"] for c in replay.llm.calls if c["role"] == "challenger")


def test_a_definition_the_obligation_does_not_cite_does_not_bind_its_witness(tmp_path):
    rec = copy.deepcopy(FIX["record"])
    for obligation in rec["obligations"]:
        if obligation["id"] in ("O6", "O7"):
            obligation["definitions"] = [d for d in obligation["definitions"] if d != "D2"]
    replay = Replay(tmp_path, [finding(COPY_TARGET, RECALL_TARGET)], rec=rec)
    replay.run()
    assert {f["disposition"] for f in replay.findings().values()} == {"out_of_scope"}
    assert replay.statuses()["P2.recall"] == "observed"


def fixer_answer() -> dict:
    """A fixer answer shaped like k1-run1's: D2 (the counting convention) scoped to the cost reference and cited by
    P1, its conformance audit and the P2 copy obligation."""
    rec = FIX["record"]
    keep = {"O1", "O2", "O6"}
    obligations = [{"id": o["id"], "claim_id": o["claim_id"], "claim_quote": o["claim_quote"],
                    "statements": [{"form": s["form"], s["form"]: copy.deepcopy(s["spec"])} for s in o["statements"]],
                    "definitions": list(o["definitions"]), "premises": list(o["premises"]),
                    "discharges": list(o["discharges"])} for o in rec["obligations"] if o["id"] in keep]
    definitions = [{k: d[k] for k in ("id", "kind", "text")} for d in rec["definitions"]]
    next(d for d in definitions if d["id"] == "D2")["scope"] = "cost_reference"
    # FIX-20: a conformance statement carries the host's precision policy and exact tolerance. The fixture card
    # predates the card's resources section, so the host leaves the spec as written; these are the values it sets
    # for a card that declares binary64, as k1 does.
    conformance = next(s for o in obligations for s in o["statements"] if s["form"] == "conformance")
    conformance["conformance"].update(precision_bits=64, tolerance=0.0)
    return {"obligations": obligations, "definitions": definitions, "premises": [], "not_formalized": []}


def test_the_fixer_is_told_and_freeze_removes_a_cost_reference_convention_from_a_witness():
    answer = fixer_answer()
    problems = mc.scope_problems(answer["obligations"], answer["definitions"])
    assert len(problems) == 1 and problems[0].startswith("O6: cites D2, scoped to the P1 cost reference")
    texts = {c["id"]: c["text"] for c in FIX["record"]["claims"]}
    record = mc.freeze(answer, FIX["card"], ["P1", "P2.copy"], 40, texts)
    assert [o["id"] for o in record["obligations"]] == ["O1", "O2", "O6"]
    cited = {o["id"]: o["definitions"] for o in record["obligations"]}
    assert "D2" in cited["O1"] and "D2" in cited["O2"] and "D2" not in cited["O6"]
    assert any(d.get("obligation_id") == "O6" and d.get("definitions") == ["D2"] for d in record["dropped"])
    # Without a scope the hash is the recorded one; with a scope it changes, so the scope is frozen.
    d2 = next(d for d in FIX["record"]["definitions"] if d["id"] == "D2")
    assert mc.definition_hash(d2) == d2["sha256"]
    assert mc.definition_hash(dict(d2, scope="cost_reference")) != d2["sha256"]


def test_the_schemas_carry_scope_and_targets():
    fixer = mc.statement_fixer_schema(["P1"], 4, profile_required=True)
    definition = fixer["properties"]["definitions"]["items"]
    assert definition["properties"]["scope"]["enum"] == ["shared", "cost_reference"]
    assert "scope" not in definition["required"]
    challenger = mc.challenger_schema([], ["O6", "O7"], ["O6.a", "O7.a"], [], [])
    item = challenger["properties"]["findings"]["items"]
    assert "targets" in item["required"]
    target = item["properties"]["targets"]["items"]
    assert target["required"] == ["statement_id", "deviation", "edits", "why"]
    assert target["properties"]["statement_id"]["enum"] == ["O6.a", "O7.a"]
    stored = json.loads((Path(m_math.__file__).parents[1] / "schemas" / "math_statement.schema.json").read_text())
    assert stored["properties"]["definitions"]["items"]["properties"]["scope"]["enum"] == ["shared", "cost_reference"]


def test_the_prompts_separate_the_three_contracts_and_state_the_materiality_rule():
    fixer = load_prompt("statement_fixer")
    assert "Three separate contracts" in fixer and '"scope": "cost_reference"' in fixer
    assert "log-sum-exp" in fixer and "does not excuse an overflow" in fixer
    mechanism = load_section("mechanism_definitions")
    assert "numerical semantics" in mechanism and "cost_reference" in mechanism
    obligations = load_section("profile_obligations")
    assert "never binds a P2 or P4 witness" in obligations
    assert "the max reduction and the subtraction are counted" in load_section("conformance_audit")
    challenge = load_section("challenge_findings")
    assert "MATERIALITY OF A WITNESS VIOLATION" in challenge and "`targets`" in challenge
    assert "confirmed witness violations fail those claims" not in challenge
    for text in (fixer, mechanism, obligations, load_section("conformance_audit"), challenge):
        assert "\u2014" not in text


# ----- 4. finite states in the construction harness -------------------------------------------------------------------

def test_nan_in_the_carried_state_fails_a_size_even_when_argmax_answers_right():
    """A witness whose state turns NaN still predicts through argmax (NumPy returns a token for NaN logits). A copy
    witness that answers every query from its finite part but carries a NaN is not a P2 success."""
    poisoned = COPY.replace("    new['h'] = h\n", "    new['h'] = h\n    new['norm'] = np.array([np.nan])\n")
    assert poisoned != COPY
    spec = dict(FIX["requests"]["O6.a"]["spec"], count=20, sizes=[8])
    result = spec_checks.audited_construction(spec, poisoned, FIX["requests"]["O6.a"]["architecture"])
    assert result["status"] == "violated"
    row = result["results"][0]
    assert not row["passed"]
    assert any("state['norm'] holds nan" in v for v in row["resource_violations"])
    clean = spec_checks.audited_construction(spec, COPY, FIX["requests"]["O6.a"]["architecture"])
    assert clean["status"] == "discharged"


def test_argmax_over_nan_logits_is_a_token_and_the_check_reads_the_state_first():
    assert int(np.argmax(np.full(4, np.nan))) == 0
    assert constructions.nonfinite({"h": np.zeros(3), "t": 4}) is None
    assert "inf" in constructions.nonfinite({"A": [np.zeros(2), np.array([1.0, np.inf])]})
    assert constructions.nonfinite({"x": float("nan")}) == "state['x'] is nan"
    assert constructions.nonfinite({"k": np.array([1, 2, 3])}) is None
