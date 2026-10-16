"""FIX-15 regressions: a precision deviation in a witness can be judged for materiality. The conforming edit restores
the frozen contract's precision by changing dtype casts and dtype arguments only, and the host replays it under the
frozen check (results/phase4-20261013T073748Z/FIX-15.md).

Fixture fixtures/calibration_math_20261015T053026Z/k2_run2_precision.json, written read-only by
results/phase4-20261013T073748Z/fix15-staging/make_fixture.py from runs/calibration-math-20261015T053026Z/k2-run2,
with the sha256 of every file it read: the card, the frozen record (architecture precision_bits 32, D2 "binary32"),
the stage M checks (the O9.a composition and O9.b binding witnesses, which compute in float64), the recorded
challenger finding "extra intermediate precision in composition and binding" (targets with no edit, because the
FIX-13 rules refused any change of precision), its recorded stage C items C-c13769a0 and C-ff8f3fa8 (unresolved), and
the stage M job requests of both witnesses.

Stage C runs for real on the recorded stage M, with the recorded finding restated with precision-restoring edits;
every finding verification and every audited construction runs on the host executor.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hybrid.math import findings, spec_checks
from hybrid.math.runner import MathRunner
from hybrid.research.ledger import Ledger
from hybrid.research.stages import c_challenge, m_math
from hybrid.research.stages import math_contracts as mc
from hybrid.research.stages.common import Budget, StageContext, load_section
from hybrid.research.store import Store
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research import test_fix13_materiality as fix13
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM, orch_config

FIX = json.loads((Path(__file__).parent / "fixtures" / "calibration_math_20261015T053026Z" / "k2_run2_precision.json")
                 .read_text(encoding="utf-8"))
CARD_ID = FIX["card"]["id"]
FINDING = FIX["finding"]
SUBJECT = FINDING["subject"]
CONTRACT = FIX["record"]["profile"]["architecture"]
SOURCES = {sid: FIX["m_checks"][sid]["text"] for sid in ("O9.a", "O9.b")}
# sha256 of the recorded sources, as the recorded stage C materiality records give them.
SOURCE_SHA = {"O9.a": "d14e5c66444864e646f7acde2159a1390a1ca0306fa69016abeade6a982f88da",
              "O9.b": "e6db3bf7dbb2cfee9caa43bdecbe6bea33465d724727f9319d807c9db16b3878"}
DEVIATION = "e = p['E'][int(token)].astype(np.float64)"
HEAD, TAIL = "    e = p['E'][int(token)]", "    return int(np.argmax(logits))"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def body(sid: str) -> str:
    """The body of update after the parameter lookup, and predict up to its return: every float64 the witness
    computes in."""
    source = SOURCES[sid]
    return source[source.index(HEAD):source.index(TAIL)]


def binary32(sid: str, *, change=lambda text: text) -> dict:
    """The precision-restoring target: one edit whose anchor is that body and whose replacement casts to float32."""
    anchor = body(sid)
    return mf.target(sid, DEVIATION, [(anchor, change(anchor.replace("np.float64", "np.float32")))],
                     why="restores the frozen binary32 execution and changes nothing else")


def restored(sid: str) -> str:
    source = SOURCES[sid]
    return source.replace(body(sid), body(sid).replace("np.float64", "np.float32"))


def variant(target: dict, bits=32):
    return findings.conforming_variant(SOURCES[target["statement_id"]], target["deviation"], target["edits"], bits)


def finding(*targets, **changes) -> dict:
    """The recorded finding as the challenger wrote it, with the given targets."""
    return dict(copy.deepcopy(FINDING), targets=[copy.deepcopy(t) for t in targets], **changes)


def piece_id(sid: str, oid: str = "O9") -> str:
    return c_challenge.finding_id(FIX["record"], f"{SUBJECT} {sid}", [oid])


class Replay:
    """Stage C on the recorded k2-run2 stage M: the frozen record and the M verdict are stored as stage M stored
    them; the challenger gives ``findings`` in the batch that shows O9.a and O9.b, as it did, and nothing in the
    others; every statement matches."""

    def __init__(self, tmp_path, findings_=()):
        self.run_dir = (tmp_path / "run").resolve()
        self.ledger = Ledger(self.run_dir)
        self.store = Store(self.run_dir, self.ledger)
        self.store.put_card(dict(copy.deepcopy(FIX["card"]), version=1))
        self.executor = mf.MathExecutor(tmp_path / "exec")
        batches = c_challenge.check_batches(FIX["m_checks"])
        where = next(i for i, batch in enumerate(batches) if "O9.a" in batch and "O9.b" in batch)
        replies = [mf.challenger_output(findings=list(findings_)) if i == where else mf.challenger_output()
                   for i in range(len(batches))]
        script = {"challenger": replies, "statement_backtranslate": mf.backtranslate_output,
                  "statement_match": mf.match_output()}
        self.llm = FakeLLM(self.run_dir, self.ledger, script,
                           models={"challenger": "gpt-6.1-sol", "statement_match": "qwen3.6-35b-a3b-mlx"})
        self.ctx = StageContext(run_id=self.run_dir.name, run_dir=self.run_dir, ledger=self.ledger, store=self.store,
                                llm=self.llm, runner=None, bench=None, cfg=orch_config(),
                                budget=Budget(1, 500, 10**9, 3600.0), lenses={},
                                math=MathRunner(self.executor, mf.IMAGE, self.run_dir))
        rec = copy.deepcopy(FIX["record"])
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

    def findings(self) -> dict:
        challenge = [json.loads((self.run_dir / e["data"]["path"]).read_text())
                     for e in self.ledger.iter_events("artifact") if e["data"]["kind"] == "challenge"][-1]
        return {f["id"]: f for f in challenge["findings"] if f.get("parent")}

    def check_artifact(self, name: str) -> dict:
        return json.loads((self.run_dir / "artifacts" / "math_check" / name).read_text())

    def construction_requests(self) -> list[dict]:
        return [r for r in self.executor.requests() if r["kind"] == "audited_construction"]


# ----- 1. the recorded run ---------------------------------------------------------------------------------------------

def test_the_fixture_is_the_recorded_run_and_its_unresolved_outcome():
    for sid in ("O9.a", "O9.b"):
        assert sha(SOURCES[sid]) == SOURCE_SHA[sid] == FIX["provenance"]["source_sha256"][sid]
        assert FIX["requests"][sid]["source"] == SOURCES[sid]
        assert FIX["requests"][sid]["architecture"] == CONTRACT == {"hidden": "d", "precision_bits": 32}
        assert SOURCES[sid].count("np.float64") == 9 and DEVIATION in SOURCES[sid]
    assert FIX["requests"]["O9.a"]["spec"]["task"] == "composition"
    assert FIX["requests"]["O9.a"]["spec"]["sizes"] == [4, 16, 32]
    assert FIX["requests"]["O9.b"]["spec"]["task"] == "binding"
    d2 = next(d for d in FIX["record"]["definitions"] if d["id"] == "D2")
    assert "binary32" in d2["text"] and "additional precision" in d2["text"]
    assert "D2" in next(o for o in FIX["record"]["obligations"] if o["id"] == "O9")["definitions"]
    # The challenger gave no edit because the FIX-13 rules refused any change of precision.
    assert [t["edits"] for t in FINDING["targets"]] == [[], []]
    assert all("explicitly prohibit" in t["why"] for t in FINDING["targets"])
    recorded = FIX["recorded_c"]
    for sid, fid in (("O9.a", "C-c13769a0"), ("O9.b", "C-ff8f3fa8")):
        item = recorded["items"][sid]
        assert item["id"] == fid and item["disposition"] == "unresolved"
        assert item["materiality"]["problems"] == ["no edit is given"]
        assert item["verification"]["status"] == "confirmed"
    assert FIX["m_claims"] and {c["claim_id"]: c["status"] for c in FIX["m_claims"]}["P2.composition"] == "observed"
    assert recorded["claims"]["P2.composition"] == "conditional"


# ----- 2. the precision-restoring edit is accepted and replayed --------------------------------------------------------

@pytest.mark.parametrize("sid", ["O9.a", "O9.b"])
def test_a_precision_restoring_edit_of_the_recorded_witness_is_accepted(sid):
    text, problems = variant(binary32(sid))
    assert problems == [] and text == restored(sid)
    assert "np.float64" not in text and findings.wider_precision_left(text, 32) == []
    assert findings.wider_precision_left(SOURCES[sid], 32) and findings.edit_kind(binary32(sid)["edits"], 32) == \
        "precision"
    # The same restoration as six line edits: casts set to float32, or a cast of a float32 array dropped.
    lines = [(DEVIATION, "e = p['E'][int(token)]"), ("yy = y.astype(np.float64)", "yy = y"),
             ("pre[g] = p['Wy'][g].astype(np.float64) @ yy + p['We'][g].astype(np.float64) @ e + "
              "p['b'][g].astype(np.float64)", "pre[g] = p['Wy'][g] @ yy + p['We'][g] @ e + p['b'][g]"),
             ("sn = r * s.astype(np.float64) + i * u", "sn = r * s.astype(np.float32) + i * u"),
             ("yn = o * np.tanh(sn32.astype(np.float64))", "yn = o * np.tanh(sn32)"),
             ("logits = y.astype(np.float64) @ p['O'].astype(np.float64)", "logits = y @ p['O']")]
    text, problems = variant(mf.target(sid, DEVIATION, lines))
    assert problems == [] and findings.wider_precision_left(text, 32) == []


def test_the_restored_witness_computes_in_binary32():
    """Every value update and predict compute is float32 once the casts are restored (Python float constants are
    weak under NumPy 2 and do not promote)."""
    text, _ = variant(binary32("O9.a"))
    seen = []
    probe = text.replace("    r = sg(pre['r'])\n", "    SEEN.extend([pre['r'].dtype, sg(pre['r']).dtype, "
                                                       "np.tanh(pre['u']).dtype])\n    r = sg(pre['r'])\n")
    probe = probe.replace(TAIL, "    SEEN.append(logits.dtype)\n" + TAIL)
    namespace = {"np": np, "SEEN": seen}
    exec(compile(probe, "<probe>", "exec"), namespace)
    state = namespace["init"]({"size": 4})
    for token in (0, 4, 13, 14):
        state = namespace["update"](state, token)
    namespace["predict"](state)
    assert seen and set(seen) == {np.dtype(np.float32)}


@pytest.mark.parametrize("sid, sizes", [("O9.a", "T=4, T=16, T=32"), ("O9.b", "hops=1, hops=4, hops=8")])
def test_the_recorded_witness_restored_to_binary32_holds_at_the_frozen_sizes(sid, sizes):
    """The materiality answer for k2-run2: at the frozen binary32 precision both O9 witnesses still meet 0.99 at
    every frozen size (count, sizes and threshold as frozen), so the deviation is immaterial."""
    request = FIX["requests"][sid]
    text, _ = variant(binary32(sid))
    result = spec_checks.audited_construction(request["spec"], text, request["architecture"])
    assert result["status"] == "discharged", result["evidence"]
    assert f"observed on the listed sizes: {sizes}" in result["evidence"]
    assert all(row["passed"] and row["accuracy"] >= 0.99 for row in result["results"])


def test_stage_c_replays_the_restored_witnesses_and_p2_composition_keeps_its_evidence(tmp_path):
    replay = Replay(tmp_path, [finding(binary32("O9.a"), binary32("O9.b"))])
    replay.run()
    pieces = replay.findings()
    assert set(pieces) == {piece_id("O9.a"), piece_id("O9.b")}
    for sid in ("O9.a", "O9.b"):
        item = pieces[piece_id(sid)]
        assert item["disposition"] == "immaterial" and item["used_by"] == ["O9"], item["reason"]
        assert item["verification"]["status"] == "confirmed"
        record = item["materiality"]
        assert record["outcome"] == "immaterial" and record["edit_kind"] == "precision"
        assert record["precision_bits"] == 32 and record["wider_precision_left"] == []
        assert record["original_sha256"] == SOURCE_SHA[sid] and record["variant_source"] == restored(sid)
        assert record["variant_sha256"] == sha(restored(sid))
        check = replay.check_artifact(record["check"]["artifact"])
        assert check["stage"] == "C" and check["purpose"] == "materiality" and check["statement_id"] == sid
        assert check["text"] == restored(sid) and check["decided"]
    assert replay.statuses()["P2.composition"] == "observed"
    assert "obligation_failed:P2.composition" not in replay.verdict()["kill_codes"]
    # The replays ran under the frozen statements and the frozen binary32 contract, unchanged.
    frozen = {s["id"]: s["spec"] for s in m_math.all_statements(FIX["record"])}
    replays = [r for r in replay.construction_requests() if r["source"] in (restored("O9.a"), restored("O9.b"))]
    assert {r["source"] for r in replays} == {restored("O9.a"), restored("O9.b")}
    for request in replays:
        sid = "O9.a" if request["source"] == restored("O9.a") else "O9.b"
        assert request["spec"] == frozen[sid] and request["architecture"] == CONTRACT


def test_the_recorded_finding_without_an_edit_stays_unresolved(tmp_path):
    replay = Replay(tmp_path, [finding(*FINDING["targets"])])
    replay.run()
    for sid in ("O9.a", "O9.b"):
        item = replay.findings()[piece_id(sid)]
        assert item["disposition"] == "unresolved" and item["materiality"]["problems"] == ["no edit is given"]
        assert item["materiality"]["edit_kind"] == "narrow" and item["materiality"]["precision_bits"] == 32
    assert replay.statuses()["P2.composition"] == "conditional"


# ----- 3. edits that are not a precision restoration are refused -------------------------------------------------------

def swap(sid: str, old: str, new: str) -> dict:
    return binary32(sid, change=lambda text: text.replace(old, new))


@pytest.mark.parametrize("target, problem", [
    # Toward another precision than the frozen one.
    (mf.target("O9.a", DEVIATION, [(body("O9.a"), body("O9.a").replace("np.float64", "np.float16"))]),
     "it sets 16-bit precision"),
    (mf.target("O9.a", DEVIATION, [(DEVIATION, "e = p['E'][int(token)].astype(np.longdouble)")]),
     "it sets 128-bit precision"),
    (mf.target("O9.a", DEVIATION, [(DEVIATION, "e = p['E'][int(token)].astype('f2')")]), "it sets 16-bit"),
    # Dropping a cast to the frozen precision.
    (mf.target("O9.a", "sn32 = sn.astype(np.float32)", [("sn32 = sn.astype(np.float32)", "sn32 = sn")]),
     "drops a 32-bit cast"),
    # Precision mixed with other changes in one edit: a weight, an operation, a new name, a reordering.
    (swap("O9.a", "sn32 = sn.astype", "sn32 = (2.0 * sn).astype"), "changes more than float dtype casts"),
    (swap("O9.a", " @ yy", " * yy"), "changes more than float dtype casts"),
    (swap("O9.a", "u = np.tanh(pre['u'])", "u = np.clip(np.tanh(pre['u']), -1.0, 1.0)"),
     "changes more than float dtype casts"),
    (swap("O9.a", "    o = sg(pre['o'])\n", ""), "a precision edit keeps the lines"),
    (swap("O9.a", "    o = sg(pre['o'])\n", "    o = pre['o']\n"), "changes more than float dtype casts"),
    (swap("O9.a", "u = np.tanh(pre['u'])", "u = np.float32(np.tanh, pre['u'])"), "changes more than float dtype"),
    # Precision mixed with a narrow edit in one target.
    (mf.target("O9.a", DEVIATION, [(DEVIATION, "e = p['E'][int(token)].astype(np.float32)"),
                                   ("    o = sg(pre['o'])\n", "    o = sg(pre['o'] )\n")]),
     "mix a restoration of the frozen 32-bit precision with other changes"),
    # A restoration that leaves the named deviation in place.
    (mf.target("O9.a", DEVIATION, [("yy = y.astype(np.float64)", "yy = y.astype(np.float32)")]),
     "still contains the deviation"),
])
def test_an_edit_that_is_not_a_precision_restoration_is_refused(target, problem):
    text, problems = variant(target)
    assert text is None and any(problem in p for p in problems), problems


def test_without_a_frozen_precision_a_precision_edit_is_held_to_the_fix13_rules():
    text, problems = variant(binary32("O9.a"), bits=None)
    assert text is None
    assert any("changes the precision names of its anchor" in p for p in problems)
    assert any("hold no part of the deviation" in p for p in problems)


def test_a_dtype_argument_and_a_trailing_constructor_dtype_count_as_casts():
    source = "def f(n):\n    a = np.zeros(n, np.float64)\n    b = np.ones(n, dtype='float64')\n    return a + b\n"
    edits = [{"anchor": "np.zeros(n, np.float64)", "replacement": "np.zeros(n, np.float32)"},
             {"anchor": "np.ones(n, dtype='float64')", "replacement": "np.ones(n, dtype='float32')"}]
    text, problems = findings.conforming_variant(source, "np.float64", edits, 32)
    assert problems == [] and "float64" not in text
    added = [{"anchor": "a + b", "replacement": "np.float32(a + b)"}]
    text, problems = findings.conforming_variant(source, "return a + b", added, 32)
    assert problems == [] and text.endswith("return np.float32(a + b)\n")
    # finfo gives numbers the witness uses; changing its dtype is not a cast.
    eps = "def g(x):\n    return x + np.finfo(np.float64).eps\n"
    text, problems = findings.conforming_variant(eps, "np.finfo(np.float64)", [
        {"anchor": "np.finfo(np.float64)", "replacement": "np.finfo(np.float32)"}], 32)
    assert text is None and any("changes more than float dtype casts" in p for p in problems)


def test_each_dtype_change_is_judged_where_it_stands():
    """A float16 cast dropped on one line is not excused by a float32 cast set elsewhere."""
    source = "def f(a, b):\n    x = a.astype(np.float16) + b.astype(np.float64)\n    return x\n"
    edits = [{"anchor": "a.astype(np.float16) + b.astype(np.float64)", "replacement": "a + b.astype(np.float32)"}]
    text, problems = findings.conforming_variant(source, "b.astype(np.float64)", edits, 32)
    assert text is None and any("drops a narrower cast" in p for p in problems), problems
    edits = [{"anchor": "a.astype(np.float16) + b.astype(np.float64)",
              "replacement": "a.astype(np.float32) + b.astype(np.float32)"}]
    text, problems = findings.conforming_variant(source, "b.astype(np.float64)", edits, 32)
    assert problems == [] and text.count("np.float32") == 2


def test_a_precision_edit_toward_another_dtype_leaves_the_witness_unresolved_and_runs_nothing(tmp_path):
    half = mf.target("O9.a", DEVIATION, [(body("O9.a"), body("O9.a").replace("np.float64", "np.float16"))])
    replay = Replay(tmp_path, [finding(half, binary32("O9.b"))])
    replay.run()
    item = replay.findings()[piece_id("O9.a")]
    assert item["disposition"] == "unresolved" and item["materiality"]["check"] is None
    assert any("it sets 16-bit precision" in p for p in item["materiality"]["problems"])
    assert replay.findings()[piece_id("O9.b")]["disposition"] == "immaterial"
    assert replay.statuses()["P2.composition"] == "conditional"
    assert not any("np.float16" in r["source"] for r in replay.construction_requests())


# ----- 4. the FIX-13 cases still hold ----------------------------------------------------------------------------------

def test_the_fix13_variants_and_refusals_hold_under_the_recorded_float64_contract():
    """The k1-run1 contract is float64. The no-max variants are narrow edits under it, and the FIX-13 refusals
    stand, the longdouble one now as a precision edit toward another precision than the frozen one."""
    bits = fix13.FIX["requests"]["O7.a"]["architecture"]["precision_bits"]
    assert bits == 64
    for source, target in ((fix13.COPY, fix13.COPY_TARGET), (fix13.RECALL, fix13.RECALL_TARGET)):
        text, problems = findings.conforming_variant(source, target["deviation"], target["edits"], bits)
        assert problems == [] and text == fix13.FIX["no_max"][target["statement_id"]]
        assert findings.edit_kind(target["edits"], bits) == "narrow"
    retuned = fix13.RETUNED
    text, problems = findings.conforming_variant(fix13.RECALL, retuned["deviation"], retuned["edits"], bits)
    assert text is None and any("adds numbers [100.0]" in p for p in problems)
    more = [{"anchor": "    s = s - s.max()\n", "replacement": "    s = s.astype(np.longdouble)\n"}]
    text, problems = findings.conforming_variant(fix13.RECALL, "s = s - s.max()", more, bits)
    assert text is None and any("it sets 128-bit precision" in p for p in problems)


# ----- 5. the challenger is told ---------------------------------------------------------------------------------------

def test_the_challenger_is_told_precision_restoring_edits_are_allowed_and_expected():
    challenge = load_section("challenge_findings")
    assert "PRECISION DEVIATIONS (FIX-15)" in challenge
    assert "the conforming edit restores the frozen precision, and you are expected to give it" in challenge
    assert "A precision deviation always has one." in challenge
    assert "do not move to a precision other than the frozen one" in challenge
    assert "\u2014" not in challenge
