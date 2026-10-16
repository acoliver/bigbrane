"""The s3.mode switch (run.toml): "judges" runs S3 as every run before Phase 3 did; "math" runs stages M and C in its
place. A resumed run keeps the mode it started with. Plumb runs M and C under the operator override, and inspect
reports tiers, definitions, the premise ledger and kill codes."""
import asyncio
import tomllib

import pytest

from hybrid.math.runner import MathRunner
from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.plumb import Deps, PlumbError, parse_stages, plumb
from hybrid.research.stages.common import MATH_ORDER, STAGE_ORDER, pipeline_order
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_orchestrator_fakes import FakeBench, FakeLLM, make_orchestrator, orch_config
from hybrid.tests.research.test_plumb import CARD as SOURCE_CARD
from hybrid.tests.research.test_plumb import build_source_run
from hybrid.bench.local import LocalExecutor
from hybrid.executor.lock import read_lock

RUN_TOML = ROOT / "hybrid" / "config" / "run.toml"


def run(orch):
    return asyncio.run(orch.run())


def only_card(orch):
    (card_id,) = orch.store.list_cards()
    return orch.store.get_card(card_id)


def events(run_dir, type):
    return [e["data"] for e in Ledger(run_dir).iter_events(type)]


def math_runner(tmp_path, run_dir):
    return MathRunner(mf.MathExecutor(tmp_path / "exec"), mf.IMAGE, run_dir)


def math_cfg(**overrides):
    cfg = orch_config(**overrides)
    cfg["s3"]["mode"] = "math"
    cfg["math"]["max_prover_attempts"] = 2
    return cfg


@pytest.fixture
def run_dir(tmp_path):
    return (tmp_path / "run-20261012T000000Z").resolve()


# ----- the config switch ------------------------------------------------------------------------------------------

def test_shipped_config_uses_the_judges_and_a_config_without_the_key_does_too(tmp_path):
    assert load_run_config(RUN_TOML)["s3"]["mode"] == "judges"
    text = RUN_TOML.read_text().replace('mode = "judges"\n', "")
    old = tmp_path / "old.toml"
    old.write_text(text)
    assert "mode" not in tomllib.loads(text)["s3"]
    cfg = load_run_config(old)
    # run.toml also turns on the refinement dialogue R (Phase 7), which follows S1 in either mode
    assert cfg["s3"]["mode"] == "judges" and pipeline_order(cfg) == STAGE_ORDER[:2] + ("R",) + STAGE_ORDER[2:]
    # run.toml also holds every card to the transformer profile (Phase 4, decision D1), and sizes the host-wide job
    # slots from the Podman VM (hybrid/slots.py)
    assert cfg["math"] == {"max_prover_attempts": 4, "max_obligations": 12, "max_statements": 16,
                           "match_instances": 1000, "profile": True, "max_parallel_jobs": "auto"}
    # a config from before the job slots existed has no max_parallel_jobs and gets the default filled in
    text = text.replace('max_parallel_jobs = "auto"\n', "")
    older = tmp_path / "older.toml"
    older.write_text(text)
    assert "max_parallel_jobs" not in tomllib.loads(text)["math"]
    assert load_run_config(older)["math"]["max_parallel_jobs"] == "auto"


def test_math_mode_and_bad_values(tmp_path):
    math = tmp_path / "math.toml"
    math.write_text(RUN_TOML.read_text().replace('mode = "judges"', 'mode = "math"'))
    assert MATH_ORDER == ("S0", "S1", "S2", "M", "C", "S4", "S5", "S6")
    assert pipeline_order(load_run_config(math)) == ("S0", "S1", "R", "S2", "M", "C", "S4", "S5", "S6")
    assert pipeline_order(load_run_config(math, {"refine.enabled": False})) == MATH_ORDER
    bad = tmp_path / "bad.toml"
    bad.write_text(RUN_TOML.read_text().replace('mode = "judges"', 'mode = "maths"'))
    with pytest.raises(ValueError, match="s3.mode must be one of"):
        load_run_config(bad)
    zero = tmp_path / "zero.toml"
    zero.write_text(RUN_TOML.read_text().replace("max_prover_attempts = 4", "max_prover_attempts = 0"))
    with pytest.raises(ValueError, match="math.max_prover_attempts"):
        load_run_config(zero)


def test_math_mode_without_a_math_runner_fails_fast(run_dir):
    with pytest.raises(ValueError, match="needs a math runner"):
        make_orchestrator(run_dir, cfg=math_cfg())


# ----- a whole line in math mode ------------------------------------------------------------------------------------

def test_math_mode_runs_m_and_c_in_place_of_s3_and_inspect_reports_them(run_dir, tmp_path, capsys):
    orch, llm = make_orchestrator(run_dir, cfg=math_cfg(), script=mf.math_script(),
                                  math=math_runner(tmp_path, run_dir))
    assert run(orch)["reason"] == "complete"
    card = only_card(orch)
    assert card["status"]["stage"] == "S6"
    assert {s: v["verdict"] for s, v in card["status"]["verdicts"].items()} == {
        s: "pass" for s in ("S0", "S1", "S2", "M", "C", "S4", "S5", "S6")}
    prompts = llm.prompts()
    assert not {"formalizer", "disprover", "obligation_checker", "feasibility_judge_a"} & set(prompts)
    # M reads each statement back and matches it before the freeze (FIX-19), so the first back-translation and
    # match come right after the fixer and before the prover; C repeats both after the challenger.
    challenged = prompts.index("challenger")
    first_read, last_read = prompts.index("statement_backtranslate"), prompts.index("statement_backtranslate", challenged)
    first_match = prompts.index("statement_match")
    last_match = len(prompts) - 1 - prompts[::-1].index("statement_match")
    assert prompts.index("statement_fixer") < first_read < first_match < prompts.index("prover") \
        < challenged < last_read < last_match < prompts.index("builder")
    start = events(run_dir, "run_start")[0]
    assert start["s3_mode"] == "math" and start["stages"] == list(MATH_ORDER)
    queued = [e["stage"] for e in events(run_dir, "queued")]
    assert queued == ["S1", "S2", "M", "C", "S4", "S5", "S6"]
    assert inspect_run(run_dir) == 0
    out = capsys.readouterr().out
    assert "Math stages M and C: 1 statement sets frozen" in out and f"{card['id']} C pass" in out
    assert "claim capability.2: observed (executed-on-examples)" in out
    dossier = (run_dir / "report" / "dossiers" / f"{card['id']}.html").read_text()
    assert "Math stages (M and C)" in dossier and "lean-proven" in dossier and "not in the ledger" not in dossier


def test_a_card_that_m_leaves_unproven_never_reaches_c(run_dir, tmp_path):
    fixer = mf.fixer_output(premises=[mf.premise("A1", "an oracle segments the input", plan=None,
                                                 does_work=False)],
                            capability_1_premises=["A1"])
    orch, llm = make_orchestrator(run_dir, cfg=math_cfg(), script=mf.math_script(fixer=fixer),
                                  math=math_runner(tmp_path, run_dir))
    run(orch)
    card = only_card(orch)
    assert card["status"]["stage"] == "M" and card["status"]["verdicts"]["M"]["verdict"] == "undetermined"
    assert "challenger" not in llm.prompts()
    assert "math gate (M): undetermined; kill codes unproven:A1" in card["status"]["verdicts"]["M"]["reasons"]


def test_enqueue_refuses_s4_without_passes_at_m_and_c(run_dir, tmp_path):
    orch, _ = make_orchestrator(run_dir, cfg=math_cfg(), script=mf.math_script(fixer=mf.fixer_output(
        premises=[mf.premise("A1", "an oracle", plan=None)], capability_1_premises=["A1"])),
        math=math_runner(tmp_path, run_dir))
    run(orch)
    card = only_card(orch)
    orch.queues = {stage: asyncio.Queue() for stage in orch.order[1:]}
    with pytest.raises(AssertionError):
        orch._enqueue("C", card["id"])
    with pytest.raises(AssertionError):
        orch._enqueue("S4", card["id"])


# ----- resume keeps the mode ----------------------------------------------------------------------------------------

def test_a_judges_run_resumed_in_math_mode_is_refused(run_dir, tmp_path):
    orch, _ = make_orchestrator(run_dir, cfg=orch_config(budgets__max_calls=3))
    run(orch)
    with pytest.raises(ValueError, match="started with s3.mode = 'judges'"):
        make_orchestrator(run_dir, cfg=math_cfg(), resume=True, math=math_runner(tmp_path, run_dir))


def test_a_run_from_before_phase_3_resumes_with_the_judges(run_dir, tmp_path):
    Ledger(run_dir).append("run_start", "orchestrator", {"run_id": run_dir.name, "resume": False})
    with pytest.raises(ValueError, match="started with s3.mode = 'judges'"):
        make_orchestrator(run_dir, cfg=math_cfg(), resume=True, math=math_runner(tmp_path, run_dir))
    orch, _ = make_orchestrator(run_dir, cfg=orch_config(), resume=True)
    assert orch.order == STAGE_ORDER


def test_a_math_run_resumes_in_math_mode_only_and_continues_from_its_last_verdict(run_dir, tmp_path):
    # explorer, critic, prior art, reduction critic, fixer, the pre-freeze back-translation and match (FIX-19),
    # two prover calls (the Lean statement and the construction); the challenger call is refused.
    orch, _ = make_orchestrator(run_dir, cfg=math_cfg(budgets__max_calls=9), script=mf.math_script(),
                                math=math_runner(tmp_path, run_dir))
    assert run(orch)["reason"].startswith("max_calls")
    card = only_card(orch)
    assert card["status"]["stage"] == "M" and card["status"]["verdicts"]["M"]["verdict"] == "pass"
    with pytest.raises(ValueError, match="started with s3.mode = 'math'"):
        make_orchestrator(run_dir, cfg=orch_config(), resume=True)
    orch2, llm2 = make_orchestrator(run_dir, cfg=math_cfg(), resume=True, script=mf.math_script(),
                                    math=math_runner(tmp_path / "second", run_dir))
    summary = run(orch2)
    assert summary["requeued_at_start"] == {"C": [card["id"]]}
    assert llm2.prompts()[0] == "challenger" and "statement_fixer" not in llm2.prompts()
    assert only_card(orch2)["status"]["stage"] == "S6"
    assert Ledger(run_dir).verify() == (True, None)


# ----- plumb ------------------------------------------------------------------------------------------------------

def test_plumb_stage_lists_for_the_math_stages():
    assert parse_stages("M,C") == ("M", "C")
    assert parse_stages("m,c,s4,s5,s6") == ("M", "C", "S4", "S5", "S6")
    with pytest.raises(PlumbError, match="needs M first"):
        parse_stages("C,S4")
    with pytest.raises(PlumbError):
        parse_stages("S3,M")


def test_plumb_runs_m_then_c_whatever_m_says(tmp_path):
    source = build_source_run(tmp_path / "runs" / "run-20261010T181300Z")
    run_dir = (tmp_path / "runs" / "plumb-20261012T000000Z").resolve()
    fixer = mf.fixer_output(premises=[mf.premise("A1", "an oracle segments the input", plan=None,
                                                 does_work=False)],
                            capability_1_premises=["A1"])

    def factory(ledger):
        directory = ledger.run_dir.resolve()
        return Deps(FakeLLM(directory, ledger, mf.math_script(fixer=fixer)),
                    LocalExecutor(directory.name, directory / "jobs" / "receipts"), FakeBench(),
                    read_lock()["reference"], math_runner(tmp_path, directory))

    summary = asyncio.run(plumb(source_run=source, card_id=SOURCE_CARD, run_dir=run_dir, reason="calibrate M and C",
                                cfg=orch_config(), deps_factory=factory, stages=("M", "C")))
    assert summary["verdicts"] == {"M": "undetermined", "C": "undetermined"} and summary["completed"]
    math = summary["math"]
    assert math["stages"]["M"]["kill_codes"] == ["unproven:A1"]
    assert math["stages"]["C"]["premises"] == {"A1": "open"} and math["stages"]["C"]["definitions"] == []
    # The fixer is asked once more because A1 has no measurement plan.
    # Two prover calls: the Lean statement and the construction are asked separately.
    # statement_match: a back-translation and a comparison before the freeze in M (FIX-19; the fixer's second
    # round reuses them, so no more calls) and the same two at C.
    assert math["role_calls"] == {"statement_fixer": 2, "prover": 2, "challenger": 1, "statement_match": 4}
    override = events(run_dir, "operator_override")[0]
    assert override["stages_run"] == ["M", "C"] and "C runs after M" in override["math_note"]
    assert inspect_run(run_dir) == 0


def test_plumb_without_a_math_runner_refuses_m(tmp_path):
    source = build_source_run(tmp_path / "runs" / "run-20261010T181300Z")

    def factory(ledger):
        directory = ledger.run_dir.resolve()
        return Deps(FakeLLM(directory, ledger), LocalExecutor(directory.name, directory / "jobs" / "receipts"),
                    FakeBench(), read_lock()["reference"])

    with pytest.raises(RuntimeError, match="need a math runner"):
        asyncio.run(plumb(source_run=source, card_id=SOURCE_CARD, run_dir=tmp_path / "runs" / "p2", reason="r",
                          cfg=orch_config(), deps_factory=factory, stages=("M", "C")))
