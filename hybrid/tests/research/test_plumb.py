"""`python -m hybrid.research plumb`: S4 to S6 on a card the gates stopped, labeled as an operator override.

The stages run for real against fakes (scripted LLM, LocalExecutor, fake bench), the same ones the
orchestrator tests use.
"""
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from hybrid.bench.local import LocalExecutor
from hybrid.executor.lock import read_lock
from hybrid.llm.errors import TransportError
from hybrid.research import plumb as plumb_module
from hybrid.research.__main__ import build_parser, main
from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT
from hybrid.research.plumb import STATEMENT, Deps, PlumbError, plumb
from hybrid.research.store import Store
from hybrid.tests.research.builders import make_card
from hybrid.tests.research.test_inspect import build_run as build_normal_run
from hybrid.tests.research.test_orchestrator_fakes import FakeBench, FakeLLM, orch_config

CARD = "card-537e161c"
REASON = "plumbing check of S4-S6; card was reduced to linear_attention at S2"
PAGES = ["index.html", "cards.html", "lineage.html", "models.html", f"dossiers/{CARD}.html"]
REDUCTION = {"reduces_to": "linear_attention", "mapping": "k = shifted token", "confidence": 0.9,
             "difference_if_any": "", "radical_enough": False, "why": "The update is a delta rule.",
             "reduction_outcome": "verified", "textbook_transplant": True, "textbook_why": "textbook EDMD"}


def build_source_run(run_dir: Path) -> Path:
    """A source run holding one card that S2 reduced, with a reduction_critic artifact, as the loop writes it."""
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    store.put_card(make_card(CARD, title="Koopman Power Readout"))
    store.set_verdict(CARD, "S0", "pass", ["card validates"], ["call-aaaaaaaaaaaa"])
    store.set_verdict(CARD, "S1", "pass", ["gate: passed with 1 residual finding"], ["call-bbbbbbbbbbbb"])
    store.put_artifact("reduction_critic", f"{CARD}-v3-abc123.json", json.dumps(REDUCTION).encode(), card_id=CARD)
    store.set_verdict(CARD, "S2", "reduces_to_known", ["reduction claim linear_attention (verified)"],
                      ["call-cccccccccccc", "job-0123456789ab"])
    ledger.append("run_end", "orchestrator", {"reason": "complete"})
    return run_dir


def tree_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def fake_factory(script=None, fail=None, bench=None):
    def build(ledger):
        run_dir = ledger.run_dir.resolve()
        return Deps(FakeLLM(run_dir, ledger, script, fail), LocalExecutor(run_dir.name, run_dir / "jobs" / "receipts"),
                    bench or FakeBench(), read_lock()["reference"])
    return build


@pytest.fixture
def source_run(tmp_path) -> Path:
    return build_source_run(tmp_path / "runs" / "run-20261010T181300Z")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    return tmp_path / "runs" / "plumb-20261010T200000Z"


def do_plumb(source_run, run_dir, **kwargs):
    kwargs.setdefault("deps_factory", fake_factory())
    return asyncio.run(plumb(source_run=source_run, card_id=CARD, run_dir=run_dir, reason=REASON,
                             cfg=orch_config(), **kwargs))


def all_events(run_dir):
    return list(Ledger(run_dir).iter_events())


# ----- the record -------------------------------------------------------------------------------


def test_override_event_is_first_and_complete(source_run, run_dir):
    summary = do_plumb(source_run, run_dir, routing={"table": "gateway_claude_local"})
    assert summary["completed"] and summary["verdicts"] == {"S4": "pass", "S5": "pass", "S6": "pass"}
    events = all_events(run_dir)
    first = events[0]
    assert (first["seq"], first["type"], first["actor"]) == (1, "operator_override", "operator")
    data = first["data"]
    assert data["statement"] == STATEMENT
    assert data["reason"] == REASON
    assert data["source_run"] == str(source_run.resolve()) and data["source_run_id"] == source_run.name
    assert data["card_id"] == CARD
    assert data["stages_run"] == ["S4", "S5", "S6"]
    assert data["routing"] == {"table": "gateway_claude_local"}
    verdicts = data["source_verdicts"]
    assert {stage: (v["verdict"] if v else None) for stage, v in verdicts.items()} == {
        "S0": "pass", "S1": "pass", "S2": "reduces_to_known", "S3": None}
    assert verdicts["S2"]["reasons"] == ["reduction claim linear_attention (verified)"]
    assert Ledger(run_dir).verify() == (True, None)
    assert [e["type"] for e in events].count("operator_override") == 1


def test_stages_run_in_order_and_only_s4_to_s6_get_verdict_events(source_run, run_dir):
    do_plumb(source_run, run_dir)
    events = all_events(run_dir)
    started = [e["data"]["stage"] for e in events if e["type"] == "stage_start"]
    assert started == ["S4", "S5", "S6"]
    recorded = [(e["data"]["stage"], e["data"]["verdict"]) for e in events if e["type"] == "verdict"]
    assert recorded == [("S4", "pass"), ("S5", "pass"), ("S6", "pass")]
    assert events[-1]["type"] == "run_end" and events[-1]["data"]["reason"] == "plumb_complete"
    assert not any(e["type"] in ("queued", "card_created", "mutation_requested", "escape_requested")
                   for e in events)


def test_card_is_copied_with_its_history_and_the_source_file_verbatim(source_run, run_dir):
    source_card_file = source_run / "cards" / CARD / "v4.json"
    do_plumb(source_run, run_dir)
    store = Store(run_dir, Ledger(run_dir))
    first = store.get_card(CARD, 1)
    source = json.loads(source_card_file.read_text())
    assert first["parent_ids"] == [] and first["version"] == 1
    assert first["status"]["verdicts"] == source["status"]["verdicts"]
    final = store.get_card(CARD)
    assert final["status"]["stage"] == "S6"
    assert {s: v["verdict"] for s, v in final["status"]["verdicts"].items()} == {
        "S0": "pass", "S1": "pass", "S2": "reduces_to_known", "S4": "pass", "S5": "pass", "S6": "pass"}
    artifacts = {(e["data"]["kind"], e["data"]["name"]): e["data"] for e in all_events(run_dir)
                 if e["type"] == "artifact"}
    copied = artifacts[("plumb_source", f"{CARD}-source-v4.json")]
    assert copied["sha256"] == hashlib.sha256(source_card_file.read_bytes()).hexdigest()
    assert (run_dir / copied["path"]).read_bytes() == source_card_file.read_bytes()
    summary = json.loads((run_dir / artifacts[("plumb_source", f"{CARD}-source-verdicts.json")]["path"]).read_text())
    assert summary["source_verdicts"]["S3"] is None
    assert json.loads((run_dir / artifacts[("reduction_critic", f"{CARD}-v3-abc123.json")]["path"]).read_text()) \
        == REDUCTION
    assert store.file_problems() == []


def test_source_run_is_unchanged_byte_for_byte(source_run, run_dir):
    before = tree_hashes(source_run)
    do_plumb(source_run, run_dir)
    assert tree_hashes(source_run) == before
    inspect_run(run_dir)
    assert tree_hashes(source_run) == before


def test_a_failed_stage_is_recorded_and_the_override_stays_first(source_run, run_dir):
    summary = do_plumb(source_run, run_dir, deps_factory=fake_factory(fail={"builder": TransportError("down")}))
    assert not summary["completed"] and summary["reason"] == "stage_error at S4"
    events = all_events(run_dir)
    assert events[0]["type"] == "operator_override"
    assert [e["data"]["stage"] for e in events if e["type"] == "stage_error"] == ["S4"]
    assert not any(e["type"] == "verdict" for e in events)


def test_candidate_failure_at_s5_ends_the_line_with_a_fail_verdict(source_run, run_dir):
    summary = do_plumb(source_run, run_dir, deps_factory=fake_factory(bench=FakeBench(status="candidate_error")))
    assert not summary["completed"] and summary["verdicts"] == {"S4": "pass", "S5": "fail"}


# ----- refusals: nothing is created -------------------------------------------------------------


def test_unknown_card_empty_reason_and_bad_run_dirs_create_nothing(source_run, run_dir):
    with pytest.raises(PlumbError, match="not in"):
        asyncio.run(plumb(source_run=source_run, card_id="card-00000000", run_dir=run_dir, reason=REASON,
                          cfg=orch_config(), deps_factory=fake_factory()))
    with pytest.raises(PlumbError, match="non-empty --reason"):
        asyncio.run(plumb(source_run=source_run, card_id=CARD, run_dir=run_dir, reason="  ", cfg=orch_config(),
                          deps_factory=fake_factory()))
    with pytest.raises(PlumbError, match="inside it"):
        asyncio.run(plumb(source_run=source_run, card_id=CARD, run_dir=source_run / "plumb", reason=REASON,
                          cfg=orch_config(), deps_factory=fake_factory()))
    assert not run_dir.exists() and not (source_run / "plumb").exists()
    run_dir.mkdir(parents=True)
    (run_dir / "something").write_text("x")
    with pytest.raises(PlumbError, match="not empty"):
        do_plumb(source_run, run_dir)


# ----- the report -------------------------------------------------------------------------------


def test_report_shows_the_override_on_every_page_and_the_dossier(source_run, run_dir, capsys):
    do_plumb(source_run, run_dir)
    assert inspect_run(run_dir) == 0
    text = capsys.readouterr().out
    assert text.startswith("*** OPERATOR OVERRIDE: PLUMBING CHECK. THIS RUN IS NOT A CARD PASSING THE GATES. ***")
    assert STATEMENT in text and REASON in text
    assert "S0 pass, S1 pass, S2 reduces_to_known, S3 none (not run)" in text
    for page in PAGES:
        html = (run_dir / "report" / page).read_text()
        assert 'class="override"' in html, page
        assert "OPERATOR OVERRIDE" in html and STATEMENT in html and REASON in html, page
        assert source_run.name in html, page
    overview = (run_dir / "report" / "index.html").read_text()
    assert overview.index("OPERATOR OVERRIDE") < overview.index("<h1>Run ")
    dossier = (run_dir / "report" / "dossiers" / f"{CARD}.html").read_text()
    assert "Provenance: copied from run" in dossier
    assert "reduction claim linear_attention (verified)" in dossier
    assert "S3 in the source run: none (stage never ran)" in dossier
    # the reduction recorded in the source run is what the overview counts, and S4 to S6 are this run's verdicts
    assert "linear_attention" in overview and "critic artifact missing" not in overview


def test_normal_runs_show_no_override(tmp_path, capsys):
    normal = build_normal_run(tmp_path / "run-20261009T120000Z")
    assert inspect_run(normal) == 0
    assert "OVERRIDE" not in capsys.readouterr().out
    for page in ("index.html", "cards.html", "lineage.html", "models.html", "dossiers/card-aaaaaaaa.html"):
        html = (normal / "report" / page).read_text()
        assert "override" not in html.lower(), page


# ----- the command ------------------------------------------------------------------------------


def test_parser_requires_source_card_and_reason():
    args = build_parser().parse_args(["plumb", "--source-run", "runs/a", "--card", CARD, "--reason", "why",
                                      "--routing", "gateway_claude_local", "--s5-mode", "podman"])
    assert (args.source_run, args.card, args.reason, args.routing, args.s5_mode) == (
        Path("runs/a"), CARD, "why", "gateway_claude_local", "podman")
    assert args.run_dir is None and args.config == Path("hybrid/config/run.toml")
    for missing in (["--card", CARD, "--reason", "r"], ["--source-run", "x", "--reason", "r"],
                    ["--source-run", "x", "--card", CARD]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["plumb", *missing])


def test_command_runs_the_chain_with_injected_dependencies(source_run, run_dir, monkeypatch, capsys):
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    monkeypatch.setattr(plumb_module, "real_deps_factory", lambda cfg, routing, stages: fake_factory())
    code = main(["plumb", "--source-run", str(source_run), "--card", CARD, "--run-dir", str(run_dir),
                 "--reason", REASON, "--config", str(ROOT / "hybrid" / "config" / "run.toml")])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("PLUMBING CHECK (operator override): " + STATEMENT)
    assert json.loads(out[out.index("{"):])["verdicts"] == {"S4": "pass", "S5": "pass", "S6": "pass"}
    first = next(iter(Ledger(run_dir).iter_events()))
    assert first["type"] == "operator_override"
    assert first["data"]["routing"]["table"] == "llxprt"


def test_command_returns_1_when_a_stage_does_not_complete(source_run, run_dir, monkeypatch, capsys):
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    monkeypatch.setattr(plumb_module, "real_deps_factory",
                        lambda cfg, routing, stages: fake_factory(bench=FakeBench(status="candidate_error")))
    code = main(["plumb", "--source-run", str(source_run), "--card", CARD, "--run-dir", str(run_dir),
                 "--reason", REASON, "--config", str(ROOT / "hybrid" / "config" / "run.toml")])
    assert code == 1


def test_command_without_a_gateway_fails_before_creating_the_run(source_run, run_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIGBRANE_GATEWAY_HOME", str(tmp_path / "gateway-home"))
    monkeypatch.delenv("BIGBRANE_GATEWAY_TOKEN", raising=False)
    code = main(["plumb", "--source-run", str(source_run), "--card", CARD, "--run-dir", str(run_dir),
                 "--reason", REASON, "--routing", "gateway_claude_local",
                 "--config", str(ROOT / "hybrid" / "config" / "run.toml")])
    assert code == 2
    assert "cannot start the plumbing check: gateway backend" in capsys.readouterr().err
    assert not run_dir.exists()


def test_command_reports_a_missing_card_without_creating_the_run(source_run, run_dir, monkeypatch, capsys):
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    code = main(["plumb", "--source-run", str(source_run), "--card", "card-00000000", "--run-dir", str(run_dir),
                 "--reason", REASON, "--config", str(ROOT / "hybrid" / "config" / "run.toml")])
    assert code == 2
    assert "card-00000000 is not in" in capsys.readouterr().err
    assert not run_dir.exists()


def test_plumb_designs_validates_and_runs_falsifier_tasks(source_run, run_dir):
    summary = do_plumb(source_run, run_dir)
    assert summary["completed"]
    assert summary["falsifiers"] == {"validated": ["iterated_shift"], "dropped": [], "decided_by_rule": 2,
                                     "claims": {"supported": 2, "refuted": 0, "inconclusive": 0, "untested": 7}}
    events = all_events(run_dir)
    assert events[0]["data"]["falsifiers"] is True
    (designed,) = [e["data"] for e in events if e["type"] == "falsifier_tasks"]
    assert designed["validated"] == ["iterated_shift"] and designed["designer_models"] == ["fake-falsifier_designer"]
    designed_at = next(i for i, e in enumerate(events) if e["type"] == "falsifier_tasks")
    s5_at = next(i for i, e in enumerate(events) if e["type"] == "verdict" and e["data"]["stage"] == "S5")
    assert designed_at < s5_at


def test_command_can_switch_falsifiers_off(source_run, run_dir, monkeypatch, capsys):
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    monkeypatch.setattr(plumb_module, "real_deps_factory", lambda cfg, routing, stages: fake_factory())
    assert build_parser().parse_args(["plumb", "--source-run", "x", "--card", CARD, "--reason", "r"]).falsifiers is None
    code = main(["plumb", "--source-run", str(source_run), "--card", CARD, "--run-dir", str(run_dir),
                 "--reason", REASON, "--no-falsifiers", "--config", str(ROOT / "hybrid" / "config" / "run.toml")])
    out = capsys.readouterr().out
    assert code == 0
    summary = json.loads(out[out.index("{"):])
    assert summary["falsifiers"]["validated"] == [] and summary["falsifiers"]["decided_by_rule"] == 0
    assert next(iter(Ledger(run_dir).iter_events()))["data"]["falsifiers"] is False


# ----- --stages S3: replay the feasibility stage --------------------------------------------------

S3_CARD = "card-46dbb349"
RESIDUAL = {"findings": [{"kind": "underspecified_step", "where": "u[3]", "problem": "float precision not stated"}]}


def build_s3_source_run(run_dir: Path) -> Path:
    """A source run with a card that S3 left undetermined and an s1_residual artifact."""
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)
    store.put_card(make_card(S3_CARD, title="Unknown-Duration Koopman Flow"))
    store.set_verdict(S3_CARD, "S0", "pass", ["card validates"], ["call-aaaaaaaaaaaa"])
    store.set_verdict(S3_CARD, "S1", "pass", ["gate: passed with 1 residual finding"], ["call-bbbbbbbbbbbb"])
    store.put_artifact("s1_residual", f"{S3_CARD}-v6-06d367.json", json.dumps(RESIDUAL).encode(), card_id=S3_CARD)
    store.set_verdict(S3_CARD, "S2", "pass", ["novel"], ["call-cccccccccccc"])
    store.set_verdict(S3_CARD, "S3", "undetermined", ["judges did not agree after 2 round(s)"], ["call-dddddddddddd"])
    ledger.append("run_end", "orchestrator", {"reason": "complete"})
    return run_dir


def test_stages_s3_replays_feasibility_only_with_the_override_and_the_s1_residual(tmp_path):
    source = build_s3_source_run(tmp_path / "runs" / "run-20261010T181300Z")
    before = tree_hashes(source)
    target = tmp_path / "runs" / "plumb-s3"
    llm_box = {}

    def factory(ledger):
        deps = fake_factory()(ledger)
        llm_box["llm"] = deps.llm
        return deps

    summary = asyncio.run(plumb(source_run=source, card_id=S3_CARD, run_dir=target, reason="S3 replay",
                                cfg=orch_config(), deps_factory=factory, stages=("S3",)))
    assert summary["completed"] and summary["verdicts"] == {"S3": "pass"} and summary["reason"] == "plumb_complete"
    assert summary["s3"]["rule"] == "judges" and summary["s3"]["checks"]["o0"]["status"] == "discharged"
    events = all_events(target)
    assert events[0]["type"] == "operator_override" and events[0]["data"]["stages_run"] == ["S3"]
    assert events[0]["data"]["source_verdicts"]["S3"]["verdict"] == "undetermined"
    assert [e["data"]["stage"] for e in events if e["type"] == "stage_start"] == ["S3"]
    assert [e["data"]["stage"] for e in events if e["type"] == "verdict"] == ["S3"]
    llm = llm_box["llm"]
    assert "builder" not in llm.prompts()
    formalizer_user = next(c["user"] for c in llm.calls if c["prompt"] == "formalizer")
    assert "float precision not stated" in formalizer_user
    assert tree_hashes(source) == before


def test_stages_s3_report_names_the_stage_and_the_checks(tmp_path, capsys):
    source = build_s3_source_run(tmp_path / "runs" / "run-20261010T181300Z")
    target = tmp_path / "runs" / "plumb-s3"
    asyncio.run(plumb(source_run=source, card_id=S3_CARD, run_dir=target, reason="S3 replay", cfg=orch_config(),
                      deps_factory=fake_factory(), stages=("S3",)))
    assert inspect_run(target) == 0
    text = capsys.readouterr().out
    assert text.startswith("*** OPERATOR OVERRIDE") and "stages run here  S3" in text
    dossier = (target / "report" / "dossiers" / f"{S3_CARD}.html").read_text()
    assert 'further down (S3) are the only ones' in dossier
    assert "Obligation checks (S3)" in dossier
    index = (target / "report" / "index.html").read_text()
    assert "OPERATOR OVERRIDE" in index and "<td>S3</td>" in index


def test_command_takes_stages_and_keeps_the_banner(tmp_path, monkeypatch, capsys):
    source = build_s3_source_run(tmp_path / "runs" / "run-20261010T181300Z")
    target = tmp_path / "runs" / "plumb-s3"
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    monkeypatch.setattr(plumb_module, "real_deps_factory", lambda cfg, routing, stages: fake_factory())
    assert build_parser().parse_args(["plumb", "--source-run", "x", "--card", CARD, "--reason", "r"]).stages \
        == "S4,S5,S6"
    code = main(["plumb", "--source-run", str(source), "--card", S3_CARD, "--run-dir", str(target),
                 "--reason", "S3 replay", "--stages", "S3", "--config", str(ROOT / "hybrid" / "config" / "run.toml")])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("PLUMBING CHECK (operator override): " + STATEMENT + "\nstages: S3\n")
    assert json.loads(out[out.index("{"):])["verdicts"] == {"S3": "pass"}


def test_command_rejects_bad_stages_before_creating_the_run(source_run, run_dir, monkeypatch, capsys):
    monkeypatch.delenv("BIGBRANE_ROUTING", raising=False)
    code = main(["plumb", "--source-run", str(source_run), "--card", CARD, "--run-dir", str(run_dir),
                 "--reason", REASON, "--stages", "S3,S5", "--config", str(ROOT / "hybrid" / "config" / "run.toml")])
    assert code == 2
    assert "consecutive" in capsys.readouterr().err
    assert not run_dir.exists()
