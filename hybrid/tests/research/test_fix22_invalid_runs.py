"""FIX-22 (decision ENV): a card-run stopped by an environment failure is invalid_environment, never scored, and
`calibrate --rerun-invalid` reruns it in a fresh run directory and rescores with the rerun in its place."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from hybrid.llm.errors import BackendError, OutputExhausted, SchemaValidationError, TransportError
from hybrid.math.runner import MathRunner
from hybrid.research import calibration as cal
from hybrid.research.__main__ import main
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM

# The two stops of calibration 20261016T141847Z (runs/calibration-math-20261016T141847Z k3-run1 and k4-run1).
K3_STOP = ("stage_error at C: statement_match: BackendError: POST http://127.0.0.1:1234/v1/chat/completions returned "
           "HTTP 400: {\"error\":\"The number of tokens to keep from the initial prompt is greater than the context "
           "length. Try to load the model with a larger context length, or provide a shorter input\"}")
K4_STOP = ("stage_error at M: statement_match: BackendError: POST http://127.0.0.1:1234/v1/chat/completions returned "
           "HTTP 400: {\"error\":\"Model unloaded.\"}")


@pytest.mark.parametrize("stopped, kinds, want", [
    (K3_STOP, None, cal.INVALID_ENVIRONMENT),
    (K4_STOP, None, cal.INVALID_ENVIRONMENT),
    ("stage_error at C: challenger: TransportError: all backend entries failed for role 'challenger': llxprt:"
     "gpt-sol-high: gateway returned HTTP 502", None, cal.INVALID_ENVIRONMENT),
    ("stage_error at M: statement_crosscheck: all backend entries failed", ["TransportError", "LLMError"],
     cal.INVALID_ENVIRONMENT),
    ("stage_error at M: statement_crosscheck: [Errno 61] Connection refused", None, cal.INVALID_ENVIRONMENT),
    ("stage_error at M: prover: OutputExhausted: all backend entries failed", None, cal.VALID),
    ("stage_error at M: prover: x", ["OutputExhausted", "TransportError", "LLMError"], cal.VALID),
    ("stage_error at C: statement_match: SchemaValidationError: output invalid after one retry", None, cal.VALID),
    # A class name decides before the phrases: a schema failure quoting "Model unloaded" is still the model's.
    ("stage_error at C: statement_match: SchemaValidationError: says Model unloaded", None, cal.VALID),
    ("stage_error at C: statement_match gave no finding for P1 P2.copy", None, cal.VALID),
    ("stage_error at M: executor infra_error in math check s1: podman", None, cal.VALID),
    ("budget: max_calls reached", None, cal.VALID),
    ("card_incomplete at M (fixture error): resources.architecture.width", None, cal.VALID),
    (None, None, cal.VALID),
])
def test_environment_stops_are_invalid_and_verdict_stops_are_not(stopped, kinds, want):
    assert cal.stop_validity(stopped, kinds) == want


def test_error_kinds_record_the_llm_error_classes_of_the_cause():
    assert cal.error_kinds(BackendError("x")) == ["BackendError", "LLMError"]
    assert cal.error_kinds(OutputExhausted("x")) == ["OutputExhausted", "TransportError", "LLMError"]
    assert cal.error_kinds(ValueError("x")) == [] and cal.error_kinds(None) == []


def test_a_verdict_level_outcome_is_never_invalid():
    # fail and not_shown cells (a held check, an unresolved match) with no stop are valid whatever they show.
    outcome = {"card": "k1", "run": 1, "run_dir": "x", "stopped": None,
               "cells": {cell: "not_shown" for cell in cal.CELLS}}
    assert cal.validity(outcome) == cal.VALID


def factory(cards, *, fail=None):
    """Deps per run directory; ``fail`` maps (card, run) to {prompt: exception} raised by the fake LLM there."""
    def build(ledger):
        key, run = cf.card_key(ledger)
        executor = cf.FastExecutor(ledger.run_dir / "exec")
        llm = FakeLLM(ledger.run_dir, ledger, cf.script(key, cards[key]), fail=(fail or {}).get((key, run)))
        return cal.Deps(llm, executor, "image", MathRunner(executor, mf.IMAGE, ledger.run_dir))
    return build


def calibrate(tmp_path, keys, runs, fail=None):
    cards = cal.load_cards(keys)
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    return asyncio.run(cal.calibrate(cards=cards, runs=runs, run_root=tmp_path / "runs", out_dir=tmp_path / "out",
                                     cfg=cfg, reason="FIX-22 test", deps_factory=factory(cards, fail=fail)))


def rerun(tmp_path, keys, fail=None):
    cards = cal.load_cards(keys)
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    return asyncio.run(cal.rerun_invalid(out_dir=tmp_path / "out", cfg=cfg, reason="FIX-22 rerun test",
                                         deps_factory=factory(cards, fail=fail)))


def digest(directory: Path) -> dict:
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


UNLOADED = BackendError("POST http://127.0.0.1:1234/v1/chat/completions returned HTTP 400: Model unloaded.")


def test_an_environment_failure_is_not_scored_and_blocks_acceptance(tmp_path):
    result = calibrate(tmp_path, ("k2",), 3, fail={("k2", 1): {"statement_match": UNLOADED}})
    report = result["report"]
    first = next(o for o in result["outcomes"] if o["run"] == 1)
    assert first["validity"] == cal.INVALID_ENVIRONMENT
    assert first["stop_error_kinds"] == ["BackendError", "LLMError"]
    assert not report["accepted"]
    assert report["runs_per_card"] == {"k2": 2}
    assert [r["run"] for r in report["cards"]["k2"]["runs"]] == [2, 3]
    assert report["cards"]["k2"]["agree"] and report["cards"]["k2"]["reproduced_every_run"]
    assert [(i["card"], i["run"], i["replaced_by"]) for i in report["invalid"]] == [("k2", 1, None)]
    assert any("1 card-run(s) invalid (environment failure), not scored: k2-run1" in p for p in report["problems"])
    events = list(Ledger(Path(first["run_dir"])).iter_events())
    error = next(e["data"] for e in events if e["type"] == "stage_error")
    assert error["error_kinds"] == ["BackendError", "LLMError"]
    end = next(e["data"] for e in events if e["type"] == "run_end")
    assert end["validity"] == cal.INVALID_ENVIRONMENT
    text = (tmp_path / "out" / "report.md").read_text()
    assert "Invalid card-runs (environment failures, decision ENV; not scored):" in text
    assert "- k2 run 1 (k2-run1): stage_error at" in text and "not replaced (rerun with --rerun-invalid)" in text


def test_a_rerun_replaces_the_invalid_card_run_and_keeps_its_directory(tmp_path):
    calibrate(tmp_path, ("k2",), 3, fail={("k2", 1): {"statement_match": UNLOADED}})
    original = tmp_path / "runs" / "k2-run1"
    before = digest(original)
    before_json = (tmp_path / "out" / "calibration.json").read_text()
    result = rerun(tmp_path, ("k2",))
    report = result["report"]
    assert report["accepted"], json.dumps(report, indent=1)[:2000]
    assert report["runs_per_card"] == {"k2": 3}
    assert [(r["run"], r["rerun"]) for r in report["cards"]["k2"]["runs"]] == [(1, 1), (2, None), (3, None)]
    fresh = tmp_path / "runs" / "k2-run1-rerun1"
    assert [(i["run_dir"], i["replaced_by"]) for i in report["invalid"]] == [(str(original), str(fresh))]
    assert digest(original) == before
    out = tmp_path / "out"
    assert (out / "calibration-before-rerun1.json").read_text() == before_json
    assert (out / "report-before-rerun1.md").exists()
    assert len((out / "outcomes.jsonl").read_text().splitlines()) == 4
    stored = json.loads((out / "calibration.json").read_text())
    assert stored["meta"]["reruns"][0]["replaced"] == [str(original)]
    assert stored["meta"]["reruns"][0]["run_dirs"] == [str(fresh)]
    override = next(Ledger(fresh).iter_events("operator_override"))["data"]
    assert override["calibration"] == {"card": "k2", "run": 1, "rerun": 1, "rerun_of": str(original)}
    text = (out / "report.md").read_text()
    assert "Accepted (plan section 12): yes" in text and "| k2 | 1 (rerun 1) |" in text
    assert "replaced by k2-run1-rerun1" in text
    # Everything valid now: a further rerun is refused.
    with pytest.raises(ValueError, match="every card-run is valid; nothing to rerun"):
        rerun(tmp_path, ("k2",))


def test_an_invalid_rerun_is_rerun_again_under_the_next_number(tmp_path):
    calibrate(tmp_path, ("k2",), 1, fail={("k2", 1): {"statement_match": UNLOADED}})
    first = rerun(tmp_path, ("k2",), fail={("k2", 1): {"statement_match": TransportError("connection refused")}})
    assert [i["replaced_by"] for i in first["report"]["invalid"]] == [None, None]
    second = rerun(tmp_path, ("k2",))
    assert [Path(o["run_dir"]).name for o in second["outcomes"]] == ["k2-run1", "k2-run1-rerun1", "k2-run1-rerun2"]
    assert all(i["replaced_by"].endswith("k2-run1-rerun2") for i in second["report"]["invalid"])


def test_a_valid_card_run_is_never_rerun(tmp_path):
    result = calibrate(tmp_path, ("k2",), 1)
    outcome = result["outcomes"][0]
    assert cal.rerun_targets(result["outcomes"]) == []
    with pytest.raises(ValueError, match="is a valid card-run; only invalid_environment card-runs are rerun"):
        cal.check_rerunnable(outcome, cal.load_cards(("k2",))["k2"])
    with pytest.raises(ValueError, match="nothing to rerun"):
        rerun(tmp_path, ("k2",))
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["k2-run1"]


def test_a_model_failure_stays_a_valid_run(tmp_path):
    bad = SchemaValidationError("output invalid after one retry: not JSON")
    bad.call_id = "call-fake"
    result = calibrate(tmp_path, ("k2",), 1, fail={("k2", 1): {"statement_fixer": bad}})
    outcome = result["outcomes"][0]
    assert outcome["stopped"].startswith("stage_error at M: statement_fixer: SchemaValidationError")
    assert outcome["stop_error_kinds"] == ["SchemaValidationError", "LLMError"] and outcome["validity"] == cal.VALID
    assert result["report"]["invalid"] == [] and result["report"]["runs_per_card"] == {"k2": 1}


def test_a_rerun_refuses_a_card_that_changed_since_the_run(tmp_path):
    calibrate(tmp_path, ("k2",), 1, fail={("k2", 1): {"statement_match": UNLOADED}})
    cards = cal.load_cards(("k2",))
    changed = dict(cards["k2"], version=cards["k2"]["version"] + 1)
    outcome = json.loads((tmp_path / "out" / "calibration.json").read_text())["outcomes"][0]
    with pytest.raises(ValueError, match="changed since"):
        cal.check_rerunnable(outcome, changed)


def test_older_records_without_validity_are_classified_from_the_stop_text():
    outcomes = [{"card": "k3", "run": 1, "run_dir": "/r/k3-run1", "stopped": K3_STOP, "cells": {}},
                {"card": "k3", "run": 2, "run_dir": "/r/k3-run2", "stopped": None, "cells": {}}]
    scored, invalid = cal.split_valid(outcomes)
    assert [o["run"] for o in scored] == [2]
    assert [(i["run"], i["validity"]) for i in invalid] == [(1, cal.INVALID_ENVIRONMENT)]
    assert [(t["outcome"]["run"], t["rerun"]) for t in cal.rerun_targets(outcomes)] == [(1, 1)]


def test_the_rerun_option_is_wired_and_refuses_out_and_run_root(tmp_path, capsys, monkeypatch):
    from hybrid.research import __main__ as entry

    monkeypatch.setattr(entry, "local_model_check", lambda args: True)
    code = main(["calibrate", "--reason", "x", "--rerun-invalid", str(tmp_path), "--out", str(tmp_path / "o")])
    assert code == 2 and "drop --out and --run-root" in capsys.readouterr().err
    args = entry.build_parser().parse_args(["calibrate", "--reason", "x", "--rerun-invalid", str(tmp_path)])
    assert args.rerun_invalid == tmp_path and args.out is None
    with pytest.raises(FileNotFoundError):
        asyncio.run(cal.rerun_invalid(out_dir=tmp_path / "missing", cfg={}, reason="x", deps_factory=None))
