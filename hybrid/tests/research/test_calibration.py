"""Calibration of the math stages on anonymized known models (Phase 4, plan section 12): the fixture cards, the
anonymization check, the driver with scripted roles, the comparison with the expected table, and the command."""
import asyncio
import copy
import json
import os
from pathlib import Path

import pytest

from hybrid.math.runner import MathRunner
from hybrid.research import calibration as cal
from hybrid.research.__main__ import build_parser, main
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.schemas import validate
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_orchestrator_fakes import FakeLLM


# ----- fixtures and anonymization -----------------------------------------------------------------------------------

def test_the_four_cards_are_schema_valid_anonymous_and_distinct():
    cards = cal.load_cards()
    assert list(cards) == ["k1", "k2", "k3", "k4"]
    assert len({card["id"] for card in cards.values()}) == 4
    for card in cards.values():
        validate("theory_card", card)
        assert cal.anonymization_problems(card) == []
        assert card["references"] == [] and "source_transplant" not in card
        assert card["status"] == {"stage": "S0", "verdicts": {}}


def test_the_expected_table_covers_every_card_and_scored_cell():
    expected = cal.load_expected()
    assert expected["runs_required"] == 3 and tuple(expected["cells"]) == cal.CELLS
    assert set(expected["cards"]) == set(cal.CARD_KEYS)
    for key, want in expected["cards"].items():
        assert set(want["cells"]) == set(cal.CELLS)
        assert want["profile"] in ("pass", "fail") and want["target"] in ("pass", "fail")
    assert expected["cards"]["k1"]["target"] == "fail"
    assert [k for k, w in expected["cards"].items() if w["profile"] == "pass"] == ["k1", "k2"]
    for key in ("k3", "k4"):
        assert {"obligation_failed:P2.nesting", "obligation_failed:P2.composition"} <= set(
            expected["cards"][key]["kill_codes_include"])


@pytest.mark.parametrize("where,text,finding", [
    ("summary", "A small Transformer with two layers.", "'transformer'"),
    ("title", "Self-attention, renamed", "'attention'"),
    ("summary", "A hidden Markov chain", "'hidden markov'"),
    ("summary", "an LSTM", "'lstm'"),
    ("summary", "a trigram table", "'trigram'"),
    ("summary", "a Hopfield memory", "'hopfield'"),
    ("summary", "the linear attention trick", "'linear attention'"),
    ("summary", "as in arXiv 1706.03762", "cites a source"),
    ("summary", "introduced in 1997", "'1997'"),
    ("summary", "following Vaswani et al.", "'vaswani'"),
    ("summary", "see https://example.org/paper", "'https://'"),
])
def test_the_anonymization_check_catches_names_terms_and_citations(where, text, finding):
    card = copy.deepcopy(cal.load_cards(("k1",))["k1"])
    card[where] = text
    problems = cal.anonymization_problems(card)
    assert any(finding in p for p in problems), problems


def test_the_anonymization_check_catches_references_and_mechanism_part_names():
    card = copy.deepcopy(cal.load_cards(("k1",))["k1"])
    card["references"] = [{"source_id": "x", "why": "y"}]
    card["mechanism"]["state"][0]["meaning"] = "Key vectors of every position read so far."
    problems = cal.anonymization_problems(card)
    assert "references must be empty" in problems
    assert any("attention-style part name: 'key'" in p for p in problems)
    # The same words are ordinary task vocabulary in the claims, where they are allowed.
    card = copy.deepcopy(cal.load_cards(("k3",))["k3"])
    card["claims"]["capabilities"][0]["prediction"] += " Every key is queried once for its value."
    assert cal.anonymization_problems(card) == []



def test_the_anonymization_check_reads_the_family_ids_and_extra_terms():
    card = copy.deepcopy(cal.load_cards(("k4",))["k4"])
    assert "exact match cache" in cal.family_terms()
    card["summary"] += " It behaves like an exact match cache."
    assert any("'exact match cache'" in p for p in cal.anonymization_problems(card))
    card = copy.deepcopy(cal.load_cards(("k4",))["k4"])
    card["summary"] += " A continuation table, also called a Zorbo table."
    assert cal.anonymization_problems(card) == []
    assert any("'zorbo'" in p for p in cal.anonymization_problems(card, extra_terms=("zorbo",)))
    card = copy.deepcopy(cal.load_cards(("k4",))["k4"])
    card["source_transplant"] = {"field": "x"}
    card["summary"] += " See doi 10.1000/xyz."
    problems = cal.anonymization_problems(card)
    assert "source_transplant must be absent (it names the source)" in problems
    assert any("cites a source: 'doi'" in p for p in problems)

def test_load_cards_refuses_a_card_that_is_not_anonymous(tmp_path):
    for key in cal.CARD_KEYS:
        (tmp_path / f"{key}.json").write_text((cal.FIXTURES / f"{key}.json").read_text())
    card = json.loads((tmp_path / "k2.json").read_text())
    card["title"] = "An LSTM written plainly"
    (tmp_path / "k2.json").write_text(json.dumps(card))
    with pytest.raises(ValueError, match="k2 is not anonymous"):
        cal.load_cards(directory=tmp_path)


# ----- the driver -----------------------------------------------------------------------------------------------------

def factory(cards, *, p2_by_run=None):
    """Deps per run directory: scripted roles for the run's card; ``p2_by_run`` maps (card, run) to whether the
    prover's constructions pass P2 (a run-to-run disagreement)."""
    def build(ledger):
        key, run = cf.card_key(ledger)
        executor = cf.FastExecutor(ledger.run_dir / "exec")
        passes = (p2_by_run or {}).get((key, run))
        llm = FakeLLM(ledger.run_dir, ledger, cf.script(key, cards[key], passes_p2=passes))
        return cal.Deps(llm, executor, "image", MathRunner(executor, mf.IMAGE, ledger.run_dir))
    return build


def run_calibration(tmp_path, keys=cal.CARD_KEYS, runs=3, **kwargs):
    cards = cal.load_cards(keys)
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    return asyncio.run(cal.calibrate(cards=cards, runs=runs, run_root=tmp_path / "runs", out_dir=tmp_path / "out",
                                     cfg=cfg, reason="calibration test", deps_factory=factory(cards, **kwargs)))


def test_three_runs_of_the_four_cards_reproduce_the_expected_table(tmp_path):
    result = run_calibration(tmp_path)
    report = result["report"]
    assert report["accepted"], json.dumps(report, indent=1)[:3000]
    assert report["runs_per_card"] == {"k1": 3, "k2": 3, "k3": 3, "k4": 3}
    for key in cal.CARD_KEYS:
        assert report["cards"][key]["agree"] and report["cards"][key]["reproduced_every_run"]
    k1 = [o for o in result["outcomes"] if o["card"] == "k1"]
    assert all(o["profile_passed"] and o["target"]["status"] == "fail" for o in k1)
    assert all(o["target"]["cost_exponent"] == pytest.approx(2.0) for o in k1)
    k3 = [o for o in result["outcomes"] if o["card"] == "k3"]
    assert all(o["stages"]["M"]["verdict"] == "fail" for o in k3)
    out = tmp_path / "out"
    assert len((out / "outcomes.jsonl").read_text().splitlines()) == 12
    stored = json.loads((out / "calibration.json").read_text())
    assert stored["report"]["accepted"] and stored["meta"]["target_rule"] == {
        "max_cost_exponent": 1.5, "state_growth": "log", "coupled_max_cost_exponent": 1.5,
        "coupled_max_state_exponent": 0.5}
    assert report["agreement_rate"] == 1.0
    assert all(report["cards"][key]["agreement_rate"] == 1.0 for key in cal.CARD_KEYS)
    text = (out / "report.md").read_text()
    assert "Accepted (plan section 12): yes" in text and "| k4 | 3 |" in text
    assert "Run-to-run agreement: 100% of scored items" in text and "- k1 (100%): all cells agree" in text


def test_every_run_is_an_override_with_the_profile_on_and_the_target_off_in_m(tmp_path):
    result = run_calibration(tmp_path, keys=("k1",), runs=1)
    outcome = result["outcomes"][0]
    ledger = Ledger(Path(outcome["run_dir"]))
    events = list(ledger.iter_events())
    assert events[0]["type"] == "operator_override"
    assert events[0]["data"]["calibration"] == {"card": "k1", "run": 1}
    assert "stages S0 to S2 are bypassed" in events[0]["data"]["statement"]
    start = next(e for e in events if e["type"] == "run_start")["data"]
    assert start["profile"] is True and start["target_in_m"] is False
    # The target gate did not decide stage M, though the separate gate rejects the card.
    assert "obligation_failed:target" not in outcome["m_kill_codes"]
    assert outcome["target"]["status"] == "fail"
    stages = [e["data"]["stage"] for e in events if e["type"] == "stage_verdict"]
    assert stages == ["M", "C"]


def test_a_run_that_disagrees_is_reported_and_fails_acceptance(tmp_path):
    # FIX-16: k3's card now states its 4-bit carried state, so no scripted construction can pass P2 under it; the
    # disagreement is scripted on k2 instead, whose second run gets the capped constructions.
    result = run_calibration(tmp_path, keys=("k2",), runs=3, p2_by_run={("k2", 2): False})
    report = result["report"]
    assert not report["accepted"]
    entry = report["cards"]["k2"]
    assert not entry["agree"]
    assert entry["agreement"]["P2.nesting"] == {"values": ["pass", "fail", "pass"], "agree": False}
    second = entry["runs"][1]
    assert not second["reproduced"]
    assert "P2.nesting: expected pass, got fail" in second["mismatches"]
    # The second run's constructions are capped, so its four P2 cells and the profile verdict split from the other
    # runs; P1, P3 (the copy limit at 4096 is beaten by neither), P5 and the target agree: four of the nine scored
    # items, so agreement is below 100%.
    split = sorted(cell for cell, item in entry["agreement"].items() if not item["agree"])
    assert split == ["P2.composition", "P2.copy", "P2.nesting", "P2.recall", "profile"]
    assert entry["agreement_rate"] == pytest.approx(4 / 9) and report["agreement_rate"] == pytest.approx(4 / 9)
    text = (tmp_path / "out" / "report.md").read_text()
    assert "Run-to-run agreement: 44% of scored items (acceptance needs 100%)." in text
    assert "- k2 (44%): disagree on" in text and "P2.nesting (pass, fail, pass)" in text


def test_fewer_runs_than_the_table_needs_are_not_accepted(tmp_path):
    report = run_calibration(tmp_path, keys=("k2",), runs=1)["report"]
    assert report["cards"]["k2"]["reproduced_every_run"]
    assert not report["accepted"] and "acceptance needs 3 runs per card" in report["problems"]


def test_the_driver_refuses_a_used_output_directory_and_an_empty_reason(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        run_calibration(tmp_path, keys=("k1",), runs=1)
    cards = cal.load_cards(("k1",))
    with pytest.raises(ValueError, match="reason"):
        asyncio.run(cal.calibrate(cards=cards, runs=1, run_root=tmp_path / "r", out_dir=tmp_path / "o2",
                                  cfg=load_run_config(ROOT / "hybrid" / "config" / "run.toml"), reason=" ",
                                  deps_factory=factory(cards)))



def test_a_single_cell_that_one_run_does_not_show_lowers_agreement_below_100_percent():
    want = cal.load_expected()["cards"]["k2"]

    def outcome(run, **cells):
        return {"card": "k2", "run": run, "cells": dict(want["cells"], **cells), "profile_passed": True,
                "m_kill_codes": [], "target": {"status": want["target"]}, "stages": {"M": {"verdict": "pass"}},
                "model_calls": 8, "stopped": None}

    expected = {"cards": {"k2": want}, "runs_required": 3}
    agreed = cal.compare(expected, [outcome(1), outcome(2), outcome(3)])
    assert agreed["accepted"] and agreed["agreement_rate"] == 1.0
    split = cal.compare(expected, [outcome(1), outcome(2), outcome(3, P3="not_shown")])
    assert not split["accepted"] and split["agreement_rate"] == pytest.approx(8 / 9)
    assert split["cards"]["k2"]["agreement"]["P3"] == {"values": ["pass", "pass", "not_shown"], "agree": False}
    assert split["cards"]["k2"]["runs"][2]["mismatches"] == ["P3: expected pass, got not_shown"]


def test_a_calibration_run_directory_is_never_resumed(tmp_path):
    # A run that stopped is rerun from scratch in a new directory: a used directory is refused before any call.
    cards = cal.load_cards(("k1",))
    cfg = cal.calibration_config(load_run_config(ROOT / "hybrid" / "config" / "run.toml"))
    used = tmp_path / "k1-run1"
    used.mkdir()
    (used / "ledger.jsonl").write_text("{}\n")
    calls = []

    def deps(ledger):
        calls.append(ledger)
        raise AssertionError("no deps for a refused run")

    with pytest.raises(ValueError, match="is not empty"):
        asyncio.run(cal.run_card(key="k1", card=cards["k1"], run_index=1, run_dir=used, cfg=cfg, reason="x",
                                 deps_factory=deps, target_rule={"max_cost_exponent": 1.5, "state_growth": "log"}))
    assert calls == [] and (used / "ledger.jsonl").read_text() == "{}\n"
    assert cfg["math"]["profile"] is True and cfg["target"]["enabled"] is False and cfg["s3"]["mode"] == "math"

def test_cells_not_shown_never_match_the_expected_table():
    assert cal.cell_outcome("observed") == "pass" and cal.cell_outcome("proven") == "pass"
    assert cal.cell_outcome("failed") == "fail" and cal.cell_outcome("refuted") == "fail"
    for status in ("unproven", "not_formalized", "blocked", "circular", "conditional", None):
        assert cal.cell_outcome(status) == "not_shown"


# ----- the command --------------------------------------------------------------------------------------------------

def test_the_calibrate_command_is_wired_and_refuses_unknown_cards(tmp_path, capsys):
    args = build_parser().parse_args(["calibrate", "--reason", "x"])
    assert (args.runs, args.cards, args.out, args.run_root) == (3, "k1,k2,k3,k4", None, None)
    code = main(["calibrate", "--reason", "x", "--cards", "k9", "--out", str(tmp_path / "o"),
                 "--run-root", str(tmp_path / "r")])
    assert code == 2 and "--cards takes keys from k1, k2, k3, k4" in capsys.readouterr().err
    assert not (tmp_path / "o").exists() and not (tmp_path / "r").exists()


# ----- the stored live result (plan section 17, Phase 4: the expected table as tests over the stored result) ---------

STORED = os.environ.get("BIGBRANE_CALIBRATION_RESULT")


@pytest.mark.skipif(not STORED, reason="set BIGBRANE_CALIBRATION_RESULT to a live calibration.json")
def test_the_stored_live_calibration_meets_the_acceptance_rule():
    stored = json.loads(Path(STORED).read_text())
    report = cal.compare(cal.load_expected(), stored["outcomes"])
    assert report["accepted"], json.dumps(report, indent=1)[:4000]
