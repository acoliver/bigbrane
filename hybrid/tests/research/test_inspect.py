import json
import re
from pathlib import Path

import pytest

from hybrid.research.__main__ import main
from hybrid.research.inspect import inspect_run
from hybrid.research.ledger import Ledger
from hybrid.research.store import Store
from hybrid.tests.research.builders import make_card

CARD_A = "card-aaaaaaaa"
CARD_B = "card-bbbbbbbb"
CALL_EXPLORER = "call-aaaaaaaaaaaa"
CALL_REDUCER = "call-bbbbbbbbbbbb"
CALL_CRITIC = "call-cccccccccccc"
JOB = "job-0123456789ab"
PAGES = ["index.html", "cards.html", "lineage.html", "models.html",
         "dossiers/card-aaaaaaaa.html", "dossiers/card-bbbbbbbb.html"]
HASH = "0" * 64


def _llm_call(ledger, transcript, *, call_id, role, model, usage, latency, error=None):
    ledger.append("llm_call", "llm", {
        "call_id": call_id, "role": role, "backend": "llxprt", "model": model,
        "prompt_sha256": HASH, "response_sha256": HASH, "usage": usage,
        "latency_s": latency, "transcript_path": str(transcript), "error": error,
    })


def build_run(run_dir: Path) -> Path:
    ledger = Ledger(run_dir)
    store = Store(run_dir, ledger)

    transcripts = run_dir / "llm" / "transcripts"
    transcripts.mkdir(parents=True)
    explorer_transcript = transcripts / f"{CALL_EXPLORER}.txt"
    explorer_transcript.write_text("SYSTEM ...\nUSER ...\nRESPONSE {\"title\": \"Alpha\"}\n", encoding="utf-8")
    reducer_transcript = transcripts / f"{CALL_REDUCER}.txt"
    reducer_transcript.write_text("reduction critic transcript\n", encoding="utf-8")

    _llm_call(ledger, explorer_transcript, call_id=CALL_EXPLORER, role="explorer", model="opus",
              usage={"input": 1200, "output": 300, "cached": 0}, latency=12.5)
    _llm_call(ledger, reducer_transcript, call_id=CALL_REDUCER, role="reduction_critic", model="opus",
              usage={"input": 900, "output": 200, "cached": None}, latency=8.0)
    _llm_call(ledger, transcripts / "missing.txt", call_id=CALL_CRITIC, role="salad_critic", model="haiku",
              usage={"input": 500, "output": None, "cached": None}, latency=3.0, error="timeout")

    store.put_card(make_card(CARD_A, title="Alpha"))
    store.put_card(make_card(CARD_B, parents=[CARD_A], title="Beta, child of Alpha"))
    store.set_verdict(CARD_A, "S1", "pass", ["toy example reproduced its expected values"], [JOB, CALL_EXPLORER])
    store.set_verdict(CARD_A, "S2", "reduces_to_known", ["update is a phase-weighted linear sum"],
                      [CALL_REDUCER, CALL_CRITIC])

    ledger.append("literature_receipt", "literature", {
        "query": "phase binding holographic memory", "source": "arxiv",
        "ts": "2026-10-09T12:00:00+00:00", "result_ids": ["arxiv:0000.00001"], "card_id": CARD_A,
    })
    bench = {
        "card_id": CARD_A,
        "per_task": {"associative_recall_k3": {"accuracy": 0.61, "kl_to_oracle": 0.42}},
        "baselines": {"gru": {"accuracy": 0.9}},
        "scaling_slopes": {"time_log_log": 1.02, "memory_log_log": 0.0},
    }
    store.put_artifact("bench_result", "candidate.json", json.dumps(bench).encode("utf-8"), card_id=CARD_A)
    store.put_artifact("receipt", f"{JOB}.json", b"{\"exit\": 0}", card_id=CARD_A)
    return run_dir


@pytest.fixture
def run_dir(tmp_path) -> Path:
    return build_run(tmp_path / "run-20261009T120000Z")


@pytest.fixture
def report(run_dir, capsys):
    assert inspect_run(run_dir) == 0
    return run_dir / "report", capsys.readouterr().out


def test_report_pages_are_written(report):
    report_dir, _ = report
    for page in PAGES:
        assert (report_dir / page).is_file(), page


def test_text_summary_shows_stages_verdicts_calls_and_ledger(report):
    _, out = report
    assert "Ledger: verified" in out
    assert "stage S2: 1" in out
    assert "stage S0: 1" in out
    assert "S1 pass: 1" in out
    assert "S2 reduces_to_known: 1" in out
    assert "opus" in out and "haiku" in out
    assert "Totals: 3 calls" in out
    assert "wall time" in out


def test_every_relative_link_resolves_to_a_file(report):
    report_dir, _ = report
    checked = 0
    for page in report_dir.rglob("*.html"):
        for href in re.findall(r'href="([^"]+)"', page.read_text(encoding="utf-8")):
            if href.startswith("#"):
                continue
            assert "://" not in href, href
            assert not href.startswith("/"), href
            assert (page.parent / href).resolve().is_file(), f"{href} in {page}"
            checked += 1
    assert checked > 10


def test_pages_have_no_scripts_or_external_resources(report):
    report_dir, _ = report
    for page in report_dir.rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert "<script" not in html
        assert not re.search(r'(src|href)="https?://', html)


def test_transcripts_are_linked_relative_to_the_run_folder(report, run_dir):
    report_dir, _ = report
    dossier = (report_dir / "dossiers" / f"{CARD_A}.html").read_text(encoding="utf-8")
    assert f'href="../../llm/transcripts/{CALL_EXPLORER}.txt"' in dossier
    assert f'href="../../llm/transcripts/{CALL_REDUCER}.txt"' in dossier


def test_missing_transcript_is_flagged_not_linked(report):
    report_dir, _ = report
    dossier = (report_dir / "dossiers" / f"{CARD_A}.html").read_text(encoding="utf-8")
    assert "transcript for call-cccccccccccc: file missing" in dossier
    assert "missing.txt" not in dossier


def test_dossier_lists_versions_with_diff_verdicts_and_receipts(report):
    report_dir, _ = report
    dossier = (report_dir / "dossiers" / f"{CARD_A}.html").read_text(encoding="utf-8")
    assert f'href="../../cards/{CARD_A}/v1.json"' in dossier
    assert f'href="../../cards/{CARD_A}/v3.json"' in dossier
    assert "+++ v2" in dossier
    assert "reduces_to_known" in dossier
    assert "phase binding holographic memory" in dossier
    assert "arxiv:0000.00001" in dossier
    assert "associative_recall_k3" in dossier
    assert "time_log_log" in dossier
    assert f"{JOB}.json" in dossier


def test_card_table_shows_stage_reached_and_why(report):
    report_dir, _ = report
    cards = (report_dir / "cards.html").read_text(encoding="utf-8")
    assert "Beta, child of Alpha" in cards
    assert "update is a phase-weighted linear sum" in cards
    assert "reduces_to_known" in cards


def test_lineage_tree_nests_the_child_under_its_parent(report):
    report_dir, _ = report
    lineage = (report_dir / "lineage.html").read_text(encoding="utf-8")
    parent = lineage.index(CARD_A)
    child = lineage.index(CARD_B)
    assert parent < child
    assert "<ul>" in lineage[parent:child]


def test_model_usage_table_has_tokens_latency_and_errors(report):
    report_dir, _ = report
    models = (report_dir / "models.html").read_text(encoding="utf-8")
    assert "<td>opus</td>" in models
    assert "<td>haiku</td>" in models
    assert "2100" in models
    assert "10.25" in models
    assert "20.50" in models


def test_tampered_ledger_fails_and_builds_no_report(run_dir, capsys):
    ledger_path = run_dir / "ledger.jsonl"
    ledger_path.write_text(ledger_path.read_text(encoding="utf-8").replace('"opus"', '"gpt"', 1), encoding="utf-8")

    assert inspect_run(run_dir) == 1
    assert "BROKEN" in capsys.readouterr().out
    assert not (run_dir / "report" / "index.html").exists()


def test_edited_artifact_is_reported_and_exit_code_is_nonzero(run_dir, capsys):
    (run_dir / "artifacts" / "bench_result" / "candidate.json").write_text("{}", encoding="utf-8")

    assert inspect_run(run_dir) == 1
    out = capsys.readouterr().out
    assert "sha256 mismatch: artifacts/bench_result/candidate.json" in out
    assert (run_dir / "report" / "index.html").is_file()


def test_missing_ledger_is_an_error(tmp_path, capsys):
    assert inspect_run(tmp_path / "nothing-here") == 2
    assert "No ledger.jsonl" in capsys.readouterr().out


def test_command_line_inspect_subcommand(run_dir, capsys):
    assert main(["inspect", str(run_dir)]) == 0
    assert "Totals: 3 calls" in capsys.readouterr().out
