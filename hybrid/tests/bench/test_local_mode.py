"""Local mode, the baseline cache, the CLI and the shipped configs, none of which need Podman."""
from __future__ import annotations

import json

import pytest

from hybrid.bench import config as bench_config
from hybrid.bench.__main__ import main
from hybrid.bench.evaluate import evaluate_candidate
from hybrid.bench.local import LocalExecutor

from .conftest import ROOT, TINY

# ladder.toml configures the test ladder (hybrid/bench/ladder.py), a different schema; it is validated in
# test_ladder_tasks.py.
CONFIGS = sorted(p for p in (ROOT / "hybrid" / "bench" / "configs").glob("*.toml") if p.name != "ladder.toml")


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    cfg = bench_config.load_config(path)
    assert set(cfg["tasks"]) == {"hmm", "vom", "recall", "compose", "fewshot"}


def test_quick_and_default_configs_do_not_share_baseline_cache_keys():
    quick, default = (bench_config.load_config(ROOT / "hybrid" / "bench" / "configs" / n) for n in ("quick.toml", "default.toml"))
    assert bench_config.task_config_hash(quick) != bench_config.task_config_hash(default)


def test_local_mode_is_labeled_untrusted_and_baselines_are_cached(work_dir, template_dir):
    executor = LocalExecutor("bench-local", work_dir / "receipts")
    cache = work_dir / "cache"
    first = evaluate_candidate(template_dir, TINY, work_dir / "run1", executor, baselines=["ngram"], cache_dir=cache,
                               log=lambda _: None)
    assert first["trusted"] is False and first["host"]["trusted"] is False and first["host"]["mode"] == "local"
    on_disk = json.loads((work_dir / "run1" / "bench" / "bench_result.json").read_text())
    assert on_disk["trusted"] is False
    assert first["host"]["receipts"]["candidate"]["status"] == "ok"
    second = evaluate_candidate(template_dir, TINY, work_dir / "run2", executor, baselines=["ngram"], cache_dir=cache,
                                log=lambda _: None)
    assert second["entrants"]["ngram"]["job_id"] == first["entrants"]["ngram"]["job_id"]   # served from the cache
    assert second["entrants"]["ngram"]["tasks"] == first["entrants"]["ngram"]["tasks"]
    assert second["trusted"] is False


def test_run_directory_must_be_fresh(work_dir, template_dir):
    executor = LocalExecutor("bench-fresh", work_dir / "receipts")
    evaluate_candidate(template_dir, TINY, work_dir / "run", executor, baselines=[], cache_dir=work_dir / "cache",
                       log=lambda _: None)
    with pytest.raises(FileExistsError):
        evaluate_candidate(template_dir, TINY, work_dir / "run", executor, baselines=[], cache_dir=work_dir / "cache",
                           log=lambda _: None)


def test_cli_eval_local_writes_markdown_and_untrusted_result(work_dir, template_dir, capsys):
    run_dir, table = work_dir / "run", work_dir / "table.md"
    code = main(["eval", "--candidate", str(template_dir), "--config", str(TINY), "--run-dir", str(run_dir),
                 "--local", "--baselines", "ngram", "--cache-dir", str(work_dir / "cache"), "--claimed-exponent", "0",
                 "--markdown", str(table)])
    assert code == 0
    out = capsys.readouterr().out
    summary = json.loads(out[out.rindex("\n{") + 1:])
    assert summary["trusted"] is False
    assert json.loads((run_dir / "bench" / "bench_result.json").read_text())["trusted"] is False
    assert "| entrant |" in table.read_text() and "ngram" in table.read_text()


def test_cli_refuses_an_existing_bench_directory(work_dir, template_dir):
    (work_dir / "run" / "bench").mkdir(parents=True)
    with pytest.raises(SystemExit):
        main(["eval", "--candidate", str(template_dir), "--config", str(TINY), "--run-dir", str(work_dir / "run"),
              "--local"])
