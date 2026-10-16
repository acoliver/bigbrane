"""End-to-end through the Podman executor: the template candidate against the GRU baseline,
then a second evaluation that takes the GRU baseline from the cache."""
from __future__ import annotations

import json

from hybrid.bench.evaluate import evaluate_candidate
from hybrid.executor.lock import read_lock

from .conftest import STAMP, TINY, podman


def labeled(kind, run_id):
    from hybrid.executor.podman import Podman
    argv = {"containers": ["ps", "-a", "--format", "{{.Names}}"],
            "volumes": ["volume", "ls", "--format", "{{.Name}}"]}[kind]
    out = Podman().run([*argv, "--filter", "label=bigbrane.owner=bigbrane",
                        "--filter", "label=bigbrane.run=" + run_id]).stdout
    return out.split()


@podman
def test_template_and_gru_through_podman(work_dir, template_dir):
    from hybrid.executor.executor import Executor
    run_id = "bench-test-" + STAMP.lower()
    executor = Executor(run_id, work_dir / "receipts")
    cache = work_dir / "cache"
    first = evaluate_candidate(template_dir, TINY, work_dir / "run1", executor, baselines=["gru"], cache_dir=cache,
                               claimed_exponent=0.0, log=lambda _: None)
    digest = read_lock()["digest"]
    assert first["trusted"] is True and first["host"]["mode"] == "podman"
    for name in ("candidate", "gru"):
        entry = first["entrants"][name]
        assert entry["status"] == "ok", entry
        receipt = first["host"]["receipts"][name]
        assert receipt["status"] == "ok" and receipt["image_digest"] == digest
        for task, block in entry["tasks"].items():
            for n, m in block["sizes"].items():
                assert m["valid"] and m["kl"] >= 0 and 0 <= m["acc"] <= 1, (name, task, n)
        assert entry["scaling"]["rss_method"] == "linux_fork_vmhwm"
        assert len(entry["scaling"]["lengths"]) == 4
    assert first["entrants"]["gru"]["scaling"]["verdict"] in ("match", "mismatch")
    assert first["host"]["evaluator_receipt"]["status"] == "ok"
    assert set(first["comparison"]["candidate"]) == {"gru"}
    assert first["entrants"]["candidate"]["tasks"]["hmm"]["sizes"]["32"]["kl"] > 0
    on_disk = json.loads((work_dir / "run1" / "bench" / "bench_result.json").read_text())
    assert on_disk["trusted"] is True and (work_dir / "run1" / "bench" / "manifest.json").is_file()

    second = evaluate_candidate(template_dir, TINY, work_dir / "run2", executor, baselines=["gru"], cache_dir=cache,
                                log=lambda _: None)
    assert second["host"]["timings_s"]["candidate_jobs"] > 0
    assert second["entrants"]["gru"]["job_id"] == first["entrants"]["gru"]["job_id"]   # from the cache
    assert second["entrants"]["gru"]["tasks"] == first["entrants"]["gru"]["tasks"]
    assert second["entrants"]["candidate"]["tasks"] == first["entrants"]["candidate"]["tasks"]  # deterministic
    assert labeled("containers", run_id) == [] and labeled("volumes", run_id) == []
