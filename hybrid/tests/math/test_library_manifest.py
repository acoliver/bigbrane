"""Check-hash manifest (plan section 5.5) and the library job path: the committed MANIFEST.json matches every file
under hybrid/math, edits are caught on the host and in the staged job, check modules import nothing outside the
stdlib, NumPy and SymPy, and check_library stages, replays and records hashes. The selfcheck battery runs on the
host here; test_library_live.py runs it in the tools image."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid.math import manifest, references, selfcheck
from hybrid.math.runner import LIBRARY_CAPS, LIBRARY_FILES, MathRunner

IMAGE = "localhost/bigbrane/tools@sha256:" + "b" * 64
STDLIB_OK = {"__future__", "argparse", "ast", "builtins", "collections", "dataclasses", "fractions", "hashlib",
             "itertools", "json", "math", "operator", "random", "re", "sys", "time", "types", "pathlib"}
# z3-solver is in the tools image; only the frozen-statement checks use it, imported where it is used.
Z3_USERS = {"spec_checks.py"}


def test_committed_manifest_matches_every_math_file():
    recorded = manifest.verify()
    assert set(recorded) == {p.name for p in manifest.MATH_DIR.glob("*.py")}
    for name in ("conservation.py", "cost.py", "constructions.py", "magic_x.py", "tiers.py", "jobmain.py",
                 "lean_check.py", "runner.py"):
        assert name in recorded


@pytest.fixture
def math_copy(tmp_path):
    directory = tmp_path / "math"
    directory.mkdir()
    for path in manifest.MATH_DIR.glob("*.py"):
        shutil.copy(path, directory / path.name)
    shutil.copy(manifest.MANIFEST_PATH, directory / "MANIFEST.json")
    return directory


def test_an_edited_check_is_caught(math_copy):
    with open(math_copy / "conservation.py", "a") as handle:
        handle.write("\n# edited by a prover\n")
    with pytest.raises(manifest.ManifestMismatch) as exc:
        manifest.verify(math_copy, math_copy / "MANIFEST.json")
    assert exc.value.problems == ["conservation.py: changed"]


def test_added_and_removed_files_are_caught(math_copy):
    (math_copy / "shortcut.py").write_text("PASS = True\n")
    (math_copy / "tiers.py").unlink()
    with pytest.raises(manifest.ManifestMismatch) as exc:
        manifest.verify(math_copy, math_copy / "MANIFEST.json")
    assert sorted(exc.value.problems) == ["shortcut.py: not in the manifest", "tiers.py: missing"]


def test_write_then_verify_round_trips(math_copy):
    (math_copy / "tiers.py").write_text("# changed\n")
    path = math_copy / "M.json"
    manifest.write(math_copy, path)
    assert manifest.verify(math_copy, path)["tiers.py"] == manifest.sha256_file(math_copy / "tiers.py")


def staged_imports(source: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module.split(".")[0]


@pytest.mark.parametrize("staged,source", sorted(LIBRARY_FILES.items()))
def test_staged_modules_import_only_stdlib_numpy_sympy_and_each_other(staged, source):
    text = (manifest.MATH_DIR / source).read_text()
    outside = set(staged_imports(text)) - STDLIB_OK - {"numpy", "sympy", "bbmath"} - (
        {"z3"} if source in Z3_USERS else set())
    assert outside == set(), f"{source} imports {outside}; it must run in the tools image as bbmath"


class HostExecutor:
    """Runs each staged job's main.py with the host Python, in a copy of the staged inputs, as the image would."""

    def __init__(self, tmp_path):
        self.tmp_path, self.jobs = tmp_path, []

    def run(self, job):
        self.jobs.append(job)
        work = self.tmp_path / "run" / job.job_id
        shutil.copytree(job.input_dir, work)
        proc = subprocess.run([sys.executable, "-B", str(work / "main.py")], capture_output=True, text=True,
                              cwd=work, timeout=600)
        out, err = self.tmp_path / f"{job.job_id}.stdout", self.tmp_path / f"{job.job_id}.stderr"
        out.write_text(proc.stdout)
        err.write_text(proc.stderr)
        return SimpleNamespace(status="ok" if proc.returncode == 0 else "candidate_error", detail="",
                               exit_code=proc.returncode, run_s=0.1, stdout_path=str(out), stderr_path=str(err))


def test_check_library_stages_verified_files_replays_and_records_hashes(tmp_path):
    executor = HostExecutor(tmp_path)
    runner = MathRunner(executor, IMAGE, tmp_path / "work")
    request = {"kind": "conservation", "claims": {"state_bits": 256,
                                                  "recall": [{"K": 64, "V": 256, "accuracy": 0.99}]}}
    record = runner.check_library(request)
    assert (record["status"], record["decided"], record["replayed"]) == ("violated", True, True)
    assert record["counterexample"]["smallest_failing_K"] == 33
    assert len(executor.jobs) == 2
    job = executor.jobs[0]
    assert job.command == ("python3", "/in/main.py") and job.caps == LIBRARY_CAPS and job.image == IMAGE
    staged = sorted(p.relative_to(job.input_dir).as_posix() for p in job.input_dir.rglob("*") if p.is_file())
    assert staged == sorted([*LIBRARY_FILES, "manifest.json", "request.json"])
    recorded = manifest.verify()
    assert record["check_hashes"] == {staged: recorded[source] for staged, source in LIBRARY_FILES.items()}
    assert record["kind"] == "conservation"


def test_check_library_cost_request_catches_the_planted_exponent_error(tmp_path):
    runner = MathRunner(HostExecutor(tmp_path), IMAGE, tmp_path / "work")
    record = runner.check_library({"kind": "cost", "source": references.PLANTED_QUADRATIC,
                                   "claim": references.REFERENCES["planted_quadratic"][1], "ns": [16, 32, 64]})
    assert record["status"] == "violated" and record["decided"]
    assert record["counterexample"]["check"] == "exponent_n"


def test_check_library_refuses_before_staging_when_the_manifest_differs(tmp_path, monkeypatch):
    bad = tmp_path / "MANIFEST.json"
    data = json.loads(manifest.MANIFEST_PATH.read_text())
    data["files"]["cost.py"] = "0" * 64
    bad.write_text(json.dumps(data))
    monkeypatch.setattr(manifest, "MANIFEST_PATH", bad)
    executor = HostExecutor(tmp_path)
    runner = MathRunner(executor, IMAGE, tmp_path / "work")
    with pytest.raises(manifest.ManifestMismatch, match="cost.py: changed"):
        runner.check_library({"kind": "selfcheck"})
    assert executor.jobs == [] and not (tmp_path / "work").exists()


def test_check_library_rejects_unknown_kinds(tmp_path):
    with pytest.raises(ValueError):
        MathRunner(HostExecutor(tmp_path), IMAGE, tmp_path).check_library({"kind": "judge"})


def test_staged_driver_refuses_a_file_edited_after_staging(tmp_path):
    runner = MathRunner(None, IMAGE, tmp_path)
    files, _ = runner.library_files({"kind": "selfcheck"})
    for name, text in files.items():
        (tmp_path / "in" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "in" / name).write_text(text)
    with open(tmp_path / "in" / "bbmath" / "cost.py", "a") as handle:
        handle.write("\n# edited\n")
    proc = subprocess.run([sys.executable, "-B", str(tmp_path / "in" / "main.py")], capture_output=True, text=True)
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert proc.returncode == 2 and report["status"] == "unknown"
    assert "bbmath/cost.py: changed" in report["evidence"]


def test_selfcheck_battery_on_the_host():
    result = selfcheck.run(quick=True)
    failed = [(c["case"], c["expected"], c["got"]) for c in result["cases"] if not c["ok"]]
    assert failed == [] and result["status"] == "discharged"
    names = {c["case"] for c in result["cases"]}
    assert {"storage_planted_overrun", "recall_planted_capacity_overrun", "cost_planted_quadratic",
            "cost_planted_quadratic_caught_by_exponent", "magic_x_oracle_alone_gives_advantage"} <= names
