"""MathRunner with a fake executor: job staging, two-run replay, host-side rejection, program checks.

The live version (real Podman, tools image) is test_tools_image_live.py, enabled by BIGBRANE_TOOLS_PODMAN=1."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid.math import lean_check as lc
from hybrid.math.runner import CHECK_FILE, LEAN_CAPS, PROGRAM_CAPS, MathRunner

IMAGE = "localhost/bigbrane/tools@sha256:" + "a" * 64
STATEMENT = "def Claim : Prop := ∀ n : Nat, n + 0 = n\n"
PROOF = "theorem claim_holds : Claim := fun _ => rfl\n"


class FakeExecutor:
    """Answers each job with the next scripted (status, exit_code, stdout) entry. "{nonce}" in stdout is filled
    from the staged Check.lean, as the real trailer would print it."""

    def __init__(self, tmp_path, script):
        self.tmp_path, self.script, self.jobs = tmp_path, list(script), []

    def run(self, job):
        self.jobs.append(job)
        status, exit_code, stdout = self.script.pop(0)
        staged = {p.name: p.read_text() for p in Path(job.input_dir).iterdir()}
        if CHECK_FILE in staged:
            nonce = re.search(r"bigbrane_check_([0-9a-f]+)", staged[CHECK_FILE]).group(1)
            stdout = stdout.replace("{nonce}", nonce)
        out, err = self.tmp_path / f"{job.job_id}.stdout", self.tmp_path / f"{job.job_id}.stderr"
        out.write_text(stdout)
        err.write_text("")
        return SimpleNamespace(status=status, detail="", exit_code=exit_code, run_s=1.5, stdout_path=str(out),
                               stderr_path=str(err))


PASS_OUT = "'claim_holds' does not depend on any axioms\nBIGBRANE_FACTS {nonce} kind=theorem exact=true claim_prop=true axioms=[]\n"
SORRY_OUT = ("'claim_holds' depends on axioms: [sorryAx]\n"
             "BIGBRANE_FACTS {nonce} kind=theorem exact=true claim_prop=true axioms=[sorryAx]\n")


def runner(tmp_path, script):
    executor = FakeExecutor(tmp_path, script)
    return MathRunner(executor, IMAGE, tmp_path / "work"), executor


def check(statement=STATEMENT, proof=PROOF):
    return lc.LeanCheck(statement, lc.statement_hash(STATEMENT), proof)


def test_lean_pass_needs_two_agreeing_runs_with_fresh_nonces(tmp_path):
    math, executor = runner(tmp_path, [("ok", 0, PASS_OUT), ("ok", 0, PASS_OUT)])
    result = math.check_lean(check())
    assert (result.status, result.decided, result.axioms) == ("pass", True, [])
    assert len(executor.jobs) == 2 and result.runs[0].nonce != result.runs[1].nonce
    job = executor.jobs[0]
    assert job.command == ("lean-check", f"/in/{CHECK_FILE}") and job.caps == LEAN_CAPS and job.image == IMAGE
    staged = (Path(job.input_dir) / CHECK_FILE).read_text()
    assert staged == lc.assemble(check(), result.runs[0].nonce)


def test_lean_sorry_rejected_and_decided(tmp_path):
    math, _ = runner(tmp_path, [("ok", 0, SORRY_OUT), ("ok", 0, SORRY_OUT)])
    result = math.check_lean(check(proof="theorem claim_holds : Claim := by sorry\n"))
    assert (result.status, result.decided) == ("reject", True)
    assert "uses sorry (sorryAx)" in result.reasons


def test_hash_mismatch_and_forbidden_constructs_never_reach_the_executor(tmp_path):
    math, executor = runner(tmp_path, [])
    restated = math.check_lean(check(statement="def Claim : Prop := True\n"))
    forbidden = math.check_lean(check(proof="#eval 1\n" + PROOF))
    assert restated.status == forbidden.status == "reject" and executor.jobs == []
    assert restated.reasons[0].startswith("statement hash mismatch")
    assert forbidden.reasons == ["proof: forbidden construct '#eval'"]


def test_lean_replay_disagreement_is_unknown(tmp_path):
    math, _ = runner(tmp_path, [("ok", 0, PASS_OUT), ("ok", 0, SORRY_OUT)])
    result = math.check_lean(check())
    assert (result.status, result.decided) == ("unknown", False)
    assert result.reasons[0].startswith("replay disagreed")


@pytest.mark.parametrize("bad", ["infra_error", "timeout", "oom"])
def test_lean_executor_failure_is_unknown(tmp_path, bad):
    math, executor = runner(tmp_path, [(bad, -1, "")])
    result = math.check_lean(check())
    assert (result.status, result.decided) == ("unknown", False) and len(executor.jobs) == 1


def report(status, counterexample=None):
    return json.dumps({"status": status, "evidence": "e", "counterexample": counterexample}) + "\n"


def test_program_decided_only_after_agreeing_replay(tmp_path):
    math, executor = runner(tmp_path, [("ok", 0, report("violated", {"x": 3})),
                                       ("ok", 0, report("violated", {"x": 3}))])
    record = math.check_program("z3", "print('x')")
    assert (record["status"], record["decided"], record["replayed"], record["tool"]) == ("violated", True, True, "z3")
    assert record["counterexample"] == {"x": 3}
    assert executor.jobs[0].command == ("python3", "/in/main.py") and executor.jobs[0].caps == PROGRAM_CAPS


def test_program_replay_disagreement_is_unknown(tmp_path):
    math, _ = runner(tmp_path, [("ok", 0, report("violated", {"x": 3})), ("ok", 0, report("violated", {"x": 4}))])
    record = math.check_program("z3", "p")
    assert (record["status"], record["decided"]) == ("unknown", False)


def test_program_unknown_is_not_replayed(tmp_path):
    math, executor = runner(tmp_path, [("ok", 0, report("unknown"))])
    record = math.check_program("sympy", "p")
    assert record["status"] == "unknown" and len(executor.jobs) == 1


def test_program_tool_must_be_known(tmp_path):
    math, _ = runner(tmp_path, [])
    with pytest.raises(ValueError):
        math.check_program("bash", "p")
