"""FIX-2 regressions for profile obligation P4 (toy learnability), from the 2026-10-13 calibration
(results/phase4-20261013T073748Z/FIX-2.md and p4-timing/REPORT.md).

Recorded fixtures in fixtures/calibration_live_20261013/:

- k1_learnability_timeout.json: card k1's O9.a check (runs/calibration-math-20261013T120132Z/k1-run1), a full
  learnability job that ran out of its 1800 s cap with empty evidence, the two learners the prover wrote for O9.a and
  O10.a (the same frozen experiment under two claims), the receipt, and the host timing of the same learner on one
  seed and 300 examples (28.90 s true training, 28.73 s wrong-answer control).
- k3_learner_isinstance.json: card k3's learner that failed with NameError on isinstance.
- k2_frozen_statements.json and k2_prover_declined.txt (FIX-1 fixtures): k2's frozen statements, where O10.a
  (sample_efficiency.prediction) and O11.a (P4.copy) are the same experiment, and the reply that left both learners
  null.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hybrid.math import cost, learnability, profile
from hybrid.math.runner import LIBRARY_CAPS, JobHangGuard, MathRunner, cpu_capped
from hybrid.research import calibration
from hybrid.research.stages import m_math
from hybrid.research.stages.common import load_prompt, load_section
from hybrid.tests.research import calibration_fakes as cf
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_profile_stage import Harness

LIVE = Path(__file__).parent / "fixtures" / "calibration_live_20261013"
STUBS = ("RECIPE = {\"epochs\": 0}\n"  # FIX-3: every learner declares its training recipe
         "def learn(layout, examples, seed):\n    return {}\n"
         "def init(layout, params):\n    return {}\n"
         "def update(state, token):\n    return state\n"
         "def predict(state):\n    return 0\n")


def live(name: str):
    return json.loads((LIVE / name).read_text(encoding="utf-8"))


def k1_timing(examples: int | None = None) -> dict:
    """The host phases of k1's learner on one seed and 300 examples, as a probe would print them. The recorded
    seconds were measured single-threaded on an idle host, where wall and CPU time agree; decision CT reads them as
    CPU seconds."""
    measured = live("k1_learnability_timeout.json")["fraction_timing"]
    phases = {phase: 0.0 for phase in learnability.SCALED_PHASES + learnability.FIXED_PHASES}
    phases.update(train=measured["train_s"], train_empty=measured["train_empty_s"],
                  train_wrong=measured["train_wrong_s"], test=measured["one_test_pass_s"],
                  test_empty=measured["one_test_pass_s"], test_wrong=measured["one_test_pass_s"])
    return {"seed": 0, "examples": examples or measured["examples"], "clock": "process_cpu",
            "total_cpu_s": sum(phases.values()), "phases_cpu": phases,
            "total_wall_s": sum(phases.values()), "phases_wall": dict(phases), "probe_accuracy": 0.003906}


def cap_report(job, kind: str, cpu_s: float) -> str:
    """What the driver prints when RLIMIT_CPU stops a job at its cap (jobmain.cap_report)."""
    cap = json.loads((Path(job.input_dir) / "request.json").read_text())["cpu_cap_s"]
    return json.dumps({"status": "unknown", "counterexample": None,
                       "evidence": f"CPU cap reached: the {kind} job used {cpu_s:.1f} s of process CPU time",
                       "cpu_cap": {"cpu_s": cpu_s, "wall_s": 2.5 * cpu_s, "cap_cpu_s": cap, "kind": kind}}) + "\n"


class ProbeExecutor(cf.FastExecutor):
    """FastExecutor whose learnability jobs for ``task`` follow a script: the probe prints ``timing`` (scaled to the
    probe's example count) or stops with ``probe_status``; the full job stops with ``full_status``. Other jobs run
    the staged driver as usual. Decision CT: status ``cpu_cap`` answers the driver's cap report (RLIMIT_CPU soft
    limit), ``cpu_kill`` the hard limit's SIGKILL (exit 137, no report), and ``timeout`` the executor's wall-clock hang
    guard."""

    def __init__(self, tmp_path, *, task="copy", timing=None, probe_status="ok", full_status="ok"):
        super().__init__(tmp_path)
        self.task, self.timing, self.probe_status, self.full_status = task, timing, probe_status, full_status

    def requests(self) -> list[dict]:
        return [json.loads((Path(job.input_dir) / "request.json").read_text()) for job in self.jobs]

    def _stopped(self, job, status):
        out, err = self.tmp_path / f"{job.job_id}.stdout", self.tmp_path / f"{job.job_id}.stderr"
        out.write_text("")
        err.write_text("")
        return SimpleNamespace(status=status, detail=f"wall cap {job.caps.wall_s:.0f}s exceeded", exit_code=137,
                               run_s=job.caps.wall_s, stdout_path=str(out), stderr_path=str(err))

    def _scripted_stop(self, job, status, kind):
        self.jobs.append(job)
        if status == "cpu_cap":
            return self._answer(job, cap_report(job, kind, job.caps.cpu_s + 0.4))
        if status == "cpu_kill":
            return dict_namespace(self._stopped(job, "candidate_error"), detail="", exit_code=137)
        return self._stopped(job, status)

    def run(self, job):
        request = json.loads((Path(job.input_dir) / "request.json").read_text())
        scripted = request.get("spec", {}).get("task") == self.task if isinstance(request.get("spec"), dict) else False
        if scripted and request["kind"] == "learnability_probe":
            if self.probe_status != "ok":
                return self._scripted_stop(job, self.probe_status, request["kind"])
            if self.timing is not None:
                self.jobs.append(job)
                timing = dict(self.timing, examples=learnability.probe_examples(request["spec"]))
                printed = {"status": "unknown", "evidence": "timing probe only, decides nothing", "probe": True,
                           "counterexample": None, "tier": "executed", "timing": timing}
                return self._answer(job, json.dumps(printed) + "\n")
        if scripted and request["kind"] == "learnability" and self.full_status != "ok":
            return self._scripted_stop(job, self.full_status, request["kind"])
        return super().run(job)


def dict_namespace(namespace, **changes):
    return SimpleNamespace(**{**vars(namespace), **changes})


def copy_learnability(executor) -> list[str]:
    return [r["kind"] for r in executor.requests() if r["kind"].startswith("learnability")
            and r["spec"]["task"] == "copy"]


def kinds(executor) -> list[str]:
    return [r["kind"] for r in executor.requests() if r["kind"].startswith("learnability")]


def k1_request() -> dict:
    fixture = live("k1_learnability_timeout.json")
    return {"kind": "learnability", "spec": fixture["spec"], "source": fixture["learners"]["O9.a"]}


# ----- 1. budget: the runner's caps and the control trainings, stated to the fixer and the prover -----------------

def test_the_fixer_template_states_the_runner_caps_and_the_cost_of_the_controls():
    text = profile.template_text(LIBRARY_CAPS)
    # Decision CT: the budget is the 1800 s CPU cap; the wall clock is a hang guard at 3 x (1800 + 30 s grace).
    assert "1800 s of process CPU time, 4096 MB of memory and 1 CPU" in text
    assert "a busy host does not count against it" in text and "wall-clock guard of 5490 s" in text
    assert "on no examples (control)" in text and "random answers (control)" in text
    assert "6 trainings on m examples, 3 on none and 9 test passes" in text
    assert "A projection above 1440 CPU s (80% of the cap)" in text
    assert "One training on m examples therefore has about 240 s of CPU time on one core" in text
    # FIX-3 moved the recipe to the prover; the budget now binds the prover's recipe instead of a frozen schedule.
    assert "The prover's recipe must fit about 240 s of CPU time per training on m examples on one core" in text
    assert "fewer than 3 seeds, lowering the threshold below 0.99 or dropping a control" in text
    # Read from the caps given, not from a literal.
    other = profile.template_text(cpu_capped(600, memory_mb=2048, pids=64))
    assert "600 s of process CPU time, 2048 MB" in other and "has about 80 s of CPU time on one core" in other
    assert "4096 MB" not in other and "wall-clock guard of 1890 s" in other


def test_the_prompts_ask_for_a_schedule_that_fits_and_for_batched_linear_algebra():
    section = load_section("profile_obligations")
    assert "chosen by the prover within the P4 budget stated below" in section
    assert "Never change m, the seeds, the threshold or the controls to fit the budget" in section
    prompt = load_prompt("prover")
    assert "- **learnability**: `source` is Python defining `RECIPE`" in prompt
    assert "`learn(layout, examples, seed)` returning parameters" in prompt
    assert "np.einsum(..., optimize=True)" in prompt and "no Python loop over examples or tokens" in prompt
    assert "A recurrence may loop over time steps" in prompt
    assert "LEARNABILITY BUDGET section" in load_section("profile_prover")


def test_the_stage_states_the_budget_of_its_own_runner_to_both_roles(tmp_path):
    h = Harness(tmp_path, "k1", protocols=False)
    h.ctx.math.library_caps = cpu_capped(900, memory_mb=2048, pids=64)
    h.m()
    fixer = next(c for c in h.llm.calls if c["prompt"] == "statement_fixer")
    assert "900 s of process CPU time, 2048 MB" in fixer["user"]
    assert "about 120 s of CPU time on one core" in fixer["user"]
    learners = [c["user"] for c in h.llm.calls if c["prompt"] == "prover" and '"form": "learnability"' in c["user"]]
    assert learners and all("LEARNABILITY BUDGET" in u and "900 s of process CPU time" in u for u in learners)
    others = [c["user"] for c in h.llm.calls if c["prompt"] == "prover" and '"form": "learnability"' not in c["user"]]
    assert others and not any("LEARNABILITY BUDGET (the runner's caps)" in u for u in others)


# ----- 2. a timeout reaches the next round with its reason and cost, and stays unknown ----------------------------

def test_the_recorded_k1_timeout_check_is_read_back_with_its_reason():
    fixture = live("k1_learnability_timeout.json")
    check = fixture["check"]
    assert check["evidence"] == "" and check["status"] == "unknown" and check["decided"] is False
    assert fixture["receipt"]["status"] == "timeout" and fixture["receipt"]["exit_code"] == 137
    text = m_math.failure_text(check)
    assert "first run timeout: wall cap 1800s exceeded" in text
    assert "job mathlib-f5484d3ea965: executor status timeout: wall cap 1800s exceeded" in text
    assert text.strip() != "status unknown:"


def fitting_timing() -> dict:
    fitting = k1_timing()
    fitting["phases_cpu"] = dict(fitting["phases_cpu"], train=10.0, train_wrong=10.0)   # projects about 600 CPU s
    return fitting


def test_a_full_job_that_reaches_the_cpu_cap_is_a_resource_failure_with_the_cap_and_the_projection(tmp_path):
    executor = ProbeExecutor(tmp_path / "exec", timing=fitting_timing(), full_status="cpu_cap")
    record = MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(k1_request())
    assert kinds(executor) == ["learnability_probe", "learnability"]   # an unknown report is not replayed
    assert executor.requests()[1]["cpu_cap_s"] == 1800 and executor.jobs[1].caps.wall_s == 5490
    assert (record["status"], record["decided"], record["counterexample"]) == ("unknown", False, None)
    resource = record["resource"]
    assert resource["kind"] == "cpu_cap" and resource["cap_cpu_s"] == 1800 and resource["cpu_s"] == 1800.4
    assert resource["wall_cap_s"] == 5490 and 590 < resource["projected_cpu_s"] < 620
    assert "reached its CPU cap of 1800 s of process CPU time" in record["evidence"]
    assert "1800.4 CPU s" in record["evidence"]
    assert "projected full check 60" in record["evidence"] and "not a pass and not a counterexample" in record["evidence"]
    check = {"status": record["status"], "evidence": record["evidence"], "reasons": record["problems"],
             "runs": record["runs"], "probe": record["probe"], "resource": resource}
    feedback = m_math.failure_text(check)
    assert "CPU time: training on true answers 10.00 s" in feedback and "resource failure" in feedback


def test_a_full_job_killed_at_the_hard_cpu_limit_is_the_same_cap_stop(tmp_path):
    executor = ProbeExecutor(tmp_path / "exec", timing=fitting_timing(), full_status="cpu_kill")
    record = MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(k1_request())
    assert record["status"] == "unknown" and record["resource"]["kind"] == "cpu_cap"
    assert record["resource"]["cpu_s"] is None and "reached its CPU cap of 1800 s" in record["evidence"]


def test_a_full_job_stopped_by_the_wall_hang_guard_is_an_environment_failure_not_a_resource_failure(tmp_path):
    executor = ProbeExecutor(tmp_path / "exec", timing=fitting_timing(), full_status="timeout")
    with pytest.raises(JobHangGuard) as exc:
        MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(k1_request())
    assert exc.value.kind == "learnability" and exc.value.caps.wall_s == 5490 and exc.value.caps.cpu_s == 1800
    assert "environment failure" in str(exc.value) and "hang guard of 5490 s" in str(exc.value)


# ----- 3. the timing probe: one seed, about 300 examples, controls counted, never deciding ------------------------

def test_the_projection_counts_both_controls_and_every_seed():
    spec = live("k1_learnability_timeout.json")["spec"]
    timing = k1_timing()
    measured = live("k1_learnability_timeout.json")["fraction_timing"]
    expected = 3 * ((measured["train_s"] + measured["train_wrong_s"]) * 10 + measured["train_empty_s"]
                    + 3 * measured["one_test_pass_s"])
    assert learnability.projected_cpu_seconds(spec, timing) == pytest.approx(expected)
    assert 1700 < expected < 1800   # REPORT.md: about 1729 to 1792 s against the 1800 s cap
    without_control = dict(timing, phases_cpu=dict(timing["phases_cpu"], train_wrong=0.0))
    assert learnability.projected_cpu_seconds(spec, without_control) < 0.55 * expected
    one_seed = dict(spec, seeds=[0])
    assert learnability.projected_cpu_seconds(one_seed, timing) == pytest.approx(expected / 3)
    # Decision CT: the wall phases are information only; inflating them changes nothing.
    contended = dict(timing, phases_wall={k: 3.5 * v for k, v in timing["phases_wall"].items()})
    assert learnability.projected_cpu_seconds(spec, contended) == pytest.approx(expected)


def test_the_recorded_k1_cost_is_refused_before_any_full_job(tmp_path):
    executor = ProbeExecutor(tmp_path / "exec", timing=k1_timing())
    record = MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(k1_request())
    assert kinds(executor) == ["learnability_probe"]
    probe_job = executor.jobs[0]
    assert probe_job.job_id.startswith("mathprobe-") and probe_job.caps.cpu_s == pytest.approx(480.0)
    assert probe_job.caps.wall_s == pytest.approx(1530.0)   # the probe's hang guard: 3 x (480 + 30 s grace)
    assert executor.requests()[0]["cpu_cap_s"] == pytest.approx(480.0)
    assert (record["status"], record["decided"], record["runs"]) == ("unknown", False, [])
    resource = record["resource"]
    assert resource["kind"] == "projected_overrun" and resource["fit_cpu_s"] == pytest.approx(1440.0)
    assert 1700 < resource["projected_cpu_s"] < 1800
    for phrase in ("projects 17", "above the 1440 CPU s admitted (80% of the 1800 s CPU cap)",
                   "CPU time: training on true answers 28.90 s",
                   "wrong-answer control 28.73 s", "np.einsum with optimize=True",
                   "The seeds, m, the threshold and both data controls stay as they are"):
        assert phrase in record["evidence"], phrase
    assert record["probe"]["job_id"] == probe_job.job_id and record["probe"]["timing"]["examples"] == 300


def test_a_probe_that_reaches_its_own_cpu_cap_refuses_the_full_job(tmp_path):
    executor = ProbeExecutor(tmp_path / "exec", probe_status="cpu_cap")
    record = MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(k1_request())
    assert kinds(executor) == ["learnability_probe"]
    resource = record["resource"]
    assert resource["kind"] == "probe_cpu_cap" and resource["projected_at_least_cpu_s"] == 1440
    assert resource["probe_cpu_s"] == pytest.approx(480.4)
    assert "reached its own CPU cap of 480 s" in record["evidence"] and record["status"] == "unknown"


def test_a_probe_stopped_by_the_wall_hang_guard_raises_an_environment_failure(tmp_path):
    executor = ProbeExecutor(tmp_path / "exec", probe_status="timeout")
    with pytest.raises(JobHangGuard) as exc:
        MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(k1_request())
    assert exc.value.kind == "learnability_probe" and exc.value.caps.wall_s == pytest.approx(1530.0)
    assert kinds(executor) == ["learnability_probe"]


def test_the_probe_decides_nothing_and_a_learner_that_fits_gets_the_full_check(tmp_path):
    spec = {"task": "copy", "size": 3, "examples": 400, "seeds": [0, 1, 2], "test_count": 40}
    probe = learnability.timing_probe(spec, cf.LEARNERS["copy"])
    assert probe["status"] == "unknown" and probe["probe"] and probe["timing"]["examples"] == 300
    assert probe["timing"]["seed"] == 0 and probe["timing"]["probe_accuracy"] >= 0.99   # a perfect learner, still unknown
    assert set(probe["timing"]["phases_cpu"]) == set(learnability.SCALED_PHASES + learnability.FIXED_PHASES)
    assert set(probe["timing"]["phases_wall"]) == set(probe["timing"]["phases_cpu"])
    assert probe["projected_cpu_s"] > 0 and "decides nothing" in probe["evidence"]
    executor = cf.FastExecutor(tmp_path / "exec")
    record = MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(
        {"kind": "learnability", "spec": spec, "source": cf.LEARNERS["copy"]})
    staged = [json.loads((Path(j.input_dir) / "request.json").read_text())["kind"] for j in executor.jobs]
    assert staged == ["learnability_probe", "learnability", "learnability"]
    assert (record["status"], record["decided"], record["resource"]) == ("discharged", True, None)
    assert record["probe"]["job_id"] not in [run["job_id"] for run in record["runs"]]


def test_a_learner_that_crashes_in_the_probe_is_reported_without_a_full_job(tmp_path):
    executor = cf.FastExecutor(tmp_path / "exec")
    spec = {"task": "copy", "size": 3, "examples": 40, "seeds": [0, 1, 2], "test_count": 40}
    record = MathRunner(executor, mf.IMAGE, tmp_path / "work").check_library(
        {"kind": "learnability", "spec": spec, "source": cf.LEARNERS["copy"].replace("def learn(layout, examples, seed):",
                                                                                     "def learn(layout, examples, seed):\n    1 / 0")})
    assert [json.loads((Path(j.input_dir) / "request.json").read_text())["kind"] for j in executor.jobs] == [
        "learnability_probe"]
    assert record["status"] == "unknown" and record["prover_fault"] and "ZeroDivisionError" in record["evidence"]


def overrun_harness(tmp_path) -> tuple[Harness, ProbeExecutor]:
    """Card k1 with the profile; the copy learner's probe prints k1's recorded phases, and the runner's CPU cap is 200 s
    (FIX-16: m is k1's stated 3000, so the recorded phases scaled from the probe's 300 examples project well above
    160 s)."""
    h = Harness(tmp_path, "k1", protocols=False)
    executor = ProbeExecutor(tmp_path / "exec2", timing=k1_timing())
    h.ctx.math = MathRunner(executor, mf.IMAGE, h.run_dir, library_caps=cpu_capped(200, memory_mb=4096, pids=64))
    return h, executor


def test_at_stage_m_a_projected_overrun_leaves_p4_unknown_and_reaches_the_next_round(tmp_path):
    h, executor = overrun_harness(tmp_path)
    h.m()
    copy_kinds = [r["kind"] for r in executor.requests() if r.get("spec", {}).get("task") == "copy"
                  and r["kind"].startswith("learnability")]
    # Round 1 probes and refuses; round 2 resends the same source, refused without a job. Never a full job.
    assert copy_kinds == ["learnability_probe"]
    slot = h.slot("M", "P4.copy")
    assert slot["status"] not in ("failed", "refuted", "observed")
    codes = h.verdict("M")["kill_codes"]
    assert "unproven:P4.copy" in codes and "obligation_failed:P4.copy" not in codes
    check = h.verdict("M")["checks"]["O7.a"]
    assert (check["status"], check["decided"]) == ("unknown", False)
    assert check["resource"]["kind"] == "unchanged_source"   # round 2 resent round 1's refused source
    probes = [e["data"] for e in h.ledger.iter_events("math_probe") if e["data"]["statement_id"] == "O7.a"]
    assert len(probes) == 1 and probes[0]["decides"] is False
    assert probes[0]["projected_cpu_s"] > probes[0]["fit_cpu_s"] and probes[0]["cap_cpu_s"] == 200
    retry = [c["user"] for c in h.llm.calls if c["prompt"] == "prover" and '"statement_id": "O7.a"' in c["user"]][1]
    assert "previous_attempt_failed" in retry and "resource failure" in retry
    assert "above the 160 CPU s admitted (80% of the 200 s CPU cap)" in retry
    assert "training on true answers 28.90 s" in retry


# ----- 4. an experiment that ran out of its cap is not run again at the same cost ---------------------------------

def k2_history():
    record = live("k2_frozen_statements.json")
    statements = {s["id"]: s for s in m_math.all_statements(record)}
    return m_math.ResourceHistory(record), statements


def test_two_claims_with_the_same_frozen_experiment_share_one_history_entry():
    fixture = live("k1_learnability_timeout.json")["same_experiment"]
    assert fixture["O9.a"]["sha256"] == fixture["O10.a"]["sha256"]
    assert fixture["O9.a"]["definitions"] == fixture["O10.a"]["definitions"]
    assert fixture["O9.a"]["claim_id"] != fixture["O10.a"]["claim_id"]
    history, statements = k2_history()
    assert history.key(statements["O10.a"]) == history.key(statements["O11.a"])
    assert history.key(statements["O10.a"]) != history.key(statements["O1.a"])


def test_after_a_cpu_cap_stop_an_equivalent_experiment_needs_a_probe_showing_a_lower_cost():
    history, statements = k2_history()
    learner = live("k1_learnability_timeout.json")["learners"]
    assert history.admit(statements["O11.a"]) is None and history.unchanged(statements["O11.a"], learner["O9.a"]) is None
    history.note(statements["O10.a"], {"attempt": 1, "resource": {"kind": "cpu_cap", "projected_cpu_s": 1300.0}},
                 learner["O9.a"])
    admit = history.admit(statements["O11.a"])
    same_cost = admit({"projected_cpu_s": 1250.0, "cap_cpu_s": 1800})
    assert "O10.a attempt 1" in same_cost and "projected 1300 CPU s" in same_cost and "1250 CPU s" in same_cost
    assert "reached its 1800 s CPU cap" in same_cost
    assert admit({"projected_cpu_s": 1000.0, "cap_cpu_s": 1800}) is None   # at most 80% of 1300 CPU s
    assert "this exact source already stopped" in history.unchanged(statements["O11.a"], learner["O9.a"])
    assert history.unchanged(statements["O11.a"], learner["O10.a"]) is None   # the second k1 source differs


def two_copy_statements(tmp_path, *, variant: bool):
    """Card k1 whose P4.copy has two copy statements of the same experiment; the full copy job reaches its CPU cap
    after a probe that projects it fits. With ``variant`` the prover sends a cosmetically different second source."""
    card = cf.without_protocols(calibration.load_cards(("k1",))["k1"])
    fixer = cf.fixer_output("k1", card)
    obligation = next(o for o in fixer["obligations"] if o["claim_id"] == "P4.copy")
    # FIX-16: both statements copy k1's stated P4 copy case (resources.learning.copy).
    obligation["statements"] = [cf.learnability("copy", card=card), cf.learnability("copy", card=card)]
    h = Harness(tmp_path, "k1", fixer=fixer, card=card)
    # A probe that projects a fit: 50 s per scaled phase per seed at the full m. FIX-16: m is k1's stated copy case
    # (3000), so the probe runs 300 examples and its scaled phases are a tenth of that.
    m = card["resources"]["learning"]["copy"]["examples"]
    scale = m / learnability.probe_examples({"examples": m})
    fitting = dict(k1_timing(), phases_cpu=dict(k1_timing()["phases_cpu"], train=50.0 / scale,
                                                 train_wrong=50.0 / scale))
    executor = ProbeExecutor(tmp_path / "exec2", timing=fitting, full_status="cpu_cap")
    h.ctx.math = MathRunner(executor, mf.IMAGE, h.run_dir)
    base = cf.prover_output("k1")

    def prover(index, user):
        answer = base(index, user)
        for attempt in answer["attempts"]:
            if attempt["statement_id"] == "O7.b" and variant:
                attempt["source"] = attempt["source"] + "\n# the same learner, written again\n"
        return answer
    h.llm.script["prover"] = prover
    return h, executor


def test_the_same_source_for_an_equivalent_experiment_is_not_run_after_a_cpu_cap_stop(tmp_path):
    h, executor = two_copy_statements(tmp_path, variant=False)
    h.m()
    copy_kinds = copy_learnability(executor)
    assert copy_kinds == ["learnability_probe", "learnability"]   # one probe and one full job in all
    checks = [json.loads((h.run_dir / "artifacts" / "math_check" / e["data"]["artifact"]).read_text())
              for e in h.ledger.iter_events("math_check") if e["data"]["statement_id"] in ("O7.a", "O7.b")]
    kinds_seen = [(c["statement_id"], c["attempt"], (c.get("resource") or {}).get("kind")) for c in checks]
    assert kinds_seen[0] == ("O7.a", 1, "cpu_cap") and ("O7.b", 1, "unchanged_source") in kinds_seen
    assert all(c["status"] == "unknown" and not c["decided"] for c in checks)
    assert "unproven:P4.copy" in h.verdict("M")["kill_codes"]


def test_a_rewritten_source_with_an_unchanged_cost_gets_a_probe_but_no_full_job(tmp_path):
    h, executor = two_copy_statements(tmp_path, variant=True)
    h.m()
    assert copy_learnability(executor) == ["learnability_probe", "learnability", "learnability_probe"]
    o7b = [json.loads((h.run_dir / "artifacts" / "math_check" / e["data"]["artifact"]).read_text())
           for e in h.ledger.iter_events("math_check") if e["data"]["statement_id"] == "O7.b"][0]
    assert o7b["resource"]["kind"] == "duplicate" and o7b["job_ids"] == []
    assert "an equivalent frozen experiment (O7.a attempt 1" in o7b["evidence"]


# ----- 5. k2: a learnability statement always brings the learner contract; a null learner is told what to send ----

def test_a_learnability_statement_outside_the_profile_still_gets_the_contract_and_budget():
    record = live("k2_frozen_statements.json")
    card = calibration.load_cards(("k2",))["k2"]
    o10 = [s for s in m_math.all_statements(record) if s["id"] == "O10.a"]
    assert m_math.obligation_of(record, "O10.a")["claim_id"] == "sample_efficiency.prediction"
    user = m_math.prover_user(card, record, o10, {})
    assert "PROFILE OBLIGATIONS" not in user
    assert "LEARNABILITY BUDGET" in user and "1800 s of process CPU time" in user and "SANDBOX." in user
    assert "learn(layout, examples, seed)" in load_prompt("prover")


def test_the_recorded_null_learners_get_feedback_naming_what_to_send(tmp_path):
    declined = json.loads((LIVE / "k2_prover_declined.txt").read_text())
    notes = {a["statement_id"]: a["notes"] for a in declined["attempts"]}
    assert "doubtful" in notes["O10.a"]
    text = m_math.no_text_failure("learnability", 4)
    assert "one of its 4 attempts was used" in text and "learn(layout, examples, seed)" in text
    assert "even if you expect the check to fail" in text
    nulls = {"n": 0}
    base = cf.prover_output("k2")

    def prover(index, user):
        answer = base(index, user)
        if '"form": "learnability"' in user and nulls["n"] == 0:
            nulls["n"] += 1
            for attempt in answer["attempts"]:
                attempt["source"] = None
                attempt["notes"] = notes["O10.a"]
        return answer
    h = Harness(tmp_path, "k2", protocols=False)
    h.llm.script["prover"] = prover
    h.m()
    retries = [c["user"] for c in h.llm.calls if c["prompt"] == "prover" and '"form": "learnability"' in c["user"]]
    assert len(retries) >= 2
    assert "a learner source defining learn(layout, examples, seed)" in retries[1]
    assert h.slot("M", "P4.copy")["status"] == "observed"


# ----- 6. k3: the names the sandbox refused ----------------------------------------------------------------------

def test_the_k3_isinstance_learner_now_runs_to_a_decided_result():
    fixture = live("k3_learner_isinstance.json")
    assert fixture["evidence_then"] == "learner failed at seed 0: NameError: name 'isinstance' is not defined"
    # The recorded learner predates FIX-3's RECIPE requirement; its declared recipe is added, its code is unchanged.
    recipe = 'RECIPE = {"recorded": "2026-10-13 calibration; schedule as written in learn"}\n'
    result = learnability.run_learnability(dict(fixture["spec"], test_count=100), recipe + fixture["learner"])
    assert not result.get("prover_fault"), result["evidence"]
    assert result["status"] in ("discharged", "violated") and "seed 0 accuracy" in result["evidence"]


def test_isinstance_is_available_and_the_refused_names_stay_refused():
    namespace = learnability.load_learner(STUBS + "def kinds(x):\n    return isinstance(x, (list, dict)), "
                                                  "isinstance(np.zeros(2), np.ndarray)\n")
    assert namespace["kinds"]([1]) == (True, True)
    for name in ("getattr", "type", "object", "vars", "eval", "__import__", "globals"):
        assert cost.audit_source(f"def f(x):\n    return {name}(x)\n"), name
    available, absent = m_math.sandbox_note().split("No other builtin exists there")
    assert "isinstance" in available and "isinstance" not in absent
