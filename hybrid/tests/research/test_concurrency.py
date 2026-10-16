"""Concurrent checks in stage M and concurrent card-runs in the calibration driver (hybrid/research/parallel.py,
hybrid/slots.py): the ledger, decisions and outcomes do not depend on which job finished first, the job bound holds,
learnability checks of one frozen experiment still run in turn, and concurrent runs of a card share no state."""
import asyncio
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from hybrid.math.runner import MathRunner
from hybrid.research import calibration as cal
from hybrid.research import parallel
from hybrid.research.ledger import Ledger
from hybrid.research.orchestrator import ROOT, load_run_config
from hybrid.research.stages import m_math
from hybrid.research.stages.common import StageError
from hybrid.slots import JobSlots
from hybrid.tests.research import math_fakes as mf
from hybrid.tests.research.test_calibration import factory


class SlottedExecutor:
    """A fake executor behind the host-wide job slots. Jobs that start together finish in reverse order (the first of
    a burst sleeps longest), so recording in completion order would differ from statement order."""

    def __init__(self, inner, job_slots: JobSlots):
        self.inner = inner
        self.slots = job_slots
        self.lock = threading.Lock()
        self.now = 0
        self.peak = 0
        self.completed: list[str] = []

    def run(self, job):
        with self.slots.hold(job.caps.memory_mb):
            with self.lock:
                self.now += 1
                self.peak = max(self.peak, self.now)
                delay = 0.02 * (self.slots.limit - self.now + 1)
            time.sleep(delay)
            try:
                return self.inner.run(job)
            finally:
                with self.lock:
                    self.now -= 1
                    self.completed.append(job.job_id)


def slotted_factory(cards, job_slots: JobSlots, made: list):
    base = factory(cards)

    def build(ledger):
        deps = base(ledger)
        executor = SlottedExecutor(deps.executor, job_slots)
        made.append(executor)
        return cal.Deps(deps.llm, executor, deps.image, MathRunner(executor, mf.IMAGE, ledger.run_dir))
    return build


KEEP = {"math_check": ("stage", "statement_id", "purpose", "status", "decided", "resource"),
        "proof_attempt": ("statement_id", "attempt", "status", "decided", "refused"),
        "math_probe": ("stage", "statement_id", "attempt", "status", "decides"),
        "math_role_call": ("stage", "role", "purpose"),
        "math_decision": ("stage", "verdict", "kill_codes", "statuses", "premises"),
        "stage_verdict": ("card_id", "stage", "verdict"),
        "loop_round": ("stage", "loop", "round_id"),
        "loop_stop": ("stage", "loop", "kind")}


def signature(run_dir: Path) -> list:
    """The run's ledger without what is random by design (job ids, call ids, artifact names, times)."""
    return [(e["type"], json.dumps([e["data"].get(k) for k in KEEP[e["type"]]], sort_keys=True))
            for e in Ledger(run_dir).iter_events() if e["type"] in KEEP]


def run_k_card(tmp_path, name, limit, key="k1"):
    cards = cal.load_cards((key,))
    made = []
    cfg = cal.calibration_config(load_run_config(ROOT / "hybrid" / "config" / "run.toml"))
    target = {k: cfg["target"][k] for k in m_math.TARGET_RULE_KEYS}
    job_slots = JobSlots(limit, tmp_path / f"slots-{name}", poll_s=0.005)
    outcome = asyncio.run(cal.run_card(key=key, card=cards[key], run_index=1, run_dir=tmp_path / name / f"{key}-run1",
                                       cfg=cfg, reason="concurrency test",
                                       deps_factory=slotted_factory(cards, job_slots, made), target_rule=target))
    return outcome, made[0]


@pytest.mark.parametrize("key", ["k1", "k3"])
def test_concurrent_checks_record_the_same_ledger_and_decisions_as_serial_checks(tmp_path, key):
    serial, serial_exec = run_k_card(tmp_path, "serial", 1, key)
    concurrent, concurrent_exec = run_k_card(tmp_path, "concurrent", 4, key)
    assert serial_exec.peak == 1
    assert 1 < concurrent_exec.peak <= 4
    assert signature(Path(concurrent["run_dir"])) == signature(Path(serial["run_dir"]))
    for field in ("cells", "profile_passed", "m_kill_codes", "target", "stopped", "role_calls"):
        assert concurrent[field] == serial[field], field
    # The jobs did finish out of statement order; the recording put them back in order.
    recorded = [j for e in Ledger(Path(concurrent["run_dir"])).iter_events("math_check")
                for j in e["data"]["job_ids"]]
    finished = [j for j in concurrent_exec.completed if j in set(recorded)]
    assert sorted(recorded) == sorted(finished) and recorded != finished


# ----- scheduling inside a round ------------------------------------------------------------------------------------

def statement(sid, form="construction"):
    return {"id": sid, "form": form, "sha256": "h-" + sid, "spec": {}}


class SameExperiment:
    """A ResourceHistory stand-in whose key puts L1 and L2 in one frozen experiment."""

    def key(self, s):
        return "exp-A" if s["id"] in ("L1", "L2") else "exp-" + s["id"]


def scripted_execute(log, failing=()):
    async def execute_check(ctx, card_id, stage, s, *, history=None, **request):
        log.append(("start", s["id"], time.monotonic()))
        await asyncio.sleep(0.05 if s["id"] != "C1" else 0.1)
        log.append(("end", s["id"], time.monotonic()))
        if s["id"] in failing:
            raise StageError(f"infra error in {s['id']}")
        return {"statement_id": s["id"], **request}
    return execute_check


def window(log, sid):
    return next(t for kind, i, t in log if kind == "start" and i == sid), \
        next(t for kind, i, t in log if kind == "end" and i == sid)


def test_independent_checks_overlap_and_one_experiment_runs_in_turn(monkeypatch):
    log = []
    monkeypatch.setattr(m_math, "execute_check", scripted_execute(log))
    statements = [statement("C1"), statement("L1", "learnability"), statement("L2", "learnability"),
                  statement("C2")]
    requests = [{"statement": s, "purpose": "prove", "attempt": 1} for s in statements]
    results = asyncio.run(m_math.execute_checks(None, "card", "M", requests, SameExperiment()))
    assert [r["statement_id"] for r in results] == ["C1", "L1", "L2", "C2"]
    c1, l1, l2, c2 = (window(log, sid) for sid in ("C1", "L1", "L2", "C2"))
    assert l2[0] >= l1[1]                       # same frozen experiment: L2 starts after L1 ended
    assert c1[0] < l1[1] and c2[0] < c1[1]      # the other checks ran beside them


def test_a_failed_check_is_raised_after_the_earlier_checks_are_recorded(monkeypatch):
    log, recorded = [], []
    monkeypatch.setattr(m_math, "execute_check", scripted_execute(log, failing=("L1",)))
    monkeypatch.setattr(m_math, "record_check", lambda ctx, card, stage, s, check: recorded.append(s["id"]) or check)
    statements = [statement("C1"), statement("L1", "learnability"), statement("L2", "learnability"),
                  statement("C2")]
    requests = [{"statement": s, "purpose": "prove", "attempt": 1} for s in statements]
    results = asyncio.run(m_math.execute_checks(None, "card", "M", requests, SameExperiment()))
    assert isinstance(results[1], StageError) and results[2] is None
    # L2 shares L1's experiment and never ran after L1 failed, as in the serial loop.
    assert not any(i == "L2" for _, i, _ in log)
    with pytest.raises(StageError, match="infra error in L1"):
        m_math.record_in_order(None, "card", "M", statements, results)
    assert recorded == ["C1"]


def test_the_check_group_of_a_learnability_statement_is_its_frozen_experiment():
    record = {"definitions": [{"id": "D1", "sha256": "d1"}],
              "obligations": [{"id": "O9", "definitions": ["D1"], "statements": [statement("O9.a", "learnability")]},
                              {"id": "O10", "definitions": ["D1"],
                               "statements": [dict(statement("O10.a", "learnability"), sha256="h-O9.a")]},
                              {"id": "O11", "definitions": [], "statements": [statement("O11.a", "learnability")]}]}
    history = m_math.ResourceHistory(record)
    o9, o10, o11 = (s for o in record["obligations"] for s in o["statements"])
    assert m_math.check_group(o9, history) == m_math.check_group(o10, history)
    assert m_math.check_group(o9, history) != m_math.check_group(o11, history)
    assert m_math.check_group(statement("P1"), history) == "statement:P1"
    assert m_math.check_group(o9, None) == "statement:O9.a"


# ----- concurrent card-runs ------------------------------------------------------------------------------------------

def test_run_limited_bounds_the_runs_keeps_the_order_and_raises_the_first_failure():
    async def scenario():
        now, peak, done = [0], [0], []

        def thunk(i, fail=False):
            async def go():
                now[0] += 1
                peak[0] = max(peak[0], now[0])
                await asyncio.sleep(0.03 * (5 - i))
                now[0] -= 1
                if fail:
                    raise ValueError(f"run {i} failed")
                return i
            return go
        results = await parallel.run_limited([thunk(i) for i in range(5)], 2, lambda i, r: done.append(i))
        assert results == [0, 1, 2, 3, 4] and peak[0] == 2 and sorted(done) == [0, 1, 2, 3, 4]
        with pytest.raises(ValueError, match="run 1 failed"):
            await parallel.run_limited([thunk(0), thunk(1, fail=True), thunk(2)], 3)
    asyncio.run(scenario())
    for bad in (0, -1, True, "2"):
        with pytest.raises(ValueError):
            parallel.check_parallel(bad)


def run_calibration(tmp_path, name, cards, runs, parallel_runs, made):
    cfg = load_run_config(ROOT / "hybrid" / "config" / "run.toml")
    return asyncio.run(cal.calibrate(
        cards=cards, runs=runs, run_root=tmp_path / name / "runs", out_dir=tmp_path / name / "out", cfg=cfg,
        reason="parallel calibration test", deps_factory=tracked(factory(cards), made), parallel=parallel_runs,
        job_slots={"limit": 4}))


def tracked(build, made):
    def wrapped(ledger):
        deps = build(ledger)
        made.append((ledger.run_dir.resolve(), deps))
        return deps
    return wrapped


class TrackedBest(m_math.BestAttempts):
    instances: list = []

    def __init__(self):
        super().__init__()
        self.job_ids: list[str] = []
        TrackedBest.instances.append(self)

    def note(self, statement, attempt, check, source):
        self.job_ids += check.get("job_ids") or []
        super().note(statement, attempt, check, source)


def span(run_dir: Path) -> tuple[datetime, datetime]:
    events = list(Ledger(run_dir).iter_events())
    return datetime.fromisoformat(events[0]["ts"]), datetime.fromisoformat(events[-1]["ts"])


def test_parallel_card_runs_share_no_state_and_give_the_serial_result(tmp_path, monkeypatch):
    monkeypatch.setattr(m_math, "BestAttempts", TrackedBest)
    TrackedBest.instances = []
    cards = cal.load_cards(("k1", "k3"))
    serial_made, parallel_made = [], []
    serial = run_calibration(tmp_path, "serial", cards, 2, 1, serial_made)
    serial_best = list(TrackedBest.instances)
    TrackedBest.instances = []
    both = run_calibration(tmp_path, "parallel", cards, 2, 4, parallel_made)
    # The same outcomes, cells and acceptance verdict, in the same order.
    strip = lambda outcomes: [{k: v for k, v in o.items() if k != "run_dir"} for o in outcomes]
    assert strip(both["outcomes"]) == strip(serial["outcomes"])
    assert both["report"]["cards"] == serial["report"]["cards"]
    assert [(o["card"], o["run"]) for o in both["outcomes"]] == [("k1", 1), ("k3", 1), ("k1", 2), ("k3", 2)]
    lines = (tmp_path / "parallel" / "out" / "outcomes.jsonl").read_text().splitlines()
    assert [(json.loads(x)["card"], json.loads(x)["run"]) for x in lines] == [("k1", 1), ("k3", 1), ("k1", 2),
                                                                                ("k3", 2)]
    assert both["meta"]["parallel"] == 4 and both["meta"]["job_slots"] == {"limit": 4}
    # The four card-runs really ran at the same time.
    spans = [span(run_dir) for run_dir, _ in parallel_made]
    assert max(start for start, _ in spans) < min(end for _, end in spans)
    # Every card-run built its own client, executor and math runner.
    for attr in ("llm", "executor", "math"):
        assert len({id(getattr(deps, attr)) for _, deps in parallel_made}) == 4
    # Each run's prover memory saw only that run's jobs: no BestAttempts holds jobs of two run directories.
    run_dirs = [run_dir for run_dir, _ in parallel_made]
    noted = [best for best in TrackedBest.instances if best.job_ids]
    assert len(noted) >= 4 and len(TrackedBest.instances) == len(serial_best)
    for best in noted:
        owners = {run_dir for run_dir in run_dirs for job in best.job_ids if (run_dir / "jobs" / job).is_dir()}
        assert len(owners) == 1, owners
    # Each run's ledger is its own card and run.
    for run_dir, _ in parallel_made:
        override = next(Ledger(run_dir).iter_events("operator_override"))["data"]["calibration"]
        assert run_dir.name == f"{override['card']}-run{override['run']}"


def test_the_calibrate_and_planted_commands_take_parallel():
    from hybrid.research.__main__ import build_parser

    args = build_parser().parse_args(["calibrate", "--reason", "x"])
    assert args.parallel == 1
    assert build_parser().parse_args(["calibrate", "--reason", "x", "--parallel", "3"]).parallel == 3
    assert build_parser().parse_args(["planted", "--parallel", "2"]).parallel == 2
    assert load_run_config(ROOT / "hybrid" / "config" / "run.toml")["math"]["max_parallel_jobs"] == "auto"


def test_max_parallel_jobs_is_validated(tmp_path):
    text = (ROOT / "hybrid" / "config" / "run.toml").read_text()
    for bad in ("0", "\"many\"", "true"):
        path = tmp_path / "run.toml"
        path.write_text(text.replace('max_parallel_jobs = "auto"', f"max_parallel_jobs = {bad}"))
        with pytest.raises(ValueError, match="max_parallel_jobs"):
            load_run_config(path)
    path.write_text(text.replace('max_parallel_jobs = "auto"\n', ""))
    assert load_run_config(path)["math"]["max_parallel_jobs"] == "auto"
