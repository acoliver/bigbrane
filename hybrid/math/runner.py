"""Run Lean, Z3 and SymPy checks in the tools image through the executor's job API.

Every check is one executor job per run (fresh container, ``--network=none``, read-only rootfs, uid 10001, the
caps below), with its inputs staged into ``<work_dir>/jobs/<job id>/in`` and its outputs under ``.../out``.
A result is decided only when two runs agree, as in ``hybrid/research/stages/obligation_checks.py``.

* ``check_lean(LeanCheck)``: rejected on the host, without running, on a statement hash mismatch or a forbidden
  construct (see lean_check.py); otherwise the assembled file is elaborated with ``lean-check`` (the image's Lean
  with the Mathlib LEAN_PATH) twice, each run with its own nonce. Pass needs both runs to pass.
* ``check_program(tool, program)``: a Python program (z3 or sympy) following the obligation-check contract: the
  last stdout line is {"status": "discharged" | "violated" | "unknown", "evidence": str, "counterexample": ...}.
  A deciding first run is replayed and counts only when the replay prints the same status and counterexample.
* ``check_library(request)``: a Phase 2 math-library check (conservation, cost, construction, magic_x or the
  selfcheck battery), or a Phase 3 check of a frozen statement (z3, sympy, simulation, audited_construction,
  predicate_agreement; see spec_checks.py). The check files are verified against MANIFEST.json on the host first (a mismatch raises
  before anything is staged), staged as the package ``bbmath`` with the driver jobmain.py as ``main.py``, re-hashed
  by the driver inside the container, and run under LIBRARY_CAPS with the same replay rule as programs. The result
  records the hashes of every staged check file.
* ``check_learnability(request)`` (also reached through ``check_library`` for kind learnability): first a timing
  probe, one job of request kind ``learnability_probe`` (learnability.timing_probe: one seed, min(m, 300) examples,
  the full schedule and both controls) under a CPU cap of PROBE_FIT of the full CPU cap divided by the seeds. The
  probe decides nothing; it is recorded in the result under ``probe`` with its own job id. When its projection of the
  full check is above PROBE_FIT of the CPU cap, when it reaches its own CPU cap or memory, or when the caller's
  ``admit`` refuses it, the full job is not run and the result is ``unknown`` with a ``resource`` record (a resource
  failure the prover can fix, never a pass or a counterexample). Otherwise the full check runs as any library check;
  a full run stopped at the CPU cap gets a ``resource`` record too, with the probe's projection. In the 2026-10-13
  calibration (card k1) a full learnability job ran out of 1800 s with no output; its cost had been about 1750 s on
  the host (results/phase4-20261013T073748Z/p4-timing/REPORT.md).

CPU budget and hang guard (decision CT, 2026-10-17). Library jobs (every request kind run by jobmain.py: the
learnability probe and check, and the Phase 2 and 3 kinds) are capped in process CPU seconds, ``Caps.cpu_s``: the
request carries it as ``cpu_cap_s`` and the driver enforces it with RLIMIT_CPU, and the probe's projection and the
admission are in CPU seconds. So a verdict does not depend on how busy the host is: on 2026-10-17, at a host load of
25 to 65, the wall-clock probes of K2's recall and composition learners projected 3341 s and 3623 s against 1440 s
admitted, where the same kind of job had projected 1007 s on 2026-10-15 and passed. The executor's wall cap stays
as a hang guard at HANG_GUARD_FACTOR times the most CPU time the job can use (hang_guard_s: the cap rounded up to whole
seconds, as RLIMIT_CPU counts, plus CPU_GRACE_S before the hard limit kills it). A library job is single-threaded
under a one-CPU quota, so its wall time is at least its CPU time; the guard fires only when a job got less than a third
of a core on average over its whole run, and for a job admitted at PROBE_FIT of the 1800 s cap (1440 CPU s against a
5490 s guard) only below about 1 / 3.8 of a core: the 2026-10-17 projections were 3.3 and 3.6 times the 2026-10-15
one, inside that margin. Basing the guard on the enforced limit matters for small caps: in the FIX-CT container smoke
a probe capped at 1.1 CPU s (soft limit 2 s) printed its cap report after 3.0 s of wall time, at the moment a guard of
three times 1.1 s expired. A hang-guard stop is not
evidence about the check: the runner raises JobHangGuard, an environment failure the stages turn into a stage error
(calibration classifies the card-run invalid_environment), never a verdict and never a resource failure the prover
is asked to fix. Lean and Z3/SymPy program jobs are not run by the driver and keep their wall caps.
"""
from __future__ import annotations

import math
import secrets
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

import json

from hybrid.executor.job import Caps, Job
from hybrid.math import lean_check, learnability, manifest
from hybrid.research.stages.common import ExecResult
from hybrid.research.stages.obligation_checks import DECIDING, outcome, parse_report, run_entry

LEAN_CAPS = Caps(wall_s=900, memory_mb=8192, pids=512, cpus=1.0, tmp_mb=64)
PROGRAM_CAPS = Caps(wall_s=300, memory_mb=2048, pids=64, cpus=1.0, tmp_mb=64)
# Library jobs: the wall cap is this multiple of the most CPU time the job can use, a hang guard only (module
# docstring), and the CPU seconds between the driver's RLIMIT_CPU soft limit (SIGXCPU, a clean cap report) and its hard
# limit (SIGKILL), sent to the driver in the request.
HANG_GUARD_FACTOR = 3
CPU_GRACE_S = 30


def hang_guard_s(cpu_s: float) -> float:
    """The wall-clock hang guard of a job capped at ``cpu_s`` CPU seconds: HANG_GUARD_FACTOR times the CPU time the
    job can use before the hard limit kills it (the soft limit is ``cpu_s`` rounded up to whole seconds)."""
    return float(HANG_GUARD_FACTOR * (math.ceil(cpu_s) + CPU_GRACE_S))


def cpu_capped(cpu_s: float, memory_mb: int, pids: int, cpus: float = 1.0, tmp_mb: int = 64) -> Caps:
    """Caps of a library job: ``cpu_s`` process CPU seconds, enforced inside the job, and the wall-clock hang guard
    hang_guard_s(cpu_s)."""
    return Caps(wall_s=hang_guard_s(cpu_s), cpu_s=cpu_s, memory_mb=memory_mb, pids=pids, cpus=cpus, tmp_mb=tmp_mb)


# The selfcheck on the plan's full grid (n up to 4096) is the longest library job. The CPU cap is the 1800 s the wall
# cap was before decision CT.
LIBRARY_CPU_S = 1800
LIBRARY_CAPS = cpu_capped(LIBRARY_CPU_S, memory_mb=4096, pids=64)
LIBRARY_KINDS = ("conservation", "cost", "construction", "magic_x", "selfcheck",
                 # Phase 3: checks of frozen statements (spec_checks.py)
                 "z3", "sympy", "simulation", "audited_construction", "predicate_agreement",
                 # Phase 4: toy learnability (profile obligation P4, learnability.py) and its timing probe
                 "learnability", "learnability_probe",
                 # FIX-4: the verification code of a challenger finding (findings.py)
                 "finding",
                 # FIX-6: a cost reference against the card's frozen equations (spec_checks.reference_conformance)
                 "conformance")
# Staged path in the job -> file under hybrid/math. Every file is covered by MANIFEST.json.
LIBRARY_FILES = {
    "main.py": "jobmain.py",
    "bbmath/__init__.py": "__init__.py",
    **{f"bbmath/{name}": name for name in ("architecture.py", "conservation.py", "constructions.py", "cost.py",
                                            "findings.py", "learnability.py", "magic_x.py", "references.py",
                                            "selfcheck.py", "spec_checks.py", "tiers.py")},
}
PROGRAM_TOOLS = ("z3", "sympy")
# Host-side files that decide a Lean check (the trailer, the scan, the verdict, this runner).
LEAN_CHECK_FILES = ("lean_check.py", "runner.py")
CHECK_FILE = "Check.lean"


@dataclass
class LeanRun:
    job_id: str
    receipt_status: str
    exit_code: int
    run_s: float
    nonce: str
    status: str            # pass | reject | infra_error
    reasons: list[str]
    facts: dict
    stdout_tail: str


@dataclass
class LeanResult:
    status: str            # pass | reject | unknown
    decided: bool
    reasons: list[str]
    statement_sha256: str
    axioms: list[str] | None
    runs: list[LeanRun] = field(default_factory=list)
    # sha256 of the host-side check files that decided this result, from the verified MANIFEST.json
    check_hashes: dict = field(default_factory=dict)


def new_job_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class JobHangGuard(Exception):
    """A CPU-capped library job reached its wall-clock hang guard: the host or the VM starved or stalled it. An
    environment failure, never a verdict or a resource failure of the check (decision CT)."""

    def __init__(self, job_id: str, kind: str, caps: Caps, run_s: float | None, detail: str):
        self.job_id, self.kind, self.caps, self.run_s = job_id, kind, caps, run_s
        super().__init__(f"environment failure: {kind} job {job_id} reached its wall-clock hang guard of "
                         f"{caps.wall_s:.0f} s (CPU cap {caps.cpu_s:.0f} s) after {run_s or 0:.1f} s: {detail}; the "
                         "host or the tools VM did not give the job CPU time, so the run is no evidence about the check")


class MathRunner:
    def __init__(self, executor, image: str, work_dir: Path, *, lean_caps: Caps = LEAN_CAPS,
                 program_caps: Caps = PROGRAM_CAPS, library_caps: Caps = LIBRARY_CAPS,
                 nonce=lambda: secrets.token_hex(16)):
        self.executor = executor
        self.image = image
        self.work_dir = Path(work_dir)
        self.lean_caps = lean_caps
        self.program_caps = program_caps
        self.library_caps = library_caps
        self.nonce = nonce

    def _job(self, job_id: str, files: dict[str, str], command: tuple, caps: Caps):
        job_dir = self.work_dir / "jobs" / job_id
        in_dir, out_dir = job_dir / "in", job_dir / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        for name, text in files.items():
            (in_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (in_dir / name).write_text(text, encoding="utf-8")
        receipt = self.executor.run(Job(job_id, self.image, in_dir, out_dir, command, caps))
        stdout = Path(receipt.stdout_path).read_text(encoding="utf-8", errors="replace")
        stderr = Path(receipt.stderr_path).read_text(encoding="utf-8", errors="replace")
        return receipt, stdout, stderr, out_dir

    # ----- Lean -------------------------------------------------------------------------------

    def lean_once(self, check: lean_check.LeanCheck) -> LeanRun:
        nonce = self.nonce()
        job_id = new_job_id("lean")
        source = lean_check.assemble(check, nonce)
        receipt, stdout, stderr, _ = self._job(job_id, {CHECK_FILE: source}, ("lean-check", f"/in/{CHECK_FILE}"),
                                               self.lean_caps)
        facts = lean_check.parse_output(stdout, nonce)
        if receipt.status not in ("ok", "candidate_error"):
            status, reasons = "infra_error", [f"executor status {receipt.status}: {receipt.detail}"[:400]]
        else:
            status, reasons = lean_check.verdict(receipt.exit_code, facts)
        return LeanRun(job_id, receipt.status, receipt.exit_code, receipt.run_s, nonce, status, reasons, facts,
                       (stdout + stderr)[-3000:])

    def check_lean(self, check: lean_check.LeanCheck) -> LeanResult:
        # The checker files must match MANIFEST.json before anything is judged (plan section 5.5).
        recorded = manifest.verify()
        hashes = {name: recorded[name] for name in LEAN_CHECK_FILES}
        digest = lean_check.statement_hash(check.statement)
        problems = check.problems()
        if problems:
            return LeanResult("reject", True, problems, digest, None, check_hashes=hashes)
        first = self.lean_once(check)
        if first.status == "infra_error":
            return LeanResult("unknown", False, first.reasons, digest, first.facts["axioms"], [first], hashes)
        second = self.lean_once(check)
        runs = [first, second]
        if second.status == "infra_error":
            return LeanResult("unknown", False, second.reasons, digest, first.facts["axioms"], runs, hashes)
        if (first.status, first.facts["axioms"]) != (second.status, second.facts["axioms"]):
            return LeanResult("unknown", False, [f"replay disagreed: {first.status} {first.facts['axioms']} then "
                                                 f"{second.status} {second.facts['axioms']}"],
                              digest, first.facts["axioms"], runs, hashes)
        return LeanResult(first.status, True, first.reasons, digest, first.facts["axioms"], runs, hashes)

    # ----- Z3 and SymPy programs --------------------------------------------------------------

    def _python_once(self, prefix: str, files: dict[str, str], caps: Caps) -> ExecResult:
        job_id = new_job_id(prefix)
        receipt, stdout, stderr, out_dir = self._job(job_id, files, ("python3", "/in/main.py"), caps)
        return ExecResult(job_id, receipt.status, receipt.detail, stdout, stderr, receipt.run_s, out_dir,
                          receipt.exit_code)

    def _replayed(self, run_once) -> dict:
        first = run_once()
        report, _ = parse_report(first.printed_json() if first.ok else None)
        second = None
        if report is not None and report["status"] in DECIDING:
            second = run_once()
        return outcome(first, second)

    def program_once(self, tool: str, program: str) -> ExecResult:
        if tool not in PROGRAM_TOOLS:
            raise ValueError(f"tool must be one of {PROGRAM_TOOLS}, got {tool!r}")
        return self._python_once(tool, {"main.py": program}, self.program_caps)

    def check_program(self, tool: str, program: str) -> dict:
        """The obligation-check record for one program: status discharged | violated | unknown, decided only
        when the replay agrees. Same contract and replay rule as obligation_checks.outcome."""
        return dict(self._replayed(lambda: self.program_once(tool, program)), tool=tool)

    # ----- math library (Phase 2) ------------------------------------------------------------------------------

    def library_files(self, request: dict, caps: Caps) -> tuple[dict[str, str], dict[str, str]]:
        """The files staged for a library job and their verified hashes (staged path -> sha256); the request carries
        the job's CPU cap (``cpu_cap_s``, from ``caps.cpu_s``) and the hard limit's grace (``cpu_grace_s``) for the
        driver to enforce. Raises
        manifest.ManifestMismatch when any file under hybrid/math differs from MANIFEST.json."""
        if request.get("kind") not in LIBRARY_KINDS:
            raise ValueError(f"request kind must be one of {LIBRARY_KINDS}, got {request.get('kind')!r}")
        if caps.cpu_s is None:
            raise ValueError("library jobs are CPU-capped (decision CT): caps.cpu_s must be set, see cpu_capped")
        recorded = manifest.verify()
        hashes = {staged: recorded[source] for staged, source in LIBRARY_FILES.items()}
        files = {staged: (manifest.MATH_DIR / source).read_text(encoding="utf-8")
                 for staged, source in LIBRARY_FILES.items()}
        files["manifest.json"] = json.dumps({"files": hashes}, indent=2, sort_keys=True)
        files["request.json"] = json.dumps(dict(request, cpu_cap_s=caps.cpu_s, cpu_grace_s=CPU_GRACE_S),
                                           sort_keys=True)
        return files, hashes

    def _library_once(self, prefix: str, kind: str, files: dict[str, str], caps: Caps) -> ExecResult:
        """One CPU-capped library job; a stop at the wall cap is the hang guard and raises JobHangGuard."""
        run = self._python_once(prefix, files, caps)
        if run.status == "timeout":
            raise JobHangGuard(run.job_id, kind, caps, run.run_s, run.detail)
        return run

    def check_library(self, request: dict, caps: Caps | None = None, *, admit=None) -> dict:
        """A library check; a learnability request goes through ``check_learnability`` and its timing probe."""
        caps = caps or self.library_caps
        if request.get("kind") == "learnability":
            return self.check_learnability(request, caps, admit=admit)
        files, hashes = self.library_files(request, caps)
        record = self._replayed(lambda: self._library_once("mathlib", request["kind"], files, caps))
        return dict(record, kind=request["kind"], check_hashes=hashes)

    # ----- learnability: timing probe, then the full check -------------------------------------------------------

    def probe_caps(self, spec: dict, caps: Caps) -> Caps:
        """The probe runs one seed of at most m examples, so a probe whose CPU time exceeds the admitted full budget
        divided by the seeds already projects a full check over that budget. Its wall cap is the hang guard of its CPU
        cap."""
        cpu = max(1.0, round(learnability.PROBE_FIT * caps.cpu_s / len(spec["seeds"]), 1))
        return replace(caps, cpu_s=cpu, wall_s=hang_guard_s(cpu))

    def probe_learnability(self, request: dict, caps: Caps) -> dict:
        """Run the timing probe for a learnability request once (never replayed: it decides nothing) and return its
        record: job, receipt status, measured CPU phases and the CPU projection of the full check against ``caps``.
        Wall times are recorded for information (``run_s``, the receipt's; ``timing.phases_wall``)."""
        spec = request["spec"]
        probe_request = {"kind": "learnability_probe", "spec": spec, "source": request["source"]}
        pcaps = self.probe_caps(spec, caps)
        files, _ = self.library_files(probe_request, pcaps)
        run = self._library_once("mathprobe", "learnability_probe", files, pcaps)
        printed = run.printed_json() if run.status in ("ok", "candidate_error") else None
        clock = (printed or {}).get("job_clock") or (printed or {}).get("cpu_cap") or {}
        record = {"job_id": run.job_id, "status": run.status, "detail": run.detail,
                  "run_s": None if run.run_s is None else round(run.run_s, 3), "cpu_s": clock.get("cpu_s"),
                  "probe_cap_cpu_s": pcaps.cpu_s, "probe_wall_cap_s": pcaps.wall_s, "cap_cpu_s": caps.cpu_s,
                  "wall_cap_s": caps.wall_s, "fit_cpu_s": learnability.PROBE_FIT * caps.cpu_s,
                  "examples": spec["examples"], "seeds": len(spec["seeds"]), "timing": None, "projected_cpu_s": None,
                  "cpu_cap_reached": cpu_cap_stop(run, printed), "evidence": (printed or {}).get("evidence"),
                  "prover_fault": bool((printed or {}).get("prover_fault")), "run": run_entry(run)}
        if printed and printed.get("timing"):
            timing = printed["timing"]
            overhead = max(0.0, clock["cpu_s"] - timing["total_cpu_s"]) if clock.get("cpu_s") is not None else 0.0
            record.update(timing=timing, projected_cpu_s=round(
                learnability.projected_cpu_seconds(spec, timing, overhead), 1))
        return record

    def check_learnability(self, request: dict, caps: Caps | None = None, *, admit=None) -> dict:
        """The timing probe, then the full check when the probe projects that it fits. ``admit(probe)`` may refuse
        the full job with a reason (stage M refuses an experiment that already ran out of its cap unless the new
        source's probe shows a lower cost)."""
        caps = caps or self.library_caps
        files, hashes = self.library_files(request, caps)
        probe = self.probe_learnability(request, caps)
        base = {"kind": "learnability", "check_hashes": hashes, "probe": probe, "replayed": False, "decided": False,
                "counterexample": None}
        if probe["status"] == "infra_error":
            return dict(base, status="unknown", evidence="", problems=[f"timing probe {probe['status']}: "
                                                                      f"{probe['detail']}"[:400]],
                        runs=[probe["run"]], resource=None)
        resource = probe_refusal(probe)
        if resource is None and probe["timing"] is None:
            # The probe stopped before timing anything: the statement cannot be read, or the learner was refused or
            # crashed. The full check would stop the same way, so its report is the result.
            evidence = probe["evidence"] or f"timing probe printed no report (executor status {probe['status']})"
            return dict(base, status="unknown", evidence=evidence, problems=[evidence[:400]], runs=[],
                        resource=None, prover_fault=probe["prover_fault"])
        if resource is None and admit is not None:
            reason = admit(probe)
            if reason:
                resource = resource_record("duplicate", probe, reason=reason)
        if resource is not None:
            text = resource_text(resource)
            return dict(base, status="unknown", evidence=text, problems=[text[:400]], runs=[], resource=resource)
        runs = []

        def full_once():
            run = self._library_once("mathlib", "learnability", files, caps)
            runs.append(run)
            return run
        record = self._replayed(full_once)
        record = dict(record, **{k: v for k, v in base.items() if k not in ("replayed", "decided",
                                                                             "counterexample")}, resource=None)
        first = runs[0]
        printed = first.printed_json() if first.status in ("ok", "candidate_error") else None
        if cpu_cap_stop(first, printed):
            resource = resource_record("cpu_cap", probe, run_s=first.run_s, detail=first.detail, job_id=first.job_id,
                                       cpu_s=((printed or {}).get("cpu_cap") or {}).get("cpu_s"))
            record.update(resource=resource, evidence=resource_text(resource))
        elif first.status == "oom":
            resource = resource_record("oom", probe, run_s=first.run_s, detail=first.detail, job_id=first.job_id)
            record.update(resource=resource, evidence=resource_text(resource))
        return record


# ----- resource failures of learnability checks ----------------------------------------------------------------------

# Exit code of a job the RLIMIT_CPU hard limit killed (SIGKILL, 128 + 9 through the supervisor). Nothing else in a
# library job sends SIGKILL: the executor's kill is the wall cap (status timeout) and a kernel OOM kill is status oom.
CPU_HARD_LIMIT_EXIT = 137


def cpu_cap_stop(run: ExecResult, printed: dict | None) -> bool:
    """Whether a library job stopped at its CPU cap: the driver's cap report, or the hard limit's SIGKILL with no
    report."""
    if printed is not None and printed.get("cpu_cap"):
        return True
    return run.status == "candidate_error" and printed is None and run.exit_code == CPU_HARD_LIMIT_EXIT


def probe_refusal(probe: dict) -> dict | None:
    """The resource record that stops a learnability check after its probe, or None when the full job may run."""
    if probe["cpu_cap_reached"]:
        return resource_record("probe_cpu_cap", probe)
    if probe["status"] in ("oom", "pids"):
        return resource_record("probe_" + probe["status"], probe)
    if probe["projected_cpu_s"] is not None and probe["projected_cpu_s"] > probe["fit_cpu_s"]:
        return resource_record("projected_overrun", probe)
    return None


def resource_record(kind: str, probe: dict, *, reason: str | None = None, run_s: float | None = None,
                    detail: str | None = None, job_id: str | None = None, cpu_s: float | None = None) -> dict:
    """A learnability check stopped for its cost: what stopped it and the numbers the prover needs to fix it. Costs
    and caps are process CPU seconds; ``run_s`` and ``probe_run_s`` are the receipts' wall seconds, for information."""
    at_least = None
    if kind == "probe_cpu_cap":
        at_least = round(probe["probe_cap_cpu_s"] * probe["seeds"], 1)
    return {"kind": kind, "cap_cpu_s": probe["cap_cpu_s"], "wall_cap_s": probe["wall_cap_s"],
            "fit_cpu_s": probe["fit_cpu_s"], "projected_cpu_s": probe["projected_cpu_s"],
            "projected_at_least_cpu_s": at_least, "probe_job": probe["job_id"], "probe_run_s": probe["run_s"],
            "probe_cpu_s": probe["cpu_s"], "probe_cap_cpu_s": probe["probe_cap_cpu_s"],
            "probe_examples": (probe["timing"] or {}).get("examples"), "examples": probe["examples"],
            "seeds": probe["seeds"], "timing": probe["timing"], "reason": reason, "run_s": run_s, "cpu_s": cpu_s,
            "detail": detail, "job_id": job_id}


def resource_text(resource: dict) -> str:
    """The resource failure in words, with the measured and projected cost; this is what the prover reads."""
    cap, fit = resource["cap_cpu_s"], resource["fit_cpu_s"]
    head = "resource failure (status unknown: not a pass and not a counterexample): "
    kind = resource["kind"]
    if kind == "cpu_cap":
        used = f"{resource['cpu_s']:.1f} CPU s, " if resource.get("cpu_s") is not None else ""
        what = (f"the full check reached its CPU cap of {cap:.0f} s of process CPU time (job {resource['job_id']}, "
                f"stopped after {used}{(resource['run_s'] or 0):.1f} s wall)")
    elif kind == "oom":
        what = f"the full check ran out of memory (job {resource['job_id']}: {resource['detail']})"
    elif kind == "probe_cpu_cap":
        what = (f"the full check was not run: the timing probe (one seed, at most 300 examples) reached its own CPU "
                f"cap of {resource['probe_cap_cpu_s']:.0f} s, so the full check of {resource['seeds']} seeds on "
                f"m={resource['examples']} examples is projected at {resource['projected_at_least_cpu_s']:.0f} CPU s "
                f"or more against the {cap:.0f} s CPU cap")
    elif kind in ("probe_oom", "probe_pids"):
        what = (f"the full check was not run: the timing probe on at most 300 examples already ran out of "
                f"{'memory' if kind == 'probe_oom' else 'processes'}")
    elif kind == "duplicate":
        what = "the full check was not run: " + (resource["reason"] or "")
    else:
        what = (f"the full check was not run: the timing probe projects {resource['projected_cpu_s']:.0f} CPU s for "
                f"{resource['seeds']} seeds on m={resource['examples']} examples, above the {fit:.0f} CPU s admitted "
                f"({learnability.PROBE_FIT:.0%} of the {cap:.0f} s CPU cap)")
    parts = [head + what + "."]
    if resource["timing"]:
        timing = resource["timing"]
        parts.append(f"Timing probe (job {resource['probe_job']}, {timing['total_cpu_s']:.1f} CPU s, "
                     f"{resource['probe_run_s'] or 0:.1f} s wall): seed "
                     f"{timing['seed']} on {timing['examples']} examples, {learnability.timing_line(timing)}"
                     + (f"; projected full check {resource['projected_cpu_s']:.0f} CPU s"
                        if resource["projected_cpu_s"] is not None else "") + ".")
    parts.append("Make each training on m examples cheaper without changing the frozen statement or its "
                 "definitions: batched matrix products over the whole batch, np.einsum with optimize=True or "
                 "reshape and @ for gradients summed over batch and time, no per-example or per-token Python loops "
                 "where a batch operation exists. The seeds, m, the threshold and both data controls stay as they are.")
    return " ".join(parts)
