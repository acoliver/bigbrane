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
  the full schedule and both controls) under a wall cap of PROBE_FIT of the full cap divided by the seeds. The probe
  decides nothing; it is recorded in the result under ``probe`` with its own job id. When its projection of the full
  check is above PROBE_FIT of the cap, when it hits its own cap or memory, or when the caller's ``admit`` refuses it,
  the full job is not run and the result is ``unknown`` with a ``resource`` record (a resource failure the prover can
  fix, never a pass or a counterexample). Otherwise the full check runs as any library check; a full run stopped by
  the wall cap gets a ``resource`` record too, with the probe's projection. In the 2026-10-13 calibration (card k1)
  a full learnability job ran out of 1800 s with no output; its cost had been about 1750 s on the host
  (results/phase4-20261013T073748Z/p4-timing/REPORT.md).
"""
from __future__ import annotations

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
# The selfcheck on the plan's full grid (n up to 4096) is the longest library job.
LIBRARY_CAPS = Caps(wall_s=1800, memory_mb=4096, pids=64, cpus=1.0, tmp_mb=64)
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
        return ExecResult(job_id, receipt.status, receipt.detail, stdout, stderr, receipt.run_s, out_dir)

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

    def library_files(self, request: dict) -> tuple[dict[str, str], dict[str, str]]:
        """The files staged for a library job and their verified hashes (staged path -> sha256). Raises
        manifest.ManifestMismatch when any file under hybrid/math differs from MANIFEST.json."""
        if request.get("kind") not in LIBRARY_KINDS:
            raise ValueError(f"request kind must be one of {LIBRARY_KINDS}, got {request.get('kind')!r}")
        recorded = manifest.verify()
        hashes = {staged: recorded[source] for staged, source in LIBRARY_FILES.items()}
        files = {staged: (manifest.MATH_DIR / source).read_text(encoding="utf-8")
                 for staged, source in LIBRARY_FILES.items()}
        files["manifest.json"] = json.dumps({"files": hashes}, indent=2, sort_keys=True)
        files["request.json"] = json.dumps(request, sort_keys=True)
        return files, hashes

    def check_library(self, request: dict, caps: Caps | None = None, *, admit=None) -> dict:
        """A library check; a learnability request goes through ``check_learnability`` and its timing probe."""
        caps = caps or self.library_caps
        if request.get("kind") == "learnability":
            return self.check_learnability(request, caps, admit=admit)
        files, hashes = self.library_files(request)
        record = self._replayed(lambda: self._python_once("mathlib", files, caps))
        return dict(record, kind=request["kind"], check_hashes=hashes)

    # ----- learnability: timing probe, then the full check -------------------------------------------------------

    def probe_caps(self, spec: dict, caps: Caps) -> Caps:
        """The probe runs one seed of at most m examples, so a probe longer than the admitted full budget divided by
        the seeds already projects a full check over that budget."""
        return replace(caps, wall_s=max(1.0, round(learnability.PROBE_FIT * caps.wall_s / len(spec["seeds"]), 1)))

    def probe_learnability(self, request: dict, caps: Caps) -> dict:
        """Run the timing probe for a learnability request once (never replayed: it decides nothing) and return its
        record: job, receipt status, measured phases and the projection of the full check against ``caps``."""
        spec = request["spec"]
        probe_request = {"kind": "learnability_probe", "spec": spec, "source": request["source"]}
        files, _ = self.library_files(probe_request)
        pcaps = self.probe_caps(spec, caps)
        run = self._python_once("mathprobe", files, pcaps)
        printed = run.printed_json() if run.status in ("ok", "candidate_error") else None
        record = {"job_id": run.job_id, "status": run.status, "detail": run.detail,
                  "run_s": None if run.run_s is None else round(run.run_s, 3),
                  "probe_cap_s": pcaps.wall_s, "cap_s": caps.wall_s, "fit_s": learnability.PROBE_FIT * caps.wall_s,
                  "examples": spec["examples"], "seeds": len(spec["seeds"]), "timing": None, "projected_s": None,
                  "evidence": (printed or {}).get("evidence"), "prover_fault": bool((printed or {}).get(
                      "prover_fault")), "run": run_entry(run)}
        if printed and printed.get("timing"):
            timing = printed["timing"]
            overhead = max(0.0, run.run_s - timing["total_s"])
            record.update(timing=timing, projected_s=round(learnability.projected_seconds(spec, timing, overhead), 1))
        return record

    def check_learnability(self, request: dict, caps: Caps | None = None, *, admit=None) -> dict:
        """The timing probe, then the full check when the probe projects that it fits. ``admit(probe)`` may refuse
        the full job with a reason (stage M refuses an experiment that already ran out of its cap unless the new
        source's probe shows a lower cost)."""
        caps = caps or self.library_caps
        files, hashes = self.library_files(request)
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
        record = self._replayed(lambda: self._python_once("mathlib", files, caps))
        record = dict(record, **{k: v for k, v in base.items() if k not in ("replayed", "decided",
                                                                             "counterexample")}, resource=None)
        first = record["runs"][0] if record["runs"] else {}
        if first.get("status") in ("timeout", "oom"):
            resource = resource_record(first["status"], probe, run_s=first.get("run_s"), detail=first.get("detail"),
                                       job_id=first.get("job_id"))
            record.update(resource=resource, evidence=resource_text(resource))
        return record


# ----- resource failures of learnability checks ----------------------------------------------------------------------

def probe_refusal(probe: dict) -> dict | None:
    """The resource record that stops a learnability check after its probe, or None when the full job may run."""
    if probe["status"] == "timeout":
        return resource_record("probe_timeout", probe)
    if probe["status"] in ("oom", "pids"):
        return resource_record("probe_" + probe["status"], probe)
    if probe["projected_s"] is not None and probe["projected_s"] > probe["fit_s"]:
        return resource_record("projected_overrun", probe)
    return None


def resource_record(kind: str, probe: dict, *, reason: str | None = None, run_s: float | None = None,
                    detail: str | None = None, job_id: str | None = None) -> dict:
    """A learnability check stopped for its cost: what stopped it and the numbers the prover needs to fix it."""
    at_least = None
    if kind == "probe_timeout":
        at_least = round(probe["probe_cap_s"] * probe["seeds"], 1)
    return {"kind": kind, "cap_s": probe["cap_s"], "fit_s": probe["fit_s"], "projected_s": probe["projected_s"],
            "projected_at_least_s": at_least, "probe_job": probe["job_id"], "probe_run_s": probe["run_s"],
            "probe_cap_s": probe["probe_cap_s"], "probe_examples": (probe["timing"] or {}).get("examples"),
            "examples": probe["examples"], "seeds": probe["seeds"], "timing": probe["timing"], "reason": reason,
            "run_s": run_s, "detail": detail, "job_id": job_id}


def resource_text(resource: dict) -> str:
    """The resource failure in words, with the measured and projected cost; this is what the prover reads."""
    cap, fit = resource["cap_s"], resource["fit_s"]
    head = "resource failure (status unknown: not a pass and not a counterexample): "
    kind = resource["kind"]
    if kind == "timeout":
        what = (f"the full check ran out of its {cap:.0f} s wall cap (job {resource['job_id']}, stopped after "
                f"{(resource['run_s'] or 0):.1f} s: {resource['detail']})")
    elif kind == "oom":
        what = f"the full check ran out of memory (job {resource['job_id']}: {resource['detail']})"
    elif kind == "probe_timeout":
        what = (f"the full check was not run: the timing probe (one seed, at most 300 examples) did not finish within "
                f"its own cap of {resource['probe_cap_s']:.0f} s, so the full check of {resource['seeds']} seeds "
                f"on m={resource['examples']} examples is projected at {resource['projected_at_least_s']:.0f} s "
                f"or more against the {cap:.0f} s cap")
    elif kind in ("probe_oom", "probe_pids"):
        what = (f"the full check was not run: the timing probe on at most 300 examples already ran out of "
                f"{'memory' if kind == 'probe_oom' else 'processes'}")
    elif kind == "duplicate":
        what = "the full check was not run: " + (resource["reason"] or "")
    else:
        what = (f"the full check was not run: the timing probe projects {resource['projected_s']:.0f} s for "
                f"{resource['seeds']} seeds on m={resource['examples']} examples, above the {fit:.0f} s admitted "
                f"({learnability.PROBE_FIT:.0%} of the {cap:.0f} s cap)")
    parts = [head + what + "."]
    if resource["timing"]:
        timing = resource["timing"]
        parts.append(f"Timing probe (job {resource['probe_job']}, {resource['probe_run_s'] or 0:.1f} s): seed "
                     f"{timing['seed']} on {timing['examples']} examples, {learnability.timing_line(timing)}"
                     + (f"; projected full check {resource['projected_s']:.0f} s" if resource["projected_s"]
                        is not None else "") + ".")
    parts.append("Make each training on m examples cheaper without changing the frozen statement or its "
                 "definitions: batched matrix products over the whole batch, np.einsum with optimize=True or "
                 "reshape and @ for gradients summed over batch and time, no per-example or per-token Python loops "
                 "where a batch operation exists. The seeds, m, the threshold and both data controls stay as they are.")
    return " ".join(parts)
