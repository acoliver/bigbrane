"""Driver for math-library checks inside the tools image, staged as /in/main.py by MathRunner.check_library.

Layout of /in: main.py (this file), bbmath/ (the check modules), manifest.json ({"files": {staged path: sha256}},
written by the host from the verified MANIFEST.json) and request.json. Before importing anything it re-hashes every
staged file against manifest.json, then runs the request and prints one JSON object as its last stdout line
(the obligation-check contract: status, evidence, counterexample, plus the check's details).

Request kinds: conservation {claims}; cost {source, claim, ns?, tolerance?}; construction {source, task, sizes,
count?, seed?, params?, threshold?}; magic_x {claim, assumption, order?, test_count?, seed?}; selfcheck {quick?};
and the frozen-statement kinds of spec_checks.py (Phase 3): z3 {spec}; sympy {spec}; simulation {spec, source};
audited_construction {spec, source}; predicate_agreement {spec, predicate, count?, seed?}; learnability
{spec, source, architecture?} (Phase 4, profile obligation P4); learnability_probe {spec, source, examples?}, the
timing probe that runs before a learnability check and decides nothing; finding {code, sources, specs?, resources?} (FIX-4, FIX-17),
the verification code a challenger finding comes with; and conformance {spec, source} (FIX-6), a cost reference
against the card's frozen equations. audited_construction and cost also take the profile's
architecture contract (architecture.py); cost takes ``precision_bits``, the card's declared precision policy (FIX-20),
and a conformance spec carries its own.
Stdlib only at the top level; NumPy and SymPy are imported by the check modules.

CPU cap (decision CT, 2026-10-17). Every request carries ``cpu_cap_s``, the job's budget in process CPU seconds
(MathRunner, from Caps.cpu_s), and ``cpu_grace_s`` (runner.CPU_GRACE_S). Run as the job's command (``python3 /in/main.py``), the driver sets RLIMIT_CPU to it
before running the check: at the soft limit the kernel sends SIGXCPU, whose handler raises CpuCapReached (a
BaseException, so a check's or a learner's ``except Exception`` cannot absorb it), and the driver prints a status
``unknown`` report with a ``cpu_cap`` record and exits 0. Linux repeats SIGXCPU every CPU second past the soft limit,
and the hard limit, ``cpu_grace_s`` later, kills the process (exit 137 through the supervisor; MathRunner reads that
as the same cap stop). The executor's wall cap is only a hang guard. The process CPU time counts from the start of
the interpreter, so the cap covers the whole job. Called in-process (the test fakes run ``main()`` in the pytest
process), the driver sets no limit. Learnability reports also carry ``job_clock``: at the end of the check, the
process CPU seconds since the interpreter started, the wall seconds since this driver started, and the CPU cap.
"""
import hashlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# File names the check modules compile submitted sources under (cost, learnability, construction, simulation).
SOURCE_FILES = ("<candidate>", "<learner>", "<construction>", "<prover>", "<finding>")
CLOCKED_KINDS = ("learnability", "learnability_probe")
_WALL_START = time.monotonic()


class CpuCapReached(BaseException):
    """The job used its CPU budget (SIGXCPU at the RLIMIT_CPU soft limit)."""


def _on_sigxcpu(signum, frame):
    raise CpuCapReached()


def cpu_used() -> float:
    """CPU seconds of this process (every thread) and its waited-for children, as RLIMIT_CPU counts the process."""
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def install_cpu_cap(cap_s: float, grace_s: int) -> None:
    """RLIMIT_CPU soft limit at the cap (whole seconds, rounded up) and hard limit ``grace_s`` later."""
    soft = max(1, math.ceil(cap_s))
    signal.signal(signal.SIGXCPU, _on_sigxcpu)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + grace_s))


def job_clock(cap_s) -> dict:
    return {"cpu_s": round(cpu_used(), 3), "wall_s": round(time.monotonic() - _WALL_START, 3), "cap_cpu_s": cap_s}


def cap_report(request: dict) -> dict:
    """The report of a job stopped at its CPU cap: status unknown, never a verdict."""
    clock = job_clock(request.get("cpu_cap_s"))
    return {"status": "unknown", "counterexample": None,
            "evidence": (f"CPU cap reached: the {request.get('kind')} job used {clock['cpu_s']:.1f} s of process CPU "
                         f"time against its cap of {request.get('cpu_cap_s')} s ({clock['wall_s']:.1f} s wall) and "
                         "was stopped before it finished; status unknown, not a pass and not a counterexample"),
            "cpu_cap": dict(clock, kind=request.get("kind"))}


def emit(result):
    print(json.dumps(result, default=str, sort_keys=True))


def verify_staged():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))["files"]
    problems = []
    for name, digest in sorted(manifest.items()):
        path = HERE / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"{name}: changed")
    staged = {p.relative_to(HERE).as_posix() for p in HERE.rglob("*.py")}
    problems += [f"{name}: not in the manifest" for name in sorted(staged - set(manifest))]
    return problems


def run(request):
    sys.path.insert(0, str(HERE))
    from bbmath import conservation, constructions, cost, learnability, magic_x, selfcheck, spec_checks

    kind = request["kind"]
    if kind == "z3":
        return spec_checks.z3_property(request["spec"])
    if kind == "sympy":
        return spec_checks.sympy_relation(request["spec"])
    if kind == "simulation":
        return spec_checks.simulation(request["spec"], request["source"])
    if kind == "audited_construction":
        return spec_checks.audited_construction(request["spec"], request["source"], request.get("architecture"))
    if kind == "conformance":
        return spec_checks.reference_conformance(request["spec"], request["source"])
    if kind == "predicate_agreement":
        options = {k: request[k] for k in ("count", "seed") if k in request}
        return spec_checks.predicate_agreement(request["spec"], request["predicate"], **options)
    if kind == "conservation":
        return conservation.report(conservation.check_claims(request["claims"]))
    if kind == "cost":
        options = {k: request[k] for k in ("tolerance", "vocab", "seed", "precision_bits") if k in request}
        if "ns" in request:
            options["ns"] = tuple(request["ns"])
        if request.get("architecture") is not None:
            options["architecture"] = request["architecture"]
        return cost.verify_cost(request["source"], cost.CostClaim(**request["claim"]), **options)
    if kind == "construction":
        options = {k: request[k] for k in ("count", "seed", "params", "threshold") if k in request}
        return constructions.run_construction(request["source"], request["task"], request["sizes"], **options)
    if kind == "magic_x":
        options = {k: request[k] for k in ("order", "test_count", "seed") if k in request}
        return magic_x.magic_x_test(magic_x.AdvantageClaim(**request["claim"]),
                                    magic_x.Assumption(**request["assumption"]), **options)
    if kind == "learnability":
        return learnability.run_learnability(request["spec"], request["source"], request.get("architecture"))
    if kind == "learnability_probe":
        options = {k: request[k] for k in ("examples",) if k in request}
        return learnability.timing_probe(request["spec"], request["source"], **options)
    if kind == "finding":
        from bbmath import findings
        return findings.verify_finding(request["code"], request["sources"], request.get("specs") or {},
                                       request.get("resources") or {})
    if kind == "selfcheck":
        return selfcheck.run(quick=request.get("quick", True))
    raise ValueError(f"unknown request kind {kind!r}")


def main(enforce_cpu_cap: bool = False):
    problems = verify_staged()
    if problems:
        emit({"status": "unknown", "evidence": "staged check files differ from the manifest: " + "; ".join(problems),
              "counterexample": None})
        return 2
    request = json.loads((HERE / "request.json").read_text(encoding="utf-8"))
    if enforce_cpu_cap:
        install_cpu_cap(request["cpu_cap_s"], request["cpu_grace_s"])
    try:
        result = run(request)
        if isinstance(result, dict) and request["kind"] in CLOCKED_KINDS:
            result = dict(result, job_clock=job_clock(request.get("cpu_cap_s")))
    except CpuCapReached:
        # Linux repeats SIGXCPU every CPU second past the soft limit; the report must not be interrupted by it.
        signal.signal(signal.SIGXCPU, signal.SIG_IGN)
        emit(cap_report(request))
        return 0
    except Exception as exc:  # the contract reports any failure to run a check as unknown
        from bbmath import cost
        source = request.get("source") if isinstance(request.get("source"), str) else None
        where = next((line for line in (cost.crash_line(exc, source, name) for name in SOURCE_FILES) if line), None)
        evidence = cost.describe_exception(exc) + (f"; raised at {where} of the submitted source" if where else "")
        emit({"status": "unknown", "evidence": evidence[:2000], "counterexample": None,
              **({"prover_fault": True, "raised_at": where} if where else {})})
        return 1
    emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(enforce_cpu_cap=True))
