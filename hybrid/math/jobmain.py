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
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# File names the check modules compile submitted sources under (cost, learnability, construction, simulation).
SOURCE_FILES = ("<candidate>", "<learner>", "<construction>", "<prover>", "<finding>")


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


def main():
    problems = verify_staged()
    if problems:
        emit({"status": "unknown", "evidence": "staged check files differ from the manifest: " + "; ".join(problems),
              "counterexample": None})
        return 2
    request = json.loads((HERE / "request.json").read_text(encoding="utf-8"))
    try:
        result = run(request)
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
    sys.exit(main())
