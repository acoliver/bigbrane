"""Checks of frozen statements (hybrid/math/spec_checks.py) on the host: expression whitelist, statement data
validation, SymPy relations, simulations, the audited construction harness and statement hashing. Z3 is in the
tools image only; its cases here are skipped without z3 and run live in spec_smoke (test_spec_checks_live.py)."""
import pytest

from hybrid.math import references, spec_checks as sc
from hybrid.math.runner import MathRunner
from hybrid.tests.math.test_library_manifest import IMAGE, HostExecutor

Z3_SPEC = {"variables": {"x": {"sort": "int", "lo": 0, "hi": 1000}}, "hypotheses": [],
           "property": "x * x != 2 * x + 3"}


def test_expression_whitelist_refuses_everything_but_arithmetic_and_listed_calls():
    names, funcs = {"x", "y"}, sc.Z3_FUNCS
    assert sc.expression_problems("And(x <= y, Not(x == 3))", names, funcs, allow_compare=True) == []
    assert "unknown name 'z'" in sc.expression_problems("x + z", names, funcs, allow_compare=True)
    assert any("call to" in p for p in sc.expression_problems("__import__('os')", names, funcs, allow_compare=True))
    assert any("Attribute" in p for p in sc.expression_problems("x.real", names, funcs, allow_compare=True))
    assert any("chained" in p for p in sc.expression_problems("0 <= x <= 3", names, funcs, allow_compare=True))
    assert any("and/or" in p for p in sc.expression_problems("x > 0 and y > 0", names, funcs, allow_compare=True))
    assert any("not a number" in p for p in sc.expression_problems("x + 'a'", names, funcs, allow_compare=True))
    assert any("comparisons" in p for p in sc.expression_problems("x < y", names, sc.SYMPY_FUNCS,
                                                                  allow_compare=False))
    assert any("Lambda" in p for p in sc.expression_problems("(lambda: 1)()", names, funcs, allow_compare=True))


def test_z3_statement_must_be_bounded():
    unbounded = {"variables": {"n": {"sort": "int"}}, "property": "n >= 0"}
    assert any("bounds" in p for p in sc.statement_problems("z3", unbounded))
    assert sc.statement_problems("z3", Z3_SPEC) == []


def test_statement_hash_is_canonical_json():
    a = {"lhs": "x", "rhs": "x", "relation": "==", "symbols": {"x": "real"}}
    b = {"symbols": {"x": "real"}, "relation": "==", "rhs": "x", "lhs": "x"}
    assert sc.spec_hash(a) == sc.spec_hash(b)
    assert sc.spec_hash(a) != sc.spec_hash(dict(a, rhs="x + 0*x + 1"))


def test_sympy_identity_is_proven_with_spot_checks():
    spec = {"symbols": {"x": "real", "y": "real"}, "lhs": "(x + y)**3",
            "rhs": "x**3 + 3*x**2*y + 3*x*y**2 + y**3", "relation": "=="}
    result = sc.sympy_relation(spec)
    assert (result["status"], result["tier"]) == ("discharged", "sympy_exact")
    assert "proven" in result["evidence"]


def test_sympy_false_identity_has_a_counterexample():
    result = sc.sympy_relation({"symbols": {"x": "real"}, "lhs": "(x + 1)**2", "rhs": "x**2 + 1",
                                "relation": "=="})
    assert result["status"] == "violated" and result["counterexample"]["relation"] == "=="


def test_sympy_inequality_by_sign_and_a_wrong_direction():
    amgm = {"symbols": {"a": "positive", "b": "positive"}, "lhs": "(a - b)**2", "rhs": "0", "relation": ">="}
    assert sc.sympy_relation(amgm)["status"] == "discharged"
    wrong = dict(amgm, relation="<")
    assert sc.sympy_relation(wrong)["status"] == "violated"


def test_sympy_unprovable_inequality_with_agreeing_points_is_unknown():
    spec = {"symbols": {"n": "positive_integer"}, "lhs": "2**n", "rhs": "n**2", "relation": ">=",
            "conditions": ["n >= 4"]}
    assert sc.sympy_relation(spec)["status"] in ("unknown", "discharged")


INVOLUTION = {
    "generator": "def generate(size, rng):\n    xs = [int(v) for v in rng.integers(-50, 50, size=size)]\n"
                 "    return {'input': xs, 'target': None}\n",
    "metric": "def metric(given, target, output):\n"
              "    pairs = list(zip(given, output['f'], output['ff']))\n"
              "    return sum(1.0 for x, fx, ffx in pairs if ffx == x) / max(1, len(pairs))\n",
    "sizes": [4, 16], "count": 20, "threshold": 1.0, "seed": 3}
GOOD_RUN = "def run(xs):\n    f = [7 - x for x in xs]\n    return {'f': f, 'ff': [7 - y for y in f]}\n"
BAD_RUN = "def run(xs):\n    f = [2 * x for x in xs]\n    return {'f': f, 'ff': [2 * y for y in f]}\n"


def test_simulation_observed_on_sizes_and_never_called_proven():
    result = sc.simulation(INVOLUTION, GOOD_RUN)
    assert (result["status"], result["tier"]) == ("discharged", "executed")
    assert "observed on the listed sizes" in result["evidence"] and "proven" not in result["evidence"]


def test_simulation_failure_names_the_size_and_instance():
    result = sc.simulation(INVOLUTION, BAD_RUN)
    assert result["status"] == "violated"
    assert result["counterexample"]["failing_size"] == 4 and "input" in result["counterexample"]


def test_simulation_refuses_prover_code_that_imports_or_reaches_private_state():
    for source in ("import os\ndef run(x):\n    return x\n", "def run(x):\n    return x.__class__\n",
                   "def run(x):\n    return getattr(x, 'a')\n"):
        result = sc.simulation(INVOLUTION, source)
        assert result["status"] == "unknown" and result["prover_fault"], source


def test_simulation_with_a_broken_metric_is_unknown_not_a_prover_fault():
    spec = dict(INVOLUTION, metric="def metric(given, target, output):\n    return 2.0\n")
    result = sc.simulation(spec, GOOD_RUN)
    assert result["status"] == "unknown" and not result.get("prover_fault")


def test_audited_construction_runs_the_harness_and_refuses_imports():
    source = references.construction_source(references.RECALL_TABLE)
    spec = {"task": "recall", "sizes": [4], "count": 10}
    assert sc.audited_construction(spec, source)["status"] == "discharged"
    refused = sc.audited_construction(spec, "import os\n" + source)
    assert refused["status"] == "unknown" and refused["prover_fault"]


def test_new_kinds_run_through_check_library_on_the_host(tmp_path):
    runner = MathRunner(HostExecutor(tmp_path), IMAGE, tmp_path / "work")
    record = runner.check_library({"kind": "simulation", "spec": INVOLUTION, "source": GOOD_RUN})
    assert (record["status"], record["decided"], record["replayed"]) == ("discharged", True, True)
    assert "bbmath/spec_checks.py" in record["check_hashes"]
    sympy = runner.check_library({"kind": "sympy", "spec": {"symbols": {"x": "real"}, "lhs": "(x+1)**2",
                                                            "rhs": "x**2+2*x+1", "relation": "=="}})
    assert sympy["status"] == "discharged" and sympy["runs"][0]["printed"]["tier"] == "sympy_exact"


def test_z3_bounded_property_and_counterexample():
    pytest.importorskip("z3")
    result = sc.z3_property(Z3_SPEC)
    assert result["status"] == "violated" and result["counterexample"] == {"x": 3}
    holds = sc.z3_property({"variables": {"x": {"sort": "int", "lo": 0, "hi": 50}}, "property": "x * x >= x"})
    assert holds["status"] == "discharged" and "proven up to bound x in [0, 50]" in holds["evidence"]


def test_predicate_agreement_catches_a_strengthened_exponent():
    pytest.importorskip("z3")
    formal = {"variables": {"n": {"sort": "int", "lo": 1, "hi": 64}, "c": {"sort": "int", "lo": 1, "hi": 64}},
              "property": "c * n * n <= 64 * n * n"}
    agree = sc.predicate_agreement(formal, "def claim(n, c):\n    return c * n * n <= 64 * n * n\n", count=200)
    assert agree["status"] == "discharged"
    stronger = sc.predicate_agreement(formal, "def claim(n, c):\n    return c * n <= 64 * n\n", count=200)
    assert stronger["status"] == "discharged"
    differs = sc.predicate_agreement(formal, "def claim(n, c):\n    return c * n * n <= 8 * n\n", count=200)
    assert differs["status"] == "violated"
