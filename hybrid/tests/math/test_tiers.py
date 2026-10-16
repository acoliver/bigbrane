"""Tool tiers and the verdict wording rule (plan section 4.2)."""
import pytest

from hybrid.math import tiers


def test_only_lean_and_sympy_exact_are_reported_as_proven():
    assert tiers.verdict_phrase(tiers.LEAN) == "proven"
    assert tiers.verdict_phrase(tiers.SYMPY_EXACT) == "proven"
    assert tiers.verdict_phrase(tiers.SMT_BOUNDED, bound="n <= 64") == "proven up to bound n <= 64"
    assert tiers.verdict_phrase(tiers.EXECUTED, sizes=["K=4", "K=8"]) == "observed on the listed sizes: K=4, K=8"
    assert "proven" not in tiers.verdict_phrase(tiers.ARGUED)


def test_bounded_and_executed_results_must_state_their_scope():
    with pytest.raises(ValueError):
        tiers.verdict_phrase(tiers.SMT_BOUNDED)
    with pytest.raises(ValueError):
        tiers.verdict_phrase(tiers.EXECUTED)


def test_argued_never_satisfies_an_obligation():
    assert [t for t in tiers.TIERS if not tiers.satisfies_obligation(t)] == [tiers.ARGUED]
    with pytest.raises(ValueError):
        tiers.satisfies_obligation("judged")
