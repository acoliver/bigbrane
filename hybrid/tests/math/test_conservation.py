"""Conservation-law library against published and hand-derived numbers (plan section 5.2)."""
import math

import pytest

from hybrid.math import conservation as cl


def test_every_law_records_statement_assumptions_and_source():
    assert set(cl.LAWS) == {"storage", "copy", "dyck", "index", "associative_recall", "sample", "read"}
    for law in cl.LAWS.values():
        assert law.statement and law.assumptions and law.source
        assert hasattr(cl, law.function)


def test_storage_k_facts_of_b_bits_need_kb_bits():
    assert cl.facts_bits(100, 10) == 1000
    assert cl.storage_check(1000, 100, 10).status == "discharged"


def test_planted_storage_overrun_names_the_smallest_failing_k():
    check = cl.storage_check(999, 100, 10)
    assert check.status == "violated" and check.kill_code == "conservation:storage"
    assert check.counterexample["smallest_failing_facts"] == 100
    assert check.counterexample["shortfall_bits"] == 1
    # 99 facts of 10 bits fit in 999 bits; the 100th does not.
    assert cl.storage_check(999, 99, 10).status == "discharged"


def test_jelassi_theorem_2_7_values():
    # error > 1 - 2^S / D^L: 10 bits, binary strings of length 20.
    assert cl.copy_error_lower_bound(10, 2, 20) == pytest.approx(1 - 2 ** -10, abs=1e-15)
    # 64 bits copy 16 hex symbols exactly; one more symbol leaves 1/16 of strings.
    assert cl.copy_error_lower_bound(64, 16, 16) == 0.0
    assert cl.copy_error_lower_bound(64, 16, 17) == pytest.approx(0.9375)
    # State well above the string: no bound.
    assert cl.copy_error_lower_bound(1000, 4, 10) == 0.0


def test_copy_claim_above_the_bound_is_violated_with_the_failing_length():
    check = cl.copy_check(10, 2, 20, 0.5)
    assert check.status == "violated"
    # 2^10 / 2^11 = 0.5 is still allowed; at L = 12 the bound is 0.25.
    assert check.counterexample["smallest_failing_L"] == 12
    assert cl.copy_check(10, 2, 11, 0.5).status == "discharged"
    assert cl.copy_check(10, 2, 12, 0.5).status == "violated"


def test_dyck_depth_needs_one_bit_per_level():
    assert cl.dyck_check(8, 8, 1.0).status == "discharged"
    check = cl.dyck_check(8, 9, 1.0)
    assert check.status == "violated" and check.counterexample["smallest_failing_depth"] == 9
    assert cl.dyck_error_lower_bound(8, 10) == pytest.approx(0.75)


def test_index_bound_exact_and_with_error():
    assert cl.streaming_index_required_bits(1024) == 1024
    # h2(0.11) = 0.499916 by hand: -0.11 log2 0.11 - 0.89 log2 0.89.
    assert cl.h2(0.11) == pytest.approx(0.499916, abs=1e-6)
    assert cl.streaming_index_required_bits(1024, 0.11) == pytest.approx(512.086, abs=0.01)
    check = cl.streaming_index_bound(1024, 1000)
    assert check.status == "violated" and check.counterexample["smallest_failing_n"] == 1001
    assert cl.streaming_index_bound(1024, 1024).status == "discharged"
    with pytest.raises(ValueError):
        cl.streaming_index_required_bits(10, 0.5)


def test_associative_recall_reduces_to_index_over_v_symbols():
    assert cl.associative_recall_required_bits(64, 256) == 64 * 8
    # Chance accuracy needs no bits.
    assert cl.associative_recall_required_bits(64, 256, 1 - 1 / 256) == pytest.approx(0.0, abs=1e-9)


def test_planted_recall_capacity_overrun():
    # K=64 values over V=256 at accuracy 0.99 need 64 * (8 - h2(0.01) - 0.01 log2 255) = 501.71 bits.
    check = cl.associative_recall_check(256, 64, 256, 0.99)
    assert check.status == "violated" and check.kill_code == "conservation:associative_recall"
    assert check.bound["required_bits"] == pytest.approx(501.713, abs=1e-3)
    assert check.counterexample["smallest_failing_K"] == 33
    assert cl.associative_recall_check(256, 32, 256, 0.99).status == "discharged"


def test_sample_bound_known_numbers():
    # 1024 hypotheses, one bit per example: binary search attains 10.
    assert cl.sample_bound(10, 0, 1) == 10
    # Permutation of 10 items, one (x, pi(x)) pair per example: ceil(21.79 / 3.32) = 7.
    assert cl.sample_bound(math.log2(math.factorial(10)), 0, math.log2(10)) == 7
    # Prior bits are subtracted: 4 of the 10 bits built in leave 6 examples.
    assert cl.sample_bound(10, 4, 1) == 6
    # Fano at error 0.1: (0.9 * 100 - 1) / 1 = 89.
    assert cl.sample_bound(100, 0, 1, 0.1) == 89
    assert cl.sample_bound(10, 20, 1) == 0


def test_sample_claim_with_too_few_examples_is_violated():
    check = cl.sample_check(10, 2, 1, 5)
    assert check.status == "violated" and check.bound["required_examples"] == 8
    assert check.counterexample["unresolved_log2_hypotheses"] == 3
    assert cl.sample_check(10, 2, 1, 8).status == "discharged"


def test_read_bound():
    assert cl.read_check(100, 99).status == "violated"
    assert cl.read_check(100, 100).status == "discharged"


def test_check_claims_reads_a_card_and_reports_the_first_violation():
    claims = {"state_bits": 256,
              "recall": [{"K": 64, "V": 256, "accuracy": 0.99}],
              "copy": [{"D": 4, "L": 64, "accuracy": 0.99}],
              "dyck": [{"depth": 300, "accuracy": 1.0, "state_bits": 512}],
              "sample": [{"log2_H": 20, "prior_bits": 0, "bits_per_example": 1, "examples": 20}],
              "reads": [{"n": 4096, "ops": 4096}]}
    checks = cl.check_claims(claims)
    assert [c.law for c in checks] == ["copy", "dyck", "associative_recall", "sample", "read"]
    report = cl.report(checks)
    assert report["status"] == "violated"
    assert report["kill_codes"] == ["conservation:associative_recall"]
    assert report["counterexample"]["law"] == "associative_recall"
    assert report["counterexample"]["smallest_failing_K"] == 33


def test_check_claims_discharged_when_everything_fits():
    report = cl.report(cl.check_claims({"state_bits": 4096, "storage": [{"facts": 100, "bits_per_fact": 8}]}))
    assert report["status"] == "discharged" and report["counterexample"] is None


@pytest.mark.parametrize("claims", [
    {"recall": [{"K": 4, "V": 16, "accuracy": 1.0}]},          # no state_bits anywhere
    {"state_bits": 8, "recall": [{"K": 4, "accuracy": 1.0}]},  # missing V
    {"state_bits": 8, "attention": []},                         # unknown section
])
def test_unreadable_claims_fail_fast(claims):
    with pytest.raises(ValueError):
        cl.check_claims(claims)


def test_report_needs_at_least_one_check():
    with pytest.raises(ValueError):
        cl.report([])


def test_find_state_collision_gives_an_executed_pigeonhole_witness():
    # A "state" of 3 bits for 4-bit strings: two strings share it and differ in what must be recalled.
    strings = [tuple((i >> b) & 1 for b in range(4)) for i in range(16)]
    witness = cl.find_state_collision(lambda s: s[:3], strings, lambda s: s)
    assert witness is not None and witness["first"][:3] == witness["second"][:3]
    assert witness["first"] != witness["second"]
    assert cl.find_state_collision(lambda s: s, strings, lambda s: s) is None
