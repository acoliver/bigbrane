"""Magic-X harness (plan section 5.1): an assumption that alone gives the advantage fails the card; an
uninformative assumption does not; an assumption proven or measured with its cost in P1 is exempt."""
import pytest

from hybrid.math import magic_x as mx
from hybrid.math import references

CLAIM = mx.AdvantageClaim(**references.MAGIC_X_CLAIM)


def test_retrieval_oracle_alone_gives_the_advantage_so_the_card_fails():
    result = mx.magic_x_test(CLAIM, mx.Assumption("X1", "store", references.RETRIEVAL_ORACLE))
    assert result["status"] == "violated" and result["kill_code"] == "magic_x"
    ce = result["counterexample"]
    assert ce["accuracy_baseline_plus_x"] >= CLAIM.value - CLAIM.tolerance
    assert ce["accuracy_baseline_alone"] < 0.2
    # The assumption's cost is measured, not declared: it scans the context on every query.
    assert ce["x_cost"]["calls"] > 0 and ce["x_cost"]["lines_max_per_call"] > 10


def test_uninformative_assumption_leaves_the_baseline_at_chance():
    result = mx.magic_x_test(CLAIM, mx.Assumption("X2", "oracle", references.POSITION_PARITY))
    assert result["status"] == "discharged"
    assert result["details"]["accuracy_baseline_plus_x"] < 0.2


@pytest.mark.parametrize("status,cost_in_p1,expected", [
    ("measured", True, "discharged"),
    ("proven", True, "discharged"),
    ("measured", False, "violated"),
    ("cited", True, "violated"),
    ("open", False, "violated"),
])
def test_exemption_needs_proven_or_measured_and_cost_in_p1(status, cost_in_p1, expected):
    assumption = mx.Assumption("X1", "store", references.RETRIEVAL_ORACLE, status, cost_in_p1)
    assert mx.magic_x_test(CLAIM, assumption)["status"] == expected


def test_assumption_that_cannot_run_is_unknown():
    result = mx.magic_x_test(CLAIM, mx.Assumption("X3", "oracle", "import os\ndef x(p, l, o):\n    return 0\n"))
    assert result["status"] == "unknown" and result["assumption_status"] == "open"


def test_assumption_sees_only_the_prefix():
    seen = []
    source = "def x(prefix, layout, ops):\n    return len(prefix)\n"
    runner = mx._XRunner(mx.Assumption("X4", "other", source))
    inst = mx.constructions.generate("recall", 4, 1, 0)[0]
    lay = mx.constructions.layout("recall", 4)
    for prefix, feature, _answer in mx._rows([inst], lay, runner):
        seen.append(feature)
        assert tuple(prefix) == inst.tokens[: len(prefix)]
    assert seen == [p + 1 for p, _ in inst.queries]


def test_ngram_baseline_backs_off_and_breaks_ties_deterministically():
    model = mx.NgramBaseline(order=1)
    model.fit([((1,), None, 5), ((1,), None, 6), ((2,), None, 7)])
    assert model.predict((1,), None) == 5
    assert model.predict((9,), None) in (5, 6, 7)
    model.fit([((3,), 42, 9)])
    assert model.predict((8,), 42) == 9


def test_unknown_kind_or_status_fails_fast():
    with pytest.raises(ValueError):
        mx.Assumption("X", "magic", "def x(p, l, o):\n    return 0\n")
    with pytest.raises(ValueError):
        mx.Assumption("X", "oracle", "def x(p, l, o):\n    return 0\n", status="likely")
