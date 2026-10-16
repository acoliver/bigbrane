"""Work hidden from the op counter (review fix 2026-10-16, cost.COSTED_WORK).

Before the fix six of these quadratic candidates, each claimed linear, verified as discharged: their per-token O(t)
work sat inside a builtin, a container operator or an array copy, which neither the op counter nor the line tracer
saw (results/review-fixes-20261016/probe_cost_before.json). Each must now fail with the touched-word check, the line
check, or a refusal.
"""
import pytest

from hybrid.math import cost, references

NS = (32, 64, 128, 256)
LINEAR = cost.CostClaim(ops="n", state="n", sizes={}, exponent=1)
TICK = "    ops.np.add(ops.np.zeros(1), 1.0)\n"
LIST_STATE = "def init(sizes, ops):\n    return {}, {'h': []}\n"
APPEND = "    h = state['h']\n    h.append(token)\n    return {'h': h}\n"


def candidate(update_body: str, predict_body: str = "    return 0\n", init: str = LIST_STATE) -> str:
    return (init + "def update(params, state, token, ops):\n" + TICK + update_body
            + "def predict(params, state, ops):\n" + predict_body)


HIDDEN = {
    "list concatenation and sum": candidate("    return {'h': state['h'] + [token]}\n",
                                            "    return sum(state['h'])\n"),
    "max and membership on a list": candidate(APPEND, "    return max(state['h']) + (99 in state['h'])\n"),
    "list slice copy": candidate("    h = state['h'][:]\n    h.append(token)\n    return {'h': h}\n"),
    "list copy method": candidate("    h = state['h'].copy()\n    h.append(token)\n    return {'h': h}\n"),
    "list built from the history": candidate(APPEND, "    return len(list(state['h']))\n"),
    "insert at the front": candidate("    h = state['h']\n    h.insert(0, token)\n    return {'h': h}\n"),
    "array concatenation": candidate(
        "    return {'h': ops.np.concatenate([state['h'], ops.np.full(1, float(token))])}\n",
        init="def init(sizes, ops):\n    return {}, {'h': ops.np.zeros(0)}\n"),
    "array copy and reshape": candidate(
        "    return {'h': ops.np.concatenate([state['h'], ops.np.full(1, float(token))])}\n",
        "    snapshot = state['h'].copy().reshape(1, -1)\n    return 0\n",
        init="def init(sizes, ops):\n    return {}, {'h': ops.np.zeros(0)}\n"),
    "array fancy index": candidate(
        "    h = state['h']\n    h.append(token)\n    return {'h': h}\n",
        "    a = ops.np.ones(len(state['h']))\n    return 0\n"),
}


@pytest.mark.parametrize("name", sorted(HIDDEN))
def test_hidden_per_token_work_fails_a_linear_claim(name):
    result = cost.verify_cost(HIDDEN[name], LINEAR, ns=NS)
    assert result["status"] == cost.VIOLATED, result["evidence"]
    assert "touched_exponent_n" in result["failed_checks"]
    (slope,) = [s for s in result["slopes"] if s["parameter"] == "n"]
    assert slope["measured"] == 1.0 and slope["touched"] > 1.8      # counted ops linear, the hidden work quadratic


def test_a_comprehension_over_the_history_is_still_caught_by_the_line_check():
    """The one hidden-quadratic probe the old checker already caught: loops and comprehensions are traced lines."""
    source = candidate(APPEND, "    return len([v for v in state['h'] if v == 3])\n")
    result = cost.verify_cost(source, LINEAR, ns=NS)
    assert result["status"] == cost.VIOLATED, result["evidence"]
    assert "lines_exponent_n" in result["failed_checks"]


def test_sorting_in_costed_code_is_refused_with_the_allowed_operations():
    source = candidate(APPEND, "    return sorted(state['h'])[0]\n")
    with pytest.raises(cost.CandidateError, match="sorted is not allowed in costed code") as caught:
        cost.verify_cost(source, LINEAR, ns=NS)
    assert cost.COSTED_WORK in str(caught.value)
    with pytest.raises(cost.CandidateError, match="list.sort"):
        cost.verify_cost(candidate("    h = state['h']\n    h.append(token)\n    h.sort()\n    return {'h': h}\n"),
                         LINEAR, ns=NS)


@pytest.mark.parametrize("body, match", [
    ("    return state['h'] == [1, 2]\n", "comparing list with list"),
    ("    return 3 in ops.np.ones(3)\n", "`in` on Counted"),
    ("    return {1} | {2}\n", "set algebra"),
    ("    return 'a b'.split()\n", "str.split"),
    ("    return f\"{state['h']}\"\n", "formatting a container"),
])
def test_work_that_cannot_be_counted_exactly_is_refused(body, match):
    with pytest.raises(cost.CandidateError, match=match):
        cost.verify_cost(candidate(APPEND, body), LINEAR, ns=NS)


@pytest.mark.parametrize("body, match", [
    ("    return max(*state['h'])\n", r"\*-unpacking"),
    ("    return dict(**state)\n", r"\*\*-unpacking"),
    ("    t = state['h']\n    return 0 < len(t) < 9\n", "chained comparison whose middle operand"),
    ("    __bb_getslice__ = 1\n    return 0\n", "'__bb_getslice__' is not allowed"),
])
def test_uncountable_syntax_is_refused_before_anything_runs(body, match):
    source = candidate(APPEND, body)
    with pytest.raises(cost.CandidateError, match=match):
        cost.load_candidate(source)
    assert any(__import__("re").search(match, p) for p in cost.audit_source(source) + cost.sandbox_problems(
        source, "cost"))


def test_the_touched_tally_is_exact():
    """`state['h'] + [token]` builds a list of t + 1 items at step t: 1 + 2 + 3 + 4 = 10 words over four tokens,
    plus the one-element zeros array TICK allocates per token."""
    source = candidate("    return {'h': state['h'] + [token]}\n")
    m = cost.measure(cost.load_candidate(source), {"n": 4})
    assert m.touched == 10 + 4 and m.ops == 4


def test_an_honest_linear_history_still_verifies():
    """Appending and reading the last item touch O(1) per token, so a linear claim with linear state holds."""
    source = candidate(APPEND, "    return state['h'][-1] + max(1, 2)\n")
    result = cost.verify_cost(source, LINEAR, ns=NS)
    assert result["status"] == cost.DISCHARGED, result["evidence"]


def test_init_and_uncosted_calls_are_not_charged_or_refused():
    source = ("def init(sizes, ops):\n    order = sorted([3, 1, 2])\n    return {}, {'h': list(order)}\n"
              "def update(params, state, token, ops):\n" + TICK + "    return state\n"
              "def predict(params, state, ops):\n    return state['h'][0]\n")
    fns = cost.load_candidate(source)
    params, state = fns.init({}, cost.OpCounter())
    assert state == {"h": [1, 2, 3]} and fns.predict(params, fns.update(params, state, 0, cost.OpCounter()),
                                                     cost.OpCounter()) == 1


def test_builtin_types_and_shadowed_names_behave_as_before():
    source = ("def init(sizes, ops):\n    return {}, {'h': [], 'kind': isinstance([], list)}\n"
              "def update(params, state, token, ops):\n" + TICK
              + "    sum = lambda xs: 7\n    h = state['h']\n    h.append(sum(h))\n    return {'h': h, 'kind': True}\n"
              "def predict(params, state, ops):\n    return 0\n")
    fns = cost.load_candidate(source)
    params, state = fns.init({}, cost.OpCounter())
    assert state["kind"] is True
    with cost.LineTracer() as tracer:
        state = fns.update(params, state, 1, cost.OpCounter())
    assert state["h"] == [7] and tracer.touched == 2     # zeros(1) of TICK and the one append; the local sum is free


@pytest.mark.parametrize("name", ["elman_rnn", "softmax_attention", "decayed_unigram", "hmm_forward"])
def test_the_honest_references_stay_within_their_order(name):
    source, claim, expected = references.REFERENCES[name]
    result = cost.verify_cost(source, cost.CostClaim(**claim), ns=NS)
    assert expected == "discharged" and result["status"] == cost.DISCHARGED, result["failures"]
    assert not [f for f in result["failures"] if f["check"].startswith("touched_exponent")]


@pytest.mark.parametrize("name", ["planted_quadratic", "planted_uncounted_loop"])
def test_the_planted_references_keep_their_verdicts(name):
    source, claim, expected = references.REFERENCES[name]
    assert cost.verify_cost(source, cost.CostClaim(**claim), ns=NS)["status"] == expected
