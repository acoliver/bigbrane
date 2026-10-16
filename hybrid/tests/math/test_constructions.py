"""Construction harness: deterministic generators, oracle self-score 1.0, hand-set mechanisms that pass, and
capacity-limited mechanisms that fail above their capacity (plan section 4.4, P2; executed tier)."""
import pytest

from hybrid.math import constructions as c
from hybrid.math import references

TASK_SIZES = [("copy", 16), ("recall", 16), ("dyck", 8), ("composition", 16), ("binding", 8)]


@pytest.mark.parametrize("task,size", TASK_SIZES)
def test_generators_are_deterministic(task, size):
    assert c.generate(task, size, 20, 3) == c.generate(task, size, 20, 3)
    assert c.generate(task, size, 20, 3) != c.generate(task, size, 20, 4)


@pytest.mark.parametrize("task,size", TASK_SIZES)
def test_oracle_self_score_is_one(task, size):
    assert c.oracle_self_score(task, size, 100, 0) == 1.0


def test_copy_layout_and_queries():
    inst = c.generate("copy", 5, 1, 0, {"D": 3})[0]
    sep = c.layout("copy", 5, {"D": 3})["SEP"]
    assert sep == 3 and inst.tokens[5] == sep and len(inst.tokens) == 10
    assert [a for _, a in inst.queries] == list(inst.tokens[:5])
    assert [p for p, _ in inst.queries] == list(range(5, 10))


def test_recall_follows_mqar_format():
    lay = c.layout("recall", 6, {"V": 8})
    inst = c.generate("recall", 6, 1, 0, {"V": 8})[0]
    split = inst.tokens.index(lay["QSEP"])
    keys = inst.tokens[0:split:2]
    assert split == 12 and len(set(keys)) == 6 and all(k < lay["n_keys"] for k in keys)
    assert all(lay["value_base"] <= v < lay["value_base"] + 8 for v in inst.tokens[1:split:2])
    assert sorted(inst.tokens[split + 1:]) == sorted(keys)


def test_dyck_reaches_the_stated_depth_and_asks_for_the_matching_closer():
    for inst in c.generate("dyck", 5, 30, 1):
        depth = peak = 0
        for token in inst.tokens:
            depth += 1 if token in (0, 1) else -1
            peak = max(peak, depth)
            assert depth >= 0
        assert depth == 0 and peak == 5


def test_composition_answer_is_f_of_g():
    lay = c.layout("composition", 3)
    tokens = [lay["GSEP"], 0, 4, 1, 3, 2, 5, lay["FSEP"], 3, 6, 4, 8, 5, 7, lay["QSEP"], 0, 2]
    assert c.solve("composition", 3, tokens) == [(15, 8), (16, 7)]


def test_binding_follows_chains():
    lay = c.layout("binding", 2)
    vb = lay["value_base"]
    # v0 = c1; v1 = v0; v2 = c2; v3 = v1; query v3 then v2.
    tokens = [0, vb + 1, 1, 0, 2, vb + 2, 3, 1, lay["QSEP"], 3, 2]
    assert c.solve("binding", 2, tokens) == [(9, vb + 1), (10, vb + 2)]


@pytest.mark.parametrize("name", sorted(references.CONSTRUCTIONS))
def test_hand_set_constructions(name):
    source, task, sizes, params, expected = references.CONSTRUCTIONS[name]
    result = c.run_construction(source, task, sizes, params=params, count=100)
    assert result["status"] == expected, result["evidence"]
    assert result["tier"] == "executed" and "observed on the listed sizes" in result["evidence"]


def test_capacity_limited_construction_fails_exactly_above_capacity():
    source = references.construction_source(references.RECALL_TABLE, 8)
    result = c.run_construction(source, "recall", [8, 9], params={"V": 16}, count=50)
    by_size = {r["size"]: r for r in result["results"]}
    assert by_size[8]["accuracy"] == 1.0 and by_size[8]["passed"]
    assert by_size[9]["accuracy"] < 0.99 and not by_size[9]["passed"]
    assert result["counterexample"]["size"] == 9
    assert result["counterexample"]["first_error"]["expected"] != result["counterexample"]["first_error"]["predicted"]


def test_state_words_grow_with_the_stored_content():
    source = references.construction_source(references.COPY_BUFFER)
    result = c.run_construction(source, "copy", [4, 16], count=10)
    assert [r["state_peak_words"] for r in result["results"]] == [6, 18]


def test_unknown_task_and_params_fail_fast():
    with pytest.raises(ValueError):
        c.layout("parity", 4)
    with pytest.raises(ValueError):
        c.layout("copy", 4, {"V": 3})
    with pytest.raises(ValueError, match="predict"):
        c.run_construction("def init(l):\n    return 0\ndef update(s, t):\n    return s\n", "copy", [4])
