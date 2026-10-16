"""Rung 1 tasks (IMPROVEMENT-PLAN.md B.2): generator determinism, published-format invariants, answer-only loss
positions, and oracle self-score 1.0 (an independent parser of the visible prefix agrees with every label)."""
from __future__ import annotations

import numpy as np
import pytest

from hybrid.bench.tasks import copy_task, mqar, ngram_lookup

IGNORE = -100


def rng(seed=0):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence((seed, 7, 1))))


# ------------------------------------------------------------------------------------------------ MQAR

def test_mqar_generator_is_deterministic():
    a = mqar.generate(rng(3), 16, seq_len=128, k_values=[4, 8, 16], vocab=256)
    b = mqar.generate(rng(3), 16, seq_len=128, k_values=[4, 8, 16], vocab=256)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    c = mqar.generate(rng(4), 16, seq_len=128, k_values=[4, 8, 16], vocab=256)
    assert not np.array_equal(a["tokens"], c["tokens"])


@pytest.mark.parametrize("seq_len,k,vocab", [(64, 4, 128), (256, 16, 512), (256, 64, 512), (512, 64, 1024)])
def test_mqar_matches_the_zoology_format(seq_len, k, vocab):
    data = mqar.generate(rng(1), 64, seq_len=seq_len, k_values=[k], vocab=vocab)
    half = vocab // 2
    for row in range(64):
        tokens, targets, dist = data["tokens"][row], data["targets"][row], data["distance"][row]
        keys, values = tokens[0:2 * k:2], tokens[1:2 * k:2]
        assert len(set(keys.tolist())) == k and len(set(values.tolist())) == k        # distinct
        assert keys.min() >= 1 and keys.max() < half                                    # keys: 1 .. V/2 - 1
        assert values.min() >= half and values.max() < vocab                            # values: V/2 .. V - 1
        scored = np.nonzero(targets != IGNORE)[0]
        assert len(scored) == k                                                         # every key queried once
        assert (scored >= 2 * k).all() and ((scored - 2 * k) % 2 == 0).all()           # even query slots
        assert sorted(tokens[scored].tolist()) == sorted(keys.tolist())                 # queries are stored keys
        for pos in scored:
            i = int(np.nonzero(keys == tokens[pos])[0][0])
            assert targets[pos] == values[i]
            assert dist[pos] == pos - 2 * i
        assert (dist[targets == IGNORE] == -1).all()
        assert tokens.min() >= 0 and tokens.max() < vocab


def test_mqar_query_gaps_follow_the_power_law():
    seq_len, k = 256, 4
    space = (seq_len - 2 * k) // 2
    data = mqar.generate(rng(2), 4000, seq_len=seq_len, k_values=[k], vocab=512)
    slots = []
    for row in range(len(data["tokens"])):
        slots += ((np.nonzero(data["targets"][row] != IGNORE)[0] - 2 * k) // 2).tolist()
    counts = np.bincount(slots, minlength=space)
    p = mqar.gap_probabilities(space, 0.01)
    assert p[0] > 50 * p[-1] / 2                       # strongly favours short gaps
    # Sampling k without replacement flattens the law a little; the first slots must still dominate.
    assert counts[0] > counts[10] > counts[100]
    assert counts[:8].sum() > counts[-60:].sum()


def test_mqar_fills_empty_positions_with_random_tokens_over_the_whole_vocabulary():
    data = mqar.generate(rng(5), 200, seq_len=128, k_values=[8], vocab=256)
    filler = data["tokens"][:, 16:][data["targets"][:, 16:] == IGNORE]
    assert filler.min() < 128 <= filler.max()           # both halves appear among fillers
    assert len(np.unique(filler)) > 200


def test_mqar_rejects_settings_outside_the_published_constraints():
    with pytest.raises(ValueError):
        mqar.generate(rng(), 2, seq_len=64, k_values=[17], vocab=512)      # 4K > N
    with pytest.raises(ValueError):
        mqar.generate(rng(), 2, seq_len=512, k_values=[8], vocab=512)      # V must exceed N


@pytest.mark.parametrize("seq_len,k_values", [(64, [4, 8, 16]), (256, [4, 64])])
def test_mqar_oracle_self_score_is_one(seq_len, k_values):
    data = mqar.generate(rng(6), 50, seq_len=seq_len, k_values=k_values, vocab=512)
    checked = 0
    for row in range(50):
        for pos in np.nonzero(data["targets"][row] != IGNORE)[0]:
            assert mqar.oracle(data["tokens"][row][:pos + 1], int(data["k"][row])) == data["targets"][row][pos]
            checked += 1
    assert checked == int(sum(data["k"]))


# ------------------------------------------------------------------------------------------------ copy

def test_copy_structure_and_answer_positions():
    data = copy_task.generate(rng(1), 40, lengths=list(range(1, 33)), alphabet=16)
    s = copy_task.special(16)
    for row in range(40):
        length = int(data["length"][row])
        tokens, targets = data["tokens"][row], data["targets"][row]
        assert tokens[0] == s["bos"] and tokens[length + 1] == s["copy"] and tokens[2 * length + 2] == s["eos"]
        np.testing.assert_array_equal(tokens[1:length + 1], tokens[length + 2:2 * length + 2])
        assert (tokens[1:length + 1] < 16).all()
        assert (tokens[2 * length + 3:] == s["pad"]).all()
        scored = np.nonzero(targets != IGNORE)[0]
        np.testing.assert_array_equal(scored, np.arange(length + 1, 2 * length + 1))   # COPY .. second-last copy
        np.testing.assert_array_equal(targets[scored], tokens[scored + 1])               # next-token labels
    assert copy_task.vocab_size(alphabet=16) == 20


def test_copy_generator_is_deterministic_and_lengths_cover_the_range():
    a = copy_task.generate(rng(9), 500, lengths=list(range(1, 33)), alphabet=16)
    b = copy_task.generate(rng(9), 500, lengths=list(range(1, 33)), alphabet=16)
    np.testing.assert_array_equal(a["tokens"], b["tokens"])
    assert set(a["length"].tolist()) == set(range(1, 33))


def test_copy_oracle_self_score_is_one():
    data = copy_task.generate(rng(2), 30, lengths=[5, 32, 64], alphabet=16)
    for row in range(30):
        for pos in np.nonzero(data["targets"][row] != IGNORE)[0]:
            assert copy_task.oracle(data["tokens"][row][:pos + 1], 16) == data["targets"][row][pos]


def test_copy_bits_bound():
    assert copy_task.bits_needed(64, 16) == 256.0


# ------------------------------------------------------------------------------------------------ lookup

@pytest.mark.parametrize("variant", ngram_lookup.VARIANTS)
def test_lookup_structure_and_unique_key(variant):
    data = ngram_lookup.generate(rng(3), 60, lengths=list(range(6, 31)), variant=variant, n=3, k=3, alphabet=16)
    s = ngram_lookup.special(16)
    for row in range(60):
        tokens, targets = data["tokens"][row], data["targets"][row]
        length = int(data["length"][row])
        real = tokens[tokens != s["pad"]]
        assert real[0] == s["bos"]
        if variant == "suffix":
            x, gram = real[1:1 + length], real[2 + length:5 + length]
            assert real[1 + length] == s["sep"]
        else:
            gram, x = real[1:4], real[5:5 + length]
            assert real[4] == s["sep"] and real[5 + length] == s["sep"]
        found = ngram_lookup.occurrences(x, gram)
        assert len(found) == 1
        scored = np.nonzero(targets != IGNORE)[0]
        assert len(scored) == 3
        np.testing.assert_array_equal(targets[scored], x[found[0] + 3:found[0] + 6])
        np.testing.assert_array_equal(targets[scored], tokens[scored + 1])


@pytest.mark.parametrize("variant", ngram_lookup.VARIANTS)
def test_lookup_oracle_self_score_is_one(variant):
    data = ngram_lookup.generate(rng(4), 40, lengths=[20, 60, 100], variant=variant, n=3, k=3, alphabet=16)
    for row in range(40):
        for pos in np.nonzero(data["targets"][row] != IGNORE)[0]:
            got = ngram_lookup.oracle(data["tokens"][row][:pos + 1], variant, 3, 16)
            assert got == data["targets"][row][pos]


def test_lookup_generator_is_deterministic():
    a = ngram_lookup.generate(rng(5), 20, lengths=[30], variant="suffix")
    b = ngram_lookup.generate(rng(5), 20, lengths=[30], variant="suffix")
    np.testing.assert_array_equal(a["tokens"], b["tokens"])


# ------------------------------------------------------------------------------------------------ host side

def test_eval_sets_keep_answers_on_the_host(work_dir):
    from hybrid.bench import ladder
    cfg = {"test_seed": 11, "tasks": {"m": {"module": "mqar", "vocab": 128, "seq_len": 64, "train_k": [4],
                                            "eval_k": [4, 8], "n_eval": 8}}}
    names = ladder.build_eval_sets(cfg, "m", work_dir)
    assert names == ["K4", "K8"]
    staged = np.load(work_dir / "m" / "K8.npz")
    assert set(staged.files) == {"tokens", "positions"}                   # no targets go to the job
    oracle = np.load(work_dir / "m" / "K8.oracle.npz")
    assert int(staged["positions"].sum()) == 8 * 8
    again = ladder.build_eval_sets(cfg, "m", work_dir)                    # idempotent
    assert again == names
    pred = oracle["targets"][oracle["positions"]].astype(np.int16)
    score = ladder.score_set("mqar", oracle, pred)
    assert score["acc"] == 1.0 and score["seq_acc"] == 1.0
    pred[0] = -1
    assert ladder.score_set("mqar", oracle, pred)["seq_acc"] == pytest.approx(7 / 8)


def test_shipped_ladder_config_validates_and_its_slices_name_known_cells():
    from hybrid.bench import ladder
    from .conftest import ROOT
    cfg = ladder.load_config(ROOT / "hybrid" / "bench" / "configs" / "ladder.toml")
    for slice_ in cfg["slices"].values():
        for cell in slice_["cells"]:
            assert cell["task"] in cfg["tasks"]
            assert all(model in cfg["models"] for model in cell["models"])
    for task in cfg["tasks"].values():
        module = ladder.TASK_MODULES[task["module"]]
        rng_ = np.random.default_rng(0)
        for _name, kwargs, _label in ladder.eval_set_defs(task):
            module.generate(rng_, 2, **kwargs)
        module.generate(rng_, 2, **ladder.train_params(task))
