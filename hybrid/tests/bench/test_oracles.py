"""Oracles against brute-force enumeration on tiny instances, and samplers against oracles."""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest

from hybrid.bench.tasks import build_task, compose, fewshot, hmm, recall, vom
from hybrid.bench.tasks.common import PARAMS, stream


# ---------------------------------------------------------------- HMM

def brute_hmm_joint(x, A, B, pi):
    """P(x) by summing over every hidden path."""
    k = len(pi)
    total = 0.0
    for z in itertools.product(range(k), repeat=len(x)):
        p = pi[z[0]] * B[z[0], x[0]]
        for t in range(1, len(x)):
            p *= A[z[t - 1], z[t]] * B[z[t], x[t]]
        total += p
    return total


def test_hmm_filter_matches_enumeration():
    A, B, pi = hmm.make_hmm(3, n_states=2, vocab=4, rho=0.7, eta=0.6)
    seqs = np.array(list(itertools.product(range(4), repeat=4)))
    out = hmm.hmm_filter(seqs, A, B, pi)
    for i, x in enumerate(seqs):
        for t in range(4):
            prefix = list(x[:t])
            denom = brute_hmm_joint(prefix, A, B, pi) if t else 1.0
            expected = [brute_hmm_joint(prefix + [v], A, B, pi) / denom for v in range(4)]
            np.testing.assert_allclose(out[i, t], expected, rtol=1e-12, atol=1e-14)


def test_hmm_task_oracle_matches_enumeration():
    task = build_task("hmm", 5, 4, 40, {"n_states": 2, "vocab": 4, "rho": 0.8, "eta": 0.7, "length": 6})
    A, B, pi = (np.array(task.meta[k]) for k in ("A", "B", "pi"))
    for prefix, dist in zip(task.prefixes, task.oracle):
        denom = brute_hmm_joint(prefix, A, B, pi)
        expected = [brute_hmm_joint(prefix + [v], A, B, pi) / denom for v in range(4)]
        np.testing.assert_allclose(dist, expected, rtol=1e-12)


def test_hmm_sampler_matches_oracle_joint():
    A, B, pi = hmm.make_hmm(4, n_states=2, vocab=4, rho=0.7, eta=0.6)
    n = 40000
    xs = hmm.sample(A, B, pi, n, 2, np.random.default_rng(0))
    counts = Counter(map(tuple, xs.tolist()))
    for x in itertools.product(range(4), repeat=2):
        p = brute_hmm_joint(list(x), A, B, pi)
        assert abs(counts[x] / n - p) < 5 * np.sqrt(p * (1 - p) / n) + 1e-4


# ---------------------------------------------------------------- variable-order Markov

def naive_vom(tree, history):
    """Longest node of the (suffix-closed) context set that is a suffix of the history."""
    best = ()
    for context in tree.dists:
        if len(context) <= len(history) and tuple(history[len(history) - len(context):]) == context:
            if len(context) > len(best):
                best = context
    return tree.dists[best]


def tiny_tree(seed=2):
    return vom.ContextTree.random(stream(seed, "vom", PARAMS), 3, (1.0, 0.7, 0.5), 0.4)


def test_vom_walk_matches_naive_longest_suffix():
    tree = tiny_tree()
    assert tree.max_depth() >= 2
    for length in range(0, 5):
        for history in itertools.product(range(3), repeat=length):
            np.testing.assert_array_equal(tree.next_dist(list(history)), naive_vom(tree, list(history)))


def test_vom_task_oracle_matches_naive():
    params = {"vocab": 3, "split_probs": [1.0, 0.7, 0.5], "alpha": 0.4, "length": 12}
    task = build_task("vom", 2, 4, 60, params)
    tree = tiny_tree(2)
    for prefix, dist in zip(task.prefixes, task.oracle):
        np.testing.assert_array_equal(dist, naive_vom(tree, prefix))


def test_vom_sampler_matches_oracle_joint():
    tree = tiny_tree()
    n = 40000
    xs = tree.sample(n, 3, np.random.default_rng(1))
    counts = Counter(map(tuple, xs.tolist()))
    total = 0.0
    for x in itertools.product(range(3), repeat=3):
        p = np.prod([naive_vom(tree, list(x[:t]))[x[t]] for t in range(3)])
        total += p
        assert abs(counts[x] / n - p) < 5 * np.sqrt(p * (1 - p) / n) + 1e-4
    assert abs(total - 1) < 1e-12


# ---------------------------------------------------------------- compose

def brute_compose_joint(seq, T, F, D, w, filler):
    """P(seq) written directly from the generative rules, independent of BracketSource."""
    p, stack = 1.0, []
    for tok in seq:
        can_open, can_close = len(stack) < D, bool(stack)
        z = (w[0] if can_open else 0) + (w[1] if can_close else 0) + w[2]
        if tok < T:
            step = w[0] / z / T if can_open else 0.0
        elif tok < 2 * T:
            step = w[1] / z if can_close and stack[-1] == tok - T else 0.0
        else:
            step = w[2] / z * filler[stack[-1] + 1 if stack else 0, tok - 2 * T]
        p *= step
        if p == 0:
            return 0.0
        if tok < T:
            stack.append(tok)
        elif tok < 2 * T:
            stack.pop()
    return p


def test_compose_oracle_matches_enumeration():
    T, F, D, w = 2, 2, 2, (0.3, 0.3, 0.4)
    src = compose.make_source(3, T, F, D, *w, 0.5)
    V = 2 * T + F
    checked = 0
    for length in range(0, 5):
        for prefix in itertools.product(range(V), repeat=length):
            denom = brute_compose_joint(prefix, T, F, D, w, src.filler) if length else 1.0
            if denom == 0:
                continue
            expected = [brute_compose_joint(prefix + (v,), T, F, D, w, src.filler) / denom for v in range(V)]
            np.testing.assert_allclose(src.next_dist(src.parse(list(prefix))), expected, rtol=1e-12, atol=1e-15)
            checked += 1
    assert checked > 100


def test_compose_task_oracle_and_close_tokens():
    params = {"n_types": 2, "n_fillers": 3, "max_depth": 3, "length": 30}
    task = build_task("compose", 4, 4, 80, params)
    T, D, w = 2, 3, (0.3, 0.3, 0.4)
    filler = np.array(task.meta["filler"])
    for i, prefix in enumerate(task.prefixes):
        denom = brute_compose_joint(prefix, T, 3, D, w, filler)
        assert denom > 0
        expected = [brute_compose_joint(prefix + [v], T, 3, D, w, filler) / denom for v in range(task.vocab_size)]
        np.testing.assert_allclose(task.oracle[i], expected, rtol=1e-10)
        stack = []
        for tok in prefix:
            if tok < T:
                stack.append(tok)
            elif tok < 2 * T:
                stack.pop()
        assert task.extras["close_token"][i] == (T + stack[-1] if stack else -1)
    assert (task.extras["close_token"] >= 0).any() and (task.extras["close_token"] < 0).any()


def test_compose_sampler_matches_oracle_joint():
    T, F, D, w = 2, 2, 2, (0.3, 0.3, 0.4)
    src = compose.make_source(3, T, F, D, *w, 0.5)
    n = 40000
    xs = src.sample(n, 3, np.random.default_rng(2))
    counts = Counter(map(tuple, xs.tolist()))
    for x in itertools.product(range(2 * T + F), repeat=3):
        p = brute_compose_joint(x, T, F, D, w, src.filler)
        assert abs(counts[x] / n - p) < 5 * np.sqrt(p * (1 - p) / n) + 1e-4


# ---------------------------------------------------------------- recall

def test_recall_answers_by_parsing():
    params = {"k_values": [2, 5], "n_keys": 8, "n_values": 4, "n_queries": 3}
    task = build_task("recall", 1, 50, 40, params)
    n_keys = 8
    for prefix, answer, group in zip(task.prefixes, task.oracle, task.group):
        sep = prefix.index(recall.SEP)
        pairs = prefix[:sep]
        keys = pairs[0::2]
        assert len(keys) == task.group_labels[group] and len(set(keys)) == len(keys)
        assert all(1 <= k <= n_keys for k in keys) and all(v > n_keys for v in pairs[1::2])
        table = dict(zip(keys, pairs[1::2]))
        assert 1 <= prefix[-1] <= n_keys and (len(prefix) - sep - 1) % 2 == 1
        assert answer == table[prefix[-1]]
    for seq in task.train:
        sep = seq.index(recall.SEP)
        table = dict(zip(seq[:sep:2], seq[1:sep:2]))
        assert all(table[q] == a for q, a in zip(seq[sep + 1::2], seq[sep + 2::2]))


# ---------------------------------------------------------------- fewshot

def consistent_rules(pairs, p):
    return [(a, b) for a in range(1, p) for b in range(p) if all((a * x + b) % p == y for x, y in pairs)]


def test_fewshot_rules_are_unique_disjoint_and_answers_exact():
    p = 7
    task = build_task("fewshot", 3, 60, 40, {"modulus": p, "shots": [2, 3], "n_queries": 2, "test_fraction": 0.3})
    train_rules = {tuple(r) for r in task.meta["train_rules"]}
    test_rules = {tuple(r) for r in task.meta["test_rules"]}
    assert not train_rules & test_rules and len(train_rules | test_rules) == p * (p - 1)
    for prefix, answer, group in zip(task.prefixes, task.oracle, task.group):
        body, query = prefix[:-2], prefix[-2:]
        assert query[1] == p and len(body) % 4 == 0
        pairs = [(body[i], body[i + 2]) for i in range(0, len(body), 4)]
        assert len(pairs) == task.group_labels[group]
        assert query[0] not in [x for x, _ in pairs]
        rules = consistent_rules(pairs, p)
        assert len(rules) == 1 and rules[0] in test_rules
        a, b = rules[0]
        assert answer == (a * query[0] + b) % p
    for seq in task.train:
        pairs = [(seq[i], seq[i + 2]) for i in range(0, len(seq), 4)]
        rules = consistent_rules(pairs, p)
        assert len(rules) == 1 and rules[0] in train_rules


# ---------------------------------------------------------------- all tasks

@pytest.mark.parametrize("name", ["hmm", "vom", "recall", "compose", "fewshot"])
def test_tasks_are_deterministic_and_seed_sensitive(name):
    a = build_task(name, 0, 8, 16, {})
    b = build_task(name, 0, 8, 16, {})
    c = build_task(name, 1, 8, 16, {})
    assert a.train == b.train and a.prefixes == b.prefixes
    np.testing.assert_array_equal(a.oracle, b.oracle)
    assert a.train != c.train
