"""Few-shot rule induction: affine maps y = (a*x + b) mod p shown in context.

Each sequence draws a fresh rule (a, b) and shows examples "x -> y ;". The rule set is split
once into disjoint train and test rules, so a model that memorizes training rules has no
consistent rule for a test sequence: with two or more distinct shown inputs an affine map mod
a prime is determined uniquely, and that map is a test rule. A test prefix shows `shots`
examples and then asks for a query input that was not among them, so lookup cannot answer it.

Tokens: numbers 0..p-1, ARROW = p, SEMI = p + 1.
"""
from __future__ import annotations

import numpy as np

from .common import ANSWER, PARAMS, TEST, TRAIN, TaskData, stream


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, int(n ** .5) + 1))


def split_rules(seed, modulus, test_fraction):
    rules = [(a, b) for a in range(1, modulus) for b in range(modulus)]
    order = stream(seed, "fewshot", PARAMS).permutation(len(rules))
    n_test = int(round(test_fraction * len(rules)))
    test = [rules[i] for i in order[:n_test]]
    train = [rules[i] for i in order[n_test:]]
    return train, test


def example(x, rule, modulus):
    a, b = rule
    return [x, modulus, (a * x + b) % modulus, modulus + 1]


def build(seed, train_pool, n_test, modulus=13, shots=(2, 4, 8), n_queries=4, test_fraction=0.3):
    shots = tuple(int(s) for s in shots)
    if not is_prime(modulus):
        raise ValueError("modulus must be prime so two examples determine the rule")
    if min(shots) < 2 or max(shots) + n_queries > modulus:
        raise ValueError("need 2 <= shots and shots + n_queries <= modulus")
    train_rules, test_rules = split_rules(seed, modulus, test_fraction)
    r = stream(seed, "fewshot", TRAIN)
    train = []
    for _ in range(train_pool):
        rule = train_rules[int(r.integers(len(train_rules)))]
        n = int(r.choice(shots)) + n_queries
        train.append([tok for x in r.permutation(modulus)[:n] for tok in example(int(x), rule, modulus)])
    r = stream(seed, "fewshot", TEST)
    prefixes, answers, groups, rule_ids = [], [], [], []
    for group, n in enumerate(shots):
        for _ in range(n_test):
            index = int(r.integers(len(test_rules)))
            rule = test_rules[index]
            xs = [int(x) for x in r.permutation(modulus)[:n + 1]]
            seq = [tok for x in xs[:n] for tok in example(x, rule, modulus)]
            query = example(xs[n], rule, modulus)
            prefixes.append(seq + query[:2])
            answers.append(query[2])
            groups.append(group)
            rule_ids.append(index)
    return TaskData(
        name="fewshot", vocab_size=modulus + 2, train=train, prefixes=prefixes, oracle_kind=ANSWER,
        oracle=np.array(answers, dtype=np.int64), group=np.array(groups, dtype=np.int64),
        group_name="shots", group_labels=list(shots), extras={"test_rule": np.array(rule_ids, dtype=np.int64)},
        meta={"modulus": modulus, "n_queries": n_queries, "test_fraction": test_fraction,
              "train_rules": [list(x) for x in train_rules], "test_rules": [list(x) for x in test_rules],
              "n_test_per_group": n_test})
