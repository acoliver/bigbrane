"""Associative recall with a varying number of pairs K. Oracle: the exact value.

Sequence: k1 v1 k2 v2 ... kK vK SEP q1 a1 q2 a2 ... with distinct keys, values drawn with
replacement, and each query key drawn from the K stored keys. A test prefix ends right
after one query key, and the oracle answer is that key's value. Groups are the K values, so
a fixed-size state that cannot hold many pairs shows up as accuracy falling with K.
"""
from __future__ import annotations

import numpy as np

from .common import ANSWER, TEST, TRAIN, TaskData, stream

SEP = 0


def key_token(i):
    return 1 + i


def value_token(j, n_keys):
    return 1 + n_keys + j


def make_sequence(r, k, n_keys, n_values, n_queries):
    keys = r.choice(n_keys, size=k, replace=False)
    values = r.integers(0, n_values, size=k)
    seq = []
    for key, value in zip(keys, values):
        seq += [key_token(int(key)), value_token(int(value), n_keys)]
    seq.append(SEP)
    asked = r.integers(0, k, size=n_queries)
    for index in asked:
        seq += [key_token(int(keys[index])), value_token(int(values[index]), n_keys)]
    return seq


def build(seed, train_pool, n_test, k_values=(2, 8, 32), n_keys=32, n_values=16, n_queries=16):
    k_values = tuple(int(k) for k in k_values)
    if max(k_values) > n_keys:
        raise ValueError("every K must be at most n_keys")
    vocab = 1 + n_keys + n_values
    r = stream(seed, "recall", TRAIN)
    train = [make_sequence(r, int(r.choice(k_values)), n_keys, n_values, n_queries) for _ in range(train_pool)]
    r = stream(seed, "recall", TEST)
    prefixes, answers, groups = [], [], []
    for group, k in enumerate(k_values):
        for _ in range(n_test):
            seq = make_sequence(r, k, n_keys, n_values, n_queries)
            query = int(r.integers(0, n_queries))
            cut = 2 * k + 1 + 2 * query + 1   # prefix ends with the query key
            prefixes.append(seq[:cut])
            answers.append(seq[cut])
            groups.append(group)
    return TaskData(
        name="recall", vocab_size=vocab, train=train, prefixes=prefixes, oracle_kind=ANSWER,
        oracle=np.array(answers, dtype=np.int64), group=np.array(groups, dtype=np.int64),
        group_name="K", group_labels=list(k_values),
        meta={"n_keys": n_keys, "n_values": n_values, "n_queries": n_queries, "sep": SEP,
              "n_test_per_group": n_test})
