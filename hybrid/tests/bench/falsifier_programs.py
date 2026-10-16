"""Falsifier task programs used by the bench and research tests (the role a falsifier designer plays)."""
from __future__ import annotations

# Iterated shift rule, a small form of card-537e161c capability 1: demonstrations "a a+s ;" then "@m q" with q
# undemonstrated, target q + m*s mod NC. Answer oracle, groups m=1 and m=2.
SHIFT = '''
import numpy as np

NC, SEP, BOS, MARK = 16, 16, 17, {1: 18, 2: 19}
V = 20


def episode(r, m):
    s = int(r.integers(1, NC))
    demos = [int(a) for a in r.choice(NC, size=6, replace=False)]
    seq = [BOS]
    for a in demos:
        seq += [a, (a + s) % NC, SEP]
    rest = [q for q in range(NC) if q not in demos]
    q = int(r.choice(rest))
    return seq + [MARK[m], q], (q + m * s) % NC


def generate(seed):
    r = np.random.default_rng(seed)
    train = []
    for _ in range(64):
        seq, target = episode(r, int(r.integers(1, 3)))
        train.append(seq + [target])
    prefixes, nxt, group = [], [], []
    for g, m in enumerate((1, 2)):
        for _ in range(32):
            seq, target = episode(r, m)
            prefixes.append(seq)
            nxt.append(target)
            group.append(g)
    return {"vocab_size": V, "train": train, "prefixes": prefixes, "next_tokens": nxt, "oracle_kind": "answer",
            "oracle": list(nxt), "group": group, "group_labels": ["m=1", "m=2"]}
'''

# A two-state Markov chain with an exact next-token distribution oracle (dist kind, one group).
MARKOV = '''
import numpy as np

P = np.array([[0.9, 0.1], [0.2, 0.8]])


def sample(r, length):
    seq = [int(r.integers(0, 2))]
    while len(seq) < length:
        seq.append(int(r.random() < P[seq[-1], 1]))
    return seq


def generate(seed):
    r = np.random.default_rng(seed)
    train = [sample(r, 24) for _ in range(48)]
    prefixes, nxt, oracle = [], [], []
    for _ in range(64):
        seq = sample(r, 13)
        prefixes.append(seq[:12])
        nxt.append(seq[12])
        oracle.append(P[seq[11]].tolist())
    return {"vocab_size": 2, "train": train, "prefixes": prefixes, "next_tokens": nxt, "oracle_kind": "dist",
            "oracle": oracle, "group": [0] * 64, "group_labels": ["all"]}
'''

SHIFT_RULE = {"claims": ["capability.1"], "quote": "Accuracy >= 0.9 at m = 2",
              "conditions": [{"entrant": "candidate", "minus": None, "metric": "acc", "group": "m=2", "op": ">=",
                              "value": 0.9}], "if_met": "supported"}
MARKOV_RULE = {"claims": ["capability.2"], "quote": "within 0.01 nats of the n-gram",
               "conditions": [{"entrant": "candidate", "minus": "ngram", "metric": "kl", "group": None, "op": "<=",
                               "value": 0.01}], "if_met": "supported"}


def design(task_id: str, program: str, rules: list[dict]) -> dict:
    """A task design as the research stage records it (task.json before validation fields are added)."""
    return {"task_id": task_id, "title": "fixture " + task_id, "protocol": "fixture protocol",
            "deviations": "none", "rules": rules, "program": program, "protocol_params": []}


def variant(program: str, old: str, new: str) -> str:
    assert old in program, old
    return program.replace(old, new)
