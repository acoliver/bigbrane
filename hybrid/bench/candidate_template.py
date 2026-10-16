"""Reference candidate. Copy this file to <candidate_dir>/candidate.py and replace the model.

Contract (full text in hybrid/bench/api.py):
- class Candidate(vocab_size, seed, budget) with fit(train), predict_proba(prefixes), describe().
- Runs inside the executor image: Python 3.12, NumPy, CPU torch, one thread, no network.
  Do not read or write files; the bench passes everything through these methods.
- predict_proba returns a float array of shape (len(prefixes), vocab_size). Row i is the
  next-token distribution after prefixes[i] and must depend only on prefixes[i] and the
  fitted parameters. Rows are non-negative and sum to 1.
- Respect budget["max_steps"], budget["max_epochs"], budget["batch_size"] and budget["lr"]
  if the model trains iteratively; seed every random generator from `seed`.
- describe() reports the parameter count and the per-sequence state size in floats that the
  model carries while reading a sequence (for a model whose state grows with context, give
  the value at 256 tokens and say so in notes).

This template is a deliberately simple bigram model with add-alpha smoothing that mixes in a
unigram. It is a working example of the API, not a baseline.
"""
from __future__ import annotations

import numpy as np


class Candidate:
    def __init__(self, vocab_size: int, seed: int, budget: dict):
        self.vocab_size = int(vocab_size)
        self.rng = np.random.default_rng(seed)  # unused here; real models seed everything from `seed`
        self.budget = dict(budget)
        self.alpha = 0.5
        self.bigram = np.zeros((self.vocab_size, self.vocab_size))
        self.unigram = np.zeros(self.vocab_size)

    def fit(self, train: list[list[int]]) -> None:
        for seq in train:
            tokens = np.asarray(seq, dtype=np.int64)
            np.add.at(self.unigram, tokens, 1.0)
            np.add.at(self.bigram, (tokens[:-1], tokens[1:]), 1.0)

    def predict_proba(self, prefixes: list[list[int]]) -> np.ndarray:
        unigram = (self.unigram + 1.0) / (self.unigram.sum() + self.vocab_size)
        last = np.array([p[-1] for p in prefixes], dtype=np.int64)
        counts = self.bigram[last] + self.alpha * self.vocab_size * unigram
        return counts / counts.sum(axis=1, keepdims=True)

    def describe(self) -> dict:
        return {"params": int(self.bigram.size + self.unigram.size),
                "state_floats_per_sequence": 1,
                "hyper": {"context": 1},          # integer hyperparameters by name (state width d, slots M, ...)
                "train_max_len": 2048,            # fit() uses every token of sequences up to this length
                "notes": "bigram with add-alpha unigram smoothing (template)"}
