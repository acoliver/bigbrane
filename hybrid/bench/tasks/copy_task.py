"""Copy task from "Repeat After Me" (Jelassi, Brandfonbrener, Kakade, Malach, ICML 2024, arXiv 2402.01032).

Self-contained (NumPy only) so the ladder runner can import it inside the executor job. The file is named
copy_task.py, not copy.py, so it never shadows the standard library `copy` module when its directory is put
on sys.path inside the job.

Paper definition (section 2, "The copy task", and section 3.1): strings "<BOS> x1 ... xL <COPY>" where each
x is drawn uniformly from an alphabet of D tokens; the model must then output x1 ... xL. Training strings have
a random length up to L_train (the length is drawn first, then each position independently), and the
reported metric is string-level accuracy: the first L generated tokens equal x.

This bench's choices (IMPROVEMENT-PLAN.md B.3): D = 16 content tokens, L_train = 32, test L in
{32, 64, 128, 256}. The paper also packs several samples into one context; this bench keeps one sample per
sequence so a test sequence has exactly one string. An <EOS> token ends the sequence; it is not scored.

Tokens: content 0 .. D-1, BOS = D, COPY = D + 1, EOS = D + 2, PAD = D + 3 (right padding only).
Sequence: BOS x1 .. xL COPY x1 .. xL EOS, length 2L + 3. Scored positions: the COPY token and the first
L - 1 copied tokens, whose targets are x1 .. xL (teacher forcing). Greedy decoding reproduces the string
exactly when every one of those L argmax predictions is correct, so string accuracy is "all L correct".

Oracle: `oracle(prefix, d)` reads the answer from the prefix alone: it finds COPY and returns the source token
at the copy offset.
"""
from __future__ import annotations

import numpy as np

IGNORE = -100
NAME = "copy"


def special(d):
    return {"bos": d, "copy": d + 1, "eos": d + 2, "pad": d + 3}


def vocab_size(alphabet=16, **_):
    return int(alphabet) + 4


def make_row(rng, length, alphabet):
    s = special(alphabet)
    x = rng.integers(0, alphabet, size=length)
    tokens = np.concatenate([[s["bos"]], x, [s["copy"]], x, [s["eos"]]]).astype(np.int64)
    targets = np.full(len(tokens), IGNORE, dtype=np.int64)
    targets[length + 1:2 * length + 1] = x
    return tokens, targets


def generate(rng, n, lengths, alphabet=16):
    """n copy sequences; each row's L is drawn uniformly from `lengths` (training uses 1 .. L_train, a test
    group uses one length). Rows are right-padded with PAD to the longest row.

    Returns tokens (n, T), targets (n, T) (IGNORE where unscored), length (n,)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    ls = lengths[rng.integers(0, len(lengths), size=n)]
    width = 2 * int(ls.max()) + 3
    pad = special(alphabet)["pad"]
    tokens = np.full((n, width), pad, dtype=np.int64)
    targets = np.full((n, width), IGNORE, dtype=np.int64)
    for row in range(n):
        t, y = make_row(rng, int(ls[row]), alphabet)
        tokens[row, :len(t)] = t
        targets[row, :len(y)] = y
    return {"tokens": tokens, "targets": targets, "length": ls}


def oracle(prefix, alphabet):
    """The next copied token after `prefix` (which must end at COPY or inside the copy), from the prefix alone."""
    prefix = list(prefix)
    s = special(alphabet)
    if prefix[0] != s["bos"] or s["copy"] not in prefix:
        raise ValueError("copy oracle: prefix has no BOS ... COPY structure")
    c = prefix.index(s["copy"])
    j = len(prefix) - 1 - c
    if j >= c - 1:
        raise ValueError("copy oracle: the whole string has already been copied")
    return int(prefix[1 + j])


def bits_needed(length, alphabet):
    """L log2 D: the information in a uniform string, the quantity in Theorem 2.7 and Corollary 2.8."""
    return float(length) * float(np.log2(alphabet))
