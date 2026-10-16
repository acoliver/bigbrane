"""n-gram lookup from "Repeat After Me" (Jelassi et al., ICML 2024, arXiv 2402.01032, section 3.5).

Self-contained (NumPy only) so the ladder runner can import it inside the executor job.

Paper definition: the model gets a sequence of L input tokens and an n-gram taken from it, and must output the
k tokens that followed that n-gram in the sequence. Two variants:

- suffix key: the sequence, a separator, then the n-gram. A model must have kept the whole sequence to answer,
  so fixed-state models fail beyond their training lengths (paper Figure 5).
- prefix key: the n-gram first, then the sequence. A fixed-state model can store the key and watch for it, so
  fixed-state models generalize here (paper Figure 6). The bench uses it as the positive control for
  fixed-state models.

Training used strings of at most 30 tokens; the paper tested up to 100. The paper does not state n and k in
the text; this bench uses n = 3 and k = 3 by default, with D = 16 content tokens, and resamples the sequence
until the chosen n-gram occurs exactly once, so the answer is unique.

Tokens: content 0 .. D-1, BOS = D, SEP = D + 1, PAD = D + 2.
- suffix: BOS x1 .. xL SEP g1 .. gn a1 .. ak
- prefix: BOS g1 .. gn SEP x1 .. xL SEP a1 .. ak
where g = x[i : i + n] and a = x[i + n : i + n + k]. Scored positions: the token before a1 (gn for suffix,
the second SEP for prefix) and a1 .. a(k-1), with targets a1 .. ak (teacher forcing).

Oracle: `oracle(prefix, variant, n, alphabet)` parses the prefix, finds the unique occurrence of the key and
returns the next answer token.
"""
from __future__ import annotations

import numpy as np

IGNORE = -100
NAME = "ngram_lookup"
VARIANTS = ("suffix", "prefix")


def special(d):
    return {"bos": d, "sep": d + 1, "pad": d + 2}


def vocab_size(alphabet=16, **_):
    return int(alphabet) + 3


def occurrences(x, gram):
    n = len(gram)
    return [i for i in range(len(x) - n + 1) if list(x[i:i + n]) == list(gram)]


def make_row(rng, length, variant, n, k, alphabet, max_tries=1000):
    if length < n + k:
        raise ValueError("ngram_lookup: length %d is below n + k = %d" % (length, n + k))
    s = special(alphabet)
    for _ in range(max_tries):
        x = rng.integers(0, alphabet, size=length)
        i = int(rng.integers(0, length - n - k + 1))
        gram = x[i:i + n]
        if len(occurrences(x, gram)) == 1:
            break
    else:
        raise RuntimeError("ngram_lookup: no unique %d-gram after %d tries (L %d, D %d)"
                           % (n, max_tries, length, alphabet))
    answer = x[i + n:i + n + k]
    if variant == "suffix":
        head = np.concatenate([[s["bos"]], x, [s["sep"]], gram])
    elif variant == "prefix":
        head = np.concatenate([[s["bos"]], gram, [s["sep"]], x, [s["sep"]]])
    else:
        raise ValueError("ngram_lookup: unknown variant %r" % (variant,))
    tokens = np.concatenate([head, answer]).astype(np.int64)
    targets = np.full(len(tokens), IGNORE, dtype=np.int64)
    first = len(head) - 1
    targets[first:first + k] = answer
    return tokens, targets


def generate(rng, n_rows, lengths, variant, n=3, k=3, alphabet=16):
    """n_rows sequences of one variant; each row's L is drawn uniformly from `lengths`; right-padded with PAD.

    Returns tokens (rows, T), targets (rows, T) (IGNORE where unscored), length (rows,)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    ls = lengths[rng.integers(0, len(lengths), size=n_rows)]
    rows = [make_row(rng, int(length), variant, n, k, alphabet) for length in ls]
    width = max(len(t) for t, _ in rows)
    tokens = np.full((n_rows, width), special(alphabet)["pad"], dtype=np.int64)
    targets = np.full((n_rows, width), IGNORE, dtype=np.int64)
    for row, (t, y) in enumerate(rows):
        tokens[row, :len(t)] = t
        targets[row, :len(y)] = y
    return {"tokens": tokens, "targets": targets, "length": ls}


def oracle(prefix, variant, n, alphabet):
    """The next answer token after `prefix` (which must end at the last key token or inside the answer)."""
    prefix = list(prefix)
    s = special(alphabet)
    if prefix[0] != s["bos"]:
        raise ValueError("ngram_lookup oracle: prefix does not start with BOS")
    if variant == "suffix":
        sep = prefix.index(s["sep"])
        x, gram, answered = prefix[1:sep], prefix[sep + 1:sep + 1 + n], prefix[sep + 1 + n:]
    elif variant == "prefix":
        gram = prefix[1:1 + n]
        if prefix[1 + n] != s["sep"]:
            raise ValueError("ngram_lookup oracle: no SEP after the prefix key")
        rest = prefix[2 + n:]
        sep = rest.index(s["sep"])
        x, answered = rest[:sep], rest[sep + 1:]
    else:
        raise ValueError("ngram_lookup oracle: unknown variant %r" % (variant,))
    if len(gram) != n:
        raise ValueError("ngram_lookup oracle: the key is incomplete")
    found = occurrences(x, gram)
    if len(found) != 1:
        raise ValueError("ngram_lookup oracle: key occurs %d times" % len(found))
    return int(x[found[0] + n + len(answered)])
