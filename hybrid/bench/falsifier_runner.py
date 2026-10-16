"""Candidate job for one card falsifier task. Standalone: standard library and NumPy only, no `hybrid` imports.

    python3 falsifier_runner.py <in_dir> <out_dir>

<in_dir> holds spec.json, candidate/ (candidate.py defining class Candidate) and data/{train,prefixes}.json.
No oracle data is ever placed here. The runner builds a fresh Candidate, fits it on every training sequence,
predicts the test prefixes and writes pred.npy, describe.json and runner.json.
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    spec = json.loads((in_dir / "spec.json").read_text())
    sys.path.insert(0, str(in_dir / "candidate"))
    Candidate = importlib.import_module("candidate").Candidate
    train = json.loads((in_dir / spec["train_file"]).read_text())["sequences"]
    prefixes = json.loads((in_dir / spec["prefix_file"]).read_text())["prefixes"]
    vocab = spec["vocab_size"]
    model = Candidate(vocab, spec["seed"], dict(spec["budget"]))
    t0 = time.perf_counter()
    model.fit([list(seq) for seq in train])
    t1 = time.perf_counter()
    probs = np.asarray(model.predict_proba([list(p) for p in prefixes]), dtype=np.float64)
    t2 = time.perf_counter()
    if probs.shape != (len(prefixes), vocab):
        raise ValueError("predict_proba shape %s, expected %s" % (probs.shape, (len(prefixes), vocab)))
    np.save(out_dir / "pred.npy", probs, allow_pickle=False)
    (out_dir / "describe.json").write_text(json.dumps(model.describe(), sort_keys=True))
    (out_dir / "runner.json").write_text(json.dumps({"task_id": spec["task_id"], "fit_s": t1 - t0,
                                                     "predict_s": t2 - t1, "python": sys.version,
                                                     "platform": platform.platform()}, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: falsifier_runner.py <in_dir> <out_dir>")
    main(sys.argv[1], sys.argv[2])
