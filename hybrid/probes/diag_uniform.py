"""Diagnostic: is the platform drift in numpy's Generator.uniform (low + (high-low) * random)?"""
import hashlib
import json

import numpy as np


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


raw = np.random.default_rng(20261007).random((10000,))
uniform = np.random.default_rng(20261007).uniform(0.2, 1, (10000,))
manual_no_fma = 0.2 + (1 - 0.2) * raw
fixture_init = np.random.default_rng(20261007).uniform(0.2, 1, (2, 2))
print(json.dumps({
    "raw_random_digest": digest(raw),
    "uniform_digest": digest(uniform),
    "uniform_equals_separately_rounded_formula": bool((uniform == manual_no_fma).all()),
    "n_uniform_differs_from_separately_rounded_formula": int((uniform != manual_no_fma).sum()),
    "fixture_hmm_init_hex": [float(x).hex() for x in fixture_init.ravel()],
}, indent=1, sort_keys=True))
