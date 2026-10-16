"""Diagnostic for the probe 8 drift: which numpy operations differ between platforms?

Run on the host and inside the executor; compare the per-operation digests."""
import hashlib
import json
import sys

import numpy as np

rng = np.random.default_rng(12345)
# Generator.random is identical across platforms; scale with separately rounded ufuncs so inputs match bit for bit.
V = 0.01 + 0.99 * rng.random((4000, 2))
M = 0.01 + 0.99 * rng.random((4000, 2, 2))
S = 0.01 + 0.99 * rng.random((4000, 2, 2))
INPUT_DIGEST = {"V": V, "M": M, "S": S}


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()[:16]


ops = {
    "vec_at_mat": np.array([v @ m for v, m in zip(V, M)]),
    "mat_at_vec": np.array([m @ v for v, m in zip(V, M)]),
    "mat_at_mat": np.array([m @ s for m, s in zip(M, S)]),
    "sum_axis1_keepdims": M.sum(axis=-1, keepdims=True),
    "sum_vec": V.sum(axis=-1),
    "divide": M / M.sum(axis=-1, keepdims=True),
    "outer_elementwise": np.array([v[:, None] * m for v, m in zip(V, M)]),
    "exp": np.exp(M), "log": np.log(M), "tanh": np.tanh(M),
    "sigmoid": 1 / (1 + np.exp(-M)),
    "X_T_at_vec": (M.reshape(8000, 2).T @ V.reshape(8000)[:8000] if False else M.reshape(-1, 2).T @ V.reshape(-1)[: M.reshape(-1, 2).shape[0]]),
    "scalar_dot_2": np.array([np.dot(v, v) for v in V]),
    "mean_sum_log": np.array([np.mean(np.sum(m * np.log(m / s), axis=1)) for m, s in zip(M, S)]),
}
report = {name: digest(value) for name, value in ops.items()}
report.update({"input_" + k: digest(v) for k, v in INPUT_DIGEST.items()})
report["numpy"] = np.__version__
cfg = np.show_config(mode="dicts")
report["blas"] = {"name": cfg["Build Dependencies"]["blas"].get("name"), "version": cfg["Build Dependencies"]["blas"].get("version")}
report["simd_baseline"] = list(cfg["SIMD Extensions"]["baseline"])
report["simd_found"] = list(cfg["SIMD Extensions"]["found"])
print(json.dumps(report, indent=1, sort_keys=True))
