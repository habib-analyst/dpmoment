"""Determinism check: same (seed, n, eps) run twice must be bit-identical
on all reported metrics. Exits non-zero on any difference."""

import sys

import numpy as np

sys.path.insert(0, "src")

from dpmoment import feature_map, make_rff_params  # noqa: E402
from experiment import run_one, sample_ground_truth  # noqa: E402

rff = make_rff_params(3, n_rff=32, seed=0)
support = np.random.default_rng(12345).uniform(-1, 1, size=(1500, 3))
Phi = feature_map(support, rff)
ref = sample_ground_truth(4000, np.random.default_rng(999999))

a = run_one(1000 + 0 * 17 + 2000, 2000, 1.0, rff, support, Phi, ref)
b = run_one(1000 + 0 * 17 + 2000, 2000, 1.0, rff, support, Phi, ref)
keys = ["dpmoment_mmd", "dpmoment_sw1", "nonprivate_mmd", "nonprivate_sw1",
        "naive_mmd", "naive_sw1"]
diffs = {k: abs(a[k] - b[k]) for k in keys}
print(diffs)
assert all(v == 0.0 for v in diffs.values()), "NON-DETERMINISTIC"
print("DETERMINISM: PASS (identical on rerun)")
