"""Minimal demo: DP synthetic data from a small synthetic sample.

Run:  python examples/demo.py
Prints MMD of the DP synthetic sample vs a ground-truth reference.
This is a teaching demo on synthetic data, not a privacy audit tool:
do not use it on real personal data without a full DP review.
"""

import math
import sys

import numpy as np

import os

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpmoment import (  # noqa: E402
    empirical_moments,
    feature_map,
    fit_measure,
    gaussian_sigma,
    make_rff_params,
    mmd_rbf,
    sample_from_measure,
)
from experiment import sample_ground_truth  # noqa: E402

rng = np.random.default_rng(0)
rff = make_rff_params(3, n_rff=32, seed=0)
support = np.random.default_rng(12345).uniform(-1, 1, size=(1500, 3))
Phi_support = feature_map(support, rff)

private = sample_ground_truth(2000, np.random.default_rng(42))
mu = empirical_moments(private, rff)
sens = 2 * math.sqrt(len(mu)) / len(private)
sigma = gaussian_sigma(1.0, 1e-5, sens)
mu_priv = mu + rng.normal(0, sigma, size=mu.shape)
w = fit_measure(Phi_support, mu_priv)
syn = sample_from_measure(support, w, 2000, rng)
ref = sample_ground_truth(4000, np.random.default_rng(999999))
print(f"eps=1.0, delta=1e-5, n=2000 -> DP synthetic MMD vs truth: "
      f"{mmd_rbf(syn, ref):.4f}")
