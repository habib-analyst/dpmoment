"""Evaluation metrics: MMD (RBF kernel) and sliced-Wasserstein-1.

Both compare a synthetic sample against a large reference sample from
the known ground-truth distribution (never against the private sample
itself, so the metric is not contaminated by finite-private-sample noise
beyond what the estimator already carries).
"""

import numpy as np


def _rbf_kernel(A, B, gamma):
    A2 = np.sum(A**2, axis=1, keepdims=True)
    B2 = np.sum(B**2, axis=1, keepdims=True).T
    D2 = np.maximum(A2 + B2 - 2.0 * (A @ B.T), 0.0)
    return np.exp(-gamma * D2)


def mmd_rbf(X, Y, gamma=1.0):
    """Biased (V-statistic) MMD; returns the MMD value (sqrt of MMD^2)."""
    Kxx = _rbf_kernel(X, X, gamma).mean()
    Kyy = _rbf_kernel(Y, Y, gamma).mean()
    Kxy = _rbf_kernel(X, Y, gamma).mean()
    return float(np.sqrt(max(Kxx + Kyy - 2.0 * Kxy, 0.0)))


def _w1_1d(a, b):
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) != len(b):
        # Quantile interpolation to a common grid.
        grid = np.linspace(0.0, 1.0, max(len(a), len(b)))
        qa = np.quantile(a, grid)
        qb = np.quantile(b, grid)
        return float(np.mean(np.abs(qa - qb)))
    return float(np.mean(np.abs(a - b)))


def sliced_wasserstein1(X, Y, n_projections=64, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    dirs = rng.standard_normal((n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    vals = []
    for v in dirs:
        vals.append(_w1_1d(X @ v, Y @ v))
    return float(np.mean(vals))
