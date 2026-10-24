"""Generalized moments: a fixed, bounded feature map phi(x) in [-1, 1]^m.

Components (d = input dimension):
  * constant 1
  * first-order terms x_i
  * second-order terms x_i * x_j for i <= j
  * random Fourier features cos(w_k . x + b_k), k = 1..n_rff, with w
    drawn once (fixed seed) from N(0, I) and b uniform on [0, 2*pi).

Every coordinate of phi lies in [-1, 1], so ||phi(x)||_2 <= sqrt(m) and
the L2 sensitivity of the empirical mean over n records is 2*sqrt(m)/n.
That single bound is what the Gaussian mechanism in privacy.py uses.
"""

import numpy as np


def make_rff_params(d, n_rff=32, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_rff, d))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_rff)
    return W, b


def feature_map(X, rff_params):
    """Return phi(X), shape (n, m). X must already lie in [-1, 1]^d."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = X.shape
    cols = [np.ones(n)]
    cols.append(X)  # first order, (n, d)
    # second order i <= j
    second = []
    for i in range(d):
        for j in range(i, d):
            second.append(X[:, i] * X[:, j])
    cols.append(np.column_stack(second))
    W, b = rff_params
    cols.append(np.cos(X @ W.T + b))
    return np.column_stack(cols)


def empirical_moments(X, rff_params):
    return feature_map(X, rff_params).mean(axis=0)
