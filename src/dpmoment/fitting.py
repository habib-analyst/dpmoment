"""Measure fitting: project the (private) moment vector back onto measures.

Candidate support: a fixed pool of points in [-1, 1]^d, drawn once from a
uniform distribution (independent of the private data). We solve

    min_{w in simplex} || Phi_support^T w - mu_target ||_2^2

by projected gradient descent with an exact simplex projection
(Duchi et al.). The fitted measure is discrete on the support; sampling
is just categorical sampling of support points. No private record is
ever copied into the support: the only private-data contact is mu_target.
"""

import numpy as np


def _project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > css)[0][-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def fit_measure(Phi_support, mu_target, n_iter=300, lr=None):
    """Phi_support: (n_support, m); mu_target: (m,). Returns weights (n_support,)."""
    Phi = np.asarray(Phi_support, dtype=float)
    mu = np.asarray(mu_target, dtype=float)
    n_support = Phi.shape[0]
    if lr is None:
        # Lipschitz constant of the quadratic objective: 2 * ||Phi^T Phi||_2.
        L = 2.0 * float(np.linalg.norm(Phi.T @ Phi, ord=2))
        lr = 1.0 / L if L > 0 else 1.0
    w = np.full(n_support, 1.0 / n_support)
    for _ in range(n_iter):
        resid = Phi.T @ w - mu  # (m,)
        grad = 2.0 * (Phi @ resid)  # (n_support,)
        w = _project_simplex(w - lr * grad)
    return w


def sample_from_measure(support, weights, n_samples, rng):
    idx = rng.choice(len(support), size=n_samples, p=weights)
    return support[idx].copy()
