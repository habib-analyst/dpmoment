"""pytest suite for dpmoment (target >= 12 tests)."""

import math
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from dpmoment import (  # noqa: E402
    check_budget,
    composed_epsilon,
    empirical_moments,
    feature_map,
    fit_measure,
    gaussian_sigma,
    make_rff_params,
    mmd_rbf,
    private_moments,
    sample_from_measure,
    sliced_wasserstein1,
)
from experiment import sample_ground_truth  # noqa: E402

RFF = make_rff_params(3, n_rff=32, seed=0)


def test_feature_map_shape_and_bounds():
    X = np.random.default_rng(0).uniform(-1, 1, size=(50, 3))
    Phi = feature_map(X, RFF)
    assert Phi.shape == (50, 42)
    assert np.all(np.abs(Phi) <= 1.0 + 1e-12)


def test_feature_map_constant_column():
    X = np.zeros((10, 3))
    assert np.all(feature_map(X, RFF)[:, 0] == 1.0)


def test_empirical_moments_shape():
    X = np.random.default_rng(1).uniform(-1, 1, size=(100, 3))
    assert empirical_moments(X, RFF).shape == (42,)


def test_gaussian_sigma_roundtrip():
    sens = 2 * math.sqrt(42) / 2000
    sigma = gaussian_sigma(1.0, 1e-5, sens)
    assert abs(composed_epsilon(sigma, 1e-5, sens) - 1.0) < 1e-9


def test_gaussian_sigma_decreases_with_epsilon():
    sens = 0.01
    assert gaussian_sigma(4.0, 1e-5, sens) < gaussian_sigma(1.0, 1e-5, sens)


def test_private_moments_shape_and_noise():
    rng = np.random.default_rng(2)
    mu = np.zeros(42)
    noisy, sigma = private_moments(mu, 1.0, 1e-5, 0.01, rng)
    assert noisy.shape == (42,)
    assert sigma > 0 and not np.allclose(noisy, mu)


def test_check_budget_pass_and_overbudget_probe():
    sens = 0.01
    sigma = gaussian_sigma(1.0, 1e-5, sens)
    implied, ok = check_budget(sigma, 1e-5, sens, 1.0)
    assert ok and abs(implied - 1.0) < 1e-9
    # Deliberate over-budget probe: halved sigma must be caught.
    implied2, ok2 = check_budget(sigma / 2, 1e-5, sens, 1.0)
    assert not ok2 and abs(implied2 - 2.0) < 1e-9


def test_invalid_privacy_params_raise():
    with pytest.raises(ValueError):
        gaussian_sigma(0.0, 1e-5, 0.01)
    with pytest.raises(ValueError):
        gaussian_sigma(1.0, 1.5, 0.01)


def test_fit_measure_weights_on_simplex():
    rng = np.random.default_rng(3)
    support = rng.uniform(-1, 1, size=(200, 3))
    Phi = feature_map(support, RFF)
    mu = Phi.mean(axis=0)
    w = fit_measure(Phi, mu, n_iter=50)
    assert abs(w.sum() - 1.0) < 1e-6
    assert np.all(w >= -1e-12)


def test_fit_measure_recovers_moments_better_than_uniform():
    rng = np.random.default_rng(4)
    support = rng.uniform(-1, 1, size=(300, 3))
    Phi = feature_map(support, RFF)
    target = Phi[:50].mean(axis=0)  # a realizable-ish target
    w = fit_measure(Phi, target, n_iter=300)
    err_fit = np.linalg.norm(Phi.T @ w - target)
    err_unif = np.linalg.norm(Phi.mean(axis=0) - target)
    assert err_fit < err_unif


def test_sample_from_measure_in_support():
    support = np.array([[0.1, 0.2, 0.3], [-0.5, 0.0, 0.5]])
    w = np.array([1.0, 0.0])
    out = sample_from_measure(support, w, 20, np.random.default_rng(5))
    assert np.allclose(out, support[0])


def test_mmd_identical_is_zero():
    X = np.random.default_rng(6).uniform(-1, 1, size=(100, 3))
    assert mmd_rbf(X, X) < 1e-9


def test_mmd_separated_is_positive():
    rng = np.random.default_rng(7)
    X = rng.normal(-0.8, 0.05, size=(200, 3))
    Y = rng.normal(0.8, 0.05, size=(200, 3))
    assert mmd_rbf(X, Y) > 0.3


def test_sliced_wasserstein_identical_is_zero():
    X = np.random.default_rng(8).uniform(-1, 1, size=(100, 3))
    assert sliced_wasserstein1(X, X, n_projections=8, seed=0) < 1e-9


def test_sliced_wasserstein_shifted_positive():
    X = np.zeros((100, 3))
    Y = np.ones((100, 3))
    assert sliced_wasserstein1(X, Y, n_projections=16, seed=0) > 0.3


def test_ground_truth_in_range():
    X = sample_ground_truth(1000, np.random.default_rng(9))
    assert X.shape == (1000, 3)
    assert np.all(X >= -1.0) and np.all(X <= 1.0)


def test_ground_truth_multimodal():
    # Three separated modes: sample mean alone must not describe it;
    # check the mixture actually spans both signs on dim 0.
    X = sample_ground_truth(5000, np.random.default_rng(10))
    assert (X[:, 0] < -0.3).mean() > 0.15
    assert (X[:, 0] > 0.3).mean() > 0.15
