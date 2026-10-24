"""Gated experiment for dpmoment.

Ground truth: 3-component Gaussian mixture in [-1, 1]^3, clipped.
Methods (all sample 2000 synthetic records):
  * nonprivate: moment matching on exact empirical moments (utility upper bound)
  * dpmoment:  moment matching on Gaussian-mechanism private moments
  * naive:     DP independent-Gaussian fit (privately estimated mean only,
               fixed public scale 0.35, unimodal) -- the naive DP baseline
Metrics vs a fixed 4000-point reference sample from ground truth:
  MMD (RBF, gamma=1.0) and sliced-Wasserstein-1 (64 projections).

Privacy: delta = 1e-5 throughout. dpmoment sensitivity = 2*sqrt(m)/n
(m = 42 moments); naive sensitivity = 2*sqrt(d)/n on the mean only.
Every run re-checks the accountant: implied epsilon from the sigma used
must be <= budget (else the run is marked invalid and excluded).
"""

import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from dpmoment import (  # noqa: E402
    empirical_moments,
    feature_map,
    fit_measure,
    gaussian_sigma,
    check_budget,
    make_rff_params,
    mmd_rbf,
    sample_from_measure,
    sliced_wasserstein1,
)

D = 3
DELTA = 1e-5
N_SYN = 2000
N_REF = 4000
N_SUPPORT = 1500
SCALE_NAIVE = 0.35

MEANS = np.array([[-0.6, -0.5, 0.3], [0.6, 0.4, -0.4], [0.0, 0.6, 0.6]])
WEIGHTS = np.array([0.40, 0.35, 0.25])
COV_SCALE = 0.08


def sample_ground_truth(n, rng):
    comp = rng.choice(len(WEIGHTS), size=n, p=WEIGHTS)
    X = MEANS[comp] + math.sqrt(COV_SCALE) * rng.standard_normal((n, D))
    return np.clip(X, -1.0, 1.0)


def paired_bootstrap_ci(diff, n_boot=2000, seed=7):
    """Percentile bootstrap CI for mean(diff); diff is per-replicate paired diffs."""
    diff = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_one(seed, n, eps, rff_params, support, Phi_support, reference):
    rng = np.random.default_rng(seed)
    private = sample_ground_truth(n, rng)
    mu_exact = empirical_moments(private, rff_params)
    m = len(mu_exact)
    sens = 2.0 * math.sqrt(m) / n
    sigma = gaussian_sigma(eps, DELTA, sens)
    implied, ok = check_budget(sigma, DELTA, sens, eps)
    assert ok, f"accountant failed: implied {implied} > budget {eps}"
    mu_priv = mu_exact + rng.normal(0.0, sigma, size=m)

    out = {"seed": seed, "n": n, "eps": eps, "sigma": sigma,
           "implied_eps": implied, "accountant_ok": bool(ok)}
    # dpmoment
    w = fit_measure(Phi_support, mu_priv)
    syn = sample_from_measure(support, w, N_SYN, rng)
    out["dpmoment_mmd"] = mmd_rbf(syn, reference)
    out["dpmoment_sw1"] = sliced_wasserstein1(syn, reference, seed=seed)
    # non-private upper bound
    w0 = fit_measure(Phi_support, mu_exact)
    syn0 = sample_from_measure(support, w0, N_SYN, rng)
    out["nonprivate_mmd"] = mmd_rbf(syn0, reference)
    out["nonprivate_sw1"] = sliced_wasserstein1(syn0, reference, seed=seed)
    # naive DP Gaussian (mean only, full budget)
    mu_mean = private.mean(axis=0)
    sens_mean = 2.0 * math.sqrt(D) / n
    sigma_mean = gaussian_sigma(eps, DELTA, sens_mean)
    implied_m, ok_m = check_budget(sigma_mean, DELTA, sens_mean, eps)
    assert ok_m
    mean_priv = mu_mean + rng.normal(0.0, sigma_mean, size=D)
    syn_n = np.clip(mean_priv + SCALE_NAIVE * rng.standard_normal((N_SYN, D)), -1.0, 1.0)
    out["naive_mmd"] = mmd_rbf(syn_n, reference)
    out["naive_sw1"] = sliced_wasserstein1(syn_n, reference, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rff_params = make_rff_params(D, n_rff=32, seed=0)
    rng_sup = np.random.default_rng(12345)
    support = rng_sup.uniform(-1.0, 1.0, size=(N_SUPPORT, D))
    Phi_support = feature_map(support, rff_params)
    reference = sample_ground_truth(N_REF, np.random.default_rng(999999))

    if args.mode == "pilot":
        configs = [(2000, eps, seed) for eps in (0.5, 1.0, 2.0, 4.0)
                   for seed in range(5)]
    else:
        configs = [(n, eps, seed) for n in (500, 2000, 8000)
                   for eps in (1.0, 4.0) for seed in range(10)]

    t0 = time.time()
    rows = []
    for n, eps, seed in configs:
        row = run_one(1000 + seed * 17 + n, n, eps, rff_params,
                      support, Phi_support, reference)
        rows.append(row)
        print(f"n={n} eps={eps} seed={seed} "
              f"dp_mmd={row['dpmoment_mmd']:.4f} naive_mmd={row['naive_mmd']:.4f} "
              f"np_mmd={row['nonprivate_mmd']:.4f}", flush=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
