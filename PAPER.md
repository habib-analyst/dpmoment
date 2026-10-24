# Differentially Private Synthetic Data via Generalized Moment Matching — empirical verification

**Project #80 · Replication of arXiv 2610.30196 (Ding, He, Zhu) · 2026-10-24**

## Claim under test

The paper introduces generalized moment matching for (ε, δ)-DP
synthetic data: privately estimate moments of a database in [-1, 1]^d,
fit a probability measure to them, sample synthetic records — with
rates minimax-optimal up to log factors (fixed dimension) for MMD and
sliced-Wasserstein-1. This project implements that estimator family
and verifies its rates empirically. It does not reproduce the paper's
theorems and uses no author code (none was found).

## Method

- Ground truth: 3-component Gaussian mixture in [-1, 1]^3, weights
  0.40/0.35/0.25, component scale 0.08, clipped. Known by construction,
  so MMD / sliced-W₁ against a 4000-point reference are exact.
- Moments: 42 bounded features (constant + first-order + second-order
  + 32 random Fourier features); L2 sensitivity 2·√42/n.
- Privacy: Gaussian mechanism, δ = 1e-5, single draw privatizing all
  moments; accountant re-checked on every run.
- Measure fit: projected gradient descent onto the simplex over a
  fixed 1500-point uniform support (independent of private data);
  2000 synthetic records sampled per run.
- Baselines: non-private moment matching (utility upper bound); naive
  DP independent-Gaussian fit (private mean only, fixed scale 0.35).
- Protocol: pilot (n = 2000, ε ∈ {0.5, 1, 2, 4}, 5 reps) → PREREG locked
  → authenticated author-code recheck (G1) → full run (n ∈ {500, 2000,
  8000} × ε ∈ {1, 4}, 10 reps, 60 runs) → 17-test pytest suite →
  determinism check. Bars never moved after the pilot.

## Results (full run, means; 95% paired bootstrap CIs)

| n | ε | DP MMD | naive MMD | non-priv MMD | MMD advantage [CI] | SW₁ advantage [CI] | utility gap |
|---|---|--------|-----------|--------------|--------------------|--------------------|-------------|
| 500 | 1.0 | 0.123 | 0.364 | 0.066 | 0.241 [0.229, 0.253] | 0.107 [0.099, 0.116] | 0.057 |
| 500 | 4.0 | 0.071 | 0.362 | 0.066 | 0.291 [0.284, 0.299] | 0.135 [0.126, 0.143] | 0.005 |
| 2000 | 1.0 | 0.071 | 0.360 | 0.058 | 0.289 [0.284, 0.295] | 0.138 [0.129, 0.148] | 0.013 |
| 2000 | 4.0 | 0.060 | 0.360 | 0.058 | 0.300 [0.297, 0.304] | 0.143 [0.135, 0.153] | 0.002 |
| 8000 | 1.0 | 0.058 | 0.362 | 0.060 | 0.304 [0.299, 0.309] | 0.152 [0.146, 0.160] | −0.002 |
| 8000 | 4.0 | 0.058 | 0.362 | 0.060 | 0.304 [0.299, 0.308] | 0.152 [0.146, 0.160] | −0.001 |

- Fitted log-log error exponents vs n: −0.271 (ε = 1.0), −0.073
  (ε = 4.0). Both negative (error decays with n), neither forced to
  theory: the ε = 4.0 exponent is shallow because the error is already
  at the non-private floor (~0.058) set by the fixed support.
- Privacy: accountant OK on all 60 full runs (max implied ε = budget);
  over-budget probe (σ halved → implied ε = 2× budget) caught.
- Competence pre-check: non-private MMD 0.058 at n = 2000, beating
  naive by 0.302 — the task is non-vacuous.
- Determinism: same-seed rerun bit-identical on all six metrics.
- Tests: 17/17 passed. Compute: pilot ~95 s + full ~271 s CPU, $0.

## Gate verdicts (all against PREREG-locked bars)

- G0 staging / competence: PASS.
- G1 author-code recheck (authenticated GitHub): PASS — no author
  code; abs page has no code links; repo searches for the arXiv ID and
  `dpmoment` return 0; the 31 code-search hits are aggregator mirrors
  that only list the paper.
- G2 empirical claim: PASS at both ε = 1.0 and ε = 4.0 (advantages far
  above the locked bars; exponents negative).
- G3 privacy + utility: PASS (accountant clean; gaps 0.013 / 0.002
  vs locked bounds 0.10 / 0.05).

## Limitations (stated, not buried)

Single 3-D synthetic mixture; one fixed feature map (not the paper's
kernel families); fixed support sets the ~0.058 MMD floor; the naive
baseline is deliberately weak (unimodal), so the win demonstrates
multimodality capture under privacy, not supremacy over all DP
synthesizers. No real data, no production privacy review — not for use
on personal data as-is.

## Reproduce

Repo payload: `~/workspace/build/dpmoment/repo_staging/` (to become
`habib-analyst/dpmoment` after browser repo creation). Run
`experiment.py --mode full`, `summarize.py`, `pytest`, and
`check_determinism.py` as documented in the README.
