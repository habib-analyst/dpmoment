# PREREG — dpmoment (project #80), LOCKED before the full run

Paper: arXiv 2610.30196, "Differentially Private Synthetic Data via
Generalized Moment Matching" (Ding, He, Zhu). This is an
implementation-and-rate-verification, NOT author code and NOT a proof of
the paper's minimax theorems. Locked 2026-10-24, after the pilot below
and before any full-run result was seen. Bars do not move after the
full run, whatever it shows.

## Design (frozen)

- Ground truth: 3-component Gaussian mixture in [-1, 1]^3, weights
  0.40/0.35/0.25, component scale 0.08, clipped. Fixed in experiment.py.
- Feature map: 1 + 3 first-order + 6 second-order + 32 random Fourier
  features = 42 moments, all coordinates in [-1, 1].
- Privacy: delta = 1e-5; Gaussian mechanism; sensitivity 2*sqrt(42)/n.
- Measure fit: 1500-point fixed uniform support, projected gradient
  descent (300 iters) onto the simplex; sample 2000 synthetic records.
- Metrics vs a fixed 4000-point reference sample: MMD (RBF gamma=1.0),
  sliced-Wasserstein-1 (64 projections).
- Baselines: non-private moment matching (utility upper bound); naive
  DP independent-Gaussian fit (privately estimated mean only, fixed
  public scale 0.35, unimodal).
- Full run: n in {500, 2000, 8000} x eps in {1.0, 4.0}, 10 replicates
  each (60 runs). Paired bootstrap CIs (2000 resamples) over replicates.

## Pilot numbers (n=2000, 5 replicates; the ONLY data used to set bars)

| eps | dpmoment MMD | naive MMD | nonprivate MMD | MMD advantage [95% CI] | SW1 advantage [95% CI] | utility gap MMD |
|-----|--------------|-----------|----------------|------------------------|------------------------|-----------------|
| 0.5 | 0.0927 | 0.3603 | 0.0593 | 0.2676 [0.255, 0.280] | 0.1275 [0.112, 0.151] | 0.0334 |
| 1.0 | 0.0737 | 0.3600 | 0.0593 | 0.2863 [0.278, 0.294] | 0.1381 [0.125, 0.158] | 0.0144 |
| 2.0 | 0.0649 | 0.3599 | 0.0593 | 0.2950 [0.289, 0.299] | 0.1426 [0.131, 0.161] | 0.0056 |
| 4.0 | 0.0625 | 0.3599 | 0.0593 | 0.2974 [0.294, 0.301] | 0.1440 [0.132, 0.163] | 0.0032 |

Pilot cost: ~95 s CPU for 20 runs. All pilot accountant checks passed.

## Locked determinative bars

- **G0 competence pre-check (must pass, else the task is vacuous):**
  non-private MMD mean < 0.15 at n=2000 in the full run, AND
  non-private beats naive by >= 0.15 MMD mean. (Pilot: 0.0593, gap 0.3007.)
- **G2 disease / load-bearing empirical claim (must pass at BOTH eps=1
  and eps=4, at n=2000):**
  (a) MMD advantage (naive minus dpmoment) mean >= 0.10 with paired
      bootstrap 95% CI lower bound > 0.05;
  (b) SW1 advantage mean >= 0.05 with CI lower bound > 0.03;
  (c) error decays with n: fitted log-log exponent of dpmoment MMD vs
      n is < 0 at both eps (reported honestly; NOT forced to -0.5 or to
      any theoretical value — a non-negative exponent fails G2).
  Bars are set well below pilot effects (pilot advantages ~0.29 MMD /
  ~0.14 SW1) to leave headroom for full-run noise, not to chase the pilot.
- **G3 load-bearing privacy + utility (must pass on ALL full runs):**
  (a) accountant: implied epsilon <= budget on every run
      (max implied eps == budget by construction, checked per run);
  (b) deliberate over-budget probe caught: halving sigma must yield
      implied epsilon = 2x budget, flagged as over budget;
  (c) utility gap (dpmoment minus non-private MMD mean) <= 0.10 at
      eps=1 and <= 0.05 at eps=4, at n=2000. (Pilot: 0.0144 / 0.0032.)
- **G1 author-code shelf (checked with the authenticated GitHub
  connector before the full run):** if official author code for
  2610.30196 exists, HARD STOP / KILL regardless of the numbers above.

## Kill conditions (any one kills, no soft pass)

- G1 finds author code; G0 competence fails; any G2 bar fails at either
  eps; any G3 bar fails; pytest suite not green; determinism check fails
  (same seed rerun not bit-identical on metrics).
