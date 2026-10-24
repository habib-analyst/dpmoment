# dpmoment — Differentially Private Synthetic Data via Generalized Moment Matching

An independent, CPU-only **replication of the estimator family** in
arXiv 2610.30196 (Ding, He, Zhu, "Differentially Private Synthetic Data
via Generalized Moment Matching"). This is **not the authors' code** —
no author code was found (see below) — and it is **not a proof** of the
paper's minimax theorems. It implements the framework's pipeline
(privately estimate generalized moments → fit a measure → sample
synthetic records) and verifies its rates empirically on synthetic
tabular data with known ground truth.

## What the paper claims

(ε, δ)-DP synthetic data for arbitrary databases of n records in
[-1, 1]^d under a specified integral probability metric: privately
estimate moments, fit a probability measure, sample records. With
enough synthetic records, the rates are minimax-optimal up to log
factors for fixed dimension, for MMD (Matérn / Laplace / NTK kernels)
and sliced-Wasserstein-1.

## What this repo verifies (measured, not claimed)

On a 3-component Gaussian mixture in [-1, 1]^3 (δ = 1e-5, 10
replicates per cell, MMD with RBF kernel vs a 4000-point reference):

| n | ε | DP moment-matching MMD | naive DP Gaussian MMD | non-private MMD |
|---|---|------------------------|----------------------|-----------------|
| 500 | 1.0 | 0.123 | 0.364 | 0.066 |
| 500 | 4.0 | 0.071 | 0.362 | 0.066 |
| 2000 | 1.0 | 0.071 | 0.360 | 0.058 |
| 2000 | 4.0 | 0.060 | 0.360 | 0.058 |
| 8000 | 1.0 | 0.058 | 0.362 | 0.060 |
| 8000 | 4.0 | 0.058 | 0.362 | 0.060 |

- **DP moment matching beats the naive DP baseline by a wide margin:**
  at n = 2000, MMD advantage 0.289 (95% paired bootstrap CI
  [0.284, 0.295]) at ε = 1.0 and 0.300 ([0.297, 0.304]) at ε = 4.0;
  sliced-W₁ advantages 0.138 ([0.129, 0.148]) and 0.143 ([0.135, 0.153]).
- **Error decays with n**, with fitted log-log exponents −0.27 (ε = 1.0)
  and −0.07 (ε = 4.0). Honest note: the ε = 4.0 exponent is shallow
  because the error is already at the non-private floor (~0.058) — the
  fit is dominated by the floor, not by a clean power law. We report
  the fitted exponents as measured and do **not** claim they match the
  paper's theoretical rates.
- **Privacy accounting verified on every run:** the Gaussian
  mechanism's implied ε equals the budget exactly (never over), and a
  deliberate over-budget probe (halved σ → implied ε = 2× budget) is
  caught by the accountant (covered by test).
- **Utility gap to the non-private upper bound is small:** 0.013 MMD
  at ε = 1.0, 0.002 at ε = 4.0 (n = 2000). At n = 8000 the DP error
  sits at the non-private floor within noise.

Full numbers, CIs, and the pre-registered bars: `PREREG.md`,
`results/full_summary.json`, `PAPER.md`.

## Method (this implementation)

- **Moments:** 42 bounded features — constant, first-order, second-order
  (i ≤ j), plus 32 random Fourier features. Every coordinate lies in
  [-1, 1], so the L2 sensitivity of the empirical mean is
  2·√42/n and one Gaussian-mechanism draw privatizes all moments.
- **Measure fit:** a fixed 1500-point uniform support in [-1, 1]^3
  (independent of the private data — the only private contact is the
  moment vector), weights fit by projected gradient descent onto the
  simplex; synthetic records are sampled from the fitted discrete
  measure.
- **Baselines:** non-private moment matching (same pipeline, exact
  moments) and a naive DP independent-Gaussian fit (privately estimated
  mean only, fixed public scale 0.35, unimodal — it cannot represent a
  3-mode mixture, which is exactly why it loses).

## Honest scope and limitations

- Synthetic data only; a single 3-D mixture ground truth; one fixed
  feature map (not the paper's Matérn/Laplace/NTK kernel families).
- The fitted measure lives on a fixed support — it cannot place mass
  where the support has no points, which sets the ~0.058 MMD floor.
- This is an empirical rate check, not a reproduction of the paper's
  theorems and not a production DP system. Do not use it on real
  personal data without a full DP review (composition across releases,
  δ accounting, and support construction all need auditing).
- The naive baseline is deliberately weak (first-moment, unimodal);
  beating it shows the moment-matching machinery captures multimodality
  under privacy, not that it beats every DP synthesizer.

## Author-code shelf check (G1)

2026-10-24, authenticated GitHub search: arXiv abs page carries no code
links; repo search for `2610.30196` = 0 hits, for `dpmoment` = 0 hits;
authenticated code search for `2610.30196` = 31 hits across 16 repos,
all arXiv aggregators / daily-paper mirrors / personal blogs (e.g.
qhduan/cn-chat-arxiv) that merely list the paper — no author
implementation. Gate passed; if author code appears later, this repo
should be re-assessed.

## Run it

```bash
pip install -e .
python examples/demo.py            # quick demo, eps=1.0, n=2000
python experiment.py --mode pilot --out pilot.jsonl
python experiment.py --mode full --out full.jsonl
python summarize.py full.jsonl --out results/full_summary.json
python -m pytest tests/ -q         # 17 tests
python check_determinism.py        # same-seed rerun is bit-identical
```

CPU-only, $0 cost, no network calls at runtime. Python ≥ 3.10,
numpy, scipy.
