"""dpmoment: differentially private synthetic data via generalized moment matching.

Replication (not the authors' code) of the estimator family in
arXiv 2610.30196 (Ding, He, Zhu): privately estimate generalized moments,
fit a probability measure to them, sample synthetic records.

Scope honesty: this is an *implementation-and-rate-verification* of the
framework on synthetic tabular data with known ground truth. It is not a
claim to reproduce the paper's minimax theorems, and it uses a fixed,
documented feature map (polynomial degree<=2 + random Fourier features)
rather than the paper's full kernel families.
"""

from .moments import feature_map, empirical_moments, make_rff_params
from .privacy import gaussian_sigma, composed_epsilon, private_moments, check_budget
from .fitting import fit_measure, sample_from_measure
from .metrics import mmd_rbf, sliced_wasserstein1

__all__ = [
    "feature_map",
    "empirical_moments",
    "make_rff_params",
    "gaussian_sigma",
    "composed_epsilon",
    "private_moments",
    "check_budget",
    "fit_measure",
    "sample_from_measure",
    "mmd_rbf",
    "sliced_wasserstein1",
]
