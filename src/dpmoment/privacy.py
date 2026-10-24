"""(epsilon, delta)-DP via the Gaussian mechanism, with an explicit accountant.

Standard calibration: for L2 sensitivity Delta and noise sigma,
  epsilon = (Delta / sigma) * sqrt(2 * ln(1.25 / delta)).
`composed_epsilon` inverts that formula, so any run can be audited:
the accountant is asked for the epsilon implied by the sigma actually
used, and the run is only valid if that value is within budget.
"""

import math


def gaussian_sigma(epsilon, delta, sensitivity):
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("need epsilon > 0 and 0 < delta < 1")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be positive")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def composed_epsilon(sigma, delta, sensitivity):
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (sensitivity / sigma) * math.sqrt(2.0 * math.log(1.25 / delta))


def private_moments(mu, epsilon, delta, sensitivity, rng):
    """Add Gaussian noise calibrated to (epsilon, delta); return (noisy, sigma)."""
    sigma = gaussian_sigma(epsilon, delta, sensitivity)
    return mu + rng.normal(0.0, sigma, size=mu.shape), sigma


def check_budget(sigma, delta, sensitivity, epsilon_budget):
    """Return (implied_epsilon, within_budget). The over-budget probe uses this."""
    implied = composed_epsilon(sigma, delta, sensitivity)
    return implied, implied <= epsilon_budget + 1e-12
