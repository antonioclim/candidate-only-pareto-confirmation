"""Anytime-valid Gaussian boundaries for candidate-only Pareto certification.

The module implements:

* a conservative Laurent--Massart bound;
* the exact least-favourable chi-bar-square cone boundary;
* a sparse coordinatewise one-sided boundary; and
* a valid hybrid rule that splits the local error budget.

All rules are indexed by intrinsic candidate/challenger sample counts. A
summable count-mass sequence preserves validity under adaptive sampling and
optional stopping.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln, zeta
from scipy.stats import chi2, norm

SpendingName = Literal["power", "log_telescoping"]
BoundaryName = Literal["laurent_massart", "chi_bar", "coordinate", "hybrid"]


@dataclass(frozen=True)
class BoundaryDecision:
    """Replayable evidence for one challenger."""

    crossed: bool
    cone_glr: float
    cone_threshold: float | None
    max_z: float
    coordinate_threshold: float | None
    witness_objective: int
    local_alpha: float
    method: str
    spending: str


def count_mass_power(count: int, exponent: float = 2.0) -> float:
    if count < 1 or not math.isfinite(exponent) or exponent <= 1.0:
        raise ValueError("count >= 1 and exponent > 1 are required")
    return 1.0 / (float(zeta(exponent, 1.0)) * count**exponent)


def count_mass_log_telescoping(count: int) -> float:
    """Exactly summable mass asymptotic to 1/[s log(s)^2]."""

    if count < 1:
        raise ValueError("count >= 1 is required")
    return math.log(2.0) * (
        1.0 / math.log(count + 1.0) - 1.0 / math.log(count + 2.0)
    )


def count_mass(
    count: int,
    spending: SpendingName = "log_telescoping",
    exponent: float = 2.0,
) -> float:
    if spending == "log_telescoping":
        return count_mass_log_telescoping(count)
    if spending == "power":
        return count_mass_power(count, exponent)
    raise ValueError(f"unknown spending schedule: {spending}")


def intrinsic_pair_alpha(
    delta: float,
    candidate_count: int,
    challenger_count: int,
    *,
    spending: SpendingName = "log_telescoping",
    exponent: float = 2.0,
) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return (
        delta
        * count_mass(candidate_count, spending, exponent)
        * count_mass(challenger_count, spending, exponent)
    )


def _mixture_weights(dimension: int) -> np.ndarray:
    if dimension < 1:
        raise ValueError("dimension >= 1 is required")
    k = np.arange(dimension + 1, dtype=float)
    logw = (
        gammaln(dimension + 1.0)
        - gammaln(k + 1.0)
        - gammaln(dimension - k + 1.0)
        - dimension * math.log(2.0)
    )
    return np.exp(logw)


def chi_bar_square_sf(value: float, dimension: int) -> float:
    """Survival function of sum_j max(Z_j, 0)^2 for standard Gaussian Z."""

    if value < 0.0:
        return 1.0
    weights = _mixture_weights(dimension)
    if value == 0.0:
        return float(1.0 - weights[0])
    ks = np.arange(1, dimension + 1)
    return float(np.dot(weights[1:], chi2.sf(value, ks)))


@lru_cache(maxsize=32768)
def chi_bar_square_isf(alpha: float, dimension: int) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if dimension < 1:
        raise ValueError("dimension >= 1 is required")
    atom_survival = 1.0 - 2.0 ** (-dimension)
    if alpha >= atom_survival:
        return 0.0
    lo = 0.0
    hi = max(1.0, float(chi2.isf(alpha, dimension)))
    while chi_bar_square_sf(hi, dimension) > alpha:
        hi *= 2.0
        if hi > 1e8:
            raise RuntimeError("failed to bracket chi-bar-square quantile")
    return float(
        brentq(
            lambda x: chi_bar_square_sf(x, dimension) - alpha,
            lo,
            hi,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=200,
        )
    )


def laurent_massart_glr_threshold(alpha: float, dimension: int) -> float:
    if not 0.0 < alpha < 1.0 or dimension < 1:
        raise ValueError("invalid alpha or dimension")
    u = math.log(1.0 / alpha)
    return 0.5 * (dimension + 2.0 * math.sqrt(dimension * u) + 2.0 * u)


@lru_cache(maxsize=32768)
def _quantised_chi_bar_glr_threshold(
    u_step: int, dimension: int, resolution: float
) -> float:
    """Conservative cached chi-bar threshold on a log-alpha grid."""

    quantised_u = u_step * resolution
    return 0.5 * chi_bar_square_isf(math.exp(-quantised_u), dimension)


def chi_bar_glr_threshold(
    alpha: float,
    dimension: int,
    *,
    conservative_log_resolution: float | None = 0.05,
) -> float:
    """Return a chi-bar GLR threshold.

    When ``conservative_log_resolution`` is positive, ``log(1/alpha)`` is
    rounded upward to a grid. The resulting threshold is no smaller than the
    exact threshold and is therefore conservative while being highly cacheable
    in sequential simulations. Pass ``None`` for the exact root.
    """

    if conservative_log_resolution is None:
        return 0.5 * chi_bar_square_isf(alpha, dimension)
    if conservative_log_resolution <= 0.0:
        raise ValueError("positive resolution or None required")
    u = math.log(1.0 / alpha)
    step = int(math.ceil(u / conservative_log_resolution))
    return _quantised_chi_bar_glr_threshold(
        step, dimension, conservative_log_resolution
    )


def coordinate_z_threshold(alpha: float, dimension: int) -> float:
    if not 0.0 < alpha < 1.0 or dimension < 1:
        raise ValueError("invalid alpha or dimension")
    return float(norm.isf(alpha / dimension))


def standardised_gap_vector(
    candidate_mean: np.ndarray,
    challenger_mean: np.ndarray,
    candidate_count: int,
    challenger_count: int,
    candidate_variance: np.ndarray,
    challenger_variance: np.ndarray,
) -> np.ndarray:
    if candidate_count < 1 or challenger_count < 1:
        raise ValueError("positive counts required")
    gap = np.asarray(challenger_mean, dtype=float) - np.asarray(
        candidate_mean, dtype=float
    )
    variance = np.asarray(challenger_variance, dtype=float) / challenger_count
    variance += np.asarray(candidate_variance, dtype=float) / candidate_count
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("positive finite variances required")
    return gap / np.sqrt(variance)


def evaluate_boundary(
    candidate_mean: np.ndarray,
    challenger_mean: np.ndarray,
    candidate_count: int,
    challenger_count: int,
    candidate_variance: np.ndarray,
    challenger_variance: np.ndarray,
    delta: float,
    *,
    method: BoundaryName = "chi_bar",
    spending: SpendingName = "log_telescoping",
    exponent: float = 2.0,
    hybrid_cone_share: float = 0.5,
) -> BoundaryDecision:
    z = standardised_gap_vector(
        candidate_mean,
        challenger_mean,
        candidate_count,
        challenger_count,
        candidate_variance,
        challenger_variance,
    )
    positive = np.maximum(z, 0.0)
    cone_glr = 0.5 * float(np.dot(positive, positive))
    witness = int(np.argmax(z))
    max_z = float(z[witness])
    alpha = intrinsic_pair_alpha(
        delta,
        candidate_count,
        challenger_count,
        spending=spending,
        exponent=exponent,
    )
    dimension = int(z.size)

    if method == "laurent_massart":
        threshold = laurent_massart_glr_threshold(alpha, dimension)
        return BoundaryDecision(
            cone_glr >= threshold, cone_glr, threshold, max_z, None,
            witness, alpha, method, spending
        )
    if method == "chi_bar":
        threshold = chi_bar_glr_threshold(alpha, dimension)
        return BoundaryDecision(
            cone_glr >= threshold, cone_glr, threshold, max_z, None,
            witness, alpha, method, spending
        )
    if method == "coordinate":
        threshold = coordinate_z_threshold(alpha, dimension)
        return BoundaryDecision(
            max_z >= threshold, cone_glr, None, max_z, threshold,
            witness, alpha, method, spending
        )
    if method == "hybrid":
        if not 0.0 < hybrid_cone_share < 1.0:
            raise ValueError("hybrid_cone_share must lie in (0, 1)")
        cone_threshold = chi_bar_glr_threshold(
            alpha * hybrid_cone_share, dimension
        )
        coordinate_threshold = coordinate_z_threshold(
            alpha * (1.0 - hybrid_cone_share), dimension
        )
        crossed = cone_glr >= cone_threshold or max_z >= coordinate_threshold
        return BoundaryDecision(
            crossed, cone_glr, cone_threshold, max_z, coordinate_threshold,
            witness, alpha, method, spending
        )
    raise ValueError(f"unknown boundary method: {method}")


def evaluate_all_challengers(
    means: np.ndarray,
    counts: np.ndarray,
    variances: np.ndarray,
    delta: float,
    *,
    method: BoundaryName = "chi_bar",
    spending: SpendingName = "log_telescoping",
    exponent: float = 2.0,
    hybrid_cone_share: float = 0.5,
) -> tuple[BoundaryDecision, ...]:
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=int)
    variances = np.asarray(variances, dtype=float)
    if means.shape != variances.shape or counts.shape != (means.shape[0],):
        raise ValueError("incompatible means, counts and variances")
    return tuple(
        evaluate_boundary(
            means[0], means[i], int(counts[0]), int(counts[i]),
            variances[0], variances[i], delta,
            method=method, spending=spending, exponent=exponent,
            hybrid_cone_share=hybrid_cone_share,
        )
        for i in range(1, means.shape[0])
    )
