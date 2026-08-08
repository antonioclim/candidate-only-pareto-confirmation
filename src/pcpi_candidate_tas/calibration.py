"""Exact-null calibration and robustness mechanisms.

The exact-null campaign uses one least-favourable boundary challenger and two
easy positive controls. For the Gaussian core cells, a shared candidate
component induces realistic cross-challenger dependence while preserving the
declared marginal covariance of every paired difference.

Robustness cells are applicability-boundary diagnostics. They do not extend the
exact theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.stats import beta, f, t

from .paired import count_mass_log_telescoping
from .accelerated import _grid_counts, _orthant_distance_batch


@dataclass(frozen=True)
class CalibrationDesign:
    sample_cap: int = 2_000
    stopping_grid: int = 5
    minimum_count: int = 8
    delta: float = 0.05
    challengers: int = 3
    easy_effect: float = 1.0
    easy_negative_coordinate: float = -0.20
    robustness_dimension: int = 3
    robustness_primary_method: str = "hybrid"
    hybrid_hotelling_share: float = 0.5
    core_cells: int = 12
    core_replications_per_cell: int = 3_000
    robustness_cells: int = 8
    robustness_replications_per_cell: int = 1_000
    familywise_confidence: float = 0.95
    gate_threshold: float = 0.065

    def validate(self) -> None:
        if self.sample_cap < 10 or self.stopping_grid < 1:
            raise ValueError("invalid cap or stopping grid")
        if not 0 < self.delta < 1:
            raise ValueError("delta must lie in (0,1)")
        if self.challengers != 3:
            raise ValueError("the frozen campaign has three challengers")
        if self.core_cells != 12 or self.robustness_cells != 8:
            raise ValueError("frozen cell count mismatch")
        if self.robustness_primary_method != "hybrid":
            raise ValueError("the frozen robustness primary method is hybrid")


def covariance_matrix(dimension: int, label: str) -> np.ndarray:
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if label == "identity":
        return np.eye(dimension)
    if label == "compound_rho_0.5":
        covariance = np.full((dimension, dimension), 0.5)
        np.fill_diagonal(covariance, 1.0)
        return covariance
    raise ValueError(f"unknown covariance label {label!r}")


def challenger_means(
    dimension: int,
    *,
    easy_effect: float = 1.0,
    negative_coordinate: float = -0.20,
) -> np.ndarray:
    if dimension < 2:
        raise ValueError("at least two objectives are required")
    means = [np.zeros(dimension)]
    for witness in (0, 1):
        mean = np.full(dimension, negative_coordinate, dtype=float)
        mean[witness] = easy_effect
        means.append(mean)
    return np.asarray(means)


def generate_exact_null_path(
    seed: int,
    dimension: int,
    covariance_label: str,
    *,
    count: int = 2_000,
    easy_effect: float = 1.0,
    negative_coordinate: float = -0.20,
) -> np.ndarray:
    """Generate challenger-by-count-by-objective Gaussian paired differences.

    Let C ~ N(0, Sigma/2) be a common candidate component and
    X_i ~ N(mu_i, Sigma/2) be independent challenger components. Then
    D_i = X_i - C has marginal covariance Sigma and
    Cov(D_i, D_j) = Sigma/2 for i != j.
    """

    if count < 2:
        raise ValueError("count must be at least two")
    covariance = covariance_matrix(dimension, covariance_label)
    half_cholesky = np.linalg.cholesky(covariance / 2.0)
    means = challenger_means(
        dimension,
        easy_effect=easy_effect,
        negative_coordinate=negative_coordinate,
    )
    rng = np.random.default_rng(seed)
    candidate = rng.normal(size=(count, dimension)) @ half_cholesky.T
    output = np.empty((3, count, dimension), dtype=float)
    for challenger, mean in enumerate(means):
        individual = rng.normal(size=(count, dimension)) @ half_cholesky.T
        output[challenger] = mean + individual - candidate
    return output


def _elliptical_t(
    rng: np.random.Generator,
    count: int,
    dimension: int,
    degrees_of_freedom: int,
    covariance: np.ndarray,
) -> np.ndarray:
    normal = rng.normal(size=(count, dimension)) @ np.linalg.cholesky(
        covariance
    ).T
    scale = rng.chisquare(degrees_of_freedom, size=count) / degrees_of_freedom
    variance_correction = math.sqrt(
        (degrees_of_freedom - 2.0) / degrees_of_freedom
    )
    return normal / np.sqrt(scale[:, None]) * variance_correction


def _centred_lognormal(
    rng: np.random.Generator,
    count: int,
    dimension: int,
    sigma: float = 0.8,
) -> np.ndarray:
    normal = rng.normal(size=(count, dimension))
    raw = np.exp(sigma * normal)
    mean = math.exp(sigma * sigma / 2.0)
    variance = (math.exp(sigma * sigma) - 1.0) * math.exp(sigma * sigma)
    return (raw - mean) / math.sqrt(variance)


def _stationary_ar1(
    rng: np.random.Generator,
    count: int,
    dimension: int,
    rho: float,
) -> np.ndarray:
    innovation = rng.normal(size=(count, dimension))
    output = np.empty_like(innovation)
    output[0] = rng.normal(size=dimension)
    innovation_scale = math.sqrt(1.0 - rho * rho)
    for index in range(1, count):
        output[index] = (
            rho * output[index - 1]
            + innovation_scale * innovation[index]
        )
    return output


def generate_robustness_path(
    seed: int,
    mechanism: str,
    *,
    count: int = 2_000,
    dimension: int = 3,
    easy_effect: float = 1.0,
    negative_coordinate: float = -0.20,
) -> np.ndarray:
    """Generate one three-challenger null/stress path.

    Challenger zero is on the null boundary. Challengers one and two are easy
    positive controls. Each challenger path is generated independently; this is
    a valid archive-level setting because the global logic does not require
    cross-challenger independence.
    """

    rng = np.random.default_rng(seed)
    means = challenger_means(
        dimension,
        easy_effect=easy_effect,
        negative_coordinate=negative_coordinate,
    )
    output = np.empty((3, count, dimension), dtype=float)
    for challenger, mean in enumerate(means):
        if mechanism == "iid Gaussian reference":
            centred = rng.normal(size=(count, dimension))
        elif mechanism == "multivariate t, df=5":
            centred = _elliptical_t(
                rng, count, dimension, 5, np.eye(dimension)
            )
        elif mechanism == "multivariate t, df=3":
            centred = _elliptical_t(
                rng, count, dimension, 3, np.eye(dimension)
            )
        elif mechanism == "centred lognormal coordinates":
            centred = _centred_lognormal(rng, count, dimension)
        elif mechanism == "1% ten-SD contamination":
            centred = rng.normal(size=(count, dimension))
            contaminated = rng.random(count) < 0.01
            centred[contaminated] = rng.normal(
                scale=10.0,
                size=(int(contaminated.sum()), dimension),
            )
        elif mechanism == "AR(1) rho=0.3":
            centred = _stationary_ar1(rng, count, dimension, 0.3)
        elif mechanism == "AR(1) rho=0.6":
            centred = _stationary_ar1(rng, count, dimension, 0.6)
        elif mechanism == "near-singular covariance":
            covariance = np.full((dimension, dimension), 0.9999)
            np.fill_diagonal(covariance, 1.0)
            centred = rng.multivariate_normal(
                np.zeros(dimension), covariance, size=count
            )
        else:
            raise ValueError(f"unknown robustness mechanism {mechanism!r}")
        output[challenger] = mean + centred
    return output


def _first_true_count(values: np.ndarray, counts: np.ndarray) -> int | None:
    locations = np.flatnonzero(values)
    return None if locations.size == 0 else int(counts[int(locations[0])])


def sequential_method_diagnostics(
    differences: np.ndarray,
    delta: float,
    *,
    methods: Iterable[str] = ("hybrid",),
    hybrid_hotelling_share: float = 0.5,
    min_count: int = 8,
    max_count: int = 2_000,
    check_every: int = 5,
) -> dict[str, dict]:
    """Evaluate coordinate, Hotelling and/or hybrid boundaries in one pass."""

    data = np.asarray(differences, dtype=float)
    if data.ndim != 3 or data.shape[0] < 1 or data.shape[2] < 1:
        raise ValueError("challenger-by-count-by-objective data are required")
    if not np.all(np.isfinite(data)):
        raise ValueError("differences must be finite")
    requested = tuple(dict.fromkeys(methods))
    if not requested or not set(requested).issubset(
        {"coordinate", "hotelling", "hybrid"}
    ):
        raise ValueError("unknown or empty method set")

    challenger_count, total, dimension = data.shape
    start = max(
        min_count,
        dimension + 2
        if any(method != "coordinate" for method in requested)
        else 2,
    )
    counts = _grid_counts(
        total,
        min_count=start,
        max_count=max_count,
        check_every=check_every,
    )
    indices = counts - 1
    sums = np.cumsum(data, axis=1)[:, indices].transpose(1, 0, 2)
    outer = np.einsum("cnm,cnk->cnmk", data, data)
    outer_sums = np.cumsum(outer, axis=1)[:, indices].transpose(
        1, 0, 2, 3
    )
    count_array = counts[:, None, None].astype(float)
    means = sums / count_array
    covariances = (
        outer_sums
        - np.einsum("gcm,gck->gcmk", sums, sums)
        / count_array[..., None]
    ) / (counts[:, None, None, None] - 1.0)
    covariances = 0.5 * (
        covariances + np.swapaxes(covariances, 2, 3)
    )

    masses = np.asarray(
        [count_mass_log_telescoping(int(count)) for count in counts]
    )
    alpha = delta * masses

    variances = np.diagonal(covariances, axis1=2, axis2=3)
    t_statistics = np.full_like(means, -np.inf)
    good = np.isfinite(variances) & (variances > 0)
    broadcast_counts = np.broadcast_to(count_array, means.shape)
    t_statistics[good] = means[good] / np.sqrt(
        variances[good] / broadcast_counts[good]
    )
    coordinate_max = np.max(t_statistics, axis=2)

    need_hotelling = any(
        method in {"hotelling", "hybrid"} for method in requested
    )
    if need_hotelling:
        distances, ranks = _orthant_distance_batch(
            means, covariances, counts[:, None]
        )
    else:
        distances = np.full(coordinate_max.shape, np.nan)
        ranks = np.zeros(coordinate_max.shape, dtype=int)

    output: dict[str, dict] = {}
    for method in requested:
        if method == "coordinate":
            coordinate_alpha = alpha
            coordinate_threshold = t.isf(
                coordinate_alpha / dimension, counts - 1
            )
            local = (
                coordinate_max > coordinate_threshold[:, None]
            )
        elif method == "hotelling":
            hotelling_threshold = (
                dimension
                * (counts - 1)
                / (counts - dimension)
                * f.isf(alpha, dimension, counts - dimension)
            )
            local = (
                np.isfinite(distances)
                & (ranks == dimension)
                & (distances > hotelling_threshold[:, None])
            )
        else:
            coordinate_alpha = alpha * (1.0 - hybrid_hotelling_share)
            hotelling_alpha = alpha * hybrid_hotelling_share
            coordinate_threshold = t.isf(
                coordinate_alpha / dimension, counts - 1
            )
            hotelling_threshold = (
                dimension
                * (counts - 1)
                / (counts - dimension)
                * f.isf(
                    hotelling_alpha, dimension, counts - dimension
                )
            )
            local = (
                coordinate_max > coordinate_threshold[:, None]
            ) | (
                np.isfinite(distances)
                & (ranks == dimension)
                & (distances > hotelling_threshold[:, None])
            )

        global_crossed = np.all(local, axis=1)
        stopping_count = _first_true_count(global_crossed, counts)
        boundary_count = _first_true_count(local[:, 0], counts)
        easy_joint = (
            np.all(local[:, 1:], axis=1)
            if challenger_count > 1
            else np.ones(len(counts), dtype=bool)
        )
        easy_count = _first_true_count(easy_joint, counts)
        output[method] = {
            "certified": stopping_count is not None,
            "stopping_count": (
                int(stopping_count)
                if stopping_count is not None
                else int(counts[-1])
            ),
            "decision_checks": (
                int(np.searchsorted(counts, stopping_count) + 1)
                if stopping_count is not None
                else len(counts)
            ),
            "boundary_ever_crossed": boundary_count is not None,
            "boundary_first_crossing_count": (
                "" if boundary_count is None else int(boundary_count)
            ),
            "easy_controls_jointly_crossed": easy_count is not None,
            "easy_controls_first_joint_crossing_count": (
                "" if easy_count is None else int(easy_count)
            ),
            "final_full_rank_all_challengers": bool(
                np.all(ranks[-1] == dimension)
            )
            if need_hotelling
            else True,
        }
    return output


def one_sided_clopper_pearson_upper(
    successes: int,
    trials: int,
    confidence: float,
) -> float:
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("invalid successes or trials")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    if successes == trials:
        return 1.0
    return float(
        beta.ppf(confidence, successes + 1, trials - successes)
    )


def bonferroni_core_upper(
    successes: int,
    trials: int,
    *,
    cells: int = 12,
    familywise_confidence: float = 0.95,
) -> float:
    per_cell_confidence = 1.0 - (
        1.0 - familywise_confidence
    ) / cells
    return one_sided_clopper_pearson_upper(
        successes, trials, per_cell_confidence
    )
