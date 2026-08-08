"""Paired multivariate-normal candidate-only certification.

The module treats one common scenario as a paired observation for all compared
systems.  For challenger i, the difference vector is
D_i = outcome_i - outcome_candidate in minimisation orientation.  The candidate
is anti-dominated by challenger i whenever at least one component of E[D_i] is
strictly positive.

The Hotelling certificate is an exact confidence-set separation argument under
i.i.d. multivariate normal paired differences with unknown positive-definite
covariance.  The coordinate certificate is a union of one-sided Student-t
bounds.  A hybrid splits the local error budget and accepts either certificate.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal
import numpy as np
from scipy.optimize import nnls
from scipy.stats import f, t

Method = Literal["hotelling", "coordinate", "hybrid"]

@dataclass(frozen=True)
class PairedDecision:
    crossed: bool
    method: str
    count: int
    local_alpha: float
    hotelling_distance: float | None
    hotelling_threshold: float | None
    coordinate_t_max: float
    coordinate_threshold: float | None
    witness_objective: int
    covariance_rank: int


def count_mass_log_telescoping(count: int) -> float:
    if count < 2:
        raise ValueError("count >= 2 is required")
    # Shifted telescope over s=2,3,...
    return math.log(3.0) * (
        1.0 / math.log(count + 1.0) - 1.0 / math.log(count + 2.0)
    )


def intrinsic_alpha(delta: float, count: int) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    return delta * count_mass_log_telescoping(count)


def sample_mean_cov(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x=np.asarray(samples,dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("samples must be a count-by-objective matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite")
    return x.mean(axis=0), np.cov(x,rowvar=False,ddof=1).reshape(x.shape[1],x.shape[1])


def hotelling_threshold(alpha: float, count: int, dimension: int) -> float:
    if not 0.0 < alpha < 1.0 or count <= dimension or dimension < 1:
        raise ValueError("alpha in (0,1) and count > dimension are required")
    return float(dimension*(count-1)/(count-dimension)*f.isf(alpha,dimension,count-dimension))


def orthant_mahalanobis_distance(mean: np.ndarray, covariance: np.ndarray, count: int) -> tuple[float,int]:
    """Minimum n*(mean-mu)'S^-1*(mean-mu) over mu <= 0.

    With x=-mu >= 0 and A'A=S^-1, solve min ||A x + A mean||^2.
    Returns infinity only when the covariance is numerically invalid; callers
    treat singular covariance as non-certification rather than evidence.
    """
    mean=np.asarray(mean,dtype=float)
    cov=np.asarray(covariance,dtype=float)
    if mean.ndim!=1 or cov.shape!=(mean.size,mean.size) or count<2:
        raise ValueError("incompatible mean/covariance/count")
    cov=(cov+cov.T)/2
    rank=int(np.linalg.matrix_rank(cov,tol=max(cov.shape)*np.finfo(float).eps*np.linalg.norm(cov,2)))
    if rank < mean.size:
        return math.nan, rank
    try:
        precision=np.linalg.inv(cov)
        # L satisfies L L^T = precision, so A=L^T yields A^T A=precision.
        L=np.linalg.cholesky(precision)
        A=L.T
        x,_=nnls(A,-A@mean)
        residual=A@(mean+x)
        return float(count*np.dot(residual,residual)),rank
    except (np.linalg.LinAlgError,ValueError):
        return math.nan,rank


def coordinate_t_statistics(mean: np.ndarray,covariance: np.ndarray,count:int) -> np.ndarray:
    var=np.diag(np.asarray(covariance,dtype=float))
    out=np.full_like(np.asarray(mean,dtype=float),-np.inf)
    good=np.isfinite(var)&(var>0)
    out[good]=np.asarray(mean,dtype=float)[good]/np.sqrt(var[good]/count)
    return out


def evaluate_paired_statistics(
    mean: np.ndarray, covariance: np.ndarray, count: int, delta: float, *,
    method: Method = "hybrid", hybrid_hotelling_share: float = 0.5,
) -> PairedDecision:
    """Evaluate one challenger from sufficient statistics."""
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.ndim != 1 or covariance.shape != (mean.size, mean.size):
        raise ValueError("incompatible mean and covariance")
    if count < 2 or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("finite statistics and count >= 2 are required")
    dimension = mean.size
    alpha = intrinsic_alpha(delta, count)
    if method not in {"hotelling", "coordinate", "hybrid"}:
        raise ValueError("unknown method")
    if not 0.0 < hybrid_hotelling_share < 1.0:
        raise ValueError("hybrid share must lie in (0,1)")
    hot_alpha = alpha if method == "hotelling" else alpha * hybrid_hotelling_share
    coord_alpha = alpha if method == "coordinate" else alpha * (1.0 - hybrid_hotelling_share)
    distance, rank = orthant_mahalanobis_distance(mean, covariance, count)
    hot_threshold = None
    hot_crossed = False
    if count > dimension and rank == dimension and math.isfinite(distance):
        hot_threshold = hotelling_threshold(hot_alpha, count, dimension)
        hot_crossed = distance > hot_threshold
    t_statistics = coordinate_t_statistics(mean, covariance, count)
    witness = int(np.argmax(t_statistics))
    t_max = float(t_statistics[witness])
    coordinate_threshold = float(t.isf(coord_alpha / dimension, count - 1))
    coordinate_crossed = t_max > coordinate_threshold
    if method == "hotelling":
        crossed = hot_crossed
    elif method == "coordinate":
        crossed = coordinate_crossed
    else:
        crossed = hot_crossed or coordinate_crossed
    return PairedDecision(
        bool(crossed), method, count, alpha,
        None if not math.isfinite(distance) else distance, hot_threshold,
        t_max, coordinate_threshold, witness, rank,
    )


def evaluate_paired_samples(
    samples: np.ndarray, delta: float, *, method: Method = "hybrid",
    hybrid_hotelling_share: float = 0.5,
) -> PairedDecision:
    x = np.asarray(samples, dtype=float)
    mean, covariance = sample_mean_cov(x)
    return evaluate_paired_statistics(
        mean, covariance, x.shape[0], delta, method=method,
        hybrid_hotelling_share=hybrid_hotelling_share,
    )


def evaluate_archive(differences: np.ndarray,delta:float,*,method:Method="hybrid",hybrid_hotelling_share:float=0.5) -> tuple[bool,tuple[PairedDecision,...]]:
    """Evaluate challenger-by-count-by-objective paired differences.

    No Bonferroni factor over challengers is required for one-sided global
    soundness: if the candidate is dominated, choose one actual dominator; a
    false global certificate implies that dominator crossed its own delta-valid
    boundary.
    """
    d=np.asarray(differences,dtype=float)
    if d.ndim!=3 or d.shape[0]<1:
        raise ValueError("differences must be challenger-by-count-by-objective")
    decisions=tuple(evaluate_paired_samples(d[i],delta,method=method,
                          hybrid_hotelling_share=hybrid_hotelling_share)
                    for i in range(d.shape[0]))
    return bool(all(x.crossed for x in decisions)),decisions


def sequential_archive_confirmation(
    differences: np.ndarray, delta: float, *, method: Method = "hybrid",
    hybrid_hotelling_share: float = 0.5, min_count: int | None = None,
    max_count: int | None = None, check_every: int = 1,
) -> dict:
    """Sequential paired confirmation using one-pass Welford statistics.

    The former implementation recomputed means and covariances from every raw
    prefix, which made dense stopping grids quadratic in the number of paired
    scenarios. This implementation updates sufficient statistics once per
    scenario and evaluates only on the declared intrinsic-count grid.
    """
    data = np.asarray(differences, dtype=float)
    if data.ndim != 3 or data.shape[0] < 1 or data.shape[2] < 1:
        raise ValueError("challenger-by-count-by-objective required")
    if not np.all(np.isfinite(data)):
        raise ValueError("differences must be finite")
    challenger_count, total, dimension = data.shape
    if check_every < 1:
        raise ValueError("check_every >= 1 required")
    start = max(3, dimension + 2) if min_count is None else max(2, int(min_count))
    end = total if max_count is None else min(total, int(max_count))
    if end < 2:
        raise ValueError("at least two paired scenarios are required")
    start = min(start, end)
    check_grid = set(range(start, end + 1, check_every))
    check_grid.add(end)

    means = np.zeros((challenger_count, dimension), dtype=float)
    m2 = np.zeros((challenger_count, dimension, dimension), dtype=float)
    last: tuple[PairedDecision, ...] = ()
    checks = 0
    for index in range(end):
        count = index + 1
        observation = data[:, index, :]
        delta_before = observation - means
        means += delta_before / count
        delta_after = observation - means
        m2 += np.einsum("ci,cj->cij", delta_before, delta_after)
        if count not in check_grid:
            continue
        covariance = m2 / (count - 1)
        last = tuple(
            evaluate_paired_statistics(
                means[challenger], covariance[challenger], count, delta,
                method=method,
                hybrid_hotelling_share=hybrid_hotelling_share,
            )
            for challenger in range(challenger_count)
        )
        checks += 1
        if all(decision.crossed for decision in last):
            return {
                "certified": True,
                "stopping_count": count,
                "decisions": last,
                "decision_checks": checks,
                "algorithm_id": "paired_sequential_confirmation",
                "stopping_grid": f"every_{check_every}_paired_counts",
            }
    return {
        "certified": False,
        "stopping_count": end,
        "decisions": last,
        "decision_checks": checks,
        "algorithm_id": "paired_sequential_confirmation",
        "stopping_grid": f"every_{check_every}_paired_counts",
    }

