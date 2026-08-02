"""Independent small-instance iMO-SCORE reimplementation.

This module is an independently written, paper-derived implementation for
independent Gaussian objectives. It is not official author code and is never
reported as an official MO-SCORE/iMO-SCORE executable.

The implementation deliberately enumerates all assignment-induced phantoms and
retains all MCE and MCI constraints. That makes it suitable for validation on
small and moderate benchmark instances, rather than for the thousands-of-arms
regime targeted by the published efficient approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .psi_comparators import pareto_mask


@dataclass(frozen=True)
class Phantom:
    values: np.ndarray
    contributors: np.ndarray
    assignment: tuple[int, ...]


@dataclass(frozen=True)
class IMOSCOREAllocation:
    weights: np.ndarray
    rate: float
    pareto_indices: np.ndarray
    nonpareto_indices: np.ndarray
    nonpareto_scores: np.ndarray
    nonpareto_shares: np.ndarray
    phantom_values: np.ndarray
    phantom_contributors: np.ndarray
    success: bool
    message: str
    iterations: int
    minimum_constraint_residual: float
    implementation_label: str = "independent_imoscore_full_constraint"


def _validate(means: np.ndarray, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(means, dtype=float)
    var = np.asarray(variances, dtype=float)
    if mu.ndim != 2 or var.shape != mu.shape:
        raise ValueError("means and variances must be same-shape matrices")
    if len(mu) < 2 or mu.shape[1] < 2:
        raise ValueError("at least two systems and two objectives are required")
    if not np.all(np.isfinite(mu)):
        raise ValueError("means must be finite")
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        raise ValueError("variances must be finite and strictly positive")
    return mu, var


def enumerate_assignment_phantoms(
    means: np.ndarray,
    pareto_indices: Iterable[int] | None = None,
    *,
    maximum_assignments: int = 250_000,
) -> list[Phantom]:
    """Enumerate assignment-induced phantom points for minimisation.

    Every Pareto system is assigned to one objective on which a potential
    falsely included system must beat it. The phantom coordinate is the best
    (smallest) coordinate among systems assigned to that objective; an empty
    coordinate is +infinity. Duplicate phantoms are removed deterministically.
    """
    mu = np.asarray(means, dtype=float)
    if pareto_indices is None:
        pareto = np.flatnonzero(pareto_mask(mu))
    else:
        pareto = np.asarray(tuple(pareto_indices), dtype=int)
    if len(pareto) == 0:
        raise ValueError("at least one Pareto system is required")
    d = mu.shape[1]
    n_assignments = d ** len(pareto)
    if n_assignments > maximum_assignments:
        raise ValueError(
            f"phantom enumeration would require {n_assignments} assignments; "
            f"limit is {maximum_assignments}"
        )

    unique: dict[tuple[float, ...], Phantom] = {}
    for assignment in product(range(d), repeat=len(pareto)):
        values = np.full(d, np.inf, dtype=float)
        contributors = np.full(d, -1, dtype=int)
        for objective in range(d):
            members = [
                pareto[pos]
                for pos, assigned_objective in enumerate(assignment)
                if assigned_objective == objective
            ]
            if members:
                member_values = mu[members, objective]
                local = int(np.argmin(member_values))
                values[objective] = float(member_values[local])
                contributors[objective] = int(members[local])
        key = tuple(float(x) for x in values)
        unique.setdefault(
            key,
            Phantom(
                values=values.copy(),
                contributors=contributors.copy(),
                assignment=tuple(int(x) for x in assignment),
            ),
        )
    return [unique[key] for key in sorted(unique, key=lambda x: tuple(np.nan_to_num(x, posinf=1e300)))]


def independent_nonpareto_scores(
    means: np.ndarray,
    variances: np.ndarray,
    nonpareto_indices: Iterable[int],
    phantoms: list[Phantom],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute independent-arm phantom scores and minimising phantom indices."""
    mu, var = _validate(means, variances)
    nonpareto = np.asarray(tuple(nonpareto_indices), dtype=int)
    if len(nonpareto) == 0:
        return np.empty(0), np.empty(0, dtype=int)
    scores = np.empty(len(nonpareto), dtype=float)
    minimisers = np.empty(len(nonpareto), dtype=int)
    for q, arm in enumerate(nonpareto):
        values = []
        for phantom in phantoms:
            finite = np.isfinite(phantom.values)
            gaps = np.maximum(mu[arm, finite] - phantom.values[finite], 0.0)
            values.append(float(0.5 * np.sum(gaps * gaps / var[arm, finite])))
        minimisers[q] = int(np.argmin(values))
        scores[q] = float(values[minimisers[q]])
    if np.any(scores <= 0) or not np.all(np.isfinite(scores)):
        raise ValueError(
            "non-Pareto independent scores must be finite and positive; "
            "the estimated Pareto partition may be numerically degenerate"
        )
    return scores, minimisers


def _mce_rate(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    dominated_pareto: int,
    putative_dominator: int,
) -> float:
    """Rate for removing one Pareto system through another Pareto system."""
    gaps = means[putative_dominator] - means[dominated_pareto]
    active = gaps > 0
    if not np.any(active):
        return 0.0
    denominator = (
        variances[putative_dominator, active] / weights[putative_dominator]
        + variances[dominated_pareto, active] / weights[dominated_pareto]
    )
    return float(0.5 * np.sum(gaps[active] ** 2 / denominator))


def _mci_rate(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    arm: int,
    phantom: Phantom,
) -> float:
    """Rate for falsely including a dominated system through a phantom."""
    finite = np.isfinite(phantom.values)
    active = finite & (means[arm] > phantom.values)
    if not np.any(active):
        return 0.0
    contributors = phantom.contributors[active]
    denominator = (
        variances[arm, active] / weights[arm]
        + variances[contributors, np.flatnonzero(active)] / weights[contributors]
    )
    gaps = means[arm, active] - phantom.values[active]
    return float(0.5 * np.sum(gaps * gaps / denominator))


def all_constraint_rates(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    pareto_indices: np.ndarray,
    nonpareto_indices: np.ndarray,
    phantoms: list[Phantom],
) -> np.ndarray:
    rates: list[float] = []
    for dominated in pareto_indices:
        for putative_dominator in pareto_indices:
            if dominated != putative_dominator:
                rates.append(
                    _mce_rate(
                        means,
                        variances,
                        weights,
                        int(dominated),
                        int(putative_dominator),
                    )
                )
    for arm in nonpareto_indices:
        for phantom in phantoms:
            rates.append(
                _mci_rate(means, variances, weights, int(arm), phantom)
            )
    return np.asarray(rates, dtype=float)


def solve_independent_imoscore(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    epsilon: float = 1e-7,
    maximum_assignments: int = 250_000,
    random_starts: int = 4,
    seed: int = 0,
) -> IMOSCOREAllocation:
    """Solve the paper-derived full-constraint independent-score allocation.

    Non-Pareto allocations are tied in inverse proportion to their independent
    phantom scores. Pareto allocations and the common minimum rate are then
    optimised subject to all pairwise MCE and all phantom MCI constraints.
    """
    mu, var = _validate(means, variances)
    k = len(mu)
    pareto = np.flatnonzero(pareto_mask(mu))
    nonpareto = np.flatnonzero(~pareto_mask(mu))
    phantoms = enumerate_assignment_phantoms(
        mu, pareto, maximum_assignments=maximum_assignments
    )
    scores, _ = independent_nonpareto_scores(mu, var, nonpareto, phantoms)
    if len(nonpareto):
        inverse = 1.0 / scores
        shares = inverse / inverse.sum()
    else:
        shares = np.empty(0)

    p = len(pareto)
    rng = np.random.default_rng(seed)

    def expand(alpha_p: np.ndarray) -> np.ndarray:
        weights = np.zeros(k, dtype=float)
        weights[pareto] = alpha_p
        remaining = 1.0 - float(alpha_p.sum())
        if len(nonpareto):
            weights[nonpareto] = remaining * shares
        return weights

    def objective(x: np.ndarray) -> float:
        return -float(x[-1])

    def feasibility(x: np.ndarray) -> float:
        alpha_p = x[:-1]
        if len(nonpareto):
            return 1.0 - epsilon - float(alpha_p.sum())
        return epsilon - abs(1.0 - float(alpha_p.sum()))

    def equality_sum(x: np.ndarray) -> float:
        return float(x[:-1].sum() - 1.0)

    def rate_constraints(x: np.ndarray) -> np.ndarray:
        weights = expand(x[:-1])
        if np.any(weights <= 0):
            return np.full(len(initial_rates), -1e6, dtype=float)
        values = all_constraint_rates(
            mu, var, weights, pareto, nonpareto, phantoms
        )
        return values - float(x[-1])

    equal = np.full(k, 1.0 / k)
    alpha_equal = equal[pareto]
    initial_rates = all_constraint_rates(
        mu, var, equal, pareto, nonpareto, phantoms
    )
    if len(initial_rates) == 0:
        raise ValueError("at least one alternative constraint is required")
    z0 = max(float(np.min(initial_rates)) * 0.7, epsilon)

    constraints: list[dict] = []
    if len(nonpareto):
        constraints.append({"type": "ineq", "fun": feasibility})
    else:
        constraints.append({"type": "eq", "fun": equality_sum})
    constraints.append({"type": "ineq", "fun": rate_constraints})

    bounds = [(epsilon, 1.0 - epsilon)] * p + [(0.0, None)]
    starts = [np.r_[alpha_equal, z0]]
    for _ in range(max(0, random_starts)):
        if len(nonpareto):
            mass = float(rng.uniform(0.25, 0.90))
            alpha = rng.dirichlet(np.ones(p)) * mass
        else:
            alpha = rng.dirichlet(np.ones(p))
        weights = expand(alpha)
        rate = float(
            np.min(
                all_constraint_rates(
                    mu, var, weights, pareto, nonpareto, phantoms
                )
            )
        )
        starts.append(np.r_[alpha, max(rate * 0.7, epsilon)])

    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-11, "maxiter": 4000, "disp": False},
        )
        if best is None or (
            result.success and (not best.success or result.x[-1] > best.x[-1])
        ):
            best = result
        elif best is not None and not best.success and result.fun < best.fun:
            best = result
    assert best is not None

    weights = expand(best.x[:-1])
    rates = all_constraint_rates(mu, var, weights, pareto, nonpareto, phantoms)
    minimum_rate = float(np.min(rates))
    residual = float(minimum_rate - best.x[-1])
    return IMOSCOREAllocation(
        weights=weights,
        rate=minimum_rate,
        pareto_indices=pareto,
        nonpareto_indices=nonpareto,
        nonpareto_scores=scores,
        nonpareto_shares=shares,
        phantom_values=np.asarray([p.values for p in phantoms], dtype=float),
        phantom_contributors=np.asarray(
            [p.contributors for p in phantoms], dtype=int
        ),
        success=bool(best.success and residual >= -1e-7),
        message=str(best.message),
        iterations=int(best.nit),
        minimum_constraint_residual=residual,
    )


def integer_allocation(
    weights: np.ndarray,
    budget: int,
    *,
    minimum_per_system: int = 2,
) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if budget < minimum_per_system * len(w):
        raise ValueError("budget is smaller than the mandatory initial allocation")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be a probability vector")
    remaining = budget - minimum_per_system * len(w)
    raw = remaining * w
    extra = np.floor(raw).astype(int)
    counts = extra + minimum_per_system
    missing = budget - int(counts.sum())
    order = np.argsort(-(raw - extra))
    counts[order[:missing]] += 1
    return counts
