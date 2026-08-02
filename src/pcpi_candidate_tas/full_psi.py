"""A conservative full-Pareto confidence-racing comparator.

The method is a valid racing baseline, not a state-of-the-art replacement for
MO-SCORE or modern Pareto-set-identification algorithms.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

from .boundaries import count_mass
from .models import FullPSIRunResult, GaussianCandidateInstance


def mean_confidence_bounds(
    means: np.ndarray,
    counts: np.ndarray,
    variances: np.ndarray,
    delta: float,
    *,
    spending: str = "log_telescoping",
    exponent: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=int)
    variances = np.asarray(variances, dtype=float)
    system_count, objective_count = means.shape
    lower = np.empty_like(means)
    upper = np.empty_like(means)
    for system in range(system_count):
        mass = count_mass(
            int(counts[system]), spending=spending, exponent=exponent
        )
        alpha = delta * mass / (system_count * objective_count)
        multiplier = float(norm.isf(alpha / 2.0))
        radius = multiplier * np.sqrt(
            variances[system] / counts[system]
        )
        lower[system] = means[system] - radius
        upper[system] = means[system] + radius
    return lower, upper


def classify_pareto_intervals(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    system_count = lower.shape[0]
    dominated = np.zeros(system_count, dtype=bool)
    nondominated = np.zeros(system_count, dtype=bool)
    for system in range(system_count):
        for competitor in range(system_count):
            if competitor == system:
                continue
            if np.all(upper[competitor] <= lower[system]):
                dominated[system] = True
                break
        if dominated[system]:
            continue
        nondominated[system] = all(
            competitor == system
            or np.any(lower[competitor] > upper[system])
            for competitor in range(system_count)
        )
    return dominated, nondominated


def run_full_psi_confidence_racing(
    instance: GaussianCandidateInstance,
    *,
    delta: float = 0.05,
    seed: int = 0,
    min_initial_pulls: int = 2,
    max_samples: int = 500_000,
    batch_per_active_system: int = 10,
    spending: str = "log_telescoping",
    trace_every: int = 1000,
) -> FullPSIRunResult:
    rng = np.random.default_rng(seed)
    system_count, objective_count = instance.means.shape
    counts = np.zeros(system_count, dtype=int)
    sums = np.zeros((system_count, objective_count), dtype=float)

    def pull_many(system: int, number: int) -> None:
        y = rng.normal(
            instance.means[system],
            np.sqrt(instance.variances[system]),
            size=(number, objective_count),
        )
        counts[system] += number
        sums[system] += y.sum(axis=0)

    for system in range(system_count):
        pull_many(system, min_initial_pulls)

    trace = []
    next_trace = int(counts.sum()) + trace_every
    lower = np.full_like(instance.means, -np.inf)
    upper = np.full_like(instance.means, np.inf)
    dominated = np.zeros(system_count, dtype=bool)
    nondominated = np.zeros(system_count, dtype=bool)

    while int(counts.sum()) < max_samples:
        means = sums / counts[:, None]
        lower, upper = mean_confidence_bounds(
            means, counts, instance.variances, delta, spending=spending
        )
        dominated, nondominated = classify_pareto_intervals(lower, upper)
        if bool(np.all(dominated | nondominated)):
            return FullPSIRunResult(
                True, int(counts.sum()), counts.copy(), sums.copy(),
                means.copy(), lower.copy(), upper.copy(),
                dominated.copy(), nondominated.copy(),
                tuple(np.flatnonzero(nondominated).tolist()),
                tuple(trace),
            )

        active = ~(dominated | nondominated)
        systems = set(np.flatnonzero(active).tolist())
        systems.add(int(np.argmax(np.max(upper - lower, axis=1))))
        remaining = max_samples - int(counts.sum())
        for system in sorted(systems):
            number = min(batch_per_active_system, remaining)
            if number <= 0:
                break
            pull_many(system, number)
            remaining -= number

        if trace_every and int(counts.sum()) >= next_trace:
            trace.append({
                "time": int(counts.sum()),
                "dominated": int(dominated.sum()),
                "nondominated": int(nondominated.sum()),
                "unclassified": int(
                    system_count - np.sum(dominated | nondominated)
                ),
            })
            next_trace += trace_every

    means = sums / counts[:, None]
    lower, upper = mean_confidence_bounds(
        means, counts, instance.variances, delta, spending=spending
    )
    dominated, nondominated = classify_pareto_intervals(lower, upper)
    return FullPSIRunResult(
        False, int(counts.sum()), counts.copy(), sums.copy(),
        means.copy(), lower.copy(), upper.copy(),
        dominated.copy(), nondominated.copy(),
        tuple(np.flatnonzero(nondominated).tolist()),
        tuple(trace),
    )
