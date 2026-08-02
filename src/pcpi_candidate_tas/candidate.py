"""Candidate-only sampling policies and stopping rules."""
from __future__ import annotations
import numpy as np

from .boundaries import BoundaryName, SpendingName, evaluate_all_challengers
from .models import CandidateRunResult, GaussianCandidateInstance
from .oracle import solve_scalar_allocation


def _regularised_gaps(means: np.ndarray, floor: float) -> np.ndarray:
    gaps = np.asarray(means[1:] - means[0], dtype=float).copy()
    for i in range(gaps.shape[0]):
        if gaps[i].max() <= floor:
            gaps[i, int(np.argmax(gaps[i]))] = floor
    return gaps


def _decision_arrays(decisions):
    return (
        np.array([d.crossed for d in decisions], dtype=bool),
        np.array([d.cone_glr for d in decisions], dtype=float),
        np.array([
            np.nan if d.cone_threshold is None else d.cone_threshold
            for d in decisions
        ], dtype=float),
        np.array([d.max_z for d in decisions], dtype=float),
        np.array([
            np.nan if d.coordinate_threshold is None else d.coordinate_threshold
            for d in decisions
        ], dtype=float),
        np.array([d.witness_objective for d in decisions], dtype=int),
    )


def run_candidate_certification(
    instance: GaussianCandidateInstance,
    *,
    delta: float = 0.05,
    policy: str = "plugin_track",
    boundary_method: BoundaryName = "chi_bar",
    spending: SpendingName = "log_telescoping",
    spending_exponent: float = 2.0,
    hybrid_cone_share: float = 0.5,
    seed: int = 0,
    min_initial_pulls: int = 2,
    max_samples: int = 200_000,
    forced_exploration: float = 0.03,
    gap_floor: float = 1e-3,
    oracle_update_every: int = 250,
    batch_size: int = 25,
    trace_every: int = 500,
    true_oracle_weights: np.ndarray | None = None,
) -> CandidateRunResult:
    """Run the practical batched candidate-only procedure.

    The implementation is deliberately distinguished from the idealised
    C-tracking policy used in the inherited expected-time theorem.
    """

    allowed = {
        "plugin_track", "uniform", "half_candidate",
        "gap_racing", "oracle_static",
    }
    if policy not in allowed:
        raise ValueError(f"unknown policy {policy!r}")
    if batch_size < 1 or oracle_update_every < 1:
        raise ValueError("positive batch and update sizes required")

    rng = np.random.default_rng(seed)
    system_count, objective_count = instance.means.shape
    counts = np.zeros(system_count, dtype=int)
    sums = np.zeros((system_count, objective_count), dtype=float)
    oracle_failures = 0

    def pull_many(system: int, number: int) -> None:
        if number <= 0:
            return
        observations = rng.normal(
            instance.means[system],
            np.sqrt(instance.variances[system]),
            size=(number, objective_count),
        )
        counts[system] += number
        sums[system] += observations.sum(axis=0)

    for system in range(system_count):
        pull_many(system, min_initial_pulls)

    static_target = None
    if policy == "uniform":
        static_target = np.full(system_count, 1.0 / system_count)
    elif policy == "half_candidate":
        static_target = np.r_[
            0.5, np.full(system_count - 1, 0.5 / (system_count - 1))
        ]
    elif policy == "oracle_static":
        static_target = (
            np.asarray(true_oracle_weights, dtype=float)
            if true_oracle_weights is not None
            else solve_scalar_allocation(
                instance.gaps, instance.variances
            ).weights
        )

    target = np.full(system_count, 1.0 / system_count)
    next_update = int(counts.sum())
    next_trace = int(counts.sum()) + trace_every
    trace = []

    while int(counts.sum()) < max_samples:
        total = int(counts.sum())
        means = sums / counts[:, None]
        decisions = evaluate_all_challengers(
            means, counts, instance.variances, delta,
            method=boundary_method, spending=spending,
            exponent=spending_exponent,
            hybrid_cone_share=hybrid_cone_share,
        )
        arrays = _decision_arrays(decisions)
        if bool(np.all(arrays[0])):
            return CandidateRunResult(
                True, total, counts.copy(), sums.copy(), means.copy(),
                *arrays, oracle_failures, target.copy(),
                boundary_method, spending, tuple(trace)
            )

        if static_target is not None:
            target = static_target
        elif policy == "gap_racing":
            margin = (
                arrays[4] - arrays[3]
                if boundary_method == "coordinate"
                else arrays[2] - arrays[1]
            )
            deficit = np.maximum(margin, 0.0) + 1e-9
            deficit /= deficit.sum()
            target = np.r_[0.5, 0.5 * deficit]
        elif total >= next_update:
            try:
                target = solve_scalar_allocation(
                    _regularised_gaps(means, gap_floor),
                    instance.variances,
                    tol=1e-9,
                    inverse_tol=1e-11,
                ).weights
            except Exception:
                oracle_failures += 1
                target = np.full(system_count, 1.0 / system_count)
            next_update = total + oracle_update_every

        target = (
            (1.0 - forced_exploration) * target
            + forced_exploration / system_count
        )
        target /= target.sum()

        step = min(batch_size, max_samples - total)
        desired = (total + step) * target - counts
        allocation = np.floor(np.maximum(desired, 0.0)).astype(int)
        if allocation.sum() > step:
            surplus = int(allocation.sum() - step)
            for system in np.argsort(desired):
                take = min(int(allocation[system]), surplus)
                allocation[system] -= take
                surplus -= take
                if surplus == 0:
                    break
        remainder = int(step - allocation.sum())
        order = np.argsort(-(desired - np.floor(desired)))
        for index in range(remainder):
            allocation[int(order[index % system_count])] += 1
        for system, number in enumerate(allocation):
            pull_many(system, int(number))

        if trace_every and total >= next_trace:
            trace.append({
                "time": total,
                "empirical_weights": (counts / counts.sum()).tolist(),
                "target_weights": target.tolist(),
                "resolved_challengers": int(np.sum(arrays[0])),
            })
            next_trace += trace_every

    means = sums / counts[:, None]
    arrays = _decision_arrays(evaluate_all_challengers(
        means, counts, instance.variances, delta,
        method=boundary_method, spending=spending,
        exponent=spending_exponent,
        hybrid_cone_share=hybrid_cone_share,
    ))
    return CandidateRunResult(
        False, int(counts.sum()), counts.copy(), sums.copy(), means.copy(),
        *arrays, oracle_failures, target.copy(),
        boundary_method, spending, tuple(trace)
    )


def run_theory_aligned_c_tracking(
    instance: GaussianCandidateInstance,
    *,
    delta: float = 0.05,
    boundary_method: BoundaryName = "chi_bar",
    spending: SpendingName = "log_telescoping",
    seed: int = 0,
    min_initial_pulls: int = 2,
    max_samples: int = 100_000,
    gap_floor: float = 1e-3,
    update_every: int = 1,
    trace_every: int = 250,
) -> CandidateRunResult:
    """Unit-pull cumulative-target implementation aligned with the theorem.

    Initial mandatory pulls are included in the cumulative target. At each
    subsequent step the implementation adds the projected oracle allocation and
    samples the system with the largest cumulative deficit. The result supports
    pathwise target convergence and almost-sure termination only on strictly
    positive instances. It is not labelled globally delta-PAC and is not assigned
    an expected-time first-order optimality guarantee. The practical batched
    policy remains a separate engineering approximation.
    """

    rng = np.random.default_rng(seed)
    system_count, objective_count = instance.means.shape
    counts = np.zeros(system_count, dtype=int)
    sums = np.zeros((system_count, objective_count), dtype=float)
    cumulative_target = np.zeros(system_count, dtype=float)
    oracle_failures = 0
    target = np.full(system_count, 1.0 / system_count)

    def pull(system: int) -> None:
        y = rng.normal(
            instance.means[system], np.sqrt(instance.variances[system])
        )
        counts[system] += 1
        sums[system] += y

    for system in range(system_count):
        for _ in range(min_initial_pulls):
            pull(system)
    # Treat mandatory exploration as part of the cumulative target so that the
    # tracking discrepancy has a transparent pathwise interpretation.
    cumulative_target = counts.astype(float).copy()

    trace = []
    while int(counts.sum()) < max_samples:
        total = int(counts.sum())
        means = sums / counts[:, None]
        arrays = _decision_arrays(evaluate_all_challengers(
            means, counts, instance.variances, delta,
            method=boundary_method, spending=spending,
        ))
        if bool(np.all(arrays[0])):
            return CandidateRunResult(
                True, total, counts.copy(), sums.copy(), means.copy(),
                *arrays, oracle_failures, target.copy(),
                boundary_method, spending, tuple(trace)
            )

        if total % update_every == 0:
            try:
                target = solve_scalar_allocation(
                    _regularised_gaps(means, gap_floor),
                    instance.variances,
                ).weights
            except Exception:
                oracle_failures += 1
                target = np.full(system_count, 1.0 / system_count)

        epsilon = min(1.0 / system_count, 1.0 / np.sqrt(max(total, 1)))
        projected = np.maximum(target, epsilon)
        projected /= projected.sum()
        cumulative_target += projected
        pull(int(np.argmax(cumulative_target - counts)))

        if trace_every and total % trace_every == 0:
            discrepancy = cumulative_target - counts
            trace.append({
                "time": total,
                "empirical_weights": (counts / counts.sum()).tolist(),
                "target_weights": projected.tolist(),
                "tracking_discrepancy_linf": float(np.max(np.abs(discrepancy))),
                "maximum_positive_deficit": float(np.max(discrepancy)),
            })

    means = sums / counts[:, None]
    arrays = _decision_arrays(evaluate_all_challengers(
        means, counts, instance.variances, delta,
        method=boundary_method, spending=spending,
    ))
    return CandidateRunResult(
        False, int(counts.sum()), counts.copy(), sums.copy(), means.copy(),
        *arrays, oracle_failures, target.copy(),
        boundary_method, spending, tuple(trace)
    )


def run_reference_c_tracking(*args, **kwargs):
    """Backward-compatible alias for :func:`run_theory_aligned_c_tracking`."""
    return run_theory_aligned_c_tracking(*args, **kwargs)
