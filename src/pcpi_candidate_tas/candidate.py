"""Candidate-only sampling policies and stopping rules.

This module deliberately separates:

* the unit-pull cumulative-target reference policy used for theorem-to-code
  traceability; and
* practical batched policies whose stopping grid, periodic updates, rounding,
  fallback semantics, and finite-precision behaviour are engineering choices.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .boundaries import BoundaryName, SpendingName, evaluate_all_challengers
from .models import CandidateRunResult, GaussianCandidateInstance
from .oracle import solve_scalar_allocation

OracleFailureMode = Literal["uniform_fallback", "fail_closed"]


def _regularised_gaps(means: np.ndarray, floor: float) -> np.ndarray:
    """Ensure that every plug-in challenger has one positive active coordinate."""
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("gap floor must be finite and non-negative")
    gaps = np.asarray(means[1:] - means[0], dtype=float).copy()
    if gaps.ndim != 2 or gaps.shape[0] < 1:
        raise ValueError("candidate and challenger means are required")
    for i in range(gaps.shape[0]):
        if gaps[i].max() <= floor:
            gaps[i, int(np.argmax(gaps[i]))] = floor
    return gaps


def _vanishing_regularisation_floor(
    total: int, scale: float, exponent: float
) -> float:
    """Positive plug-in floor converging to zero."""
    if total < 1:
        raise ValueError("total must be positive")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("regularisation scale must be finite and non-negative")
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("regularisation exponent must be finite and positive")
    return float(scale / (float(total) ** exponent))


def _requested_oracle_tolerances(
    total: int, scale: float, exponent: float, floor: float
) -> tuple[float, float]:
    """Decreasing requested numerical tolerances for the reference.

    The exact tracking theorem concerns an exact-real-arithmetic oracle, or an
    approximation error tending to zero. Floating-point software reaches a
    machine-scale floor and is therefore an approximation to that theorem object.
    """
    if total < 1:
        raise ValueError("total must be positive")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("tolerance scale must be finite and positive")
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("tolerance exponent must be finite and positive")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("tolerance floor must be finite and positive")
    outer = max(float(floor), float(scale) / (float(total) ** exponent))
    inverse = max(float(np.finfo(float).eps * 16.0), outer * 0.01)
    return outer, inverse


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


def _validate_common_run_inputs(
    instance: GaussianCandidateInstance,
    delta: float,
    min_initial_pulls: int,
    max_samples: int,
    trace_every: int,
) -> tuple[int, int, int]:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if min_initial_pulls < 1:
        raise ValueError("min_initial_pulls must be positive")
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    system_count, objective_count = instance.means.shape
    initial_total = system_count * int(min_initial_pulls)
    if max_samples < initial_total:
        raise ValueError("max_samples must cover mandatory initial pulls")
    return system_count, objective_count, initial_total


def _validate_target(target: np.ndarray, system_count: int) -> np.ndarray:
    result = np.asarray(target, dtype=float)
    if result.shape != (system_count,):
        raise ValueError("target must match the system count")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("target weights must be finite and non-negative")
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("target weights must have positive mass")
    return result / total


def _mix_forced_exploration(
    base_target: np.ndarray, forced_exploration: float
) -> np.ndarray:
    """Apply the configured mixture once, without compounding between updates."""
    if not np.isfinite(forced_exploration) or not 0.0 <= forced_exploration <= 1.0:
        raise ValueError("forced_exploration must lie in [0,1]")
    base = _validate_target(base_target, len(base_target))
    projected = (
        (1.0 - forced_exploration) * base
        + forced_exploration / len(base)
    )
    return projected / projected.sum()


def _largest_deficit_batch_allocation(
    counts: np.ndarray, target: np.ndarray, step: int
) -> np.ndarray:
    """Allocate a batch by repeated largest final-count deficit.

    At each unit of the batch, the routine pulls the system with the largest
    deficit relative to ``(total + step) * target``. This avoids the negative-
    deficit rounding bug of a fractional-remainder implementation and is fully
    deterministic under index-order tie breaking.
    """
    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("counts must be a non-negative vector")
    if step < 0:
        raise ValueError("step must be non-negative")
    target = _validate_target(target, counts.size)
    allocation = np.zeros_like(counts)
    desired_final = (int(counts.sum()) + int(step)) * target
    for _ in range(int(step)):
        deficits = desired_final - (counts + allocation)
        allocation[int(np.argmax(deficits))] += 1
    return allocation


def _gap_racing_base_target(
    arrays: tuple[np.ndarray, ...], boundary_method: BoundaryName
) -> np.ndarray:
    crossed, cone_values, cone_thresholds, max_z, coordinate_thresholds, _ = arrays
    if boundary_method == "coordinate":
        margin = coordinate_thresholds - max_z
    elif boundary_method == "hybrid":
        cone_margin = cone_thresholds - cone_values
        coordinate_margin = coordinate_thresholds - max_z
        margin = np.minimum(cone_margin, coordinate_margin)
    else:
        margin = cone_thresholds - cone_values
    unresolved = np.where(crossed, 0.0, np.maximum(margin, 0.0))
    challenger_mass = unresolved + 1e-12
    challenger_mass /= challenger_mass.sum()
    return np.r_[0.5, 0.5 * challenger_mass]


def _batched_algorithm_id(policy: str) -> str:
    return {
        "plugin_track": "batched_plugin_tracking",
        "uniform": "uniform_candidate_only",
        "half_candidate": "half_candidate_baseline",
        "gap_racing": "gap_racing_diagnostic",
        "oracle_static": "oracle_static_diagnostic",
    }[policy]


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
    gap_floor_exponent: float | None = 0.25,
    oracle_update_every: int = 250,
    batch_size: int = 25,
    trace_every: int = 500,
    true_oracle_weights: np.ndarray | None = None,
    oracle_failure_mode: OracleFailureMode = "uniform_fallback",
) -> CandidateRunResult:
    """Run a practical batched candidate-only procedure.

    The policy is evaluated only at batch boundaries. It is an engineering
    implementation and does not inherit the reference tracking theorem. Forced
    exploration is mixed once with the current base target; it does not compound
    while an oracle target is held between updates.
    """
    allowed = {
        "plugin_track", "uniform", "half_candidate",
        "gap_racing", "oracle_static",
    }
    if policy not in allowed:
        raise ValueError(f"unknown policy {policy!r}")
    if batch_size < 1 or oracle_update_every < 1:
        raise ValueError("positive batch and update sizes required")
    if oracle_failure_mode not in {"uniform_fallback", "fail_closed"}:
        raise ValueError("unknown oracle_failure_mode")
    if not np.isfinite(gap_floor) or gap_floor < 0.0:
        raise ValueError("gap_floor must be finite and non-negative")
    if gap_floor_exponent is not None and (
        not np.isfinite(gap_floor_exponent) or gap_floor_exponent <= 0.0
    ):
        raise ValueError("gap_floor_exponent must be positive or None")

    system_count, objective_count, initial_total = _validate_common_run_inputs(
        instance, delta, min_initial_pulls, max_samples, trace_every
    )
    _mix_forced_exploration(
        np.full(system_count, 1.0 / system_count), forced_exploration
    )

    rng = np.random.default_rng(seed)
    counts = np.zeros(system_count, dtype=int)
    sums = np.zeros((system_count, objective_count), dtype=float)
    oracle_failures = 0
    oracle_updates = 0
    decision_checks = 0
    batch_count = 0

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
            else solve_scalar_allocation(instance.gaps, instance.variances).weights
        )
        static_target = _validate_target(static_target, system_count)

    base_target = np.full(system_count, 1.0 / system_count)
    projected_target = _mix_forced_exploration(base_target, forced_exploration)
    next_update = initial_total
    next_trace = initial_total + trace_every if trace_every else None
    trace: list[dict] = []

    while int(counts.sum()) < max_samples:
        total = int(counts.sum())
        means = sums / counts[:, None]
        decisions = evaluate_all_challengers(
            means, counts, instance.variances, delta,
            method=boundary_method, spending=spending,
            exponent=spending_exponent,
            hybrid_cone_share=hybrid_cone_share,
        )
        decision_checks += 1
        arrays = _decision_arrays(decisions)
        if bool(np.all(arrays[0])):
            return CandidateRunResult(
                True, total, counts.copy(), sums.copy(), means.copy(),
                *arrays, oracle_failures, projected_target.copy(),
                boundary_method, spending, tuple(trace),
                _batched_algorithm_id(policy), False, "certified",
                "batch_boundary", oracle_updates, decision_checks, batch_count,
            )

        oracle_updated = False
        current_floor = None
        if static_target is not None:
            base_target = static_target
        elif policy == "gap_racing":
            base_target = _gap_racing_base_target(arrays, boundary_method)
        elif total >= next_update:
            current_floor = (
                gap_floor if gap_floor_exponent is None
                else _vanishing_regularisation_floor(
                    total, gap_floor, gap_floor_exponent
                )
            )
            try:
                base_target = solve_scalar_allocation(
                    _regularised_gaps(means, current_floor),
                    instance.variances,
                    tol=1e-9,
                    inverse_tol=1e-11,
                ).weights
            except Exception as exc:
                oracle_failures += 1
                if oracle_failure_mode == "fail_closed":
                    raise RuntimeError("batched plug-in oracle failed") from exc
                base_target = np.full(system_count, 1.0 / system_count)
            oracle_updated = True
            oracle_updates += 1
            next_update = total + oracle_update_every

        base_target = _validate_target(base_target, system_count)
        projected_target = _mix_forced_exploration(
            base_target, forced_exploration
        )

        step = min(batch_size, max_samples - total)
        allocation = _largest_deficit_batch_allocation(
            counts, projected_target, step
        )
        for system, number in enumerate(allocation):
            pull_many(system, int(number))
        batch_count += 1

        post_total = int(counts.sum())
        if trace_every and next_trace is not None and post_total >= next_trace:
            trace.append({
                "time": post_total,
                "pre_batch_time": total,
                "empirical_weights": (counts / counts.sum()).tolist(),
                "base_target_weights": base_target.tolist(),
                "projected_target_weights": projected_target.tolist(),
                "allocation": allocation.tolist(),
                "resolved_challengers_before_batch": int(np.sum(arrays[0])),
                "oracle_updated": oracle_updated,
                "oracle_failures": oracle_failures,
                "regularisation_floor": current_floor,
                "stopping_grid": "batch_boundary",
            })
            while next_trace <= post_total:
                next_trace += trace_every

    means = sums / counts[:, None]
    arrays = _decision_arrays(evaluate_all_challengers(
        means, counts, instance.variances, delta,
        method=boundary_method, spending=spending,
        exponent=spending_exponent,
        hybrid_cone_share=hybrid_cone_share,
    ))
    decision_checks += 1
    final_certified = bool(np.all(arrays[0]))
    return CandidateRunResult(
        final_certified, int(counts.sum()), counts.copy(), sums.copy(),
        means.copy(), *arrays, oracle_failures, projected_target.copy(),
        boundary_method, spending, tuple(trace),
        _batched_algorithm_id(policy), False,
        "certified_at_cap" if final_certified else "sample_cap",
        "batch_boundary", oracle_updates, decision_checks, batch_count,
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
    gap_floor_exponent: float = 0.25,
    oracle_tolerance_scale: float = 2e-10,
    oracle_tolerance_exponent: float = 0.25,
    oracle_tolerance_floor: float = 1e-13,
    update_every: int = 1,
    trace_every: int = 250,
    fail_on_oracle_error: bool = True,
) -> CandidateRunResult:
    """Run the reference unit-pull cumulative-target reference implementation.

    The plug-in gap floor vanishes, the first oracle update occurs immediately
    after mandatory exploration, and oracle failures are fail-closed by default.
    Requested numerical tolerances decrease to a documented floating-point
    floor. The mathematical target-convergence theorem concerns the exact-real-
    arithmetic policy (or vanishing approximation error); this executable is a
    finite-precision approximation.
    """
    if update_every < 1:
        raise ValueError("update_every must be positive")
    if not np.isfinite(gap_floor) or gap_floor < 0.0:
        raise ValueError("gap_floor must be finite and non-negative")
    system_count, objective_count, initial_total = _validate_common_run_inputs(
        instance, delta, min_initial_pulls, max_samples, trace_every
    )

    rng = np.random.default_rng(seed)
    counts = np.zeros(system_count, dtype=int)
    sums = np.zeros((system_count, objective_count), dtype=float)
    cumulative_target = np.zeros(system_count, dtype=float)
    oracle_failures = 0
    oracle_updates = 0
    decision_checks = 0
    pull_iterations = 0
    base_target = np.full(system_count, 1.0 / system_count)
    projected_target = base_target.copy()
    oracle_outer_tolerance, oracle_inverse_tolerance = _requested_oracle_tolerances(
        initial_total, oracle_tolerance_scale,
        oracle_tolerance_exponent, oracle_tolerance_floor,
    )

    def pull(system: int) -> None:
        y = rng.normal(
            instance.means[system], np.sqrt(instance.variances[system])
        )
        counts[system] += 1
        sums[system] += y

    for system in range(system_count):
        for _ in range(min_initial_pulls):
            pull(system)
    cumulative_target = counts.astype(float).copy()

    next_update = initial_total
    next_trace = initial_total + trace_every if trace_every else None
    trace: list[dict] = []

    while int(counts.sum()) < max_samples:
        total = int(counts.sum())
        means = sums / counts[:, None]
        arrays = _decision_arrays(evaluate_all_challengers(
            means, counts, instance.variances, delta,
            method=boundary_method, spending=spending,
        ))
        decision_checks += 1
        if bool(np.all(arrays[0])):
            return CandidateRunResult(
                True, total, counts.copy(), sums.copy(), means.copy(),
                *arrays, oracle_failures, projected_target.copy(),
                boundary_method, spending, tuple(trace),
                "reference_unit_pull_c_tracking", True, "certified",
                "every_pull", oracle_updates, decision_checks, pull_iterations,
            )

        current_gap_floor = _vanishing_regularisation_floor(
            total, gap_floor, gap_floor_exponent
        )
        oracle_outer_tolerance, oracle_inverse_tolerance = _requested_oracle_tolerances(
            total, oracle_tolerance_scale, oracle_tolerance_exponent,
            oracle_tolerance_floor,
        )
        oracle_updated = False
        if total >= next_update:
            try:
                base_target = solve_scalar_allocation(
                    _regularised_gaps(means, current_gap_floor),
                    instance.variances,
                    tol=oracle_outer_tolerance,
                    inverse_tol=oracle_inverse_tolerance,
                ).weights
            except Exception as exc:
                oracle_failures += 1
                if fail_on_oracle_error:
                    raise RuntimeError(
                        "the theorem-aligned reference oracle failed"
                    ) from exc
                base_target = np.full(system_count, 1.0 / system_count)
            oracle_updated = True
            oracle_updates += 1
            next_update = total + update_every

        epsilon = min(1.0 / system_count, 1.0 / np.sqrt(max(total, 1)))
        projected_target = np.maximum(base_target, epsilon)
        projected_target /= projected_target.sum()
        cumulative_target += projected_target
        pull_system = int(np.argmax(cumulative_target - counts))
        pull(pull_system)
        pull_iterations += 1
        post_total = int(counts.sum())

        if trace_every and next_trace is not None and post_total >= next_trace:
            discrepancy = cumulative_target - counts
            trace.append({
                "time": post_total,
                "pre_pull_time": total,
                "empirical_weights": (counts / counts.sum()).tolist(),
                "base_target_weights": base_target.tolist(),
                "projected_target_weights": projected_target.tolist(),
                "tracking_discrepancy_linf": float(np.max(np.abs(discrepancy))),
                "maximum_positive_deficit": float(np.max(discrepancy)),
                "minimum_deficit": float(np.min(discrepancy)),
                "regularisation_floor": current_gap_floor,
                "oracle_outer_tolerance": oracle_outer_tolerance,
                "oracle_inverse_tolerance": oracle_inverse_tolerance,
                "oracle_updated": oracle_updated,
                "oracle_failures": oracle_failures,
                "pull_system": pull_system,
                "stopping_grid": "every_pull",
            })
            while next_trace <= post_total:
                next_trace += trace_every

    means = sums / counts[:, None]
    arrays = _decision_arrays(evaluate_all_challengers(
        means, counts, instance.variances, delta,
        method=boundary_method, spending=spending,
    ))
    decision_checks += 1
    final_certified = bool(np.all(arrays[0]))
    return CandidateRunResult(
        final_certified, int(counts.sum()), counts.copy(), sums.copy(),
        means.copy(), *arrays, oracle_failures, projected_target.copy(),
        boundary_method, spending, tuple(trace),
        "reference_unit_pull_c_tracking", True,
        "certified_at_cap" if final_certified else "sample_cap",
        "every_pull", oracle_updates, decision_checks, pull_iterations,
    )


def run_reference_c_tracking(*args, **kwargs):
    """Backward-compatible alias for :func:`run_theory_aligned_c_tracking`."""
    return run_theory_aligned_c_tracking(*args, **kwargs)


def run_reference_target_tracking(*args, **kwargs):
    """Alias retained for proof-to-code traceability documents."""
    return run_theory_aligned_c_tracking(*args, **kwargs)
