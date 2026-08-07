"""Vectorised execution kernels with reference-equivalence tests.

The accelerated functions preserve the statistical contracts of the
reference implementation while reducing repeated prefix computation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import f, t

from .application import (
    ApplicationCalibration,
    BatteryPolicy,
    BatterySystemParameters,
)
from .paired import count_mass_log_telescoping


def evaluate_archive_fast(
    policies: tuple[BatteryPolicy, ...],
    prices: np.ndarray,
    loads: np.ndarray,
    calibration: ApplicationCalibration,
    system: BatterySystemParameters = BatterySystemParameters(),
) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    loads = np.asarray(loads, dtype=float)
    if prices.ndim != 2 or loads.shape != prices.shape:
        raise ValueError("prices and loads must be aligned scenario matrices")
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(loads)):
        raise ValueError("finite scenario values are required")
    if np.any(loads <= 0):
        raise ValueError("strictly positive load is required")
    if not policies:
        raise ValueError("at least one policy is required")

    scenario_count, horizon = prices.shape
    policy_count = len(policies)
    site_load = loads * calibration.load_scale_mw_per_system_mwh
    aggressiveness = np.asarray([policy.aggressiveness for policy in policies])
    thresholds = [
        calibration.threshold_for(policy.policy_id) for policy in policies
    ]
    charge_threshold = np.asarray([
        -np.inf if item.charge_price_eur_per_mwh is None
        else item.charge_price_eur_per_mwh
        for item in thresholds
    ])
    discharge_threshold = np.asarray([
        np.inf if item.discharge_price_eur_per_mwh is None
        else item.discharge_price_eur_per_mwh
        for item in thresholds
    ])
    reserve_threshold = np.asarray([
        np.inf if item.reserve_load_mw is None else item.reserve_load_mw
        for item in thresholds
    ])

    minimum_energy = system.minimum_soc_fraction * system.capacity_mwh
    maximum_energy = system.maximum_soc_fraction * system.capacity_mwh
    initial_energy = system.initial_soc_fraction * system.capacity_mwh
    energy = np.full((scenario_count, policy_count), initial_energy)
    operating_cost = np.zeros_like(energy)
    throughput = np.zeros_like(energy)
    peak_import = np.zeros_like(energy)
    dt = system.time_step_hours
    maximum_power = system.power_mw * aggressiveness[None, :]
    active = aggressiveness[None, :] > 0

    for hour in range(horizon):
        current_price = prices[:, hour, None]
        current_load = site_load[:, hour, None]
        charge_condition = (
            active
            & (current_price <= charge_threshold[None, :])
            & (energy < maximum_energy)
        )
        charge_capacity = (
            maximum_energy - energy
        ) / (system.charge_efficiency * dt)
        charge = np.where(
            charge_condition,
            np.minimum(maximum_power, np.maximum(charge_capacity, 0.0)),
            0.0,
        )
        discharge_condition = (
            active
            & ~charge_condition
            & (
                (current_price >= discharge_threshold[None, :])
                | (current_load >= reserve_threshold[None, :])
            )
            & (energy > minimum_energy)
        )
        discharge_capacity = (
            (energy - minimum_energy)
            * system.discharge_efficiency
            / dt
        )
        discharge = np.where(
            discharge_condition,
            np.minimum(
                np.minimum(maximum_power, np.maximum(discharge_capacity, 0.0)),
                current_load,
            ),
            0.0,
        )
        energy = (
            energy
            + system.charge_efficiency * charge * dt
            - discharge * dt / system.discharge_efficiency
        )
        if np.any(energy < minimum_energy - 1e-9) or np.any(
            energy > maximum_energy + 1e-9
        ):
            raise RuntimeError("battery energy left the declared SOC interval")
        grid_import = np.maximum(0.0, current_load + charge - discharge)
        operating_cost += current_price * grid_import * dt
        throughput += (charge + discharge) * dt
        peak_import = np.maximum(peak_import, grid_import)

    terminal_shortfall = np.maximum(0.0, initial_energy - energy)
    terminal_recharge = terminal_shortfall / system.charge_efficiency
    terminal_cost = (
        calibration.terminal_energy_value_eur_per_mwh * terminal_recharge
    )
    return np.stack(
        (
            operating_cost + terminal_cost,
            peak_import,
            throughput + terminal_recharge,
        ),
        axis=2,
    )


def _grid_counts(total, *, min_count, max_count, check_every):
    if check_every < 1:
        raise ValueError("check_every >= 1 required")
    end = total if max_count is None else min(total, int(max_count))
    if end < 2:
        raise ValueError("at least two observations are required")
    start = min(max(2, int(min_count)), end)
    counts = np.arange(start, end + 1, check_every, dtype=int)
    if counts.size == 0 or counts[-1] != end:
        counts = np.append(counts, end)
    return counts


def _orthant_distance_batch(means, covariances, counts):
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    dimension = means.shape[-1]
    flat_means = means.reshape(-1, dimension)
    flat_cov = covariances.reshape(-1, dimension, dimension)
    flat_counts = np.broadcast_to(
        np.asarray(counts, dtype=float), means.shape[:-1]
    ).reshape(-1)
    symmetric = 0.5 * (flat_cov + np.swapaxes(flat_cov, 1, 2))
    eigenvalues = np.linalg.eigvalsh(symmetric)
    tolerance = (
        dimension * np.finfo(float).eps
        * np.maximum(eigenvalues[:, -1], 1.0)
    )
    full_rank = eigenvalues[:, 0] > tolerance
    precision = np.zeros_like(symmetric)
    precision[full_rank] = np.linalg.inv(symmetric[full_rank])
    q = np.einsum("bij,bj->bi", precision, flat_means)
    best = np.full(len(flat_means), np.inf)
    found = np.zeros(len(flat_means), dtype=bool)
    eps = 5e-10
    for mask in range(1 << dimension):
        active = [
            index for index in range(dimension) if mask & (1 << index)
        ]
        inactive = [
            index for index in range(dimension) if not mask & (1 << index)
        ]
        x = np.zeros_like(flat_means)
        feasible = full_rank.copy()
        if active:
            matrix = precision[:, active][:, :, active]
            rhs = -q[:, active]
            try:
                active_x = np.linalg.solve(matrix, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            x[:, active] = active_x
            feasible &= np.all(active_x >= -eps, axis=1)
        gradient = np.einsum(
            "bij,bj->bi", precision, flat_means + x
        )
        if inactive:
            feasible &= np.all(gradient[:, inactive] >= -eps, axis=1)
        residual = flat_means + x
        values = flat_counts * np.einsum(
            "bi,bij,bj->b", residual, precision, residual
        )
        improve = feasible & np.isfinite(values) & (values < best)
        best[improve] = values[improve]
        found |= feasible & np.isfinite(values)
    best[~found] = np.nan
    ranks = np.sum(eigenvalues > tolerance[:, None], axis=1)
    return (
        best.reshape(means.shape[:-1]),
        ranks.reshape(means.shape[:-1]),
    )


def sequential_candidate_fast(
    differences,
    delta,
    *,
    method="hybrid",
    hybrid_hotelling_share=0.5,
    min_count=8,
    max_count=None,
    check_every=5,
):
    data = np.asarray(differences, dtype=float)
    challenger_count, total, dimension = data.shape
    counts = _grid_counts(
        total,
        min_count=max(
            min_count,
            dimension + 2 if method != "coordinate" else 2,
        ),
        max_count=max_count,
        check_every=check_every,
    )
    indices = counts - 1
    sums = np.cumsum(data, axis=1)[:, indices].transpose(1, 0, 2)
    outer = np.einsum("cnm,cnk->cnmk", data, data)
    outer_sums = np.cumsum(outer, axis=1)[:, indices].transpose(1, 0, 2, 3)
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
    coordinate_alpha = (
        alpha
        if method == "coordinate"
        else alpha * (1.0 - hybrid_hotelling_share)
    )
    hotelling_alpha = (
        alpha if method == "hotelling"
        else alpha * hybrid_hotelling_share
    )
    variances = np.diagonal(covariances, axis1=2, axis2=3)
    t_statistics = np.full_like(means, -np.inf)
    good = np.isfinite(variances) & (variances > 0)
    broadcast_counts = np.broadcast_to(count_array, means.shape)
    t_statistics[good] = means[good] / np.sqrt(
        variances[good] / broadcast_counts[good]
    )
    coordinate_max = np.max(t_statistics, axis=2)
    coordinate_threshold = t.isf(
        coordinate_alpha / dimension, counts - 1
    )
    coordinate_crossed = (
        coordinate_max > coordinate_threshold[:, None]
    )
    hotelling_crossed = np.zeros_like(coordinate_crossed)
    if method != "coordinate":
        distances, ranks = _orthant_distance_batch(
            means, covariances, counts[:, None]
        )
        hotelling_threshold = (
            dimension
            * (counts - 1)
            / (counts - dimension)
            * f.isf(
                hotelling_alpha,
                dimension,
                counts - dimension,
            )
        )
        hotelling_crossed = (
            np.isfinite(distances)
            & (ranks == dimension)
            & (distances > hotelling_threshold[:, None])
        )
    if method == "coordinate":
        local_crossed = coordinate_crossed
    elif method == "hotelling":
        local_crossed = hotelling_crossed
    else:
        local_crossed = coordinate_crossed | hotelling_crossed
    global_crossed = np.all(local_crossed, axis=1)
    if np.any(global_crossed):
        grid_index = int(np.flatnonzero(global_crossed)[0])
        certified = True
    else:
        grid_index = len(counts) - 1
        certified = False
    return {
        "certified": certified,
        "stopping_count": int(counts[grid_index]),
        "decision_checks": grid_index + 1,
    }


def sequential_full_psi_coordinate_fast(
    outcomes,
    delta,
    *,
    min_count=12,
    max_count=None,
    check_every=5,
):
    outcomes = np.asarray(outcomes, dtype=float)
    total, arm_count, dimension = outcomes.shape
    counts = _grid_counts(
        total,
        min_count=max(3, min_count),
        max_count=max_count,
        check_every=check_every,
    )
    indices = counts - 1
    pairs = [
        (first, second)
        for first in range(arm_count)
        for second in range(first + 1, arm_count)
    ]
    first_indices = np.asarray([first for first, _ in pairs])
    second_indices = np.asarray([second for _, second in pairs])
    differences = (
        outcomes[:, first_indices] - outcomes[:, second_indices]
    )
    sums = np.cumsum(differences, axis=0)[indices]
    square_sums = np.cumsum(differences * differences, axis=0)[indices]
    count_array = counts[:, None, None].astype(float)
    means = sums / count_array
    variances = (
        square_sums - sums * sums / count_array
    ) / (count_array - 1.0)
    variances = np.maximum(variances, 0.0)
    pair_count = len(pairs)
    masses = np.asarray(
        [count_mass_log_telescoping(int(count)) for count in counts]
    )
    pair_alpha = delta * masses / pair_count
    radii = (
        t.isf(
            pair_alpha / (2.0 * dimension),
            counts - 1,
        )[:, None, None]
        * np.sqrt(variances / count_array)
    )
    pair_lower = means - radii
    pair_upper = means + radii
    grid_count = len(counts)
    lower = np.full(
        (grid_count, arm_count, arm_count, dimension), np.nan
    )
    upper = np.full_like(lower, np.nan)
    lower[:, first_indices, second_indices] = pair_lower
    upper[:, first_indices, second_indices] = pair_upper
    lower[:, second_indices, first_indices] = -pair_upper
    upper[:, second_indices, first_indices] = -pair_lower
    dominated = np.zeros((grid_count, arm_count), dtype=bool)
    nondominated = np.zeros_like(dominated)
    for arm in range(arm_count):
        other = np.arange(arm_count) != arm
        dominated[:, arm] = np.any(
            np.all(upper[:, other, arm] <= 0.0, axis=2), axis=1
        )
        nondominated[:, arm] = np.all(
            np.any(lower[:, other, arm] > 0.0, axis=2), axis=1
        )
    completed = np.all(dominated | nondominated, axis=1)
    if np.any(completed):
        grid_index = int(np.flatnonzero(completed)[0])
        success = True
    else:
        grid_index = grid_count - 1
        success = False
    return {
        "completed": success,
        "stopping_count": int(counts[grid_index]),
        "dominated": dominated[grid_index].copy(),
        "nondominated": nondominated[grid_index].copy(),
        "decision_checks": grid_index + 1,
    }
