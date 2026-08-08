"""Transparent, causal battery-policy application model.

The module defines a deliberately stylised representative-site storage model.
It is designed to test candidate-only confirmation, not to reproduce DK1 market
operation or electrochemical battery ageing.

Critical design rules
---------------------
1. Policy thresholds are calibrated from development data only.
2. Confirmation-scenario futures never determine current thresholds.
3. System load is mapped to a representative site by one frozen development
   scaling factor, never by each scenario's future maximum.
4. A terminal energy settlement discourages end-of-horizon energy depletion.
5. Cost, peak import and grid-side throughput remain separate minimisation
   objectives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import numpy as np


OBJECTIVE_NAMES = (
    "net_energy_cost_eur",
    "peak_grid_import_mw",
    "grid_side_throughput_mwh",
)


@dataclass(frozen=True)
class BatteryPolicy:
    """Quantile-threshold policy defined before confirmation."""

    policy_id: str
    charge_quantile: float
    discharge_quantile: float
    load_reserve_quantile: float
    aggressiveness: float

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        for name, value in (
            ("charge_quantile", self.charge_quantile),
            ("discharge_quantile", self.discharge_quantile),
            ("load_reserve_quantile", self.load_reserve_quantile),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if self.aggressiveness < 0.0 or self.aggressiveness > 1.0:
            raise ValueError("aggressiveness must lie in [0,1]")
        if self.aggressiveness > 0 and not self.charge_quantile < self.discharge_quantile:
            raise ValueError("active policies require charge_quantile < discharge_quantile")


@dataclass(frozen=True)
class BatterySystemParameters:
    """Physical and representative-site parameters.

    The values are illustrative engineering settings, not calibrated DK1 asset
    parameters.  Capacity and power are in MWh and MW for a one-hour time step.
    """

    capacity_mwh: float = 10.0
    power_mw: float = 4.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc_fraction: float = 0.50
    minimum_soc_fraction: float = 0.10
    maximum_soc_fraction: float = 0.90
    representative_peak_mw: float = 9.0
    development_load_reference_quantile: float = 0.95
    terminal_energy_value_quantile: float = 0.50
    time_step_hours: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity_mwh <= 0 or self.power_mw <= 0:
            raise ValueError("positive capacity and power are required")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("charge_efficiency must lie in (0,1]")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("discharge_efficiency must lie in (0,1]")
        if not (
            0 <= self.minimum_soc_fraction
            < self.initial_soc_fraction
            < self.maximum_soc_fraction
            <= 1
        ):
            raise ValueError("SOC fractions must satisfy min < initial < max")
        if self.representative_peak_mw <= 0:
            raise ValueError("representative_peak_mw must be positive")
        for name, value in (
            ("development_load_reference_quantile", self.development_load_reference_quantile),
            ("terminal_energy_value_quantile", self.terminal_energy_value_quantile),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must lie in (0,1)")
        if self.time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive")


@dataclass(frozen=True)
class SelectionWeights:
    """Weights for the pre-confirmation compromise selection rule."""

    cost: float = 0.45
    peak: float = 0.40
    throughput: float = 0.15

    def __post_init__(self) -> None:
        values = np.asarray([self.cost, self.peak, self.throughput], dtype=float)
        if np.any(values < 0):
            raise ValueError("selection weights must be non-negative")
        if not np.isclose(float(values.sum()), 1.0, atol=1e-12):
            raise ValueError("selection weights must sum to one")

    def as_array(self) -> np.ndarray:
        return np.asarray([self.cost, self.peak, self.throughput], dtype=float)


@dataclass(frozen=True)
class PolicyThresholds:
    """Frozen development-data thresholds for one active policy."""

    policy_id: str
    charge_price_eur_per_mwh: float | None
    discharge_price_eur_per_mwh: float | None
    reserve_load_mw: float | None


@dataclass(frozen=True)
class ApplicationCalibration:
    """Development-only application calibration."""

    load_reference_system_mwh: float
    load_scale_mw_per_system_mwh: float
    terminal_energy_value_eur_per_mwh: float
    thresholds: tuple[PolicyThresholds, ...]
    calibration_split: str
    development_rows: int

    def threshold_for(self, policy_id: str) -> PolicyThresholds:
        for threshold in self.thresholds:
            if threshold.policy_id == policy_id:
                return threshold
        raise KeyError(policy_id)

    def to_dict(self) -> dict:
        return {
            "load_reference_system_mwh": self.load_reference_system_mwh,
            "load_scale_mw_per_system_mwh": self.load_scale_mw_per_system_mwh,
            "terminal_energy_value_eur_per_mwh": self.terminal_energy_value_eur_per_mwh,
            "thresholds": [asdict(item) for item in self.thresholds],
            "calibration_split": self.calibration_split,
            "development_rows": self.development_rows,
        }


def policy_archive() -> tuple[BatteryPolicy, ...]:
    """Return the frozen 13-policy archive.

    Twelve active policies form a deterministic space-filling design over the
    declared quantile/aggressiveness ranges; the idle policy is a control.
    """

    designs = (
        (0.15, 0.75, 0.90, 0.50),
        (0.15, 0.80, 0.85, 0.75),
        (0.20, 0.75, 0.90, 0.75),
        (0.20, 0.80, 0.85, 1.00),
        (0.25, 0.70, 0.90, 0.50),
        (0.25, 0.75, 0.85, 0.75),
        (0.25, 0.80, 0.80, 1.00),
        (0.30, 0.70, 0.90, 0.75),
        (0.30, 0.75, 0.85, 1.00),
        (0.35, 0.70, 0.85, 0.75),
        (0.35, 0.75, 0.80, 1.00),
        (0.40, 0.70, 0.80, 1.00),
    )
    policies = [BatteryPolicy("policy_00_idle", 0.0, 1.0, 1.0, 0.0)]
    policies.extend(
        BatteryPolicy(f"policy_{index:02d}", *values)
        for index, values in enumerate(designs, start=1)
    )
    return tuple(policies)


def policy_design_diagnostics(policies: Iterable[BatteryPolicy]) -> dict:
    active = [policy for policy in policies if policy.aggressiveness > 0]
    matrix = np.asarray(
        [
            [
                policy.charge_quantile,
                policy.discharge_quantile,
                policy.load_reserve_quantile,
                policy.aggressiveness,
            ]
            for policy in active
        ],
        dtype=float,
    )
    lower = np.asarray([0.15, 0.70, 0.80, 0.50])
    upper = np.asarray([0.40, 0.80, 0.90, 1.00])
    scaled = (matrix - lower) / (upper - lower)
    pairwise = np.sqrt(np.sum((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(pairwise, np.inf)
    return {
        "active_policy_count": len(active),
        "normalised_minimum_pairwise_distance": float(pairwise.min()),
        "normalised_nearest_neighbour_distances": [
            float(value) for value in pairwise.min(axis=1)
        ],
        "parameter_ranges": {
            "charge_quantile": [float(matrix[:, 0].min()), float(matrix[:, 0].max())],
            "discharge_quantile": [float(matrix[:, 1].min()), float(matrix[:, 1].max())],
            "load_reserve_quantile": [float(matrix[:, 2].min()), float(matrix[:, 2].max())],
            "aggressiveness": [float(matrix[:, 3].min()), float(matrix[:, 3].max())],
        },
    }


def calibrate_application(
    data: dict,
    policies: tuple[BatteryPolicy, ...],
    system: BatterySystemParameters = BatterySystemParameters(),
    *,
    split: str = "development",
) -> ApplicationCalibration:
    mask = np.asarray(data["split"]) == split
    prices = np.asarray(data["price"], dtype=float)[mask]
    loads = np.asarray(data["load"], dtype=float)[mask]
    if len(prices) < 2 or len(prices) != len(loads):
        raise ValueError("at least two aligned development observations are required")
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(loads)):
        raise ValueError("finite development price and load are required")
    if np.any(loads <= 0):
        raise ValueError("development load must be positive")

    load_reference = float(
        np.quantile(loads, system.development_load_reference_quantile)
    )
    load_scale = system.representative_peak_mw / load_reference
    site_loads = loads * load_scale
    terminal_value = float(
        np.quantile(prices, system.terminal_energy_value_quantile)
    )

    thresholds: list[PolicyThresholds] = []
    for policy in policies:
        if policy.aggressiveness == 0:
            thresholds.append(PolicyThresholds(policy.policy_id, None, None, None))
            continue
        thresholds.append(
            PolicyThresholds(
                policy_id=policy.policy_id,
                charge_price_eur_per_mwh=float(
                    np.quantile(prices, policy.charge_quantile)
                ),
                discharge_price_eur_per_mwh=float(
                    np.quantile(prices, policy.discharge_quantile)
                ),
                reserve_load_mw=float(
                    np.quantile(site_loads, policy.load_reserve_quantile)
                ),
            )
        )

    return ApplicationCalibration(
        load_reference_system_mwh=load_reference,
        load_scale_mw_per_system_mwh=load_scale,
        terminal_energy_value_eur_per_mwh=terminal_value,
        thresholds=tuple(thresholds),
        calibration_split=split,
        development_rows=len(prices),
    )


def _validate_scenario(price: np.ndarray, load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prices = np.asarray(price, dtype=float)
    loads = np.asarray(load, dtype=float)
    if prices.ndim != 1 or loads.ndim != 1 or prices.shape != loads.shape:
        raise ValueError("price and load must be aligned one-dimensional arrays")
    if len(prices) < 1:
        raise ValueError("non-empty scenarios are required")
    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(loads)):
        raise ValueError("finite scenario values are required")
    if np.any(loads <= 0):
        raise ValueError("strictly positive load is required")
    return prices, loads


def simulate_policy_with_trace(
    policy: BatteryPolicy,
    thresholds: PolicyThresholds,
    price: np.ndarray,
    load: np.ndarray,
    calibration: ApplicationCalibration,
    system: BatterySystemParameters = BatterySystemParameters(),
) -> dict:
    """Simulate one policy and return objectives plus a full dispatch trace."""

    prices, system_load = _validate_scenario(price, load)
    if thresholds.policy_id != policy.policy_id:
        raise ValueError("policy/threshold identity mismatch")

    dt = system.time_step_hours
    site_load = system_load * calibration.load_scale_mw_per_system_mwh

    minimum_energy = system.minimum_soc_fraction * system.capacity_mwh
    maximum_energy = system.maximum_soc_fraction * system.capacity_mwh
    initial_energy = system.initial_soc_fraction * system.capacity_mwh
    energy = initial_energy

    charges = np.zeros_like(prices)
    discharges = np.zeros_like(prices)
    imports = np.zeros_like(prices)
    states = np.empty(len(prices) + 1, dtype=float)
    states[0] = energy

    for hour, (current_price, current_load) in enumerate(zip(prices, site_load)):
        charge = 0.0
        discharge = 0.0
        if policy.aggressiveness > 0:
            assert thresholds.charge_price_eur_per_mwh is not None
            assert thresholds.discharge_price_eur_per_mwh is not None
            assert thresholds.reserve_load_mw is not None
            if (
                current_price <= thresholds.charge_price_eur_per_mwh
                and energy < maximum_energy
            ):
                charge = min(
                    system.power_mw * policy.aggressiveness,
                    (maximum_energy - energy) / (system.charge_efficiency * dt),
                )
            elif (
                current_price >= thresholds.discharge_price_eur_per_mwh
                or current_load >= thresholds.reserve_load_mw
            ) and energy > minimum_energy:
                discharge = min(
                    system.power_mw * policy.aggressiveness,
                    (energy - minimum_energy) * system.discharge_efficiency / dt,
                    current_load,
                )

        energy = (
            energy
            + system.charge_efficiency * charge * dt
            - discharge * dt / system.discharge_efficiency
        )
        if energy < minimum_energy - 1e-10 or energy > maximum_energy + 1e-10:
            raise RuntimeError("battery energy left the declared SOC interval")

        charges[hour] = charge
        discharges[hour] = discharge
        imports[hour] = max(0.0, current_load + charge - discharge)
        states[hour + 1] = energy

    # A fixed development-price settlement removes the incentive to exhaust the
    # battery at the end of the 24-hour horizon.  It is a cost adjustment only;
    # the virtual settlement is not included in peak import.
    terminal_shortfall = max(0.0, initial_energy - energy)
    terminal_recharge_grid = terminal_shortfall / system.charge_efficiency
    terminal_cost = (
        calibration.terminal_energy_value_eur_per_mwh * terminal_recharge_grid
    )

    operating_cost = float(np.dot(prices, imports) * dt)
    total_cost = operating_cost + terminal_cost
    throughput = float(np.sum(charges + discharges) * dt + terminal_recharge_grid)
    peak_import = float(np.max(imports))

    return {
        "objectives": np.asarray([total_cost, peak_import, throughput], dtype=float),
        "price_eur_per_mwh": prices.copy(),
        "source_system_load_mwh": system_load.copy(),
        "site_load_mw": site_load,
        "charge_mw": charges,
        "discharge_mw": discharges,
        "grid_import_mw": imports,
        "energy_state_mwh": states,
        "operating_cost_eur": operating_cost,
        "terminal_settlement_cost_eur": float(terminal_cost),
        "terminal_shortfall_mwh": float(terminal_shortfall),
        "terminal_recharge_grid_mwh": float(terminal_recharge_grid),
        "charge_hours": int(np.count_nonzero(charges > 0)),
        "discharge_hours": int(np.count_nonzero(discharges > 0)),
    }


def simulate_policy(
    policy: BatteryPolicy,
    thresholds: PolicyThresholds,
    price: np.ndarray,
    load: np.ndarray,
    calibration: ApplicationCalibration,
    system: BatterySystemParameters = BatterySystemParameters(),
) -> np.ndarray:
    return simulate_policy_with_trace(
        policy, thresholds, price, load, calibration, system
    )["objectives"]


def evaluate_archive(
    policies: tuple[BatteryPolicy, ...],
    prices: np.ndarray,
    loads: np.ndarray,
    calibration: ApplicationCalibration,
    system: BatterySystemParameters = BatterySystemParameters(),
) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    loads = np.asarray(loads, dtype=float)
    if prices.ndim != 2 or loads.ndim != 2 or prices.shape != loads.shape:
        raise ValueError("prices and loads must be aligned scenario matrices")
    output = np.empty((len(prices), len(policies), len(OBJECTIVE_NAMES)), dtype=float)
    for scenario in range(len(prices)):
        for index, policy in enumerate(policies):
            output[scenario, index] = simulate_policy(
                policy,
                calibration.threshold_for(policy.policy_id),
                prices[scenario],
                loads[scenario],
                calibration,
                system,
            )
    return output


def pareto_mask(means: np.ndarray) -> np.ndarray:
    values = np.asarray(means, dtype=float)
    mask = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        for challenger in range(len(values)):
            if (
                index != challenger
                and np.all(values[challenger] <= values[index])
                and np.any(values[challenger] < values[index])
            ):
                mask[index] = False
                break
    return mask


def compromise_scores(
    outcomes: np.ndarray,
    weights: SelectionWeights = SelectionWeights(),
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(outcomes, dtype=float).mean(axis=0)
    mask = pareto_mask(means)
    lower = means.min(axis=0)
    upper = means.max(axis=0)
    scaled = (means - lower) / np.maximum(upper - lower, 1e-12)
    return scaled @ weights.as_array(), mask


def select_compromise_candidate(
    outcomes: np.ndarray,
    weights: SelectionWeights = SelectionWeights(),
) -> int:
    scores, mask = compromise_scores(outcomes, weights)
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError("empty empirical Pareto set")
    return int(indices[np.argmin(scores[indices])])


def candidate_differences(
    outcomes: np.ndarray,
    candidate: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if candidate < 0 or candidate >= outcomes.shape[1]:
        raise ValueError("candidate index out of range")
    challengers = tuple(index for index in range(outcomes.shape[1]) if index != candidate)
    return (
        np.stack(
            [outcomes[:, index] - outcomes[:, candidate] for index in challengers]
        ),
        challengers,
    )


def application_contract(
    system: BatterySystemParameters,
    weights: SelectionWeights,
    calibration: ApplicationCalibration,
) -> dict:
    return {
        "application_status": "stylised representative-site simulation model",
        "objective_orientation": "minimisation",
        "objective_names": list(OBJECTIVE_NAMES),
        "system_parameters": asdict(system),
        "selection_weights": asdict(weights),
        "calibration": calibration.to_dict(),
        "causality_contract": (
            "all thresholds and scaling factors are frozen from development data "
            "before confirmation; current actions use current price/load, current "
            "energy and frozen thresholds only"
        ),
        "terminal_contract": (
            "a development-price terminal settlement penalises energy shortfall; "
            "the virtual settlement is excluded from peak import"
        ),
        "non_claims": [
            "not a DK1 market-dispatch model",
            "not an electrochemical degradation model",
            "not a validated annual battery-performance model",
            "not a causal forecast or bidding optimiser",
        ],
    }
