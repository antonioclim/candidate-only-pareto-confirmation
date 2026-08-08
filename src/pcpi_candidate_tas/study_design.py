"""Study design and development-only hard-archive construction.

The module keeps development evidence separate from held-out confirmation
evidence while constructing a policy pool, selecting hard neighbours and
freezing a reproducible design.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from typing import Iterable

import numpy as np

from .application import (
    BatteryPolicy,
    BatterySystemParameters,
    SelectionWeights,
    application_contract,
    calibrate_application,
    compromise_scores,
    evaluate_archive,
    policy_archive,
    select_compromise_candidate,
)
from .multiseason import ScenarioGeneratorParameters, generate_multiseason_scenarios
from .paired import orthant_mahalanobis_distance, sample_mean_cov


@dataclass(frozen=True)
class SeedPlan:
    archive_design: int = 660_100
    archive_validation_start: int = 661_000
    application_development_start: int = 1_000_000
    application_confirmation_start: int = 2_000_000
    null_calibration_start: int = 3_000_000
    robustness_start: int = 10_000_000
    sensitivity_start: int = 20_000_000
    bootstrap_start: int = 30_000_000

    def validate(self) -> None:
        values = list(asdict(self).values())
        if len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("seed-family roots must be unique non-negative integers")


@dataclass(frozen=True)
class HardArchiveDesign:
    design_scenarios: int = 800
    validation_replications: int = 100
    validation_scenarios: int = 160
    local_step_charge: float = 0.025
    local_step_discharge: float = 0.025
    local_step_reserve: float = 0.025
    local_step_aggressiveness: float = 0.125
    hard_neighbours: int = 13
    minimum_parameter_distance: float = 0.02
    minimum_orthant_effect: float = 0.10
    maximum_orthant_effect: float = 0.50
    maximum_modal_candidate_rate: float = 0.85
    minimum_distinct_candidates: int = 3
    minimum_median_pareto_size: float = 3.0

    def validate(self) -> None:
        if self.design_scenarios < 100 or self.validation_scenarios < 50:
            raise ValueError("substantive development evidence is required")
        if self.validation_replications < 20:
            raise ValueError("too few archive-validation replications")
        if self.hard_neighbours < 1:
            raise ValueError("at least one hard neighbour is required")
        if not 0 < self.maximum_modal_candidate_rate <= 1:
            raise ValueError("invalid modal-candidate threshold")


@dataclass(frozen=True)
class ConfirmatoryDesign:
    application_replications: int = 400
    development_scenarios_per_replication: int = 400
    confirmation_scenarios_per_replication: int = 2_000
    common_sample_cap: int = 2_000
    stopping_grid: int = 5
    delta: float = 0.05
    primary_candidate_method: str = "hybrid"
    primary_comparator: str = "full_psi_coordinate_racing"
    null_replications_per_cell: int = 3_000
    robustness_replications_per_cell: int = 1_000
    sensitivity_replications_per_cell: int = 200
    bootstrap_draws: int = 10_000
    alpha_primary: float = 0.05
    sensitivity_ci_level: float = 0.99

    def validate(self) -> None:
        if self.application_replications < 100:
            raise ValueError("application replication count is too small")
        if self.null_replications_per_cell < 1_000:
            raise ValueError("null calibration needs at least 1000 replications per cell")
        if self.stopping_grid < 1 or self.common_sample_cap < self.stopping_grid:
            raise ValueError("invalid stopping grid or cap")
        if not 0 < self.delta < 1:
            raise ValueError("delta must lie in (0,1)")


@dataclass(frozen=True)
class StudyDesign:
    seed_plan: SeedPlan = SeedPlan()
    archive: HardArchiveDesign = HardArchiveDesign()
    confirmatory: ConfirmatoryDesign = ConfirmatoryDesign()
    source_data_policy: str = "user_supplied_not_redistributed"
    example_data_policy: str = "deterministic_synthetic_fixture"
    primary_estimand: str = (
        "paired mean difference in restricted stopping count at the common cap: "
        "full-set coordinate racing minus candidate-only hybrid"
    )
    secondary_estimands: tuple[str, ...] = (
        "candidate-only completion probability by the common cap",
        "full-set completion probability by the common cap",
        "paired median stopping-count difference",
        "candidate identity distribution and development score margin",
        "empirical held-out archive-relative Pareto rate",
    )
    primary_noninferiority_or_superiority_claim: bool = False

    def validate(self) -> None:
        self.seed_plan.validate()
        self.archive.validate()
        self.confirmatory.validate()
        if self.primary_noninferiority_or_superiority_claim:
            raise ValueError("different answer maps cannot support a superiority label")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "seed_plan": asdict(self.seed_plan),
            "archive": asdict(self.archive),
            "confirmatory": asdict(self.confirmatory),
            "source_data_policy": self.source_data_policy,
            "example_data_policy": self.example_data_policy,
            "primary_estimand": self.primary_estimand,
            "secondary_estimands": list(self.secondary_estimands),
            "primary_noninferiority_or_superiority_claim": False,
        }


def _clip_policy(policy_id: str, values: tuple[float, float, float, float]) -> BatteryPolicy | None:
    charge, discharge, reserve, aggressiveness = values
    charge = float(np.clip(charge, 0.05, 0.55))
    discharge = float(np.clip(discharge, 0.60, 0.95))
    reserve = float(np.clip(reserve, 0.65, 0.98))
    aggressiveness = float(np.clip(aggressiveness, 0.25, 1.00))
    if charge >= discharge:
        return None
    return BatteryPolicy(policy_id, charge, discharge, reserve, aggressiveness)


def local_policy_pool(
    anchor: BatteryPolicy,
    original: Iterable[BatteryPolicy] | None = None,
    design: HardArchiveDesign = HardArchiveDesign(),
) -> tuple[BatteryPolicy, ...]:
    """Construct a deterministic local lattice plus the original pilot archive."""

    design.validate()
    if anchor.aggressiveness <= 0:
        raise ValueError("hard-neighbour anchor must be active")
    original = tuple(original or policy_archive())
    policies: dict[tuple[float, float, float, float], BatteryPolicy] = {}

    for policy in original:
        key = (
            round(policy.charge_quantile, 8),
            round(policy.discharge_quantile, 8),
            round(policy.load_reserve_quantile, 8),
            round(policy.aggressiveness, 8),
        )
        policies[key] = policy

    index = 0
    for dc, dd, dr, da in product((-1, 0, 1), repeat=4):
        values = (
            anchor.charge_quantile + dc * design.local_step_charge,
            anchor.discharge_quantile + dd * design.local_step_discharge,
            anchor.load_reserve_quantile + dr * design.local_step_reserve,
            anchor.aggressiveness + da * design.local_step_aggressiveness,
        )
        policy = _clip_policy(f"local_{index:03d}", values)
        index += 1
        if policy is None:
            continue
        key = tuple(round(value, 8) for value in values)
        policies.setdefault(key, policy)

    for dimension, step in enumerate(
        (
            design.local_step_charge,
            design.local_step_discharge,
            design.local_step_reserve,
            design.local_step_aggressiveness,
        )
    ):
        for sign in (-2, 2):
            values = [
                anchor.charge_quantile,
                anchor.discharge_quantile,
                anchor.load_reserve_quantile,
                anchor.aggressiveness,
            ]
            values[dimension] += sign * step
            policy = _clip_policy(f"axis_{dimension}_{sign:+d}", tuple(values))
            if policy is None:
                continue
            key = tuple(round(value, 8) for value in values)
            policies.setdefault(key, policy)

    output = []
    for key in sorted(policies):
        policy = policies[key]
        if policy.policy_id.startswith(("local_", "axis_")):
            digest = sha256(repr(key).encode("utf-8")).hexdigest()[:10]
            policy = BatteryPolicy(f"hard_{digest}", *key)
        output.append(policy)
    return tuple(output)


def _parameter_vector(policy: BatteryPolicy) -> np.ndarray:
    return np.asarray(
        [
            policy.charge_quantile,
            policy.discharge_quantile,
            policy.load_reserve_quantile,
            policy.aggressiveness,
        ],
        dtype=float,
    )


def _parameter_distance(first: BatteryPolicy, second: BatteryPolicy) -> float:
    lower = np.asarray([0.05, 0.60, 0.65, 0.25])
    upper = np.asarray([0.55, 0.95, 0.98, 1.00])
    return float(
        np.linalg.norm(
            (_parameter_vector(first) - _parameter_vector(second)) / (upper - lower)
        )
    )


def policy_hardness_rows(
    policies: tuple[BatteryPolicy, ...],
    outcomes: np.ndarray,
    candidate_index: int,
) -> list[dict]:
    """Measure paired distance from each challenger mean to the dominance cone."""

    values = np.asarray(outcomes, dtype=float)
    if values.ndim != 3 or values.shape[1] != len(policies):
        raise ValueError("outcomes do not match the policy archive")
    if not 0 <= candidate_index < len(policies):
        raise ValueError("candidate_index is out of range")

    candidate = policies[candidate_index]
    rows: list[dict] = []
    for index, policy in enumerate(policies):
        if index == candidate_index:
            continue
        differences = values[:, index] - values[:, candidate_index]
        mean, covariance = sample_mean_cov(differences)
        distance, rank = orthant_mahalanobis_distance(mean, covariance, len(differences))
        effect = (
            float(np.sqrt(distance / len(differences)))
            if np.isfinite(distance) and distance >= 0
            else float("nan")
        )
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
        rows.append(
            {
                "policy_index": index,
                "policy_id": policy.policy_id,
                "charge_quantile": policy.charge_quantile,
                "discharge_quantile": policy.discharge_quantile,
                "load_reserve_quantile": policy.load_reserve_quantile,
                "aggressiveness": policy.aggressiveness,
                "candidate_policy_id": candidate.policy_id,
                "mean_cost_difference": float(mean[0]),
                "mean_peak_difference": float(mean[1]),
                "mean_throughput_difference": float(mean[2]),
                "maximum_raw_witness": float(np.max(mean)),
                "minimum_raw_difference": float(np.min(mean)),
                "maximum_standardised_witness": float(np.max(mean / standard_deviation)),
                "orthant_effect": effect,
                "covariance_rank": int(rank),
                "tradeoff": bool(np.any(mean < 0) and np.any(mean > 0)),
                "candidate_empirically_safe": bool(np.max(mean) > 0),
                "parameter_distance_from_candidate": _parameter_distance(policy, candidate),
            }
        )
    return rows


def _select_hard_neighbours(
    rows: list[dict],
    policies: tuple[BatteryPolicy, ...],
    design: HardArchiveDesign,
) -> list[int]:
    eligible = [
        row
        for row in rows
        if row["tradeoff"]
        and row["candidate_empirically_safe"]
        and np.isfinite(row["orthant_effect"])
        and design.minimum_orthant_effect
        <= row["orthant_effect"]
        <= design.maximum_orthant_effect
    ]
    eligible.sort(key=lambda row: (row["orthant_effect"], row["policy_id"]))
    selected: list[int] = []
    for row in eligible:
        index = int(row["policy_index"])
        if all(
            _parameter_distance(policies[index], policies[chosen])
            >= design.minimum_parameter_distance
            for chosen in selected
        ):
            selected.append(index)
        if len(selected) == design.hard_neighbours:
            return selected
    for row in eligible:
        index = int(row["policy_index"])
        if index not in selected:
            selected.append(index)
        if len(selected) == design.hard_neighbours:
            return selected
    raise RuntimeError("insufficient eligible hard neighbours")


def construct_hard_archive(
    data: dict,
    *,
    seed_plan: SeedPlan = SeedPlan(),
    design: HardArchiveDesign = HardArchiveDesign(),
    system: BatterySystemParameters = BatterySystemParameters(),
    weights: SelectionWeights = SelectionWeights(),
    generator: ScenarioGeneratorParameters = ScenarioGeneratorParameters(),
) -> dict:
    """Construct one frozen, moderately hard archive from development evidence."""

    seed_plan.validate()
    design.validate()
    original = policy_archive()
    original_calibration = calibrate_application(
        data, original, system, split="development"
    )
    prices, loads, _ = generate_multiseason_scenarios(
        data,
        "development",
        design.design_scenarios,
        seed_plan.archive_design,
        parameters=generator,
    )
    original_outcomes = evaluate_archive(
        original, prices, loads, original_calibration, system
    )
    original_candidate_index = select_compromise_candidate(original_outcomes, weights)
    original_candidate = original[original_candidate_index]

    pool = local_policy_pool(original_candidate, original, design)
    pool_calibration = calibrate_application(data, pool, system, split="development")
    pool_outcomes = evaluate_archive(pool, prices, loads, pool_calibration, system)
    candidate_pool_index = next(
        index
        for index, policy in enumerate(pool)
        if np.allclose(_parameter_vector(policy), _parameter_vector(original_candidate))
    )
    hardness = policy_hardness_rows(pool, pool_outcomes, candidate_pool_index)
    selected = _select_hard_neighbours(hardness, pool, design)
    final_indices = [candidate_pool_index] + selected

    final_policies = []
    for position, pool_index in enumerate(final_indices):
        source = pool[pool_index]
        final_policies.append(
            BatteryPolicy(
                "candidate_anchor_00" if position == 0 else f"hard_{position:02d}",
                source.charge_quantile,
                source.discharge_quantile,
                source.load_reserve_quantile,
                source.aggressiveness,
            )
        )
    final_policies = tuple(final_policies)
    hardness_by_index = {int(row["policy_index"]): row for row in hardness}
    final_rows = []
    for position, pool_index in enumerate(final_indices):
        policy = final_policies[position]
        source = pool[pool_index]
        row = {
            "archive_position": position,
            "policy_id": policy.policy_id,
            "source_policy_id": source.policy_id,
            "role": "candidate_anchor" if position == 0 else "hard_neighbour",
            "charge_quantile": policy.charge_quantile,
            "discharge_quantile": policy.discharge_quantile,
            "load_reserve_quantile": policy.load_reserve_quantile,
            "aggressiveness": policy.aggressiveness,
        }
        if position > 0:
            row.update(
                {
                    key: value
                    for key, value in hardness_by_index[pool_index].items()
                    if key
                    not in {
                        "policy_index",
                        "policy_id",
                        "charge_quantile",
                        "discharge_quantile",
                        "load_reserve_quantile",
                        "aggressiveness",
                    }
                }
            )
        final_rows.append(row)

    return {
        "original_candidate_index": original_candidate_index,
        "original_candidate_id": original_candidate.policy_id,
        "pool_size": len(pool),
        "final_archive_size": len(final_policies),
        "final_policies": final_policies,
        "final_rows": final_rows,
        "hardness_rows": hardness,
        "archive_design_seed": seed_plan.archive_design,
        "design_scenarios": design.design_scenarios,
        "calibration": pool_calibration,
        "system": system,
        "weights": weights,
        "generator": generator,
    }


def validate_hard_archive(
    data: dict,
    policies: tuple[BatteryPolicy, ...],
    *,
    seed_plan: SeedPlan = SeedPlan(),
    design: HardArchiveDesign = HardArchiveDesign(),
    system: BatterySystemParameters = BatterySystemParameters(),
    weights: SelectionWeights = SelectionWeights(),
    generator: ScenarioGeneratorParameters = ScenarioGeneratorParameters(),
) -> dict:
    """Evaluate selection difficulty on independent development-only scenarios."""

    calibration = calibrate_application(data, policies, system, split="development")
    rows = []
    for replication in range(design.validation_replications):
        prices, loads, _ = generate_multiseason_scenarios(
            data,
            "development",
            design.validation_scenarios,
            seed_plan.archive_validation_start + replication,
            parameters=generator,
        )
        outcomes = evaluate_archive(policies, prices, loads, calibration, system)
        scores, front = compromise_scores(outcomes, weights)
        candidate = select_compromise_candidate(outcomes, weights)
        pareto_indices = np.flatnonzero(front)
        ordered = pareto_indices[np.argsort(scores[pareto_indices], kind="stable")]
        runner = int(ordered[1]) if len(ordered) > 1 else None
        rows.append(
            {
                "replication": replication,
                "seed": seed_plan.archive_validation_start + replication,
                "candidate_index": candidate,
                "candidate_policy_id": policies[candidate].policy_id,
                "pareto_set_size": int(front.sum()),
                "candidate_score": float(scores[candidate]),
                "runner_up_index": runner,
                "runner_up_policy_id": None if runner is None else policies[runner].policy_id,
                "score_margin": float("nan")
                if runner is None
                else float(scores[runner] - scores[candidate]),
            }
        )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["candidate_policy_id"]] = counts.get(row["candidate_policy_id"], 0) + 1
    modal_id, modal_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    metrics = {
        "replications": len(rows),
        "candidate_counts": dict(sorted(counts.items())),
        "distinct_candidates": len(counts),
        "modal_candidate": modal_id,
        "modal_candidate_rate": modal_count / len(rows),
        "median_pareto_set_size": float(
            np.median([row["pareto_set_size"] for row in rows])
        ),
        "median_score_margin": float(
            np.nanmedian([row["score_margin"] for row in rows])
        ),
        "minimum_score_margin": float(
            np.nanmin([row["score_margin"] for row in rows])
        ),
    }
    metrics["difficulty_gate_passed"] = bool(
        metrics["modal_candidate_rate"] <= design.maximum_modal_candidate_rate
        and metrics["distinct_candidates"] >= design.minimum_distinct_candidates
        and metrics["median_pareto_set_size"] >= design.minimum_median_pareto_size
    )
    return {"rows": rows, "metrics": metrics}


def binomial_mcse(probability: float, replications: int) -> float:
    if not 0 <= probability <= 1 or replications < 1:
        raise ValueError("invalid probability or replication count")
    return float(np.sqrt(probability * (1 - probability) / replications))


def replication_precision_table(
    design: ConfirmatoryDesign = ConfirmatoryDesign(),
) -> list[dict]:
    design.validate()
    rows = []
    for campaign, replications in (
        ("application", design.application_replications),
        ("null_calibration", design.null_replications_per_cell),
        ("robustness", design.robustness_replications_per_cell),
    ):
        for probability in (0.05, 0.50, 0.90, 0.95):
            mcse = binomial_mcse(probability, replications)
            rows.append(
                {
                    "campaign": campaign,
                    "replications": replications,
                    "probability": probability,
                    "monte_carlo_se": mcse,
                    "normal_approx_95_half_width": 1.96 * mcse,
                }
            )
    return rows
