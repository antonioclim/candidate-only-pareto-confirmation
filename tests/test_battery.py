from pathlib import Path
import numpy as np
import pytest

from pcpi_candidate_tas.battery import (
    BatterySystemParameters,
    SelectionWeights,
    calibrate_application,
    candidate_differences,
    evaluate_archive,
    load_opsd_fixture,
    pareto_mask,
    policy_archive,
    policy_design_diagnostics,
    select_compromise_candidate,
    simulate_policy_with_trace,
)
from pcpi_candidate_tas.multiseason import load_multiseason_subset

ROOT = Path(__file__).resolve().parents[1]
MULTISEASON = ROOT / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"


def _calibration():
    data = load_multiseason_subset(MULTISEASON)
    policies = policy_archive()
    return data, policies, calibrate_application(data, policies)


def test_fixture_and_archive():
    data = load_opsd_fixture(
        ROOT / "data/synthetic_six_day/synthetic_six_day_fixture.csv"
    )
    assert len(data["price"]) == 144
    assert (data["split"] == "development").sum() == 72

    # Use the fixture development rows for one fixed calibration.
    pseudo = {
        "price": data["price"],
        "load": data["load"],
        "split": data["split"],
    }
    policies = policy_archive()
    calibration = calibrate_application(pseudo, policies)
    price = data["price"][data["split"] == "development"].reshape(3, 24)
    load = data["load"][data["split"] == "development"].reshape(3, 24)
    output = evaluate_archive(policies, price, load, calibration)
    assert output.shape == (3, 13, 3)
    assert np.all(np.isfinite(output))


def test_candidate_differences():
    data, policies, calibration = _calibration()
    mask = data["split"] == "development"
    price = data["price"][mask].reshape(4, 24)
    load = data["load"][mask].reshape(4, 24)
    outcomes = evaluate_archive(policies, price, load, calibration)
    candidate = select_compromise_candidate(outcomes)
    differences, challengers = candidate_differences(outcomes, candidate)
    assert differences.shape == (12, 4, 3)
    assert candidate not in challengers
    assert pareto_mask(outcomes.mean(axis=0))[candidate]


def test_policy_design_is_space_filling_and_frozen():
    policies = policy_archive()
    diagnostics = policy_design_diagnostics(policies)
    assert len(policies) == 13
    assert diagnostics["active_policy_count"] == 12
    assert diagnostics["normalised_minimum_pairwise_distance"] > 0.53
    assert len({policy.policy_id for policy in policies}) == len(policies)


def test_calibration_uses_development_only():
    data, policies, calibration = _calibration()
    changed = {key: np.array(value, copy=True) for key, value in data.items()}
    confirmation = changed["split"] == "confirmation"
    changed["price"][confirmation] += 10000
    changed["load"][confirmation] *= 10
    recalibrated = calibrate_application(changed, policies)
    assert calibration.to_dict() == recalibrated.to_dict()


def test_policy_is_causal_given_frozen_thresholds():
    data, policies, calibration = _calibration()
    policy = policies[4]
    base_price = np.linspace(30, 60, 24)
    base_load = np.linspace(1800, 3000, 24)
    altered_price = base_price.copy()
    altered_load = base_load.copy()
    altered_price[12:] = altered_price[12:][::-1] + 50
    altered_load[12:] = altered_load[12:][::-1] * 1.2

    first = simulate_policy_with_trace(
        policy,
        calibration.threshold_for(policy.policy_id),
        base_price,
        base_load,
        calibration,
    )
    second = simulate_policy_with_trace(
        policy,
        calibration.threshold_for(policy.policy_id),
        altered_price,
        altered_load,
        calibration,
    )
    assert np.allclose(first["charge_mw"][:12], second["charge_mw"][:12])
    assert np.allclose(first["discharge_mw"][:12], second["discharge_mw"][:12])
    assert np.allclose(first["energy_state_mwh"][:13], second["energy_state_mwh"][:13])


def test_fixed_load_scale_and_terminal_settlement():
    data, policies, calibration = _calibration()
    policy = policies[1]
    price = np.full(24, 100.0)
    load_a = np.linspace(1800, 3000, 24)
    load_b = load_a * 2.0
    result_a = simulate_policy_with_trace(
        policy, calibration.threshold_for(policy.policy_id), price, load_a, calibration
    )
    result_b = simulate_policy_with_trace(
        policy, calibration.threshold_for(policy.policy_id), price, load_b, calibration
    )
    assert np.allclose(
        result_b["site_load_mw"] / result_a["site_load_mw"], 2.0
    )
    assert result_a["terminal_settlement_cost_eur"] >= 0
    assert result_a["terminal_shortfall_mwh"] >= 0


def test_parameter_and_weight_validation():
    with pytest.raises(ValueError):
        BatterySystemParameters(minimum_soc_fraction=0.6)
    with pytest.raises(ValueError):
        SelectionWeights(0.5, 0.5, 0.5)
