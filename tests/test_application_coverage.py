from pathlib import Path
import csv
import numpy as np
import pytest

import pcpi_candidate_tas.application as application_module
from pcpi_candidate_tas.application import (
    ApplicationCalibration,
    BatteryPolicy,
    BatterySystemParameters,
    PolicyThresholds,
    SelectionWeights,
    calibrate_application,
    candidate_differences,
    evaluate_archive,
    policy_archive,
    select_compromise_candidate,
    simulate_policy_with_trace,
)
from pcpi_candidate_tas.battery import generate_scenarios, load_opsd_fixture
from pcpi_candidate_tas.multiseason import (
    ScenarioGeneratorParameters,
    generate_multiseason_scenarios,
    load_multiseason_subset,
)
from pcpi_candidate_tas.opsd_full import (
    chronological_split,
    extract_official_market_year,
)

ROOT = Path(__file__).resolve().parents[1]
MULTISEASON = ROOT / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy_id": ""},
        {"charge_quantile": -0.1},
        {"discharge_quantile": 1.1},
        {"load_reserve_quantile": 1.1},
        {"aggressiveness": -0.1},
        {"aggressiveness": 1.1},
        {"charge_quantile": 0.8, "discharge_quantile": 0.7},
    ],
)
def test_policy_validation_branches(kwargs):
    base = dict(
        policy_id="x",
        charge_quantile=0.2,
        discharge_quantile=0.8,
        load_reserve_quantile=0.9,
        aggressiveness=0.5,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        BatteryPolicy(**base)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity_mwh": 0},
        {"power_mw": 0},
        {"charge_efficiency": 0},
        {"discharge_efficiency": 1.1},
        {"representative_peak_mw": 0},
        {"development_load_reference_quantile": 0},
        {"terminal_energy_value_quantile": 1},
        {"time_step_hours": 0},
    ],
)
def test_system_parameter_validation_branches(kwargs):
    with pytest.raises(ValueError):
        BatterySystemParameters(**kwargs)


def test_selection_weight_negative_and_array():
    with pytest.raises(ValueError):
        SelectionWeights(-0.1, 0.6, 0.5)
    assert np.allclose(SelectionWeights().as_array(), [0.45, 0.40, 0.15])


def test_calibration_and_scenario_error_branches(monkeypatch):
    policies = policy_archive()
    with pytest.raises(ValueError):
        calibrate_application(
            {"price": np.array([1.0]), "load": np.array([1.0]), "split": np.array(["development"])},
            policies,
        )
    with pytest.raises(ValueError):
        calibrate_application(
            {
                "price": np.array([1.0, np.nan]),
                "load": np.array([1.0, 2.0]),
                "split": np.array(["development", "development"]),
            },
            policies,
        )
    with pytest.raises(ValueError):
        calibrate_application(
            {
                "price": np.array([1.0, 2.0]),
                "load": np.array([1.0, 0.0]),
                "split": np.array(["development", "development"]),
            },
            policies,
        )

    data = load_multiseason_subset(MULTISEASON)
    calibration = calibrate_application(data, policies)
    with pytest.raises(KeyError):
        calibration.threshold_for("missing")
    with pytest.raises(ValueError):
        simulate_policy_with_trace(
            policies[0],
            calibration.threshold_for(policies[0].policy_id),
            np.zeros((2, 2)),
            np.ones(4),
            calibration,
        )
    with pytest.raises(ValueError):
        simulate_policy_with_trace(
            policies[0],
            calibration.threshold_for(policies[0].policy_id),
            np.array([]),
            np.array([]),
            calibration,
        )
    with pytest.raises(ValueError):
        simulate_policy_with_trace(
            policies[0],
            calibration.threshold_for(policies[0].policy_id),
            np.array([1.0, np.nan]),
            np.array([1.0, 2.0]),
            calibration,
        )
    with pytest.raises(ValueError):
        simulate_policy_with_trace(
            policies[0],
            calibration.threshold_for(policies[0].policy_id),
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0]),
            calibration,
        )
    with pytest.raises(ValueError):
        simulate_policy_with_trace(
            policies[0],
            PolicyThresholds("wrong", None, None, None),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            calibration,
        )
    with pytest.raises(ValueError):
        evaluate_archive(
            policies,
            np.zeros((2, 2, 2)),
            np.zeros((2, 2, 2)),
            calibration,
        )
    with pytest.raises(ValueError):
        candidate_differences(np.zeros((2, 3, 2)), 3)

    monkeypatch.setattr(
        application_module,
        "compromise_scores",
        lambda *args, **kwargs: (np.zeros(2), np.zeros(2, dtype=bool)),
    )
    with pytest.raises(RuntimeError):
        select_compromise_candidate(np.zeros((2, 2, 3)))


def test_legacy_generator_and_whole_day_validation(tmp_path):
    data = load_opsd_fixture(
        ROOT / "data/synthetic_six_day/synthetic_six_day_fixture.csv"
    )
    price, load = generate_scenarios(data, "development", 3, 4)
    assert price.shape == load.shape == (3, 24)
    bad = {key: np.array(value, copy=True) for key, value in data.items()}
    first_development = int(np.flatnonzero(bad["split"] == "development")[0])
    bad["split"][first_development] = "confirmation"
    with pytest.raises(ValueError):
        generate_scenarios(bad, "development", 1, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_hours": 1},
        {"smoothing_window": 0},
        {"shared_level_sd": -1},
        {"multiplicative_price_sd": -1},
        {"multiplicative_load_sd": -1},
        {"additive_price_sd_eur_per_mwh": -1},
        {"additive_load_sd_system_mwh": -1},
        {"season_sampling": "bad"},
    ],
)
def test_generator_parameter_validation_branches(kwargs):
    with pytest.raises(ValueError):
        ScenarioGeneratorParameters(**kwargs)


def test_multiseason_loader_and_window_uniform_branches(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "utc_timestamp,season,split,DK_1_price_day_ahead_EUR_MWh,"
        "DK_1_load_actual_entsoe_transparency_MWh\n"
        "x,winter,development,nan,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_multiseason_subset(bad)
    bad.write_text(
        "utc_timestamp,season,split,DK_1_price_day_ahead_EUR_MWh,"
        "DK_1_load_actual_entsoe_transparency_MWh\n"
        "x,winter,development,1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_multiseason_subset(bad)

    data = load_multiseason_subset(MULTISEASON)
    parameters = ScenarioGeneratorParameters(
        season_sampling="window_uniform",
        additive_load_sd_system_mwh=1e6,
    )
    price, load, labels, metadata = generate_multiseason_scenarios(
        data,
        "confirmation",
        5,
        77,
        parameters=parameters,
        return_metadata=True,
    )
    assert price.shape == load.shape == (5, 24)
    assert len(labels) == 5
    assert metadata["clipped_load_values"] > 0


def test_official_source_error_branches(tmp_path):
    wrong_hash = tmp_path / "wrong.csv"
    wrong_hash.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256_mismatch"):
        extract_official_market_year(wrong_hash)

    missing_columns = tmp_path / "missing.csv"
    missing_columns.write_text("utc_timestamp,x\n2019-01-01T00:00:00Z,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required DK1 columns"):
        extract_official_market_year(missing_columns, verify_hash=False)

    no_rows = tmp_path / "none.csv"
    with no_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "utc_timestamp",
                "DK_1_price_day_ahead",
                "DK_1_load_actual_entsoe_transparency",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "utc_timestamp": "2020-01-01T00:00:00Z",
                "DK_1_price_day_ahead": "1",
                "DK_1_load_actual_entsoe_transparency": "2",
            }
        )
    with pytest.raises(ValueError, match="no complete 2019"):
        extract_official_market_year(no_rows, verify_hash=False)

    with pytest.raises(ValueError, match="utc_timestamp"):
        chronological_split([{"x": "1"}, {"x": "2"}], 0.5)
