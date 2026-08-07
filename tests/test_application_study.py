from pathlib import Path
import numpy as np

from pcpi_candidate_tas.application import (
    BatterySystemParameters,
    SelectionWeights,
    application_contract,
    calibrate_application,
    evaluate_archive,
    policy_archive,
    select_compromise_candidate,
)
from pcpi_candidate_tas.multiseason import (
    ScenarioGeneratorParameters,
    generate_multiseason_scenarios,
    load_multiseason_subset,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"


def test_application_contract_and_candidate_selection():
    data = load_multiseason_subset(DATA)
    policies = policy_archive()
    system = BatterySystemParameters()
    weights = SelectionWeights()
    calibration = calibrate_application(data, policies, system)
    prices, loads, seasons, metadata = generate_multiseason_scenarios(
        data,
        "development",
        40,
        11,
        return_metadata=True,
    )
    outcomes = evaluate_archive(policies, prices, loads, calibration, system)
    candidate = select_compromise_candidate(outcomes, weights)
    contract = application_contract(system, weights, calibration)
    assert outcomes.shape == (40, 13, 3)
    assert candidate in range(13)
    assert set(metadata["season_counts"]) == {"autumn", "spring", "summer", "winter"}
    assert contract["application_status"].startswith("stylised")
    assert len(contract["objective_names"]) == 3


def test_balanced_season_sampling_and_determinism():
    data = load_multiseason_subset(DATA)
    parameters = ScenarioGeneratorParameters(season_sampling="balanced")
    first = generate_multiseason_scenarios(
        data, "confirmation", 41, 99, parameters=parameters, return_metadata=True
    )
    second = generate_multiseason_scenarios(
        data, "confirmation", 41, 99, parameters=parameters, return_metadata=True
    )
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    counts = list(first[3]["season_counts"].values())
    assert max(counts) - min(counts) <= 1
    assert first[3]["clipped_load_values"] == 0


def test_generator_parameter_validation():
    try:
        ScenarioGeneratorParameters(shared_smooth_weight=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid shared_smooth_weight accepted")
