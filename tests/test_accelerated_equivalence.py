from pathlib import Path
import numpy as np
from pcpi_candidate_tas.application import (
    calibrate_application, candidate_differences,
    evaluate_archive, policy_archive,
)
from pcpi_candidate_tas.multiseason import (
    generate_multiseason_scenarios, load_multiseason_subset,
)
from pcpi_candidate_tas.paired import sequential_archive_confirmation
from pcpi_candidate_tas.accelerated import (
    evaluate_archive_fast, sequential_candidate_fast,
    sequential_full_psi_coordinate_fast,
)
from pcpi_candidate_tas.psi_comparators import (
    sequential_full_psi_coordinate,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"

def fixture():
    data = load_multiseason_subset(DATA)
    policies = policy_archive()
    calibration = calibrate_application(data, policies)
    prices, loads, _ = generate_multiseason_scenarios(
        data, "confirmation", 160, 717001
    )
    return policies, calibration, prices, loads

def test_application_equivalence():
    policies, calibration, prices, loads = fixture()
    scalar = evaluate_archive(policies, prices, loads, calibration)
    fast = evaluate_archive_fast(policies, prices, loads, calibration)
    assert np.allclose(scalar, fast, rtol=1e-12, atol=1e-8)

def test_candidate_equivalence():
    policies, calibration, prices, loads = fixture()
    outcomes = evaluate_archive_fast(
        policies, prices, loads, calibration
    )
    differences, _ = candidate_differences(outcomes, 1)
    for method in ("coordinate", "hotelling", "hybrid"):
        reference = sequential_archive_confirmation(
            differences, 0.05, method=method,
            min_count=8, max_count=160, check_every=5
        )
        fast = sequential_candidate_fast(
            differences, 0.05, method=method,
            min_count=8, max_count=160, check_every=5
        )
        assert reference["certified"] == fast["certified"]
        assert reference["stopping_count"] == fast["stopping_count"]

def test_full_set_equivalence():
    policies, calibration, prices, loads = fixture()
    outcomes = evaluate_archive_fast(
        policies, prices, loads, calibration
    )
    reference = sequential_full_psi_coordinate(
        outcomes, 0.05, min_count=12,
        max_count=160, check_every=5
    )
    fast = sequential_full_psi_coordinate_fast(
        outcomes, 0.05, min_count=12,
        max_count=160, check_every=5
    )
    assert reference["completed"] == fast["completed"]
    assert reference["stopping_count"] == fast["stopping_count"]
    assert np.array_equal(reference["dominated"], fast["dominated"])
    assert np.array_equal(
        reference["nondominated"], fast["nondominated"]
    )
