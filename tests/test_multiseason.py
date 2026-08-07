from pathlib import Path
import numpy as np
from pcpi_candidate_tas.multiseason import (
    ScenarioGeneratorParameters,
    load_multiseason_subset,
    seasonal_windows,
    generate_multiseason_scenarios,
)

DATA = Path(__file__).parents[1] / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"


def test_multiseason_load_and_windows():
    data = load_multiseason_subset(DATA)
    assert len(data["price"]) == 285
    assert set(data["season"]) == {"winter", "spring", "summer", "autumn"}
    dev = seasonal_windows(data, "development")
    conf = seasonal_windows(data, "confirmation")
    assert len(dev) == 4
    assert len(conf) >= 4
    assert all(len(p) == 24 and len(l) == 24 for _, p, l in dev + conf)


def test_multiseason_scenarios_are_deterministic_and_finite():
    data = load_multiseason_subset(DATA)
    parameters = ScenarioGeneratorParameters()
    first = generate_multiseason_scenarios(
        data, "confirmation", 12, 7, parameters=parameters
    )
    second = generate_multiseason_scenarios(
        data, "confirmation", 12, 7, parameters=parameters
    )
    assert np.allclose(first[0], second[0])
    assert np.allclose(first[1], second[1])
    assert np.array_equal(first[2], second[2])
    assert first[0].shape == (12, 24)
    assert np.all(first[1] > 0)
