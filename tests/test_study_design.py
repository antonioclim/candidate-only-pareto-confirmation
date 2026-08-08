from pathlib import Path
import json

import numpy as np
import pytest

from pcpi_candidate_tas.application import BatteryPolicy, policy_archive
from pcpi_candidate_tas.multiseason import load_multiseason_subset
from pcpi_candidate_tas.study_design import (
    ConfirmatoryDesign,
    HardArchiveDesign,
    StudyDesign,
    SeedPlan,
    binomial_mcse,
    construct_hard_archive,
    local_policy_pool,
    replication_precision_table,
    validate_hard_archive,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/synthetic_multiseason/synthetic_multiseason_fixture.csv"


def small_design() -> HardArchiveDesign:
    return HardArchiveDesign(
        design_scenarios=140,
        validation_replications=20,
        validation_scenarios=70,
        hard_neighbours=5,
        minimum_parameter_distance=0.0,
        minimum_orthant_effect=0.01,
        maximum_orthant_effect=2.0,
        maximum_modal_candidate_rate=1.0,
        minimum_distinct_candidates=1,
        minimum_median_pareto_size=1.0,
    )


def test_design_validation_and_precision():
    design = StudyDesign()
    design.validate()
    assert design.to_dict()["confirmatory"]["application_replications"] == 400
    assert np.isclose(binomial_mcse(0.5, 400), 0.025)
    assert len(replication_precision_table()) == 12
    with pytest.raises(ValueError):
        SeedPlan(archive_design=-1).validate()
    with pytest.raises(ValueError):
        ConfirmatoryDesign(application_replications=10).validate()


def test_local_pool_is_deterministic_and_unique():
    anchor = policy_archive()[1]
    first = local_policy_pool(anchor, design=small_design())
    second = local_policy_pool(anchor, design=small_design())
    assert first == second
    keys = {
        (
            p.charge_quantile,
            p.discharge_quantile,
            p.load_reserve_quantile,
            p.aggressiveness,
        )
        for p in first
    }
    assert len(keys) == len(first)
    with pytest.raises(ValueError):
        local_policy_pool(
            BatteryPolicy("idle", 0.0, 1.0, 1.0, 0.0),
            design=small_design(),
        )


def test_hard_archive_build_and_validation():
    data = load_multiseason_subset(DATA)
    constructed = construct_hard_archive(data, design=small_design())
    assert constructed["final_archive_size"] == 6
    assert constructed["final_policies"][0].policy_id == "candidate_anchor_00"
    assert len(constructed["hardness_rows"]) >= 5
    validation = validate_hard_archive(
        data, constructed["final_policies"], design=small_design()
    )
    assert validation["metrics"]["replications"] == 20
    assert sum(validation["metrics"]["candidate_counts"].values()) == 20


def test_public_study_design_file():
    path = ROOT / "config/STUDY_DESIGN.json"
    payload = json.loads(path.read_text())
    assert payload["specification"] == "pcpi-study-design/1.0"
    assert len(payload["design_sha256"]) == 64
