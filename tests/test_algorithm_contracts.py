import json
from pathlib import Path

import numpy as np
import pytest

from pcpi_candidate_tas import (
    GaussianCandidateInstance,
    algorithm_contract,
    algorithm_contracts,
)
from pcpi_candidate_tas.boundaries import BoundaryDecision
from pcpi_candidate_tas.candidate import (
    _gap_racing_base_target,
    _largest_deficit_batch_allocation,
    _mix_forced_exploration,
    run_candidate_certification,
    run_theory_aligned_c_tracking,
)
from pcpi_candidate_tas.paired_artifacts import (
    build_paired_certificate,
    write_raw_paired,
)
from pcpi_candidate_tas.theory_status import one_sided_theory_status


def easy_instance():
    return GaussianCandidateInstance.from_arrays(
        [[0.0, 0.0], [1.2, -0.1], [-0.1, 1.2]],
        np.ones((3, 2)),
    )


def _decision(crossed=False, cone=0.0, cone_threshold=1.0,
              max_z=0.0, coordinate_threshold=1.0):
    return BoundaryDecision(
        crossed, cone, cone_threshold, max_z, coordinate_threshold,
        0, 0.01, "hybrid", "log_telescoping",
    )


def test_algorithm_registry_is_unique_and_conservative():
    contracts = algorithm_contracts()
    ids = [contract.algorithm_id for contract in contracts]
    assert len(ids) == len(set(ids))
    assert algorithm_contract("reference_unit_pull_c_tracking").theorem_coverage
    assert "No inherited tracking theorem" in algorithm_contract(
        "batched_plugin_tracking"
    ).theorem_coverage
    assert sum("Pathwise allocation convergence" in c.theorem_coverage
               for c in contracts) == 1


def test_forced_exploration_does_not_compound_when_reapplied_to_base():
    base = np.array([0.8, 0.1, 0.1])
    first = _mix_forced_exploration(base, 0.03)
    second = _mix_forced_exploration(base, 0.03)
    assert np.allclose(first, second)
    assert np.allclose(first, np.array([0.786, 0.107, 0.107]))


def test_largest_deficit_batch_fixes_negative_fractional_remainder_bug():
    counts = np.array([3, 3, 0])
    target = np.array([0.5431223842839688, 0.4217870659627395,
                       0.03509054975329172])
    allocation = _largest_deficit_batch_allocation(counts, target, 1)
    # The former fractional-remainder code incorrectly pulled system 1 even
    # though its final-count deficit was negative.
    assert allocation.tolist() == [1, 0, 0]
    assert allocation.sum() == 1


def test_largest_deficit_batch_is_integer_and_exact():
    rng = np.random.default_rng(44)
    for system_count in range(2, 9):
        for _ in range(100):
            counts = rng.integers(0, 100, size=system_count)
            target = rng.dirichlet(np.ones(system_count))
            step = int(rng.integers(0, 50))
            allocation = _largest_deficit_batch_allocation(counts, target, step)
            assert allocation.dtype.kind in "iu"
            assert np.all(allocation >= 0)
            assert int(allocation.sum()) == step


def test_hybrid_gap_racing_uses_closest_branch():
    decisions = (
        _decision(False, cone=0.9, cone_threshold=1.0,
                  max_z=0.0, coordinate_threshold=5.0),
        _decision(False, cone=0.0, cone_threshold=5.0,
                  max_z=0.9, coordinate_threshold=1.0),
    )
    arrays = (
        np.array([d.crossed for d in decisions]),
        np.array([d.cone_glr for d in decisions]),
        np.array([d.cone_threshold for d in decisions]),
        np.array([d.max_z for d in decisions]),
        np.array([d.coordinate_threshold for d in decisions]),
        np.array([d.witness_objective for d in decisions]),
    )
    target = _gap_racing_base_target(arrays, "hybrid")
    assert np.isclose(target[0], 0.5)
    assert np.isclose(target[1], target[2])


def test_batched_result_exposes_algorithm_and_batch_grid():
    result = run_candidate_certification(
        easy_instance(), delta=0.1, seed=7, max_samples=30_000,
        batch_size=20, oracle_update_every=200, trace_every=20,
    )
    assert result.certified
    assert result.algorithm_id == "batched_plugin_tracking"
    assert not result.theorem_covered
    assert result.stopping_grid == "batch_boundary"
    assert result.termination_reason == "certified"
    assert result.trace
    for entry in result.trace:
        assert entry["time"] == entry["pre_batch_time"] + sum(entry["allocation"])
        assert np.isclose(sum(entry["projected_target_weights"]), 1.0)


def test_reference_updates_immediately_and_exposes_every_pull_grid(monkeypatch):
    import pcpi_candidate_tas.candidate as module
    calls = []
    real = module.solve_scalar_allocation

    def wrapped(*args, **kwargs):
        calls.append(kwargs.copy())
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "solve_scalar_allocation", wrapped)
    result = run_theory_aligned_c_tracking(
        easy_instance(), delta=0.1, seed=8, max_samples=20_000,
        update_every=17, trace_every=20,
    )
    assert result.certified
    assert calls
    assert result.algorithm_id == "reference_unit_pull_c_tracking"
    assert result.theorem_covered
    assert result.stopping_grid == "every_pull"
    assert result.trace
    for entry in result.trace:
        assert entry["time"] == entry["pre_pull_time"] + 1
        assert entry["maximum_positive_deficit"] < 2.0
        assert entry["minimum_deficit"] > -1.0


def test_final_cap_is_rechecked(monkeypatch):
    import pcpi_candidate_tas.candidate as module
    instance = easy_instance()
    calls = {"count": 0}

    def fake(*args, **kwargs):
        calls["count"] += 1
        crossed = calls["count"] >= 2
        return tuple(
            _decision(crossed=crossed, cone=2.0 if crossed else 0.0,
                      cone_threshold=1.0)
            for _ in range(instance.n_challengers)
        )

    monkeypatch.setattr(module, "evaluate_all_challengers", fake)
    initial = instance.means.shape[0] * 2
    result = run_candidate_certification(
        instance, delta=0.1, policy="uniform", seed=1,
        max_samples=initial + 1, batch_size=1, trace_every=0,
    )
    assert result.certified
    assert result.termination_reason == "certified_at_cap"


def test_batched_fail_closed_mode(monkeypatch):
    import pcpi_candidate_tas.candidate as module
    monkeypatch.setattr(
        module,
        "solve_scalar_allocation",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced")),
    )
    with pytest.raises(RuntimeError, match="batched plug-in oracle failed"):
        run_candidate_certification(
            easy_instance(), delta=0.1, max_samples=20,
            batch_size=2, oracle_update_every=1,
            oracle_failure_mode="fail_closed",
        )


def test_input_validation_and_static_target_validation():
    instance = easy_instance()
    with pytest.raises(ValueError):
        run_candidate_certification(instance, max_samples=5)
    with pytest.raises(ValueError):
        run_candidate_certification(instance, forced_exploration=1.1)
    with pytest.raises(ValueError):
        run_candidate_certification(instance, gap_floor_exponent=0.0)
    with pytest.raises(ValueError):
        run_candidate_certification(
            instance, policy="oracle_static",
            true_oracle_weights=np.array([0.5, -0.1, 0.6]),
        )
    with pytest.raises(ValueError):
        run_theory_aligned_c_tracking(instance, update_every=0)


def test_paired_builder_fails_on_raw_difference_mismatch(tmp_path):
    rng = np.random.default_rng(3)
    differences = np.stack([
        rng.multivariate_normal([0.7, -0.1], np.eye(2), size=40),
        rng.multivariate_normal([-0.1, 0.7], np.eye(2), size=40),
    ])
    raw = tmp_path / "raw.csv"
    write_raw_paired(raw, differences, ["a", "b"], ["f1", "f2"])
    altered = differences.copy()
    altered[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="raw evidence"):
        build_paired_certificate(
            altered, 0.1, "candidate", ["a", "b"], ["f1", "f2"], raw
        )
    with pytest.raises(ValueError, match="collision"):
        build_paired_certificate(
            differences, 0.1, "a", ["a", "b"], ["f1", "f2"], raw
        )


def test_cli_describe_algorithms(capsys):
    from pcpi_candidate_tas.cli import main
    assert main(["describe-algorithms"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == len(algorithm_contracts())
    assert any(item["algorithm_id"] == "independent_paired_verifier"
               for item in payload)


def test_status_contract():
    status = one_sided_theory_status()
    assert status.algorithm_contract_registry_available
    assert status.batched_stopping_grid_is_batch_boundary
    assert status.batched_forced_exploration_is_non_compounding
    assert status.batched_integer_allocation_uses_largest_deficit
    assert status.final_cap_boundary_is_rechecked
    assert status.paired_builder_verifies_raw_evidence_consistency


def test_incremental_paired_grid_matches_direct_prefix_evaluation():
    from pcpi_candidate_tas.paired import (
        evaluate_archive,
        sequential_archive_confirmation,
    )
    rng = np.random.default_rng(55)
    differences = np.stack([
        rng.multivariate_normal([0.5, -0.1], [[1.0, 0.3], [0.3, 1.0]], size=120),
        rng.multivariate_normal([-0.1, 0.5], [[1.0, 0.2], [0.2, 1.0]], size=120),
    ])
    result = sequential_archive_confirmation(
        differences, 0.1, method="hybrid",
        min_count=8, max_count=120, check_every=7,
    )
    grid = list(range(8, 121, 7))
    if grid[-1] != 120:
        grid.append(120)
    expected_certified = False
    expected_stop = 120
    for count in grid:
        certified, _ = evaluate_archive(
            differences[:, :count], 0.1, method="hybrid"
        )
        if certified:
            expected_certified = True
            expected_stop = count
            break
    assert result["certified"] == expected_certified
    assert result["stopping_count"] == expected_stop
    assert result["decision_checks"] <= len(grid)
    assert result["algorithm_id"] == "paired_sequential_confirmation"


def test_paired_artifact_input_contract_branches(tmp_path):
    import pcpi_candidate_tas.paired_artifacts as module
    with pytest.raises(ValueError, match="candidate_id"):
        module._validate_identity_contract("", ["a"], ["f1"])
    with pytest.raises(ValueError, match="challenger_ids"):
        module._validate_identity_contract("c", [], ["f1"])
    with pytest.raises(ValueError, match="unique"):
        module._validate_identity_contract("c", ["a", "a"], ["f1"])
    with pytest.raises(ValueError, match="objective_names"):
        module._validate_identity_contract("c", ["a"], [])
    with pytest.raises(ValueError, match="unique"):
        module._validate_identity_contract("c", ["a"], ["f1", "f1"])

    good = np.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="challenger-by-count"):
        module._validate_differences(np.zeros((3, 2)), ["a"], ["f1", "f2"])
    with pytest.raises(ValueError, match="challenger count"):
        module._validate_differences(good, ["a", "b"], ["f1", "f2"])
    with pytest.raises(ValueError, match="two paired"):
        module._validate_differences(np.zeros((1, 1, 2)), ["a"], ["f1", "f2"])
    with pytest.raises(ValueError, match="objective count"):
        module._validate_differences(good, ["a"], ["f1"])
    bad = good.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module._validate_differences(bad, ["a"], ["f1", "f2"])
    with pytest.raises(ValueError, match="unique challenger"):
        module.write_raw_paired(tmp_path / "x.csv", good, ["a", "a"], ["f1", "f2"])
    with pytest.raises(ValueError, match="unique objective"):
        module.write_raw_paired(tmp_path / "x.csv", good, ["a"], ["f1", "f1"])


def test_read_raw_paired_rejects_malformed_files(tmp_path):
    from pcpi_candidate_tas.paired_artifacts import read_raw_paired
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("bad,header\n1,2\n")
    with pytest.raises(ValueError, match="columns"):
        read_raw_paired(wrong, ["a"], ["f1"])

    unknown = tmp_path / "unknown.csv"
    unknown.write_text("challenger_id,scenario_index,f1\nb,0,1\nb,1,2\n")
    with pytest.raises(ValueError, match="unknown"):
        read_raw_paired(unknown, ["a"], ["f1"])

    unbalanced = tmp_path / "unbalanced.csv"
    unbalanced.write_text(
        "challenger_id,scenario_index,f1\n"
        "a,0,1\na,1,2\nb,0,1\n"
    )
    with pytest.raises(ValueError, match="balanced"):
        read_raw_paired(unbalanced, ["a", "b"], ["f1"])

    bad_index = tmp_path / "bad_index.csv"
    bad_index.write_text(
        "challenger_id,scenario_index,f1\n"
        "a,0,1\na,2,2\n"
    )
    with pytest.raises(ValueError, match="index"):
        read_raw_paired(bad_index, ["a"], ["f1"])


def test_replay_detects_semantic_statistic_and_verdict_mutations(tmp_path):
    import copy
    from pcpi_candidate_tas.paired_artifacts import (
        build_paired_certificate, canonical_json, replay_paired_certificate,
        sha256_bytes, write_raw_paired,
    )
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(61)
    differences = np.stack([
        rng.multivariate_normal([0.7, -0.1], np.eye(2), size=60),
        rng.multivariate_normal([-0.1, 0.7], np.eye(2), size=60),
    ])
    raw = tmp_path / "raw.csv"
    write_raw_paired(raw, differences, ["a", "b"], ["f1", "f2"])
    artifact = build_paired_certificate(
        differences, 0.1, "candidate", ["a", "b"], ["f1", "f2"], raw
    )
    schema = root / "schemas/pcpi_paired_candidate_certificate.schema.json"
    assert replay_paired_certificate(artifact, raw, schema)["valid"]

    def mutate(mutator):
        changed = copy.deepcopy(artifact)
        mutator(changed["semantic_core"])
        changed["semantic_sha256"] = sha256_bytes(
            canonical_json(changed["semantic_core"])
        )
        return changed

    cases = [
        (lambda s: s["sufficient_statistics"].__setitem__("count", 59), "count"),
        (lambda s: s["sufficient_statistics"]["means"][0].__setitem__(0, 99.0), "mean"),
        (lambda s: s["sufficient_statistics"]["covariances"][0][0].__setitem__(0, 99.0), "covariance"),
        (lambda s: s["certificate"].__setitem__("verdict", "notCertified" if s["certificate"]["verdict"] == "certified" else "certified"), "verdict"),
        (lambda s: s["certificate"]["crossed"].__setitem__(0, not s["certificate"]["crossed"][0]), "crossed"),
        (lambda s: s["certificate"]["coordinate_t_max"].pop(), "length"),
        (lambda s: s["certificate"]["coordinate_t_max"].__setitem__(0, s["certificate"]["coordinate_t_max"][0] + 1.0), "coordinate_t_max"),
    ]
    for mutator, message in cases:
        with pytest.raises(ValueError, match=message):
            replay_paired_certificate(mutate(mutator), raw)
