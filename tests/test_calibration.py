import numpy as np

from pcpi_candidate_tas.paired import sequential_archive_confirmation
from pcpi_candidate_tas.calibration import (
    CalibrationDesign,
    bonferroni_core_upper,
    challenger_means,
    covariance_matrix,
    generate_exact_null_path,
    generate_robustness_path,
    sequential_method_diagnostics,
)


def test_execution_contract_and_means():
    contract = CalibrationDesign()
    contract.validate()
    means = challenger_means(4)
    assert means.shape == (3, 4)
    assert np.allclose(means[0], 0)
    assert means[1, 0] > 0 and means[2, 1] > 0


def test_exact_path_marginal_and_cross_covariance():
    data = generate_exact_null_path(
        123,
        2,
        "compound_rho_0.5",
        count=100_000,
    )
    target = covariance_matrix(2, "compound_rho_0.5")
    assert np.allclose(np.cov(data[0], rowvar=False), target, atol=0.025)
    cross = np.cov(data[0].T, data[1].T)[:2, 2:]
    assert np.allclose(cross, target / 2, atol=0.025)


def test_fast_diagnostics_match_reference():
    data = generate_exact_null_path(
        456, 4, "identity", count=300
    )
    diagnostics = sequential_method_diagnostics(
        data,
        0.05,
        methods=("coordinate", "hotelling", "hybrid"),
        max_count=300,
        check_every=5,
    )
    for method in ("coordinate", "hotelling", "hybrid"):
        reference = sequential_archive_confirmation(
            data,
            0.05,
            method=method,
            min_count=8,
            max_count=300,
            check_every=5,
        )
        assert diagnostics[method]["certified"] == reference["certified"]
        assert (
            diagnostics[method]["stopping_count"]
            == reference["stopping_count"]
        )


def test_robustness_generators_are_finite_and_centred():
    mechanisms = [
        "iid Gaussian reference",
        "multivariate t, df=5",
        "multivariate t, df=3",
        "centred lognormal coordinates",
        "1% ten-SD contamination",
        "AR(1) rho=0.3",
        "AR(1) rho=0.6",
        "near-singular covariance",
    ]
    for index, mechanism in enumerate(mechanisms):
        data = generate_robustness_path(
            1000 + index,
            mechanism,
            count=20_000,
        )
        assert data.shape == (3, 20_000, 3)
        assert np.all(np.isfinite(data))
        assert np.max(np.abs(data[0].mean(axis=0))) < 0.15


def test_bonferroni_upper_is_monotone():
    assert bonferroni_core_upper(0, 3000) < 0.01
    assert bonferroni_core_upper(50, 3000) < bonferroni_core_upper(
        100, 3000
    )


def test_validation_error_branches():
    import pytest
    from pcpi_candidate_tas.calibration import (
        one_sided_clopper_pearson_upper,
    )

    for contract in [
        CalibrationDesign(sample_cap=5),
        CalibrationDesign(delta=1.0),
        CalibrationDesign(challengers=2),
        CalibrationDesign(core_cells=11),
        CalibrationDesign(robustness_primary_method="coordinate"),
    ]:
        with pytest.raises(ValueError):
            contract.validate()

    with pytest.raises(ValueError):
        covariance_matrix(0, "identity")
    with pytest.raises(ValueError):
        covariance_matrix(2, "unknown")
    with pytest.raises(ValueError):
        challenger_means(1)
    with pytest.raises(ValueError):
        generate_exact_null_path(1, 2, "identity", count=1)
    with pytest.raises(ValueError):
        generate_robustness_path(1, "unknown", count=10)

    with pytest.raises(ValueError):
        sequential_method_diagnostics(np.zeros((2, 3)), 0.05)
    bad = np.zeros((1, 10, 2))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        sequential_method_diagnostics(bad, 0.05)
    with pytest.raises(ValueError):
        sequential_method_diagnostics(
            np.zeros((1, 10, 2)), 0.05, methods=()
        )

    coordinate_only = sequential_method_diagnostics(
        np.zeros((1, 20, 2)),
        0.05,
        methods=("coordinate",),
        max_count=20,
        check_every=5,
    )
    assert "coordinate" in coordinate_only

    with pytest.raises(ValueError):
        one_sided_clopper_pearson_upper(-1, 10, 0.95)
    with pytest.raises(ValueError):
        one_sided_clopper_pearson_upper(1, 10, 1.0)
    assert one_sided_clopper_pearson_upper(10, 10, 0.95) == 1.0
