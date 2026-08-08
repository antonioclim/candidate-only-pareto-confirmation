from pcpi_candidate_tas.theory_status import one_sided_theory_status


def test_calibration_status_is_conservative():
    status = one_sided_theory_status()
    assert status.gaussian_core_calibration_completed
    assert status.gaussian_core_gate_passed
    assert status.easy_positive_controls_crossed_in_all_core_paths
    assert not status.distribution_free_validity_claimed
    assert status.temporal_dependence_failure_boundary_empirically_confirmed
    assert status.strong_ar1_false_certification_inflation_observed
    assert not status.mild_ar1_validity_claimed
    assert not status.heavy_tail_robustness_theorem_claimed
    assert not status.skewness_robustness_theorem_claimed
    assert not status.contamination_robustness_theorem_claimed
    assert status.near_singular_positive_definite_gaussian_stress_passed
