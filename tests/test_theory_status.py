from pcpi_candidate_tas.theory_status import one_sided_theory_status

def test_one_sided_theory_mapping_is_conservative():
    status = one_sided_theory_status()
    assert status.problem_is_one_sided_positive_confirmation
    assert status.archive_relative_claim_only
    assert status.certified_proposition_is_strict_positive_witness_separation
    assert status.certified_proposition_implies_ordinary_nondominance
    assert not status.equality_boundary_is_certifiable
    assert not status.global_delta_pac_identification_claimed
    assert not status.false_or_boundary_instance_termination_guaranteed
    assert status.pathwise_target_convergence_claimed
    assert status.positive_instance_almost_sure_termination_claimed
    assert not status.expected_time_first_order_optimality_claimed
    assert not status.generic_track_and_stop_expected_time_theorem_directly_applicable

def test_prior_art_and_code_mapping_are_explicit():
    status = one_sided_theory_status()
    assert status.scalar_equalisation_and_inverse_root_architecture_is_prior_art
    assert status.identity_covariance_positive_part_projection_is_prior_art
    assert not status.generic_candidate_specific_dominance_cone_information_claimed_new
    assert status.unequal_diagonal_variance_selected_candidate_projection_is_specialised_refinement
    assert status.executable_reference_control_flow_matches_pathwise_construction
    assert not status.practical_batched_implementation_theorem_covered
    assert not status.temporally_dependent_scenarios_covered
