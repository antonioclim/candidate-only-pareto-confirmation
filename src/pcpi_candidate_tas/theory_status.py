"""Machine-readable conservative status of the one-sided theory."""
from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class OneSidedTheoryStatus:
    problem_is_one_sided_positive_confirmation: bool = True
    archive_relative_claim_only: bool = True
    certified_proposition_is_strict_positive_witness_separation: bool = True
    certified_proposition_implies_ordinary_nondominance: bool = True
    equality_boundary_is_certifiable: bool = False
    global_delta_pac_identification_claimed: bool = False
    false_or_boundary_instance_termination_guaranteed: bool = False
    lower_bound_for_positive_instances_available: bool = True
    unique_continuous_target_under_strict_separation: bool = True
    pathwise_target_convergence_claimed: bool = True
    positive_instance_almost_sure_termination_claimed: bool = True
    expected_time_first_order_optimality_claimed: bool = False
    generic_track_and_stop_expected_time_theorem_directly_applicable: bool = False
    scalar_equalisation_and_inverse_root_architecture_is_prior_art: bool = True
    identity_covariance_positive_part_projection_is_prior_art: bool = True
    generic_candidate_specific_dominance_cone_information_claimed_new: bool = False
    unequal_diagonal_variance_selected_candidate_projection_is_specialised_refinement: bool = True
    change_of_measure_principle_is_prior_art: bool = True
    cumulative_target_tracking_principle_is_prior_art: bool = True
    mathematical_reference_oracle_error_must_vanish: bool = True
    executable_reference_control_flow_matches_pathwise_construction: bool = True
    executable_finite_precision_is_literal_exact_theorem_object: bool = False
    executable_reference_requests_decreasing_oracle_tolerance: bool = True
    reference_regularisation_floor_vanishes: bool = True
    reference_oracle_failures_are_fail_closed: bool = True
    cumulative_target_positive_deficit_bound_is_k_minus_one: bool = True
    chi_bar_zero_atom_uses_strict_crossing: bool = True
    practical_batched_implementation_theorem_covered: bool = False
    algorithm_contract_registry_available: bool = True
    batched_stopping_grid_is_batch_boundary: bool = True
    batched_forced_exploration_is_non_compounding: bool = True
    batched_integer_allocation_uses_largest_deficit: bool = True
    final_cap_boundary_is_rechecked: bool = True
    paired_builder_verifies_raw_evidence_consistency: bool = True
    equality_boundary_covered: bool = False
    temporally_dependent_scenarios_covered: bool = False
    gaussian_core_calibration_completed: bool = True
    gaussian_core_gate_passed: bool = True
    easy_positive_controls_crossed_in_all_core_paths: bool = True
    distribution_free_validity_claimed: bool = False
    temporal_dependence_failure_boundary_empirically_confirmed: bool = True
    strong_ar1_false_certification_inflation_observed: bool = True
    mild_ar1_validity_claimed: bool = False
    heavy_tail_robustness_theorem_claimed: bool = False
    skewness_robustness_theorem_claimed: bool = False
    contamination_robustness_theorem_claimed: bool = False
    near_singular_positive_definite_gaussian_stress_passed: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

def one_sided_theory_status() -> OneSidedTheoryStatus:
    return OneSidedTheoryStatus()
