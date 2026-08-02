"""Machine-readable status of the one-sided sequential theory.

The candidate-only procedure is a positive-instance confirmation rule: it may
certify an archive-relative nondominance claim, but it is not required to stop
and return a complementary answer under every false or boundary parameter.
Consequently, generic fixed-confidence identification theorems that assume a
total answer map and almost-sure termination for every model are not invoked as
black-box expected-time guarantees here.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class OneSidedTheoryStatus:
    problem_is_one_sided_positive_confirmation: bool = True
    archive_relative_claim_only: bool = True
    global_delta_pac_identification_claimed: bool = False
    false_or_boundary_instance_termination_guaranteed: bool = False
    lower_bound_for_positive_instances_available: bool = True
    unique_continuous_target_under_strict_separation: bool = True
    pathwise_target_convergence_claimed: bool = True
    positive_instance_almost_sure_termination_claimed: bool = True
    expected_time_first_order_optimality_claimed: bool = False
    generic_track_and_stop_expected_time_theorem_directly_applicable: bool = False
    scalar_equalisation_and_inverse_root_architecture_is_prior_art: bool = True
    candidate_specific_dominance_cone_information_is_intended_contribution: bool = True
    change_of_measure_principle_is_prior_art: bool = True
    cumulative_target_tracking_principle_is_prior_art: bool = True
    reference_unit_pull_implementation_matches_claimed_pathwise_construction: bool = True
    practical_batched_implementation_theorem_covered: bool = False
    equality_boundary_covered: bool = False
    temporally_dependent_scenarios_covered: bool = False
    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

def one_sided_theory_status() -> OneSidedTheoryStatus:
    """Return the conservative one-sided theorem-to-prior-art mapping."""
    return OneSidedTheoryStatus()
