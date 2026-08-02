import numpy as np
from pcpi_candidate_tas import GaussianCandidateInstance
from pcpi_candidate_tas.candidate import run_theory_aligned_c_tracking
from pcpi_candidate_tas.theory_status import one_sided_theory_status

def easy():
    return GaussianCandidateInstance.from_arrays(
        [[0.,0.],[1.2,-.1],[-.1,1.2]],np.ones((3,2))
    )

def test_theory_aligned_tracking_records_bounded_deficit():
    result=run_theory_aligned_c_tracking(
        easy(),delta=.1,seed=12,max_samples=25000,update_every=10,trace_every=20
    )
    assert result.certified and result.trace
    assert max(x["maximum_positive_deficit"] for x in result.trace) <= 1.0000001
    assert max(x["tracking_discrepancy_linf"] for x in result.trace) < 3.0

def test_expected_time_status_names_reference_coverage():
    status=one_sided_theory_status()
    assert status.reference_unit_pull_implementation_matches_claimed_pathwise_construction
    assert not status.practical_batched_implementation_theorem_covered
