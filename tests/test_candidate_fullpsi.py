import numpy as np

from pcpi_candidate_tas import GaussianCandidateInstance
from pcpi_candidate_tas.candidate import (
    run_candidate_certification, run_reference_c_tracking,
)
from pcpi_candidate_tas.full_psi import (
    classify_pareto_intervals, run_full_psi_confidence_racing,
)


def easy_instance():
    return GaussianCandidateInstance.from_arrays(
        [[0.0, 0.0], [1.4, -0.2], [-0.2, 1.4]],
        np.ones((3, 2)),
    )


def test_candidate_boundaries_certify_easy_instance():
    instance = easy_instance()
    for method in [
        "laurent_massart", "chi_bar", "coordinate", "hybrid"
    ]:
        result = run_candidate_certification(
            instance, delta=0.1, boundary_method=method,
            seed=3, max_samples=30_000, batch_size=20,
            oracle_update_every=200,
        )
        assert result.certified
        assert np.all(result.crossed)


def test_reference_tracking_certifies_easy_instance():
    result = run_reference_c_tracking(
        easy_instance(), delta=0.1, seed=4,
        max_samples=30_000, update_every=20,
    )
    assert result.certified


def test_interval_classification_exact_boxes():
    lower = np.array([
        [0.0, 0.0], [1.0, -0.2], [-0.2, 1.0], [2.0, 2.0]
    ])
    upper = lower.copy()
    dominated, nondominated = classify_pareto_intervals(lower, upper)
    assert nondominated[:3].all()
    assert dominated[3]


def test_full_psi_racing_completes_easy_instance():
    instance = GaussianCandidateInstance.from_arrays(
        [[0.0, 0.0], [1.4, -0.2], [-0.2, 1.4], [1.6, 1.6]],
        np.ones((4, 2)),
    )
    result = run_full_psi_confidence_racing(
        instance, delta=0.1, seed=2, max_samples=100_000,
        batch_per_active_system=20,
    )
    assert result.completed
    assert 3 not in result.estimated_pareto_set
    assert 0 in result.estimated_pareto_set
