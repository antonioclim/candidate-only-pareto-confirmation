import numpy as np

from pcpi_candidate_tas.imoscore import (
    all_constraint_rates,
    enumerate_assignment_phantoms,
    independent_nonpareto_scores,
    integer_allocation,
    solve_independent_imoscore,
)
from pcpi_candidate_tas.psi_comparators import pareto_mask


def instance():
    means = np.array(
        [
            [0.10, 0.85, 0.55],
            [0.55, 0.10, 0.85],
            [0.85, 0.55, 0.10],
            [0.62, 0.65, 0.66],
            [0.74, 0.76, 0.70],
            [0.95, 0.95, 0.95],
        ]
    )
    variances = np.array(
        [
            [0.8, 1.0, 1.2],
            [1.1, 0.9, 1.0],
            [1.0, 1.2, 0.8],
            [1.0, 1.0, 1.0],
            [1.3, 0.9, 1.1],
            [1.0, 1.0, 1.0],
        ]
    )
    return means, variances


def test_phantoms_are_deterministic_and_finite_where_assigned():
    means, _ = instance()
    pareto = np.flatnonzero(pareto_mask(means))
    a = enumerate_assignment_phantoms(means, pareto)
    b = enumerate_assignment_phantoms(means, pareto)
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert np.array_equal(x.values, y.values)
        finite = np.isfinite(x.values)
        assert np.all(x.contributors[finite] >= 0)
        assert np.all(x.contributors[~finite] == -1)


def test_nonpareto_shares_are_inverse_score_proportional():
    means, variances = instance()
    pareto = np.flatnonzero(pareto_mask(means))
    nonpareto = np.flatnonzero(~pareto_mask(means))
    phantoms = enumerate_assignment_phantoms(means, pareto)
    scores, _ = independent_nonpareto_scores(
        means, variances, nonpareto, phantoms
    )
    result = solve_independent_imoscore(means, variances)
    expected = (1.0 / scores) / np.sum(1.0 / scores)
    assert np.allclose(result.nonpareto_shares, expected)


def test_solver_returns_positive_probability_vector_and_improves_equal_rate():
    means, variances = instance()
    result = solve_independent_imoscore(means, variances, seed=4)
    assert result.success, result.message
    assert np.all(result.weights > 0)
    assert np.isclose(result.weights.sum(), 1.0)
    pareto = result.pareto_indices
    nonpareto = result.nonpareto_indices
    phantoms = enumerate_assignment_phantoms(means, pareto)
    equal = np.full(len(means), 1.0 / len(means))
    equal_rate = np.min(
        all_constraint_rates(
            means, variances, equal, pareto, nonpareto, phantoms
        )
    )
    assert result.rate >= equal_rate - 1e-9
    assert result.minimum_constraint_residual >= -1e-7


def test_scale_invariance_when_means_and_variances_are_scaled_consistently():
    means, variances = instance()
    a = solve_independent_imoscore(means, variances, seed=5)
    scale = 7.5
    b = solve_independent_imoscore(
        means * scale,
        variances * scale**2,
        seed=5,
    )
    assert np.allclose(a.weights, b.weights, atol=2e-5)
    assert np.isclose(a.rate, b.rate, rtol=2e-5, atol=1e-9)


def test_integer_allocation_sums_to_budget():
    counts = integer_allocation(np.array([0.1, 0.2, 0.3, 0.4]), 101)
    assert counts.sum() == 101
    assert np.all(counts >= 2)
