import math
import numpy as np
from scipy.stats import norm

from pcpi_candidate_tas.boundaries import (
    chi_bar_glr_threshold, chi_bar_square_isf, chi_bar_square_sf,
    count_mass_log_telescoping, count_mass_power, evaluate_boundary,
    intrinsic_pair_alpha, laurent_massart_glr_threshold,
)


def test_log_spending_telescopes():
    total = sum(
        count_mass_log_telescoping(s) for s in range(1, 200_000)
    )
    expected = 1.0 - math.log(2.0) / math.log(200_001.0)
    assert abs(total - expected) < 1e-12


def test_power_spending_nearly_sums_to_one():
    total = sum(count_mass_power(s, 2.0) for s in range(1, 100_000))
    assert 0.99999 < total < 1.0


def test_pair_alpha_is_valid():
    alpha = intrinsic_pair_alpha(0.05, 10, 20)
    assert 0.0 < alpha < 0.05


def test_chi_bar_dimension_one_matches_one_sided_normal():
    for alpha in [0.1, 0.05, 0.01, 1e-4]:
        q = chi_bar_square_isf(alpha, 1)
        assert abs(q - norm.isf(alpha) ** 2) < 1e-9
        assert abs(chi_bar_square_sf(q, 1) - alpha) < 1e-10


def test_chi_bar_is_sharper_than_laurent_massart():
    for dimension in [1, 2, 5, 10]:
        for alpha in [0.05, 1e-3, 1e-6]:
            assert (
                chi_bar_glr_threshold(alpha, dimension)
                < laurent_massart_glr_threshold(alpha, dimension)
            )


def test_hybrid_accepts_sparse_and_dense_evidence():
    candidate = np.zeros(3)
    variance = np.ones(3)
    sparse = evaluate_boundary(
        candidate, np.array([2.0, -1.0, -1.0]), 100, 100,
        variance, variance, 0.1, method="hybrid"
    )
    dense = evaluate_boundary(
        candidate, np.array([0.8, 0.8, 0.8]), 100, 100,
        variance, variance, 0.1, method="hybrid"
    )
    assert sparse.crossed
    assert dense.crossed
