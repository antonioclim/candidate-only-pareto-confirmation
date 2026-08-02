import numpy as np
from pcpi_candidate_tas.oracle import (
    solve_generic_slsqp, solve_scalar_allocation,
)


def test_scalar_oracle_matches_generic_solver():
    rng = np.random.default_rng(52)
    for _ in range(5):
        gaps = rng.uniform(0.1, 1.2, size=(5, 3))
        variances = rng.uniform(0.4, 2.0, size=(6, 3))
        scalar = solve_scalar_allocation(gaps, variances)
        generic = solve_generic_slsqp(gaps, variances)
        assert abs(scalar.rate - generic.rate) < 1e-8
        assert np.ptp(scalar.equalised_information) < 1e-9
