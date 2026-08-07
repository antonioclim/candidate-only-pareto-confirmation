import json
from pathlib import Path
import numpy as np
import pytest
from pcpi_candidate_tas.boundaries import evaluate_boundary
from pcpi_candidate_tas.candidate import (
    _regularised_gaps, _requested_oracle_tolerances,
    _vanishing_regularisation_floor, run_candidate_certification,
    run_reference_target_tracking, run_theory_aligned_c_tracking,
)
from pcpi_candidate_tas.models import GaussianCandidateInstance
from pcpi_candidate_tas.oracle import (
    g_derivative, g_limit, g_value, inverse_g,
    solve_generic_slsqp, solve_scalar_allocation,
)
from pcpi_candidate_tas.theory_status import one_sided_theory_status

def test_chi_bar_zero_atom_uses_strict_crossing():
    decision=evaluate_boundary(
        np.array([0.0]),np.array([-1.0]),1,1,np.array([1.0]),np.array([1.0]),
        0.99,method="chi_bar",spending="power",exponent=3.0,
    )
    assert decision.cone_threshold==0.0
    assert decision.cone_glr==0.0
    assert not decision.crossed

def test_vanishing_regularisation_removes_fixed_floor_bias():
    means=np.array([[0.0],[5e-4]])
    true_gaps=means[1:]-means[0]
    fixed=_regularised_gaps(means,1e-3)
    assert fixed[0,0]==1e-3
    floor=_vanishing_regularisation_floor(10_000,1e-3,0.25)
    vanishing=_regularised_gaps(means,floor)
    assert floor<true_gaps[0,0]
    assert np.allclose(vanishing,true_gaps)

def test_requested_oracle_tolerance_decreases_to_documented_floor():
    early=_requested_oracle_tolerances(10,2e-10,0.25,1e-13)
    late=_requested_oracle_tolerances(1_000_000,2e-10,0.25,1e-13)
    extreme=_requested_oracle_tolerances(10**20,2e-10,0.25,1e-13)
    assert late[0]<early[0]
    assert extreme[0]==1e-13
    assert early[1]<early[0]

def test_scalar_oracle_first_order_root_and_equalisation():
    gaps=np.array([[0.7,-0.2,0.3],[0.2,0.6,-0.1],[0.5,0.1,0.4]])
    variances=np.array([[1.2,0.8,1.1],[0.9,1.4,0.7],[1.5,0.6,1.0],[0.8,1.3,1.2]])
    result=solve_scalar_allocation(gaps,variances)
    assert np.all(result.weights>0)
    assert abs(result.weights.sum()-1.0)<1e-12
    assert np.ptp(result.equalised_information)<1e-9
    generic=solve_generic_slsqp(gaps,variances)
    assert abs(result.rate-generic.rate)<1e-8
    q=result.q_star; rhos=result.weights[1:]/result.weights[0]
    primes=np.array([1/g_derivative(rhos[i],gaps[i],variances[0],variances[i+1]) for i in range(gaps.shape[0])])
    h=1+rhos.sum()-q*primes.sum()
    assert abs(h)<1e-7
    assert q<min(g_limit(gaps[i],variances[0],variances[i+1]) for i in range(gaps.shape[0]))

def _simulate(targets):
    counts=np.zeros(targets.shape[1]); cumulative=np.zeros(targets.shape[1]); hist=[]
    for target in targets:
        cumulative+=target; arm=int(np.argmax(cumulative-counts)); counts[arm]+=1; hist.append((cumulative-counts).copy())
    return np.asarray(hist)

def test_cumulative_target_general_bound_and_counterexample_to_unit_bound():
    data=json.loads((Path(__file__).parents[1]/"evidence/examples/tracking_counterexample_seed12345.json").read_text())
    deficits=_simulate(np.asarray(data["targets"],float))
    assert deficits.max()>1.0
    k=deficits.shape[1]
    assert deficits.max()<k-1.0
    assert deficits.min()>-1.0

def test_reference_policy_records_vanishing_floor_and_tolerance():
    instance=GaussianCandidateInstance.from_arrays([[0.,0.],[0.8,-0.1],[-0.1,0.8]],np.ones((3,2)))
    result=run_theory_aligned_c_tracking(instance,delta=0.1,seed=5,max_samples=20_000,update_every=10,trace_every=20)
    assert result.certified and result.oracle_failures==0
    floors=[e["regularisation_floor"] for e in result.trace]
    tols=[e["oracle_outer_tolerance"] for e in result.trace]
    assert floors and tols
    assert all(b<a for a,b in zip(floors,floors[1:]))
    assert all(b<=a for a,b in zip(tols,tols[1:]))

def test_theory_status_is_explicit():
    status=one_sided_theory_status()
    assert status.reference_regularisation_floor_vanishes
    assert status.reference_oracle_failures_are_fail_closed
    assert status.cumulative_target_positive_deficit_bound_is_k_minus_one
    assert status.chi_bar_zero_atom_uses_strict_crossing
    assert status.mathematical_reference_oracle_error_must_vanish
    assert status.executable_reference_requests_decreasing_oracle_tolerance
    assert not status.executable_finite_precision_is_literal_exact_theorem_object


def test_validation_branches_and_practical_fallback(monkeypatch):
    with pytest.raises(ValueError):
        _vanishing_regularisation_floor(0, 1e-3, 0.25)
    with pytest.raises(ValueError):
        _vanishing_regularisation_floor(10, -1.0, 0.25)
    with pytest.raises(ValueError):
        _vanishing_regularisation_floor(10, 1e-3, 0.0)
    with pytest.raises(ValueError):
        _requested_oracle_tolerances(0, 1e-8, 0.25, 1e-13)
    with pytest.raises(ValueError):
        _requested_oracle_tolerances(10, 0.0, 0.25, 1e-13)
    with pytest.raises(ValueError):
        _requested_oracle_tolerances(10, 1e-8, 0.0, 1e-13)
    with pytest.raises(ValueError):
        _requested_oracle_tolerances(10, 1e-8, 0.25, 0.0)

    gaps=np.array([0.5,-0.2]); var0=np.array([1.0,1.0]); vari=np.array([2.0,1.0])
    assert g_value(1.0,gaps,var0,vari)>0.0
    assert inverse_g(0.0,gaps,var0,vari)==0.0

    instance=GaussianCandidateInstance.from_arrays(
        [[0.,0.],[0.01,-1.0],[-1.0,0.01]], np.ones((3,2))
    )
    import pcpi_candidate_tas.candidate as candidate_module
    with monkeypatch.context() as patch:
        patch.setattr(
            candidate_module, 'solve_scalar_allocation',
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('forced')),
        )
        result=run_candidate_certification(
            instance, delta=.01, policy='plugin_track', seed=2,
            max_samples=26, batch_size=5, oracle_update_every=1,
            trace_every=1,
        )
        assert result.oracle_failures>0
        assert result.trace

    alias_result=run_reference_target_tracking(
        instance, delta=.01, seed=3, max_samples=6,
        update_every=1, trace_every=0,
    )
    assert alias_result.stopping_time==6


def test_log_q_fallback_handles_extreme_relative_scale() -> None:
    """The outer optimum can lie more than 1e-14 below qmax.

    This deterministic regression case previously failed because the outer
    bracket was expressed as a fixed relative fraction of qmax.  The hybrid
    raw-q/log-q solver must return a finite, normalised allocation.
    """
    gaps = np.array([
        [
            4.0861120806364095,
            -27.566956021031263,
            1.167603003075863,
            16.157310491215153,
            1.7199716115743844,
        ]
    ])
    variances = np.array([
        [
            0.00018282787215184276,
            515.0486780617111,
            2.1142382618171633,
            122796096.0682303,
            1.0896843046682315e-11,
        ],
        [
            219330508.07552138,
            3.0708245460201837,
            37344881780.921524,
            2.2454069600817356e-07,
            18498248505.619316,
        ],
    ])
    result = solve_scalar_allocation(gaps, variances, tol=1e-11, inverse_tol=1e-13)
    assert np.all(np.isfinite(result.weights))
    assert np.isfinite(result.rate)
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-14)
    assert result.q_star / g_limit(gaps[0], variances[0], variances[1]) < 1e-14
