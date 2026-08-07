from pathlib import Path
import json
import numpy as np
import pytest

from pcpi_candidate_tas import GaussianCandidateInstance
from pcpi_candidate_tas.boundaries import (
    count_mass, count_mass_log_telescoping, count_mass_power,
    intrinsic_pair_alpha, chi_bar_square_sf, chi_bar_square_isf,
    laurent_massart_glr_threshold, chi_bar_glr_threshold,
    coordinate_z_threshold, standardised_gap_vector, evaluate_boundary,
    evaluate_all_challengers,
)
from pcpi_candidate_tas.candidate import (
    run_candidate_certification, run_theory_aligned_c_tracking,
)
from pcpi_candidate_tas.theory_status import one_sided_theory_status
from pcpi_candidate_tas.information import (
    challenger_information, information_vector, direct_quadratic_projection,
)
from pcpi_candidate_tas.models import GaussianCandidateInstance
from pcpi_candidate_tas.multiseason import (
    load_multiseason_subset, seasonal_windows, generate_multiseason_scenarios,
)
from pcpi_candidate_tas.official_repo_tns import run_official_repo_tns
from pcpi_candidate_tas.opsd_full import verify_official_source, chronological_split
from pcpi_candidate_tas.paired import (
    count_mass_log_telescoping as paired_mass, intrinsic_alpha,
    sample_mean_cov, hotelling_threshold, orthant_mahalanobis_distance,
    coordinate_t_statistics, evaluate_paired_samples, evaluate_archive,
    sequential_archive_confirmation,
)
from pcpi_candidate_tas.psi_comparators import (
    classify_full_pareto_coordinate, sequential_full_psi_coordinate,
    psips_inspired_posterior_classifier, imo_score_inspired_static_weights,
)


def easy_instance():
    return GaussianCandidateInstance.from_arrays(
        [[0.,0.],[1.4,-.2],[-.2,1.4]],np.ones((3,2))
    )


def test_information_geometry_contracts():
    gaps=np.array([[1.,-1.],[-.5,2.]])
    variances=np.ones((3,2))
    weights=np.array([.4,.3,.3])
    vec=information_vector(weights,gaps,variances)
    assert vec.shape==(2,) and np.all(vec>0)
    assert challenger_information(0,.2,gaps[0],variances[0],variances[1])==0
    assert challenger_information(.2,.2,[-1.,0.],variances[0],variances[1])==0
    with pytest.raises(ValueError): information_vector(np.array([.5,.5]),gaps,variances)
    cost,l0,li=direct_quadratic_projection(.4,.3,[0.,0.],[1.,-1.],[1.,1.],[1.,1.])
    assert cost>0 and np.isclose(l0[0],li[0]) and l0[1]==0 and li[1]==-1


def test_model_validation_and_properties():
    x=easy_instance()
    assert x.n_challengers==2 and x.n_objectives==2 and x.strict_positive_witness_candidate and x.strictly_nondominated_candidate
    assert x.gaps.shape==(2,2)
    for means,var,ids,names in [
        ([[0,0]],[[1,1]],('a',),('x','y')),
        ([[0,0],[1,1]],[[1,1]],('a','b'),('x','y')),
        ([[0,0],[np.nan,1]],[[1,1],[1,1]],('a','b'),('x','y')),
        ([[0,0],[1,1]],[[1,1],[0,1]],('a','b'),('x','y')),
        ([[0,0],[1,1]],[[1,1],[1,1]],('a','a'),('x','y')),
        ([[0,0],[1,1]],[[1,1],[1,1]],('a','b'),('x','x')),
    ]:
        with pytest.raises(ValueError):
            GaussianCandidateInstance(np.asarray(means),np.asarray(var),ids,names)


def test_boundary_validation_and_all_methods():
    assert count_mass_power(2)>0 and count_mass_log_telescoping(2)>0
    assert count_mass(2,'power',2)>0 and count_mass(2,'log_telescoping')>0
    with pytest.raises(ValueError): count_mass_power(0)
    with pytest.raises(ValueError): count_mass_log_telescoping(0)
    with pytest.raises(ValueError): count_mass(2,'bad')
    with pytest.raises(ValueError): intrinsic_pair_alpha(2,1,1)
    assert chi_bar_square_sf(0,2)>0 and chi_bar_square_sf(-1,2)==1
    assert chi_bar_square_isf(.1,2)>0
    with pytest.raises(ValueError): chi_bar_square_isf(0,2)
    assert laurent_massart_glr_threshold(.1,2)>0
    assert chi_bar_glr_threshold(.1,2,conservative_log_resolution=None)>0
    with pytest.raises(ValueError): chi_bar_glr_threshold(.1,2,conservative_log_resolution=0)
    assert coordinate_z_threshold(.1,2)>0
    with pytest.raises(ValueError): standardised_gap_vector([0],[1],0,1,[1],[1])
    with pytest.raises(ValueError): standardised_gap_vector([0],[1],1,1,[0],[0])
    for method in ['laurent_massart','chi_bar','coordinate','hybrid']:
        d=evaluate_boundary([0,0],[4,4],20,20,[1,1],[1,1],.2,method=method)
        assert d.crossed
    with pytest.raises(ValueError): evaluate_boundary([0],[1],2,2,[1],[1],.1,method='bad')
    with pytest.raises(ValueError): evaluate_boundary([0],[1],2,2,[1],[1],.1,method='hybrid',hybrid_cone_share=1)
    with pytest.raises(ValueError): evaluate_all_challengers(np.zeros((2,2)),np.ones(3),np.ones((2,2)),.1)


def test_candidate_policy_branches_and_caps(monkeypatch):
    instance=easy_instance()
    for policy in ['uniform','half_candidate','gap_racing','oracle_static']:
        result=run_candidate_certification(instance,delta=.1,policy=policy,seed=5,
            max_samples=30000,batch_size=20,oracle_update_every=100,
            true_oracle_weights=np.array([.4,.3,.3]) if policy=='oracle_static' else None)
        assert result.certified
    with pytest.raises(ValueError): run_candidate_certification(instance,policy='bad')
    with pytest.raises(ValueError): run_candidate_certification(instance,batch_size=0)
    hard=GaussianCandidateInstance.from_arrays([[0,0],[.01,-1],[-1,.01]],np.ones((3,2)))
    capped=run_candidate_certification(hard,delta=.01,policy='uniform',seed=1,max_samples=20,batch_size=5)
    assert not capped.certified
    import pcpi_candidate_tas.candidate as module
    monkeypatch.setattr(module,'solve_scalar_allocation',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('x')))
    with pytest.raises(RuntimeError):
        run_theory_aligned_c_tracking(instance,delta=.1,seed=1,max_samples=20,update_every=1,trace_every=1)
    fallback=run_theory_aligned_c_tracking(instance,delta=.1,seed=1,max_samples=20,update_every=1,trace_every=1,fail_on_oracle_error=False)
    assert fallback.oracle_failures>0


def test_paired_validation_sequential_and_singular_cases():
    rng=np.random.default_rng(9)
    x=rng.normal([.8,-.1],[.4,.4],size=(80,2))
    mean,cov=sample_mean_cov(x)
    assert mean.shape==(2,) and cov.shape==(2,2)
    assert paired_mass(2)>0 and intrinsic_alpha(.1,2)>0
    assert hotelling_threshold(.1,10,2)>0
    distance,rank=orthant_mahalanobis_distance(mean,cov,len(x))
    assert np.isfinite(distance) and rank==2
    singular=np.column_stack([x[:,0],x[:,0]])
    sm,sc=sample_mean_cov(singular)
    sd,srank=orthant_mahalanobis_distance(sm,sc,len(singular))
    assert np.isnan(sd) and srank<2
    stats=coordinate_t_statistics(mean,cov,len(x));assert stats.shape==(2,)
    for method in ['hotelling','coordinate','hybrid']:
        decision=evaluate_paired_samples(x,.1,method=method)
        assert decision.count==80
    with pytest.raises(ValueError): evaluate_paired_samples(x,.1,method='bad')
    with pytest.raises(ValueError): evaluate_paired_samples(x,.1,method='hybrid',hybrid_hotelling_share=1)
    differences=np.stack([x,x+np.array([.2,0])])
    cert,dec=evaluate_archive(differences,.1,method='hybrid')
    assert len(dec)==2
    seq=sequential_archive_confirmation(differences,.1,method='hybrid',min_count=5,max_count=80,check_every=7)
    assert seq['stopping_count']<=80
    with pytest.raises(ValueError): evaluate_archive(np.zeros((2,2)),.1)
    with pytest.raises(ValueError): sequential_archive_confirmation(differences,.1,check_every=0)


def test_psi_comparator_paths():
    rng=np.random.default_rng(3)
    means=np.array([[0,0],[1,-.2],[-.2,1],[2,2.]])
    outcomes=rng.normal(means,0.15,size=(180,4,2))
    dom,nd=classify_full_pareto_coordinate(outcomes[:100],.2)
    assert dom.shape==(4,) and nd.shape==(4,)
    seq=sequential_full_psi_coordinate(outcomes,.2,min_count=20,max_count=180,check_every=20)
    assert seq['stopping_count']<=180
    post=psips_inspired_posterior_classifier(outcomes,.3,draws=25,seed=2,min_count=20,max_count=80,check_every=20)
    assert post['estimated_pareto'].shape==(4,)
    weights=imo_score_inspired_static_weights(outcomes[:30])
    assert np.isclose(weights.sum(),1) and np.all(weights>0)


def test_multiseason_error_contracts(tmp_path):
    bad=tmp_path/'bad.csv';bad.write_text('a,b\n1,2\n')
    with pytest.raises(ValueError): load_multiseason_subset(bad)
    data=load_multiseason_subset(Path(__file__).parents[1]/'data/synthetic_multiseason/synthetic_multiseason_fixture.csv')
    with pytest.raises(ValueError): seasonal_windows(data,'development',1)
    with pytest.raises(ValueError): seasonal_windows(data,'missing',24)
    with pytest.raises(ValueError): generate_multiseason_scenarios(data,'confirmation',0,1)


def test_opsd_and_official_adapter_contracts(tmp_path):
    wrong=tmp_path/'wrong.csv';wrong.write_text('x')
    status=verify_official_source(wrong)
    assert status['exists'] and not status['valid'] and status['reason']=='sha256_mismatch'
    with pytest.raises(ValueError): chronological_split([{'utc_timestamp':'x'}],1)
    with pytest.raises(ValueError): run_official_repo_tns(np.zeros((2,3)))


def test_one_sided_theory_status_serialisation():
    status=one_sided_theory_status()
    assert status.to_dict()['expected_time_first_order_optimality_claimed'] is False

def test_paired_error_and_noncertification_paths():
    with pytest.raises(ValueError): paired_mass(1)
    with pytest.raises(ValueError): intrinsic_alpha(0,2)
    with pytest.raises(ValueError): sample_mean_cov(np.zeros((1,2)))
    with pytest.raises(ValueError): hotelling_threshold(.1,2,2)
    with pytest.raises(ValueError): orthant_mahalanobis_distance(np.zeros(2),np.eye(3),5)
    rng=np.random.default_rng(22)
    differences=np.stack([rng.normal([0,0],[1,1],size=(12,2))])
    result=sequential_archive_confirmation(differences,.01,method='hybrid',min_count=5,max_count=12,check_every=4)
    assert not result['certified'] and result['stopping_count']==12
