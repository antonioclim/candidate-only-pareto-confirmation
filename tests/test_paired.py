import numpy as np
from pcpi_candidate_tas.paired import *

def test_count_spending_partial_mass():
 assert 0.8<sum(count_mass_log_telescoping(s) for s in range(2,100000))<1.0

def test_hotelling_crosses_easy_positive():
 rng=np.random.default_rng(1);x=rng.multivariate_normal([.8,.3],[[1,.3],[.3,1]],size=80)
 assert evaluate_paired_samples(x,.1,method='hotelling').crossed

def test_hotelling_does_not_cross_null():
 rng=np.random.default_rng(2);x=rng.multivariate_normal([-.2,-.1],np.eye(2),size=80)
 assert not evaluate_paired_samples(x,.05,method='hotelling').crossed

def test_coordinate_sparse():
 rng=np.random.default_rng(3);x=rng.multivariate_normal([.9,-.3,-.2],np.eye(3),size=60)
 assert evaluate_paired_samples(x,.1,method='coordinate').crossed

def test_hybrid_accepts_either():
 rng=np.random.default_rng(4);x=rng.multivariate_normal([.7,-.2],np.eye(2),size=100)
 assert evaluate_paired_samples(x,.1,method='hybrid').crossed

def test_singular_hotelling_refuses():
 x=np.column_stack([np.arange(10),np.arange(10)])
 d=evaluate_paired_samples(x,.05,method='hotelling')
 assert not d.crossed and d.covariance_rank<2
