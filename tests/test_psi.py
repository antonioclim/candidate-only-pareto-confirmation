import numpy as np
from pcpi_candidate_tas.psi_comparators import *

def test_pareto_mask():
 y=np.array([[0,1],[1,0],[2,2]])
 assert pareto_mask(y).tolist()==[True,True,False]

def test_full_coordinate_easy():
 rng=np.random.default_rng(4);means=np.array([[0,1],[1,0],[2,2.]])
 x=rng.normal(means,0.2,size=(120,3,2));r=sequential_full_psi_coordinate(x,.1,min_count=10)
 assert r['completed'] and r['nondominated'][:2].all() and r['dominated'][2]

def test_static_weights():
 rng=np.random.default_rng(5);x=rng.normal(size=(30,5,2));w=imo_score_inspired_static_weights(x)
 assert np.isclose(w.sum(),1) and np.all(w>0)
