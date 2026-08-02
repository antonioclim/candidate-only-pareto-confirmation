from pathlib import Path
import numpy as np
from pcpi_candidate_tas.battery import *
ROOT=Path(__file__).resolve().parents[1]

def test_fixture_and_archive():
 d=load_opsd_fixture(ROOT/'data/opsd_fixture/opsd_dk1_2019_01_01_01_06_fixture.csv')
 assert len(d['price'])==144 and (d['split']=='development').sum()==72
 p,l=generate_scenarios(d,'development',10,1)
 out=evaluate_archive(policy_archive(),p,l)
 assert out.shape==(10,13,3) and np.all(np.isfinite(out))

def test_candidate_differences():
 d=load_opsd_fixture(ROOT/'data/opsd_fixture/opsd_dk1_2019_01_01_01_06_fixture.csv')
 p,l=generate_scenarios(d,'development',20,2);o=evaluate_archive(policy_archive(),p,l);c=select_compromise_candidate(o)
 diff,ch=candidate_differences(o,c)
 assert diff.shape==(12,20,3) and c not in ch
