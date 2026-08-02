from pathlib import Path
import numpy as np
from pcpi_candidate_tas.multiseason import (
    load_multiseason_subset, seasonal_windows, generate_multiseason_scenarios,
)

DATA=Path(__file__).parents[1]/"data/multiseason/dk1_2019_multiseason_mirror_subset.csv"

def test_multiseason_load_and_windows():
    data=load_multiseason_subset(DATA)
    assert len(data["price"])==285
    assert set(data["season"])=={"winter","spring","summer","autumn"}
    dev=seasonal_windows(data,"development")
    conf=seasonal_windows(data,"confirmation")
    assert len(dev)==4
    assert len(conf)>=4
    assert all(len(p)==24 and len(l)==24 for _,p,l in dev+conf)

def test_multiseason_scenarios_are_deterministic_and_finite():
    data=load_multiseason_subset(DATA)
    a=generate_multiseason_scenarios(data,"confirmation",12,7)
    b=generate_multiseason_scenarios(data,"confirmation",12,7)
    assert np.allclose(a[0],b[0]) and np.allclose(a[1],b[1])
    assert np.array_equal(a[2],b[2])
    assert a[0].shape==(12,24) and np.all(a[1]>0)
