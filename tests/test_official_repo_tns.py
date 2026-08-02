import os
from pathlib import Path
import numpy as np
import pytest
from pcpi_candidate_tas.official_repo_tns import run_official_repo_tns

def test_pinned_repo_tns_source_contract():
    root=Path(__file__).resolve().parents[1]
    source_available=(root/'third_party'/'psips_pinned').exists() or bool(os.environ.get('PCPI_PSIPS_SOURCE'))
    means=np.array([[.2,.8],[.8,.2],[.45,.45],[.9,.9]])
    if source_available:
        r=run_official_repo_tns(means,delta=.2,seed=3,max_steps=80000)
        assert r.stopped and r.correct and set(r.estimated_pareto)=={0,1,2}
    else:
        with pytest.raises(RuntimeError,match='Pinned source unavailable'):
            run_official_repo_tns(means,delta=.2,seed=3,max_steps=80000)
