"""Adapter for a pinned source-equivalent extraction of the PSIPS-repository
Python Track-and-Stop path.

It is not the C++ PSIPS posterior-sampling executable and is never labelled
official PSIPS.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os,sys
import numpy as np
PINNED_COMMIT="0b78a2fb7c30a997eb6c76b0ff0290274856e681"
@dataclass(frozen=True)
class OfficialRepoTnSResult:
    estimated_pareto:tuple[int,...];correct:bool;total_arm_pulls:int;stopped:bool
    pinned_commit:str=PINNED_COMMIT
    comparator_label:str="psips-repository-python-track-and-stop"
def _load():
    root=Path(__file__).resolve().parents[2]
    parent=Path(os.environ.get("PCPI_PSIPS_SOURCE",root/"third_party"))
    if not (parent/"psips_pinned").exists():
        raise RuntimeError("Pinned source unavailable; set PCPI_PSIPS_SOURCE")
    sys.path.insert(0,str(parent))
    from psips_pinned.ext.track_and_stop import track_and_stop
    return track_and_stop
def run_official_repo_tns(means_minimisation,delta=.1,covariance=None,seed=0,max_steps=80000):
    means=np.asarray(means_minimisation,float)
    if means.ndim!=2 or means.shape[1]!=2:raise ValueError("pinned comparator extraction is limited to two objectives")
    cov=np.eye(2) if covariance is None else np.asarray(covariance,float)
    ans,correct,pulls,stopped=_load()(-means,np.log(1/delta),tracking="C",seed=int(seed),cov=cov,max_steps=int(max_steps))
    return OfficialRepoTnSResult(tuple(map(int,np.asarray(ans).tolist())),correct,pulls,stopped)
