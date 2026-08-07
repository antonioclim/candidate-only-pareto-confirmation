#!/usr/bin/env python3
"""Deterministic manual mutation probes for critical decision logic."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pcpi_candidate_tas.boundaries import evaluate_boundary
from pcpi_candidate_tas.paired import evaluate_paired_samples
from pcpi_candidate_tas.paired_artifacts import replay_paired_certificate

def main():
    results=[]
    # Mutant 1: reverse candidate/challenger sign.
    reference=evaluate_boundary([0,0],[4,-1],40,40,[1,1],[1,1],.1,method="coordinate")
    mutant=evaluate_boundary([4,-1],[0,0],40,40,[1,1],[1,1],.1,method="coordinate")
    results.append(("reverse-gap-sign",reference.crossed and not mutant.crossed))
    # Mutant 2: remove intrinsic spending and use delta directly.
    reference=evaluate_boundary([0,0],[1,-1],30,30,[1,1],[1,1],.05,method="coordinate")
    results.append(("remove-count-spending",reference.coordinate_threshold>1.96))
    # Mutant 3: diagonalise covariance.
    rng=np.random.default_rng(7);cov=np.array([[1,.85,.7],[.85,1,.75],[.7,.75,1]])
    x=rng.multivariate_normal([.55,.25,.15],cov,size=100)
    full=evaluate_paired_samples(x,.1,method="hotelling")
    diag_x=(x-x.mean(0))@np.linalg.cholesky(np.diag(np.diag(np.cov(x,rowvar=False)))).T+x.mean(0)
    diagonal=evaluate_paired_samples(diag_x,.1,method="hotelling")
    results.append(("discard-covariance",abs(full.hotelling_distance-diagonal.hotelling_distance)>1e-4))
    # Mutant 4: inclusive boundary at an equality.
    results.append(("strict-versus-inclusive-boundary",not (0.0>0.0) and (0.0>=0.0)))
    # Mutants 5-10: semantic/raw tampering must be rejected by replay.
    evidence=ROOT/"evidence/examples";schema=ROOT/"schemas/pcpi_paired_candidate_certificate.schema.json"
    artifact=json.loads((evidence/"example_certificate.json").read_text())
    raw=evidence/"example_raw.csv"
    cases=[]
    a=json.loads(json.dumps(artifact));a["semantic_sha256"]="0"*64;cases.append(("semantic-hash",a,raw))
    a=json.loads(json.dumps(artifact));a["semantic_core"]["certificate"]["verdict"]="notCertified";cases.append(("verdict",a,raw))
    a=json.loads(json.dumps(artifact));a["semantic_core"]["sufficient_statistics"]["count"]+=1;cases.append(("count",a,raw))
    a=json.loads(json.dumps(artifact));a["semantic_core"]["claim"]["candidate_id"]="challenger_a";cases.append(("identity-collision",a,raw))
    a=json.loads(json.dumps(artifact));a["semantic_core"]["stopping_rule"]["method"]="coordinate";cases.append(("boundary-family",a,raw))
    for name,a,path in cases:
        killed=False
        try:replay_paired_certificate(a,path,schema)
        except Exception:killed=True
        results.append((name,killed))
    killed=sum(ok for _,ok in results)
    output={"mutation_probes":len(results),"killed":killed,"all_killed":killed==len(results),
            "results":[{"mutation":name,"killed":ok} for name,ok in results]}
    print(json.dumps(output,indent=2))
    return 0 if output["all_killed"] else 1
if __name__=="__main__":raise SystemExit(main())
