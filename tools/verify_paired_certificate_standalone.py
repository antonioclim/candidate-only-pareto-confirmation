#!/usr/bin/env python3
"""Standalone verifier for PCPI paired candidate certificates.

The verifier intentionally does not import ``pcpi_candidate_tas``.  It
reconstructs the certificate directly from the JSON Schema, raw CSV and the
declared Hotelling/coordinate/hybrid boundary.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from jsonschema import Draft202012Validator
from scipy.optimize import nnls
from scipy.stats import f,t

def canonical_json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def count_mass(count):
    if count<2:raise ValueError("count >= 2 required")
    return math.log(3.0)*(1/math.log(count+1)-1/math.log(count+2))

def read_raw(path,challenger_ids,objective_names):
    rows={identifier:[] for identifier in challenger_ids}
    with open(path,newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle)
        expected=["challenger_id","scenario_index",*objective_names]
        if reader.fieldnames!=expected:raise ValueError("raw columns mismatch")
        for row in reader:
            identifier=row["challenger_id"]
            if identifier not in rows:raise ValueError("unknown challenger")
            rows[identifier].append((int(row["scenario_index"]),[float(row[x]) for x in objective_names]))
    counts={len(value) for value in rows.values()}
    if len(counts)!=1 or not counts or min(counts)<2:raise ValueError("balanced paired scenarios required")
    output=[]
    for identifier in challenger_ids:
        ordered=sorted(rows[identifier])
        if [index for index,_ in ordered]!=list(range(len(ordered))):raise ValueError("scenario index mismatch")
        output.append([value for _,value in ordered])
    return np.asarray(output,float)

def orthant_distance(mean,covariance,count):
    covariance=(covariance+covariance.T)/2
    rank=int(np.linalg.matrix_rank(covariance))
    if rank<len(mean):return math.nan,rank
    precision=np.linalg.inv(covariance)
    L=np.linalg.cholesky(precision);A=L.T
    x,_=nnls(A,-A@mean);residual=A@(mean+x)
    return float(count*np.dot(residual,residual)),rank

def decide(samples,delta,method,share):
    mean=samples.mean(axis=0);cov=np.cov(samples,rowvar=False,ddof=1).reshape(samples.shape[1],samples.shape[1])
    count,dimension=samples.shape;alpha=delta*count_mass(count)
    hot_alpha=alpha if method=="hotelling" else alpha*share
    coord_alpha=alpha if method=="coordinate" else alpha*(1-share)
    distance,rank=orthant_distance(mean,cov,count)
    hthr=None;hcross=False
    if count>dimension and rank==dimension and math.isfinite(distance):
        hthr=float(dimension*(count-1)/(count-dimension)*f.isf(hot_alpha,dimension,count-dimension))
        hcross=distance>hthr
    variances=np.diag(cov);statistics=np.full(dimension,-np.inf)
    good=np.isfinite(variances)&(variances>0)
    statistics[good]=mean[good]/np.sqrt(variances[good]/count)
    witness=int(np.argmax(statistics));tmax=float(statistics[witness])
    cthr=float(t.isf(coord_alpha/dimension,count-1));ccross=tmax>cthr
    crossed=hcross if method=="hotelling" else ccross if method=="coordinate" else hcross or ccross
    return {"crossed":bool(crossed),"hotelling_distance":None if not math.isfinite(distance) else distance,
        "hotelling_threshold":hthr,"coordinate_t_max":tmax,"coordinate_threshold":cthr,
        "witness_objective":witness,"covariance_rank":rank,"mean":mean,"covariance":cov}

def replay(artifact,raw_path,schema_path):
    schema=json.loads(Path(schema_path).read_text())
    Draft202012Validator(schema).validate(artifact)
    semantic=artifact["semantic_core"]
    if sha256_bytes(canonical_json(semantic))!=artifact["semantic_sha256"]:raise ValueError("semantic hash mismatch")
    if file_sha256(raw_path)!=semantic["raw_evidence"]["sha256"]:raise ValueError("raw evidence hash mismatch")
    claim=semantic["claim"]
    if claim["candidate_id"] in claim["challenger_ids"]:raise ValueError("candidate/challenger collision")
    raw=read_raw(raw_path,claim["challenger_ids"],claim["objective_names"])
    sufficient=semantic["sufficient_statistics"]
    if raw.shape[1]!=sufficient["count"]:raise ValueError("count mismatch")
    means=np.asarray([x.mean(axis=0) for x in raw]);covariances=np.asarray([np.cov(x,rowvar=False,ddof=1) for x in raw])
    if not np.allclose(means,sufficient["means"],rtol=0,atol=1e-10):raise ValueError("mean mismatch")
    if not np.allclose(covariances,sufficient["covariances"],rtol=0,atol=1e-10):raise ValueError("covariance mismatch")
    rule=semantic["stopping_rule"]
    decisions=[decide(x,claim["delta"],rule["method"],rule["hybrid_hotelling_share"]) for x in raw]
    certified=all(x["crossed"] for x in decisions);declared=semantic["certificate"]
    if declared["verdict"]!=("certified" if certified else "notCertified"):raise ValueError("verdict mismatch")
    exact={"crossed":[x["crossed"] for x in decisions],
        "witness_objective_indices":[x["witness_objective"] for x in decisions],
        "covariance_ranks":[x["covariance_rank"] for x in decisions]}
    for key,value in exact.items():
        if declared[key]!=value:raise ValueError(key+" mismatch")
    numeric={"hotelling_distances":[x["hotelling_distance"] for x in decisions],
        "hotelling_thresholds":[x["hotelling_threshold"] for x in decisions],
        "coordinate_t_max":[x["coordinate_t_max"] for x in decisions],
        "coordinate_thresholds":[x["coordinate_threshold"] for x in decisions]}
    for key,values in numeric.items():
        for left,right in zip(declared[key],values):
            if left is None or right is None:
                if left is not None or right is not None:raise ValueError(key+" mismatch")
            elif not np.isclose(float(left),float(right),rtol=0,atol=1e-10):
                raise ValueError(key+" mismatch")
    return {"valid":True,"certified":certified,"method":rule["method"],"stopping_count":int(raw.shape[1]),
            "verifier":"standalone-pcpi-paired-v1"}

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--artifact",required=True);parser.add_argument("--raw",required=True);parser.add_argument("--schema",required=True)
    args=parser.parse_args(argv)
    print(json.dumps(replay(json.loads(Path(args.artifact).read_text()),args.raw,args.schema),sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
