"""Strict paired-certificate artefact with raw-data binding and replay."""
from __future__ import annotations
from pathlib import Path
import csv,hashlib,json
import numpy as np
from jsonschema import Draft202012Validator
from .paired import evaluate_archive

def canonical_json(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def sha256_bytes(b):return hashlib.sha256(b).hexdigest()
def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def write_raw_paired(path,differences,challenger_ids,objective_names):
    d=np.asarray(differences,float)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['challenger_id','scenario_index',*objective_names])
        for i,cid in enumerate(challenger_ids):
            for s,row in enumerate(d[i]):w.writerow([cid,s,*[format(float(v),'.17g') for v in row]])

def read_raw_paired(path,challenger_ids,objective_names):
    index={x:i for i,x in enumerate(challenger_ids)};rows={x:[] for x in challenger_ids}
    with open(path,newline='',encoding='utf-8') as f:
        r=csv.DictReader(f)
        if r.fieldnames!=['challenger_id','scenario_index',*objective_names]:raise ValueError('raw columns mismatch')
        for z in r:
            if z['challenger_id'] not in index:raise ValueError('unknown challenger')
            rows[z['challenger_id']].append((int(z['scenario_index']),[float(z[o]) for o in objective_names]))
    counts={len(v) for v in rows.values()}
    if len(counts)!=1 or not counts or min(counts)<2:raise ValueError('balanced paired scenarios required')
    out=[]
    for cid in challenger_ids:
        seq=sorted(rows[cid]);
        if [x[0] for x in seq]!=list(range(len(seq))):raise ValueError('scenario index mismatch')
        out.append([x[1] for x in seq])
    return np.asarray(out,float)

def build_paired_certificate(differences,delta,candidate_id,challenger_ids,objective_names,raw_path,*,method='hybrid',hybrid_hotelling_share=.5):
    certified,dec=evaluate_archive(differences,delta,method=method,hybrid_hotelling_share=hybrid_hotelling_share)
    sem={'specification':'pcpi-paired-candidate-certificate/0.6',
         'claim':{'type':'candidateStrictlyParetoNondominatedInMean','candidate_id':candidate_id,'challenger_ids':list(challenger_ids),'objective_names':list(objective_names),'orientation':'minimise','delta':float(delta),'model':'iidPairedMultivariateNormalUnknownPositiveDefiniteCovariance'},
         'stopping_rule':{'method':method,'spending':'log_telescoping','hybrid_hotelling_share':float(hybrid_hotelling_share)},
         'sufficient_statistics':{'count':int(differences.shape[1]),'means':[np.mean(x,axis=0).tolist() for x in differences],'covariances':[np.cov(x,rowvar=False,ddof=1).reshape(differences.shape[2],differences.shape[2]).tolist() for x in differences]},
         'certificate':{'verdict':'certified' if certified else 'notCertified','crossed':[x.crossed for x in dec],'hotelling_distances':[x.hotelling_distance for x in dec],'hotelling_thresholds':[x.hotelling_threshold for x in dec],'coordinate_t_max':[x.coordinate_t_max for x in dec],'coordinate_thresholds':[x.coordinate_threshold for x in dec],'witness_objective_indices':[x.witness_objective for x in dec],'covariance_ranks':[x.covariance_rank for x in dec]},
         'raw_evidence':{'media_type':'text/csv','sha256':file_sha256(raw_path),'relative_path':Path(raw_path).name}}
    return {'semantic_core':sem,'semantic_sha256':sha256_bytes(canonical_json(sem)),'execution_metadata':{'generator':'pcpi-candidate-certification','generator_version':'1.1.0'}}

def replay_paired_certificate(artifact,raw_path,schema_path=None):
    if schema_path:Draft202012Validator(json.loads(Path(schema_path).read_text())).validate(artifact)
    sem=artifact['semantic_core']
    if sha256_bytes(canonical_json(sem))!=artifact['semantic_sha256']:raise ValueError('semantic hash mismatch')
    if file_sha256(raw_path)!=sem['raw_evidence']['sha256']:raise ValueError('raw evidence hash mismatch')
    c=sem['claim'];d=read_raw_paired(raw_path,c['challenger_ids'],c['objective_names'])
    if d.shape[1]!=sem['sufficient_statistics']['count']:raise ValueError('count mismatch')
    means=np.asarray([x.mean(axis=0) for x in d]);cov=np.asarray([np.cov(x,rowvar=False,ddof=1) for x in d])
    if not np.allclose(means,sem['sufficient_statistics']['means'],rtol=0,atol=1e-10):raise ValueError('mean mismatch')
    if not np.allclose(cov,sem['sufficient_statistics']['covariances'],rtol=0,atol=1e-10):raise ValueError('covariance mismatch')
    rule=sem['stopping_rule'];certified,dec=evaluate_archive(d,c['delta'],method=rule['method'],hybrid_hotelling_share=rule['hybrid_hotelling_share'])
    cert=sem['certificate']
    if cert['verdict']!=('certified' if certified else 'notCertified'):raise ValueError('verdict mismatch')
    for key,vals in [('crossed',[x.crossed for x in dec]),('witness_objective_indices',[x.witness_objective for x in dec]),('covariance_ranks',[x.covariance_rank for x in dec])]:
        if cert[key]!=vals:raise ValueError(key+' mismatch')
    numeric={
      'hotelling_distances':[x.hotelling_distance for x in dec],
      'hotelling_thresholds':[x.hotelling_threshold for x in dec],
      'coordinate_t_max':[x.coordinate_t_max for x in dec],
      'coordinate_thresholds':[x.coordinate_threshold for x in dec],
    }
    for key,vals in numeric.items():
        declared=cert[key]
        if len(declared)!=len(vals):raise ValueError(key+' length mismatch')
        for a,b in zip(declared,vals):
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError(key+' mismatch')
            elif not np.isclose(float(a),float(b),rtol=0,atol=1e-10):raise ValueError(key+' mismatch')
    if c['candidate_id'] in c['challenger_ids']:raise ValueError('candidate/challenger collision')
    return {'valid':True,'certified':certified,'method':rule['method'],'stopping_count':int(d.shape[1])}
