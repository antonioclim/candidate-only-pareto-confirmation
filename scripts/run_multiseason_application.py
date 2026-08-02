from pathlib import Path
import argparse,csv,json,sys,time,math
import numpy as np
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"retargeted_kit/src"))
from pcpi_candidate_tas.multiseason import load_multiseason_subset,generate_multiseason_scenarios
from pcpi_candidate_tas.battery import policy_archive,evaluate_archive,select_compromise_candidate,candidate_differences,pareto_mask
from pcpi_candidate_tas.paired import sequential_archive_confirmation
from pcpi_candidate_tas.psi_comparators import sequential_full_psi_coordinate

OUT=ROOT/"evidence/powered"
REPLICATIONS=100
DEV_SCENARIOS=220
CONF_SCENARIOS=600
CAP_CANDIDATE=600
CAP_FULL=500

def bootstrap_median(values,seed,draws=4000):
    x=np.asarray(values,float);rng=np.random.default_rng(seed)
    sims=np.median(rng.choice(x,size=(draws,len(x)),replace=True),axis=1)
    return float(np.quantile(sims,.025)),float(np.quantile(sims,.975))

def required_paired_n(differences,alpha=.05,power=.80):
    x=np.asarray(differences,float);sd=float(x.std(ddof=1));effect=abs(float(x.mean()))
    if effect==0:return math.inf
    return int(math.ceil(((norm.ppf(1-alpha/2)+norm.ppf(power))*sd/effect)**2))

def run_chunk(start,count):
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_multiseason_subset(ROOT/"data/multiseason/dk1_2019_multiseason_mirror_subset.csv")
    policies=policy_archive();rows=[];started=time.time()
    for rep in range(start,min(start+count,REPLICATIONS)):
        p,l,dev_seasons=generate_multiseason_scenarios(data,"development",DEV_SCENARIOS,10000+rep)
        development=evaluate_archive(policies,p,l)
        candidate=select_compromise_candidate(development)
        p,l,conf_seasons=generate_multiseason_scenarios(data,"confirmation",CONF_SCENARIOS,20000+rep)
        confirmation=evaluate_archive(policies,p,l)
        differences,_=candidate_differences(confirmation,candidate)
        empirical_pareto=bool(pareto_mask(confirmation.mean(axis=0))[candidate])
        for method in ["coordinate","hybrid","hotelling"]:
            result=sequential_archive_confirmation(differences,.05,method=method,min_count=8,max_count=CAP_CANDIDATE,check_every=5)
            rows.append({"replicate":rep,"candidate_policy":policies[candidate].policy_id,
                "method":method,"decision_scope":"candidate_only","completed":bool(result["certified"]),
                "stopping_count":int(result["stopping_count"]),"cap":CAP_CANDIDATE,
                "empirical_heldout_candidate_pareto":empirical_pareto,
                "development_season_count":len(set(dev_seasons)),
                "confirmation_season_count":len(set(conf_seasons))})
        full=sequential_full_psi_coordinate(confirmation,.05,min_count=12,max_count=CAP_FULL,check_every=10)
        rows.append({"replicate":rep,"candidate_policy":policies[candidate].policy_id,
            "method":"full_psi_coordinate_racing","decision_scope":"full_pareto",
            "completed":bool(full["completed"]),"stopping_count":int(full["stopping_count"]),
            "cap":CAP_FULL,"empirical_heldout_candidate_pareto":empirical_pareto,
            "development_season_count":len(set(dev_seasons)),
            "confirmation_season_count":len(set(conf_seasons))})
    path=OUT/f"application_chunk_{start:03d}_{min(start+count,REPLICATIONS):03d}.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({"chunk":path.name,"replicates":len(set(r["replicate"] for r in rows)),
                      "rows":len(rows),"elapsed_seconds":time.time()-started},indent=2))

def aggregate():
    rows=[]
    for path in sorted(OUT.glob("application_chunk_*.csv")):
        with path.open(newline="",encoding="utf-8") as f:rows.extend(csv.DictReader(f))
    for r in rows:
        for key in ["replicate","stopping_count","cap","development_season_count","confirmation_season_count"]:
            r[key]=int(r[key])
        for key in ["completed","empirical_heldout_candidate_pareto"]:
            r[key]=r[key]=="True"
    reps=sorted(set(r["replicate"] for r in rows))
    if reps!=list(range(REPLICATIONS)):raise RuntimeError(f"incomplete chunks: {len(reps)}")
    raw=OUT/"multiseason_application_raw.csv"
    with raw.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for method in ["coordinate","hybrid","hotelling","full_psi_coordinate_racing"]:
        cell=[r for r in rows if r["method"]==method];vals=[r["stopping_count"] for r in cell]
        lo,hi=bootstrap_median(vals,77+len(method))
        summary.append({"method":method,"decision_scope":cell[0]["decision_scope"],"runs":len(cell),
            "completion_rate":sum(r["completed"] for r in cell)/len(cell),
            "median_stopping_count":float(np.median(vals)),"median_bootstrap_ci_low":lo,
            "median_bootstrap_ci_high":hi,"mean_stopping_count":float(np.mean(vals)),
            "restricted_mean_at_cap":float(np.mean(np.minimum(vals,[r["cap"] for r in cell]))),
            "empirical_candidate_pareto_rate":sum(r["empirical_heldout_candidate_pareto"] for r in cell)/len(cell)})
    with (OUT/"multiseason_application_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    by={(r["replicate"],r["method"]):r["stopping_count"] for r in rows}
    paired=[by[(rep,"full_psi_coordinate_racing")]-by[(rep,"hybrid")] for rep in reps]
    candidate_counts={}
    for rep in reps:
        policy=next(r["candidate_policy"] for r in rows if r["replicate"]==rep)
        candidate_counts[policy]=candidate_counts.get(policy,0)+1
    metadata={"replications":REPLICATIONS,"development_scenarios":DEV_SCENARIOS,
        "confirmation_scenarios":CONF_SCENARIOS,"candidate_cap":CAP_CANDIDATE,
        "full_set_cap":CAP_FULL,"primary_paired_contrast":"full_psi_coordinate_racing minus hybrid candidate-only",
        "pilot_replications":20,"normal_approximation_required_n_for_paired_mean_difference":required_paired_n(paired[:20]),
        "precision_rationale":"n=100 gives a worst-case 95% binomial half-width of approximately 0.098 and exceeds the pilot paired-effect requirement.",
        "candidate_policy_counts":candidate_counts,
        "data_status":"four-season pinned public-mirror subset; not the official complete OPSD package",
        "data_sha256":json.loads((ROOT/"data/multiseason/MULTISEASON_PROVENANCE.json").read_text())["output_sha256"],
        "critical_qualification":"Full-set and candidate-only methods answer different questions; stopping counts quantify decision-scope cost rather than universal algorithmic superiority."}
    (OUT/"multiseason_application_metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--start",type=int);parser.add_argument("--count",type=int,default=20);parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args()
    aggregate() if args.aggregate else run_chunk(args.start,args.count)
