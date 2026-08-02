from pathlib import Path
import argparse,csv,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"retargeted_kit/src"))
from pcpi_candidate_tas import GaussianCandidateInstance
from pcpi_candidate_tas.candidate import run_theory_aligned_c_tracking,run_candidate_certification

OUT=ROOT/"evidence/tracker"
REPLICATIONS=20

def instances():
    return {
        "balanced":GaussianCandidateInstance.from_arrays(
            [[0,0],[.50,-.20],[-.20,.50],[.45,-.15],[-.15,.45]],np.ones((5,2))),
        "heterogeneous":GaussianCandidateInstance.from_arrays(
            [[0,0],[.25,-.20],[-.20,.55],[.40,-.10],[-.10,.40]],
            np.array([[2.0,2.0],[1,1],[1,2],[2,1],[1,1]],float)),
    }

def bootstrap_median(x,seed,draws=3000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    z=np.median(rng.choice(x,size=(draws,len(x)),replace=True),axis=1)
    return float(np.quantile(z,.025)),float(np.quantile(z,.975))

def run_chunk(start,count):
    OUT.mkdir(parents=True,exist_ok=True);rows=[];started=time.time()
    for instance_index,(name,instance) in enumerate(instances().items()):
        for rep in range(start,min(start+count,REPLICATIONS)):
            seed=30000+1000*instance_index+rep
            t=time.time()
            reference=run_theory_aligned_c_tracking(instance,delta=.05,boundary_method="hybrid",
                seed=seed,max_samples=30000,update_every=250,trace_every=250)
            discrepancy=max([x.get("maximum_positive_deficit",0.0) for x in reference.trace] or [0.0])
            rows.append({"instance":name,"replicate":rep,"method":"theory_aligned_c_tracking",
                "certified":reference.certified,"stopping_time":reference.stopping_time,
                "wall_seconds":time.time()-t,"oracle_failures":reference.oracle_failures,
                "max_positive_tracking_deficit":discrepancy,"theorem_covered":True})
            for method,policy in [("batched_plugin","plugin_track"),("uniform","uniform"),("oracle_static","oracle_static")]:
                t=time.time()
                result=run_candidate_certification(instance,delta=.05,policy=policy,boundary_method="hybrid",
                    seed=seed,max_samples=30000,batch_size=25,oracle_update_every=250)
                rows.append({"instance":name,"replicate":rep,"method":method,
                    "certified":result.certified,"stopping_time":result.stopping_time,
                    "wall_seconds":time.time()-t,"oracle_failures":result.oracle_failures,
                    "max_positive_tracking_deficit":np.nan,"theorem_covered":False})
    path=OUT/f"tracker_chunk_{start:03d}_{min(start+count,REPLICATIONS):03d}.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({"chunk":path.name,"rows":len(rows),"elapsed_seconds":time.time()-started},indent=2))

def aggregate():
    rows=[]
    for path in sorted(OUT.glob("tracker_chunk_*.csv")):
        with path.open(newline="",encoding="utf-8") as f:rows.extend(csv.DictReader(f))
    for r in rows:
        r["replicate"]=int(r["replicate"]);r["stopping_time"]=int(r["stopping_time"])
        r["wall_seconds"]=float(r["wall_seconds"]);r["oracle_failures"]=int(r["oracle_failures"])
        r["certified"]=r["certified"]=="True";r["theorem_covered"]=r["theorem_covered"]=="True"
        r["max_positive_tracking_deficit"]=float(r["max_positive_tracking_deficit"])
    for name in instances():
        if sorted(set(r["replicate"] for r in rows if r["instance"]==name))!=list(range(REPLICATIONS)):
            raise RuntimeError(f"incomplete chunks for {name}")
    with (OUT/"tracker_alignment_raw.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for name in instances():
        for method in ["theory_aligned_c_tracking","batched_plugin","uniform","oracle_static"]:
            cell=[r for r in rows if r["instance"]==name and r["method"]==method]
            vals=[r["stopping_time"] for r in cell];lo,hi=bootstrap_median(vals,500+len(name)+len(method))
            summary.append({"instance":name,"method":method,"runs":len(cell),
                "completion_rate":sum(r["certified"] for r in cell)/len(cell),
                "median_stopping_time":float(np.median(vals)),"median_ci_low":lo,"median_ci_high":hi,
                "mean_stopping_time":float(np.mean(vals)),
                "median_wall_seconds":float(np.median([r["wall_seconds"] for r in cell])),
                "oracle_failure_rate":sum(r["oracle_failures"]>0 for r in cell)/len(cell),
                "maximum_recorded_positive_tracking_deficit":float(np.nanmax([r["max_positive_tracking_deficit"] for r in cell])) if method=="theory_aligned_c_tracking" else np.nan,
                "theorem_covered":cell[0]["theorem_covered"]})
    with (OUT/"tracker_alignment_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    meta={"replications_per_cell":REPLICATIONS,"instances":list(instances()),
        "methods":["theory_aligned_c_tracking","batched_plugin","uniform","oracle_static"],
        "primary_theory_mapping":"Only theory_aligned_c_tracking is covered by the inherited C-tracking theorem. Batched methods are engineering approximations."}
    (OUT/"tracker_alignment_metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--start",type=int);parser.add_argument("--count",type=int,default=5);parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args();aggregate() if args.aggregate else run_chunk(args.start,args.count)
