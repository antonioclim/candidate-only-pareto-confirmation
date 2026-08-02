from pathlib import Path
import argparse,csv,json,sys,time
import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"retargeted_kit/src"))
from pcpi_candidate_tas.paired import sequential_archive_confirmation
OUT=ROOT/"evidence/robustness"
MODELS=["gaussian_iid","student_t5","contaminated_2pct","ar1_0.5","near_singular_rho_0.99","heteroskedastic"]
METHODS=["coordinate","hybrid","hotelling"]
NULL_PATHS=1000
POWER_PATHS=300
COUNT=200

def draw(model,rng,count,mean):
    m=len(mean)
    if model=="gaussian_iid":return rng.normal(mean,1,size=(count,m))
    if model=="student_t5":return mean+rng.standard_t(5,size=(count,m))*np.sqrt(3/5)
    if model=="contaminated_2pct":
        x=rng.normal(mean,1,size=(count,m));mask=rng.random(count)<.02
        x[mask]+=rng.normal(0,8,size=(mask.sum(),m));return x
    if model=="ar1_0.5":
        innovation=rng.normal(size=(count,m));x=np.empty_like(innovation);x[0]=mean+innovation[0]
        for s in range(1,count):x[s]=mean+.5*(x[s-1]-mean)+np.sqrt(.75)*innovation[s]
        return x
    if model=="near_singular_rho_0.99":
        cov=np.full((m,m),.99);np.fill_diagonal(cov,1)
        return rng.multivariate_normal(mean,cov,size=count)
    if model=="heteroskedastic":
        scale=rng.lognormal(mean=-.5*.35**2,sigma=.35,size=(count,1))
        return mean+rng.normal(size=(count,m))*scale
    raise ValueError(model)

def cp_upper(successes,n,confidence=.95):
    return 1.0 if successes==n else float(beta.ppf(confidence,successes+1,n-successes))

def run_model(model):
    OUT.mkdir(parents=True,exist_ok=True);model_index=MODELS.index(model);rows=[];started=time.time()
    for method_index,method in enumerate(METHODS):
        false=0
        for rep in range(NULL_PATHS):
            rng=np.random.default_rng(400000+100000*model_index+1000*method_index+rep)
            x=draw(model,rng,COUNT,np.zeros(3))
            result=sequential_archive_confirmation(x[None,:,:],.05,method=method,min_count=8,max_count=COUNT,check_every=10)
            false+=int(result["certified"])
        completed=[];stopping=[]
        for rep in range(POWER_PATHS):
            rng=np.random.default_rng(800000+100000*model_index+1000*method_index+rep)
            x=draw(model,rng,COUNT,np.array([.38,-.08,-.08]))
            result=sequential_archive_confirmation(x[None,:,:],.05,method=method,min_count=8,max_count=COUNT,check_every=5)
            completed.append(bool(result["certified"]));stopping.append(result["stopping_count"])
        rows.append({"model":model,"method":method,"null_paths":NULL_PATHS,
            "false_certifications":false,"false_certification_rate":false/NULL_PATHS,
            "one_sided_cp_upper_95":cp_upper(false,NULL_PATHS),"power_paths":POWER_PATHS,
            "alternative_completion_rate":sum(completed)/POWER_PATHS,
            "alternative_median_stopping_count":float(np.median(stopping)),
            "alternative_mean_stopping_count":float(np.mean(stopping)),
            "declared_model_valid":model in {"gaussian_iid","near_singular_rho_0.99"}})
    path=OUT/f"robustness_{model}.csv"
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({"model":model,"elapsed_seconds":time.time()-started,"rows":rows},indent=2))

def aggregate():
    rows=[]
    for model in MODELS:
        path=OUT/f"robustness_{model}.csv"
        with path.open(newline="",encoding="utf-8") as f:rows.extend(csv.DictReader(f))
    with (OUT/"robustness_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    meta={"models":MODELS,"methods":METHODS,"null_paths_per_cell":NULL_PATHS,
        "alternative_paths_per_cell":POWER_PATHS,"maximum_count":COUNT,
        "total_null_paths":NULL_PATHS*len(MODELS)*len(METHODS),
        "total_alternative_paths":POWER_PATHS*len(MODELS)*len(METHODS),
        "interpretation":"Gaussian results are calibration checks. Heavy-tail, contamination, dependence and heteroskedastic cells are stress tests outside the theorem and define the applicability boundary."}
    (OUT/"robustness_metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--model",choices=MODELS);parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args();aggregate() if args.aggregate else run_model(args.model)
