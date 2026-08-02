"""Small, transparent battery-policy confirmation fixture.

This is deliberately not a full market optimiser.  It creates a finite archive
of causal threshold policies, selects one policy on development scenarios and
confirms it on independent OPSD-derived confirmation scenarios.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np

@dataclass(frozen=True)
class BatteryPolicy:
    policy_id: str
    charge_quantile: float
    discharge_quantile: float
    load_reserve_quantile: float
    aggressiveness: float


def policy_archive() -> tuple[BatteryPolicy,...]:
    policies=[BatteryPolicy("policy_00_idle",0.0,1.0,1.0,0.0)]
    idx=1
    for cq,dq,rq,a in [
        (.15,.75,.90,.50),(.15,.80,.85,.75),(.20,.75,.90,.75),(.20,.80,.85,1.00),
        (.25,.70,.90,.50),(.25,.75,.85,.75),(.25,.80,.80,1.00),(.30,.70,.90,.75),
        (.30,.75,.85,1.00),(.35,.70,.85,.75),(.35,.75,.80,1.00),(.40,.70,.80,1.00)]:
        policies.append(BatteryPolicy(f"policy_{idx:02d}",cq,dq,rq,a));idx+=1
    return tuple(policies)


def load_opsd_fixture(path:Path|str) -> dict:
    rows=[]
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    price=np.array([float(r['DK_1_price_day_ahead_EUR_MWh']) for r in rows])
    load=np.array([float(r['DK_1_load_actual_entsoe_transparency_MWh']) for r in rows])
    split=np.array([r['split'] for r in rows])
    return {"price":price,"load":load,"split":split,
            "timestamps":np.array([r['utc_timestamp'] for r in rows])}


def _base_days(data:dict,split:str) -> tuple[np.ndarray,np.ndarray]:
    mask=data['split']==split
    price=data['price'][mask]
    load=data['load'][mask]
    if len(price)%24: raise ValueError("whole days required")
    return price.reshape(-1,24),load.reshape(-1,24)


def generate_scenarios(data:dict,split:str,count:int,seed:int,*,price_noise:float=.055,load_noise:float=.025) -> tuple[np.ndarray,np.ndarray]:
    p,l=_base_days(data,split);rng=np.random.default_rng(seed)
    prices=[];loads=[]
    for _ in range(count):
        day=int(rng.integers(len(p)))
        # Paired CRN scenario shared by every policy.  Smooth within-day shocks
        # preserve the observed daily profile while creating independent blocks.
        hour_noise=rng.normal(size=24)
        smooth=np.convolve(hour_noise,np.ones(3)/3,mode='same')
        ps=p[day]*(1+price_noise*smooth)+rng.normal(0,1.5,size=24)
        ls=l[day]*(1+load_noise*smooth)+rng.normal(0,15,size=24)
        prices.append(ps);loads.append(np.clip(ls,1,None))
    return np.asarray(prices),np.asarray(loads)


def simulate_policy(policy:BatteryPolicy,price:np.ndarray,load:np.ndarray,*,capacity:float=10.0,pmax:float=4.0,eta:float=.95) -> np.ndarray:
    price=np.asarray(price,float);load=np.asarray(load,float)
    site=load/load.max()*9.0
    low=float(np.quantile(price,policy.charge_quantile)) if policy.aggressiveness else -np.inf
    high=float(np.quantile(price,policy.discharge_quantile)) if policy.aggressiveness else np.inf
    reserve=float(np.quantile(site,policy.load_reserve_quantile)) if policy.aggressiveness else np.inf
    energy=.5*capacity;throughput=0.;grid=[]
    for p,l in zip(price,site):
        charge=discharge=0.
        if policy.aggressiveness and p<=low and energy<.9*capacity:
            charge=min(pmax*policy.aggressiveness,(.9*capacity-energy)/eta)
        elif policy.aggressiveness and (p>=high or l>=reserve) and energy>.1*capacity:
            discharge=min(pmax*policy.aggressiveness,(energy-.1*capacity)*eta,l)
        energy += eta*charge-discharge/eta
        throughput += charge+discharge
        grid.append(max(0.,l+charge-discharge))
    grid=np.asarray(grid)
    cost=float(np.dot(price,grid))
    peak=float(grid.max())
    return np.array([cost,peak,throughput],float)


def evaluate_archive(policies:tuple[BatteryPolicy,...],prices:np.ndarray,loads:np.ndarray) -> np.ndarray:
    out=np.empty((len(prices),len(policies),3))
    for s in range(len(prices)):
        for a,policy in enumerate(policies):out[s,a]=simulate_policy(policy,prices[s],loads[s])
    return out


def pareto_mask(means:np.ndarray) -> np.ndarray:
    y=np.asarray(means,float);n=len(y);mask=np.ones(n,bool)
    for i in range(n):
        for k in range(n):
            if i!=k and np.all(y[k]<=y[i]) and np.any(y[k]<y[i]):mask[i]=False;break
    return mask


def select_compromise_candidate(outcomes:np.ndarray) -> int:
    means=outcomes.mean(axis=0);pm=pareto_mask(means);indices=np.flatnonzero(pm)
    lo=means.min(axis=0);hi=means.max(axis=0);scaled=(means-lo)/np.maximum(hi-lo,1e-12)
    # Cost and peak are primary; throughput is a smaller degradation proxy.
    score=.45*scaled[:,0]+.40*scaled[:,1]+.15*scaled[:,2]
    return int(indices[np.argmin(score[indices])])


def candidate_differences(outcomes:np.ndarray,candidate:int) -> tuple[np.ndarray,tuple[int,...]]:
    challengers=tuple(i for i in range(outcomes.shape[1]) if i!=candidate)
    # challenger - candidate in minimisation orientation
    return np.stack([outcomes[:,i]-outcomes[:,candidate] for i in challengers]),challengers
