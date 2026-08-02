"""Comparator layer for paired multi-objective outcomes.

The coordinate-racing method is a valid but conservative full-Pareto-set
comparator.  The posterior classifier is PSIPS-inspired only: it is not the
official PSIPS implementation and carries no state-of-the-art or frequentist
optimality claim.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import t
from .paired import count_mass_log_telescoping


def _pair_bounds(diff:np.ndarray,alpha:float) -> tuple[np.ndarray,np.ndarray]:
    n=diff.shape[0];mean=diff.mean(axis=0);sd=diff.std(axis=0,ddof=1)
    r=t.isf(alpha/(2*diff.shape[1]),n-1)*sd/np.sqrt(n)
    return mean-r,mean+r


def classify_full_pareto_coordinate(outcomes:np.ndarray,delta:float) -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(outcomes,float);n,arms,m=x.shape
    mass=count_mass_log_telescoping(n)
    # simultaneous intervals over unordered pairs and objectives
    pairs=arms*(arms-1)//2
    alpha=delta*mass/max(pairs,1)
    dominated=np.zeros(arms,bool);nondominated=np.zeros(arms,bool)
    lower=np.empty((arms,arms,m));upper=np.empty_like(lower)
    lower.fill(np.nan);upper.fill(np.nan)
    for a in range(arms):
        for b in range(a+1,arms):
            lo,up=_pair_bounds(x[:,a]-x[:,b],alpha)
            lower[a,b]=lo;upper[a,b]=up
            lower[b,a]=-up;upper[b,a]=-lo
    for a in range(arms):
        for b in range(arms):
            if a==b:continue
            # b dominates a if outcome_b - outcome_a <= 0 in every objective.
            if np.all(upper[b,a] <= 0): dominated[a]=True;break
        if dominated[a]:continue
        nondominated[a]=all(b==a or np.any(lower[b,a]>0) for b in range(arms))
    return dominated,nondominated


def sequential_full_psi_coordinate(outcomes:np.ndarray,delta:float,*,min_count:int=10,max_count:int|None=None,check_every:int=1) -> dict:
    x=np.asarray(outcomes,float);end=len(x) if max_count is None else min(len(x),max_count)
    start=max(min_count,3);grid=list(range(start,end+1,check_every));
    if not grid or grid[-1]!=end:grid.append(end)
    for s in grid:
        dom,nd=classify_full_pareto_coordinate(x[:s],delta)
        if np.all(dom|nd):return {"completed":True,"stopping_count":s,"dominated":dom,"nondominated":nd}
    dom,nd=classify_full_pareto_coordinate(x[:end],delta)
    return {"completed":False,"stopping_count":end,"dominated":dom,"nondominated":nd}


def pareto_mask(means:np.ndarray) -> np.ndarray:
    y=np.asarray(means,float);n=len(y);mask=np.ones(n,bool)
    for i in range(n):
        for j in range(n):
            if i!=j and np.all(y[j]<=y[i]) and np.any(y[j]<y[i]):mask[i]=False;break
    return mask


def psips_inspired_posterior_classifier(outcomes:np.ndarray,delta:float,*,draws:int=500,seed:int=0,min_count:int=12,max_count:int|None=None,check_every:int=5) -> dict:
    """Research heuristic, not the official PSIPS algorithm."""
    x=np.asarray(outcomes,float);rng=np.random.default_rng(seed);end=len(x) if max_count is None else min(len(x),max_count)
    arms=x.shape[1];m=x.shape[2]
    start=max(min_count,m+3);grid=list(range(start,end+1,check_every));
    if not grid or grid[-1]!=end:grid.append(end)
    labels=np.zeros((draws,arms),bool)
    for s in grid:
        mean=x[:s].mean(axis=0)
        labels=np.zeros((draws,arms),bool)
        for a in range(arms):
            cov=np.cov(x[:s,a],rowvar=False,ddof=1).reshape(m,m)
            cov=(cov+cov.T)/2 + 1e-9*np.eye(m)
            samples=rng.multivariate_normal(mean[a],cov/s,size=draws)
            if a==0:post=np.empty((draws,arms,m));post[:,a]=samples
            else:post[:,a]=samples
        for d in range(draws):labels[d]=pareto_mask(post[d])
        p=labels.mean(axis=0)
        decided=np.maximum(p,1-p)>=1-delta/max(arms,1)
        if np.all(decided):return {"completed":True,"stopping_count":s,"pareto_probability":p,"estimated_pareto":p>.5}
    p=labels.mean(axis=0)
    return {"completed":False,"stopping_count":end,"pareto_probability":p,"estimated_pareto":p>.5}


def imo_score_inspired_static_weights(pilot_outcomes:np.ndarray) -> np.ndarray:
    """An explicit non-official hardness heuristic inspired by SCORE allocation."""
    x=np.asarray(pilot_outcomes,float);means=x.mean(axis=0);arms=len(means)
    hardness=np.ones(arms)
    for a in range(arms):
        gaps=[]
        for b in range(arms):
            if a!=b:gaps.append(np.max(np.abs(means[a]-means[b])))
        hardness[a]=1/max(min(gaps),1e-3)**2
    return hardness/hardness.sum()
