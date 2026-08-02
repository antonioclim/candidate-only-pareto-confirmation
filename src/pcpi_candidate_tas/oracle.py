"""Unique scalar allocation oracle."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq, minimize
from .models import AllocationResult
from .information import information_vector

def _active(gaps_i,var0,vari):
    mask=np.asarray(gaps_i,float)>0
    if not np.any(mask): raise ValueError("positive gap required")
    return np.asarray(gaps_i,float)[mask]**2, np.asarray(var0,float)[mask], np.asarray(vari,float)[mask]

def g_value(r,gaps_i,var0,vari):
    d2,v0,vi=_active(gaps_i,var0,vari)
    return float(.5*np.sum(r*d2/(vi+r*v0)))

def g_derivative(r,gaps_i,var0,vari):
    d2,v0,vi=_active(gaps_i,var0,vari)
    return float(.5*np.sum(d2*vi/(vi+r*v0)**2))

def g_limit(gaps_i,var0,vari):
    d2,v0,_=_active(gaps_i,var0,vari)
    return float(.5*np.sum(d2/v0))

def inverse_g(q,gaps_i,var0,vari,xtol=1e-12):
    lim=g_limit(gaps_i,var0,vari)
    if q<0 or q>=lim: raise ValueError("q outside inverse range")
    if q==0: return 0.0
    hi=1.0
    while g_value(hi,gaps_i,var0,vari)<=q:
        hi*=2
        if hi>1e18: raise RuntimeError("inverse bracket failed")
    return float(brentq(lambda r:g_value(r,gaps_i,var0,vari)-q,0,hi,xtol=xtol,maxiter=160))

def _rho_prime(q,gaps,variances,xtol):
    rho=np.empty(gaps.shape[0]); prime=np.empty(gaps.shape[0])
    for i in range(gaps.shape[0]):
        rho[i]=inverse_g(q,gaps[i],variances[0],variances[i+1],xtol)
        prime[i]=1/g_derivative(rho[i],gaps[i],variances[0],variances[i+1])
    return rho,prime

def solve_scalar_allocation(gaps,variances,tol=1e-10,inverse_tol=1e-12):
    gaps=np.asarray(gaps,float); variances=np.asarray(variances,float)
    if variances.shape!=(gaps.shape[0]+1,gaps.shape[1]): raise ValueError("shape mismatch")
    if np.any(np.max(gaps,axis=1)<=0): raise ValueError("strict separation required")
    qmax=min(g_limit(gaps[i],variances[0],variances[i+1]) for i in range(gaps.shape[0]))
    lo=max(np.finfo(float).eps,qmax*1e-14); hi=qmax*(1-1e-11)
    def h(q):
        rho,p=_rho_prime(q,gaps,variances,inverse_tol)
        return 1+rho.sum()-q*p.sum()
    if not (h(lo)>0 and h(hi)<0):
        for power in range(12,17):
            hi=qmax*(1-10**(-power))
            if h(hi)<0: break
        else: raise RuntimeError("outer root bracket failed")
    q,res=brentq(h,lo,hi,xtol=tol,maxiter=160,full_output=True)
    rho,_=_rho_prime(q,gaps,variances,inverse_tol)
    D=1+rho.sum(); w=np.r_[1/D,rho/D]; info=information_vector(w,gaps,variances)
    return AllocationResult(w,float(info.min()),float(q),info,float(tol),int(res.iterations))

def solve_generic_slsqp(gaps,variances):
    gaps=np.asarray(gaps,float); n=gaps.shape[0]
    w=np.full(n+1,1/(n+1)); z=.9*information_vector(w,gaps,variances).min()
    x0=np.r_[w,max(z,1e-12)]
    cons=[{"type":"eq","fun":lambda x:float(x[:-1].sum()-1)}]
    for i in range(n):
        cons.append({"type":"ineq","fun":lambda x,i=i:float(information_vector(x[:-1],gaps,variances)[i]-x[-1])})
    res=minimize(lambda x:-x[-1],x0,method="SLSQP",
                 bounds=[(1e-10,1)]*(n+1)+[(0,None)],constraints=cons,
                 options={"ftol":1e-12,"maxiter":5000})
    if not res.success: raise RuntimeError(res.message)
    info=information_vector(res.x[:-1],gaps,variances)
    return AllocationResult(res.x[:-1],float(info.min()),math.nan,info,math.nan,int(res.nit))
