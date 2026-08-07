"""Unique scalar allocation oracle."""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
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

def _log_g_from_log_r(log_r,gaps_i,var0,vari):
    """Return log(g(exp(log_r))) in a scale-stable form."""
    d2,v0,vi=_active(gaps_i,var0,vari)
    log_terms=(math.log(.5)+np.log(d2)+log_r
               -np.logaddexp(np.log(vi),log_r+np.log(v0)))
    return float(logsumexp(log_terms))


def _log_deficit(log_r,gaps_i,var0,vari):
    """Return log(g_limit - g(exp(log_r))) without catastrophic cancellation."""
    d2,v0,vi=_active(gaps_i,var0,vari)
    log_terms=(math.log(.5)+np.log(d2)+np.log(vi)-np.log(v0)
               -np.logaddexp(np.log(vi),log_r+np.log(v0)))
    return float(logsumexp(log_terms))


def inverse_g_log(q,gaps_i,var0,vari,xtol=1e-12):
    """Logarithm of the unique ratio ``r`` satisfying ``g(r)=q``.

    The root is bracketed around the appropriate first-order asymptote in
    log-ratio space.  Unlike a fixed lower bound such as ``log(r)=-745``, this
    remains valid when the mathematically required ratio is smaller than the
    least positive representable floating-point number.  Near the finite upper
    endpoint a log-deficit equation avoids catastrophic cancellation.
    """
    lim=g_limit(gaps_i,var0,vari)
    if q<0 or q>=lim: raise ValueError("q outside inverse range")
    if q==0: return -math.inf
    tolerance=max(float(xtol),4*np.finfo(float).eps)
    d2,v0,vi=_active(gaps_i,var0,vari)

    if q<=.5*lim:
        log_target=math.log(q)
        # g(r) ~ A_0 r as r -> 0, with A_0=.5*sum(d2/vi).
        log_A0=float(logsumexp(math.log(.5)+np.log(d2)-np.log(vi)))
        centre=log_target-log_A0
        f=lambda u:_log_g_from_log_r(u,gaps_i,var0,vari)-log_target
        lo,hi=centre-8.0,centre+8.0
        step=8.0
        for _ in range(80):
            flo,fhi=f(lo),f(hi)
            if flo<=0<=fhi: break
            if flo>0: lo-=step
            if fhi<0: hi+=step
            step*=1.5
        else: raise RuntimeError("log inverse bracket failed")
    else:
        target=lim-q
        if target<=0: raise ValueError("q outside inverse range")
        log_target=math.log(target)
        # lim-g(r) ~ A_inf/r as r -> infinity.
        log_Ainf=float(logsumexp(math.log(.5)+np.log(d2)+np.log(vi)-2*np.log(v0)))
        centre=log_Ainf-log_target
        f=lambda u:_log_deficit(u,gaps_i,var0,vari)-log_target
        lo,hi=centre-8.0,centre+8.0
        step=8.0
        for _ in range(80):
            flo,fhi=f(lo),f(hi)
            if flo>=0>=fhi: break
            if flo<0: lo-=step
            if fhi>0: hi+=step
            step*=1.5
        else: raise RuntimeError("log inverse bracket failed")

    flo,fhi=f(lo),f(hi)
    if flo==0: return float(lo)
    if fhi==0: return float(hi)
    if flo*fhi>0: raise RuntimeError("log inverse bracket failed")
    return float(brentq(f,lo,hi,xtol=tolerance,maxiter=240))


def inverse_g(q,gaps_i,var0,vari,xtol=1e-12):
    log_r=inverse_g_log(q,gaps_i,var0,vari,xtol)
    if log_r==-math.inf: return 0.0
    if log_r>math.log(np.finfo(float).max): return math.inf
    return float(math.exp(log_r))


def _log_g_derivative_from_log_r(log_r,gaps_i,var0,vari):
    d2,v0,vi=_active(gaps_i,var0,vari)
    log_terms=(math.log(.5)+np.log(d2)+np.log(vi)
               -2*np.logaddexp(np.log(vi),log_r+np.log(v0)))
    return float(logsumexp(log_terms))


def _rho_logs_and_prime_logs(q,gaps,variances,xtol):
    log_rho=np.empty(gaps.shape[0]); log_prime=np.empty(gaps.shape[0])
    for i in range(gaps.shape[0]):
        log_rho[i]=inverse_g_log(q,gaps[i],variances[0],variances[i+1],xtol)
        log_prime[i]=-_log_g_derivative_from_log_r(
            log_rho[i],gaps[i],variances[0],variances[i+1]
        )
    return log_rho,log_prime


def solve_scalar_allocation(gaps,variances,tol=1e-10,inverse_tol=1e-12):
    gaps=np.asarray(gaps,float); variances=np.asarray(variances,float)
    if variances.shape!=(gaps.shape[0]+1,gaps.shape[1]): raise ValueError("shape mismatch")
    if np.any(np.max(gaps,axis=1)<=0): raise ValueError("strict separation required")
    qmax=min(g_limit(gaps[i],variances[0],variances[i+1]) for i in range(gaps.shape[0]))
    if not np.isfinite(qmax) or qmax<=0: raise ValueError("positive finite information limit required")

    def balance(q):
        # Root of q D'(q)/D(q)-1.  Log-domain sums remain stable even when
        # some candidate/challenger ratios are outside the representable raw
        # scale.
        log_rho,log_prime=_rho_logs_and_prime_logs(q,gaps,variances,inverse_tol)
        log_D=float(logsumexp(np.r_[0.0,log_rho]))
        log_ratio=math.log(q)+float(logsumexp(log_prime))-log_D
        if log_ratio>50: return math.exp(50)-1
        return math.expm1(log_ratio)

    # Fast path for ordinary scales: solve directly in q as in the original
    # scalar oracle.  The log-q fallback below removes the arbitrary relative
    # lower bracket only when the fast path cannot enclose the root.
    lo=max(np.nextafter(0.0,1.0),qmax*1e-14)
    hi=np.nextafter(qmax,0.0)
    flo=balance(lo); fhi=balance(hi)
    bracketed=bool(flo<0 and fhi>0)
    if not bracketed:
        for power in range(13,4,-1):
            lo=max(np.nextafter(0.0,1.0),qmax*10.0**(-power))
            hi=qmax*(1.0-10.0**(-power))
            flo=balance(lo); fhi=balance(hi)
            if flo<0 and fhi>0:
                bracketed=True
                break

    relative_tol=max(4*np.finfo(float).eps,min(float(tol),1e-6))
    if bracketed:
        q,res=brentq(
            balance,lo,hi,xtol=np.finfo(float).tiny,rtol=relative_tol,
            maxiter=240,full_output=True
        )
    else:
        def balance_from_log_q(log_q):
            return balance(math.exp(log_q))

        log_qmax=math.log(qmax)
        log_hi=None; log_fhi=None
        for power in range(4,15):
            fraction=1.0-10.0**(-power)
            candidate=log_qmax+math.log(fraction)
            value=balance_from_log_q(candidate)
            if value>0:
                log_hi,log_fhi=candidate,value
                break
        if log_hi is None:
            raise RuntimeError("outer root upper bracket failed")

        log_tiny=math.log(np.nextafter(0.0,1.0))
        log_lo=log_hi-8.0
        log_flo=balance_from_log_q(log_lo)
        step=8.0
        for _ in range(120):
            if log_flo<0: break
            log_lo-=step
            if log_lo<=log_tiny:
                log_lo=log_tiny
                log_flo=balance_from_log_q(log_lo)
                break
            step=min(step*1.25,64.0)
            log_flo=balance_from_log_q(log_lo)
        if not (log_flo<0 and log_fhi>0):
            raise RuntimeError("outer root lower bracket failed")

        log_tolerance=max(4*np.finfo(float).eps,min(float(tol),1e-6))
        log_q,res=brentq(
            balance_from_log_q,log_lo,log_hi,xtol=log_tolerance,
            rtol=max(4*np.finfo(float).eps,log_tolerance),
            maxiter=240,full_output=True
        )
        q=math.exp(log_q)
    log_rho,_=_rho_logs_and_prime_logs(q,gaps,variances,inverse_tol)
    log_D=float(logsumexp(np.r_[0.0,log_rho]))
    log_weights=np.r_[-log_D,log_rho-log_D]
    w=np.exp(log_weights)
    w/=w.sum()
    # At the mathematical root every challenger information equals w0*q.
    rate=float(math.exp(math.log(q)-log_D))
    info=np.full(gaps.shape[0],rate,dtype=float)
    return AllocationResult(w,rate,float(q),info,float(tol),int(res.iterations))

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
