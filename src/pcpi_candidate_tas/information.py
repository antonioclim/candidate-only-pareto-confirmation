"""Information geometry for candidate-only dominance alternatives."""
from __future__ import annotations
import numpy as np

def challenger_information(w0, wi, gaps_i, var0, vari) -> float:
    gaps_i = np.asarray(gaps_i, dtype=float)
    var0 = np.asarray(var0, dtype=float)
    vari = np.asarray(vari, dtype=float)
    mask = gaps_i > 0
    if not np.any(mask) or w0 <= 0 or wi <= 0:
        return 0.0
    d = gaps_i[mask]
    return float(0.5 * np.sum(w0 * wi * d*d / (w0*vari[mask] + wi*var0[mask])))

def information_vector(weights, gaps, variances):
    weights = np.asarray(weights, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if weights.shape != (gaps.shape[0] + 1,):
        raise ValueError("incompatible weights")
    return np.array([
        challenger_information(weights[0], weights[i+1], gaps[i],
                               variances[0], variances[i+1])
        for i in range(gaps.shape[0])
    ])

def direct_quadratic_projection(w0, wi, mu0, mui, var0, vari):
    mu0=np.asarray(mu0,float); mui=np.asarray(mui,float)
    var0=np.asarray(var0,float); vari=np.asarray(vari,float)
    lam0=mu0.copy(); lami=mui.copy(); mask=mui>mu0
    if np.any(mask):
        denom=w0/var0[mask]+wi/vari[mask]
        pooled=(w0*mu0[mask]/var0[mask]+wi*mui[mask]/vari[mask])/denom
        lam0[mask]=pooled; lami[mask]=pooled
    cost=.5*(w0*np.sum((mu0-lam0)**2/var0)+wi*np.sum((mui-lami)**2/vari))
    return float(cost),lam0,lami
