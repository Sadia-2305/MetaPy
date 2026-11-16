"""Independent meta-analysis calculator (does NOT import MetaPy). Formulas from primary sources."""
import math
import numpy as np
from scipy.stats import norm, t as tdist, chi2

def J(df):                      # Hedges small-sample correction (Hedges 1981)
    return 1 - 3 / (4 * df - 1)

def ppc2(mpre_i, mpost_i, sdpre_i, n_i, mpre_c, mpost_c, sdpre_c, n_c):
    """Morris (2008) eq. 8: g = c_p * [(Mpost_i-Mpre_i)-(Mpost_c-Mpre_c)] / SDpre_pooled"""
    sp = math.sqrt(((n_i - 1) * sdpre_i**2 + (n_c - 1) * sdpre_c**2) / (n_i + n_c - 2))
    return J(n_i + n_c - 2) * ((mpost_i - mpre_i) - (mpost_c - mpre_c)) / sp

def ppc2_var(g, n_i, n_c, r, variant="metapy"):
    """Morris (2008) eq. 25. variant='morris': Delta estimated by g everywhere (literal eq. 25).
       variant='metapy': d = g/J inside the bracket, then multiplied by J^2."""
    N = n_i + n_c; a = N / (n_i * n_c); c = J(N - 2)
    if variant == "morris":
        return 2 * c**2 * (1 - r) * a * (N - 2) / (N - 4) * (1 + g**2 / (2 * (1 - r) * a)) - g**2
    d = g / c
    return c**2 * (2 * (1 - r) * a * (N - 2) / (N - 4) * (1 + d**2 / (2 * (1 - r) * a)) - d**2 / c**2)

def endpoint_g(mi, sdi, ni, mc, sdc, nc):
    sp = math.sqrt(((ni - 1) * sdi**2 + (nc - 1) * sdc**2) / (ni + nc - 2))
    return J(ni + nc - 2) * (mi - mc) / sp

def var_g_ind(g, ni, nc):       # Borenstein 2009 eq 4.20/4.24 on g scale
    d = g / J(ni + nc - 2)
    return J(ni + nc - 2)**2 * ((ni + nc) / (ni * nc) + d**2 / (2 * (ni + nc)))

def reml(y, v, tol=1e-12, maxit=1000):
    """REML tau^2 by the fixed-point/Fisher-scoring iteration (Viechtbauer 2005), truncated at 0."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    tau2 = max(0.0, np.var(y, ddof=1) - v.mean())
    for _ in range(maxit):
        w = 1 / (v + tau2); mu = np.sum(w * y) / np.sum(w)
        new = np.sum(w**2 * ((y - mu)**2 - v)) / np.sum(w**2) + 1 / np.sum(w)
        new = max(0.0, new)
        if abs(new - tau2) < tol:
            return new
        tau2 = new
    return tau2

def dl(y, v):
    w = 1 / v; mu = np.sum(w * y) / np.sum(w); q = np.sum(w * (y - mu)**2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    return max(0.0, (q - (len(y) - 1)) / c)

def pm(y, v):
    from scipy.optimize import brentq
    k = len(y)
    f = lambda t2: np.sum((y - np.sum(y / (v + t2)) / np.sum(1 / (v + t2)))**2 / (v + t2)) - (k - 1)
    if f(0) <= 0: return 0.0
    hi = 1.0
    while f(hi) > 0: hi *= 2
    return brentq(f, 0, hi, xtol=1e-14)

def pool(y, v, tau2, hk=True, modified=True):
    y, v = np.asarray(y, float), np.asarray(v, float); k = len(y)
    w = 1 / (v + tau2); sw = w.sum(); mu = np.sum(w * y) / sw
    if hk:
        q = np.sum(w * (y - mu)**2) / (k - 1)
        if modified: q = max(q, 1.0)
        se = math.sqrt(q / sw); crit = tdist.ppf(0.975, k - 1); p = 2 * tdist.sf(abs(mu / se), k - 1)
    else:
        se = math.sqrt(1 / sw); crit = norm.ppf(0.975); p = 2 * norm.sf(abs(mu / se))
    w0 = 1 / v; mu0 = np.sum(w0 * y) / w0.sum(); Q = np.sum(w0 * (y - mu0)**2)
    s2 = (k - 1) * w0.sum() / (w0.sum()**2 - np.sum(w0**2))
    out = dict(mu=mu, se=se, lo=mu - crit * se, hi=mu + crit * se, p=p, tau2=tau2, Q=Q, Qp=chi2.sf(Q, k - 1),
               I2=100 * tau2 / (tau2 + s2), I2_Q=max(0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0, weights=100 * w / sw)
    if k >= 3:
        pse = math.sqrt(tau2 + se**2); pc = tdist.ppf(0.975, k - 2)
        out.update(pi_lo=mu - pc * pse, pi_hi=mu + pc * pse)
    return out
