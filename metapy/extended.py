"""Extended statistical methods (MetaPy 5.1).

Heterogeneity estimators : Hedges (HE), Hunter-Schmidt (HS, HSk), Sidik-Jonkman (SJ), maximum likelihood (ML),
                           empirical Bayes (EB), Paule-Mandel with moderators (PM)   (Viechtbauer 2005; DerSimonian &
                           Kacker 2007; Sidik & Jonkman 2005; Morris 1983; Paule & Mandel 1982)
Binary data, 2x2 tables  : Mantel-Haenszel odds ratio, risk ratio and risk difference (Mantel & Haenszel 1959;
                           Robins, Breslow & Greenland 1986; Greenland & Robins 1985; Sato, Greenland & Robins 1989)
                           and Peto's one-step odds ratio (Yusuf et al. 1985)
Selection models         : step-function selection models with any number of cut-points (Vevea & Hedges 1995)
Influence                : externally standardized residuals, DFFITS, Cook's distance, covariance ratio, leave-one-out
                           tau^2 and Q, hat values, DFBETAS (Viechtbauer & Cheung 2010)
Small-study effects      : test of excess significance (Ioannidis & Trikalinos 2007)

All functions take plain arrays and return dictionaries or data frames, so they can be tested in isolation.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import binom, chi2, norm

Z975 = float(norm.ppf(0.975))

TAU2_METHODS = {
    "REML": "restricted maximum likelihood",
    "ML": "maximum likelihood",
    "DL": "DerSimonian-Laird",
    "PM": "Paule-Mandel",
    "EB": "empirical Bayes",
    "HE": "Hedges",
    "HS": "Hunter-Schmidt",
    "HSk": "Hunter-Schmidt with small-sample correction",
    "SJ": "Sidik-Jonkman",
}


# ======================================================================= helpers
def _X(k, X):
    return np.ones((k, 1)) if X is None else np.asarray(X, float)


def _proj(w, X):
    """P = W - W X (X'WX)^-1 X'W for weights w."""
    W = np.diag(w)
    return W - W @ X @ np.linalg.solve(X.T @ W @ X, X.T @ W)


def _rss(y, v, X, tau2):
    return float(y @ _proj(1.0 / (v + tau2), X) @ y)


# ======================================================================= tau^2 estimators
def tau2_he(y, v, X=None):
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = _X(len(y), X)
    k, p = X.shape
    P = np.eye(k) - X @ np.linalg.solve(X.T @ X, X.T)          # ordinary least squares residual projector
    return max(0.0, (float(y @ P @ y) - float(np.sum(np.diag(P) * v))) / (k - p))


def tau2_hs(y, v, X=None, small_sample=False):
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = _X(len(y), X)
    k, p = X.shape
    w = 1 / v
    q = float(y @ _proj(w, X) @ y)
    if small_sample:
        q = q * k / (k - p)
    return max(0.0, (q - k) / float(np.sum(w)))


def tau2_sj(y, v, X=None, tau2_init=None):
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = _X(len(y), X)
    k, p = X.shape
    t0 = float(np.var(y, ddof=1) * (k - 1) / k) if tau2_init is None else float(tau2_init)
    if t0 <= 0:
        t0 = 1e-8
    return float(t0 * _rss(y, v, X, t0) / (k - p))


def _loglik(tau2, y, v, X, reml):
    w = 1 / (v + tau2)
    XtWX = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    r = y - X @ beta
    ll = -0.5 * (np.sum(np.log(v + tau2)) + np.sum(w * r ** 2))
    if reml:
        ll -= 0.5 * np.linalg.slogdet(XtWX)[1]
    return float(ll)


def tau2_ml(y, v, X=None, reml=False):
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = _X(len(y), X)
    k, p = X.shape
    if k - p < 1:
        return 0.0
    f = lambda t: -_loglik(t, y, v, X, reml)
    hi = max(1.0, float(np.var(y)) * 10 + float(np.max(v)) * 10)
    grid = np.concatenate([[0.0], np.geomspace(1e-6, hi, 60)])
    vals = [f(g) for g in grid]
    i = int(np.argmin(vals))
    lo_b, hi_b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    if i == 0:
        lo_b, hi_b = 0.0, grid[1]
    opt = minimize_scalar(f, bounds=(lo_b, hi_b), method="bounded", options={"xatol": 1e-12})
    cands = [(0.0, f(0.0)), (float(opt.x), float(opt.fun))]
    return min(cands, key=lambda c: c[1])[0]


def tau2_pm(y, v, X=None):
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = _X(len(y), X)
    k, p = X.shape
    if k - p < 1:
        return 0.0
    g = lambda t: _rss(y, v, X, t) - (k - p)
    if g(0.0) <= 0:
        return 0.0
    hi = max(1.0, float(np.var(y)) * 10)
    while g(hi) > 0:
        hi *= 2
    return float(brentq(g, 0.0, hi, xtol=1e-14))


def tau2_eb(y, v, X=None, maxiter=1000, tol=1e-12):
    """Morris (1983) empirical Bayes estimator, by fixed-point iteration (as in metafor)."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = _X(len(y), X)
    k, p = X.shape
    if k - p < 1:
        return 0.0
    tau2 = max(0.0, tau2_he(y, v, X))
    for _ in range(maxiter):
        w = 1 / (v + tau2)
        beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        r = y - X @ beta
        new = max(0.0, float(np.sum(w * (k / (k - p) * r ** 2 - v)) / np.sum(w)))
        if abs(new - tau2) < tol:
            tau2 = new
            break
        tau2 = new
    return tau2


def tau2_dl(y, v, X=None):
    y, v = np.asarray(y, float), np.asarray(v, float)
    X = _X(len(y), X)
    k, p = X.shape
    P = _proj(1 / v, X)
    return max(0.0, (float(y @ P @ y) - (k - p)) / float(np.trace(P)))


def estimate_tau2(y, v, method="REML", X=None):
    m = method.upper() if method not in ("HSk",) else method
    if m in ("HSK",):
        m = "HSk"
    if m == "REML":
        return tau2_ml(y, v, X, reml=True)
    if m == "ML":
        return tau2_ml(y, v, X)
    if m == "DL":
        return tau2_dl(y, v, X)
    if m == "PM":
        return tau2_pm(y, v, X)
    if m == "EB":
        return tau2_eb(y, v, X)
    if m == "HE":
        return tau2_he(y, v, X)
    if m == "HS":
        return tau2_hs(y, v, X)
    if m == "HSk":
        return tau2_hs(y, v, X, small_sample=True)
    if m == "SJ":
        return tau2_sj(y, v, X)
    raise ValueError(f"Unknown tau^2 estimator: {method}")


def compare_tau2(y, v) -> pd.DataFrame:
    """Random-effects estimate under every tau^2 estimator (Wald inference), as a sensitivity table."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    w0 = 1 / v
    k = len(y)
    s2 = (k - 1) * float(np.sum(w0)) / (float(np.sum(w0)) ** 2 - float(np.sum(w0 ** 2)))
    rows = []
    for m, label in TAU2_METHODS.items():
        t2 = estimate_tau2(y, v, m)
        w = 1 / (v + t2)
        mu = float(np.sum(w * y) / np.sum(w))
        se = math.sqrt(1 / np.sum(w))
        rows.append({"estimator": label, "code": m, "tau2": t2, "tau": math.sqrt(t2), "I2": 100 * t2 / (t2 + s2),
                     "estimate": mu, "se": se, "ci_lo": mu - Z975 * se, "ci_hi": mu + Z975 * se})
    return pd.DataFrame(rows)


# ======================================================================= Mantel-Haenszel and Peto
def _tables(ai, bi, ci, di):
    a, b, c, d = (np.asarray(x, float) for x in (ai, bi, ci, di))
    return a, b, c, d, a + b, c + d, a + b + c + d


def _escalc_binary(measure, a, b, c, d):
    """Study-level estimates for Cochran's Q (0.5 added to all cells of tables with a zero cell, as in metafor)."""
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a2, b2, c2, d2 = (np.where(zero, x + 0.5, x) for x in (a, b, c, d))
    n1, n2 = a2 + b2, c2 + d2
    if measure == "OR":
        return np.log(a2 * d2 / (b2 * c2)), 1 / a2 + 1 / b2 + 1 / c2 + 1 / d2
    if measure == "RR":
        return np.log((a2 / n1) / (c2 / n2)), 1 / a2 - 1 / n1 + 1 / c2 - 1 / n2
    # RD uses the raw counts (no correction needed for the estimate; variance with uncorrected proportions)
    n1r, n2r = a + b, c + d
    p1, p2 = a / n1r, c / n2r
    return p1 - p2, p1 * (1 - p1) / n1r + p2 * (1 - p2) / n2r


def mantel_haenszel(ai, bi, ci, di, measure="OR", level=0.95) -> dict[str, Any]:
    """Mantel-Haenszel common-effect estimate for 2x2 tables (ai, bi = events / non-events in group 1; ci, di in group 2).

    Variances: Robins-Breslow-Greenland (log OR), Greenland-Robins (log RR), Sato-Greenland-Robins (RD), as in metafor.
    Estimates are on the log scale for OR and RR.
    """
    a, b, c, d, n1, n2, n = _tables(ai, bi, ci, di)
    measure = measure.upper()
    keep = n > 0
    if measure in ("OR", "RR"):
        keep &= ~(((a == 0) & (c == 0)) | ((b == 0) & (d == 0)))    # tables with no information
    a, b, c, d, n1, n2, n = (x[keep] for x in (a, b, c, d, n1, n2, n))
    k = int(len(a))
    if k < 1:
        return {"status": "Not estimable: no informative tables", "k": 0}
    if measure == "OR":
        R, S = np.sum(a * d / n), np.sum(b * c / n)
        est = math.log(R / S)
        P, Q = (a + d) / n, (b + c) / n
        var = (np.sum(P * a * d / n) / (2 * R ** 2) + np.sum(P * b * c / n + Q * a * d / n) / (2 * R * S)
               + np.sum(Q * b * c / n) / (2 * S ** 2))
    elif measure == "RR":
        R, S = np.sum(a * n2 / n), np.sum(c * n1 / n)
        est = math.log(R / S)
        var = np.sum((n1 * n2 * (a + c) - a * c * n) / n ** 2) / (R * S)
    elif measure == "RD":
        wsum = np.sum(n1 * n2 / n)
        est = float(np.sum((a * n2 - c * n1) / n) / wsum)
        P = (n1 ** 2 * c - n2 ** 2 * a + n1 * n2 * (n2 - n1) / 2) / n ** 2
        Q = (a * (n2 - c) + c * (n1 - a)) / (2 * n)
        var = (est * np.sum(P) + np.sum(Q)) / wsum ** 2
    else:
        raise ValueError("measure must be OR, RR or RD")
    se = math.sqrt(float(var))
    zc = float(norm.ppf(0.5 + level / 2))
    yi, vi = _escalc_binary(measure, a, b, c, d)
    ok = np.isfinite(yi) & np.isfinite(vi) & (vi > 0)
    qe = float(np.sum((yi[ok] - est) ** 2 / vi[ok]))
    dfq = int(ok.sum()) - 1
    out = {"status": "Pooled", "method": f"Mantel-Haenszel ({measure})", "measure": measure, "k": k, "estimate": est,
           "se": se, "ci_lo": est - zc * se, "ci_hi": est + zc * se, "z": est / se, "p": float(2 * norm.sf(abs(est / se))),
           "QE": qe, "QE_df": dfq, "QE_p": float(chi2.sf(qe, dfq)) if dfq > 0 else float("nan"),
           "I2": max(0.0, (qe - dfq) / qe) * 100 if qe > 0 else 0.0}
    if measure in ("OR", "RR"):
        out.update({"exp_estimate": math.exp(est), "exp_ci_lo": math.exp(out["ci_lo"]), "exp_ci_hi": math.exp(out["ci_hi"])})
    if measure == "OR":    # Cochran-Mantel-Haenszel test (without continuity correction)
        e_a = n1 * (a + c) / n
        var_a = n1 * n2 * (a + c) * (b + d) / (n ** 2 * (n - 1))
        cmh = float((np.sum(a) - np.sum(e_a)) ** 2 / np.sum(var_a))
        out.update({"CMH": cmh, "CMH_p": float(chi2.sf(cmh, 1))})
    return out


def peto(ai, bi, ci, di, level=0.95) -> dict[str, Any]:
    """Peto's one-step odds ratio (Yusuf et al. 1985); estimate on the log scale."""
    a, b, c, d, n1, n2, n = _tables(ai, bi, ci, di)
    keep = (n > 1) & ~(((a == 0) & (c == 0)) | ((b == 0) & (d == 0)))
    a, b, c, d, n1, n2, n = (x[keep] for x in (a, b, c, d, n1, n2, n))
    k = int(len(a))
    if k < 1:
        return {"status": "Not estimable: no informative tables", "k": 0}
    O_E = a - n1 * (a + c) / n
    V = n1 * n2 * (a + c) * (b + d) / (n ** 2 * (n - 1))
    est = float(np.sum(O_E) / np.sum(V))
    se = math.sqrt(1 / float(np.sum(V)))
    zc = float(norm.ppf(0.5 + level / 2))
    qe = float(np.sum(O_E ** 2 / V) - np.sum(O_E) ** 2 / np.sum(V))
    dfq = k - 1
    lo, hi = est - zc * se, est + zc * se
    return {"status": "Pooled", "method": "Peto (OR)", "measure": "OR", "k": k, "estimate": est, "se": se, "ci_lo": lo,
            "ci_hi": hi, "z": est / se, "p": float(2 * norm.sf(abs(est / se))), "QE": qe, "QE_df": dfq,
            "QE_p": float(chi2.sf(qe, dfq)) if dfq > 0 else float("nan"),
            "I2": max(0.0, (qe - dfq) / qe) * 100 if qe > 0 else 0.0,
            "exp_estimate": math.exp(est), "exp_ci_lo": math.exp(lo), "exp_ci_hi": math.exp(hi)}


# ======================================================================= step-function selection models
def _hess(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    E = np.eye(n) * h
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j]) - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return H


def selection_stepfun(y, v, steps: Sequence[float] = (0.025,), alternative: str = "greater", X=None,
                      tau2_fixed: float | None = None) -> dict[str, Any]:
    """Vevea-Hedges step-function selection model with any number of cut-points on the one-sided p value.

    steps: increasing cut-points in (0, 1); the last interval ends at 1. Interval 1 has weight 1; the other intervals have
    weights delta_j estimated by maximum likelihood, as in metafor::selmodel(type="stepfun").
    alternative: "greater" (p values for positive effects) or "less".
    Returns adjusted coefficients with Wald CIs, tau^2, the selection weights and the likelihood-ratio test of no selection.
    """
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    X = _X(k, X)
    p = X.shape[1]
    sgn = 1.0 if alternative == "greater" else -1.0
    cuts = np.array(sorted(set(float(s) for s in steps if 0 < float(s) < 1)) + [1.0])
    J = len(cuts)
    sei = np.sqrt(v)
    pvals = norm.sf(sgn * y / sei)
    interval = np.searchsorted(cuts, pvals, side="left")         # interval index of each study (0 = most significant)
    counts = np.bincount(interval, minlength=J)
    if k < p + 1 + J or (counts == 0).any():
        return {"status": "Not estimable: " + ("too few studies" if k < p + 1 + J else
                "at least one p-value interval contains no studies"), "counts": counts.tolist(), "steps": cuts.tolist()}
    zcrit = norm.isf(cuts)                                       # one-sided p < cut  <=> sgn*y/se > zcrit

    def unpack(par):
        beta = par[:p]
        tau2 = math.exp(par[p]) if tau2_fixed is None else tau2_fixed
        ld = par[p + (1 if tau2_fixed is None else 0):]
        delta = np.concatenate([[1.0], np.exp(ld)])
        return beta, tau2, delta

    def nll(par, fixed_delta=False):
        beta, tau2, delta = unpack(par)
        if fixed_delta:
            delta = np.ones(J)
        mu = X @ beta
        s = np.sqrt(v + tau2)
        # probability of each interval for each study
        upper = np.concatenate([[np.inf] * 1, zcrit[:-1]])       # p in (cut_{j-1}, cut_j]  <=> z in (zcrit_j, zcrit_{j-1}]
        lower = zcrit
        # z = sgn*y/se ; y = sgn*z*se. P(z in (l, u]) under y ~ N(mu, s^2)
        cdf = lambda zz: norm.cdf((zz * sei - sgn * mu) / s)
        probs = np.column_stack([cdf(np.minimum(upper[j], 1e300)) - cdf(lower[j]) if np.isfinite(upper[j])
                                 else 1 - cdf(lower[j]) for j in range(J)])
        probs = np.clip(probs, 1e-300, None)
        A = probs @ delta
        ll = norm.logpdf(y, mu, s) + np.log(delta[interval]) - np.log(A)
        return -float(np.sum(ll))

    from .stats import pool
    beta0 = np.linalg.lstsq(X * (1 / np.sqrt(v))[:, None], y / np.sqrt(v), rcond=None)[0]
    t0 = math.log(max(estimate_tau2(y, v, "ML", X), 1e-4))
    base = list(beta0) + ([t0] if tau2_fixed is None else [])
    opts = {"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12}
    fit0 = min((minimize(nll, np.array(base[:p] + ([t0 + d] if tau2_fixed is None else [])), args=(True,), method="Nelder-Mead",
                         options=opts) for d in (0.0, -2.0, 1.0)), key=lambda r: r.fun)
    starts = []
    for d in (0.0, -2.0):
        for ld in (0.0, -1.0, 0.5):
            starts.append(np.array(list(fit0.x[:p]) + ([fit0.x[p] + d] if tau2_fixed is None else []) + [ld] * (J - 1)))
    fit1 = min((minimize(nll, s0, method="Nelder-Mead", options=opts) for s0 in starts), key=lambda r: r.fun)
    for _ in range(3):                                            # polish with BFGS then Nelder-Mead
        r = minimize(nll, fit1.x, method="BFGS", options={"gtol": 1e-9})
        if r.fun <= fit1.fun:
            fit1 = r
        r = minimize(nll, fit1.x, method="Nelder-Mead", options=opts)
        if r.fun <= fit1.fun:
            fit1 = r
    beta, tau2, delta = unpack(fit1.x)
    try:
        cov = np.linalg.inv(_hess(nll, fit1.x))
        se_beta = np.sqrt(np.clip(np.diag(cov)[:p], 0, None))
        se_ld = np.sqrt(np.clip(np.diag(cov)[p + (1 if tau2_fixed is None else 0):], 0, None))
        se_delta = delta[1:] * se_ld                              # delta method
    except np.linalg.LinAlgError:
        se_beta = np.full(p, np.nan)
        se_delta = np.full(J - 1, np.nan)
    lrt = max(0.0, 2 * (fit0.fun - fit1.fun))
    out = {"status": "Performed" + (" (fewer than 20 studies: estimates are unstable)" if k < 20 else ""),
           "k": k, "steps": cuts.tolist(), "counts": counts.tolist(), "beta": beta.tolist(), "se_beta": se_beta.tolist(),
           "ci_lo": (beta - Z975 * se_beta).tolist(), "ci_hi": (beta + Z975 * se_beta).tolist(), "tau2": tau2,
           "delta": delta.tolist(), "se_delta": [float("nan")] + list(se_delta), "LRT": lrt, "LRT_df": J - 1,
           "p_selection": float(chi2.sf(lrt, J - 1)), "loglik": -fit1.fun, "loglik_no_selection": -fit0.fun,
           "alternative": alternative}
    out.update({"adjusted_estimate": float(beta[0]), "se": float(se_beta[0]), "ci_lo0": float(beta[0] - Z975 * se_beta[0]),
                "ci_hi0": float(beta[0] + Z975 * se_beta[0])})
    return out


# ======================================================================= influence diagnostics
def influence_full(y, v, labels=None, method="REML") -> pd.DataFrame:
    """metafor-style case diagnostics for the random-effects model (Viechtbauer & Cheung 2010)."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    labels = list(labels) if labels is not None else [str(i + 1) for i in range(k)]
    X = np.ones((k, 1))
    t2 = estimate_tau2(y, v, method)
    w = 1 / (v + t2)
    vb = 1 / np.sum(w)
    b = float(np.sum(w * y) * vb)
    hat = w * vb
    qe_full = float(np.sum((y - np.sum(y / v) / np.sum(1 / v)) ** 2 / v))
    rows = []
    for i in range(k):
        m = np.arange(k) != i
        t2i = estimate_tau2(y[m], v[m], method)
        wi = 1 / (v[m] + t2i)
        vbi = 1 / np.sum(wi)
        bi = float(np.sum(wi * y[m]) * vbi)
        rstud = (y[i] - bi) / math.sqrt(v[i] + t2i + vbi)
        dffits = (b - bi) / math.sqrt(hat[i] * (t2i + v[i]))
        cook = (b - bi) ** 2 / vb
        covr = vbi / vb
        w0 = 1 / v[m]
        qe_del = float(np.sum((y[m] - np.sum(w0 * y[m]) / np.sum(w0)) ** 2 * w0))
        dfbs = (b - bi) / math.sqrt(vbi)
        rows.append({"study": labels[i], "rstudent": rstud, "dffits": dffits, "cook_d": cook, "cov_r": covr,
                     "tau2_del": t2i, "QE_del": qe_del, "hat": hat[i], "weight_pct": 100 * hat[i], "dfbetas": dfbs})
    df = pd.DataFrame(rows)
    p = 1
    flag = ((df["rstudent"].abs() > norm.ppf(0.975)) | (df["dffits"].abs() > 3 * math.sqrt(p / (k - p)))
            | (df["cook_d"] > chi2.ppf(0.5, p)) | (df["hat"] > 3 * p / k) | (df["dfbetas"].abs() > 1))
    # metafor's rule: influential if any of DFFITS, Cook's D, hat or DFBETAS exceed their cut-offs
    infl = ((df["dffits"].abs() > 3 * math.sqrt(p / (k - p))) | (df["cook_d"] > chi2.ppf(0.5, p))
            | (df["hat"] > 3 * p / k) | (df["dfbetas"].abs() > 1))
    df["influential"] = np.where(infl, "Yes", "")
    df["outlier"] = np.where(df["rstudent"].abs() > norm.ppf(0.975), "Yes", "")
    del flag
    return df


# ======================================================================= test of excess significance
def tes(y, v, theta: float | None = None, alpha=0.05, test: str = "chi2") -> dict[str, Any]:
    """Test of excess significance (Ioannidis & Trikalinos 2007).

    O = number of studies significant (two-sided p < alpha); E = sum of their power to detect theta (default: the
    common-effect estimate). One-sided test of O > E by the chi-square statistic of the original article (or the
    exact binomial test with the mean power).
    """
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    sei = np.sqrt(v)
    if theta is None:
        w = 1 / v
        theta = float(np.sum(w * y) / np.sum(w))
    zc = norm.isf(alpha / 2)
    O = int(np.sum(np.abs(y / sei) > zc))
    power = norm.sf(zc - theta / sei) + norm.cdf(-zc - theta / sei)
    E = float(np.sum(power))
    out = {"k": k, "theta": theta, "O": O, "E": E, "OE_ratio": O / E if E > 0 else float("nan"), "power": power.tolist()}
    if test == "binom":
        pm = E / k
        out.update({"test": "binomial", "p": float(binom.sf(O - 1, k, pm))})
    else:
        A = (O - E) ** 2 / E + (O - E) ** 2 / (k - E) if 0 < E < k else float("nan")
        p_two = float(chi2.sf(A, 1))
        out.update({"test": "chi-square", "statistic": A, "p": p_two / 2 if O > E else 1 - p_two / 2})
    return out
