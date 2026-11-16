"""Additional meta-analytic methods.

Heterogeneity      : Q-profile confidence intervals for tau^2, tau, I^2, H^2 (Viechtbauer 2007)
Small-study / bias : PET, PEESE and conditional PET-PEESE (Stanley & Doucouliagos 2014)
                     three-parameter selection model, 3PSM (Vevea & Hedges 1995; Iyengar & Greenhouse 1988)
                     Peters' regression test for binary outcomes (Peters et al. 2006)
                     fail-safe N (Rosenthal 1979; Orwin 1983) - reported for completeness only
Influence          : Baujat plot coordinates (Baujat et al. 2002)
Moderators         : multiple meta-regression with a permutation test (Higgins & Thompson 2004)
Bayesian           : normal-normal hierarchical model with half-normal prior on tau, by exact grid
                     integration (in the spirit of bayesmeta; Roever 2020)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import chi2, halfnorm, norm, t as tdist

from . import stats


# ------------------------------------------------------------------ heterogeneity CIs
def _genq(tau2, y, v, X):
    w = 1 / (v + tau2)
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    r = y - X @ beta
    return float(np.sum(w * r ** 2))


def tau2_qprofile(y, v, X=None, level=0.95) -> dict[str, float]:
    """Q-profile CI for tau^2 (Viechtbauer 2007), and derived CIs for tau, I^2 and H^2."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    X = np.ones((k, 1)) if X is None else X
    df = k - X.shape[1]
    if df < 1:
        return {}
    a = (1 - level) / 2
    q_hi, q_lo = chi2.ppf(1 - a, df), chi2.ppf(a, df)

    def bound(target):
        f = lambda t2: _genq(t2, y, v, X) - target
        if f(0.0) < 0:
            return 0.0
        hi = max(1.0, float(np.var(y)) * 10)
        while f(hi) > 0:
            hi *= 2
            if hi > 1e8:
                return float("inf")
        return float(brentq(f, 0.0, hi, xtol=1e-12))
    lo, up = bound(q_hi), bound(q_lo)
    w = 1 / v
    s2 = (k - 1) * w.sum() / (w.sum() ** 2 - np.sum(w ** 2))
    i2 = lambda t2: 100 * t2 / (t2 + s2)
    return {"tau2_lo": lo, "tau2_hi": up, "tau_lo": math.sqrt(lo), "tau_hi": math.sqrt(up),
            "I2_lo": i2(lo), "I2_hi": i2(up), "H2_lo": (lo + s2) / s2, "H2_hi": (up + s2) / s2}


# ------------------------------------------------------------------ PET-PEESE
def _wls(y, X, w):
    W = np.diag(w)
    XtWX = X.T @ W @ X
    beta = np.linalg.solve(XtWX, X.T @ W @ y)
    r = y - X @ beta
    k, p = X.shape
    s2 = float(np.sum(w * r ** 2) / (k - p))          # multiplicative dispersion, as in lm(weights = 1/v)
    cov = s2 * np.linalg.inv(XtWX)
    return beta, np.sqrt(np.diag(cov)), k - p


def pet_peese(y, v, alpha=0.05) -> dict[str, Any]:
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if k < 4:
        return {"status": "Not performed: fewer than 4 studies"}
    se = np.sqrt(v)
    out = {}
    for name, x in (("PET", se), ("PEESE", v)):
        b, s, df = _wls(y, np.column_stack([np.ones(k), x]), 1 / v)
        tval = b[0] / s[0]
        crit = tdist.ppf(0.975, df)
        out[name] = {"estimate": float(b[0]), "se": float(s[0]), "ci_lo": float(b[0] - crit * s[0]), "ci_hi": float(b[0] + crit * s[0]),
                     "p": float(2 * tdist.sf(abs(tval), df)), "slope": float(b[1]), "slope_p": float(2 * tdist.sf(abs(b[1] / s[1]), df))}
    # conditional rule: if PET rejects H0: effect = 0 (one-sided, in the direction of the effect), report PEESE
    p_one = out["PET"]["p"] / 2 if np.sign(out["PET"]["estimate"]) == np.sign(np.average(y, weights=1 / v)) else 1 - out["PET"]["p"] / 2
    chosen = "PEESE" if p_one < alpha else "PET"
    return {"status": "Performed" + (" (fewer than 10 studies: low power, interpret with caution)" if k < 10 else ""),
            "PET": out["PET"], "PEESE": out["PEESE"], "conditional": chosen, "conditional_estimate": out[chosen]["estimate"],
            "conditional_ci": (out[chosen]["ci_lo"], out[chosen]["ci_hi"])}


# ------------------------------------------------------------------ selection model (3PSM)
def selection_3psm(y, v, cut=0.025, direction: int | None = None) -> dict[str, Any]:
    """Step-function selection model with one cut-point on the one-sided p value (default .025, i.e. two-sided .05).

    Effects with one-sided p < cut are published with weight 1, others with relative weight delta.
    Parameters (mu, tau^2, delta) by maximum likelihood; LRT of delta = 1 (no selection).
    """
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if k < 6:
        return {"status": "Not performed: fewer than 6 studies"}
    if direction is None:
        direction = 1 if np.average(y, weights=1 / v) >= 0 else -1
    yy = direction * y
    se = np.sqrt(v)
    zcut = norm.isf(cut)
    sig = (yy / se) > zcut
    if sig.all() or (~sig).all():
        return {"status": "Not estimable: all studies fall on one side of the selection cut-point"}

    def nll(par, fixed_delta=None):
        mu, ltau2 = par[0], par[1]
        tau2 = math.exp(ltau2)
        delta = 1.0 if fixed_delta is not None else math.exp(par[2])
        s = np.sqrt(v + tau2)
        p_sig = norm.sf((zcut * se - mu) / s)                     # P(one-sided p < cut)
        A = p_sig + delta * (1 - p_sig)
        ll = norm.logpdf(yy, mu, s) - np.log(A) + np.where(sig, 0.0, math.log(delta))
        return -float(np.sum(ll))

    re = stats.pool(yy, v, "random", "REML", "Wald")
    t0 = math.log(max(re["tau2"], 1e-3))
    best0 = min((minimize(nll, [re["estimate"], t0 + d], args=(1.0,), method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 4000}) for d in (0, -2, 1)), key=lambda r: r.fun)
    best1 = min((minimize(nll, [re["estimate"], t0 + d, ld], method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 8000}) for d in (0, -2) for ld in (0.0, -1.0, 1.0)),
                key=lambda r: r.fun)
    mu, tau2, delta = best1.x[0], math.exp(best1.x[1]), math.exp(best1.x[2])
    # standard errors from the numerical Hessian on (mu, log tau2, log delta)
    def hess(f, x, h=1e-4):
        n = len(x)
        Hm = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
                Hm[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h * h)
        return Hm
    try:
        cov = np.linalg.inv(hess(nll, best1.x))
        se_mu = math.sqrt(max(cov[0, 0], 0))
    except np.linalg.LinAlgError:
        se_mu = float("nan")
    lrt = max(0.0, 2 * (best0.fun - best1.fun))
    est = direction * mu
    return {"status": "Performed" + (" (fewer than 20 studies: estimates are unstable)" if k < 20 else ""),
            "adjusted_estimate": est, "se": se_mu, "ci_lo": est - 1.959964 * se_mu, "ci_hi": est + 1.959964 * se_mu,
            "tau2": tau2, "delta": delta, "unadjusted_ml_estimate": direction * best0.x[0],
            "LRT": lrt, "p_selection": float(chi2.sf(lrt, 1)), "n_significant": int(sig.sum()), "k": k, "cut": cut}


# ------------------------------------------------------------------ Peters, fail-safe N
def peters(y, v, n_total) -> dict[str, Any]:
    """Peters et al. (2006): weighted regression of the log OR on 1/N (weights 1/v); intercept-free test of the slope."""
    y, v, n = np.asarray(y, float), np.asarray(v, float), np.asarray(n_total, float)
    k = len(y)
    if k < 3:
        return {"status": "Not performed: fewer than 3 studies"}
    b, s, df = _wls(y, np.column_stack([np.ones(k), 1 / n]), 1 / v)
    tval = b[1] / s[1]
    return {"status": "Performed" + (" (fewer than 10 studies: low power)" if k < 10 else ""), "slope": float(b[1]), "se": float(s[1]),
            "t": float(tval), "df": df, "p": float(2 * tdist.sf(abs(tval), df))}


def failsafe_n(y, v, trivial: float = 0.1) -> dict[str, Any]:
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    z = y / np.sqrt(v)
    zsum = z.sum()
    rosenthal = max(0.0, (zsum / norm.isf(0.05)) ** 2 - k)
    mean_es = float(np.mean(y))
    orwin = max(0.0, k * (abs(mean_es) - trivial) / trivial) if trivial > 0 else float("nan")
    return {"status": "Reported for completeness: fail-safe N ignores heterogeneity and is not recommended for inference "
                      "(Cochrane Handbook 13.3.5.6)",
            "rosenthal_N": rosenthal, "orwin_N": orwin, "orwin_trivial_effect": trivial}


# ------------------------------------------------------------------ Baujat
def baujat(y, v, labels) -> pd.DataFrame:
    y, v = np.asarray(y, float), np.asarray(v, float)
    w = 1 / v
    mu = np.sum(w * y) / w.sum()
    rows = []
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        w_i = 1 / v[m]
        mu_i = np.sum(w_i * y[m]) / w_i.sum()
        var_i = 1 / w_i.sum()
        rows.append({"study": labels[i], "contribution_to_Q": float(w[i] * (y[i] - mu) ** 2),
                     "influence_on_estimate": float((mu - mu_i) ** 2 / var_i)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ multiple meta-regression with permutation test
def _mr_stats(y, v, X, tau_method, knha):
    """QM and coefficient test statistics of a mixed-effects meta-regression (numpy only; used for permutations)."""
    k, p = X.shape
    tau2 = stats.estimate_tau2(y, v, tau_method, X)
    w = 1 / (v + tau2)
    XtWX = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(XtWX)
    beta = cov @ (X.T @ (w * y))
    if knha:
        r = y - X @ beta
        cov = cov * max(1.0, float(np.sum(w * r ** 2)) / (k - p))
    se = np.sqrt(np.diag(cov))
    b1, c1 = beta[1:], cov[1:, 1:]
    qm = float(b1 @ np.linalg.solve(c1, b1)) if p > 1 else float("nan")
    return qm, beta / se


def metareg_permutation(df: pd.DataFrame, cols: list[str], n_perm: int = 1000, seed: int = 2024, **kw) -> dict[str, Any]:
    """Permutation test (Higgins & Thompson 2004): moderator rows are permuted jointly and the model refitted."""
    base = stats.meta_regression(df, cols, **kw)
    if base.get("status") != "Estimated":
        return base
    X0 = np.asarray(base["X"], float)
    d = df.dropna(subset=cols).reset_index(drop=True)
    y, v = d["yi"].to_numpy(float), d["vi"].to_numpy(float)
    tau_method, knha = kw.get("tau_method", "REML"), kw.get("knha", True)
    rng = np.random.default_rng(seed)
    qm0 = base["QM"]
    stat0 = np.abs(base["coefficients"]["statistic"].to_numpy())
    ge_qm = 0
    ge = np.zeros_like(stat0)
    done = 0
    for _ in range(n_perm):
        order = rng.permutation(len(d))
        Xp = X0.copy()
        Xp[:, 1:] = X0[order, 1:]
        if np.linalg.matrix_rank(Xp) < Xp.shape[1]:
            continue
        try:
            qm, st = _mr_stats(y, v, Xp, tau_method, knha)
        except np.linalg.LinAlgError:
            continue
        done += 1
        ge_qm += qm >= qm0 - 1e-12
        ge += np.abs(st) >= stat0 - 1e-12
    base["permutation"] = {"n": done, "QM_p": (ge_qm + 1) / (done + 1),
                           "coef_p": ((ge + 1) / (done + 1)).tolist()}
    base["coefficients"] = base["coefficients"].assign(p_permutation=base["permutation"]["coef_p"])
    return base


# ------------------------------------------------------------------ Bayesian random-effects (grid integration)
def bayes_re(y, v, tau_prior_scale: float = 0.5, mu_prior_sd: float = 10.0, mu_prior_mean: float = 0.0,
             grid: int = 4000) -> dict[str, Any]:
    """Normal-normal model: y_i ~ N(theta_i, v_i), theta_i ~ N(mu, tau^2), mu ~ N(m0, s0^2), tau ~ half-normal(scale).

    p(tau | y) is computed exactly on a grid (mu integrated analytically); p(mu | y) is the resulting
    mixture of normals. Default priors: weakly informative half-normal(0.5) for tau on the SMD / log-ratio
    scale (Roever et al. 2021) and a vague N(0, 10^2) for mu."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    upper = max(5 * tau_prior_scale, 3 * math.sqrt(max(np.var(y), 1e-6))) + 1.0
    taus = np.linspace(0, upper, grid)
    logpost = np.empty(grid)
    means = np.empty(grid)
    sds = np.empty(grid)
    for i, tau in enumerate(taus):
        s2 = v + tau ** 2
        w = 1 / s2
        prec = w.sum() + 1 / mu_prior_sd ** 2
        m = (np.sum(w * y) + mu_prior_mean / mu_prior_sd ** 2) / prec
        # log marginal likelihood p(y | tau) with mu integrated out (normal-normal conjugacy)
        ll = (-0.5 * np.sum(np.log(2 * math.pi * s2)) - 0.5 * np.sum(w * y ** 2) - 0.5 * mu_prior_mean ** 2 / mu_prior_sd ** 2
              + 0.5 * prec * m ** 2 - 0.5 * math.log(prec) - math.log(mu_prior_sd))
        logpost[i] = ll + halfnorm.logpdf(tau, scale=tau_prior_scale)
        means[i], sds[i] = m, math.sqrt(1 / prec)
    wts = np.exp(logpost - logpost.max())
    wts /= np.trapezoid(wts, taus)
    cdf_tau = np.concatenate([[0], np.cumsum((wts[1:] + wts[:-1]) / 2 * np.diff(taus))])
    cdf_tau /= cdf_tau[-1]
    q = lambda c, p_: float(np.interp(p_, c, taus))
    h = np.diff(taus)
    pt = wts * np.concatenate([[h[0] / 2], (h[:-1] + h[1:]) / 2, [h[-1] / 2]])    # trapezoid weights
    pt /= pt.sum()
    # p(mu | y) and the predictive distribution are mixtures of normals over the tau grid: invert their CDFs exactly
    def mix_q(p_, sd_):
        f = lambda x: float(np.sum(pt * norm.cdf(x, means, sd_))) - p_
        lo, hi = float(np.min(means - 10 * sd_)), float(np.max(means + 10 * sd_))
        return float(brentq(f, lo, hi, xtol=1e-12))
    qm = lambda p_: mix_q(p_, sds)
    # shortest 95% interval (as reported by bayesmeta), in addition to the equal-tailed one
    from scipy.optimize import minimize_scalar as _msc
    _w = _msc(lambda a_: qm(a_ + 0.95) - qm(a_), bounds=(1e-6, 0.05 - 1e-6), method="bounded", options={"xatol": 1e-8})
    short_lo, short_hi = qm(float(_w.x)), qm(float(_w.x) + 0.95)
    post_mean = float(np.sum(pt * means))
    psd = np.sqrt(sds ** 2 + taus ** 2)
    return {"mu_mean": post_mean, "mu_median": qm(0.5), "mu_lo": qm(0.025), "mu_hi": qm(0.975),
            "mu_lo_shortest": short_lo, "mu_hi_shortest": short_hi,
            "P(mu > 0)": float(np.sum(pt * norm.sf(0, means, sds))),
            "tau_median": q(cdf_tau, 0.5), "tau_lo": q(cdf_tau, 0.025), "tau_hi": q(cdf_tau, 0.975),
            "pred_lo": mix_q(0.025, psd), "pred_hi": mix_q(0.975, psd),
            "priors": f"mu ~ N({mu_prior_mean}, {mu_prior_sd}^2); tau ~ half-normal(scale {tau_prior_scale})"}
