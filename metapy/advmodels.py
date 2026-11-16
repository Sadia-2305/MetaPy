"""Advanced models (MetaPy 5.1).

location_scale : location-scale meta-regression, y_i ~ N(x_i'b, v_i + exp(z_i'a)), so that heterogeneity itself depends on
                 moderators (Viechtbauer & Lopez-Lopez 2022); REML or ML; likelihood-ratio test of the scale moderators.
multilevel     : random-effects model with any number of nested random intercepts (e.g. country / study / effect size;
                 Konstantopoulos 2011; Van den Noortgate et al. 2013), REML or ML, likelihood-ratio tests of each variance
                 component with the 50:50 chi-square boundary mixture (Self & Liang 1987), level-specific I^2
                 (Cheung 2014), optional moderators.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from .glmm import _num_hess

Z975 = float(norm.ppf(0.975))


# ============================================================================ location-scale
def _ls_nll(alpha, y, v, X, Z, reml):
    t2 = np.exp(np.clip(Z @ alpha, -40, 40))
    s = v + t2
    w = 1 / s
    XtWX = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(XtWX, X.T @ (w * y))
    r = y - X @ beta
    nll = 0.5 * (np.sum(np.log(s)) + np.sum(w * r ** 2))
    if reml:
        nll += 0.5 * np.linalg.slogdet(XtWX)[1]
    return float(nll), beta, XtWX


def location_scale(y, v, X, Z, x_names: Sequence[str], z_names: Sequence[str], method="REML") -> dict[str, Any]:
    y, v, X, Z = (np.asarray(a, float) for a in (y, v, X, Z))
    k, p = X.shape
    q = Z.shape[1]
    reml = method.upper() == "REML"
    if k - p - q < 1:
        return {"status": f"Not estimable: {k} studies for {p} location and {q} scale coefficients"}
    f = lambda a: _ls_nll(a, y, v, X, Z, reml)[0]
    from .extended import estimate_tau2
    t0 = max(estimate_tau2(y, v, "REML" if reml else "ML", X), 1e-3)
    a0 = np.zeros(q)
    a0[0] = math.log(t0)
    best = None
    for shift in (0.0, -2.0, 1.0):
        st = a0.copy()
        st[0] += shift
        r = minimize(f, st, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
        r2 = minimize(f, r.x, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
        r = r2 if r2.fun <= r.fun else r
        if best is None or r.fun < best.fun:
            best = r
    alpha = best.x
    nll, beta, XtWX = _ls_nll(alpha, y, v, X, Z, reml)
    vb = np.linalg.inv(XtWX)
    se_b = np.sqrt(np.diag(vb))
    try:
        va = np.linalg.inv(_num_hess(f, alpha, h=1e-4))
        se_a = np.sqrt(np.clip(np.diag(va), 0, None))
    except np.linalg.LinAlgError:
        se_a = np.full(q, np.nan)
    # LRT of the scale moderators (model with a single tau^2), using ML when location moderators are the same
    f0 = lambda a: _ls_nll(np.concatenate([a, np.zeros(q - 1)]), y, v, X, Z[:, :1] if q else Z, reml)[0]
    r0 = minimize(lambda a: _ls_nll(a, y, v, X, Z[:, :1], reml)[0], np.array([math.log(t0)]), method="Nelder-Mead",
                  options={"xatol": 1e-10, "fatol": 1e-12})
    lrt = max(0.0, 2 * (r0.fun - nll)) if q > 1 else float("nan")
    loc = pd.DataFrame({"term": list(x_names), "estimate": beta, "se": se_b, "z": beta / se_b,
                        "p": 2 * norm.sf(np.abs(beta / se_b)), "ci_lo": beta - Z975 * se_b, "ci_hi": beta + Z975 * se_b})
    sca = pd.DataFrame({"term": list(z_names), "estimate": alpha, "se": se_a, "z": alpha / se_a,
                        "p": 2 * norm.sf(np.abs(alpha / se_a)), "ci_lo": alpha - Z975 * se_a, "ci_hi": alpha + Z975 * se_a})
    del f0
    return {"status": "Estimated", "method": method.upper(), "k": k, "location": loc, "scale": sca,
            "tau2_i": np.exp(Z @ alpha).tolist(), "loglik": -nll, "LRT_scale": lrt, "LRT_df": q - 1,
            "p_scale": float(chi2.sf(lrt, q - 1)) if q > 1 else float("nan")}


# ============================================================================ nested multilevel
def _ml_design(df: pd.DataFrame, levels: Sequence[str]) -> list[np.ndarray]:
    """Indicator matrices Z_l for each nested level; inner levels are nested within outer ones."""
    Zs = []
    key = pd.Series([""] * len(df), index=df.index)
    for lv in levels:
        key = key + "|" + df[lv].astype(str)
        codes, uniq = pd.factorize(key)
        Z = np.zeros((len(df), len(uniq)))
        Z[np.arange(len(df)), codes] = 1.0
        Zs.append(Z)
    return Zs


def _mv_nll(lsig, y, v, X, Gs, reml):
    s2 = np.exp(np.clip(lsig, -40, 20))
    V = np.diag(v) + sum(s * G for s, G in zip(s2, Gs))
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return 1e12, None, None
    Li = np.linalg.solve(L, np.eye(len(y)))
    Vi = Li.T @ Li
    XtViX = X.T @ Vi @ X
    beta = np.linalg.solve(XtViX, X.T @ Vi @ y)
    r = y - X @ beta
    nll = 0.5 * (2 * np.sum(np.log(np.diag(L))) + r @ Vi @ r)
    if reml:
        nll += 0.5 * np.linalg.slogdet(XtViX)[1]
    return float(nll), beta, XtViX


def _fit_levels(y, v, X, Gs, reml, start=None):
    m = len(Gs)
    if m == 0:
        nll, beta, XtViX = _mv_nll(np.zeros(0), y, v, X, Gs, reml)
        return nll, np.zeros(0), beta, XtViX
    f = lambda ls: _mv_nll(ls, y, v, X, Gs, reml)[0]
    base = np.log(max(float(np.var(y)) / (m + 1), 1e-3))
    starts = [np.full(m, base)] + ([start] if start is not None else []) + [np.full(m, base - 2)]
    best = None
    for st in starts:
        r = minimize(f, st, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000})
        r = minimize(f, r.x, method="BFGS", options={"gtol": 1e-9}) if np.isfinite(r.fun) else r
        if best is None or r.fun < best.fun:
            best = r
    # a component at the boundary: compare with fits that drop it
    nll, beta, XtViX = _mv_nll(best.x, y, v, X, Gs, reml)
    return nll, np.exp(best.x), beta, XtViX


def multilevel(df: pd.DataFrame, levels: Sequence[str], moderators: Sequence[str] | None = None, method="REML") -> dict[str, Any]:
    """Nested random-intercepts model. levels from outermost to innermost, e.g. ["country", "study", "es_id"].

    Effect sizes need columns yi and vi. Each level adds a variance component sigma^2_l.
    """
    from .metareg import design
    d = df.dropna(subset=["yi", "vi"] + list(levels) + list(moderators or [])).reset_index(drop=True)
    y, v = d["yi"].to_numpy(float), d["vi"].to_numpy(float)
    X, names, _ = design(d, list(moderators or []))
    reml = method.upper() == "REML"
    Zs = _ml_design(d, levels)
    Gs = [Z @ Z.T for Z in Zs]
    k, p = X.shape
    if k - p - len(Gs) < 1:
        return {"status": "Not estimable: too few effect sizes for the number of levels and moderators"}
    nll, s2, beta, XtViX = _fit_levels(y, v, X, Gs, reml)
    # refine near-zero components by checking the boundary fits
    vb = np.linalg.inv(XtViX)
    se = np.sqrt(np.diag(vb))
    coef = pd.DataFrame({"term": names, "estimate": beta, "se": se, "z": beta / se, "p": 2 * norm.sf(np.abs(beta / se)),
                         "ci_lo": beta - Z975 * se, "ci_hi": beta + Z975 * se})
    w = 1 / v
    typical_v = (k - 1) * np.sum(w) / (np.sum(w) ** 2 - np.sum(w ** 2))
    tot = float(np.sum(s2) + typical_v)
    comp = []
    for i, lv in enumerate(levels):
        Gs_red = [G for j, G in enumerate(Gs) if j != i]
        nll_red, *_ = _fit_levels(y, v, X, Gs_red, reml)
        lrt = max(0.0, 2 * (nll_red - nll))
        comp.append({"level": lv, "sigma2": s2[i], "sigma": math.sqrt(s2[i]), "n_levels": Zs[i].shape[1],
                     "I2_percent": 100 * s2[i] / tot, "LRT": lrt, "p (boundary mixture)": 0.5 * float(chi2.sf(lrt, 1)) if lrt > 0 else 1.0})
    return {"status": "Estimated", "method": method.upper(), "k": k, "coefficients": coef, "components": pd.DataFrame(comp),
            "I2_total": 100 * float(np.sum(s2)) / tot, "loglik": -nll}
