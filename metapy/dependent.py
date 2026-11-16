"""Models for dependent effect sizes (several effects per study).

* ``multilevel``  three-level random-effects model (effects nested in studies), REML,
                  with optional moderators, variance components, I² by level
                  (Cheung 2014) and likelihood-ratio tests for each level.
* ``che_rve``     correlated-hierarchical-effects (CHE) working model with an assumed
                  within-study sampling correlation rho, followed by cluster-robust
                  (CR2) standard errors with Satterthwaite degrees of freedom
                  (Pustejovsky & Tipton 2018, 2022). This is the current recommended
                  default when the sampling correlations are unknown.
Both reduce to the ordinary random-effects model when every study has one effect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm, t as tdist


# ------------------------------------------------------------------ helpers
def _design(df: pd.DataFrame, moderators: list[str] | None) -> tuple[np.ndarray, list[str]]:
    k = len(df)
    cols, names = [np.ones(k)], ["Intercept"]
    for c in moderators or []:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().all():
            cols.append(num.to_numpy(float))
            names.append(c)
        else:
            dummies = pd.get_dummies(df[c].astype(str), drop_first=True, dtype=float)
            for dc in dummies.columns:
                cols.append(dummies[dc].to_numpy())
                names.append(f"{c} = {dc}")
    return np.column_stack(cols), names


def _sampling_blocks(v: np.ndarray, idx: list[np.ndarray], rho: float) -> list[np.ndarray]:
    """Within-study sampling covariance: var v_i, covariance rho * sqrt(v_i v_j) (constant-correlation working model)."""
    out = []
    for ix in idx:
        s = np.sqrt(v[ix])
        S = rho * np.outer(s, s)
        np.fill_diagonal(S, v[ix])
        out.append(S)
    return out


@dataclass
class DepFit:
    model: str
    k: int
    n_clusters: int
    coef: pd.DataFrame
    sigma2_between: float           # between-study (level 3)
    sigma2_within: float            # within-study (level 2)
    loglik: float
    rho: float = 0.0
    se_type: str = "model-based"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def estimate(self) -> float:
        return float(self.coef["estimate"].iloc[0])


def _reml_parts(y, X, S_blocks, idx, s2b, s2w, method="REML"):
    """-2 x (restricted) log-likelihood and GLS pieces for given variance components."""
    p = X.shape[1]
    XtWX = np.zeros((p, p))
    XtWy = np.zeros(p)
    logdet = 0.0
    Winv = []
    for S, ix in zip(S_blocks, idx):
        n = len(ix)
        V = S + s2b * np.ones((n, n)) + s2w * np.eye(n)
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return np.inf, None
        logdet += 2 * np.sum(np.log(np.diag(L)))
        Vi = np.linalg.inv(V)
        Winv.append(Vi)
        Xj, yj = X[ix], y[ix]
        XtWX += Xj.T @ Vi @ Xj
        XtWy += Xj.T @ Vi @ yj
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return np.inf, None
    beta = cov @ XtWy
    rss = 0.0
    for Vi, ix in zip(Winv, idx):
        r = y[ix] - X[ix] @ beta
        rss += r @ Vi @ r
    val = logdet + rss
    if method == "REML":
        val += np.linalg.slogdet(XtWX)[1]
    return val, (beta, cov, Winv)


def fit_dependent(df: pd.DataFrame, cluster: str = "study_id", moderators: list[str] | None = None,
                  rho: float = 0.0, method: str = "REML", fix_within_zero: bool = False,
                  fix_between_zero: bool = False) -> DepFit:
    """Fit the three-level / CHE model by (RE)ML. rho = 0 gives the classic three-level model."""
    d = df.dropna(subset=["yi", "vi"] + list(moderators or [])).reset_index(drop=True)
    y, v = d["yi"].to_numpy(float), d["vi"].to_numpy(float)
    X, names = _design(d, moderators)
    groups = d[cluster].astype(str).to_numpy()
    uniq = list(dict.fromkeys(groups))
    idx = [np.where(groups == g)[0] for g in uniq]
    S_blocks = _sampling_blocks(v, idx, rho)

    def neg(s2b, s2w):
        if fix_between_zero:
            s2b = 0.0
        if fix_within_zero:
            s2w = 0.0
        return _reml_parts(y, X, S_blocks, idx, s2b, s2w, method)[0]

    start = max(1e-4, float(np.var(y)) / 2)
    cands = []
    # interior optimum on the log scale, several starting values
    for a0 in (start, start / 10, start * 3):
        for b0 in (start, start / 10, start * 3):
            r = minimize(lambda th: neg(math.exp(th[0]), math.exp(th[1])), x0=[math.log(a0), math.log(b0)],
                         method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
            cands.append((math.exp(r.x[0]), math.exp(r.x[1]), r.fun))
    # boundaries: one component (or both) exactly zero
    for which in ("b", "w"):
        r = minimize(lambda th: neg(math.exp(th[0]), 0.0) if which == "b" else neg(0.0, math.exp(th[0])),
                     x0=[math.log(start)], method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-12})
        val = math.exp(r.x[0])
        cands.append((val, 0.0, r.fun) if which == "b" else (0.0, val, r.fun))
    cands.append((0.0, 0.0, neg(0.0, 0.0)))
    s2b, s2w, _ = min(cands, key=lambda c: c[2])
    if fix_between_zero:
        s2b = 0.0
    if fix_within_zero:
        s2w = 0.0
    s2b = 0.0 if s2b < 1e-10 else s2b
    s2w = 0.0 if s2w < 1e-10 else s2w
    fval, (beta, cov, Winv) = _reml_parts(y, X, S_blocks, idx, s2b, s2w, method)
    se = np.sqrt(np.diag(cov))
    k, m, p = len(y), len(idx), X.shape[1]
    df_t = max(1, m - p)
    stat = beta / se
    pv = 2 * tdist.sf(np.abs(stat), df_t)
    crit = tdist.ppf(0.975, df_t)
    coef = pd.DataFrame({"term": names, "estimate": beta, "se": se, "ci_lo": beta - crit * se, "ci_hi": beta + crit * se,
                         "statistic": stat, "df": df_t, "p": pv})
    fit = DepFit("CHE" if rho else "three-level", k, m, coef, s2b, s2w, -0.5 * fval, rho,
                 "model-based (t, df = clusters - p)")
    fit.extra.update({"X": X, "names": names, "y": y, "v": v, "idx": idx, "S_blocks": S_blocks, "Winv": Winv,
                      "cov": cov, "beta": beta, "data": d, "method": method})
    # I² by level (Cheung 2014), typical within-study variance as in Higgins & Thompson
    w0 = 1 / v
    s2 = (k - 1) * w0.sum() / (w0.sum() ** 2 - np.sum(w0 ** 2)) if k > 1 else float("nan")
    tot = s2b + s2w + s2
    fit.extra.update({"typical_v": s2, "I2_between": 100 * s2b / tot, "I2_within": 100 * s2w / tot,
                      "I2_total": 100 * (s2b + s2w) / tot})
    return fit


def level_tests(df: pd.DataFrame, cluster: str = "study_id", moderators: list[str] | None = None, rho: float = 0.0) -> pd.DataFrame:
    """REML likelihood-ratio tests: is the between-study / within-study variance needed?
    p values use the 50:50 chi-square mixture for a boundary parameter (Self & Liang 1987)."""
    full = fit_dependent(df, cluster, moderators, rho)
    rows = []
    for name, kw in (("Between-study variance (level 3)", {"fix_between_zero": True}),
                     ("Within-study variance (level 2)", {"fix_within_zero": True})):
        red = fit_dependent(df, cluster, moderators, rho, **kw)
        lrt = max(0.0, 2 * (full.loglik - red.loglik))
        rows.append({"test": name, "LRT": lrt, "p (chi2 df=1)": float(chi2.sf(lrt, 1)),
                     "p (boundary mixture)": float(0.5 * chi2.sf(lrt, 1)) if lrt > 0 else 1.0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ cluster-robust (CR2) inference
def _sqrtm_psd(M: np.ndarray, inverse: bool = False) -> np.ndarray:
    w, U = np.linalg.eigh((M + M.T) / 2)
    w = np.clip(w, 1e-12, None)
    return (U * (w ** (-0.5 if inverse else 0.5))) @ U.T


def cr2_core(X: np.ndarray, y: np.ndarray, W_blocks: list[np.ndarray], idx: list[np.ndarray], beta: np.ndarray):
    """CR2 cluster-robust variance and Satterthwaite df, exactly as in Pustejovsky & Tipton (2018) and clubSandwich.

    Working weights W_j = Phi_j^{-1}; adjustment A_j = D_j' B_j^{-1/2} D_j with D_j the upper Cholesky factor of Phi_j
    and B_j = D_j [(I - H) Phi (I - H)']_jj D_j'. For contrast c, df = (tr G)^2 / sum(G^2) with G_ij = q_i' Phi q_j,
    q_j = (I - H)_j' A_j' W_j X_j M c (Satterthwaite approximation under the working model).
    Returns (V_R, list of q-builders) via a function computing df for any contrast.
    """
    N, p = X.shape
    W = np.zeros((N, N))
    Phi = np.zeros((N, N))
    Phis = []
    for Wj, ix in zip(W_blocks, idx):
        Pj = np.linalg.inv(Wj)
        W[np.ix_(ix, ix)] = Wj
        Phi[np.ix_(ix, ix)] = Pj
        Phis.append(Pj)
    M = np.linalg.inv(X.T @ W @ X)
    IH = np.eye(N) - X @ M @ X.T @ W
    e = y - X @ beta
    A = []
    for Pj, ix in zip(Phis, idx):
        Dj = np.linalg.cholesky((Pj + Pj.T) / 2).T
        Bj = Dj @ (IH[ix, :] @ Phi @ IH[ix, :].T) @ Dj.T
        A.append(Dj.T @ _sqrtm_psd(Bj, inverse=True) @ Dj)
    meat = np.zeros((p, p))
    for Aj, Wj, ix in zip(A, W_blocks, idx):
        u = X[ix].T @ Wj @ Aj @ e[ix]
        meat += np.outer(u, u)
    V = M @ meat @ M

    def df(c):
        qs = [IH[ix, :].T @ (Aj.T @ (Wj @ (X[ix] @ (M @ c)))) for Aj, Wj, ix in zip(A, W_blocks, idx)]
        Q = np.column_stack(qs)
        G = Q.T @ Phi @ Q
        s2 = float(np.sum(G ** 2))
        return float(np.trace(G) ** 2 / s2) if s2 > 0 else float("nan")
    return V, df


def robust_cr2(fit: DepFit) -> pd.DataFrame:
    """CR2 cluster-robust variance with Satterthwaite df under the fitted working model (Pustejovsky & Tipton 2018)."""
    e = fit.extra
    X, y, idx, Wb, beta = e["X"], e["y"], e["idx"], e["Winv"], e["beta"]
    p = X.shape[1]
    V, dff = cr2_core(X, y, Wb, idx, beta)
    se = np.sqrt(np.diag(V))
    rows = []
    for c_i in range(p):
        c = np.zeros(p)
        c[c_i] = 1.0
        df = dff(c)
        tval = beta[c_i] / se[c_i]
        crit = tdist.ppf(0.975, df)
        rows.append({"term": e["names"][c_i], "estimate": beta[c_i], "se (CR2)": se[c_i], "df (Satterthwaite)": df,
                     "ci_lo": beta[c_i] - crit * se[c_i], "ci_hi": beta[c_i] + crit * se[c_i], "t": tval,
                     "p": float(2 * tdist.sf(abs(tval), df)),
                     "note": "df < 4: robust test unreliable (Tipton 2015)" if df < 4 else ""})
    return pd.DataFrame(rows)


def che_rve(df: pd.DataFrame, cluster: str = "study_id", moderators: list[str] | None = None, rho: float = 0.6) -> dict[str, Any]:
    """Correlated-hierarchical-effects working model + CR2 robust inference (Pustejovsky & Tipton 2022)."""
    fit = fit_dependent(df, cluster, moderators, rho)
    rob = robust_cr2(fit)
    return {"fit": fit, "robust": rob, "rho": rho}


def multilevel(df: pd.DataFrame, cluster: str = "study_id", moderators: list[str] | None = None) -> dict[str, Any]:
    fit = fit_dependent(df, cluster, moderators, 0.0)
    out = {"fit": fit, "level_tests": level_tests(df, cluster, moderators, 0.0)}
    return out


def dependency_summary(df: pd.DataFrame, cluster: str = "study_id") -> dict[str, Any]:
    sizes = df.groupby(cluster).size()
    return {"effects": int(len(df)), "clusters": int(len(sizes)), "max_effects_per_cluster": int(sizes.max()) if len(sizes) else 0,
            "clusters_with_several_effects": int((sizes > 1).sum())}
