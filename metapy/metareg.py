"""Meta-regression with interactions, tests of linear hypotheses, predictions and cluster-robust inference (MetaPy 5.1).

Model          : mixed-effects meta-regression y = X b + u + e, u ~ N(0, tau^2), e ~ N(0, v), any tau^2 estimator from
                 metapy.extended; Wald-type (z) or Knapp-Hartung (t / F) inference (Knapp & Hartung 2003).
Terms          : "a", "b", "a:b" (interaction) and "a*b" (main effects plus interaction); categorical moderators are
                 dummy-coded against their first level (alphabetical) unless a reference level is given.
Hypotheses     : L b = 0 for any matrix L (Wald chi-square or F with Knapp-Hartung).
Predictions    : fitted values at chosen moderator values, with confidence and prediction intervals as in metafor.
Robust         : CR2 cluster-robust variance with Satterthwaite degrees of freedom for any fitted model
                 (Tipton 2015; Pustejovsky & Tipton 2018).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as fdist, norm, t as tdist

from .extended import estimate_tau2


# ----------------------------------------------------------------------------- design matrix
def _expand_terms(terms: Sequence[str]) -> list[str]:
    out: list[str] = []
    for t in terms:
        t = t.strip()
        if "*" in t:
            parts = [p.strip() for p in t.split("*")]
            for p_ in parts:
                if p_ not in out:
                    out.append(p_)
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    it = f"{parts[i]}:{parts[j]}"
                    if it not in out:
                        out.append(it)
        elif t and t not in out:
            out.append(t)
    return out


def _column_block(data: pd.DataFrame, col: str, ref: dict[str, str] | None):
    num = pd.to_numeric(data[col], errors="coerce")
    if num.notna().all():
        return num.to_numpy(float).reshape(-1, 1), [col], {"type": "numeric"}
    s = data[col].astype(str)
    levels = sorted(s.unique())
    r = (ref or {}).get(col, levels[0])
    if r not in levels:
        r = levels[0]
    others = [lv for lv in levels if lv != r]
    M = np.column_stack([(s == lv).to_numpy(float) for lv in others]) if others else np.zeros((len(s), 0))
    return M, [f"{col} = {lv}" for lv in others], {"type": "categorical", "levels": levels, "reference": r}


def design(data: pd.DataFrame, terms: Sequence[str], reference: dict[str, str] | None = None, intercept: bool = True):
    """Build X for the given terms. Returns X, column names and a coding description used for predictions."""
    terms = _expand_terms(terms)
    blocks, names, coding = [], [], {}
    if intercept:
        blocks.append(np.ones((len(data), 1)))
        names.append("Intercept")
    cache: dict[str, tuple] = {}
    for t in terms:
        parts = [p.strip() for p in t.split(":")]
        mats = []
        for p_ in parts:
            if p_ not in cache:
                cache[p_] = _column_block(data, p_, reference)
                coding[p_] = cache[p_][2]
            mats.append(cache[p_])
        M, nm = mats[0][0], mats[0][1]
        for M2, nm2, _ in mats[1:]:
            M = np.column_stack([M[:, i] * M2[:, j] for i in range(M.shape[1]) for j in range(M2.shape[1])])
            nm = [f"{a} × {b}" for a in nm for b in nm2]
        blocks.append(M)
        names.extend(nm)
    X = np.hstack(blocks) if blocks else np.zeros((len(data), 0))
    return X, names, {"terms": terms, "coding": coding, "intercept": intercept}


def _row_for(values: dict[str, Any], spec: dict) -> np.ndarray:
    """Design row for one set of moderator values (numeric value, or category label)."""
    row = [1.0] if spec["intercept"] else []
    cache = {}
    for var, cd in spec["coding"].items():
        if cd["type"] == "numeric":
            cache[var] = np.array([float(values[var])])
        else:
            lab = str(values[var])
            cache[var] = np.array([1.0 if lab == lv else 0.0 for lv in cd["levels"] if lv != cd["reference"]])
    for t in spec["terms"]:
        parts = [p.strip() for p in t.split(":")]
        vec = cache[parts[0]]
        for p_ in parts[1:]:
            vec = np.array([a * b for a in vec for b in cache[p_]])
        row.extend(vec.tolist())
    return np.array(row)


# ----------------------------------------------------------------------------- fit
@dataclass
class MetaRegFit:
    beta: np.ndarray
    vcov: np.ndarray
    names: list[str]
    tau2: float
    k: int
    p: int
    test: str
    X: np.ndarray
    y: np.ndarray
    v: np.ndarray
    spec: dict = field(default_factory=dict)
    QE: float = float("nan")
    tau2_null: float = float("nan")
    s2: float = 1.0

    @property
    def df(self) -> int:
        return self.k - self.p

    def table(self, level=0.95) -> pd.DataFrame:
        se = np.sqrt(np.diag(self.vcov))
        stat = self.beta / se
        if self.test == "knha":
            crit = tdist.ppf(0.5 + level / 2, self.df)
            p = 2 * tdist.sf(np.abs(stat), self.df)
        else:
            crit = norm.ppf(0.5 + level / 2)
            p = 2 * norm.sf(np.abs(stat))
        return pd.DataFrame({"term": self.names, "estimate": self.beta, "se": se, "statistic": stat, "p": p,
                             "ci_lo": self.beta - crit * se, "ci_hi": self.beta + crit * se})


def fit(y, v, X, names, tau_method="REML", test="knha", spec=None) -> MetaRegFit:
    y, v, X = np.asarray(y, float), np.asarray(v, float), np.asarray(X, float)
    k, p = X.shape
    if k - p < 1:
        raise ValueError(f"Not estimable: {k} studies for {p} coefficients")
    if np.linalg.matrix_rank(X) < p:
        raise ValueError("Not estimable: moderators are collinear (some coefficients cannot be separated)")
    tau2 = 0.0 if tau_method.upper() in ("FE", "EE", "CE") else estimate_tau2(y, v, tau_method, X)
    w = 1 / (v + tau2)
    XtWX = X.T @ (w[:, None] * X)
    vb = np.linalg.inv(XtWX)
    beta = vb @ X.T @ (w * y)
    s2 = 1.0
    if test == "knha":
        r = y - X @ beta
        s2 = max(1.0, float(np.sum(w * r ** 2)) / (k - p))      # truncated at 1 (as in the modified method)
        vb = vb * s2
    w0 = 1 / v
    P0 = np.diag(w0) - (w0[:, None] * X) @ np.linalg.solve(X.T @ (w0[:, None] * X), (w0[:, None] * X).T)
    qe = float(y @ P0 @ y)
    t2n = estimate_tau2(y, v, tau_method) if tau_method.upper() not in ("FE", "EE", "CE") else 0.0
    return MetaRegFit(beta, vb, list(names), tau2, k, p, test, X, y, v, spec or {}, qe, t2n, s2)


def fit_df(data: pd.DataFrame, terms: Sequence[str], tau_method="REML", test="knha", reference=None) -> MetaRegFit:
    cols = sorted({p_.strip() for t in _expand_terms(terms) for p_ in t.split(":")})
    d = data.dropna(subset=cols + ["yi", "vi"]).reset_index(drop=True)
    X, names, spec = design(d, terms, reference)
    return fit(d["yi"], d["vi"], X, names, tau_method, test, spec)


def r2(fit_: MetaRegFit) -> float:
    return max(0.0, (fit_.tau2_null - fit_.tau2) / fit_.tau2_null) * 100 if fit_.tau2_null > 0 else 0.0


# ----------------------------------------------------------------------------- hypotheses
def linear_hypothesis(fit_: MetaRegFit, L, rhs=None) -> dict[str, Any]:
    """Test L b = rhs. With Knapp-Hartung, F(q, k - p); otherwise Wald chi-square(q)."""
    L = np.atleast_2d(np.asarray(L, float))
    q = L.shape[0]
    rhs = np.zeros(q) if rhs is None else np.asarray(rhs, float)
    d = L @ fit_.beta - rhs
    W = float(d @ np.linalg.solve(L @ fit_.vcov @ L.T, d))
    if fit_.test == "knha":
        F = W / q
        return {"statistic": F, "test": f"F({q}, {fit_.df})", "p": float(fdist.sf(F, q, fit_.df)), "q": q}
    return {"statistic": W, "test": f"chi-square({q})", "p": float(chi2.sf(W, q)), "q": q}


def test_terms(fit_: MetaRegFit, pattern: str) -> dict[str, Any]:
    """Omnibus test of all coefficients whose name starts with 'pattern' (e.g. a categorical moderator)."""
    idx = [i for i, n in enumerate(fit_.names) if n == pattern or n.startswith(pattern + " = ") or n.startswith(pattern + " ×")
           or f"× {pattern}" in n]
    L = np.zeros((len(idx), fit_.p))
    for r_, i in enumerate(idx):
        L[r_, i] = 1.0
    out = linear_hypothesis(fit_, L)
    out["terms"] = [fit_.names[i] for i in idx]
    return out


def omnibus(fit_: MetaRegFit) -> dict[str, Any]:
    idx = [i for i, n in enumerate(fit_.names) if n != "Intercept"]
    L = np.zeros((len(idx), fit_.p))
    for r_, i in enumerate(idx):
        L[r_, i] = 1.0
    return linear_hypothesis(fit_, L)


# ----------------------------------------------------------------------------- predictions
def predict(fit_: MetaRegFit, newdata: Sequence[dict[str, Any]] | None = None, Xnew=None, level=0.95) -> pd.DataFrame:
    """Predicted average effect with CI and prediction interval (pred ± crit * sqrt(se^2 + tau^2), as in metafor)."""
    if Xnew is None:
        Xnew = np.array([_row_for(nd, fit_.spec) for nd in newdata])
    Xnew = np.atleast_2d(np.asarray(Xnew, float))
    pred = Xnew @ fit_.beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xnew, fit_.vcov, Xnew))
    crit = tdist.ppf(0.5 + level / 2, fit_.df) if fit_.test == "knha" else norm.ppf(0.5 + level / 2)
    pse = np.sqrt(se ** 2 + fit_.tau2)
    out = pd.DataFrame({"pred": pred, "se": se, "ci_lo": pred - crit * se, "ci_hi": pred + crit * se,
                        "pi_lo": pred - crit * pse, "pi_hi": pred + crit * pse})
    if newdata is not None:
        out.insert(0, "at", [", ".join(f"{k_}={v_}" for k_, v_ in nd.items()) for nd in newdata])
    return out


# ----------------------------------------------------------------------------- robust variance
def _sqrtm_psd(M, inverse=False):
    w, U = np.linalg.eigh((M + M.T) / 2)
    w = np.clip(w, 1e-12, None)
    return (U * (w ** (-0.5 if inverse else 0.5))) @ U.T


def robust(fit_: MetaRegFit, cluster: Sequence[Any]) -> pd.DataFrame:
    """CR2 cluster-robust standard errors with Satterthwaite df under the fitted working weights 1/(v + tau^2)."""
    X, y = fit_.X, fit_.y
    w = 1 / (fit_.v + fit_.tau2)
    cl = pd.Series(list(cluster)).astype(str)
    groups = [np.where(cl.to_numpy() == g)[0] for g in pd.unique(cl)]
    sw = np.sqrt(w)
    Xt, yt = X * sw[:, None], y * sw
    N, p = X.shape
    B = np.linalg.inv(Xt.T @ Xt)
    beta = B @ Xt.T @ yt
    M = np.eye(N) - Xt @ B @ Xt.T
    et = M @ yt
    A = [_sqrtm_psd(M[np.ix_(ix, ix)], inverse=True) for ix in groups]
    meat = np.zeros((p, p))
    for Aj, ix in zip(A, groups):
        u = Xt[ix].T @ Aj @ et[ix]
        meat += np.outer(u, u)
    V = B @ meat @ B
    se = np.sqrt(np.diag(V))
    rows = []
    for ci in range(p):
        c = np.zeros(p)
        c[ci] = 1.0
        avecs = [M[ix, :].T @ (Aj @ Xt[ix] @ B @ c) for Aj, ix in zip(A, groups)]
        G = np.array([[a1 @ a2 for a2 in avecs] for a1 in avecs])
        df = float(np.trace(G) ** 2 / np.sum(G ** 2)) if np.sum(G ** 2) > 0 else float("nan")
        tval = beta[ci] / se[ci]
        crit = tdist.ppf(0.975, df)
        rows.append({"term": fit_.names[ci], "estimate": beta[ci], "se (CR2)": se[ci], "df (Satterthwaite)": df, "t": tval,
                     "p": float(2 * tdist.sf(abs(tval), df)), "ci_lo": beta[ci] - crit * se[ci], "ci_hi": beta[ci] + crit * se[ci],
                     "note": "df < 4: robust test unreliable (Tipton 2015)" if df < 4 else ""})
    return pd.DataFrame(rows)
