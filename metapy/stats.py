"""Meta-analytic statistics for MetaPy.

Pooling (fixed / random with DL, REML, PM tau-squared; Wald or Hartung-Knapp
CIs; prediction intervals) is adapted from MetaPy 3.3's pairwise engine. Added:
subgroup analysis, mixed-effects meta-regression, leave-one-out, influence
diagnostics, cumulative meta-analysis, Egger, Begg and trim-and-fill.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, kendalltau, norm, t as tdist


# ----------------------------------------------------------- tau-squared
def tau2_dl(y, v):
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    return max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0


def tau2_reml(y, v, X=None):
    k = len(y)
    X = np.ones((k, 1)) if X is None else X
    p = X.shape[1]
    if k - p < 1:
        return 0.0

    def nll(tau2):
        w = 1 / (v + tau2)
        XtWX = X.T @ (w[:, None] * X)
        try:
            beta = np.linalg.solve(XtWX, X.T @ (w * y))
        except np.linalg.LinAlgError:
            return 1e10
        r = y - X @ beta
        sign, logdet = np.linalg.slogdet(XtWX)
        return 0.5 * (np.sum(np.log(v + tau2)) + logdet + np.sum(w * r ** 2))

    upper = max(1.0, float(np.var(y)) * 10 + float(np.max(v)) * 10)
    opt = minimize_scalar(nll, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    cands = [(0.0, nll(0.0)), (float(opt.x), float(opt.fun))]
    return min(cands, key=lambda x: x[1])[0]


def tau2_pm(y, v):
    k = len(y)
    if k < 2:
        return 0.0

    def f(tau2):
        w = 1 / (v + tau2)
        mu = np.sum(w * y) / np.sum(w)
        return np.sum(w * (y - mu) ** 2) - (k - 1)
    if f(0.0) <= 0:
        return 0.0
    hi = max(1.0, float(np.var(y)) * 10)
    while f(hi) > 0:
        hi *= 2
    return float(brentq(f, 0.0, hi, xtol=1e-12))


def estimate_tau2(y, v, method="REML", X=None):
    m = method.upper()
    if m in ("ML", "EB", "HE", "HS", "HSK", "SJ") or (m == "PM" and X is not None):
        from . import extended
        return extended.estimate_tau2(np.asarray(y, float), np.asarray(v, float), "HSk" if m == "HSK" else m, X)
    if m == "DL":
        if X is None:
            return tau2_dl(y, v)
        return _tau2_dl_reg(y, v, X)
    if m == "PM":
        return tau2_pm(y, v) if X is None else _tau2_dl_reg(y, v, X)
    return tau2_reml(y, v, X)


def _tau2_dl_reg(y, v, X):
    w = 1 / v
    W = np.diag(w)
    k, p = X.shape
    P = W - W @ X @ np.linalg.inv(X.T @ W @ X) @ X.T @ W
    qe = float(y @ P @ y)
    tr = float(np.trace(P))
    return max(0.0, (qe - (k - p)) / tr) if tr > 0 else 0.0


# ----------------------------------------------------------- pooling
def pool(y, v, model="random", tau_method="REML", ci_method="HK", min_k=2) -> dict[str, Any]:
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = len(y)
    res: dict[str, Any] = {"k": k, "model": model, "tau_method": tau_method if model == "random" else "—",
                           "ci_method": ci_method if model == "random" else "Wald (z)", "status": "Pooled"}
    if k < min_k:
        res["status"] = f"Not pooled: {k} study effect(s); at least {min_k} needed"
        return res
    w0 = 1 / v
    mu0 = float(np.sum(w0 * y) / np.sum(w0))
    q = float(np.sum(w0 * (y - mu0) ** 2))
    dfq = k - 1
    res.update({"Q": q, "Q_df": dfq, "Q_p": float(chi2.sf(q, dfq)) if dfq > 0 else float("nan"),
                "I2": max(0.0, (q - dfq) / q) * 100 if q > 0 else 0.0,
                "H2": q / dfq if dfq > 0 and q > 0 else float("nan")})
    if model == "fixed":
        tau2 = 0.0
    else:
        tau2 = estimate_tau2(y, v, tau_method)
    if model == "random" and k >= 2:
        # I2 and H2 consistent with the chosen tau2 estimator (Higgins & Thompson 2002; as in metafor)
        s2 = (k - 1) * float(np.sum(w0)) / (float(np.sum(w0)) ** 2 - float(np.sum(w0 ** 2)))
        res["I2"] = 100 * tau2 / (tau2 + s2)
        res["H2"] = (tau2 + s2) / s2
    w = 1 / (v + tau2)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y) / sw)
    se0 = math.sqrt(1 / sw)
    if model == "random" and ci_method.upper() in ("HK", "HARTUNG-KNAPP", "KNHA") and k >= 2:
        qhk = float(np.sum(w * (y - mu) ** 2)) / (k - 1)
        se = math.sqrt(max(qhk, 1.0) / sw)  # modified HK: never narrower than Wald
        crit = float(tdist.ppf(0.975, k - 1))
        z = mu / se
        p = float(2 * tdist.sf(abs(z), k - 1))
        stat_label = f"t({k - 1})"
    else:
        se = se0
        crit = float(norm.ppf(0.975))
        z = mu / se
        p = float(2 * norm.sf(abs(z)))
        stat_label = "z"
    res.update({"estimate": mu, "se": se, "ci_lo": mu - crit * se, "ci_hi": mu + crit * se, "stat": z,
                "stat_label": stat_label, "p": p, "tau2": tau2, "tau": math.sqrt(tau2),
                "weights": (w / sw * 100).tolist()})
    if model == "random" and k >= 3:
        pse = math.sqrt(tau2 + se ** 2)
        pc = float(tdist.ppf(0.975, k - 2))
        res.update({"pi_lo": mu - pc * pse, "pi_hi": mu + pc * pse})
    return res


# ----------------------------------------------------------- subgroups
def subgroup(df: pd.DataFrame, col: str, **kw) -> dict[str, Any]:
    rows = []
    for level, g in df.groupby(df[col].astype(str), sort=True):
        r = pool(g["yi"], g["vi"], **kw)
        r["subgroup"] = level
        r["studies"] = g["study_id"].nunique()
        rows.append(r)
    table = pd.DataFrame(rows)
    valid = table[table["status"] == "Pooled"]
    out = {"table": table, "variable": col}
    if len(valid) >= 2:
        est = valid["estimate"].to_numpy(float)
        var = (valid["se"].to_numpy(float)) ** 2
        w = 1 / var
        mu = np.sum(w * est) / np.sum(w)
        qb = float(np.sum(w * (est - mu) ** 2))
        out.update({"Q_between": qb, "df": len(valid) - 1, "p_between": float(chi2.sf(qb, len(valid) - 1))})
    else:
        out["note"] = "Test for subgroup differences needs at least two subgroups with 2+ studies each."
    return out


# ----------------------------------------------------------- meta-regression
def meta_regression(df: pd.DataFrame, cols: list[str], tau_method="REML", knha=True) -> dict[str, Any]:
    data = df.dropna(subset=cols).copy()
    k = len(data)
    pieces, names = [np.ones((k, 1))], ["Intercept"]
    for c in cols:
        num = pd.to_numeric(data[c], errors="coerce")
        if num.notna().all():
            pieces.append(num.to_numpy(float).reshape(-1, 1))
            names.append(c)
        else:
            dummies = pd.get_dummies(data[c].astype(str), drop_first=True, dtype=float)
            for dc in dummies.columns:
                pieces.append(dummies[dc].to_numpy().reshape(-1, 1))
                names.append(f"{c} = {dc}")
    X = np.hstack(pieces)
    p = X.shape[1]
    y, v = data["yi"].to_numpy(float), data["vi"].to_numpy(float)
    if k - p < 1:
        return {"status": f"Not estimable: {k} studies for {p} coefficients", "k": k}
    tau2 = estimate_tau2(y, v, tau_method, X)
    tau2_null = estimate_tau2(y, v, tau_method)
    w = 1 / (v + tau2)
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    resid = y - X @ beta
    W0 = np.diag(1 / v)
    P0 = W0 - W0 @ X @ np.linalg.inv(X.T @ W0 @ X) @ X.T @ W0
    qe = float(y @ P0 @ y)
    if knha:
        s2 = max(1.0, float(np.sum(w * resid ** 2)) / (k - p))
        cov = cov * s2
        se = np.sqrt(np.diag(cov))
        stat = beta / se
        pv = 2 * tdist.sf(np.abs(stat), k - p)
        crit = tdist.ppf(0.975, k - p)
        test = f"t({k - p})"
    else:
        se = np.sqrt(np.diag(cov))
        stat = beta / se
        pv = 2 * norm.sf(np.abs(stat))
        crit = norm.ppf(0.975)
        test = "z"
    b1, c1 = beta[1:], cov[1:, 1:]
    qm = float(b1 @ np.linalg.inv(c1) @ b1) if p > 1 else float("nan")
    qm_p = float(chi2.sf(qm, p - 1)) if not knha else float(1 - __import__("scipy.stats", fromlist=["f"]).f.cdf(qm / (p - 1), p - 1, k - p))
    coef = pd.DataFrame({"term": names, "estimate": beta, "se": se, "ci_lo": beta - crit * se, "ci_hi": beta + crit * se,
                         "statistic": stat, "p": pv})
    r2 = max(0.0, (tau2_null - tau2) / tau2_null) * 100 if tau2_null > 0 else 0.0
    return {"status": "Estimated", "k": k, "coefficients": coef, "tau2_residual": tau2, "R2_percent": r2,
            "QE": qe, "QE_df": k - p, "QE_p": float(chi2.sf(qe, k - p)), "QM": qm, "QM_p": qm_p, "test": test,
            "moderators": cols, "X": X, "names": names}


# ----------------------------------------------------------- sensitivity
def leave_one_out(df: pd.DataFrame, **kw) -> pd.DataFrame:
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        sub = df.reset_index(drop=True).drop(index=i)
        res = pool(sub["yi"], sub["vi"], **kw)
        rows.append({"omitted": r["label"], "estimate": res.get("estimate"), "ci_lo": res.get("ci_lo"), "ci_hi": res.get("ci_hi"),
                     "p": res.get("p"), "I2": res.get("I2"), "tau2": res.get("tau2")})
    return pd.DataFrame(rows)


def influence(df: pd.DataFrame, **kw) -> pd.DataFrame:
    d = df.reset_index(drop=True)
    full = pool(d["yi"], d["vi"], **kw)
    rows = []
    for i, r in d.iterrows():
        sub = d.drop(index=i)
        res = pool(sub["yi"], sub["vi"], **kw)
        if res["status"] != "Pooled":
            continue
        resid = r["yi"] - res["estimate"]
        denom = math.sqrt(r["vi"] + res["tau2"] + res["se"] ** 2)
        cook = (full["estimate"] - res["estimate"]) ** 2 / full["se"] ** 2
        rows.append({"study": r["label"], "studentised_residual": resid / denom, "cooks_distance": cook,
                     "estimate_without": res["estimate"], "tau2_without": res["tau2"], "I2_without": res["I2"],
                     "flag": "Potentially influential" if abs(resid / denom) > 1.96 or cook > 4 / len(d) else ""})
    return pd.DataFrame(rows)


def cumulative(df: pd.DataFrame, order_col: str = "year", **kw) -> pd.DataFrame:
    d = df.copy()
    d["_ord"] = pd.to_numeric(d[order_col], errors="coerce")
    d = d.sort_values(["_ord", "label"]).reset_index(drop=True)
    rows = []
    for i in range(len(d)):
        sub = d.iloc[: i + 1]
        if i > 0:
            res = pool(sub["yi"], sub["vi"], **kw)
        else:
            se = math.sqrt(sub["vi"].iloc[0])
            res = {"estimate": sub["yi"].iloc[0], "ci_lo": sub["yi"].iloc[0] - 1.96 * se, "ci_hi": sub["yi"].iloc[0] + 1.96 * se, "p": float(2 * norm.sf(abs(sub["yi"].iloc[0] / se))), "I2": 0.0}
        rows.append({"added": d["label"].iloc[i], order_col: d["_ord"].iloc[i], "k": i + 1, "estimate": res.get("estimate"),
                     "ci_lo": res.get("ci_lo"), "ci_hi": res.get("ci_hi"), "p": res.get("p"), "I2": res.get("I2")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- small-study effects
def egger(y, v) -> dict[str, Any]:
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if k < 3:
        return {"status": "Not performed: fewer than 3 studies"}
    se = np.sqrt(v)
    X = np.column_stack([np.ones(k), 1 / se])
    zz = y / se
    beta, *_ = np.linalg.lstsq(X, zz, rcond=None)
    res = zz - X @ beta
    s2 = float(np.sum(res ** 2) / (k - 2))
    cov = s2 * np.linalg.inv(X.T @ X)
    t_ = beta[0] / math.sqrt(cov[0, 0])
    return {"status": "Performed" + (" (fewer than 10 studies: low power, interpret with caution)" if k < 10 else ""),
            "intercept": float(beta[0]), "se": math.sqrt(cov[0, 0]), "t": float(t_), "df": k - 2,
            "p": float(2 * tdist.sf(abs(t_), k - 2))}


def begg(y, v) -> dict[str, Any]:
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if k < 3:
        return {"status": "Not performed: fewer than 3 studies"}
    w = 1 / v
    mu = np.sum(w * y) / np.sum(w)
    vstar = v - 1 / np.sum(w)
    zs = (y - mu) / np.sqrt(np.maximum(vstar, 1e-12))
    tau, p = kendalltau(zs, v)
    return {"status": "Performed" + (" (fewer than 10 studies: low power)" if k < 10 else ""), "kendall_tau": float(tau), "p": float(p)}


def trim_and_fill(y, v, model="random", tau_method="REML", ci_method="HK", side: str | None = None, max_iter=50) -> dict[str, Any]:
    """Duval & Tweedie (2000) trim-and-fill with the L0 estimator."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k = len(y)
    if k < 3:
        return {"status": "Not performed: fewer than 3 studies"}
    base = pool(y, v, model, tau_method, ci_method)
    if side is None:  # impute on the side where Egger suggests missing studies
        e = egger(y, v)
        side = "left" if e.get("intercept", 0) > 0 else "right"
    sign = 1 if side == "left" else -1   # flip so that missing studies are on the left
    ys = sign * y
    k0 = 0
    mu = sign * base["estimate"]
    for _ in range(max_iter):
        order = np.argsort(ys)
        trimmed = order[: k - k0] if k0 > 0 else order
        r = pool(ys[trimmed], v[trimmed], model, tau_method, "Wald")   # same model as the analysis (as in metafor)
        mu = r["estimate"]
        dev = ys - mu
        ranks = pd.Series(np.abs(dev)).rank(method="first").to_numpy()
        tn = float(np.sum(ranks[dev > 0]))
        new_k0 = max(0, int(round((4 * tn - k * (k + 1)) / (2 * k - 1))))
        if new_k0 == k0:
            break
        k0 = min(new_k0, k - 1)
    if k0 == 0:
        return {"status": "Performed", "k0": 0, "side": side, "adjusted": base, "filled_y": [], "filled_v": []}
    order = np.argsort(ys)[::-1][:k0]  # most extreme on the opposite side
    fy = 2 * mu - ys[order]
    filled_y, filled_v = sign * fy, v[order]
    adj = pool(np.concatenate([y, filled_y]), np.concatenate([v, filled_v]), model, tau_method, ci_method)
    return {"status": "Performed", "k0": int(k0), "side": side, "adjusted": adj,
            "filled_y": filled_y.tolist(), "filled_v": filled_v.tolist()}
