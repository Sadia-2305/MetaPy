from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, t


def _tau2_dl(y: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / v
    fixed_mean = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - fixed_mean) ** 2))
    df = len(y) - 1
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    return max(0.0, (q - df) / c) if c > 0 else 0.0


def _tau2_reml(y: np.ndarray, v: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0

    def objective(tau2: float) -> float:
        vi = v + tau2
        wi = 1.0 / vi
        sw = np.sum(wi)
        mu = np.sum(wi * y) / sw
        residual = np.sum(wi * (y - mu) ** 2)
        return float(0.5 * (np.sum(np.log(vi)) + np.log(sw) + residual))

    upper = max(1.0, float(np.var(y)) * 10.0 + float(np.max(v)) * 10.0)
    opt = minimize_scalar(objective, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    candidates = [(0.0, objective(0.0)), (float(opt.x), float(opt.fun)), (upper, objective(upper))]
    return min(candidates, key=lambda x: x[1])[0]


def pool_effects(
    effects: pd.DataFrame,
    *,
    effect_col: str = "effect_estimate",
    variance_col: str = "sampling_variance",
    model: str = "random",
    tau_method: str = "REML",
    ci_method: str = "Hartung-Knapp",
    min_studies: int = 2,
    transform_back: bool | None = None,
) -> dict[str, Any]:
    """Pool independent rows with positive, finite variances.

    Assumes dependence has already been handled by the selected shared-control
    policy. It does not implement multilevel or robust-variance meta-analysis.
    """
    valid = effects.copy()
    valid[effect_col] = pd.to_numeric(valid[effect_col], errors="coerce")
    valid[variance_col] = pd.to_numeric(valid[variance_col], errors="coerce")
    valid = valid[np.isfinite(valid[effect_col]) & np.isfinite(valid[variance_col]) & (valid[variance_col] > 0)]
    k = len(valid)
    studies = int(valid["study_id"].nunique()) if "study_id" in valid else k
    base = {
        "effect_count": k, "study_count": studies, "model": model,
        "tau_method": tau_method if model == "random" else "not applicable",
        "ci_method": ci_method if model == "random" else "normal approximation",
        "pooled_effect": None, "ci_lower": None, "ci_upper": None,
        "standard_error": None, "p_value": None, "tau_squared": None,
        "tau": None, "i_squared_percent": None, "q_statistic": None,
        "q_p_value": None, "prediction_interval_lower": None,
        "prediction_interval_upper": None,
        "eggers_intercept": None,
        "eggers_p_value": None,
        "publication_bias_test_status": "Not assessed",
        "status": "Not pooled",
    }
    if k < min_studies:
        base["status"] = f"Not pooled — at least {min_studies} valid comparisons are required; found {k}."
        return base

    y = valid[effect_col].to_numpy(dtype=float)
    v = valid[variance_col].to_numpy(dtype=float)
    w_fixed = 1.0 / v
    sum_w_fixed = float(np.sum(w_fixed))
    mu_fixed = float(np.sum(w_fixed * y) / sum_w_fixed)
    q = float(np.sum(w_fixed * (y - mu_fixed) ** 2))
    df_q = k - 1
    q_p = float(chi2.sf(q, df_q)) if df_q > 0 else float("nan")
    i2 = max(0.0, (q - df_q) / q) * 100.0 if q > 0 else 0.0

    if model == "fixed":
        mu = mu_fixed
        se = math.sqrt(1.0 / sum_w_fixed)
        crit = norm.ppf(0.975)
        ci_lo, ci_hi = mu - crit * se, mu + crit * se
        pval = float(2 * norm.sf(abs(mu / se)))
        tau2 = 0.0
        pi_lo = pi_hi = None
    elif model == "random":
        if tau_method.upper() == "DL":
            tau2 = _tau2_dl(y, v)
        elif tau_method.upper() == "REML":
            tau2 = _tau2_reml(y, v)
        else:
            raise ValueError("tau_method must be 'REML' or 'DL'.")
        w = 1.0 / (v + tau2)
        sw = float(np.sum(w))
        mu = float(np.sum(w * y) / sw)
        se_model = math.sqrt(1.0 / sw)
        q_re = float(np.sum(w * (y - mu) ** 2))
        if ci_method.lower() in {"hartung-knapp", "hk", "modified hartung-knapp", "mkh"}:
            q_factor = max(1.0, q_re / max(k - 1, 1))
            se = math.sqrt(q_factor / sw)
            crit = t.ppf(0.975, df=k - 1)
            ci_lo, ci_hi = mu - crit * se, mu + crit * se
            pval = float(2 * t.sf(abs(mu / se), df=k - 1))
        else:
            se = se_model
            crit = norm.ppf(0.975)
            ci_lo, ci_hi = mu - crit * se, mu + crit * se
            pval = float(2 * norm.sf(abs(mu / se)))
        if k >= 3:
            pred_se = math.sqrt(tau2 + se_model ** 2)
            pred_crit = t.ppf(0.975, df=max(k - 2, 1))
            pi_lo, pi_hi = mu - pred_crit * pred_se, mu + pred_crit * pred_se
        else:
            pi_lo = pi_hi = None
    else:
        raise ValueError("model must be 'fixed' or 'random'.")

    base.update({
        "pooled_effect": mu, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "standard_error": se, "p_value": pval,
        "tau_squared": tau2, "tau": math.sqrt(tau2),
        "i_squared_percent": i2, "q_statistic": q, "q_p_value": q_p,
        "prediction_interval_lower": pi_lo, "prediction_interval_upper": pi_hi,
        "status": "Pooled",
    })

    # Egger regression is not attempted on small or internally dependent sets.
    if "study_id" in valid and valid["study_id"].nunique() < 10:
        base["publication_bias_test_status"] = "Not performed — fewer than 10 independent studies"
    elif "study_id" in valid and valid["study_id"].astype(str).duplicated().any():
        base["publication_bias_test_status"] = "Not performed — multiple effects from one study would violate the simple Egger regression independence assumption"
    else:
        # Egger regression: standardized normal deviate on precision.
        precision = 1.0 / np.sqrt(v)
        sn = y / np.sqrt(v)
        X = np.column_stack([np.ones(len(y)), precision])
        beta, *_ = np.linalg.lstsq(X, sn, rcond=None)
        residual = sn - X @ beta
        df_egger = len(y) - 2
        if df_egger > 0:
            s2 = float(np.sum(residual ** 2) / df_egger)
            covariance = s2 * np.linalg.inv(X.T @ X)
            se_intercept = math.sqrt(max(float(covariance[0, 0]), 0.0))
            t_stat = float(beta[0] / se_intercept) if se_intercept > 0 else float("inf")
            base["eggers_intercept"] = float(beta[0])
            base["eggers_p_value"] = float(2 * t.sf(abs(t_stat), df_egger))
            base["publication_bias_test_status"] = "Calculated — interpret cautiously; funnel asymmetry is not proof of publication bias"
        else:
            base["publication_bias_test_status"] = "Not performed — insufficient degrees of freedom"

    if transform_back is None:
        transform_back = bool(valid.get("estimand", pd.Series(dtype=str)).astype(str).isin(["log_rr", "log_or"]).all()) if "estimand" in valid else False
    if transform_back and base["pooled_effect"] is not None:
        for key in ["pooled_effect", "ci_lower", "ci_upper", "prediction_interval_lower", "prediction_interval_upper"]:
            if base[key] is not None:
                base[key + "_transformed"] = math.exp(base[key])
    return base


def pool_by_outcome(
    results: pd.DataFrame,
    *,
    include_col: str = "include_in_primary_pool",
    model: str = "random",
    tau_method: str = "REML",
    ci_method: str = "Hartung-Knapp",
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    data = results.copy()
    if include_col in data:
        data = data[data[include_col].fillna(False).astype(bool)]
    if data.empty:
        return pd.DataFrame([{"status": "No comparisons passed the pooling gates."}])
    group_cols = ["outcome"]
    if "duration" in data and data["duration"].notna().any():
        group_cols.append("duration")
    rows = []
    for key, part in data.groupby(group_cols, dropna=False, sort=False):
        pooled = pool_effects(part, model=model, tau_method=tau_method, ci_method=ci_method)
        if not isinstance(key, tuple):
            key = (key,)
        row = {name: value for name, value in zip(group_cols, key)}
        if "estimand" in part and part["estimand"].nunique(dropna=True) == 1:
            row["estimand"] = part["estimand"].dropna().iloc[0]
        elif "estimand" in part:
            row["estimand"] = "Mixed estimands — do not pool together"
        row.update(pooled)
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_study_out(
    group: pd.DataFrame,
    *,
    model: str = "random",
    tau_method: str = "REML",
    ci_method: str = "Hartung-Knapp",
) -> pd.DataFrame:
    if group.empty or "study_id" not in group:
        return pd.DataFrame()
    rows = []
    for study in sorted(group["study_id"].astype(str).unique()):
        subset = group[group["study_id"].astype(str) != study]
        pooled = pool_effects(subset, model=model, tau_method=tau_method, ci_method=ci_method)
        rows.append({"study_excluded": study, **pooled})
    return pd.DataFrame(rows)
