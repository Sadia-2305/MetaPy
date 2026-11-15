from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

from .schema import as_float, is_missing

ESTIMANDS = {
    "morris_ppc2": "Morris PPC2 — difference in change standardised by pooled baseline SD",
    "posttest_smd": "Posttest Hedges' g — standardised post-intervention group difference",
    "change_smd": "Change-score Hedges' g — standardised between-group difference in change",
    "raw_posttest_md": "Raw posttest mean difference",
    "raw_change_md": "Raw between-group difference in change",
    "log_rr": "Log risk ratio",
    "log_or": "Log odds ratio",
    "risk_difference": "Risk difference",
}


def hedges_j(df: float) -> float:
    if df <= 0:
        return float("nan")
    return float(math.exp(gammaln(df / 2.0) - 0.5 * math.log(df / 2.0) - gammaln((df - 1.0) / 2.0)))


def _f(row: pd.Series | dict, key: str) -> float | None:
    return as_float(row.get(key))


def _valid_n(row: pd.Series | dict) -> tuple[float | None, float | None]:
    n_i, n_c = _f(row, "n_i"), _f(row, "n_c")
    if n_i is None or n_c is None or n_i < 2 or n_c < 2:
        return None, None
    return n_i, n_c


def _pooled_sd(sd_i: float, n_i: float, sd_c: float, n_c: float) -> tuple[float, float]:
    df = n_i + n_c - 2.0
    if df <= 0:
        return float("nan"), df
    variance = ((n_i - 1.0) * sd_i ** 2 + (n_c - 1.0) * sd_c ** 2) / df
    return math.sqrt(variance), df


def _estimand_matches(method: Any, estimand: str) -> bool:
    if is_missing(method):
        return False
    text = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    synonyms = {
        "morris_ppc2": {"morris_ppc2", "ppc2", "pretest_posttest_control_ppc2", "baseline_sd_standardized_difference_in_change"},
        "posttest_smd": {"posttest_smd", "posttest_hedges_g", "hedges_g_posttest", "cohens_d_posttest"},
        "change_smd": {"change_smd", "change_score_smd", "smcc", "standardized_difference_in_change_score"},
        "raw_posttest_md": {"raw_posttest_md", "posttest_mean_difference", "md_posttest"},
        "raw_change_md": {"raw_change_md", "change_mean_difference", "md_change"},
        "log_rr": {"log_rr", "risk_ratio", "rr"},
        "log_or": {"log_or", "odds_ratio", "or"},
        "risk_difference": {"risk_difference", "rd"},
    }
    return text in synonyms.get(estimand, {estimand})


def _get_direct_variance(row: pd.Series | dict, estimand: str) -> tuple[float | None, str | None]:
    method = row.get("reported_effect_method")
    if not _estimand_matches(method, estimand):
        return None, None
    if estimand == "raw_posttest_md":
        md_lo, md_hi = _f(row, "md_ci_lower"), _f(row, "md_ci_upper")
        if md_lo is not None and md_hi is not None and md_hi > md_lo:
            se_md = (md_hi - md_lo) / (2.0 * norm.ppf(0.975))
            return se_md ** 2, "Variance derived from retained raw-MD 95% CI; researcher confirmed raw posttest MD estimand"
    var = _f(row, "effect_variance")
    if var is not None and var > 0:
        return var, "Direct reported sampling variance; method confirmed as matching selected estimand"
    se = _f(row, "effect_se")
    if se is not None and se > 0:
        return se ** 2, "Direct reported SE squared; method confirmed as matching selected estimand"
    lo, hi = _f(row, "ci_lower"), _f(row, "ci_upper")
    if lo is not None and hi is not None and hi > lo:
        level = _f(row, "ci_level") or 0.95
        if 0.5 < level < 1.0:
            z = norm.ppf(0.5 + level / 2.0)
            se_ci = (hi - lo) / (2.0 * z)
            return se_ci ** 2, f"Variance derived from {level:.0%} CI; method confirmed as matching selected estimand"
    return None, None


def _raw_continuous(row: pd.Series | dict, estimand: str, direction: int, n_i: float, n_c: float,
                    correlation_override: tuple[float, float] | None = None,
                    variance_n_c: float | None = None,
                    correlation_method_label: str | None = None) -> dict:
    result = {"effect_estimate": None, "variance": None, "se": None, "estimate_source": None,
              "variance_method": None, "status": "Not calculable", "candidate_reason": None,
              "reported_effect_comparison": None, "approximation_note": None}
    df = n_i + n_c - 2.0
    j = hedges_j(df)

    if estimand == "morris_ppc2":
        keys = ["mean_pre_i", "mean_post_i", "sd_pre_i", "mean_pre_c", "mean_post_c", "sd_pre_c"]
        vals = {k: _f(row, k) for k in keys}
        if any(v is None for v in vals.values()):
            result["candidate_reason"] = "Morris PPC2 point estimate requires pre/post means and baseline SDs in both groups."
            return result
        sd0, df0 = _pooled_sd(vals["sd_pre_i"], n_i, vals["sd_pre_c"], n_c)
        if not math.isfinite(sd0) or sd0 <= 0:
            result["candidate_reason"] = "Pooled baseline SD is non-positive or not calculable."
            return result
        delta_i = vals["mean_post_i"] - vals["mean_pre_i"]
        delta_c = vals["mean_post_c"] - vals["mean_pre_c"]
        d = (delta_i - delta_c) / sd0
        g = j * d * direction
        result.update({"effect_estimate": g, "estimate_source": "Independently calculated from raw pre/post means and pooled baseline SD",
                       "status": "Point estimate calculated; variance route pending"})

        r_i = _f(row, "r_prepost_i")
        r_c = _f(row, "r_prepost_c")
        if correlation_override is not None:
            r_i, r_c = correlation_override
        if r_i is not None and r_c is not None and -1 <= r_i <= 1 and -1 <= r_c <= 1:
            sdi_change_sq = vals["sd_pre_i"]**2 + (_f(row, "sd_post_i") or 0)**2 - 2*r_i*vals["sd_pre_i"]*(_f(row, "sd_post_i") or 0)
            sdc_change_sq = vals["sd_pre_c"]**2 + (_f(row, "sd_post_c") or 0)**2 - 2*r_c*vals["sd_pre_c"]*(_f(row, "sd_post_c") or 0)
            if sdi_change_sq >= 0 and sdc_change_sq >= 0 and _f(row, "sd_post_i") is not None and _f(row, "sd_post_c") is not None:
                n_c_var = variance_n_c or n_c
                # Delta-method approximation: variance of the between-group
                # difference in mean change divided by baseline pooled SD,
                # plus first-order uncertainty in the pooled baseline SD.
                variance_delta = sdi_change_sq / n_i + sdc_change_sq / n_c_var
                d_variance = variance_delta / (sd0 ** 2) + d ** 2 / (2.0 * max(df0, 1.0))
                variance = (j ** 2) * d_variance
                if variance > 0 and math.isfinite(variance):
                    result["variance"] = variance
                    result["se"] = math.sqrt(variance)
                    result["variance_method"] = (
                        "Approximate delta-method variance using group-specific pre/post SDs and "
                        + (correlation_method_label or ("researcher-approved correlation sensitivity values" if correlation_override is not None else "source group-specific pre/post correlations"))
                        + "; baseline SD uncertainty included"
                    )
                    result["status"] = "Effect and variance calculated"
                    result["approximation_note"] = "Approximate delta-method variance; inspect the method description and report it explicitly."
        return result

    if estimand in {"posttest_smd", "change_smd"}:
        if estimand == "posttest_smd":
            mean_i, mean_c = _f(row, "mean_post_i"), _f(row, "mean_post_c")
            sd_i, sd_c = _f(row, "sd_post_i"), _f(row, "sd_post_c")
            if None in (mean_i, mean_c, sd_i, sd_c):
                result["candidate_reason"] = "Posttest SMD requires posttest means and SDs in both groups."
                return result
        else:
            needed = ["mean_pre_i", "mean_post_i", "sd_pre_i", "sd_post_i", "mean_pre_c", "mean_post_c", "sd_pre_c", "sd_post_c"]
            vals = {k: _f(row, k) for k in needed}
            if any(v is None for v in vals.values()):
                result["candidate_reason"] = "Change-score SMD requires pre/post means and SDs in both groups."
                return result
            r_i = correlation_override[0] if correlation_override else _f(row, "r_prepost_i")
            r_c = correlation_override[1] if correlation_override else _f(row, "r_prepost_c")
            if r_i is None or r_c is None or not (-1 <= r_i <= 1 and -1 <= r_c <= 1):
                result["candidate_reason"] = "Change-score SMD requires valid group-specific correlations; no imputation was made."
                return result
            sd_i_sq = vals["sd_pre_i"]**2 + vals["sd_post_i"]**2 - 2*r_i*vals["sd_pre_i"]*vals["sd_post_i"]
            sd_c_sq = vals["sd_pre_c"]**2 + vals["sd_post_c"]**2 - 2*r_c*vals["sd_pre_c"]*vals["sd_post_c"]
            if sd_i_sq <= 0 or sd_c_sq <= 0:
                result["candidate_reason"] = "Derived change-score SD is non-positive."
                return result
            mean_i = vals["mean_post_i"] - vals["mean_pre_i"]
            mean_c = vals["mean_post_c"] - vals["mean_pre_c"]
            sd_i, sd_c = math.sqrt(sd_i_sq), math.sqrt(sd_c_sq)
        pooled, df = _pooled_sd(sd_i, n_i, sd_c, n_c)
        if pooled <= 0 or not math.isfinite(pooled):
            result["candidate_reason"] = "Pooled SD is non-positive or not calculable."
            return result
        d = (mean_i - mean_c) / pooled
        g = j * d * direction
        n_c_var = variance_n_c or n_c
        df_var = n_i + n_c_var - 2.0
        variance = (j ** 2) * ((1.0 / n_i) + (1.0 / n_c_var) + (d ** 2) / (2.0 * max(df_var, 1.0)))
        result.update({"effect_estimate": g, "variance": variance, "se": math.sqrt(variance),
                       "estimate_source": "Independently calculated from raw group means and SDs",
                       "variance_method": "Hedges g independent-group SMD variance approximation",
                       "status": "Effect and variance calculated"})
        return result

    if estimand in {"raw_posttest_md", "raw_change_md"}:
        if estimand == "raw_posttest_md":
            mean_i, mean_c = _f(row, "mean_post_i"), _f(row, "mean_post_c")
            sd_i, sd_c = _f(row, "sd_post_i"), _f(row, "sd_post_c")
        else:
            vals = {k: _f(row, k) for k in ["mean_pre_i", "mean_post_i", "sd_pre_i", "sd_post_i", "r_prepost_i", "mean_pre_c", "mean_post_c", "sd_pre_c", "sd_post_c", "r_prepost_c"]}
            if any(v is None for v in vals.values()):
                result["candidate_reason"] = "Raw change MD requires means, SDs and both group-specific correlations."
                return result
            mean_i = vals["mean_post_i"] - vals["mean_pre_i"]
            mean_c = vals["mean_post_c"] - vals["mean_pre_c"]
            sd_i = math.sqrt(max(0, vals["sd_pre_i"]**2 + vals["sd_post_i"]**2 - 2*vals["r_prepost_i"]*vals["sd_pre_i"]*vals["sd_post_i"]))
            sd_c = math.sqrt(max(0, vals["sd_pre_c"]**2 + vals["sd_post_c"]**2 - 2*vals["r_prepost_c"]*vals["sd_pre_c"]*vals["sd_post_c"]))
        if None in (mean_i, mean_c, sd_i, sd_c):
            result["candidate_reason"] = "Raw MD requires compatible means and SDs in both groups."
            return result
        n_c_var = variance_n_c or n_c
        variance = sd_i**2/n_i + sd_c**2/n_c_var
        result.update({"effect_estimate": (mean_i - mean_c) * direction, "variance": variance, "se": math.sqrt(variance),
                       "estimate_source": "Independently calculated from raw group means",
                       "variance_method": "Independent-group raw mean difference variance",
                       "status": "Effect and variance calculated"})
        return result

    result["candidate_reason"] = f"Raw-data calculation is not implemented for estimand '{estimand}'."
    return result


def _binary_effect(row: pd.Series | dict, estimand: str, correction: float = 0.5, direction: int = 1) -> dict:
    ei, ni = _f(row, "events_i"), _f(row, "total_i") or _f(row, "n_i")
    ec, nc = _f(row, "events_c"), _f(row, "total_c") or _f(row, "n_c")
    out = {"effect_estimate": None, "variance": None, "se": None, "estimate_source": None,
           "variance_method": None, "status": "Not calculable", "candidate_reason": None,
           "reported_effect_comparison": None, "approximation_note": None}
    if None in (ei, ni, ec, nc) or ni <= 0 or nc <= 0 or not (0 <= ei <= ni and 0 <= ec <= nc):
        out["candidate_reason"] = "Binary effect requires valid event counts and totals in both groups."
        return out
    if estimand == "risk_difference":
        p1, p0 = ei/ni, ec/nc
        est = (p1 - p0) * direction
        var = p1*(1-p1)/ni + p0*(1-p0)/nc
        out.update({"effect_estimate": est, "variance": var, "se": math.sqrt(var),
                    "estimate_source": "Calculated from event counts and group totals",
                    "variance_method": "Wald independent-group risk-difference variance",
                    "status": "Effect and variance calculated"})
        return out
    a, b, c, d = ei, ni-ei, ec, nc-ec
    cc_used = False
    if min(a, b, c, d) == 0:
        a, b, c, d = a+correction, b+correction, c+correction, d+correction
        ni_adj, nc_adj = a+b, c+d
        cc_used = True
    else:
        ni_adj, nc_adj = ni, nc
    if estimand == "log_rr":
        p1, p0 = a/ni_adj, c/nc_adj
        if p1 <= 0 or p0 <= 0:
            out["candidate_reason"] = "Risk ratio is undefined with zero cells under current correction."
            return out
        est = math.log(p1/p0) * direction
        var = (1/a - 1/ni_adj) + (1/c - 1/nc_adj)
    else:
        est = math.log((a*d)/(b*c)) * direction
        var = 1/a + 1/b + 1/c + 1/d
    out.update({"effect_estimate": est, "variance": var, "se": math.sqrt(var),
                "estimate_source": "Calculated from event counts and group totals",
                "variance_method": ("Log risk-ratio sampling variance" if estimand == "log_rr" else "Log odds-ratio sampling variance") + (f"; continuity correction {correction:g} applied" if cc_used else ""),
                "status": "Effect and variance calculated"})
    return out


def calculate_effects(
    prepared: pd.DataFrame,
    estimand: str,
    *,
    record_approvals: dict[str, bool] | None = None,
    global_decisions: dict[str, Any] | None = None,
    correlation_sensitivity: list[float] | None = None,
    shared_control_policy: str = "block",
    zero_cell_correction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate row-level effects and variances; never silently impute.

    `correlation_sensitivity` is explicitly separate from primary calculations.
    """
    if estimand not in ESTIMANDS:
        raise ValueError(f"Unsupported estimand: {estimand}. Choose one of {list(ESTIMANDS)}")
    record_approvals = record_approvals or {}
    global_decisions = global_decisions or {}
    out_rows: list[dict[str, Any]] = []

    # Identify likely shared controls by study/outcome/time and same control N.
    work = prepared.copy()
    work["shared_control_comparison_count"] = 1
    if "study_id" in work and "outcome" in work:
        group_cols = ["study_id", "outcome"]
        if "duration" in work:
            group_cols.append("duration")
        for _, idxs in work.groupby(group_cols, dropna=False).groups.items():
            ids = list(idxs)
            sub = work.loc[ids]
            # An explicit Control_ID is the strongest available workbook-level
            # signal. When identifiers are supplied, only repeated identifiers
            # are flagged; otherwise a same-N heuristic is used conservatively.
            repeated_groups: list[list[Any]] = []
            if "control_id" in sub and sub["control_id"].notna().any():
                known = sub[sub["control_id"].notna()]
                for _, control_rows in known.groupby("control_id", dropna=False).groups.items():
                    control_ids = list(control_rows)
                    if len(control_ids) > 1:
                        repeated_groups.append(control_ids)
                unknown_ids = list(sub.index[sub["control_id"].isna()])
                if len(unknown_ids) > 1 and "n_c" in sub:
                    unknown = sub.loc[unknown_ids]
                    control_sizes = unknown["n_c"].dropna()
                    if len(control_sizes) == len(unknown) and control_sizes.nunique() == 1:
                        repeated_groups.append(unknown_ids)
            else:
                control_sizes = sub["n_c"].dropna() if "n_c" in sub else pd.Series(dtype=float)
                if len(sub) > 1 and len(control_sizes) == len(sub) and control_sizes.nunique() == 1:
                    repeated_groups.append(ids)
            for control_ids in repeated_groups:
                work.loc[control_ids, "shared_control_comparison_count"] = len(control_ids)
                work.loc[control_ids, "possible_shared_control"] = True

    sensitivity_rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        rid = row["record_id"]
        n_i, n_c = _valid_n(row)
        direction = row.get("effect_direction_multiplier")
        if direction is None or is_missing(direction):
            direction = 1
            direction_confirmed = False
        else:
            direction = int(direction)
            direction_confirmed = True

        approved = bool(record_approvals.get(rid, row.get("record_review_approved", False)))
        design_ok = str(row.get("study_design_status", "")).startswith("Parallel independent groups")
        if not design_ok and bool(global_decisions.get("parallel_independent_groups_confirmed")):
            design_ok = True
        conflict = bool(row.get("has_field_conflict", False))
        if n_i is None or n_c is None:
            raw = {"effect_estimate": None, "variance": None, "se": None, "estimate_source": None,
                   "variance_method": None, "status": "Not calculable", "candidate_reason": "Valid group sample sizes are required.",
                   "reported_effect_comparison": None, "approximation_note": None}
        elif estimand in {"log_rr", "log_or", "risk_difference"}:
            raw = _binary_effect(row, estimand, correction=zero_cell_correction, direction=direction)
        else:
            raw = _raw_continuous(row, estimand, direction, n_i, n_c)

        direct_var, direct_var_method = _get_direct_variance(row, estimand)
        if direct_var is not None and raw.get("variance") is None:
            raw["variance"] = direct_var
            raw["se"] = math.sqrt(direct_var)
            raw["variance_method"] = direct_var_method
            if raw.get("effect_estimate") is not None:
                raw["status"] = "Effect and direct compatible variance available"

        reported = _f(row, "reported_effect")
        reported_compatible = _estimand_matches(row.get("reported_effect_method"), estimand)
        reported_md = _f(row, "reported_md_estimate")
        if raw.get("effect_estimate") is None and estimand == "raw_posttest_md" and reported_md is not None and _estimand_matches(row.get("reported_effect_method"), estimand):
            raw["effect_estimate"] = reported_md * direction
            raw["estimate_source"] = "Researcher-reported raw MD retained; MD CI method explicitly confirmed"
            raw["status"] = "Reported raw MD available; variance route pending" if raw.get("variance") is None else "Reported raw MD and compatible variance available"
        if raw.get("effect_estimate") is None and reported is not None and reported_compatible:
            raw["effect_estimate"] = reported * direction
            raw["estimate_source"] = "Researcher-reported effect retained; reported method explicitly matches selected estimand"
            raw["status"] = "Reported effect available; variance route pending" if raw.get("variance") is None else "Reported effect and compatible variance available"
            if raw.get("variance") is None and n_i is not None and n_c is not None and estimand in {"posttest_smd"}:
                df = n_i + n_c - 2
                j = hedges_j(df)
                d = reported / j if j else reported
                raw["variance"] = (j**2) * ((1/n_i)+(1/n_c)+(d**2/(2*df)))
                raw["se"] = math.sqrt(raw["variance"])
                raw["variance_method"] = "Hedges g independent-group variance approximation from confirmed reported posttest SMD and group Ns"
                raw["status"] = "Reported effect and variance calculated"

        # Compare reported estimate only when it is known to be the same estimand.
        if raw.get("effect_estimate") is not None and reported is not None and reported_compatible:
            diff = raw["effect_estimate"] - (reported * direction)
            raw["reported_effect_comparison"] = {
                "reported_oriented": reported * direction,
                "calculated_minus_reported": diff,
                "absolute_difference": abs(diff),
                "interpretation": "Close agreement (absolute difference <= 0.10)" if abs(diff) <= 0.10 else "Material difference (>0.10); inspect extraction and calculation inputs",
            }
        elif reported is not None and not reported_compatible:
            raw["reported_effect_comparison"] = {
                "reported_value": reported,
                "interpretation": "Reported effect not used because its estimand/method was not confirmed to match the selected analysis.",
            }
        if estimand == "raw_posttest_md" and reported_md is not None and not _estimand_matches(row.get("reported_effect_method"), estimand):
            raw["reported_effect_comparison"] = {
                "reported_md_estimate": reported_md,
                "interpretation": "Reported MD and CI text retained, but not used until the researcher confirms it is an unadjusted posttest MD for this estimand.",
            }

        # Shared-control policy: never split true sample size for point estimate
        # or Hedges' correction. Apply adjustment only to the variance stage.
        shared_n = int(row.get("shared_control_comparison_count", 1) or 1)
        if shared_n > 1 and shared_control_policy == "block":
            raw["variance"] = None
            raw["se"] = None
            raw["variance_method"] = None
            raw["status"] = "Variance blocked — possible shared control requires a dependence decision"
        elif shared_n > 1 and shared_control_policy == "split_control_variance" and raw.get("variance") is not None and n_i and n_c:
            # Re-run variance with N_control/m only. For estimates already calculated,
            # use the SMD variance approximation; for PPC2, re-evaluate its control
            # component with actual correlations if available.
            n_c_eff = n_c / shared_n
            if estimand in {"posttest_smd", "change_smd"} and raw.get("effect_estimate") is not None:
                df = n_i + n_c - 2
                j = hedges_j(df)
                d = raw["effect_estimate"] / (j * direction) if j else raw["effect_estimate"]
                v = j**2 * (1/n_i + 1/n_c_eff + d**2/(2*max(n_i+n_c_eff-2, 1)))
                raw["variance"] = v
                raw["se"] = math.sqrt(v)
                raw["variance_method"] += "; shared-control sample-size split applied for variance weights only"
            elif estimand == "morris_ppc2" and raw.get("variance") is not None:
                r_i = _f(row, "r_prepost_i")
                r_c = _f(row, "r_prepost_c")
                if r_i is not None and r_c is not None:
                    # Recalculate with true Ns/J for the point estimate; N_control/m
                    # is used only in the variance contribution.
                    recalc = _raw_continuous(row, estimand, direction, n_i, n_c, variance_n_c=n_c_eff)
                    raw["variance"] = recalc.get("variance")
                    raw["se"] = recalc.get("se")
                    raw["variance_method"] = (recalc.get("variance_method") or "") + "; shared-control sample-size split applied for variance weights only"
            elif estimand in {"raw_posttest_md", "raw_change_md"} and raw.get("variance") is not None:
                recalc = _raw_continuous(row, estimand, direction, n_i, n_c, variance_n_c=n_c_eff)
                raw["variance"] = recalc.get("variance")
                raw["se"] = recalc.get("se")
                raw["variance_method"] = (recalc.get("variance_method") or "") + "; shared-control sample-size split applied for variance weights only"

        unit_required = estimand in {"raw_posttest_md", "raw_change_md"}
        unit_confirmed = not unit_required or not is_missing(row.get("unit"))
        can_pool = (
            raw.get("effect_estimate") is not None and raw.get("variance") is not None and raw.get("variance") > 0
            and approved and design_ok and direction_confirmed and not conflict and unit_confirmed
            and (shared_n <= 1 or shared_control_policy in {"split_control_variance", "reviewer_confirmed_independent"})
        )
        blockers = []
        if not approved: blockers.append("individual record approval pending")
        if not design_ok: blockers.append("independent-group design not confirmed")
        if not direction_confirmed: blockers.append("outcome direction not confirmed")
        if conflict: blockers.append("source field conflict unresolved")
        if not unit_confirmed: blockers.append("unit not supplied; raw mean difference cannot be interpreted or pooled")
        if raw.get("effect_estimate") is None: blockers.append(raw.get("candidate_reason") or "effect estimate unavailable")
        if raw.get("variance") is None: blockers.append("compatible sampling variance unavailable")
        if shared_n > 1 and shared_control_policy == "block": blockers.append("possible shared control requires dependence handling")

        row_result = {
            "record_id": rid, "study_id": row.get("study_id"), "outcome": row.get("outcome"),
            "duration": row.get("duration"), "intervention_arm": row.get("intervention_arm"),
            "estimand": estimand, "estimand_label": ESTIMANDS[estimand],
            "effect_estimate": raw.get("effect_estimate"), "sampling_variance": raw.get("variance"),
            "standard_error": raw.get("se"), "estimate_source": raw.get("estimate_source"),
            "variance_method": raw.get("variance_method"), "calculation_status": raw.get("status"),
            "candidate_reason": raw.get("candidate_reason"),
            "reported_effect": reported, "reported_effect_method": row.get("reported_effect_method"),
            "reported_effect_comparison": str(raw.get("reported_effect_comparison")) if raw.get("reported_effect_comparison") is not None else None,
            "direction_confirmed": direction_confirmed, "record_review_approved": approved,
            "study_design_confirmed": design_ok, "possible_shared_control": shared_n > 1,
            "shared_control_comparison_count": shared_n, "shared_control_policy": shared_control_policy,
            "include_in_primary_pool": bool(can_pool), "pooling_blockers": "; ".join(blockers),
            "approximation_note": raw.get("approximation_note"),
        }
        out_rows.append(row_result)

        # Separate sensitivity analysis: create a full scenario dataset. Source
        # correlations are retained; only missing group values are varied.
        if estimand == "morris_ppc2" and correlation_sensitivity and raw.get("effect_estimate") is not None:
            actual_i, actual_c = _f(row, "r_prepost_i"), _f(row, "r_prepost_c")
            for r in correlation_sensitivity:
                if not (-1 <= float(r) <= 1):
                    continue
                if n_i is None or n_c is None:
                    continue
                sens_r_i = actual_i if actual_i is not None else float(r)
                sens_r_c = actual_c if actual_c is not None else float(r)
                sens_n_c = (n_c / shared_n) if (shared_n > 1 and shared_control_policy == "split_control_variance") else n_c
                missing_i, missing_c = actual_i is None, actual_c is None
                sensitivity_label = None
                if missing_i and missing_c:
                    sensitivity_label = "researcher-approved sensitivity correlations for both missing groups"
                elif missing_i or missing_c:
                    sensitivity_label = "source correlation plus researcher-approved sensitivity correlation for the missing group"
                sens = _raw_continuous(
                    row, estimand, direction, n_i, n_c,
                    correlation_override=(sens_r_i, sens_r_c) if (missing_i or missing_c) else None,
                    variance_n_c=sens_n_c,
                    correlation_method_label=sensitivity_label,
                )
                # If a compatible direct variance was supplied, keep it rather
                # than replacing it with an assumed-correlation calculation.
                if direct_var is not None:
                    sens["variance"] = direct_var
                    sens["se"] = math.sqrt(direct_var)
                    sens["variance_method"] = direct_var_method
                sens_eligible = approved and design_ok and direction_confirmed and not conflict
                if shared_n > 1 and shared_control_policy == "block":
                    sens_eligible = False
                unit_required = estimand in {"raw_posttest_md", "raw_change_md"}
                if unit_required and is_missing(row.get("unit")):
                    sens_eligible = False
                if not approved: sens_block = "individual record approval pending"
                elif not design_ok: sens_block = "independent-group design not confirmed"
                elif not direction_confirmed: sens_block = "outcome direction not confirmed"
                elif conflict: sens_block = "source field conflict unresolved"
                elif shared_n > 1 and shared_control_policy == "block": sens_block = "possible shared control requires dependence handling"
                elif sens.get("variance") is None: sens_block = "compatible sensitivity variance unavailable"
                else: sens_block = ""
                assumption = []
                if actual_i is None: assumption.append(f"assumed r_I={float(r):g}")
                if actual_c is None: assumption.append(f"assumed r_C={float(r):g}")
                if not assumption:
                    assumption.append("no assumption; both source correlations retained")
                sensitivity_rows.append({
                    "record_id": rid, "study_id": row.get("study_id"), "outcome": row.get("outcome"),
                    "duration": row.get("duration"), "assumed_correlation_scenario": float(r),
                    "r_I_used": sens_r_i, "r_C_used": sens_r_c,
                    "effect_estimate": sens.get("effect_estimate"), "sampling_variance": sens.get("variance"),
                    "standard_error": sens.get("se"), "variance_method": sens.get("variance_method"),
                    "assumption_flag": "; ".join(assumption) + " — separate sensitivity scenario, not source data",
                    "sensitivity_pool_eligible": bool(sens_eligible and sens.get("variance") is not None),
                    "sensitivity_pooling_blocker": sens_block,
                    "shared_control_policy": shared_control_policy,
                })

    results = pd.DataFrame(out_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    return results, sensitivity
