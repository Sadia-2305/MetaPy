"""MetaPy statistical reconstruction engine (v1.0).

This module reconstructs sampling variances/SEs for continuous-outcome
Hedges' g records from statistics explicitly supplied by the researcher.
It does not access external study sources, silently impute correlations, or
substitute one estimand for another.

Supported routes (subject to the checks documented in each result):
1. Effect-specific variance, SE, or CI for Hedges' g.
2. Group-specific pre/post correlations supplied directly.
3. Correlations derived from SD(change), SE(change), paired t, exact paired p,
   or a CI for the within-group mean change, when sufficient values exist.
4. Morris PPC2 variance using group-specific/explicit common correlations.
5. Conventional independent-group SMD variance (LS3 approximation) only when
   the requested estimand is explicitly posttest_smd or change_smd.
6. Optional sensitivity scenarios only after researcher authorization.

The module is intentionally independent of the existing harmonization engine.
It accepts dictionaries (or a pandas DataFrame via reconstruct_table).
"""
from __future__ import annotations

import math
import re
from statistics import NormalDist
from typing import Any, Iterable, Mapping

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - fallback is used only for normal-limit cases
    student_t = None


VERSION = "1.0.0"


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower()) if value is not None else ""


def _num(value: Any) -> float | None:
    """Parse a finite numeric value. Inequalities such as '<0.05' are rejected."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip().replace(",", "")
    if not text or text.startswith(("<", ">", "≤", "≥")):
        return None
    match = re.fullmatch(r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?", text)
    if not match:
        return None
    try:
        number = float(text.replace("−", "-"))
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def _pick(record: Mapping[str, Any], *keys: str) -> Any:
    lookup = {_norm(k): v for k, v in record.items()}
    for key in keys:
        nk = _norm(key)
        if nk in lookup and lookup[nk] is not None:
            return lookup[nk]
    return None


def _v(record: Mapping[str, Any], *keys: str) -> float | None:
    return _num(_pick(record, *keys))


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}


def _j_correction(df: float) -> float:
    if df <= 1:
        raise ValueError("Hedges' correction requires df > 1.")
    return math.exp(math.lgamma(df / 2.0) - 0.5 * math.log(df / 2.0) - math.lgamma((df - 1.0) / 2.0))


def _t_critical(level: float, df: float | None) -> tuple[float, str]:
    if not 0 < level < 1:
        raise ValueError("ci_level must be between 0 and 1.")
    tail = (1.0 + level) / 2.0
    if df is not None and df > 0 and student_t is not None:
        return float(student_t.ppf(tail, df)), "Student's t critical value"
    return NormalDist().inv_cdf(tail), "normal critical value (df unavailable)"


def _effect_ci_to_se(low: float, high: float, level: float, df: float | None) -> tuple[float, str]:
    if high <= low:
        raise ValueError("Effect CI upper bound must exceed its lower bound.")
    critical, label = _t_critical(level, df)
    return (high - low) / (2.0 * critical), label


def _derive_group_correlation(record: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Recover one arm's pre/post correlation without imputing it."""
    suffixes = (("1", "i", "intervention", "t", "treatment") if arm == "1"
                else ("2", "c", "control", "comparator"))
    def gv(*stems: str) -> float | None:
        keys = []
        for stem in stems:
            for suffix in suffixes:
                keys.extend((f"{stem}_{suffix}", f"{stem}{suffix}", f"{suffix}_{stem}"))
        return _v(record, *keys)

    n = gv("n")
    sd_pre = gv("sd_pre", "pre_sd")
    sd_post = gv("sd_post", "post_sd")
    sd_change = gv("sd_change", "change_sd")
    # Recover a group's SD from its reported mean SE or mean CI when available.
    if n is not None and n > 1:
        df_mean = gv("mean_df", "df_mean") or (n - 1)
        for which, current in (("pre", sd_pre), ("post", sd_post)):
            mean_se = gv(f"{which}_se", f"se_{which}")
            lo = gv(f"{which}_ci_low", f"ci_{which}_low", f"mean_{which}_ci_low")
            hi = gv(f"{which}_ci_high", f"ci_{which}_high", f"mean_{which}_ci_high")
            if current is None and mean_se is not None and mean_se > 0:
                derived_sd = mean_se * math.sqrt(n)
                if which == "pre":
                    sd_pre = derived_sd
                else:
                    sd_post = derived_sd
            elif current is None and lo is not None and hi is not None and hi > lo:
                level_mean = gv(f"{which}_ci_level", f"ci_{which}_level") or 0.95
                crit, _ = _t_critical(level_mean, df_mean)
                derived_se = (hi - lo) / (2.0 * crit)
                derived_sd = derived_se * math.sqrt(n)
                if which == "pre":
                    sd_pre = derived_sd
                else:
                    sd_post = derived_sd
    se_change = gv("se_change", "change_se")
    mean_change = gv("mean_change", "change_mean")
    paired_t = gv("paired_t", "t_paired", "within_group_t")
    paired_p = gv("paired_p", "p_paired", "within_group_p")
    ci_low = gv("change_ci_low", "ci_change_low", "mean_change_ci_low")
    ci_high = gv("change_ci_high", "ci_change_high", "mean_change_ci_high")
    ci_level = gv("change_ci_level", "ci_change_level") or _v(record, "change_ci_level") or 0.95
    paired_df = gv("paired_df", "df_paired") or (n - 1 if n is not None else None)

    # Group-specific explicitly supplied correlation has priority.
    direct_keys = (f"r_{arm}", f"r{arm}", f"correlation_{arm}",
                   "r_intervention" if arm == "1" else "r_control",
                   "prepost_correlation_intervention" if arm == "1" else "prepost_correlation_control")
    direct = _v(record, *direct_keys)
    if direct is not None:
        if not -1 <= direct <= 1:
            return {"r": None, "method": "invalid_direct_correlation", "reason": f"Group {arm} correlation is outside [-1, 1]."}
        return {"r": direct, "method": "reported_group_correlation", "assumptions": []}

    provenance = []
    if sd_change is None and se_change is not None and n is not None and n > 0:
        sd_change = se_change * math.sqrt(n)
        provenance.append("SD(change) derived as SE(change) × sqrt(n)")

    if sd_change is None and paired_t is not None and mean_change is None:
        # Group means may be supplied separately; derive the signed mean change.
        mean_pre = gv("mean_pre", "pre_mean")
        mean_post = gv("mean_post", "post_mean")
        if mean_pre is not None and mean_post is not None:
            mean_change = mean_post - mean_pre

    if sd_change is None and paired_t is not None and mean_change is not None and n is not None and n > 0 and paired_t != 0:
        sd_change = abs(mean_change) * math.sqrt(n) / abs(paired_t)
        provenance.append("SD(change) derived from paired t, mean change and n")

    paired_p_sidedness = str(gv("paired_p_sidedness", "p_sidedness") or _pick(record, "paired_p_sidedness") or "").strip().lower()
    if sd_change is None and paired_p is not None and mean_change is not None and n is not None and n > 1 and 0 < paired_p < 1:
        if paired_p_sidedness in {"two-sided", "two sided", "two-tailed", "two tailed", "2-sided", "2-tailed"}:
            if paired_df is None or paired_df <= 0:
                paired_df = n - 1
            if student_t is not None:
                t_abs = float(student_t.isf(paired_p / 2.0, paired_df))
                if t_abs > 0:
                    sd_change = abs(mean_change) * math.sqrt(n) / t_abs
                    provenance.append("SD(change) derived from exact two-sided paired p, df, mean change and n")
        else:
            provenance.append("Exact paired p was not used because its one-/two-sided convention was not supplied")

    if sd_change is None and ci_low is not None and ci_high is not None and n is not None and n > 0:
        se, crit_label = _effect_ci_to_se(ci_low, ci_high, ci_level, paired_df)
        sd_change = se * math.sqrt(n)
        provenance.append(f"SD(change) derived from within-group change CI using {crit_label}")

    if sd_pre is None or sd_post is None or sd_pre <= 0 or sd_post <= 0 or sd_change is None or sd_change < 0:
        return {"r": None, "method": "not_reconstructable", "reason": "Need valid pre/post SDs plus SD(change), SE(change), paired t, exact paired p, or a within-group change CI.", "attempts": provenance}

    r = (sd_pre ** 2 + sd_post ** 2 - sd_change ** 2) / (2.0 * sd_pre * sd_post)
    if not -1.000001 <= r <= 1.000001:
        return {"r": None, "method": "incompatible_change_statistics", "reason": f"Derived correlation {r:.6g} falls outside [-1, 1]; supplied statistics are internally incompatible or rounded beyond tolerance.", "attempts": provenance}
    r = max(-1.0, min(1.0, r))
    provenance.append("Pre/post correlation derived from SD(pre), SD(post), and SD(change)")
    return {"r": r, "method": "derived_from_change_statistics", "assumptions": [], "provenance": provenance}


def _ls3_variance(g: float, n1: float, n2: float) -> float:
    """Large-sample independent-groups SMD variance (LS3/RevMan-style)."""
    if n1 <= 1 or n2 <= 1 or int(n1) != n1 or int(n2) != n2:
        raise ValueError("Both group sample sizes must be integers greater than 1.")
    denom = n1 + n2 - 3.94
    if denom <= 0:
        raise ValueError("Sample sizes are too small for this LS3 approximation.")
    return 1.0 / n1 + 1.0 / n2 + (g * g) / (2.0 * denom)


def _morris_ppc2_variance(g: float, n1: float, n2: float, r1: float, r2: float) -> tuple[float, dict[str, float]]:
    """Morris PPC2 Hedges-g variance using group-specific correlations.

    Implements the finite-sample form described in Morris (2008) and exposed
    by the Campbell Collaboration calculator. The correlation used in the
    sampling-variance expression is the sample-size-weighted group average.
    """
    if n1 <= 1 or n2 <= 1 or int(n1) != n1 or int(n2) != n2:
        raise ValueError("PPC2 requires integer group sample sizes greater than 1.")
    total = n1 + n2
    df = total - 2.0
    if df <= 2:
        raise ValueError("PPC2 finite-sample variance requires n1+n2 > 4.")
    if not (-1 <= r1 <= 1 and -1 <= r2 <= 1):
        raise ValueError("Both group correlations must lie in [-1, 1].")
    r_bar = (n1 * r1 + n2 * r2) / total
    j = _j_correction(df)
    d = g / j
    a = total / (n1 * n2)
    one_minus_r = 1.0 - r_bar
    if one_minus_r <= 0:
        raise ValueError("Weighted pre/post correlation is 1.0; PPC2 variance cannot be reconstructed from this expression.")
    v_d = (2.0 * one_minus_r * a * ((total - 2.0) / (total - 4.0))
           * (1.0 + (d * d) / (2.0 * one_minus_r * a)) - (d * d) / (j * j))
    v_g = v_d * (j * j)
    if not math.isfinite(v_g) or v_g <= 0:
        raise ValueError(f"PPC2 variance was non-positive/invalid ({v_g!r}); check estimand and supplied statistics.")
    return v_g, {"r_weighted": r_bar, "j_correction": j, "cohens_d_equivalent": d, "df": df}


def _derive_md_ci_route(record: Mapping[str, Any], g: float | None, estimand: str,
                        n1: float | None, n2: float | None) -> dict[str, Any] | None:
    """Use an MD CI only when its posttest, unadjusted meaning is explicit."""
    md = _v(record, "md", "mean_difference", "between_group_md")
    low = _v(record, "md_ci_low", "mean_difference_ci_low", "between_group_md_ci_low")
    high = _v(record, "md_ci_high", "mean_difference_ci_high", "between_group_md_ci_high")
    # Parse a combined raw-scale field such as "-35.97 (-80.98 to 9.04)".
    if md is None or low is None or high is None:
        combined = _pick(record, "md_95_ci", "md_ci", "md_95_percent_ci", "md_(95%_ci)")
        if combined is not None:
            numbers = re.findall(r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?", str(combined).replace(",", ""))
            if len(numbers) >= 3:
                parsed = [_num(x) for x in numbers[:3]]
                if all(x is not None for x in parsed):
                    md, low, high = parsed
    md_estimand = str(_pick(record, "md_estimand", "mean_difference_estimand") or "").strip().lower()
    unadjusted = _truth(_pick(record, "md_unadjusted", "unadjusted_md_confirmed"))
    if md is None:
        return None
    # An explicitly labelled raw MD SE or compatible t/exact p can be used
    # without requiring an MD confidence interval.
    se_md = _v(record, "md_se", "mean_difference_se", "between_group_md_se")
    if se_md is None:
        md_t = _v(record, "md_t", "mean_difference_t", "between_group_t")
        md_p = _v(record, "md_p", "mean_difference_p", "between_group_p")
        md_df_tmp = _v(record, "md_df", "mean_difference_df") or (n1 + n2 - 2 if n1 and n2 else None)
        md_test = _norm(_pick(record, "md_test_type", "mean_difference_test_type") or "")
        if md_t is not None and md_t != 0 and md_test in {"independentt", "independentttest", "twosamplet", "studentttest", "equalvarianceindependentt"}:
            se_md = abs(md / md_t)
        elif md_p is not None and 0 < md_p < 1 and md_df_tmp is not None and student_t is not None and md_test in {"independentt", "independentttest", "twosamplet", "studentttest", "equalvarianceindependentt"} and _norm(_pick(record, "md_p_sidedness", "p_sidedness") or "") in {"twosided", "twotailed", "2sided", "2tailed"}:
            t_abs = float(student_t.isf(md_p / 2.0, md_df_tmp))
            if t_abs > 0:
                se_md = abs(md) / t_abs
        elif md_test in {"onewaybetweenanova", "betweengroupanova", "independentonewayanova", "equalvarianceonewayanova"}:
            md_f = _v(record, "md_f", "between_group_f", "f_statistic")
            f_df1 = _v(record, "md_f_df1", "f_df1", "numerator_df")
            if md_f is not None and md_f > 0 and f_df1 == 1:
                se_md = abs(md) / math.sqrt(md_f)
    if low is None or high is None:
        if se_md is None:
            return None
    if estimand != "posttest_smd" or md_estimand not in {"posttest", "posttest_mean_difference", "post-treatment"} or not unadjusted:
        return {"failed": "A raw MD CI is present, but it is not explicitly confirmed as an unadjusted posttest MD for the same posttest SMD estimand; it cannot be converted to Hedges' g uncertainty."}
    if n1 is None or n2 is None or n1 <= 1 or n2 <= 1:
        return {"failed": "MD-CI reconstruction requires valid group sample sizes."}
    ci_level = _v(record, "md_ci_level", "ci_level") or 0.95
    df = _v(record, "md_df", "mean_difference_df") or (n1 + n2 - 2)
    critical_label = "reported MD SE or exact MD test-statistic conversion"
    if se_md is None:
        se_md, critical_label = _effect_ci_to_se(low, high, ci_level, df)
    pooled_sd = se_md / math.sqrt(1.0 / n1 + 1.0 / n2)
    if pooled_sd <= 0:
        return {"failed": "MD CI implied a non-positive pooled SD."}
    j = _j_correction(n1 + n2 - 2)
    implied_g = j * md / pooled_sd
    if g is not None and not math.isclose(implied_g, g, rel_tol=0.15, abs_tol=0.15):
        return {"failed": f"MD-CI implied g ({implied_g:.4f}) does not agree with reported g ({g:.4f}); likely different estimator, adjustment, or extraction. Route not used."}
    selected_g = implied_g if g is None else g
    variance = _ls3_variance(selected_g, n1, n2)
    return {"variance": variance, "effect_estimate": selected_g,
            "method": "unadjusted_posttest_MD_CI_to_pooled_SD_then_LS3",
            "route": "derived_approximation",
            "assumptions": ["MD CI is explicitly confirmed as unadjusted posttest MD for the same posttest SMD estimand.",
                            "Equal-variance independent groups; pooled SD recovered from the MD standard error.",
                            "Sampling variance uses the LS3 independent-groups SMD approximation."],
            "details": {"se_md": se_md, "pooled_sd_implied": pooled_sd,
                        "critical_value_method": critical_label, "implied_g": implied_g}}


def reconstruct_uncertainty(record: Mapping[str, Any], *, estimand: str | None = None,
                            design: str | None = None) -> dict[str, Any]:
    """Try every supported, compatible uncertainty-reconstruction route.

    Canonical input fields (aliases are accepted for most):
      effect_estimate / hedges_g / g, effect_variance / g_variance,
      effect_se / g_se, effect_ci_low / g_ci_low, effect_ci_high / g_ci_high,
      n1, n2, estimand; correlation1/r1, correlation2/r2, common_correlation;
      sd_pre_1, sd_post_1, sd_change_1 (and group 2); se_change_1;
      mean_change_1, paired_t_1, paired_p_1, paired_df_1;
      change_ci_low_1/change_ci_high_1; md, md_ci_low, md_ci_high,
      md_estimand, md_unadjusted.

    The parallel independent-group design must be explicitly supplied through
    the design argument or record field for PPC2/independent-SMD routes.

    Returns a transparent dictionary. No route is selected from an incompatible
    estimand. Correlation imputation requires explicit researcher approval.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping/dictionary.")

    selected_estimand = (estimand or _pick(record, "estimand") or "").strip().lower()
    selected_design = (design or _pick(record, "design", "study_design") or "").strip().lower()
    design_key = _norm(selected_design)
    parallel_design_confirmed = design_key in {
        "parallelindependent", "parallelgroup", "parallelrct", "randomizedparallel",
        "parallelrandomizedcontrolledtrial", "parallelindependentgroups"
    }
    shared_control = _truth(_pick(record, "shared_control", "shared_comparator", "common_control"))
    effect = _v(record, "effect_estimate", "hedges_g", "hedgesg", "g")
    n1 = _v(record, "n1", "n_i", "n_intervention", "n_treatment")
    n2 = _v(record, "n2", "n_c", "n_control", "n_comparator")
    provenance = {"record_id": _pick(record, "record_id", "comparison_id", "study_id"),
                  "source_sheet": _pick(record, "source_sheet", "sheet"),
                  "source_row": _pick(record, "source_row", "excel_row"),
                  "shared_control": shared_control}
    attempts: list[dict[str, Any]] = []
    blockers: list[str] = []

    def resolved(variance: float, method: str, route: str, assumptions=None, details=None,
                 effect_override=None) -> dict[str, Any]:
        estimate = effect if effect_override is None else effect_override
        if estimate is None:
            return {"status": "needs_researcher_input", "effect_estimate": None,
                    "variance": None, "se": None, "method": method, "route": route,
                    "assumptions": assumptions or [], "provenance": provenance,
                    "attempted_routes": attempts, "blocking_issues": ["A compatible effect estimate is unavailable."],
                    "details": details or {}}
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("A reconstructed variance must be finite and positive.")
        se = math.sqrt(variance)
        level = _v(record, "ci_level", "effect_ci_level") or 0.95
        ci_df = _v(record, "ci_df", "effect_ci_df") or ((n1 + n2 - 2) if n1 and n2 else None)
        critical, critical_label = _t_critical(level, ci_df)
        return {"status": "resolved", "effect_estimate": estimate,
                "variance": variance, "se": se,
                "ci": {"level": level, "lower": estimate - critical * se,
                       "upper": estimate + critical * se, "critical_value_method": critical_label},
                "method": method, "route": route, "assumptions": assumptions or [],
                "provenance": provenance, "attempted_routes": attempts,
                "blocking_issues": [],
                "dependence_note": ("Shared-control/shared-comparator covariance across multiple comparisons is not calculated here; handle it in the multi-arm pooling stage." if shared_control else None),
                "details": details or {}}

    # 1) Direct variance/SE/effect-specific CI have first priority.
    direct_var = _v(record, "effect_variance", "hedges_g_variance", "g_variance", "variance_g", "vi")
    if direct_var is not None and direct_var > 0:
        attempts.append({"route": "direct_variance", "status": "accepted"})
        return resolved(direct_var, "reported_effect_specific_variance", "direct_reported")
    if direct_var is not None:
        attempts.append({"route": "direct_variance", "status": "rejected", "reason": "Variance must be positive."})
        blockers.append("Reported effect-specific variance was non-positive and was not used.")
    else:
        attempts.append({"route": "direct_variance", "status": "not_available"})

    direct_se = _v(record, "effect_se", "hedges_g_se", "g_se", "se_g")
    if direct_se is not None:
        if direct_se > 0:
            attempts.append({"route": "direct_effect_se", "status": "accepted"})
            return resolved(direct_se ** 2, "reported_effect_specific_SE_squared", "direct_reported")
        blockers.append("Reported effect-specific SE must be positive.")
    attempts.append({"route": "direct_effect_se", "status": "not_available" if direct_se is None else "invalid"})

    ci_low = _v(record, "effect_ci_low", "hedges_g_ci_low", "g_ci_low")
    ci_high = _v(record, "effect_ci_high", "hedges_g_ci_high", "g_ci_high")
    if ci_low is not None and ci_high is not None and _truth(_pick(record, "ci_is_effect_specific", "effect_ci_confirmed")):
        try:
            level = _v(record, "effect_ci_level", "ci_level") or 0.95
            ci_df = _v(record, "effect_ci_df", "ci_df") or ((n1 + n2 - 2) if n1 and n2 else None)
            se, critical_label = _effect_ci_to_se(ci_low, ci_high, level, ci_df)
            if effect is not None and not (ci_low < effect < ci_high):
                blockers.append("Effect-specific CI does not contain the reported effect estimate; CI route rejected.")
                attempts.append({"route": "effect_specific_CI", "status": "rejected", "reason": "CI does not contain effect estimate"})
            else:
                attempts.append({"route": "effect_specific_CI", "status": "accepted"})
                return resolved(se ** 2, "effect_specific_CI_to_SE", "exact_derivation",
                                details={"critical_value_method": critical_label})
        except Exception as exc:
            blockers.append(f"Effect CI could not be used: {exc}")
    else:
        attempts.append({"route": "effect_specific_CI", "status": "not_available_or_not_confirmed_effect_specific"})

    # Recover group correlations from all available within-group statistics.
    group1 = _derive_group_correlation(record, "1")
    group2 = _derive_group_correlation(record, "2")
    for arm, info in (("1", group1), ("2", group2)):
        attempts.append({"route": f"group_{arm}_prepost_correlation", "status": "derived" if info.get("r") is not None else "unavailable", "method": info.get("method"), "reason": info.get("reason"), "provenance": info.get("provenance") or info.get("attempts")})

    # Explicit common correlation is an accepted supplied value, not an assumed default.
    common_r = _v(record, "common_correlation", "r_common", "prepost_correlation", "r_prepost")
    if common_r is not None and not -1 <= common_r <= 1:
        blockers.append("Explicit common pre/post correlation is outside [-1, 1].")
        common_r = None

    # 2) Morris PPC2: must have PPC2 estimand; never borrow independent SMD variance.
    if selected_estimand in {"morris_ppc2", "ppc2", "morris"}:
        if not parallel_design_confirmed:
            attempts.append({"route": "morris_ppc2", "status": "blocked", "reason": "Parallel independent-group design has not been explicitly confirmed."})
            blockers.append("Morris PPC2 variance requires an explicitly classified parallel independent-group design.")
        elif effect is None or n1 is None or n2 is None:
            attempts.append({"route": "morris_ppc2", "status": "unavailable", "reason": "Requires Hedges' g and both group sample sizes."})
        else:
            r1, r2 = group1.get("r"), group2.get("r")
            correlation_source = "derived_or_reported_group_correlations"
            if r1 is None or r2 is None:
                if common_r is not None:
                    r1 = r2 = common_r
                    correlation_source = "explicit_researcher_supplied_common_correlation"
                else:
                    r1 = r2 = None
            if r1 is not None and r2 is not None:
                try:
                    variance, details = _morris_ppc2_variance(effect, n1, n2, r1, r2)
                    details.update({"r1": r1, "r2": r2, "correlation_source": correlation_source})
                    attempts.append({"route": "morris_ppc2", "status": "accepted"})
                    assumptions = ["Morris PPC2 estimand retained; no posttest/change-score estimand substitution.",
                                   "Group-specific pre/post correlations were reported or reconstructed from within-group statistics." if correlation_source.startswith("derived") else "One common pre/post correlation was explicitly supplied by the researcher for both groups.",
                                   "Morris PPC2 finite-sample variance expression used; Hedges correction applied."]
                    if shared_control:
                        assumptions.append("Marginal variance only; covariance with other comparisons sharing this control is not calculated here.")
                    return resolved(variance, "Morris_PPC2_Hedges_g_variance", "exact_derivation" if correlation_source.startswith("derived") else "researcher_supplied_input", assumptions, details)
                except Exception as exc:
                    blockers.append(f"Morris PPC2 variance route failed: {exc}")
                    attempts.append({"route": "morris_ppc2", "status": "rejected", "reason": str(exc)})
            else:
                attempts.append({"route": "morris_ppc2", "status": "pending", "reason": "Both group-specific correlations, or an explicitly supplied common correlation, are required."})

            # Approved sensitivity-only scenarios. No single value is chosen by default.
            approved = _truth(_pick(record, "correlation_imputation_approved", "imputation_approved", "researcher_approved_imputation"))
            if approved:
                raw_values = _pick(record, "imputation_values", "correlation_sensitivity_values")
                if raw_values is None:
                    values = [0.30, 0.50, 0.70]
                elif isinstance(raw_values, (list, tuple)):
                    values = [x for x in (_num(z) for z in raw_values) if x is not None]
                else:
                    values = []
                values = [x for x in values if -1 <= x <= 1]
                sensitivity = []
                for r in values:
                    try:
                        v, d = _morris_ppc2_variance(effect, n1, n2, r, r)
                        sensitivity.append({"assumed_common_r": r, "variance": v, "se": math.sqrt(v), "details": d})
                    except Exception as exc:
                        sensitivity.append({"assumed_common_r": r, "status": "invalid", "reason": str(exc)})
                selected_r = _v(record, "selected_imputed_correlation", "researcher_selected_correlation")
                if selected_r is not None and selected_r in values:
                    v, d = _morris_ppc2_variance(effect, n1, n2, selected_r, selected_r)
                    attempts.append({"route": "researcher_approved_correlation", "status": "selected_by_researcher"})
                    return resolved(v, "Morris_PPC2_variance_with_researcher_selected_r", "researcher_assumption",
                                    [f"Researcher explicitly approved and selected common pre/post correlation r={selected_r}.",
                                     "Sensitivity scenarios are included in the returned details."],
                                    {"selected_r": selected_r, "sensitivity": sensitivity, **d})
                return {"status": "needs_researcher_input", "effect_estimate": effect,
                        "variance": None, "se": None, "method": "Morris_PPC2_correlation_sensitivity_only",
                        "route": "researcher_assumption_pending_selection",
                        "assumptions": ["Researcher approved correlation sensitivity analysis; no single correlation has been selected."],
                        "provenance": provenance, "attempted_routes": attempts,
                        "blocking_issues": ["Select one correlation only after reviewing the sensitivity scenarios."],
                        "sensitivity": sensitivity}

            # No correlation imputation without approval.
            blockers.append("Morris PPC2 variance remains unresolved: neither both group correlations nor an explicitly supplied common correlation are available. Correlation imputation has not been authorized.")

    # 3) Conditional raw MD CI route: strict estimand and adjustment checks.
    md_route = _derive_md_ci_route(record, effect, selected_estimand, n1, n2)
    if md_route is not None and not parallel_design_confirmed:
        md_route = {"failed": "Raw MD uncertainty cannot be converted to an independent-group SMD until a parallel independent-group design is explicitly confirmed."}
    if md_route is not None:
        if "variance" in md_route:
            attempts.append({"route": "unadjusted_posttest_MD_CI", "status": "accepted"})
            return resolved(md_route["variance"], md_route["method"], md_route["route"], md_route["assumptions"], md_route["details"], md_route.get("effect_estimate"))
        attempts.append({"route": "unadjusted_posttest_MD_CI", "status": "rejected", "reason": md_route["failed"]})
        blockers.append(md_route["failed"])
    else:
        attempts.append({"route": "unadjusted_posttest_MD_CI", "status": "not_available"})


    # 4) Conventional independent SMD variance only for compatible SMD estimands.
    if selected_estimand in {"posttest_smd", "change_smd", "independent_smd"} and effect is not None and n1 is not None and n2 is not None and parallel_design_confirmed:
        try:
            variance = _ls3_variance(effect, n1, n2)
            attempts.append({"route": "independent_group_LS3", "status": "accepted", "estimand": selected_estimand})
            assumptions = [f"Estimand explicitly specified as {selected_estimand}.",
                           "Parallel independent-group design explicitly confirmed.",
                           "Large-sample LS3 variance approximation used; not a Morris PPC2 variance."]
            if shared_control:
                assumptions.append("This is a marginal variance for one contrast only. Shared-control covariance across multiple comparisons is not handled by this module and must be handled during multi-arm pooling.")
            return resolved(variance, "LS3_independent_group_SMD_variance_approximation", "derived_approximation",
                            assumptions,
                            {"formula": "1/n1 + 1/n2 + g^2/[2*(n1+n2-3.94)]"})
        except Exception as exc:
            attempts.append({"route": "independent_group_LS3", "status": "rejected", "reason": str(exc)})
            blockers.append(f"Independent-group LS3 route failed: {exc}")
    elif selected_estimand in {"posttest_smd", "change_smd", "independent_smd"} and not parallel_design_confirmed:
        attempts.append({"route": "independent_group_LS3", "status": "blocked", "reason": "Parallel independent-group design has not been explicitly confirmed."})
        blockers.append("Independent-group SMD variance requires an explicitly classified parallel independent-group design.")
    elif selected_estimand not in {"morris_ppc2", "ppc2", "morris"}:
        attempts.append({"route": "independent_group_LS3", "status": "not_used", "reason": "Estimand is not explicitly a compatible independent-group SMD."})

    # If a non-PPC2 route had enough within-group data to derive correlation,
    # report it but do not treat that correlation itself as a g variance.
    if not blockers:
        blockers.append("No supported, estimand-compatible route could reconstruct the sampling variance from the supplied statistics.")
    return {"status": "needs_researcher_input", "effect_estimate": effect,
            "variance": None, "se": None, "method": None, "route": None,
            "assumptions": [], "provenance": provenance,
            "attempted_routes": attempts, "blocking_issues": blockers,
            "correlation_diagnostics": {"group_1": group1, "group_2": group2,
                                        "explicit_common_correlation": common_r}}


def reconstruct_table(data: Any, *, estimand: str | None = None,
                      design: str | None = None) -> list[dict[str, Any]]:
    """Run reconstruction on a list of dictionaries or pandas DataFrame."""
    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        records = data.to_dict(orient="records")
    else:
        records = list(data)
    results = []
    for idx, record in enumerate(records, start=1):
        result = reconstruct_uncertainty(record, estimand=estimand, design=design)
        result["input_row_number"] = idx
        results.append(result)
    return results


def summarize_results(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Compact resolved/pending count for notebook reporting."""
    resolved_count = pending_count = 0
    for item in results:
        if item.get("status") == "resolved":
            resolved_count += 1
        else:
            pending_count += 1
    return {"total": resolved_count + pending_count,
            "resolved": resolved_count, "needs_researcher_input": pending_count}
