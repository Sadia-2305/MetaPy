"""MetaPy 3.2 data preparation, harmonisation and researcher review engine.

The module converts already-screened workbook records into a separate,
analysis-oriented working dataset. It does NOT overwrite the uploaded workbook,
calculate final effect sizes, calculate sampling variances, or pool studies.

Only explicitly approved researcher decisions are applied. Ambiguous or
unsupported actions are retained as findings/review items rather than guessed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from .dependency_engine import reconcile_dependency_reviews, dependency_impact_table

VERSION = "3.2.2"

REVIEW_STATUSES = {"Pending", "Approved", "Rejected", "Deferred"}

DIRECTION_OPTIONS = [
    {"value": "higher_more", "label": "Higher values mean more of the measured outcome", "explanation": "For example, a higher anxiety-scale score means more anxiety symptoms, or a higher biomarker concentration means a higher measured concentration."},
    {"value": "higher_less", "label": "Higher values mean less of the measured outcome", "explanation": "Use this only when the scale or measurement is coded so higher values indicate less of the construct."},
    {"value": "context_dependent", "label": "Meaning depends on the instrument, timing or clinical context", "explanation": "This is important for measures such as cortisol, where the interpretation can depend on sampling time and clinical context."},
    {"value": "uncertain", "label": "Not sure — keep the direction unresolved", "explanation": "MetaPy will preserve the record but will not claim that positive or negative effects have a consistent beneficial meaning."},
]

DESIGN_OPTIONS = [
    {"value": "confirm_parallel", "label": "All relevant studies use separate parallel treatment and control groups", "explanation": "Confirms the project-level independent parallel-group design setting for these records."},
    {"value": "mixed_designs", "label": "Some studies use crossover, paired, cluster or another design", "explanation": "MetaPy will flag study-specific design classification as required before selecting statistical formulas."},
    {"value": "design_uncertain", "label": "I am not sure — keep study design review pending", "explanation": "Do not apply independent-group formulas until each study's design is clear."},
]

UNIT_OPTIONS = [
    {"value": "same_within_study", "label": "Both groups use the same unit within each study", "explanation": "Different studies may still use different units when a compatible standardised effect is used, but the groups being compared within one study must be on a compatible scale."},
    {"value": "units_need_harmonisation", "label": "Some studies use different units between groups or time points", "explanation": "The affected records need a documented unit conversion before an effect estimate is calculated."},
    {"value": "unit_not_reported", "label": "Units are not reported or I cannot confirm them", "explanation": "Keep the affected records flagged. MetaPy cannot invent a unit or conversion factor."},
]

METHOD_COMPATIBILITY_OPTIONS = [
    {"value": "keep_ancova_wait", "label": "Keep ANCOVA as primary and obtain adjusted effects with SEs or CIs", "explanation": "This respects the currently approved ANCOVA choice. The current workbook does not yet show enough clearly identified ANCOVA-adjusted estimates and uncertainty for the records."},
    {"value": "request_morris_reconsideration", "label": "Reconsider Morris PPC2 as primary (requires a new approval)", "explanation": "The workbook has pre/post data for many records. This option only sends the primary-method decision back for approval; it does not silently change or approve the method."},
    {"value": "keep_ppc2_alternative", "label": "Keep PPC2 data as an alternative, but retain ANCOVA as primary", "explanation": "PPC2 candidates remain separate. They are not substituted for ANCOVA-adjusted estimates."},
]

MISSING_PPC2_OPTIONS = [
    {"value": "retrieve_baseline_sd", "label": "Retrieve the missing baseline SDs from the original report or supplement", "explanation": "This is the preferred exact-data route when the study report contains the missing baseline standard deviations."},
    {"value": "derive_from_valid_statistic", "label": "Check whether another reported statistic can validly provide the missing baseline SD", "explanation": "MetaPy will not use a post-treatment between-group MD confidence interval as a substitute for a missing baseline SD."},
    {"value": "hold_missing_sd", "label": "Keep the affected records pending", "explanation": "The records remain in the project but are not marked ready for the PPC2 point-estimate calculation."},
    {"value": "exclude_missing_sd", "label": "Exclude only these affected records from the selected primary analysis", "explanation": "The records remain preserved in the source and audit, but are marked out of the primary analysis. A protocol-based reason must be recorded."},
]

CORRECTION_OPTIONS = [
    {"value": "no_correction_needed", "label": "No extraction correction is currently needed", "explanation": "MetaPy will retain the extracted values unchanged. This can be approved only after the researcher has reviewed the flagged records."},
    {"value": "cell_corrections_needed", "label": "Specific cell-level corrections are needed", "explanation": "MetaPy will keep the preparation incomplete until the exact sheet, row, field, corrected value and source are supplied through the correction workflow."},
    {"value": "correction_not_sure", "label": "I need to review the source reports first", "explanation": "No data will be changed. The correction review remains pending."},
]


def _safe(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(_safe(value) or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _split_semicolon(value: Any) -> list[str]:
    return [x.strip() for x in _safe(value).split(";") if x.strip()]


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _safe(value).lower())


def _decision_map(queue: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if queue is None or queue.empty:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for _, row in queue.iterrows():
        result[_safe(row.get("Category"))] = {
            "choice": _safe(row.get("Your choice")),
            "status": _safe(row.get("Status")) or "Pending",
            "details": _json_obj(row.get("Decision details (structured)")),
            "exclusion_reason": _safe(row.get("Exclusion reason")),
            "decision_id": _safe(row.get("Decision ID")),
        }
    return result


def _option_lookup(options: list[dict[str, Any]], value: str) -> dict[str, Any] | None:
    return next((item for item in options if str(item.get("value")) == value), None)


def _classify_outcome(raw: Any) -> tuple[str, str]:
    key = _slug(raw)
    mapping = {
        "hama": ("Anxiety", "Validated anxiety questionnaire score"),
        "hadsa": ("Anxiety", "Validated anxiety questionnaire score"),
        "bai": ("Anxiety", "Validated anxiety questionnaire score"),
        "dassanxiety": ("Anxiety", "Validated anxiety questionnaire score"),
        "gad7": ("Anxiety", "Validated anxiety questionnaire score"),
        "stai": ("Anxiety", "Validated anxiety questionnaire score"),
        "crp": ("CRP", "Inflammatory biomarker concentration"),
        "hscrp": ("CRP", "Inflammatory biomarker concentration"),
        "il6": ("IL-6", "Inflammatory biomarker concentration"),
        "interleukin6": ("IL-6", "Inflammatory biomarker concentration"),
        "tnf": ("TNF-α", "Inflammatory biomarker concentration"),
        "tnfa": ("TNF-α", "Inflammatory biomarker concentration"),
        "tumornecrosisfactoralpha": ("TNF-α", "Inflammatory biomarker concentration"),
        "cortisol": ("Cortisol", "Hormone biomarker concentration"),
    }
    if key in mapping:
        return mapping[key]
    if "anxiety" in key:
        return "Anxiety", "Anxiety-related measure; instrument needs confirmation"
    if "cortisol" in key:
        return "Cortisol", "Hormone biomarker concentration"
    if "crp" in key:
        return "CRP", "Inflammatory biomarker concentration"
    if "interleukin6" in key or key == "il6":
        return "IL-6", "Inflammatory biomarker concentration"
    if "tnf" in key:
        return "TNF-α", "Inflammatory biomarker concentration"
    return (_safe(raw) or "Unclassified outcome", "Outcome type requires researcher review")


def _candidate_data(row: pd.Series) -> tuple[dict[str, Any], list[str]]:
    payload = _json_obj(row.get("candidate_effect_methods"))
    candidates = [c for c in payload.get("candidates", []) if isinstance(c, dict)]
    estimands = sorted({str(c.get("estimand")) for c in candidates if c.get("estimand")})
    return payload, estimands


def _candidate_for(payload: dict[str, Any], estimand: str) -> dict[str, Any] | None:
    for candidate in payload.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("estimand")) == estimand and bool(candidate.get("eligible_for_selection", False)):
            return candidate
    return None


def _source_has(provenance: str, token: str) -> bool:
    return f"{token.lower()}:" in provenance.lower()


def _reporting_format(row: pd.Series, provenance: str) -> tuple[str, list[str]]:
    formats: list[str] = []
    has_pre = _source_has(provenance, "pre_m1") and _source_has(provenance, "pre_m2")
    has_post = _source_has(provenance, "post_m1") and _source_has(provenance, "post_m2")
    has_baseline_sd = _source_has(provenance, "pre_s1") and _source_has(provenance, "pre_s2")
    has_reported_g = _number(row.get("reported_hedges_g")) is not None
    has_change = any(_source_has(provenance, x) for x in ("sd_change1", "se_change1", "paired_t1", "change_ci_low1"))
    has_adjusted = any(x in provenance.lower() for x in ("ancova", "adjusted_effect", "adjustedtreatment", "regressioncoefficient"))
    has_ci = "ci" in provenance.lower() or _source_has(provenance, "md_95_ci")
    has_sd = has_baseline_sd or (_source_has(provenance, "post_s1") and _source_has(provenance, "post_s2"))

    if has_pre and has_post and has_sd:
        formats.append("Pre/post group means with SDs and sample sizes")
    elif has_pre and has_post:
        formats.append("Pre/post group means; some SD information is missing")
    if has_change:
        formats.append("Change-score summary information")
    if has_adjusted:
        formats.append("Adjusted effect estimate")
    if has_ci:
        formats.append("Confidence interval or mean-difference summary")
    if has_reported_g:
        formats.append("Reported Hedges' g")
    if not formats:
        formats.append("Reporting format not confidently recognised")
    return " + ".join(dict.fromkeys(formats)), formats


def _outcome_type(outcome_group: str, reporting_formats: list[str], row: pd.Series) -> str:
    if outcome_group in {"Anxiety", "CRP", "IL-6", "TNF-α", "Cortisol"}:
        return "Continuous"
    candidate_payload = _json_obj(row.get("candidate_effect_methods"))
    if candidate_payload.get("candidates"):
        return "Continuous (candidate effect-size routes detected)"
    return "Unclear — researcher review needed"


def _has_adjusted_effect(provenance: str, row: pd.Series | None = None) -> bool:
    p = provenance.lower()
    # A label alone is not enough: generic inverse variance needs an adjusted
    # effect estimate AND uncertainty (SE, variance or a compatible CI).
    has_adjusted_label = any(token in p for token in (
        "ancova", "adjusted_effect", "adjustedtreatment", "adjustedestimate",
        "regressioncoefficient", "beta_adjusted", "adjustedhedgesg"
    ))
    if not has_adjusted_label or row is None:
        return False
    has_effect = any(_number(row.get(col)) is not None for col in (
        "selected_hedges_g", "independently_calculated_hedges_g", "reported_hedges_g"
    ))
    has_uncertainty = any(_number(row.get(col)) is not None for col in (
        "standard_error", "variance"
    )) or ("ci" in p and any(token in p for token in ("adjusted", "ancova", "regression")))
    return bool(has_effect and has_uncertainty)


def _get_review_option(label: str, options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"value": x["value"], "label": x["label"], "explanation": x.get("explanation", "")} for x in options]


def _review_item(review_id: str, category: str, subject: str, finding: str,
                 why: str, suggestion: str, options: list[dict[str, Any]],
                 affected_records: list[str] | None = None, severity: str = "Warning") -> dict[str, Any]:
    return {
        "Review ID": review_id,
        "Category": category,
        "Priority": "High" if severity in {"Critical", "Error"} else "Medium",
        "Subject": subject,
        "What MetaPy found": finding,
        "Why it matters": why,
        "MetaPy suggestion": suggestion,
        "Available options": json.dumps(_get_review_option("", options), ensure_ascii=False),
        "Your choice": "",
        "Status": "Pending",
        "Affected record IDs": "; ".join(affected_records or []),
        "Decision recorded (UTC)": "",
        "Severity": severity,
    }


def _issue(issue_id: str, severity: str, category: str, subject: str,
           finding: str, impact: str, action: str, affected: list[str] | None = None,
           blocking: bool = False) -> dict[str, Any]:
    return {
        "Issue ID": issue_id,
        "Severity": severity,
        "Category": category,
        "Subject": subject,
        "What MetaPy found": finding,
        "Why it matters": impact,
        "Recommended next step": action,
        "Affected record IDs": "; ".join(affected or []),
        "Blocking for primary analysis": bool(blocking),
        "Status": "Open",
    }


def _unit_issue_columns() -> list[str]:
    return ["Issue ID", "Severity", "Category", "Subject", "What MetaPy found", "Why it matters",
            "Recommended next step", "Affected record IDs", "Blocking for primary analysis", "Status"]


def _reconcile_unit_findings(findings: pd.DataFrame, review_items: pd.DataFrame) -> pd.DataFrame:
    """Rebuild unit findings from the current parent/child researcher decisions.

    This removes stale unit warnings and recreates one current parent status plus
    a specific blocker only when a conversion or unresolved-unit action remains.
    """
    out = findings.copy(deep=True) if isinstance(findings, pd.DataFrame) else pd.DataFrame(columns=_unit_issue_columns())
    if out.empty:
        out = pd.DataFrame(columns=_unit_issue_columns())
    issue_ids = out.get("Issue ID", pd.Series(index=out.index, dtype=str)).fillna("").astype(str)
    out = out.loc[~issue_ids.str.startswith("DP-PREP-UNIT-")].copy()

    if not isinstance(review_items, pd.DataFrame) or review_items.empty:
        return out.reset_index(drop=True)

    categories = review_items.get("Category", pd.Series(index=review_items.index, dtype=str)).fillna("").astype(str)
    parents = review_items[categories.eq("Measurement unit check")]
    children = review_items[categories.eq("Unit verification follow-up")]
    rebuilt: list[dict[str, Any]] = []

    for _, parent in parents.iterrows():
        parent_id = _safe(parent.get("Review ID"))
        if _safe(parent.get("Status")) != "Approved":
            continue
        parent_choice = _safe(parent.get("Your choice"))
        affected = _split_semicolon(parent.get("Affected record IDs"))
        subject = _safe(parent.get("Subject"))

        def add_parent(status_text: str, message: str, action: str, blocking: bool, severity: str = "Information"):
            rebuilt.append({
                "Issue ID": f"DP-{parent_id}", "Severity": severity,
                "Category": "Measurement unit", "Subject": subject,
                "What MetaPy found": message,
                "Why it matters": "Treatment and control measurements must be on compatible scales within each study. Different units between studies can often be handled with a standardised effect size.",
                "Recommended next step": action,
                "Affected record IDs": "; ".join(affected),
                "Blocking for primary analysis": bool(blocking), "Status": status_text,
            })

        if parent_choice == "same_within_study":
            add_parent(
                "Resolved — researcher confirmed within-study unit compatibility",
                "The researcher confirmed that the compared groups and time points use compatible units within each study.",
                "No numerical unit conversion is required for this review item. Retain the verification in the audit trail.", False,
            )
            continue

        if parent_choice not in {"units_need_harmonisation", "unit_not_reported"}:
            continue

        child_matches = children[children.get("Depends on", pd.Series(index=children.index, dtype=str)).fillna("").astype(str).eq(parent_id)]
        child = child_matches.iloc[0].to_dict() if not child_matches.empty else {}
        child_status = _safe(child.get("Status"))
        child_choice = _safe(child.get("Your choice"))
        child_id = _safe(child.get("Review ID")) or f"PREP-UNIT-FOLLOWUP-{_slug(subject.replace('Check measurement units for ', '')).upper()}"

        if child_status == "Approved" and child_choice == "same_units_verified":
            add_parent(
                "Resolved — dependent verification confirmed compatible units",
                "The initial unit concern was checked in the follow-up form, and the researcher confirmed compatible units within each affected study.",
                "No numerical conversion is required for these records. The parent warning has been resolved by the dependent review.", False,
            )
        elif child_status == "Approved" and child_choice == "conversion_factor_known":
            add_parent(
                "Advanced — numerical conversion still required",
                "The researcher confirmed that a conversion factor is available, but has not yet entered and applied the verified numerical factor.",
                "Enter the factor, source unit, target unit and supporting source through the structured conversion workflow.", False,
            )
            rebuilt.append(_issue(
                f"DP-{child_id}-FACTOR", "Critical", "Unit harmonisation",
                f"Enter and verify the conversion factor for {subject.replace('Check measurement units for ', '')}",
                "A conversion factor is said to be available, but no numerical conversion has been entered or applied.",
                "The affected group values may still be on incompatible scales.",
                "Provide the verified numerical factor, source/target units and source reference before calculations.", affected, True,
            ))
        elif child_status == "Approved" and child_choice == "unit_unresolved":
            add_parent(
                "Advanced — dependent review confirms units remain unresolved",
                "The researcher could not verify within-study unit compatibility in the follow-up review.",
                "Review the original reports or keep the affected records out of the calculation set until clarified.", False,
            )
            rebuilt.append(_issue(
                f"DP-{child_id}-UNRESOLVED", "Critical", "Unit harmonisation",
                f"Within-study unit compatibility remains unresolved for {subject.replace('Check measurement units for ', '')}",
                "The researcher could not verify whether treatment and control values are comparable.",
                "A within-study scale mismatch could invalidate the treatment-versus-control effect estimate.",
                "Check the source report or keep the affected records out of the calculation set until resolved.", affected, True,
            ))
        else:
            add_parent(
                "Open — dependent unit verification is not yet approved",
                "The parent decision indicated possible unit incompatibility, but the follow-up verification has not been approved.",
                "Complete the dependent unit-verification form. Confirm compatible units, enter a documented conversion factor, or keep the records unresolved.",
                True, "Warning",
            )

    if rebuilt:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(rebuilt)], ignore_index=True)
    return out.reset_index(drop=True)


def _reported_effect_counts(prepared: pd.DataFrame) -> dict[str, int]:
    """Count reported effects by evidence state, including post-review labels."""
    keys = {
        "reported_effect_estimates_present": 0,
        "reported_g_assumed_ppc2_unverified": 0,
        "reported_ppc2_confirmed_by_researcher": 0,
        "reported_effect_method_not_established": 0,
    }
    if prepared is None or prepared.empty:
        return keys
    status = prepared.get("Reported g method status", pd.Series("", index=prepared.index)).fillna("").astype(str)
    has_reported = prepared.get("_has_reported_g", prepared.get("Reported Hedges g", pd.Series(index=prepared.index, dtype=float)).notna()).fillna(False).astype(bool)
    unverified_labels = {"Assumed Morris PPC2 — unverified", "Retain as unverified PPC2"}
    reported_only = prepared.get("_reported_only", pd.Series(True, index=prepared.index)).fillna(False).astype(bool)
    unverified = status.isin(unverified_labels) & reported_only & has_reported
    confirmed = status.eq("Researcher confirms PPC2 from original report") & has_reported
    unknown = has_reported & ~unverified & ~confirmed
    return {
        "reported_effect_estimates_present": int(has_reported.sum()),
        "reported_g_assumed_ppc2_unverified": int(unverified.sum()),
        "reported_ppc2_confirmed_by_researcher": int(confirmed.sum()),
        "reported_effect_method_not_established": int(unknown.sum()),
    }


def _refresh_summary_statuses(summary: dict[str, Any], prepared: pd.DataFrame,
                              review_items: pd.DataFrame | None = None) -> dict[str, Any]:
    """Refresh reported-effect, calculation and researcher-review counters."""
    out = dict(summary or {})
    out.update(_reported_effect_counts(prepared))
    if prepared is None or prepared.empty:
        out.update({"raw_data_effect_estimate_candidates": 0,
                    "independent_effect_estimates_calculated": 0,
                    "sampling_variances_calculated": 0,
                    "records_with_primary_route_candidate": 0})
    else:
        route_status = prepared.get("Primary effect route status", pd.Series("", index=prepared.index)).fillna("").astype(str)
        out["records_with_primary_route_candidate"] = int(prepared.get("Primary effect route eligible", pd.Series(False, index=prepared.index)).fillna(False).astype(bool).sum())
        out["raw_data_effect_estimate_candidates"] = int(route_status.str.contains("candidate available", case=False, na=False).sum())
        out["independent_effect_estimates_calculated"] = int(prepared.get("Independent effect-size calculation status", pd.Series("", index=prepared.index)).astype(str).str.startswith("Calculated").sum())
        out["sampling_variances_calculated"] = int(prepared.get("Sampling variance calculation status", pd.Series("", index=prepared.index)).astype(str).str.startswith("Calculated").sum())
    if isinstance(review_items, pd.DataFrame) and not review_items.empty:
        dependency = review_items.get("Dependency state", pd.Series("Active", index=review_items.index)).fillna("Active").astype(str)
        statuses = review_items.get("Status", pd.Series("", index=review_items.index)).fillna("").astype(str)
        active = ~dependency.str.startswith("Inactive") & ~statuses.eq("Not applicable")
        out["preparation_review_items"] = int(active.sum())
        out["dynamic_dependent_review_items"] = int((review_items.get("Is dynamic", pd.Series(False, index=review_items.index)).fillna(False).astype(bool) & active).sum())
        out["preparation_reviews_approved"] = int((active & statuses.eq("Approved")).sum())
        out["preparation_reviews_pending"] = int((active & statuses.isin(["Pending", "Deferred"])).sum())
    return out


def _reconcile_reported_raw_route_finding(findings: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    """Refresh mismatch warning without overwriting any unrelated finding row."""
    out = findings.copy(deep=True) if isinstance(findings, pd.DataFrame) else pd.DataFrame(columns=_unit_issue_columns())
    if out.empty:
        out = pd.DataFrame(columns=_unit_issue_columns())
    out = out.loc[out.get("Issue ID", pd.Series(index=out.index, dtype=str)).astype(str).ne("DP-REPORTED-PLUS-RAW-ROUTE")].copy()
    if prepared is None or prepared.empty:
        return out.reset_index(drop=True)
    reported_status = prepared.get("Reported g method status", pd.Series("", index=prepared.index)).fillna("").astype(str)
    confirmed = reported_status.eq("Researcher confirms PPC2 from original report")
    reported_values = prepared.get("Reported Hedges g", pd.Series(index=prepared.index, dtype=float)).notna()
    raw_prepost = prepared.get("_has_raw_prepost", pd.Series(False, index=prepared.index)).fillna(False).astype(bool)
    baseline_ok = prepared.get("_has_baseline_sd_both_groups", pd.Series(False, index=prepared.index)).fillna(False).astype(bool)
    eligible = prepared.get("Primary effect route eligible", pd.Series(False, index=prepared.index)).fillna(False).astype(bool)
    reported_only = prepared.get("_reported_only", pd.Series(False, index=prepared.index)).fillna(False).astype(bool)
    missing = prepared[reported_values & raw_prepost & ~baseline_ok & ~eligible & ~confirmed & ~reported_only]
    if not missing.empty:
        issue = _issue(
            "DP-REPORTED-PLUS-RAW-ROUTE", "Warning", "Reported versus raw-data route",
            f"{len(missing)} record(s) have both reported g and raw pre/post data, but neither route is currently approved for the primary analysis",
            "These records contain a reported Hedges' g as well as incomplete raw pre/post measurements.",
            "The reported effect is not automatically interchangeable with a recalculated PPC2 estimate.",
            "Confirm the reported effect-size method from the original report, or keep the reported value separate until its method is established.",
            missing["Preparation record ID"].astype(str).tolist(), True,
        )
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame([issue])], ignore_index=True)
    return out.reset_index(drop=True)


def _reconcile_baseline_sd_findings(findings: pd.DataFrame, prepared: pd.DataFrame,
                                    review_items: pd.DataFrame | None = None) -> pd.DataFrame:
    """Keep missing raw SDs distinct from a separately confirmed reported PPC2 effect."""
    out = findings.copy(deep=True) if isinstance(findings, pd.DataFrame) else pd.DataFrame(columns=_unit_issue_columns())
    if out.empty:
        out = pd.DataFrame(columns=_unit_issue_columns())
    if prepared is None or prepared.empty:
        return out.reset_index(drop=True)
    raw_prepost = prepared.get("_has_raw_prepost", pd.Series(False, index=prepared.index)).fillna(False).astype(bool)
    baseline_ok = prepared.get("_has_baseline_sd_both_groups", pd.Series(False, index=prepared.index)).fillna(False).astype(bool)
    missing = prepared[raw_prepost & ~baseline_ok]
    base_id, action_id = "DP-PPC2-BASELINE-SD", "DP-PPC2-BASELINE-SD-ACTION"
    if missing.empty:
        return out.loc[~out["Issue ID"].astype(str).isin({base_id, action_id})].reset_index(drop=True)

    reported_status = missing.get("Reported g method status", pd.Series("", index=missing.index)).fillna("").astype(str)
    reported_values = missing.get("Reported Hedges g", pd.Series(index=missing.index, dtype=float)).notna()
    confirmed = reported_status.eq("Researcher confirms PPC2 from original report") & reported_values
    all_confirmed = bool(confirmed.all()) and len(missing) > 0
    ids = missing["Preparation record ID"].astype(str).tolist()
    base = _issue(
        base_id, "Warning" if all_confirmed else "Critical", "Missing PPC2 input",
        f"{len(missing)} raw pre/post record(s) lack baseline SDs in both groups",
        ("Raw-data PPC2 cannot be independently recalculated because baseline SDs are missing. A separate reported PPC2 estimate has been confirmed by the researcher for every affected record." if all_confirmed else "Raw-data PPC2 cannot be independently recalculated because baseline SDs are missing, and at least one affected record has no separately confirmed reported PPC2 estimate."),
        "Raw-data recalculation remains unavailable. Retaining a reported effect does not verify its variance; sampling variance must be assessed separately.",
        ("Retain the source-reported PPC2 effect with its provenance. Keep raw-data recalculation marked blocked and assess sampling variance through a compatible, documented route." if all_confirmed else "Retrieve both missing baseline SDs from a verified source, or keep the affected record out of the current primary analysis until a compatible effect estimate is established."),
        ids, not all_confirmed,
    )
    base["Status"] = "Open — raw-data recalculation unavailable; reported effect retained" if all_confirmed else "Open — raw-data PPC2 route blocked"
    positions = out.index[out["Issue ID"].astype(str).eq(base_id)].tolist() if not out.empty else []
    if positions:
        for key, value in base.items(): out.at[positions[0], key] = value
        if len(positions) > 1: out = out.drop(index=positions[1:])
    else:
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame([base])], ignore_index=True)

    categories = review_items.get("Category", pd.Series(index=review_items.index, dtype=str)).fillna("").astype(str) if isinstance(review_items, pd.DataFrame) and not review_items.empty else pd.Series(dtype=str)
    source_rows = review_items[categories.eq("Baseline SD source verification")] if isinstance(review_items, pd.DataFrame) and not review_items.empty else pd.DataFrame()
    source = source_rows.iloc[0].to_dict() if not source_rows.empty else {}
    source_status, source_choice = _safe(source.get("Status")), _safe(source.get("Your choice"))
    action_pos = out.index[out["Issue ID"].astype(str).eq(action_id)].tolist() if not out.empty else []
    action_values = None
    if source_status == "Approved" and source_choice == "no_valid_statistic_found" and all_confirmed:
        action_values = _issue(
            action_id, "Information", "Missing PPC2 input",
            f"{len(missing)} record(s) cannot be independently recalculated from raw data",
            "No valid alternative statistic was found for the missing baseline SDs. The separately reported PPC2 effects were confirmed by the researcher and are retained; this does not complete raw-data recalculation.",
            "The reported effect estimate and the independently recalculated raw-data effect are distinct. Sampling variance remains a separate calculation requirement.",
            "Retain the reported estimates with provenance; keep raw recalculation marked blocked and assess sampling variance using a compatible route in the calculation stage.", ids, False,
        )
        action_values["Status"] = "Resolved for reported-effect retention — raw recalculation unavailable"
    elif source_status == "Approved" and source_choice == "valid_source_statistic_found":
        action_values = _issue(
            action_id, "Critical", "Missing PPC2 input", "Verified baseline SD values still need to be entered",
            "A candidate source statistic was identified, but MetaPy has not received the verified numerical baseline SD values.",
            "The raw-data PPC2 point estimate cannot be independently recalculated without the actual values.",
            "Enter the verified baseline SD values and source through the structured correction workflow.", ids, True,
        )
        action_values["Status"] = "Open — verified numerical values required"
    elif source_status == "Approved" and source_choice == "no_valid_statistic_found":
        action_values = _issue(
            action_id, "Warning", "Missing PPC2 input", "No valid statistic was found to recover the missing baseline SDs",
            "The researcher checked for an alternative statistic but did not find a valid value. The affected raw-data PPC2 route remains unavailable.",
            "Without a separately confirmed reported PPC2 estimate or the missing baseline SDs, this record cannot enter the selected PPC2 route.",
            "Keep the records held or obtain the missing baseline SDs from a verified source.", ids, True,
        )
        action_values["Status"] = "Open — no valid alternative statistic found"
    if action_values is not None:
        if action_pos:
            for key, value in action_values.items(): out.at[action_pos[0], key] = value
            if len(action_pos) > 1: out = out.drop(index=action_pos[1:])
        else:
            out = pd.concat([out.reset_index(drop=True), pd.DataFrame([action_values])], ignore_index=True)
    return out.reset_index(drop=True)


def _safe_id(study: Any, arm: Any, outcome: Any, duration: Any, index: int) -> str:
    raw = "|".join((_safe(study), _safe(arm), _safe(outcome), _safe(duration)))
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()[:48]
    return f"REC-{slug or 'record'}-{index:03d}"


def _study_design_label(design: str) -> str:
    return {
        "parallel_independent": "Parallel independent groups (project-level selection)",
        "dependent_or_unknown": "Paired/crossover/repeated or not yet classified",
    }.get(design, f"Project-level design setting: {design or 'not selected'}")


def _get_units_in_sheets(sheets: dict[str, Any], provenance: str) -> bool:
    for sheet in (sheets or {}).values():
        columns = [str(c).lower() for c in getattr(sheet, "columns", [])]
        if any("unit" in c or "measurement" in c and "unit" in c for c in columns):
            return True
    return "unit" in provenance.lower()


def _get_value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index and _number(row.get(name)) is not None:
            return _number(row.get(name))
    return None


def _build_prepared_records(session: dict[str, Any], queue: pd.DataFrame,
                            decisions: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    review = session.get("review", {})
    harmonization = review.get("harmonization", {}) if isinstance(review, dict) else {}
    records = harmonization.get("records", pd.DataFrame())
    if not isinstance(records, pd.DataFrame) or records.empty:
        raise ValueError("No canonical harmonised records were found. Re-inspect the workbook before data preparation.")

    primary = decisions.get("Primary analysis method", {})
    primary_choice = primary.get("choice", "") if primary.get("status") == "Approved" else ""
    reported = decisions.get("Reported effect-size method", {})
    reported_assumed = (
        reported.get("status") == "Approved" and reported.get("choice") == "assume_ppc2_unverified"
    )
    reported_excluded = (
        reported.get("status") == "Approved" and reported.get("choice") == "exclude_reported"
    )
    variance = decisions.get("Sampling variance plan", {})
    variance_approved = variance.get("status") == "Approved"
    variance_details = variance.get("details", {})
    correlation_values = variance_details.get("correlation_values", []) if variance_approved else []
    correlation_values = sorted({float(x) for x in correlation_values if _number(x) is not None and 0 <= float(x) <= 1})
    multi = decisions.get("Multi-arm study dependence", {})
    multi_choice = multi.get("choice", "") if multi.get("status") == "Approved" else ""
    eligibility = decisions.get("Study eligibility", {})
    eligibility_choice = eligibility.get("choice", "") if eligibility.get("status") == "Approved" else ""

    rows: list[dict[str, Any]] = []
    for i, (_, source_row) in enumerate(records.iterrows(), start=1):
        row = source_row.copy()
        study_id = _safe(row.get("study_id"))
        arm = _safe(row.get("arm"))
        outcome_raw = _safe(row.get("outcome"))
        duration = _safe(row.get("duration"))
        rec_id = _safe(row.get("preparation_record_id")) or _safe_id(study_id, arm, outcome_raw, duration, i)
        outcome_group, outcome_description = _classify_outcome(outcome_raw)
        provenance = _safe(row.get("source_value_provenance"))
        reporting_text, reporting_parts = _reporting_format(row, provenance)
        candidate_payload, candidate_estimands = _candidate_data(row)
        ppc2_candidate = _candidate_for(candidate_payload, "morris_ppc2")
        ancova_present = _has_adjusted_effect(provenance, row)
        raw_prepost = _source_has(provenance, "pre_m1") and _source_has(provenance, "pre_m2") and _source_has(provenance, "post_m1") and _source_has(provenance, "post_m2")
        has_reported_g = _number(row.get("reported_hedges_g")) is not None
        reported_only = has_reported_g and not raw_prepost
        if reported_excluded and reported_only:
            effect_status = "Excluded from current primary analysis by researcher decision"
            eligible_primary = False
        elif primary_choice == "morris_ppc2":
            if ppc2_candidate:
                effect_status = "PPC2 candidate available — ready for calculation execution"
                eligible_primary = True
            elif reported_only and reported_assumed:
                effect_status = "Reported Hedges' g retained as unverified PPC2 assumption"
                eligible_primary = True
            else:
                effect_status = "No eligible PPC2 point-estimate route identified"
                eligible_primary = False
        elif primary_choice == "change_smd":
            change_candidate = _candidate_for(candidate_payload, "change_smd")
            effect_status = "Change-score SMD candidate available" if change_candidate else "Change-score variability or a compatible route is missing"
            eligible_primary = bool(change_candidate)
        elif primary_choice == "ancova_giv":
            effect_status = "ANCOVA-adjusted effect and its SE/CI available" if ancova_present else "No clearly identified ANCOVA-adjusted effect with uncertainty found"
            eligible_primary = bool(ancova_present)
        elif primary_choice == "not_decided":
            effect_status = "Primary estimand is not selected"
            eligible_primary = False
        else:
            effect_status = "Primary-method decision is not yet approved"
            eligible_primary = False

        if reported_only and not reported_assumed and not reported_excluded:
            effect_status += "; reported method remains unverified"
            eligible_primary = False

        # Preserve original group sample sizes. A split-control value is prepared
        # separately for later variance/weight calculations only.
        n1 = _get_value(row, "n_intervention")
        n2 = _get_value(row, "n_control")
        rows.append({
            "Preparation record ID": rec_id,
            "Study ID": study_id,
            "Intervention arm": arm or "Not detected",
            "Outcome label": outcome_group,
            "Original outcome / instrument": outcome_raw,
            "Outcome data type": _outcome_type(outcome_group, reporting_parts, row),
            "Outcome description": outcome_description,
            "Outcome direction": "Needs researcher confirmation",
            "Effect direction multiplier": None,
            "Cortisol sampling context status": "Not yet reviewed",
            "Unit metadata status": "Unit column not detected; within-study consistency needs review",
            "Actual pre-post correlation (intervention)": _number(row.get("r_prepost_intervention_source")),
            "Actual pre-post correlation (control)": _number(row.get("r_prepost_control_source")),
            "Actual correlation availability": (
                "Both group-specific source correlations detected"
                if _number(row.get("r_prepost_intervention_source")) is not None and _number(row.get("r_prepost_control_source")) is not None
                else "One or both group-specific source correlations missing"
            ),
            "Correlation route recommendation": (
                "Prefer actual source correlations for primary variance; use assumed grid only as separate sensitivity analysis"
                if _number(row.get("r_prepost_intervention_source")) is not None and _number(row.get("r_prepost_control_source")) is not None
                else "Actual group-specific correlations not both available; researcher may consider a separate sensitivity analysis"
            ),
            "Individual record approval": "Pending — researcher sign-off required",
            "Reporting format": reporting_text,
            "Study design": _study_design_label(_safe(session.get("design"))),
            "Study design confirmation": "Project-level setting; study-level confirmation pending",
            "Analysis structure": "Pairwise treatment-versus-control comparison" if n1 and n2 else "Analysis structure not confidently recognised",
            "Analysis structure detail": "Possible shared control" if arm else "Single comparison or arm not detected",
            "Follow-up / duration": duration,
            "N intervention (source)": n1,
            "N control (source)": n2,
            "N control for sample-size-based variance routes": n2,
            "N control for selected primary variance route": n2,
            "Shared-control comparison count": 1,
            "Shared-control preparation rule status": "No shared control detected or not yet handled",
            "Reported Hedges g": _number(row.get("reported_hedges_g")),
            "Reported Cohen d": _number(row.get("reported_cohens_d")),
            "Previously selected Hedges g (source harmonisation)": _number(row.get("selected_hedges_g")),
            "Previously calculated Hedges g (source harmonisation)": _number(row.get("independently_calculated_hedges_g")),
            "Previously available variance (source harmonisation)": _number(row.get("variance")),
            "Previously available SE (source harmonisation)": _number(row.get("standard_error")),
            "Candidate estimands detected": "; ".join(candidate_estimands),
            "Candidate calculations (structured JSON)": json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True),
            "Selected primary estimand": primary_choice or "Not approved",
            "Primary effect route status": effect_status,
            "Primary effect route eligible": bool(eligible_primary),
            "Reported g method status": "Assumed Morris PPC2 — unverified" if reported_only and reported_assumed else ("Reported method not established" if has_reported_g else "Not reported"),
            "Reported effect estimate status": (
                "Reported Hedges' g present — unverified PPC2 assumption" if reported_only and reported_assumed else
                ("Reported Hedges' g present — method not established" if has_reported_g else "No reported effect estimate")
            ),
            "Independent effect-size calculation status": (
                "Candidate available — raw-data calculation not yet executed" if ppc2_candidate and primary_choice == "morris_ppc2" else
                ("Blocked — required baseline SD missing" if raw_prepost and (not _source_has(provenance, "pre_s1") or not _source_has(provenance, "pre_s2")) else
                 ("Not independently recalculated — reported-only record" if reported_only else "No compatible raw-data calculation completed"))
            ),
            "Sampling variance calculation status": (
                "Source variance supplied — verification required" if _number(row.get("variance")) is not None else
                ("Actual group-specific source correlations available; use these for primary variance reconstruction"
                 if _number(row.get("r_prepost_intervention_source")) is not None and _number(row.get("r_prepost_control_source")) is not None else
                 ("Not calculated — approved correlation sensitivity assumptions recorded" if variance_approved and correlation_values else "Not calculated — variance inputs require review"))
            ),
            "Variance preparation status": (
                "Actual group-specific source correlations detected; assumed grid is reserved for separate sensitivity analysis"
                if _number(row.get("r_prepost_intervention_source")) is not None and _number(row.get("r_prepost_control_source")) is not None else
                (f"Sensitivity grid approved: {', '.join(str(x) for x in correlation_values)}; values remain assumptions"
                 if variance_approved and variance.get("choice") == "approved_correlation_sensitivity" and correlation_values
                 else "Variance information not calculated in data preparation")
            ),
            "Variance correlation assumptions": "; ".join(str(x) for x in correlation_values),
            "Study eligibility decision": eligibility_choice or "Not approved",
            "Researcher disposition": "Not yet reviewed at record level",
            "Preparation status": "Prepared for researcher review" if eligible_primary else "Needs review / not ready for selected estimand",
            "Source locations": _safe(row.get("source_locations")),
            "Source value provenance": provenance,
            "Original harmonized outcome key": outcome_raw,
            "_has_raw_prepost": bool(raw_prepost),
            "_has_baseline_sd_both_groups": bool(_source_has(provenance, "pre_s1") and _source_has(provenance, "pre_s2")),
            "_has_reported_g": bool(has_reported_g),
            "_reported_only": bool(reported_only),
            "_has_ancova": bool(ancova_present),
            "_candidate_estimands": candidate_estimands,
            "_study_id": study_id,
            "_arm": arm,
            "_outcome_group": outcome_group,
            "_duration": duration,
        })

    prepared = pd.DataFrame(rows)

    # Detect shared-control groups from canonical records. The sample size used
    # for future variance weighting is split only after the researcher approved
    # the split-control option. Source Ns and effect-estimation Ns remain intact.
    if not prepared.empty:
        group_cols = ["_study_id", "Original outcome / instrument", "_duration"]
        for _, idxs in prepared.groupby(group_cols, dropna=False).groups.items():
            group = prepared.loc[list(idxs)]
            arms = [x for x in group["_arm"].astype(str).tolist() if x]
            controls = [_number(x) for x in group["N control (source)"].tolist()]
            nonmissing = [x for x in controls if x is not None]
            if len(set(arms)) >= 2 and len(nonmissing) >= 2 and len(set(nonmissing)) == 1:
                count = len(set(arms))
                if multi_choice == "split_control":
                    for index in idxs:
                        n_control = _number(prepared.at[index, "N control (source)"])
                        if n_control is not None:
                            split_n = n_control / count
                            prepared.at[index, "N control for sample-size-based variance routes"] = split_n
                            if primary_choice in {"morris_ppc2", "change_smd"}:
                                prepared.at[index, "N control for selected primary variance route"] = split_n
                                primary_n_note = "Split-control N prepared for selected change-based variance route"
                            else:
                                prepared.at[index, "N control for selected primary variance route"] = n_control
                                primary_n_note = "Primary method uses a direct adjusted estimate/SE route; split N is stored only for alternative sample-size-based routes"
                            prepared.at[index, "Shared-control comparison count"] = count
                            prepared.at[index, "Shared-control preparation rule status"] = primary_n_note
                            prepared.at[index, "Analysis structure detail"] = f"Shared control; split across {count} comparisons for sample-size-based pairwise variance routes"
                elif multi_choice == "hold_multiarm":
                    for index in idxs:
                        prepared.at[index, "Researcher disposition"] = "Hold pending — shared-control handling not approved"
                        prepared.at[index, "Preparation status"] = "Held pending multi-arm handling"
                else:
                    for index in idxs:
                        prepared.at[index, "Shared-control comparison count"] = count
                        prepared.at[index, "Analysis structure detail"] = f"Shared control detected; selected handling: {multi_choice or 'not approved'}"

    # Determine whether the workbook actually exposes unit metadata.
    unit_metadata_detected = _get_units_in_sheets(session.get("sheets", {}), " ".join(prepared.get("Source value provenance", pd.Series(dtype=str)).astype(str).tolist()))
    if not unit_metadata_detected:
        prepared["Unit metadata status"] = "Not detected in uploaded workbook"
    else:
        # A single non-missing unit value in a worksheet is evidence of no
        # within-worksheet mismatch, not external verification of the unit.
        unit_map = {}
        for sheet_name, frame in (session.get("sheets", {}) or {}).items():
            cols = [c for c in getattr(frame, "columns", []) if "unit" in _safe(c).lower()]
            if not cols:
                continue
            values = set()
            for col in cols:
                values.update(_safe(v) for v in frame[col].dropna().tolist() if _safe(v))
            unit_map[str(sheet_name).lower()] = values
        for idx, row in prepared.iterrows():
            source_locations = _safe(row.get("Source locations")).lower()
            matching = [vals for name, vals in unit_map.items() if name in source_locations]
            if matching and all(len(vals) == 1 for vals in matching):
                prepared.at[idx, "Unit metadata status"] = "Same single unit declared within source worksheet; no within-record mismatch detected (not externally verified)"
            else:
                prepared.at[idx, "Unit metadata status"] = "Unit column detected; compatibility still requires researcher review"

    # Global review-form approval is not individual comparison sign-off.
    prepared["Individual record approval"] = "Pending — researcher sign-off required"
    prepared["Record-level readiness"] = "Not approved for final analysis until this comparison is signed off"

    return prepared, records.copy(deep=True)


def _build_review_items(session: dict[str, Any], prepared: pd.DataFrame,
                        decisions: dict[str, dict[str, Any]]) -> pd.DataFrame:
    items: list[dict[str, Any]] = []
    if prepared.empty:
        return pd.DataFrame()

    primary_choice = decisions.get("Primary analysis method", {}).get("choice", "") if decisions.get("Primary analysis method", {}).get("status") == "Approved" else ""
    # Critical method/data compatibility check. A reported g assumed to be PPC2
    # is not an ANCOVA-adjusted estimate.
    if primary_choice == "ancova_giv" and not prepared["Primary effect route eligible"].any():
        items.append(_review_item(
            "PREP-METHOD-COMPATIBILITY", "Primary method versus available data",
            "ANCOVA selected, but adjusted estimates were not detected",
            "The approved primary method is ANCOVA. The workbook has pre/post summaries and reported Hedges' g values, but MetaPy did not identify study-level ANCOVA-adjusted estimates with their SEs or confidence intervals.",
            "An assumed PPC2 Hedges' g is not an ANCOVA estimate. These methods cannot be relabelled or substituted for one another.",
            "Choose whether to keep ANCOVA and obtain compatible adjusted estimates, reconsider PPC2 through the original decision form, or keep PPC2 only as a separate alternative.",
            METHOD_COMPATIBILITY_OPTIONS, severity="Critical",
        ))

    multi_decision = decisions.get("Multi-arm study dependence", {})
    if primary_choice == "ancova_giv" and multi_decision.get("status") == "Approved" and multi_decision.get("choice") == "split_control":
        items.append(_review_item(
            "PREP-MULTIARM-ANCOVA-COMPATIBILITY", "Multi-arm method compatibility",
            "Split-control approval needs review alongside ANCOVA",
            "The researcher approved splitting shared-control sample sizes. However, the currently selected primary method is ANCOVA generic inverse variance, which usually uses the study's adjusted estimate and its reported SE or confidence interval.",
            "Dividing a sample size does not change an already-reported ANCOVA standard error and does not, by itself, account for dependence between adjusted comparisons.",
            "Keep multi-arm pooling on hold until a method compatible with the supplied adjusted estimates is selected, or reconsider the primary estimand.",
            [
                {"value": "hold_ancova_multiarm", "label": "Keep ANCOVA comparisons on hold until dependence is handled", "explanation": "Preserves the current ANCOVA choice but blocks pooling of dependent comparisons."},
                {"value": "request_robust_multiarm", "label": "Reconsider multivariate or robust-variance handling", "explanation": "This requires an appropriate dependence model and sufficient information; it will send the existing multi-arm decision back for approval."},
                {"value": "request_combine_arms", "label": "Reconsider combining sufficiently similar intervention arms", "explanation": "Combining arms is only suitable when clinically and methodologically justified; it will require a new approval."},
                {"value": "revisit_primary_method", "label": "Reconsider the primary effect-size method", "explanation": "Return to the primary-method form and choose a method that matches the available data and protocol."},
            ], severity="Critical",
        ))

    # Project design is a project-wide choice; ask the researcher to confirm it
    # applies to the included studies, not just the upload screen default.
    design = _safe(session.get("design"))
    if design == "parallel_independent":
        items.append(_review_item(
            "PREP-DESIGN-CONFIRM", "Study design confirmation",
            "Confirm study designs across the included trials",
            "The upload screen selected parallel independent groups. The workbook itself does not independently prove that every study has this design.",
            "A crossover, paired, cluster-randomised or repeated-measures study may need different calculations or dependence handling.",
            "Confirm all studies are parallel independent-group trials, identify a mixed-design dataset, or keep the design unresolved.",
            DESIGN_OPTIONS, severity="Warning",
        ))

    # Outcome direction must be explicit before interpreting pooled signs.
    outcome_groups = sorted(prepared["Outcome label"].dropna().astype(str).unique().tolist())
    for outcome in outcome_groups:
        records = prepared[prepared["Outcome label"].eq(outcome)]
        items.append(_review_item(
            f"PREP-DIRECTION-{_slug(outcome).upper()}", "Outcome direction",
            f"Confirm how to interpret values for {outcome}",
            f"MetaPy grouped {len(records)} record(s) under {outcome}. It has not assumed that a positive or negative effect always means benefit.",
            "MetaPy needs to know what higher and lower values mean for this outcome. The interpretation can vary by instrument or clinical context.",
            "Select the direction supported by the measurement instrument or original study. For cortisol, consider sampling time and context.",
            DIRECTION_OPTIONS, affected_records=records["Preparation record ID"].tolist(), severity="Warning",
        ))

    # Biomarker units are not visible in this workbook. SMDs may accommodate
    # different units across trials, but treatment/control values within each
    # study must be compatible.
    if not _get_units_in_sheets(session.get("sheets", {}), " ".join(prepared.get("Source value provenance", pd.Series(dtype=str)).astype(str).tolist())):
        for outcome in [x for x in outcome_groups if x in {"CRP", "IL-6", "TNF-α", "Cortisol"}]:
            records = prepared[prepared["Outcome label"].eq(outcome)]
            items.append(_review_item(
                f"PREP-UNIT-{_slug(outcome).upper()}", "Measurement unit check",
                f"Check measurement units for {outcome}",
                f"MetaPy did not detect an explicit unit column for {outcome} in the uploaded workbook.",
                "For a standardised effect, different studies can sometimes use different units, but the two groups within each study must be on compatible units. Incorrect unit mixing can distort the effect.",
                "Confirm within-study unit consistency, flag studies requiring conversion, or keep this check unresolved.",
                UNIT_OPTIONS, affected_records=records["Preparation record ID"].tolist(), severity="Warning",
            ))

    # The general 'correct extraction' approval is not a specific cell edit.
    eligibility = decisions.get("Study eligibility", {})
    if eligibility.get("status") == "Approved" and eligibility.get("choice") == "correct_extraction":
        items.append(_review_item(
            "PREP-CELL-CORRECTION", "Cell-level correction review",
            "Approved correction pathway has no specific cell changes yet",
            "The researcher approved correcting extraction where needed, but no exact worksheet, row, column, corrected value and supporting source were supplied in this decision.",
            "MetaPy must not guess which value is wrong or overwrite an original value. A general approval is not enough to make a numerical correction.",
            "Confirm that no correction is currently needed, or mark that specific cell-level corrections must be entered before finalising data preparation.",
            CORRECTION_OPTIONS, severity="Warning",
        ))

    # PPC2 records for which the raw point-estimate route is blocked by missing
    # baseline SD information. In the current workbook this identifies the two
    # Study 6 cortisol arms; the post-treatment MD CI is not a baseline SD.
    raw_blocked = prepared[
        prepared["_has_raw_prepost"]
        & ~prepared["_has_baseline_sd_both_groups"]
        & prepared["Outcome label"].eq("Cortisol")
    ]
    if not raw_blocked.empty:
        items.append(_review_item(
            "PREP-PPC2-MISSING-BASELINE-SD", "Missing PPC2 input",
            f"{len(raw_blocked)} cortisol record(s) are missing baseline SD information",
            "These records include pre/post means and sample sizes, but MetaPy could not find baseline SDs for both groups. The reported post-treatment MD confidence interval does not provide those missing baseline SDs.",
            "Morris PPC2 needs the baseline SDs to calculate its point estimate. A pre–post correlation is needed later for its variance, but a correlation cannot replace a missing baseline SD.",
            "Retrieve the missing baseline SDs, check for another valid reported statistic, keep these records pending, or exclude only these records from the primary analysis with a documented reason.",
            MISSING_PPC2_OPTIONS, affected_records=raw_blocked["Preparation record ID"].tolist(), severity="Critical" if primary_choice == "morris_ppc2" else "Warning",
        ))

    # Unverified reported-only PPC2 values are explicitly visible as a review item.
    reported_assumed = prepared[prepared["Reported g method status"].eq("Assumed Morris PPC2 — unverified")]
    if not reported_assumed.empty:
        items.append(_review_item(
            "PREP-REPORTED-G-ASSUMPTION", "Reported effect-size provenance",
            f"{len(reported_assumed)} reported-only Hedges' g value(s) are provisionally labelled PPC2",
            "The approved working assumption classifies these reported Hedges' g values as Morris PPC2, but the calculation method has not been independently verified from the study reports.",
            "They may be retained provisionally for a planned analysis, but the assumption must be visible and tested. The reported effect values should not be described as independently recalculated.",
            "Keep the unverified PPC2 assumption for the planned analysis, keep these values separate until source verification, or mark them pending.",
            [
                {"value": "retain_unverified_ppc2", "label": "Retain as unverified PPC2 values", "explanation": "Keeps the approved working assumption visible in every downstream table."},
                {"value": "separate_until_verified", "label": "Keep separate until the original method is verified", "explanation": "Does not include these reported-only values in the primary PPC2 set yet."},
                {"value": "hold_reported_g", "label": "Keep these records pending", "explanation": "Preserves them but does not mark them ready for the primary analysis."},
            ], affected_records=reported_assumed["Preparation record ID"].tolist(), severity="Warning",
        ))

    # Reported Hedges g may also accompany raw data that are incomplete for the
    # selected route. These are not included in the earlier "reported-only"
    # global assumption because their records do contain some raw measurements.
    unsupported_reported = prepared[
        prepared["_has_reported_g"]
        & prepared["_has_raw_prepost"]
        & ~prepared["_has_baseline_sd_both_groups"]
    ]
    if not unsupported_reported.empty:
        items.append(_review_item(
            "PREP-REPORTED-G-INCOMPLETE-RAW", "Reported effect-size provenance",
            f"{len(unsupported_reported)} reported Hedges g value(s) accompany incomplete raw data",
            "These records contain pre/post means and a reported Hedges g, but the raw baseline SD information needed for a Morris PPC2 recalculation is incomplete. The approved assumption for reported-only records does not automatically cover these mixed raw-plus-reported records.",
            "MetaPy must not assume that a reported effect size is PPC2 merely because other reported-only records were provisionally classified that way.",
            "Check the original study method, keep the reported value separate, or leave the affected records pending.",
            [
                {"value": "confirm_ppc2_from_source", "label": "Original report confirms Morris PPC2", "explanation": "Use this only after checking the source report; the confirmation is researcher-provided."},
                {"value": "separate_reported_value", "label": "Keep reported g separate from the PPC2 raw-data route", "explanation": "The raw record remains available, but the reported g is not assigned to PPC2."},
                {"value": "hold_incomplete_record", "label": "Keep these records pending", "explanation": "No primary-analysis value is selected until the missing information is resolved."},
            ], affected_records=unsupported_reported["Preparation record ID"].tolist(), severity="Warning",
        ))

    return pd.DataFrame(items)


def _build_findings(prepared: pd.DataFrame, review_items: pd.DataFrame,
                    decisions: dict[str, dict[str, Any]], session: dict[str, Any]) -> pd.DataFrame:
    findings: list[dict[str, Any]] = []
    if prepared.empty:
        return pd.DataFrame()

    primary = decisions.get("Primary analysis method", {})
    primary_choice = primary.get("choice", "") if primary.get("status") == "Approved" else ""
    if primary_choice == "ancova_giv" and not prepared["Primary effect route eligible"].any():
        findings.append(_issue(
            "DP-ANCOVA-NO-DATA", "Critical", "Effect-estimand compatibility",
            "ANCOVA selected but compatible adjusted estimates were not found",
            "The uploaded workbook does not show clearly identified ANCOVA-adjusted estimates with SEs or confidence intervals for the canonical records.",
            "The current data cannot be relabelled as ANCOVA. Raw pre/post data and reported PPC2 assumptions are different estimates.",
            "Resolve the primary-method mismatch before the selected primary analysis can proceed.", blocking=True,
        ))

    missing_n = prepared[prepared["N intervention (source)"].isna() | prepared["N control (source)"].isna()]
    if not missing_n.empty:
        findings.append(_issue(
            "DP-MISSING-N", "Error", "Sample size", f"{len(missing_n)} record(s) have missing group sample sizes",
            "MetaPy could not identify an intervention or control sample size for these records.",
            "Sample sizes are needed for effect-size corrections and sampling variances.",
            "Check the source worksheet or keep the affected records pending.", missing_n["Preparation record ID"].tolist(), True,
        ))

    missing_study = prepared[prepared["Study ID"].astype(str).str.strip().isin(["", "nan", "None"])]
    if not missing_study.empty:
        findings.append(_issue(
            "DP-MISSING-STUDY-ID", "Error", "Study identity", f"{len(missing_study)} record(s) have no clear study ID",
            "MetaPy could not identify the study for these records.",
            "Records without a reliable study identity cannot be safely matched, grouped or checked for shared participants.",
            "Check the source study identifier and correct it only against a verified source.",
            missing_study["Preparation record ID"].tolist(), True,
        ))

    # Flag records that contain a reported g and raw-data candidates when the
    # selected primary estimand does not yet have a confirmed source route.
    route_missing = prepared[
        prepared["Reported Hedges g"].notna()
        & prepared["_has_raw_prepost"]
        & ~prepared["_has_baseline_sd_both_groups"]
        & ~prepared["Primary effect route eligible"].astype(bool)
        & ~prepared["_reported_only"]
    ]
    if not route_missing.empty:
        findings.append(_issue(
            "DP-REPORTED-PLUS-RAW-ROUTE", "Warning", "Reported versus raw-data route",
            f"{len(route_missing)} record(s) have both reported g and raw pre/post data, but the selected route is not eligible",
            "These records contain a reported Hedges' g as well as some pre/post data.",
            "The raw-data route may be incomplete or may not match the selected estimand, so the reported effect is not automatically interchangeable.",
            "Review the method and missing raw inputs for each affected record; do not substitute the reported value without confirming its estimand.",
            route_missing["Preparation record ID"].tolist(), True,
        ))

    if not prepared["Outcome data type"].astype(str).str.startswith("Continuous").all():
        unclassified = prepared[~prepared["Outcome data type"].astype(str).str.startswith("Continuous")]
        findings.append(_issue(
            "DP-OUTCOME-UNCLEAR", "Warning", "Outcome classification", f"{len(unclassified)} record(s) have unclear outcome type",
            "MetaPy could not confidently classify all outcomes as continuous measures.",
            "The outcome type determines which effect-size family is appropriate.",
            "Select or correct the outcome type before calculations.", unclassified["Preparation record ID"].tolist(), True,
        ))

    # All records have a project-level design context; only an explicit
    # researcher confirmation resolves the design-level check.
    design_review = review_items[review_items["Review ID"].eq("PREP-DESIGN-CONFIRM")] if not review_items.empty else pd.DataFrame()
    design_confirmed = bool(not design_review.empty and _safe(design_review.iloc[0].get("Status")) == "Approved" and _safe(design_review.iloc[0].get("Your choice")) == "confirm_parallel")
    if _safe(session.get("design")) == "parallel_independent" and not design_confirmed:
        findings.append(_issue(
            "DP-DESIGN-PROJECT-LEVEL", "Warning", "Study design",
            "Parallel independent design is currently a project-level setting",
            "MetaPy applied the selected project design context to the prepared records. The individual study reports were not independently verified by this software.",
            "A crossover, cluster, paired or repeated-measures trial may need a different design-specific calculation.",
            "Confirm the design review item or classify affected studies separately.", blocking=True,
        ))

    if not review_items.empty:
        pending = review_items[(review_items["Status"].astype(str).isin(["Pending", "Deferred"])) & (review_items.get("Dependency state", pd.Series("Active", index=review_items.index)).astype(str) != "Inactive")]
        if not pending.empty:
            findings.append(_issue(
                "DP-RESEARCHER-REVIEW-PENDING", "Warning", "Researcher review",
                f"{len(pending)} data-preparation review item(s) are still pending",
                "MetaPy has identified decisions about outcome direction, measurement units, study design, or source-data handling that have not been resolved.",
                "Some findings may not prevent creating the prepared dataset, but they should be resolved or explicitly deferred before statistical analysis.",
                "Complete or defer each preparation review form.", pending["Review ID"].astype(str).tolist(), True,
            ))

    # Missing PPC2 baseline SD is only a blocking item when PPC2 is the selected
    # primary method. Do not use the post-treatment MD CI as a substitute.
    if primary_choice in {"morris_ppc2", "ancova_giv", "change_smd"}:
        missing_baseline = prepared[prepared["_has_raw_prepost"] & ~prepared["_has_baseline_sd_both_groups"]]
        if not missing_baseline.empty:
            findings.append(_issue(
                "DP-PPC2-BASELINE-SD", "Critical" if primary_choice == "morris_ppc2" else "Warning", "Missing PPC2 input",
                f"{len(missing_baseline)} raw pre/post record(s) lack baseline SDs in both groups",
                "The PPC2 point estimate needs baseline SDs. A pre–post correlation is needed for variance later, but it does not replace baseline SD information.",
                "Without both baseline SDs, these records cannot be recalculated using the raw-data PPC2 route.",
                "Retrieve the missing baseline SDs or keep/exclude the affected records with an explicit researcher decision.",
                missing_baseline["Preparation record ID"].tolist(), primary_choice == "morris_ppc2",
            ))

    if not review_items.empty:
        # Approved selections that still imply an unresolved technical task are
        # retained as findings rather than treated as completed transformations.
        for _, item in review_items.iterrows():
            if _safe(item.get("Status")) != "Approved":
                continue
            review_id = _safe(item.get("Review ID"))
            choice = _safe(item.get("Your choice"))
            if _safe(item.get("Category")) == "Measurement unit check" and choice in {"units_need_harmonisation", "unit_not_reported"}:
                findings.append(_issue(
                    f"DP-{review_id}", "Warning", "Measurement unit", _safe(item.get("Subject")),
                    "The researcher did not confirm compatible units within each study, or indicated that unit conversion is required.",
                    "A within-study unit mismatch can make a treatment-versus-control comparison invalid.",
                    "Check the source reports and apply a documented unit conversion if needed.",
                    _split_semicolon(item.get("Affected record IDs")), True,
                ))
            if _safe(item.get("Category")) == "Outcome direction" and choice == "uncertain":
                findings.append(_issue(
                    f"DP-{review_id}", "Warning", "Outcome direction", _safe(item.get("Subject")),
                    "The researcher has left the outcome direction unresolved.",
                    "MetaPy cannot confidently harmonise the direction of effects across this outcome.",
                    "Check the scale manual or study report before interpreting effect signs.",
                    _split_semicolon(item.get("Affected record IDs")), True,
                ))
            if review_id == "PREP-PPC2-MISSING-BASELINE-SD" and choice in {"retrieve_baseline_sd", "derive_from_valid_statistic"}:
                findings.append(_issue(
                    "DP-PPC2-BASELINE-SD-ACTION", "Warning", "Missing PPC2 input", _safe(item.get("Subject")),
                    "The selected action is to retrieve or derive missing baseline SD information, but no SD values have been entered or verified yet.",
                    "The point estimate cannot be calculated through the raw PPC2 route until both baseline SDs are available.",
                    "Enter the verified baseline SD values through the source correction workflow or keep the affected records pending.",
                    _split_semicolon(item.get("Affected record IDs")), True,
                ))
            if review_id == "PREP-REPORTED-G-INCOMPLETE-RAW" and choice in {"separate_reported_value", "hold_incomplete_record"}:
                findings.append(_issue(
                    "DP-REPORTED-G-INCOMPLETE-RAW", "Warning", "Reported effect-size provenance", _safe(item.get("Subject")),
                    "The reported Hedges g values remain separate from the selected raw-data PPC2 route.",
                    "The reported method is not established by the available workbook fields.",
                    "Check the original study report before using the reported values in the primary analysis.",
                    _split_semicolon(item.get("Affected record IDs")), True,
                ))

    correction_item = review_items[review_items["Review ID"].eq("PREP-CELL-CORRECTION")] if not review_items.empty else pd.DataFrame()
    if not correction_item.empty and _safe(correction_item.iloc[0].get("Status")) == "Approved":
        correction_choice = _safe(correction_item.iloc[0].get("Your choice"))
        if correction_choice in {"cell_corrections_needed", "correction_not_sure"}:
            findings.append(_issue(
                "DP-CELL-CORRECTIONS-UNRESOLVED", "Warning", "Cell-level correction review",
                "Specific extraction corrections remain unresolved",
                "The researcher indicated that source review or cell-level corrections are still needed. No numerical values were changed.",
                "An approved general correction pathway does not identify a corrected value or prove the source value is wrong.",
                "Use the cell correction workflow to specify the exact sheet, row, field, corrected value and source.", blocking=True,
            ))

    # Cascading decisions can resolve a parent item but still leave a dependent action.
    if not review_items.empty:
        for _, item in review_items.iterrows():
            if _safe(item.get("Status")) != "Approved":
                continue
            rid = _safe(item.get("Review ID"))
            choice = _safe(item.get("Your choice"))
            ids = _split_semicolon(item.get("Affected record IDs"))
            if _safe(item.get("Category")) == "Unit verification follow-up" and choice == "conversion_factor_known":
                findings.append(_issue(
                    f"DP-{rid}-FACTOR", "Critical", "Unit harmonisation", _safe(item.get("Subject")),
                    "A conversion factor is said to be available, but no numerical conversion has been entered or applied.",
                    "The approval identifies a next step; it does not perform the conversion.",
                    "Enter the verified conversion factor through the structured conversion workflow before calculations.", ids, True,
                ))
            if _safe(item.get("Category")) == "Unit verification follow-up" and choice == "unit_unresolved":
                findings.append(_issue(
                    f"DP-{rid}-UNRESOLVED", "Critical", "Unit harmonisation", _safe(item.get("Subject")),
                    "Within-study unit compatibility remains unverified.",
                    "Treatment and control values may not be directly comparable.",
                    "Check the original report or keep these records out of the calculation set until resolved.", ids, True,
                ))
            if _safe(item.get("Category")) == "Cortisol sampling context" and choice != "same_sampling_context":
                findings.append(_issue(
                    f"DP-{rid}-TIMING", "Warning", "Cortisol sampling context", _safe(item.get("Subject")),
                    "Cortisol sampling context differs or is not reported.",
                    "Interpretation and comparability of cortisol outcomes may be affected.",
                    "Retain the context flag and assess comparability before interpreting pooled direction.", ids, True,
                ))
            if _safe(item.get("Category")) == "Baseline SD source verification" and choice == "valid_source_statistic_found":
                findings.append(_issue(
                    f"DP-{rid}-VALUE", "Critical", "Baseline SD verification", _safe(item.get("Subject")),
                    "A candidate source statistic was identified, but the actual verified baseline SD value has not been entered.",
                    "The PPC2 point estimate still cannot be calculated from an unentered value.",
                    "Enter the verified value and its source through the cell-correction workflow.", ids, True,
                ))
            if _safe(item.get("Category")) == "Study-specific design" and choice != "parallel_independent":
                findings.append(_issue(
                    f"DP-{rid}-ROUTE", "Critical", "Study design", _safe(item.get("Subject")),
                    "A non-parallel study design has been confirmed.",
                    "The standard independent parallel-group calculation route is not appropriate without design-specific handling.",
                    "Keep this study out of the parallel-group calculation route until a compatible design-specific method is implemented.", ids, True,
                ))

    columns = ["Issue ID", "Severity", "Category", "Subject", "What MetaPy found", "Why it matters", "Recommended next step", "Affected record IDs", "Blocking for primary analysis", "Status"]
    result = pd.DataFrame(findings, columns=columns)
    result = _reconcile_unit_findings(result, review_items)
    result = _reconcile_reported_raw_route_finding(result, prepared)
    return _reconcile_baseline_sd_findings(result, prepared, review_items)


def _reconcile_prepared_reported_routes(prepared: pd.DataFrame) -> pd.DataFrame:
    """Reconcile final route state after all researcher decisions are applied.

    Review rows may be saved in an order where a later baseline-SD action sets
    a record to Held, even though a separate reported PPC2 provenance decision
    subsequently/previously confirms a usable reported point estimate. This
    final pass makes the reported-effect route explicit without pretending the
    raw-data PPC2 calculation or its sampling variance has been completed.
    """
    out = prepared.copy(deep=True)
    if out.empty:
        return out
    required = {
        "Selected primary estimand", "Reported Hedges g", "Reported g method status",
        "Primary effect route eligible", "Primary effect route status",
        "Preparation status", "Independent effect-size calculation status",
        "Sampling variance calculation status", "_has_raw_prepost",
        "_has_baseline_sd_both_groups", "_has_reported_g",
    }
    if not required.issubset(set(out.columns)):
        return out

    primary_ppc2 = out["Selected primary estimand"].fillna("").astype(str).eq("morris_ppc2")
    raw_prepost = out["_has_raw_prepost"].fillna(False).astype(bool)
    baseline_available = out["_has_baseline_sd_both_groups"].fillna(False).astype(bool)
    reported_present = out["_has_reported_g"].fillna(False).astype(bool) & out["Reported Hedges g"].notna()
    provenance_confirmed = out["Reported g method status"].fillna("").astype(str).eq(
        "Researcher confirms PPC2 from original report"
    )
    mask = primary_ppc2 & raw_prepost & ~baseline_available & reported_present & provenance_confirmed

    # Never reverse a separately recorded exclusion or a non-parallel study
    # design decision. Those are independent gates, not consequences of the
    # missing baseline SD.
    if "Researcher disposition" in out.columns:
        disposition = out["Researcher disposition"].fillna("").astype(str)
        mask &= ~disposition.str.contains("Excluded|Held pending shared-control", case=False, regex=True)
    if "Study design" in out.columns:
        design = out["Study design"].fillna("").astype(str)
        mask &= design.str.contains("Parallel independent", case=False, regex=False)
    if "Unit metadata status" in out.columns:
        units = out["Unit metadata status"].fillna("").astype(str)
        # Do not allow a confirmed reported estimate to bypass a known unit
        # mismatch or unresolved within-study unit comparability.
        mask &= ~units.str.contains("conversion required|remains unresolved|not reported|not confirmed", case=False, regex=True)

    if mask.any():
        out.loc[mask, "Primary effect route eligible"] = True
        out.loc[mask, "Primary effect route status"] = (
            "Reported Hedges g confirmed by researcher as Morris PPC2; retained from source, not independently recalculated"
        )
        if "Preparation status" in out.columns:
            out.loc[mask, "Preparation status"] = (
                "Reported PPC2 effect retained; raw-data recalculation unavailable; sampling variance still requires assessment"
            )
        if "Independent effect-size calculation status" in out.columns:
            out.loc[mask, "Independent effect-size calculation status"] = (
                "Blocked — baseline SD missing; reported effect estimate retained"
            )
        if "Reported effect estimate status" in out.columns:
            out.loc[mask, "Reported effect estimate status"] = (
                "Reported Hedges' g — PPC2 method researcher-confirmed, not independently recalculated"
            )
        # Intentionally do not change Sampling variance calculation status.
    return out


def _apply_review_answers(prepared: pd.DataFrame, review_items: pd.DataFrame) -> pd.DataFrame:
    out = prepared.copy(deep=True)
    if review_items is None or review_items.empty:
        return out
    for _, item in review_items.iterrows():
        if _safe(item.get("Status")) != "Approved":
            continue
        choice = _safe(item.get("Your choice"))
        category = _safe(item.get("Category"))
        if category == "Outcome direction":
            ids = _split_semicolon(item.get("Affected record IDs"))
            labels = {
                "higher_more": "Higher values mean more of the measured outcome",
                "higher_less": "Higher values mean less of the measured outcome",
                "context_dependent": "Direction depends on instrument, timing or context",
                "uncertain": "Direction not confirmed",
            }
            out.loc[out["Preparation record ID"].isin(ids), "Outcome direction"] = labels.get(choice, "Needs researcher confirmation")
            multiplier = -1 if choice == "higher_less" else (1 if choice == "higher_more" else None)
            out.loc[out["Preparation record ID"].isin(ids), "Effect direction multiplier"] = multiplier
        elif category == "Study design confirmation":
            label = {
                "confirm_parallel": "Confirmed parallel independent groups by researcher",
                "mixed_designs": "Mixed designs reported — study-specific classification required",
                "design_uncertain": "Design remains unresolved",
            }.get(choice, "Design remains unresolved")
            out["Study design confirmation"] = label
            if choice == "confirm_parallel":
                out["Study design"] = "Parallel independent groups — researcher confirmed"
            elif choice == "mixed_designs":
                out["Study design"] = "Mixed design — study-specific confirmation pending"
            else:
                out["Study design"] = "Study design unresolved"
        elif category == "Measurement unit check":
            ids = _split_semicolon(item.get("Affected record IDs"))
            label = {
                "same_within_study": "Researcher confirmed compatible units within each study",
                "units_need_harmonisation": "Unit conversion required before calculation",
                "unit_not_reported": "Unit not reported / not confirmed",
            }.get(choice, "Not reviewed")
            out.loc[out["Preparation record ID"].isin(ids), "Unit metadata status"] = label
        elif category == "Study-specific design":
            ids = _split_semicolon(item.get("Affected record IDs"))
            design_labels = {
                "parallel_independent": "Parallel independent groups",
                "crossover": "Crossover design",
                "cluster": "Cluster-randomised design",
                "paired_matched": "Paired / matched design",
                "longitudinal_other": "Other longitudinal / repeated-measures design",
                "uncertain": "Study design unresolved",
            }
            label = design_labels.get(choice, "Study design unresolved")
            out.loc[out["Preparation record ID"].isin(ids), "Study design"] = label
            out.loc[out["Preparation record ID"].isin(ids), "Study design confirmation"] = "Study-specific design confirmed by researcher" if choice != "uncertain" else "Study-specific design unresolved"
            if choice != "parallel_independent":
                out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = False
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Requires design-specific calculation route"
        elif category == "Unit verification follow-up":
            ids = _split_semicolon(item.get("Affected record IDs"))
            unit_labels = {
                "same_units_verified": "Researcher verified compatible units within each study",
                "conversion_factor_known": "Conversion factor identified; numeric conversion not yet applied",
                "unit_unresolved": "Unit compatibility remains unresolved",
            }
            out.loc[out["Preparation record ID"].isin(ids), "Unit metadata status"] = unit_labels.get(choice, "Unit review pending")
            if choice != "same_units_verified":
                out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = False
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Awaiting verified unit harmonisation"
        elif category == "Cortisol sampling context":
            ids = _split_semicolon(item.get("Affected record IDs"))
            timing_labels = {
                "same_sampling_context": "Sampling context confirmed comparable by researcher",
                "different_sampling_context": "Sampling context differs; interpretation requires review",
                "timing_not_reported": "Sampling context not reported",
            }
            out.loc[out["Preparation record ID"].isin(ids), "Cortisol sampling context status"] = timing_labels.get(choice, "Sampling context pending")
        elif category == "Baseline SD source verification":
            ids = _split_semicolon(item.get("Affected record IDs"))
            if choice == "valid_source_statistic_found":
                out.loc[out["Preparation record ID"].isin(ids), "Missing input action"] = "Candidate source statistic identified; verified numeric SD entry still required"
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Awaiting verified baseline SD value entry"
            else:
                out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = False
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Held pending valid baseline SD information"
        elif category == "Cell-level correction review":
            label = {
                "no_correction_needed": "Researcher reviewed — no correction required",
                "cell_corrections_needed": "Specific cell corrections still required; no values changed",
                "correction_not_sure": "Source review pending; no values changed",
            }.get(choice, "Not reviewed")
            out["Extraction correction status"] = label
        elif category == "Missing PPC2 input":
            ids = _split_semicolon(item.get("Affected record IDs"))
            label = {
                "retrieve_baseline_sd": "Retrieve missing baseline SDs",
                "derive_from_valid_statistic": "Check a valid alternative statistic; no derivation applied yet",
                "hold_missing_sd": "Held pending missing baseline SD",
                "exclude_missing_sd": "Excluded from selected primary analysis by researcher decision",
            }.get(choice, "Not reviewed")
            out.loc[out["Preparation record ID"].isin(ids), "Missing input action"] = label
            if choice == "retrieve_baseline_sd":
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Awaiting verified baseline SD extraction"
            elif choice == "derive_from_valid_statistic":
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Awaiting valid derivation source; no SD derived"
            elif choice == "hold_missing_sd":
                out.loc[out["Preparation record ID"].isin(ids), "Researcher disposition"] = "Held pending missing baseline SD"
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Held pending missing baseline SD"
                out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = False
            elif choice == "exclude_missing_sd":
                out.loc[out["Preparation record ID"].isin(ids), "Researcher disposition"] = "Excluded from selected primary analysis"
                out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Excluded from selected primary analysis"
                out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = False
        elif category == "Reported effect-size provenance":
            ids = _split_semicolon(item.get("Affected record IDs"))
            if _safe(item.get("Review ID")) == "PREP-REPORTED-G-INCOMPLETE-RAW":
                label = {
                    "confirm_ppc2_from_source": "Researcher confirms PPC2 from original report",
                    "separate_reported_value": "Reported g kept separate from PPC2 raw route",
                    "hold_incomplete_record": "Held pending method/source verification",
                }.get(choice, "Not reviewed")
                if choice != "confirm_ppc2_from_source":
                    out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = False
                    out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Pending reported effect-size method verification"
                else:
                    out.loc[out["Preparation record ID"].isin(ids), "Reported g method status"] = label
                    if out.loc[out["Preparation record ID"].isin(ids), "Selected primary estimand"].eq("morris_ppc2").all():
                        out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = True
                        out.loc[out["Preparation record ID"].isin(ids), "Primary effect route status"] = "Reported Hedges g confirmed by researcher as Morris PPC2; retained from source, not independently recalculated"
                        out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Eligible for later calculation/variance review using reported estimate"
            else:
                label = {
                    "retain_unverified_ppc2": "Retain as unverified PPC2",
                    "separate_until_verified": "Separate until source verification",
                    "hold_reported_g": "Held pending source verification",
                }.get(choice, "Not reviewed")
                out.loc[out["Preparation record ID"].isin(ids), "Reported g method status"] = label
                if choice in {"separate_until_verified", "hold_reported_g"}:
                    out.loc[out["Preparation record ID"].isin(ids), "Primary effect route eligible"] = False
                    out.loc[out["Preparation record ID"].isin(ids), "Preparation status"] = "Pending reported effect-size method verification"
    # Resolve cross-decision ordering after every approved response has been
    # applied. The reported PPC2 point-estimate route may be retained even when
    # raw-data recalculation is blocked by missing baseline SDs; variance stays
    # explicitly pending.
    return _reconcile_prepared_reported_routes(out)


def _build_transform_log(prepared: pd.DataFrame, decisions: dict[str, dict[str, Any]],
                         review_items: pd.DataFrame,
                         correction_history: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    changes: list[dict[str, Any]] = []
    def add(action: str, count: int, reason: str, decision: str, nature: str = "Derived metadata"):
        if count:
            changes.append({
                "Transformation ID": f"TR-{len(changes)+1:03d}",
                "Transformation": action,
                "Records affected": int(count),
                "Reason": reason,
                "Decision source": decision,
                "Transformation type": nature,
                "Original source values overwritten": "No",
                "Timestamp (UTC)": datetime.now(timezone.utc).isoformat(),
            })

    add("Created stable preparation record IDs", len(prepared), "One ID per canonical study/arm/outcome/duration record.", "MetaPy deterministic rule")
    add("Classified outcomes into broad outcome groups", int(prepared["Outcome label"].nunique()), "Used the detected outcome/instrument labels; original outcome keys retained.", "MetaPy rule-based suggestion")
    add("Added four-dimensional data classification", len(prepared), "Outcome type, reporting format, study design and analysis structure are added as derived fields.", "MetaPy rule-based suggestion")
    multi = decisions.get("Multi-arm study dependence", {})
    split_rows = int((prepared["Shared-control comparison count"] > 1).sum()) if not prepared.empty else 0
    if multi.get("status") == "Approved" and multi.get("choice") == "split_control":
        add("Prepared split-control sample sizes for sample-size-based pairwise variance routes", split_rows, "Original control Ns are preserved. A separate N is divided for future formulas that use group sample sizes; this split is not applied to a direct ANCOVA generic-inverse-variance SE/CI.", multi.get("decision_id", "Researcher decision"), "Approved analysis preparation rule")
    reported = decisions.get("Reported effect-size method", {})
    assumed_count = int(prepared["Reported g method status"].eq("Assumed Morris PPC2 — unverified").sum()) if not prepared.empty else 0
    if reported.get("status") == "Approved" and reported.get("choice") == "assume_ppc2_unverified":
        add("Marked reported-only Hedges g values as unverified PPC2 assumptions", assumed_count, "Method label is a researcher-approved working assumption; no external verification was performed.", reported.get("decision_id", "Researcher decision"), "Approved classification assumption")
    if not review_items.empty:
        done = review_items[review_items["Status"].eq("Approved")]
        if not done.empty:
            add("Applied approved data-preparation review selections", len(done), "Approved direction, design, unit and record-review selections were added to the prepared copy.", "Preparation review forms", "Researcher-approved metadata update")
    for correction in correction_history or []:
        changes.append({
            "Transformation ID": f"TR-{len(changes)+1:03d}",
            "Transformation": f"Corrected {correction.get('Worksheet')}!{correction.get('Column')} at Excel row {correction.get('Excel row')}",
            "Records affected": 1,
            "Reason": f"{correction.get('Reason')}: {correction.get('Previous value')} → {correction.get('Corrected value')}",
            "Decision source": f"Researcher confirmed against {correction.get('Evidence source')}",
            "Transformation type": "Verified cell correction in separate working copy",
            "Original source values overwritten": "No — original upload retained unchanged",
            "Timestamp (UTC)": correction.get("Timestamp (UTC)", ""),
        })
    return pd.DataFrame(changes)


def _build_classification_table(prepared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions = [
        ("1. Outcome data type", "Outcome data type"),
        ("2. Reporting format", "Reporting format"),
        ("3. Study design", "Study design"),
        ("4. Analysis structure", "Analysis structure"),
    ]
    for dimension_label, column in dimensions:
        if column not in prepared:
            continue
        counts = prepared[column].fillna("Not classified").astype(str).value_counts(dropna=False)
        for value, count in counts.items():
            rows.append({"Classification dimension": dimension_label, "Detected category": value, "Record count": int(count)})
    return pd.DataFrame(rows)


def _make_summary(prepared: pd.DataFrame, findings: pd.DataFrame, review_items: pd.DataFrame,
                  decisions: dict[str, dict[str, Any]], session: dict[str, Any]) -> dict[str, Any]:
    primary = decisions.get("Primary analysis method", {})
    primary_choice = primary.get("choice", "") if primary.get("status") == "Approved" else None
    eligible = int(prepared["Primary effect route eligible"].fillna(False).astype(bool).sum()) if not prepared.empty else 0
    reported_counts = _reported_effect_counts(prepared)
    reported_assumed = int(reported_counts["reported_g_assumed_ppc2_unverified"])
    split_rows = int((prepared["Shared-control comparison count"] > 1).sum()) if not prepared.empty else 0
    review_pending = int(review_items[(~review_items.get("Dependency state", pd.Series("Active", index=review_items.index)).astype(str).str.startswith("Inactive"))]["Status"].astype(str).isin(["Pending", "Deferred"]).sum()) if not review_items.empty else 0
    review_approved = int((review_items["Status"].astype(str) == "Approved").sum()) if not review_items.empty else 0
    blockers = int(findings.get("Blocking for primary analysis", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not findings.empty else 0
    return {
        "preparation_engine_version": VERSION,
        "source_filename": _safe(session.get("source_filename")),
        "original_workbook_modified": False,
        "approved_cell_corrections_applied": len(session.get("correction_history", [])),
        "worksheet_count": len(session.get("sheets", {})),
        "canonical_records": len(prepared),
        "outcome_groups": int(prepared["Outcome label"].nunique()) if not prepared.empty else 0,
        "primary_estimand_approved": primary_choice,
        "records_with_primary_route_candidate": eligible,
        **reported_counts,
        "reported_g_assumed_ppc2_unverified": reported_assumed,
        "shared_control_records_prepared_for_variance_weighting": split_rows,
        "preparation_review_items": int((~review_items.get("Dependency state", pd.Series("Active", index=review_items.index)).astype(str).str.startswith("Inactive")).sum()) if not review_items.empty else 0,
        "dynamic_dependent_review_items": int(review_items.get("Is dynamic", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not review_items.empty else 0,
        "preparation_reviews_approved": review_approved,
        "preparation_reviews_pending": review_pending,
        "record_level_signoffs_required": int(len(prepared)),
        "record_level_signoffs_completed": int((prepared.get("Individual record approval", pd.Series(dtype=str)).astype(str) == "Approved").sum()) if not prepared.empty else 0,
        "record_level_signoffs_pending": int((prepared.get("Individual record approval", pd.Series(dtype=str)).astype(str) != "Approved").sum()) if not prepared.empty else 0,
        "blocking_findings": blockers,
        "data_preparation_status": (
            "Needs project-level researcher review" if review_pending or blockers else
            ("Needs individual record sign-off" if not prepared.empty and (prepared.get("Individual record approval", pd.Series(dtype=str)).astype(str) != "Approved").any()
             else "Prepared — final checks passed")
        ),
        "not_yet_performed": ["Final effect-size execution", "Sampling-variance calculation", "Meta-analysis pooling"],
        "scope_note": "Workbook-internal data preparation only. No external source verification or statistical pooling was performed.",
    }


def build_preparation_bundle(session: dict[str, Any], decision_queue: pd.DataFrame,
                             previous_state: str | dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a preparation copy, classification, issue log and review queue."""
    if not isinstance(session, dict) or not isinstance(session.get("review"), dict):
        raise ValueError("Inspect an Excel workbook in MetaPy before starting data preparation.")
    if decision_queue is None or decision_queue.empty:
        raise ValueError("No researcher decision queue was found. Inspect the workbook and save decisions first.")

    decisions = _decision_map(decision_queue)
    prepared, originals = _build_prepared_records(session, decision_queue, decisions)
    review_items = _build_review_items(session, prepared, decisions)

    # Preserve existing base review answers before generating cascading tasks.
    previous = bundle_from_state(previous_state) if previous_state else None
    previous_items = previous.get("review_items") if previous else None
    if isinstance(previous_items, pd.DataFrame) and not previous_items.empty:
        old = previous_items.set_index("Review ID", drop=False)
        for idx, row in review_items.iterrows():
            review_id = _safe(row.get("Review ID"))
            if review_id in old.index:
                oldrow = old.loc[review_id]
                review_items.at[idx, "Your choice"] = _safe(oldrow.get("Your choice"))
                review_items.at[idx, "Status"] = _safe(oldrow.get("Status")) or "Pending"
                review_items.at[idx, "Decision recorded (UTC)"] = _safe(oldrow.get("Decision recorded (UTC)"))

    review_items = reconcile_dependency_reviews(
        review_items, prepared, previous_items=previous_items, decision_map=decisions
    )
    prepared = _apply_review_answers(prepared, review_items)
    findings = _build_findings(prepared, review_items, decisions, session)

    # Add workbook-level validation issues from the original review and the
    # harmonizer; never silently overwrite source values to resolve them.
    for source_label, frame in (("Workbook validation", session.get("review", {}).get("issues")),
                                ("Harmonization", session.get("review", {}).get("harmonization", {}).get("issues"))):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for idx, row in frame.iterrows():
                sev = _safe(row.get("severity")) or _safe(row.get("Severity")) or "Warning"
                findings.loc[len(findings)] = {
                    "Issue ID": f"{source_label[:3].upper()}-{idx+1:03d}",
                    "Severity": sev,
                    "Category": source_label,
                    "Subject": _safe(row.get("finding")) or _safe(row.get("title")) or _safe(row.get("code")) or "Source validation finding",
                    "What MetaPy found": _safe(row.get("what_it_means")) or _safe(row.get("explanation")) or _safe(row.get("message")),
                    "Why it matters": _safe(row.get("analysis impact")) or _safe(row.get("impact")),
                    "Recommended next step": _safe(row.get("recommended_action")) or "Review the source workbook and retain the original value until resolved.",
                    "Affected record IDs": "",
                    "Blocking for primary analysis": sev.lower() in {"critical", "error"},
                    "Status": "Open",
                }

    transformations = _build_transform_log(prepared, decisions, review_items, session.get("correction_history", []))
    classification = _build_classification_table(prepared)
    summary = _make_summary(prepared, findings, review_items, decisions, session)
    summary = _refresh_summary_statuses(summary, prepared, review_items)
    return {
        "summary": summary,
        "prepared_records": prepared,
        "canonical_source_records": originals.copy(deep=True),
        "classification": classification,
        "findings": findings,
        "review_items": review_items,
        "dependency_map": dependency_impact_table(review_items),
        "transformations": transformations,
        "decision_map": decisions,
        "source_filename": _safe(session.get("source_filename")),
        "correction_history": list(session.get("correction_history", [])),
        "session_design": _safe(session.get("design")),
        "working_path": _safe(session.get("working_path")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_preparation_review(bundle: dict[str, Any], review_id: str,
                             choice: Any, status: Any) -> dict[str, Any]:
    """Save one structured preparation decision without changing other items."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("review_items"), pd.DataFrame):
        raise ValueError("Build the prepared dataset before saving a preparation review.")
    review_id = _safe(review_id)
    choice = _safe(choice)
    status = _safe(status) or "Pending"
    previous_items = bundle["review_items"].copy(deep=True)
    items = previous_items.copy(deep=True)
    matches = items.index[items["Review ID"].astype(str).eq(review_id)].tolist()
    if not matches:
        raise ValueError("The selected preparation review item was not found.")
    if status not in REVIEW_STATUSES:
        raise ValueError("Choose Pending, Approved, Rejected or Deferred.")
    idx = matches[0]
    try:
        options = json.loads(_safe(items.at[idx, "Available options"]) or "[]")
    except Exception:
        options = []
    valid = {str(o.get("value")) for o in options if isinstance(o, dict)}
    if choice and choice not in valid:
        raise ValueError("Select one of the options shown for this preparation item.")
    if status == "Approved" and not choice:
        raise ValueError("Select an option before approving this preparation item.")
    items.at[idx, "Your choice"] = choice
    items.at[idx, "Status"] = status
    items.at[idx, "Decision recorded (UTC)"] = datetime.now(timezone.utc).isoformat()

    updated = dict(bundle)
    items = reconcile_dependency_reviews(items, bundle["prepared_records"], previous_items=previous_items, decision_map=bundle.get("decision_map", {}), changed_review_id=review_id)
    updated["review_items"] = items
    updated["dependency_map"] = dependency_impact_table(items)
    updated["prepared_records"] = _apply_review_answers(bundle["prepared_records"], items)
    updated["transformations"] = _build_transform_log(updated["prepared_records"], updated.get("decision_map", {}), items, updated.get("correction_history", []))
    # Record-level review choices are metadata only. A request to reconsider an
    # estimand must be confirmed again in the original researcher decision form.
    findings = updated.get("findings", pd.DataFrame()).copy(deep=True)
    if not findings.empty and status not in {"Pending", "Deferred"}:
        stale_validation = f"VAL-{review_id}"
        findings = findings[findings["Issue ID"].astype(str).ne(stale_validation)].copy()
    if not findings.empty:
        subject = _safe(items.at[idx, "Subject"])
        matches_f = findings.index[findings["Subject"].astype(str).eq(subject)].tolist()
        if matches_f:
            findings.at[matches_f[0], "Status"] = f"Reviewed: {choice}" if status == "Approved" else status
    # Reconcile the main dynamic findings with the new saved selection.
    pending_count = int(items[(~items.get("Dependency state", pd.Series("Active", index=items.index)).astype(str).str.startswith("Inactive"))]["Status"].astype(str).isin(["Pending", "Deferred"]).sum())
    pending_rows = findings.index[findings["Issue ID"].eq("DP-RESEARCHER-REVIEW-PENDING")].tolist() if not findings.empty else []
    if pending_rows:
        fidx = pending_rows[0]
        if pending_count == 0:
            findings.at[fidx, "Subject"] = "All active data-preparation review items have a recorded decision"
            findings.at[fidx, "What MetaPy found"] = "Every active preparation review item has an approved, rejected or explicitly deferred researcher disposition."
            findings.at[fidx, "Recommended next step"] = "Continue with final preparation validation. Any remaining technical findings are listed separately."
            findings.at[fidx, "Status"] = "Resolved — all preparation review forms have a recorded choice"
            findings.at[fidx, "Blocking for primary analysis"] = False
        else:
            findings.at[fidx, "Subject"] = f"{pending_count} data-preparation review item(s) remain pending or deferred"
            findings.at[fidx, "What MetaPy found"] = "Some active preparation review items still have no final researcher decision."
            findings.at[fidx, "Recommended next step"] = "Complete or explicitly defer each pending preparation review form."
            findings.at[fidx, "Status"] = "Open"
            findings.at[fidx, "Blocking for primary analysis"] = True

    if review_id == "PREP-DESIGN-CONFIRM" and status == "Approved" and choice == "confirm_parallel":
        rows = findings.index[findings["Issue ID"].eq("DP-DESIGN-PROJECT-LEVEL")].tolist() if not findings.empty else []
        if rows:
            findings.at[rows[0], "Status"] = "Resolved — researcher confirmed parallel independent-group design"
            findings.at[rows[0], "Blocking for primary analysis"] = False

    if review_id == "PREP-PPC2-MISSING-BASELINE-SD" and status == "Approved" and choice in {"hold_missing_sd", "exclude_missing_sd"}:
        rows = findings.index[findings["Issue ID"].eq("DP-PPC2-BASELINE-SD")].tolist() if not findings.empty else []
        if rows:
            findings.at[rows[0], "Status"] = f"Resolved for current primary set — affected records {choice}"
            findings.at[rows[0], "Blocking for primary analysis"] = False

    if review_id == "PREP-CELL-CORRECTION" and status == "Approved" and choice == "no_correction_needed":
        rows = findings.index[findings["Issue ID"].eq("DP-CELL-CORRECTIONS-UNRESOLVED")].tolist() if not findings.empty else []
        if rows:
            findings = findings.drop(index=rows)

    # Normalize the index after possible row removals so .loc[len(findings)]
    # cannot overwrite an unrelated finding when adding a new issue.
    findings = findings.reset_index(drop=True)

    # Add explicit unresolved findings for approved choices that require more data.
    if status == "Approved" and choice == "units_need_harmonisation":
        findings.loc[len(findings)] = {
            "Issue ID": f"DP-{review_id}-CONVERSION", "Severity": "Warning", "Category": "Measurement unit",
            "Subject": "Unit conversion is still required", "What MetaPy found": "The researcher indicated that units differ within at least some records.",
            "Why it matters": "A numerical conversion has not been supplied or applied.",
            "Recommended next step": "Provide a documented conversion factor and source before calculating the affected effect sizes.",
            "Affected record IDs": _safe(items.at[idx, "Affected record IDs"]), "Blocking for primary analysis": True, "Status": "Open",
        }
    if status == "Approved" and choice == "unit_not_reported":
        findings.loc[len(findings)] = {
            "Issue ID": f"DP-{review_id}-UNIT-UNKNOWN", "Severity": "Warning", "Category": "Measurement unit",
            "Subject": "Measurement unit remains unconfirmed", "What MetaPy found": "The researcher could not confirm whether the two groups in each study use compatible units.",
            "Why it matters": "A within-study unit mismatch could distort a comparison.",
            "Recommended next step": "Review the source report or leave affected records out of the calculation until clarified.",
            "Affected record IDs": _safe(items.at[idx, "Affected record IDs"]), "Blocking for primary analysis": True, "Status": "Open",
        }
    if status == "Approved" and review_id.startswith("PREP-DIRECTION-") and choice == "uncertain":
        findings.loc[len(findings)] = {
            "Issue ID": f"DP-{review_id}-UNCERTAIN", "Severity": "Warning", "Category": "Outcome direction",
            "Subject": "Outcome direction remains unresolved", "What MetaPy found": "The researcher marked the direction as uncertain.",
            "Why it matters": "MetaPy cannot harmonise the sign of effects across this outcome.",
            "Recommended next step": "Check the instrument manual or original report before interpreting effect signs.",
            "Affected record IDs": _safe(items.at[idx, "Affected record IDs"]), "Blocking for primary analysis": True, "Status": "Open",
        }

    findings = _reconcile_unit_findings(findings, items)
    findings = _reconcile_reported_raw_route_finding(findings, updated["prepared_records"])
    findings = _reconcile_baseline_sd_findings(findings, updated["prepared_records"], items)
    updated["findings"] = findings
    updated["classification"] = _build_classification_table(updated["prepared_records"])
    updated["transformations"] = _build_transform_log(updated["prepared_records"], updated.get("decision_map", {}), items, updated.get("correction_history", []))
    updated["created_utc"] = datetime.now(timezone.utc).isoformat()
    summary = dict(updated.get("summary", {}))
    summary["preparation_reviews_pending"] = pending_count
    summary["preparation_reviews_approved"] = int((items["Status"].astype(str) == "Approved").sum())
    summary["blocking_findings"] = int(findings.get("Blocking for primary analysis", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not findings.empty else 0
    summary["data_preparation_status"] = "Needs researcher review" if pending_count or summary["blocking_findings"] else "Prepared — review choices recorded; run validation"
    summary = _refresh_summary_statuses(summary, updated["prepared_records"], items)
    updated["summary"] = summary
    return updated


def validate_preparation_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Re-check record identities, review decisions and preparation blockers."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("prepared_records"), pd.DataFrame):
        raise ValueError("Build the prepared dataset before running preparation validation.")
    findings = bundle.get("findings", pd.DataFrame()).copy(deep=True)
    items = bundle.get("review_items", pd.DataFrame())
    prepared = bundle["prepared_records"]

    existing_issue_ids = set(findings.get("Issue ID", pd.Series(dtype=str)).astype(str)) if not findings.empty else set()
    if isinstance(items, pd.DataFrame) and not items.empty:
        pending = items[(items["Status"].astype(str).isin(["Pending", "Deferred"])) & (~items.get("Dependency state", pd.Series("Active", index=items.index)).astype(str).str.startswith("Inactive"))]
        for _, row in pending.iterrows():
            rid = _safe(row.get("Review ID"))
            issue_id = f"VAL-{rid}"
            if issue_id not in existing_issue_ids:
                findings.loc[len(findings)] = {
                    "Issue ID": issue_id, "Severity": "Warning", "Category": "Preparation review",
                    "Subject": _safe(row.get("Subject")),
                    "What MetaPy found": f"This review item is {row.get('Status')}; no action has been applied.",
                    "Why it matters": "MetaPy must not infer a researcher decision that has not been recorded.",
                    "Recommended next step": "Approve a structured option or explicitly defer the issue.",
                    "Affected record IDs": _safe(row.get("Affected record IDs")),
                    "Blocking for primary analysis": True, "Status": _safe(row.get("Status")),
                }
                existing_issue_ids.add(issue_id)

    if prepared["Preparation record ID"].duplicated().any() and "VAL-DUPLICATE-PREP-ID" not in existing_issue_ids:
        dup = prepared.loc[prepared["Preparation record ID"].duplicated(), "Preparation record ID"].astype(str).tolist()
        findings.loc[len(findings)] = {
            "Issue ID": "VAL-DUPLICATE-PREP-ID", "Severity": "Error", "Category": "Record identity",
            "Subject": "Duplicate preparation record IDs", "What MetaPy found": "; ".join(dup),
            "Why it matters": "Each canonical analysis record must have a stable unique identifier.",
            "Recommended next step": "Rebuild the preparation dataset and inspect the source matching logic.",
            "Affected record IDs": "; ".join(dup), "Blocking for primary analysis": True, "Status": "Open",
        }

    if "N control for sample-size-based variance routes" in prepared and "VAL-BAD-SPLIT-N" not in existing_issue_ids:
        bad = prepared[prepared["N control for sample-size-based variance routes"].apply(lambda x: (_number(x) is not None and _number(x) <= 0))]
        if not bad.empty:
            findings.loc[len(findings)] = {
                "Issue ID": "VAL-BAD-SPLIT-N", "Severity": "Error", "Category": "Shared-control preparation",
                "Subject": "Invalid split control sample size", "What MetaPy found": f"{len(bad)} record(s) have non-positive adjusted control sample sizes.",
                "Why it matters": "The later variance calculation needs a positive effective sample size.",
                "Recommended next step": "Recheck the number of comparisons sharing each control group.",
                "Affected record IDs": "; ".join(bad["Preparation record ID"].astype(str)),
                "Blocking for primary analysis": True, "Status": "Open",
            }

    findings = _reconcile_unit_findings(findings, items)
    findings = _reconcile_reported_raw_route_finding(findings, prepared)
    findings = _reconcile_baseline_sd_findings(findings, prepared, items)
    summary = dict(bundle.get("summary", {}))
    pending_count = int((items["Status"].astype(str).isin(["Pending", "Deferred"]) & (~items.get("Dependency state", pd.Series("Active", index=items.index)).astype(str).str.startswith("Inactive"))).sum()) if isinstance(items, pd.DataFrame) and not items.empty else 0
    approved_count = int((items["Status"].astype(str) == "Approved").sum()) if isinstance(items, pd.DataFrame) and not items.empty else 0
    blocking = int(findings.get("Blocking for primary analysis", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not findings.empty else 0
    error_count = int(findings.get("Severity", pd.Series(dtype=str)).astype(str).str.lower().isin(["critical", "error"]).sum()) if not findings.empty else 0
    summary.update({
        "preparation_reviews_pending": pending_count,
        "preparation_reviews_approved": approved_count,
        "blocking_findings": blocking,
        "critical_or_error_findings": error_count,
        "data_preparation_status": "Needs researcher review" if pending_count or blocking or error_count else "Prepared — final checks passed",
        "final_preparation_validation_utc": datetime.now(timezone.utc).isoformat(),
    })
    summary = _refresh_summary_statuses(summary, prepared, items)
    validated_bundle = dict(bundle)
    validated_bundle["findings"] = findings
    validated_bundle["summary"] = summary
    validated_bundle["dependency_map"] = dependency_impact_table(validated_bundle.get("review_items", pd.DataFrame()))
    return validated_bundle, findings


def correction_source_options(session: dict[str, Any]) -> dict[str, list[Any]]:
    """Return safe worksheet, row and column choices for the structured correction form."""
    sheets = session.get("sheets", {}) if isinstance(session, dict) else {}
    result: dict[str, list[Any]] = {"sheets": [], "rows": [], "columns": []}
    if not isinstance(sheets, dict):
        return result
    for name, frame in sheets.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            result["sheets"].append(str(name))
    return result


def correction_row_column_options(session: dict[str, Any], sheet_name: str) -> dict[str, list[Any]]:
    """Return actual Excel row numbers (header is row 1) and exact column names."""
    sheets = session.get("sheets", {}) if isinstance(session, dict) else {}
    frame = sheets.get(sheet_name) if isinstance(sheets, dict) else None
    if not isinstance(frame, pd.DataFrame):
        return {"rows": [], "columns": []}
    return {
        "rows": [(f"Excel row {i + 2}", i + 2) for i in range(len(frame))],
        "columns": [(str(c), str(c)) for c in frame.columns],
    }


def _coerce_correction_value(old_value: Any, new_value: Any, column: pd.Series) -> Any:
    """Coerce a verified correction and enforce basic column-specific rules."""
    text = _safe(new_value)
    if not text:
        raise ValueError("Enter a replacement cell value. Blank corrections are not allowed in this form.")
    if text.startswith("="):
        raise ValueError("Excel formulas are not accepted as correction values. Enter the verified value itself.")
    numeric_column = pd.api.types.is_numeric_dtype(column) or _number(old_value) is not None
    if numeric_column:
        number = _number(text)
        if number is None:
            raise ValueError("This column contains numeric values. Enter a valid number.")
        name = re.sub(r"[^a-z0-9]+", "_", str(column.name).strip().lower()).strip("_")
        compact = name.replace("_", "")
        is_sample_size = (
            compact in {"n", "ni", "nc", "n1", "n2", "nintervention", "ncontrol", "nexperimental", "ncomparator", "nparticipants"}
            or name.startswith("n_")
            or "sample_size" in name
            or "participants_n" in name
        )
        is_sd = name.startswith("sd_") or compact.startswith("sd") or "standard_deviation" in name
        if is_sample_size:
            if number <= 0 or not float(number).is_integer():
                raise ValueError("Sample size must be a positive whole number. Check the source before correcting it.")
            return int(number)
        if is_sd and number <= 0:
            raise ValueError("A standard deviation must be greater than zero. Enter a source-verified positive value.")
        # Preserve integer cells when a whole number is entered.
        if float(number).is_integer() and (pd.api.types.is_integer_dtype(column) or isinstance(old_value, int)):
            return int(number)
        return float(number)
    return text


def apply_approved_cell_correction(
    session: dict[str, Any], decision_queue: pd.DataFrame, preparation_bundle: dict[str, Any],
    sheet_name: str, excel_row: Any, column_name: str, new_value: Any,
    reason: str, evidence_source: str, source_checked: bool,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Apply one explicitly approved cell correction to the working copy, then re-inspect.

    The original upload is never written. All upstream decision approvals are reset
    to Pending after the workbook is re-analysed, so the researcher must reconfirm them.
    """
    if not isinstance(session, dict) or not session.get("working_path"):
        raise ValueError("Inspect the workbook first. No separate MetaPy working copy is available.")
    if not isinstance(preparation_bundle, dict) or not isinstance(preparation_bundle.get("review_items"), pd.DataFrame):
        raise ValueError("Build the prepared dataset first.")
    review_items = preparation_bundle["review_items"]
    matches = review_items.index[review_items["Review ID"].astype(str).eq("PREP-CELL-CORRECTION")].tolist()
    if not matches:
        raise ValueError("No cell-correction review item is active for this workbook.")
    review = review_items.loc[matches[0]]
    if _safe(review.get("Status")) != "Approved" or _safe(review.get("Your choice")) != "cell_corrections_needed":
        raise ValueError("First approve 'Specific cell-level corrections are needed' in the preparation review form.")
    if not source_checked:
        raise ValueError("Confirm that you checked the replacement value against a source before applying it.")
    allowed_reasons = {
        "transcription_error", "source_report_clarification", "sample_size_correction",
        "outcome_or_arm_label", "other_verified_correction"
    }
    allowed_sources = {
        "original_publication", "supplementary_material", "author_correspondence",
        "researcher_verified_extraction"
    }
    if reason not in allowed_reasons:
        raise ValueError("Choose a correction reason from the list.")
    if evidence_source not in allowed_sources:
        raise ValueError("Choose the source used to verify the correction.")
    sheets = session.get("sheets", {})
    if sheet_name not in sheets or not isinstance(sheets[sheet_name], pd.DataFrame):
        raise ValueError("Choose a valid worksheet.")
    frame = sheets[sheet_name]
    try:
        row_num = int(excel_row)
    except (TypeError, ValueError):
        raise ValueError("Choose a valid Excel row number.")
    if row_num < 2 or row_num > len(frame) + 1:
        raise ValueError("Choose a data row from the selected worksheet. Do not select the header row.")
    if column_name not in frame.columns:
        raise ValueError("Choose a valid column from the selected worksheet.")
    df_idx = row_num - 2
    old_value = frame.iloc[df_idx][column_name]
    corrected_value = _coerce_correction_value(old_value, new_value, frame[column_name])
    if _safe(old_value) == _safe(corrected_value):
        raise ValueError("The replacement value is identical to the existing value. No correction is needed.")

    working_path = Path(session["working_path"])
    if not working_path.exists():
        raise FileNotFoundError("The MetaPy working copy is missing. Please re-inspect the original workbook.")
    wb = load_workbook(working_path)
    try:
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"Worksheet '{sheet_name}' is no longer present in the working copy.")
        ws = wb[sheet_name]
        col_num = list(frame.columns).index(column_name) + 1
        cell = ws.cell(row=row_num, column=col_num)
        workbook_old = cell.value
        # Ensure the in-memory DataFrame and working-copy cell have not drifted.
        old_num = _number(old_value)
        workbook_num = _number(workbook_old)
        same_source_value = (
            math.isclose(old_num, workbook_num, rel_tol=1e-12, abs_tol=1e-12)
            if old_num is not None and workbook_num is not None
            else _safe(workbook_old) == _safe(old_value)
        )
        if not same_source_value:
            raise ValueError("The working copy differs from the inspected worksheet. Re-inspect before correcting.")
        cell.value = corrected_value
        wb.save(working_path)
    finally:
        wb.close()

    from ..review.single_workbook_workflow import _make_session_from_upload
    from ..review.researcher_decision_engine import build_decision_queue
    new_session = _make_session_from_upload(
        str(working_path), "auto", _safe(session.get("design")) or "parallel_independent",
        compare_reported_g=bool(session.get("compare_reported_g", False)),
    )
    new_session["source_filename"] = _safe(session.get("source_filename")) or working_path.name
    history = list(session.get("correction_history", []))
    entry = {
        "Correction ID": f"COR-{len(history) + 1:03d}",
        "Timestamp (UTC)": datetime.now(timezone.utc).isoformat(),
        "Worksheet": sheet_name,
        "Excel row": row_num,
        "Column": str(column_name),
        "Previous value": _safe(old_value),
        "Corrected value": str(corrected_value),
        "Reason": reason,
        "Evidence source": evidence_source,
        "Researcher verified source": True,
        "Original workbook modified": False,
        "Downstream decisions reset": True,
    }
    new_session["correction_history"] = history + [entry]

    # Recreate the upstream decision queue against the revised workbook. Keep the
    # previous option where it remains available, but reset every status to Pending.
    old_map = {}
    if isinstance(decision_queue, pd.DataFrame) and not decision_queue.empty:
        for _, old_row in decision_queue.iterrows():
            old_map[_safe(old_row.get("Category"))] = old_row
    old_reported = old_map.get("Reported effect-size method")
    assume_reported = bool(old_reported is not None and _safe(old_reported.get("Your choice")) == "assume_ppc2_unverified")
    new_queue = build_decision_queue(new_session["review"], assume_reported_g_is_ppc2=assume_reported)
    for idx, new_row in new_queue.iterrows():
        old_row = old_map.get(_safe(new_row.get("Category")))
        if old_row is None:
            continue
        try:
            options = json.loads(_safe(new_row.get("Available options")) or "[]")
            valid = {str(o.get("value")) for o in options if isinstance(o, dict)}
        except Exception:
            valid = set()
        old_choice = _safe(old_row.get("Your choice"))
        if old_choice in valid:
            new_queue.at[idx, "Your choice"] = old_choice
        new_queue.at[idx, "Status"] = "Pending"
        if "Decision recorded (UTC)" in new_queue.columns:
            new_queue.at[idx, "Decision recorded (UTC)"] = ""
        # Carry structured context only if the choice remains available.
        if old_choice in valid:
            new_queue.at[idx, "Decision details (structured)"] = _safe(old_row.get("Decision details (structured)")) or "{}"
    return new_session, new_queue, entry


def bundle_to_state(bundle: dict[str, Any]) -> str:
    """Serialize dataframes to JSON for reliable Gradio session state."""
    payload: dict[str, Any] = {}
    for key, value in bundle.items():
        if isinstance(value, pd.DataFrame):
            payload[key] = {"__dataframe__": value.to_json(orient="records", force_ascii=False, date_format="iso")}
        else:
            payload[key] = value
    return json.dumps(payload, ensure_ascii=False, default=str)


def bundle_from_state(value: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if not value:
        return None
    if isinstance(value, dict):
        return value
    try:
        payload = json.loads(value)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for key, item in list(payload.items()):
        if isinstance(item, dict) and "__dataframe__" in item:
            try:
                payload[key] = pd.DataFrame(json.loads(item["__dataframe__"]))
            except Exception:
                payload[key] = pd.DataFrame()
    return payload


def _write_frame(writer, frame: pd.DataFrame, sheet_name: str) -> None:
    (frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)).to_excel(writer, index=False, sheet_name=sheet_name[:31])


def export_preparation_workbook(bundle: dict[str, Any], path: str | Path) -> str:
    """Export a separate, auditable data-preparation workbook."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("prepared_records"), pd.DataFrame):
        raise ValueError("Build the prepared dataset before exporting.")
    out_path = str(path)
    summary = bundle.get("summary", {})
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame([{"Item": k, "Value": json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list, tuple)) else v} for k, v in summary.items()]).to_excel(writer, index=False, sheet_name="Preparation Summary")
        public_records = bundle["prepared_records"].drop(columns=[c for c in bundle["prepared_records"].columns if c.startswith("_")], errors="ignore")
        _write_frame(writer, public_records, "Prepared Dataset")
        _write_frame(writer, bundle.get("canonical_source_records", pd.DataFrame()), "Canonical Source Records")
        _write_frame(writer, bundle.get("classification", pd.DataFrame()), "Four-Dimension Classification")
        _write_frame(writer, bundle.get("findings", pd.DataFrame()), "Data Quality Findings")
        _write_frame(writer, bundle.get("review_items", pd.DataFrame()), "Researcher Review Items")
        _write_frame(writer, bundle.get("dependency_map", pd.DataFrame()), "Decision Dependency Map")
        _write_frame(writer, bundle.get("transformations", pd.DataFrame()), "Transformation Log")
        _write_frame(writer, pd.DataFrame(bundle.get("correction_history", [])), "Correction History")
        _write_frame(writer, pd.DataFrame([
            {"Field": "Original workbook", "Description": "Source workbook is retained unchanged; all preparation fields are derived into a separate dataset."},
            {"Field": "Outcome data type", "Description": "Continuous, binary, ordinal, count/rate, survival, correlation, proportion or unclear."},
            {"Field": "Reporting format", "Description": "The summary-statistic format supplied by each record; mixed formats are allowed."},
            {"Field": "Study design", "Description": "The design context for the study, distinguishing project-level assumptions from study-level confirmation."},
            {"Field": "Analysis structure", "Description": "Pairwise, network, diagnostic accuracy, multivariate, multilevel or other structure."},
            {"Field": "N control for sample-size-based variance routes", "Description": "Separate field for approved shared-control splitting. Original control N is never overwritten; this is not used for direct generic-inverse-variance SE/CI inputs."},
            {"Field": "Primary effect route status", "Description": "Availability of a compatible candidate route only; final effect calculation is a later stage."},
            {"Field": "Variance preparation status", "Description": "Records approved variance strategies but does not calculate sampling variance."},
            {"Field": "Candidate calculations (structured JSON)", "Description": "All detected candidate routes, their effect estimates, inputs, methods, assumptions and blockers, retained for the next execution stage."},
            {"Field": "Decision dependency map", "Description": "Shows which researcher choices activate, change or invalidate other preparation review tasks. Dependent approvals are reopened when their upstream choice changes."},
        ]), "Data Dictionary")
    return out_path
