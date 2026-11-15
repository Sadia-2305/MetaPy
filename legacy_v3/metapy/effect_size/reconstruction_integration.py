"""Integrate statistical uncertainty reconstruction with MetaPy harmonization.

This module adapts the canonical records created by harmonization_engine to the
standalone variance_reconstruction engine. It scans every matching worksheet
representation, rejects conflicting repeated fields, preserves provenance, and
updates harmonized records before readiness issues are emitted.

No external study verification is performed and no assumptions are silently
introduced.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping

import pandas as pd

from .harmonization_engine import (
    _duration_key,
    _norm,
    _outcome_key,
    _study_arm,
)
from .variance_reconstruction import reconstruct_uncertainty, VERSION as RECONSTRUCTION_VERSION


# Normalized aliases for supplementary statistical information. These are
# intentionally explicit; a generic "SD", "SE" or "p" column is not guessed.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    # Effect-specific uncertainty
    "effect_variance": (
        "hedgesgvariance", "variancehedgesg", "hedgesgvar", "varhedgesg",
        "smdvariance", "variancesmd", "effectsizevariance", "effectvariance",
    ),
    "effect_se": (
        "hedgesgse", "sehedgesg", "smdse", "sesmd", "effectsize se",
        "effectse", "seeffectsize",
    ),
    "effect_ci_low": (
        "hedgesgcilower", "hedgesglowerci", "cilowerhedgesg",
        "smdcilower", "smdlowerci", "hedgesgci_lower", "hedgesg95cilower",
    ),
    "effect_ci_high": (
        "hedgesgciupper", "hedgesgupperci", "ciupperhedgesg",
        "smdciupper", "smdupperci", "hedgesgci_upper", "hedgesg95ciupper",
    ),
    "effect_ci_level": ("hedgesgcilevel", "hedgesg95cilevel", "hedgesgconfidencelevel", "effectcilevel"),
    "effect_ci_df": ("hedgesgcidegreesoffreedom", "hedgesgcidf", "effectcidf"),
    # Raw group means and SDs
    "mean_pre_1": ("meanprei", "premeani", "meanpreintervention", "interventionpremean", "preinterventionmean"),
    "sd_pre_1": ("sdprei", "presdi", "sdpreintervention", "interventionpresd", "presdintervention"),
    "mean_post_1": ("meanposti", "postmeani", "meanpostintervention", "interventionpostmean", "postinterventionmean"),
    "sd_post_1": ("sdposti", "postsdi", "sdpostintervention", "interventionpostsd", "postsdintervention"),
    "mean_pre_2": ("meanprec", "premeanc", "meanprecontrol", "controlpremean", "precontrolmean"),
    "sd_pre_2": ("sdprec", "presdc", "sdprecontrol", "controlpresd", "presdcontrol"),
    "mean_post_2": ("meanpostc", "postmeanc", "meanpostcontrol", "controlpostmean", "postcontrolmean"),
    "sd_post_2": ("sdpostc", "postsdc", "sdpostcontrol", "controlpostsd", "postsdcontrol"),
    # Change-score statistics by group
    "sd_change_1": ("sdchangei", "changesdi", "sdchangeintervention", "interventionchangesd"),
    "se_change_1": ("sechangei", "changesei", "sechangeintervention", "interventionchangese"),
    "mean_change_1": ("meanchangei", "changemeani", "meanchangeintervention", "interventionmeanchange"),
    "paired_t_1": ("pairedti", "tpairedi", "within grouptintervention", "pairedtintervention"),
    "paired_p_1": ("pairedpi", "ppairedi", "within grouppintervention", "pairedpintervention"),
    "paired_df_1": ("paireddfi", "dfpairedi", "paireddfintervention"),
    "paired_p_sidedness_1": ("pairedpsidednessi", "psidednessi", "pairedpconventionintervention", "pairedptwosidedi", "pairedptwosidedintervention"),
    "change_ci_low_1": ("changecilowi", "cilowchangei", "meanchangecilowi"),
    "change_ci_high_1": ("changecihi", "cihighchangei", "meanchangecihi"),
    "change_ci_level_1": ("changecileveli", "cilevelchangei"),
    "sd_change_2": ("sdchangec", "changesdc", "sdchangecontrol", "controlchangesd"),
    "se_change_2": ("sechangec", "changesec", "sechangecontrol", "controlchangese"),
    "mean_change_2": ("meanchangec", "changemeanc", "meanchangecontrol", "controlmeanchange"),
    "paired_t_2": ("pairedtc", "tpairedc", "pairedtcontrol"),
    "paired_p_2": ("pairedpc", "ppairedc", "pairedpcontrol"),
    "paired_df_2": ("paireddfc", "dfpairedc", "paireddfcontrol"),
    "paired_p_sidedness_2": ("pairedpsidednessc", "psidednessc", "pairedpconventioncontrol", "pairedptwosidedc", "pairedptwosidedcontrol"),
    "change_ci_low_2": ("changecilowc", "cilowchangec", "meanchangecilowc"),
    "change_ci_high_2": ("changecihighc", "cihighchangec", "meanchangecihighc"),
    "change_ci_level_2": ("changecilevelc", "cilevelchangec"),
    # Within-group mean SE and confidence intervals (pre/post)
    "pre_se_1": ("presei", "seprei", "preseintervention", "sepreintervention"),
    "post_se_1": ("postsei", "seposti", "postseintervention", "sepostintervention"),
    "pre_ci_low_1": ("precilowi", "cilowprei", "meanprecilowi"),
    "pre_ci_high_1": ("precihighi", "cihighprei", "meanprecihighi"),
    "post_ci_low_1": ("postcilowi", "cilowposti", "meanpostcilowi"),
    "post_ci_high_1": ("postcihighi", "cihighposti", "meanpostcihighi"),
    "pre_se_2": ("presec", "seprec", "presecontrol", "seprecontrol"),
    "post_se_2": ("postsec", "sepostc", "postsecontrol", "sepostcontrol"),
    "pre_ci_low_2": ("precilowc", "cilowprec", "meanprecilowc"),
    "pre_ci_high_2": ("precihighc", "cihighprec", "meanprecihighc"),
    "post_ci_low_2": ("postcilowc", "cilowpostc", "meanpostcilowc"),
    "post_ci_high_2": ("postcihighc", "cihighpostc", "meanpostcihighc"),
    "mean_df_1": ("meandfi", "dfmeani", "interventionmeandf"),
    "mean_df_2": ("meandfc", "dfmeanc", "controlmeandf"),
    # Pre/post correlations
    "r_1": ("rpreposti", "prepostcorrelationi", "prepostcorrelationintervention", "correlationprepostintervention", "interventionprepostcorrelation"),
    "r_2": ("rprepostc", "prepostcorrelationc", "prepostcorrelationcontrol", "correlationprepostcontrol", "controlprepostcorrelation", "rcontrolprepost"),
    "common_correlation": ("rprepost", "prepostcorrelation", "commonprepostcorrelation", "withinpersoncorrelation", "rcommon", "commonr", "commoncorrelation"),
    # Between-group raw MD and its uncertainty. These remain MD-scale until
    # explicit same-estimand and unadjusted conditions are satisfied.
    "md": ("md", "meandifference", "betweengroupmd", "posttestmd", "posttreatmentmd"),
    "md_ci_low": ("mdcilow", "meandifferencecilow", "betweengroupmdcilow"),
    "md_ci_high": ("mdcihigh", "meandifferencecihigh", "betweengroupmdcihigh"),
    "md_95_ci": ("md95ci", "mdci", "md95percentci", "md(95%ci)"),
    "md_se": ("mdse", "meandifferencese", "betweengroupmdse"),
    "md_t": ("mdt", "meandifferencet", "betweengroupt"),
    "md_p": ("mdp", "meandifferencep", "betweengroupp"),
    "md_df": ("mddf", "meandifferencedf", "betweengroupdf"),
    "md_test_type": ("mdtesttype", "meandifferencetesttype"),
    "md_p_sidedness": ("mdpsidedness", "meandifferencepsidedness"),
    "md_f": ("mdf", "betweengroupf", "fstatisticmd"),
    "md_f_df1": ("mdfdf1", "fdf1", "numeratordf"),
    "md_estimand": ("mdestimand", "meandifferenceestimand"),
    "md_unadjusted": ("mdunadjusted", "unadjustedmdconfirmed", "unadjustedmd"),
    "md_ci_level": ("mdcilevel", "meandifferencecilevel"),
    # Explicit design / researcher-supplied options
    "design": ("studydesign", "designtype", "randomizationdesign"),
    "estimand": ("effectestimand", "estimand"),
    "correlation_imputation_approved": ("correlationimputationapproved", "imputationapproved", "researcherapprovedimputation"),
    "imputation_values": ("correlationsensitivityvalues", "imputationvalues"),
    "selected_imputed_correlation": ("selectedimputedcorrelation", "researcherselectedcorrelation"),
}


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def _num(value: Any) -> float | None:
    if _is_missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _choose_column(columns, aliases):
    alias_set = {_norm(x) for x in aliases}
    for col in columns:
        if _norm(col) in alias_set:
            return col
    return None


def _same(a: Any, b: Any, *, exact: bool = False) -> bool:
    if _is_missing(a) and _is_missing(b):
        return True
    if _is_missing(a) or _is_missing(b):
        return False
    an, bn = _num(a), _num(b)
    if an is not None and bn is not None:
        return an == bn if exact else math.isclose(an, bn, rel_tol=1e-6, abs_tol=0.005)
    return str(a).strip().casefold() == str(b).strip().casefold()


def _record_key(study_id, arm, outcome, duration):
    return (str(study_id), str(arm or ""), str(outcome), str(duration or ""))


def _collect_source_rows(sheets: Mapping[str, pd.DataFrame]) -> dict[tuple, list[dict[str, Any]]]:
    """Index all workbook rows by the exact canonical key used by harmonizer."""
    indexed: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for sheet_name, frame in sheets.items():
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        study_col = _choose_column(frame.columns, ("studyid", "study", "studyidentifier", "studycode"))
        outcome_col = _choose_column(frame.columns, ("scale", "biomarker", "outcome", "outcomename", "marker", "anxietyscale"))
        duration_col = _choose_column(frame.columns, ("duration", "followupduration", "timepointduration", "timepoint", "followup"))
        if study_col is None or outcome_col is None:
            continue
        chosen = {field: _choose_column(frame.columns, aliases) for field, aliases in FIELD_ALIASES.items()}
        for row_idx, row in frame.iterrows():
            if row.isna().all() or _is_missing(row.get(study_col)):
                continue
            sid, arm = _study_arm(row.get(study_col), row.get(outcome_col))
            out = _outcome_key(row.get(outcome_col), arm)
            dur = _duration_key(row.get(duration_col)) if duration_col else ""
            key = _record_key(sid, arm, out, dur)
            values = {}
            for field, col in chosen.items():
                if col is not None and not _is_missing(row.get(col)):
                    values[field] = row.get(col)
            indexed[key].append({
                "source_sheet": sheet_name,
                "source_row": int(row_idx) + 2,
                "source_values": values,
            })
    return indexed


def _merged_stats(source_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Merge complementary fields; exclude fields that conflict across sources."""
    candidates: dict[str, list[tuple[Any, str]]] = defaultdict(list)
    locations = []
    for item in source_rows:
        sheet = item["source_sheet"]
        locations.append(f"{sheet}!row {item['source_row']}")
        for field, value in item["source_values"].items():
            candidates[field].append((value, sheet))
    merged: dict[str, Any] = {}
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for field, items in candidates.items():
        values = [x[0] for x in items]
        exact = field in {"md_df", "md_f_df1", "paired_df_1", "paired_df_2", "mean_df_1", "mean_df_2"}
        if any(not _same(values[0], value, exact=exact) for value in values[1:]):
            conflicts[field] = [{"value": value, "sheet": sheet} for value, sheet in items]
        else:
            merged[field] = values[0]
    return merged, conflicts, sorted(set(locations))


def _shared_control_keys(records: pd.DataFrame) -> set[tuple]:
    """Return keys with a possible repeated comparator (same study/outcome/time/Nc)."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for rec in records.to_dict(orient="records"):
        n2 = _num(rec.get("n_control"))
        if n2 is None:
            continue
        groups[(str(rec.get("study_id")), str(rec.get("outcome")), str(rec.get("duration")), n2)].append(rec)
    result = set()
    for group_key, group in groups.items():
        arms = {str(r.get("arm") or "") for r in group}
        if len(group) > 1 and len(arms) > 1:
            for rec in group:
                result.add(_record_key(rec.get("study_id"), rec.get("arm"), rec.get("outcome"), rec.get("duration")))
    return result


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def apply_reconstruction(
    harmonization: dict[str, Any],
    sheets: Mapping[str, pd.DataFrame],
    *,
    estimand: str,
    design: str,
) -> dict[str, Any]:
    """Run the reconstruction cascade on every canonical harmonized record.

    The function mutates only the returned in-memory harmonization object,
    never the researcher's source workbook.
    """
    records = harmonization["records"].copy()
    if records.empty:
        harmonization["reconstruction"] = {
            "version": RECONSTRUCTION_VERSION,
            "records": pd.DataFrame(),
            "summary": {"total": 0, "resolved": 0, "needs_researcher_input": 0},
        }
        return harmonization

    source_index = _collect_source_rows(sheets)
    possible_shared = _shared_control_keys(records)
    reconstructed: list[dict[str, Any]] = []
    conflicts_by_record: dict[int, dict[str, Any]] = {}

    for idx, rec in records.iterrows():
        key = _record_key(rec.get("study_id"), rec.get("arm"), rec.get("outcome"), rec.get("duration"))
        source_rows = source_index.get(key, [])
        merged, field_conflicts, locations = _merged_stats(source_rows)
        conflicts_by_record[idx] = field_conflicts

        payload: dict[str, Any] = {
            **merged,
            "record_id": f"{key[0]}::{key[1]}::{key[2]}::{key[3]}",
            "study_id": rec.get("study_id"),
            "source_sheet": "; ".join(sorted({x["source_sheet"] for x in source_rows})) or rec.get("source_locations"),
            "source_row": "; ".join(str(x["source_row"]) for x in source_rows),
            "effect_estimate": _num(rec.get("selected_hedges_g")),
            "n1": _num(rec.get("n_intervention")),
            "n2": _num(rec.get("n_control")),
            "estimand": estimand,
            "design": design,
            "shared_control": key in possible_shared,
        }
        # Only mark an effect CI as effect-specific if its column was explicitly
        # labelled with Hedges-g/SMD and both bounds were consistently extracted.
        if "effect_ci_low" in merged and "effect_ci_high" in merged:
            payload["ci_is_effect_specific"] = True
        # The effect CI level/df are not silently filled from unrelated CIs.
        # A clearly named Hedges-g CI gets a conventional 95% level if not stated.
        if payload.get("ci_is_effect_specific") and payload.get("effect_ci_level") is None:
            payload["effect_ci_level"] = 0.95

        result = reconstruct_uncertainty(payload, estimand=estimand, design=design)
        result["canonical_key"] = key
        result["source_locations"] = locations or [str(rec.get("source_locations") or "")]
        result["field_conflicts"] = field_conflicts
        reconstructed.append(result)

        # Replace any earlier variance calculation with the new engine's result.
        # Never retain a variance produced by an incompatible fallback.
        if result.get("status") == "resolved":
            records.at[idx, "variance"] = result.get("variance")
            records.at[idx, "standard_error"] = result.get("se")
            ci = result.get("ci") or {}
            critical_method = ci.get("critical_value_method")
            records.at[idx, "ci_lower_reconstructed"] = ci.get("lower")
            records.at[idx, "ci_upper_reconstructed"] = ci.get("upper")
            records.at[idx, "ci_level_reconstructed"] = ci.get("level")
            records.at[idx, "ci_critical_value_method"] = critical_method
            # Preserve the legacy *_normal fields only when the engine actually
            # used a normal critical value. A Student-t CI must not be mislabeled.
            records.at[idx, "ci_lower_95_normal"] = ci.get("lower") if critical_method and "normal" in critical_method.lower() else None
            records.at[idx, "ci_upper_95_normal"] = ci.get("upper") if critical_method and "normal" in critical_method.lower() else None
            records.at[idx, "variance_method"] = result.get("method")
            records.at[idx, "prepost_correlation_used"] = (
                (result.get("details") or {}).get("r_weighted")
                if (result.get("details") or {}).get("r_weighted") is not None
                else (result.get("details") or {}).get("r_common")
            )
        else:
            records.at[idx, "variance"] = None
            records.at[idx, "standard_error"] = None
            records.at[idx, "ci_lower_95_normal"] = None
            records.at[idx, "ci_upper_95_normal"] = None
            records.at[idx, "ci_lower_reconstructed"] = None
            records.at[idx, "ci_upper_reconstructed"] = None
            records.at[idx, "ci_level_reconstructed"] = None
            records.at[idx, "ci_critical_value_method"] = None
            records.at[idx, "variance_method"] = None
            records.at[idx, "prepost_correlation_used"] = None

    # Add a compact, reviewable audit trail directly to the harmonized record.
    audit_columns = {
        "reconstruction_status": [],
        "reconstruction_route": [],
        "reconstruction_assumptions": [],
        "reconstruction_attempted_routes": [],
        "reconstruction_blocking_issues": [],
        "reconstruction_details": [],
        "reconstruction_source_provenance": [],
        "reconstruction_dependence_note": [],
        "reconstruction_field_conflicts": [],
    }
    for result in reconstructed:
        audit_columns["reconstruction_status"].append(result.get("status"))
        audit_columns["reconstruction_route"].append(result.get("route"))
        audit_columns["reconstruction_assumptions"].append(_json_text(result.get("assumptions", [])))
        audit_columns["reconstruction_attempted_routes"].append(_json_text(result.get("attempted_routes", [])))
        audit_columns["reconstruction_blocking_issues"].append(_json_text(result.get("blocking_issues", [])))
        audit_columns["reconstruction_details"].append(_json_text(result.get("details") or result.get("correlation_diagnostics") or result.get("sensitivity")))
        audit_columns["reconstruction_source_provenance"].append(_json_text(result.get("provenance", {})))
        audit_columns["reconstruction_dependence_note"].append(result.get("dependence_note"))
        audit_columns["reconstruction_field_conflicts"].append(_json_text(result.get("field_conflicts", {})))
    for column, values in audit_columns.items():
        records[column] = values

    # Update calculation log using canonical record order.
    calc_log = harmonization["calculation_log"].copy()
    if len(calc_log) == len(records):
        calc_log["reconstructed_variance"] = records["variance"].tolist()
        calc_log["reconstructed_se"] = records["standard_error"].tolist()
        calc_log["reconstruction_method"] = records["variance_method"].tolist()
        calc_log["reconstruction_status"] = audit_columns["reconstruction_status"]
        calc_log["reconstruction_route"] = audit_columns["reconstruction_route"]
        calc_log["reconstruction_attempts"] = audit_columns["reconstruction_attempted_routes"]
        calc_log["reconstruction_blockers"] = audit_columns["reconstruction_blocking_issues"]
        calc_log["reconstruction_assumptions"] = audit_columns["reconstruction_assumptions"]
        calc_log["reconstruction_source_provenance"] = audit_columns["reconstruction_source_provenance"]
        calc_log["possible_shared_control"] = [bool(r.get("shared_control")) for r in reconstructed]

    # Replace the old per-record G_VARIANCE_UNRESOLVED findings with grouped
    # outcome-level findings emitted only after every supported route ran.
    old_issues = harmonization["issues"].copy()
    if not old_issues.empty and "code" in old_issues.columns:
        old_issues = old_issues.loc[old_issues["code"] != "G_VARIANCE_UNRESOLVED"].copy()
    unresolved_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_idx, result in enumerate(reconstructed):
        if result.get("status") != "resolved":
            outcome = str(records.iloc[row_idx].get("outcome") or "Unspecified outcome")
            unresolved_groups[outcome].append({"record": records.iloc[row_idx].to_dict(), "result": result})

    extra_issue_rows = []
    for outcome, group in sorted(unresolved_groups.items()):
        blocker_counts = Counter(
            blocker
            for item in group
            for blocker in (item["result"].get("blocking_issues") or [])
        )
        top_blockers = [f"{count} record(s): {message}" for message, count in blocker_counts.most_common(4)]
        study_ids = sorted({str(item["record"].get("study_id")) for item in group})
        locations = sorted({str(item["record"].get("source_locations")) for item in group})
        attempts = Counter(
            attempt.get("route", "unknown")
            for item in group
            for attempt in item["result"].get("attempted_routes", [])
        )
        route_text = ", ".join(f"{route} ({count} attempt(s))" for route, count in sorted(attempts.items()))
        message = (
            f"MetaPy attempted the supported uncertainty-reconstruction routes for {len(group)} canonical record(s) under the selected estimand '{estimand}'. "
            + (" Main unresolved reasons: " + "; ".join(top_blockers) if top_blockers else "No compatible uncertainty route was established.")
            + (f" Routes checked: {route_text}." if route_text else "")
            + " No correlation or other missing statistic was imputed."
        )
        extra_issue_rows.append({
            "severity": "Warning",
            "code": "G_VARIANCE_RECONSTRUCTION_PENDING",
            "finding": "Sampling variance remains unresolved after supported reconstruction routes were attempted",
            "study_id": ", ".join(study_ids),
            "arm": "",
            "outcome": outcome,
            "source_locations": "; ".join(locations),
            "reported_g": None,
            "calculated_g": None,
            "difference": None,
            "message": message,
        })
    # Conflicting supplementary statistics are never silently discarded.
    for row_idx, conflicts in conflicts_by_record.items():
        if not conflicts:
            continue
        rec = records.iloc[row_idx]
        shown = []
        sheets_involved = set()
        for field, values in conflicts.items():
            rendered = []
            for item in values:
                sheets_involved.add(str(item.get("sheet")))
                rendered.append(f"{item.get('sheet')}={item.get('value')}")
            shown.append(f"{field}: " + " | ".join(rendered))
        extra_issue_rows.append({
            "severity": "Warning",
            "code": "RECONSTRUCTION_INPUT_FIELD_CONFLICT",
            "finding": "Conflicting supplementary statistics were excluded from uncertainty reconstruction",
            "study_id": str(rec.get("study_id")),
            "arm": str(rec.get("arm") or ""),
            "outcome": str(rec.get("outcome")),
            "source_locations": str(rec.get("source_locations") or ""),
            "reported_g": rec.get("reported_hedges_g"),
            "calculated_g": rec.get("selected_hedges_g"),
            "difference": None,
            "message": (
                f"For Study {rec.get('study_id')} ({rec.get('arm') or 'unspecified arm'}; {rec.get('outcome')}), "
                "MetaPy found repeated supplementary statistics that disagree across worksheet representations. "
                "The conflicting field(s) were excluded from the reconstruction payload rather than choosing one value: "
                + "; ".join(shown)
                + ". Review the source cells before using any affected uncertainty route."
            ),
        })
    if extra_issue_rows:
        additions = pd.DataFrame(extra_issue_rows)
        combined_rows = old_issues.to_dict(orient="records") + additions.to_dict(orient="records")
        old_issues = pd.DataFrame(combined_rows)

    # Update summary and readiness from the new reconstruction results.
    resolved_count = sum(r.get("status") == "resolved" for r in reconstructed)
    variance_count = int(records["variance"].notna().sum())
    total = int(len(records))
    summary = dict(harmonization["summary"])
    route_counts = Counter(r.get("route") or "unresolved" for r in reconstructed)
    summary.update({
        "variance_available_count": variance_count,
        "variance_unresolved_count": total - variance_count,
        "reconstruction_engine_version": RECONSTRUCTION_VERSION,
        "reconstruction_resolved_count": resolved_count,
        "reconstruction_pending_count": total - resolved_count,
        "reconstruction_route_counts": dict(route_counts),
        "reconstruction_method": "Comprehensive statistical reconstruction cascade",
        "harmonization_ready": bool(total > 0 and records["selected_hedges_g"].notna().all() and variance_count == total),
    })

    harmonization["records"] = records
    harmonization["calculation_log"] = calc_log
    harmonization["issues"] = old_issues.reset_index(drop=True)
    harmonization["summary"] = summary
    harmonization["reconstruction"] = {
        "version": RECONSTRUCTION_VERSION,
        "records": pd.DataFrame(reconstructed),
        "summary": {"total": total, "resolved": resolved_count, "needs_researcher_input": total - resolved_count},
    }
    return harmonization


__all__ = ["apply_reconstruction", "FIELD_ALIASES"]
