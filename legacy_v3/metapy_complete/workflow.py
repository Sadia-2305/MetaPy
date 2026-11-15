from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .effect_sizes import ESTIMANDS, calculate_effects
from .meta_analysis import leave_one_study_out, pool_by_outcome
from .preparation import prepare_records
from .reporting import export_report, normalize_plot_settings
from .sensitivity import pool_correlation_sensitivity
from .workbook import canonicalize_workbook, load_excel_workbook


def build_researcher_review_table(canonical: pd.DataFrame, prepared: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build editable row-by-row review table. All approvals default to False."""
    if canonical.empty:
        return pd.DataFrame(columns=[
            "record_id", "study_id", "outcome", "duration", "intervention_arm", "study_design_source",
            "direction_multiplier", "parallel_independent_groups_confirmed",
            "reported_effect_method_source", "reported_effect_method_override", "approve_record"
        ])
    source = prepared if prepared is not None and not prepared.empty else canonical
    rows = []
    for r in source.to_dict("records"):
        direction = r.get("effect_direction_multiplier")
        rows.append({
            "record_id": r.get("record_id"),
            "study_id": r.get("study_id"),
            "outcome": r.get("outcome"),
            "duration": r.get("duration"),
            "intervention_arm": r.get("intervention_arm"),
            "study_design_source": r.get("study_design"),
            "direction_multiplier": direction,
            "parallel_independent_groups_confirmed": bool(str(r.get("study_design_status", "")).startswith("Parallel independent groups")),
            "reported_effect_method_source": r.get("reported_effect_method"),
            "reported_effect_method_override": None,
            "approve_record": bool(r.get("record_review_approved", False)),
        })
    return pd.DataFrame(rows)


def _safe_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "approved", "y"}
    return bool(value)


def decisions_from_review_table(review_table: pd.DataFrame, *, confirm_all_parallel: bool = False) -> tuple[dict[str, bool], dict[str, Any]]:
    approvals: dict[str, bool] = {}
    directions: dict[str, int] = {}
    designs: dict[str, bool] = {}
    effect_methods: dict[str, str] = {}
    if review_table is not None and not review_table.empty:
        for row in review_table.to_dict("records"):
            rid = str(row.get("record_id"))
            if rid in {"", "nan", "None"}:
                continue
            approvals[rid] = _safe_bool(row.get("approve_record", False))
            value = row.get("direction_multiplier")
            try:
                ivalue = int(float(value))
                if ivalue in (-1, 1):
                    directions[rid] = ivalue
            except (TypeError, ValueError):
                pass
            designs[rid] = _safe_bool(row.get("parallel_independent_groups_confirmed", False))
            method_override = row.get("reported_effect_method_override")
            if not pd.isna(method_override) and str(method_override).strip():
                effect_methods[rid] = str(method_override).strip()
    decisions = {
        "record_direction_multipliers": directions,
        "record_design_confirmations": designs,
        "record_effect_method_overrides": effect_methods,
        "parallel_independent_groups_confirmed": bool(confirm_all_parallel),
    }
    return approvals, decisions


def run_analysis(
    workbook_path: str | Path,
    *,
    estimand: str = "morris_ppc2",
    review_table: pd.DataFrame | None = None,
    record_approvals: dict[str, bool] | None = None,
    global_decisions: dict[str, Any] | None = None,
    correlation_sensitivity: list[float] | None = None,
    shared_control_policy: str = "block",
    model: str = "random",
    tau_method: str = "REML",
    ci_method: str = "Hartung-Knapp",
    zero_cell_correction: float = 0.5,
    plot_settings: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    if estimand not in ESTIMANDS:
        raise ValueError(f"Select a supported estimand: {list(ESTIMANDS)}")
    source = Path(workbook_path)
    sheets = load_excel_workbook(source)
    canonical, conflicts, unmapped = canonicalize_workbook(sheets)
    if canonical.empty:
        raise ValueError("No usable comparison rows were found. Fill StudyID and Outcome in the input template, then upload the completed workbook.")

    if review_table is not None:
        approvals_from_table, decisions_from_table_ = decisions_from_review_table(review_table)
        record_approvals = approvals_from_table
        supplied = global_decisions or {}
        supplied.update(decisions_from_table_)
        global_decisions = supplied
    record_approvals = record_approvals or {}
    global_decisions = global_decisions or {}

    prepared, findings, transformation_log, prep_summary_frame = prepare_records(
        canonical,
        global_decisions=global_decisions,
        record_approvals=record_approvals,
    )

    effects, sensitivity_records = calculate_effects(
        prepared,
        estimand,
        record_approvals=record_approvals,
        global_decisions=global_decisions,
        correlation_sensitivity=correlation_sensitivity,
        shared_control_policy=shared_control_policy,
        zero_cell_correction=zero_cell_correction,
    )
    pooled = pool_by_outcome(effects, model=model, tau_method=tau_method, ci_method=ci_method)
    sensitivity_pooled = pool_correlation_sensitivity(
        sensitivity_records, model=model, tau_method=tau_method, ci_method=ci_method
    )

    loo_rows = []
    if not effects.empty:
        eligible = effects[effects["include_in_primary_pool"].fillna(False).astype(bool)]
        group_cols = ["outcome"] + (["duration"] if "duration" in eligible and eligible["duration"].notna().any() else [])
        for key, group in eligible.groupby(group_cols, dropna=False, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            loo = leave_one_study_out(group, model=model, tau_method=tau_method, ci_method=ci_method)
            if not loo.empty:
                for k, v in zip(group_cols, key):
                    loo[k] = v
                loo_rows.append(loo)
    loo_frames = [frame.dropna(axis=1, how="all") for frame in loo_rows if not frame.empty]
    loo_results = pd.concat(loo_frames, ignore_index=True, sort=False) if loo_frames else pd.DataFrame()

    approvals_n = int(prepared.get("record_review_approved", pd.Series(dtype=bool)).sum()) if not prepared.empty else 0
    both_corr_n = int((prepared.get("r_prepost_i_source", pd.Series(dtype=float)).notna() & prepared.get("r_prepost_c_source", pd.Series(dtype=float)).notna()).sum()) if not prepared.empty else 0
    eligible_n = int(effects.get("include_in_primary_pool", pd.Series(dtype=bool)).sum()) if not effects.empty else 0
    effect_n = int(effects.get("effect_estimate", pd.Series(dtype=float)).notna().sum()) if not effects.empty else 0
    raw_calculated_n = int(effects.get("estimate_source", pd.Series(dtype=str)).fillna("").str.contains("Independently calculated|Calculated from", case=False, regex=True).sum()) if not effects.empty else 0
    reported_retained_n = int(effects.get("estimate_source", pd.Series(dtype=str)).fillna("").str.contains("Researcher-reported", case=False, regex=False).sum()) if not effects.empty else 0
    variance_n = int(effects.get("sampling_variance", pd.Series(dtype=float)).notna().sum()) if not effects.empty else 0
    pooled_n = 0 if pooled.empty else int((pooled.get("status", pd.Series(dtype=str)) == "Pooled").sum())
    pending_approval_n = len(prepared) - approvals_n
    quality_codes = {"source_field_conflict", "intervention_arm_not_specified", "one_prepost_correlation_missing", "prepost_correlation_out_of_range"}
    quality_findings = findings[findings.get("finding_code", pd.Series(dtype=str)).isin(quality_codes)].copy() if not findings.empty else pd.DataFrame()
    researcher_review_items = findings[~findings.get("finding_code", pd.Series(dtype=str)).isin(quality_codes)].copy() if not findings.empty else pd.DataFrame()
    if not unmapped.empty:
        unmapped_quality = unmapped.copy()
        unmapped_quality["finding_type"] = "Data quality review"
        unmapped_quality["finding_code"] = "UNMAPPED_NONEMPTY_COLUMNS"
        unmapped_quality["description"] = "Non-empty workbook columns were not mapped to a recognised MetaPy field; review the original source fields."
        unmapped_quality["status"] = "Pending review"
        quality_findings = pd.concat([quality_findings, unmapped_quality], ignore_index=True)
    plot_settings = normalize_plot_settings(plot_settings)
    summary = {
        "metapy_version": __version__,
        "source_filename": source.name,
        "source_workbook_modified": False,
        "worksheets_imported": len(sheets),
        "canonical_comparisons": len(canonical),
        "selected_estimand": estimand,
        "selected_estimand_label": ESTIMANDS[estimand],
        "analysis_model": model,
        "tau_estimator": tau_method if model == "random" else "not applicable",
        "confidence_interval_method": ci_method if model == "random" else "normal approximation",
        "shared_control_policy": shared_control_policy,
        "records_with_effect_estimate": effect_n,
        "records_with_raw_data_calculated_effect": raw_calculated_n,
        "records_with_compatible_reported_effect_retained": reported_retained_n,
        "records_with_sampling_variance": variance_n,
        "records_with_both_source_correlations": both_corr_n,
        "record_level_approvals": approvals_n,
        "record_level_approvals_pending": pending_approval_n,
        "records_eligible_for_primary_pool": eligible_n,
        "outcome_groups_pooled": pooled_n,
        "preparation_status": str(prep_summary_frame.iloc[0]["preparation_status"]) if not prep_summary_frame.empty else "Not run",
        "effect_calculation_status": "Completed — raw-data and compatible reported-effect routes evaluated; unresolved estimates/variances remain explicitly blank",
        "pooling_status": "Completed for eligible outcome groups" if pooled_n else "No primary outcome group passed all pooling gates",
        "correlation_sensitivity_scenarios": ", ".join(str(x) for x in correlation_sensitivity) if correlation_sensitivity else "Not run",
        "data_quality_findings": len(quality_findings),
        "researcher_review_items": len(researcher_review_items),
        "analysis_blocked_comparisons": int((~effects.get("include_in_primary_pool", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()) if not effects.empty else 0,
        "source_conflicts": len(conflicts),
        "unmapped_nonempty_columns": len(unmapped),
        "visualisation_theme": plot_settings["theme"],
        "forest_plot_width_inches": plot_settings["forest_width_inches"],
        "forest_plot_row_height_inches": plot_settings["forest_row_height_inches"],
        "source_authenticity_check": "Not performed — MetaPy only assesses the supplied workbook's internal structure and calculations",
    }

    audit_rows = [
        {"decision": "Selected estimand", "value": estimand, "status": "Researcher selected"},
        {"decision": "Global independent-group confirmation", "value": str(global_decisions.get("parallel_independent_groups_confirmed", False)), "status": "Project-level decision; does not replace record review"},
        {"decision": "Number of record approvals", "value": f"{approvals_n}/{len(prepared)}", "status": "Individual comparison approvals"},
        {"decision": "Shared-control handling", "value": shared_control_policy, "status": "Researcher-selected analysis rule"},
        {"decision": "Random-effects estimator", "value": f"{tau_method} + {ci_method}", "status": "Analysis settings"},
        {"decision": "Correlation sensitivity grid", "value": str(correlation_sensitivity), "status": "Assumptions only; never source data"},
        {"decision": "Zero-cell correction", "value": zero_cell_correction, "status": "Applied only to binary RR/OR where a zero cell exists"},
        {"decision": "Visualisation settings", "value": str(plot_settings), "status": "Researcher-selected forest/funnel plot theme and size"},
        {"decision": "Source workbook protection", "value": "Original workbook read-only", "status": "Output is a separate report"},
    ]
    for item in prepared.to_dict("records"):
        rid = item.get("record_id")
        if item.get("record_review_approved") or item.get("study_design_confirmation_source") != "not confirmed" or item.get("effect_direction_source") != "not confirmed":
            audit_rows.append({
                "decision": f"Record-level decision — {rid}",
                "value": (
                    f"approved={bool(item.get('record_review_approved'))}; "
                    f"direction_multiplier={item.get('effect_direction_multiplier')}; "
                    f"design={item.get('study_design_confirmation_source')}; "
                    f"reported_method={item.get('reported_effect_method')}"
                ),
                "status": "Record-level decision record",
            })
    decision_audit = pd.DataFrame(audit_rows)

    result = {
        "source_path": str(source), "sheets": sheets, "canonical": canonical,
        "prepared": prepared, "conflicts": conflicts, "unmapped_columns": unmapped,
        "findings": findings, "quality_findings": quality_findings,
        "researcher_review_items": researcher_review_items,
        "transformation_log": transformation_log,
        "preparation_summary": prep_summary_frame, "effects": effects,
        "pooled": pooled, "sensitivity_records": sensitivity_records,
        "sensitivity_pooled": sensitivity_pooled, "leave_one_out": loo_results,
        "summary": summary, "decision_audit": decision_audit,
    }
    if output_path:
        result["report_path"] = export_report(result, output_path, plot_settings=plot_settings)
    return result
