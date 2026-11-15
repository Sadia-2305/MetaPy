"""MetaPy single-workbook review workflow (prototype v1).

Purpose
-------
Read one Excel workbook as a connected project, inspect all worksheets together,
identify internal inconsistencies/overlapping representations, and allow the
researcher to correct cell values in a separate working copy.

Scope and safeguards
--------------------
* No web, journal, DOI, PubMed, or external study verification is performed.
* All findings are based only on information inside the uploaded workbook.
* Original uploaded workbook is never modified.
* Corrections are written only to a timestamped working copy.
* Cell-level change history is retained during the current Colab runtime.
* No missing values are invented and no effect-size method is silently selected.
* Classification readiness is separated from effect-calculation and pooling readiness.

Install in Colab at:
    metapy/review/single_workbook_workflow.py

Then:
    from metapy.review import single_workbook_workflow as sw
    app = sw.build_app()
    app.launch(share=True, debug=True)
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from itertools import combinations
from typing import Any

import pandas as pd
from openpyxl import load_workbook

from ..effect_size.harmonization_engine import harmonize_workbook
from ..effect_size.reconstruction_integration import apply_reconstruction


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def _norm(value: Any) -> str:
    """Normalize a header/value for matching without changing source data."""
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def _as_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _same_value(left: Any, right: Any, *, tolerance: float = 1e-10) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True
    if _is_missing(left) != _is_missing(right):
        return False
    a = _as_float(left)
    b = _as_float(right)
    if a is not None and b is not None:
        return math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)
    return str(left).strip() == str(right).strip()


def _python_value(value: Any) -> Any:
    """Convert pandas/numpy values into values accepted by openpyxl."""
    if _is_missing(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.to_pydatetime()
    return value


def _find_column(columns, exact_names=(), contains=(), exclude=()):
    """Find a source column using normalized exact names then token matching."""
    normalized = [(column, _norm(column)) for column in columns]
    exact = {_norm(x) for x in exact_names}
    for column, name in normalized:
        if name in exact:
            return column
    required = [_norm(x) for x in contains]
    excluded = [_norm(x) for x in exclude]
    for column, name in normalized:
        if required and all(token in name for token in required) and not any(
            token in name for token in excluded
        ):
            return column
    return None


def _study_and_arm(raw_study_id: Any, outcome_text: Any = "") -> tuple[str, str]:
    """Recognize numeric study IDs and common arm labels such as 06-Meditation."""
    raw = "" if _is_missing(raw_study_id) else str(raw_study_id).strip()
    text = "" if _is_missing(outcome_text) else str(outcome_text).strip()

    numeric_id = raw
    tail = ""
    match = re.match(r"^0*(\d+)(?:\s*[-_: ]\s*(.*))?$", raw)
    if match:
        numeric_id = str(int(match.group(1)))
        tail = match.group(2) or ""
    else:
        numeric_id = _norm(raw)

    combined = f"{tail} {text}".lower()
    arm = ""
    for arm_name in ("meditation", "yoga"):
        if arm_name in combined:
            arm = arm_name
            break

    return numeric_id, arm


def _clean_outcome(outcome: Any, arm: str) -> str:
    text = "" if _is_missing(outcome) else str(outcome).strip().lower()
    if arm:
        text = re.sub(rf"\b{re.escape(arm)}\b", " ", text, flags=re.I)
    return _norm(text)


def _sheet_role(df: pd.DataFrame) -> str:
    names = [_norm(c) for c in df.columns]
    has_mean = any("mean" in n for n in names)
    has_sd = any("sd" in n or "standarddeviation" in n for n in names)
    has_effect = any(
        ("hedges" in n or "effectsize" in n or n in {"smd", "smdvalue"})
        and not any(x in n for x in ("se", "variance", "cilower", "ciupper"))
        for n in names
    )
    if has_mean or has_sd:
        return "Detailed outcome data"
    if has_effect:
        return "Effect-size / summary data"
    return "Other / unclassified table"


def _effect_column(df: pd.DataFrame):
    # Prefer Hedges-g columns and explicitly exclude uncertainty columns.
    for column in df.columns:
        name = _norm(column)
        if ("hedges" in name or name.startswith("smdhedges")) and not any(
            token in name for token in ("se", "variance", "var", "cilower", "ciupper")
        ):
            return column
    return _find_column(df.columns, exact_names=("effectsize", "smd"))


def _outcome_column(df: pd.DataFrame):
    return _find_column(
        df.columns,
        exact_names=("scale", "biomarker", "anxietyscale", "outcome", "outcomename", "marker"),
    )


def _study_column(df: pd.DataFrame):
    return _find_column(df.columns, exact_names=("studyid", "study", "studyidentifier", "studycode"))


def _duration_column(df: pd.DataFrame):
    return _find_column(df.columns, exact_names=("duration", "followupduration", "timepointduration"))


def _n_intervention_column(df: pd.DataFrame):
    return _find_column(
        df.columns,
        exact_names=("ni", "nintervention", "nexperimental", "nactive", "n1", "ninterventiongroup"),
    )


def _n_control_column(df: pd.DataFrame):
    return _find_column(
        df.columns,
        exact_names=("nc", "ncontrol", "ncomparator", "nplacebo", "n2", "ncontrolgroup"),
    )


def _make_record(sheet_name: str, df: pd.DataFrame, row_index: int, row: pd.Series):
    study_col = _study_column(df)
    outcome_col = _outcome_column(df)
    duration_col = _duration_column(df)
    effect_col = _effect_column(df)
    ni_col = _n_intervention_column(df)
    nc_col = _n_control_column(df)

    study_raw = row.get(study_col) if study_col else None
    outcome_raw = row.get(outcome_col) if outcome_col else None
    study_id, arm = _study_and_arm(study_raw, outcome_raw)
    outcome = _clean_outcome(outcome_raw, arm)
    duration = row.get(duration_col) if duration_col else None
    effect = row.get(effect_col) if effect_col else None
    ni = row.get(ni_col) if ni_col else None
    nc = row.get(nc_col) if nc_col else None

    has_mean_columns = any("mean" in _norm(c) for c in df.columns)
    has_sd_columns = any("sd" in _norm(c) or "standarddeviation" in _norm(c) for c in df.columns)
    has_raw_measurements = bool(has_mean_columns and has_sd_columns)

    key = (
        study_id,
        arm,
        outcome,
        "" if _is_missing(duration) else str(duration).strip(),
    )
    return {
        "key": key,
        "sheet": sheet_name,
        "excel_row": int(row_index) + 2,
        "study_id": study_id,
        "study_id_original": study_raw,
        "arm": arm,
        "outcome": outcome,
        "outcome_original": outcome_raw,
        "duration": duration,
        "effect": effect,
        "n_intervention": ni,
        "n_control": nc,
        "has_raw_measurements": has_raw_measurements,
        "row": row,
    }


def read_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    """Read every worksheet from one .xlsx workbook, retaining row positions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError("MetaPy single-workbook workflow currently accepts .xlsx files only.")
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl", dtype=object)
    if not sheets:
        raise ValueError("The uploaded workbook does not contain readable worksheets.")
    return sheets


# ---------------------------------------------------------------------------
# Workbook-level internal validation
# ---------------------------------------------------------------------------


def analyze_workbook(
    sheets: dict[str, pd.DataFrame],
    estimand: str = "posttest_smd",
    design: str = "parallel_independent",
) -> dict[str, Any]:
    """Inspect the workbook as one connected dataset; no external lookups."""
    issues: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    issue_counter = 0

    def add_issue(
        severity: str,
        category: str,
        code: str,
        title: str,
        explanation: str,
        recommendation: str,
        *,
        sheets_text: str = "",
        rows_text: str = "",
        study_ids: str = "",
        columns_text: str = "",
        observed_values: str = "",
        rule: str = "",
        impact: str = "",
    ):
        nonlocal issue_counter
        issue_counter += 1
        issues.append({
            "issue_id": f"WB-{issue_counter:04d}",
            "severity": severity,
            "category": category,
            "code": code,
            "finding": title,
            "what_it_means": explanation,
            "recommended_action": recommendation,
            "worksheet(s)": sheets_text,
            "Excel row(s)": rows_text,
            "study ID(s)": study_ids,
            "column(s)": columns_text,
            "observed value(s)": observed_values,
            "rule / method": rule,
            "analysis impact": impact,
            "status": ("Informational — no correction required" if severity == "Information" else "Detected — needs review/correction"),
        })

    # Inventory and canonical record extraction.
    for sheet_name, frame in sheets.items():
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        populated = frame.loc[~frame.isna().all(axis=1)]
        role = _sheet_role(frame)
        study_col = _study_column(frame)
        outcome_col = _outcome_column(frame)
        effect_col = _effect_column(frame)
        n_records = 0
        sheet_records = []
        if study_col is not None:
            for idx, row in frame.iterrows():
                if row.isna().all():
                    continue
                # Require a study identifier and at least an outcome/effect/data field.
                if _is_missing(row.get(study_col)):
                    continue
                rec = _make_record(sheet_name, frame, idx, row)
                if rec["outcome"] or effect_col is not None:
                    records.append(rec)
                    sheet_records.append(rec)
                    n_records += 1

        outcomes = sorted({
            str(rec["outcome_original"]) for rec in sheet_records
            if not _is_missing(rec["outcome_original"])
        })
        inventory_rows.append({
            "worksheet": sheet_name,
            "data rows": int(len(populated)),
            "detected records": n_records,
            "table type": role,
            "study ID column": str(study_col) if study_col is not None else "Not recognized",
            "outcome column": str(outcome_col) if outcome_col is not None else "Not recognized",
            "effect-size column": str(effect_col) if effect_col is not None else "Not detected",
            "outcomes detected": "; ".join(outcomes),
            "columns": int(frame.shape[1]),
        })

    # A. Cross-worksheet overlap and consistency checks.
    by_key: dict[tuple, list[dict[str, Any]]] = {}
    for record in records:
        if record["study_id"] and record["outcome"]:
            by_key.setdefault(record["key"], []).append(record)

    cross_sheet_pairs = 0
    matched_pairs = 0
    mismatched_pairs = 0

    # Matching source rows are handled silently: the harmonization engine
    # creates one canonical record and merges complementary non-missing data.
    # This block only keeps workbook-level counters; genuine field conflicts
    # are reported by the harmonizer after all relevant fields are compared.
    for key, group in by_key.items():
        for left, right in combinations(group, 2):
            if left["sheet"] == right["sheet"]:
                continue
            cross_sheet_pairs += 1
            left_fields = {
                "effect": left.get("effect"),
                "n_intervention": left.get("n_intervention"),
                "n_control": left.get("n_control"),
            }
            right_fields = {
                "effect": right.get("effect"),
                "n_intervention": right.get("n_intervention"),
                "n_control": right.get("n_control"),
            }
            different = any(
                not (_is_missing(left_fields[field]) and _is_missing(right_fields[field]))
                and not _same_value(left_fields[field], right_fields[field], tolerance=1e-6)
                for field in left_fields
            )
            if different:
                mismatched_pairs += 1
            else:
                matched_pairs += 1

    # B. Missing/invalid numerical information, by exact worksheet cell.
    missing_sd_by_sheet: dict[str, list[dict[str, Any]]] = {}
    invalid_values: list[dict[str, Any]] = []
    invalid_ns: list[dict[str, Any]] = []
    duplicate_keys_by_sheet: dict[str, dict[tuple, list[int]]] = {}

    for sheet_name, frame in sheets.items():
        study_col = _study_column(frame)
        outcome_col = _outcome_column(frame)
        duration_col = _duration_column(frame)
        ni_col = _n_intervention_column(frame)
        nc_col = _n_control_column(frame)
        effect_col = _effect_column(frame)
        sd_columns = [c for c in frame.columns if "sd" in _norm(c) or "standarddeviation" in _norm(c)]
        mean_columns = [c for c in frame.columns if "mean" in _norm(c)]

        for idx, row in frame.iterrows():
            if row.isna().all():
                continue
            excel_row = int(idx) + 2
            sid = row.get(study_col) if study_col else None
            outcome = row.get(outcome_col) if outcome_col else None

            # Missing SDs are reported only in rows that otherwise contain study/outcome data.
            if sid is not None and not _is_missing(sid) and sd_columns:
                for col in sd_columns:
                    if _is_missing(row.get(col)):
                        # Don't flag a wholly empty SD column in a table unless it has means or effect data.
                        if mean_columns or (effect_col is not None and not _is_missing(row.get(effect_col))):
                            missing_sd_by_sheet.setdefault(sheet_name, []).append({
                                "row": excel_row, "study_id": sid, "outcome": outcome, "column": str(col)
                            })
                    else:
                        val = _as_float(row.get(col))
                        if val is None:
                            invalid_values.append({"sheet": sheet_name, "row": excel_row, "study": sid, "column": str(col), "value": row.get(col), "reason": "SD is not a finite number"})
                        elif val < 0:
                            invalid_values.append({"sheet": sheet_name, "row": excel_row, "study": sid, "column": str(col), "value": val, "reason": "SD cannot be negative"})
                        elif val == 0:
                            invalid_values.append({"sheet": sheet_name, "row": excel_row, "study": sid, "column": str(col), "value": val, "reason": "SD is zero; verify whether this is real or a data-entry issue"})

            # Validate sample sizes only when values are present.
            for ncol, label in ((ni_col, "intervention"), (nc_col, "control")):
                if ncol is None or _is_missing(row.get(ncol)):
                    continue
                value = _as_float(row.get(ncol))
                if value is None or value <= 0 or not float(value).is_integer():
                    invalid_ns.append({"sheet": sheet_name, "row": excel_row, "study": sid, "column": str(ncol), "value": row.get(ncol), "group": label})

            # Duplicate comparison key within a single worksheet.
            if study_col and not _is_missing(sid):
                rec = _make_record(sheet_name, frame, idx, row)
                if rec["outcome"]:
                    duplicate_keys_by_sheet.setdefault(sheet_name, {}).setdefault(rec["key"], []).append(excel_row)

    # Missing SD cells are assessed by the effect-size harmonizer below.
    # A blank SD is not automatically a blocking issue when a reported effect
    # and a valid alternative uncertainty route are already available.

    if invalid_values:
        details = "; ".join(
            f"{x['sheet']} row {x['row']} {x['column']}={x['value']} ({x['reason']})"
            for x in invalid_values[:12]
        )
        add_issue(
            "Warning", "Numerical validity", "INVALID_STANDARD_DEVIATION",
            "One or more standard deviation values need checking",
            f"MetaPy identified {len(invalid_values)} SD value(s) that are non-numeric, negative or zero. A zero SD may be valid in rare circumstances, but it requires verification.",
            "Check each affected cell against the value recorded in the researcher's dataset. Correct only a confirmed data-entry error. Do not replace a value automatically.",
            sheets_text="; ".join(sorted({x['sheet'] for x in invalid_values})),
            rows_text="; ".join(sorted({f"{x['sheet']}:{x['row']}" for x in invalid_values})),
            study_ids=", ".join(sorted({str(x['study']) for x in invalid_values})),
            columns_text=", ".join(sorted({x['column'] for x in invalid_values})),
            observed_values=details,
            rule="SDs must be finite and non-negative; zero is flagged for researcher review rather than automatically rejected.",
            impact="Affected raw-data calculations remain blocked until checked.",
        )

    if invalid_ns:
        details = "; ".join(
            f"{x['sheet']} row {x['row']} {x['column']}={x['value']}"
            for x in invalid_ns[:12]
        )
        add_issue(
            "Warning", "Numerical validity", "INVALID_SAMPLE_SIZE",
            "One or more sample sizes are invalid",
            f"MetaPy identified {len(invalid_ns)} sample-size value(s) that are missing numeric form, non-positive or non-integer.",
            "Check the affected sample-size cells against the researcher's data. Sample sizes should normally be positive whole numbers. Correct only verified entry errors.",
            sheets_text="; ".join(sorted({x['sheet'] for x in invalid_ns})),
            rows_text="; ".join(sorted({f"{x['sheet']}:{x['row']}" for x in invalid_ns})),
            study_ids=", ".join(sorted({str(x['study']) for x in invalid_ns})),
            columns_text=", ".join(sorted({x['column'] for x in invalid_ns})), observed_values=details,
            rule="Intervention and control sample sizes must be positive integers.",
            impact="Affected records cannot enter effect-size calculation or pooling until corrected.",
        )

    for sheet_name, keymap in duplicate_keys_by_sheet.items():
        duplicates = [(key, rows) for key, rows in keymap.items() if len(rows) > 1]
        if duplicates:
            details = "; ".join(f"Study {key[0]}, outcome {key[2]}, rows {rows}" for key, rows in duplicates[:10])
            add_issue(
                "Warning", "Duplicate records", "DUPLICATE_KEY_WITHIN_WORKSHEET",
                "The same study/outcome/comparison key appears more than once in one worksheet",
                f"MetaPy found {len(duplicates)} repeated study/arm/outcome/duration key(s) inside '{sheet_name}'. Some may be legitimate multiple timepoints or comparisons, but the current key does not distinguish them.",
                "Check whether the repeated rows represent different arms, timepoints, outcomes or reports. If so, add a clear comparison/timepoint identifier in the workbook before analysis. Do not delete rows automatically.",
                sheets_text=sheet_name, rows_text=details,
                rule="Within a worksheet, compare normalized study ID + arm + outcome + duration.",
                impact="Affected records need disambiguation before an analysis dataset is created.",
            )

    # C. Effect-size harmonization. This replaces the old blanket warning for
    # missing SE/variance columns: try valid reconstruction routes first.
    harmonization = harmonize_workbook(
        sheets, estimand=estimand, design=design
    )
    # Run the comprehensive uncertainty-reconstruction cascade on the
    # harmonizer's canonical records before creating readiness issues.
    harmonization = apply_reconstruction(
        harmonization, sheets, estimand=estimand, design=design
    )
    for h_issue in harmonization["issues"].to_dict("records"):
        code = h_issue.get("code", "HARMONIZATION_REVIEW")
        if code == "RECONSTRUCTION_INPUT_FIELD_CONFLICT":
            recommendation = (
                "Compare the listed worksheet cells with the researcher's original extraction. Correct only a confirmed discrepancy, then re-run the workbook review. MetaPy excluded the conflicting field from reconstruction and did not choose one worksheet value."
            )
            impact = "The affected supplementary statistic was excluded from the reconstruction payload. Review is required before the affected record can be considered classification- and calculation-ready."
            columns = "Conflicting supplementary statistical values"
            rule = "When the same canonical comparison has inconsistent repeated values across worksheet representations, exclude the conflicting field from calculations and preserve all source locations."
        elif code == "G_VARIANCE_RECONSTRUCTION_PENDING":
            recommendation = (
                "Review the record-level reconstruction attempts and the exact source cells shown in the calculation log. "
                "Supply a missing statistic only from the researcher's dataset or publication extraction. For Morris PPC2, provide valid group-specific pre/post correlations or sufficient change-score statistics to derive them; do not substitute a posttest-SMD variance. "
                "Do not enter an assumed correlation unless a separate researcher-approved sensitivity workflow is used."
            )
            impact = "The effect estimate may be available, but its sampling variance is unresolved. These records must not enter statistical pooling until a compatible uncertainty route is established."
            columns = "Effect-specific variance/SE/CI; SD(change); SE(change); paired t/p; group pre/post SDs and correlations; compatible between-group statistics"
            rule = "Attempt all supported direct and derived uncertainty routes; preserve estimand compatibility; do not silently impute missing values or substitute another estimand."
        else:
            recommendation = (
                "Review the original source cells and confirm that the selected estimand matches the protocol. MetaPy has retained all source values and calculation history; it has not averaged conflicting values."
            )
            impact = "Check the extraction discrepancy before using the affected record in an analysis."
            columns = "Conflicting repeated source fields or effect-size calculations"
            rule = "Merge compatible source rows by study ID, arm, outcome and timepoint; compare repeated values field-by-field; exclude conflicting fields from calculations; use the selected estimand for independent effect reconstruction."
        add_issue(
            h_issue.get("severity", "Warning"),
            "Effect-size harmonization",
            code,
            h_issue.get("finding", "Effect-size harmonization requires review"),
            h_issue.get("message", "MetaPy detected an effect-size issue."),
            recommendation,
            sheets_text=h_issue.get("source_locations", ""),
            rows_text=h_issue.get("source_locations", ""),
            study_ids=str(h_issue.get("study_id", "")),
            columns_text=columns,
            observed_values=f"Reported={h_issue.get('reported_g')}; calculated={h_issue.get('calculated_g')}; difference={h_issue.get('difference')}",
            rule=rule,
            impact=impact,
        )

    # D. Identify multi-arm studies with a repeated control sample size.
    unique_records: dict[tuple, dict[str, Any]] = {}
    for rec in records:
        if rec["key"] not in unique_records:
            unique_records[rec["key"]] = rec
        else:
            # Prefer detailed row for shared-control inspection, but retain all locations elsewhere.
            if rec["has_raw_measurements"] and not unique_records[rec["key"]]["has_raw_measurements"]:
                unique_records[rec["key"]] = rec

    shared_control_groups: dict[tuple, list[dict[str, Any]]] = {}
    for rec in unique_records.values():
        if rec["arm"] and not _is_missing(rec["n_control"]):
            group_key = (rec["study_id"], rec["outcome"], str(rec["duration"]))
            shared_control_groups.setdefault(group_key, []).append(rec)

    for key, group in shared_control_groups.items():
        arms = {r["arm"] for r in group}
        controls = [_as_float(r["n_control"]) for r in group]
        controls_nonmissing = [x for x in controls if x is not None]
        if len(arms) >= 2 and len(controls_nonmissing) >= 2 and len(set(controls_nonmissing)) == 1:
            locations = "; ".join(f"{r['sheet']} row {r['excel_row']} ({r['arm']}, N_control={r['n_control']})" for r in group)
            add_issue(
                "Information", "Study design / dependence", "POSSIBLE_SHARED_CONTROL_GROUP",
                "Multiple intervention arms appear to share the same control group",
                f"Study {key[0]} has {len(arms)} intervention arms for outcome '{key[1]}' and duration {key[2]}, with the same control sample size ({controls_nonmissing[0]}). This is consistent with a multi-arm study that may use one shared control group.",
                "Retain the arms as separate comparisons but identify the shared control group in the analysis structure. The later pairwise-analysis engine must use an appropriate method to account for dependence; do not treat the comparisons as fully independent by default.",
                sheets_text="; ".join(sorted({r['sheet'] for r in group})),
                rows_text=locations, study_ids=str(key[0]),
                columns_text="StudyID / arm / N_C or N_Control",
                rule="Within the same study/outcome/duration, detect two or more named intervention arms with identical non-missing control sample sizes.",
                impact="Informational structural flag; it must be handled by the dependence/multi-arm module before pooling.",
            )

    # Summary and readiness.
    inventory = pd.DataFrame(inventory_rows)
    issue_df = pd.DataFrame(issues, columns=[
        "issue_id", "severity", "category", "code", "finding", "what_it_means",
        "recommended_action", "worksheet(s)", "Excel row(s)", "study ID(s)",
        "column(s)", "observed value(s)", "rule / method", "analysis impact", "status",
    ])

    total_rows = int(sum(
        int(frame.loc[~frame.isna().all(axis=1)].shape[0]) for frame in sheets.values()
    ))
    warnings = int((issue_df["severity"] == "Warning").sum()) if not issue_df.empty else 0
    critical = int((issue_df["severity"] == "Critical").sum()) if not issue_df.empty else 0
    info = int((issue_df["severity"] == "Information").sum()) if not issue_df.empty else 0
    harmonization_issues = harmonization["issues"]
    canonical_source_conflicts = 0
    if not harmonization_issues.empty:
        source_conflict_rows = harmonization_issues[
            harmonization_issues["code"].isin(["CROSS_SHEET_FIELD_CONFLICT", "DUPLICATE_SOURCE_G_CONFLICT"])
        ]
        if not source_conflict_rows.empty:
            key_cols = [c for c in ("study_id", "arm", "outcome") if c in source_conflict_rows.columns]
            canonical_source_conflicts = int(source_conflict_rows[key_cols].drop_duplicates().shape[0]) if key_cols else len(source_conflict_rows)
    cross_sheet_conflicts = max(mismatched_pairs, canonical_source_conflicts)
    has_structural_blocker = bool(
        critical
        or (not issue_df.empty and issue_df["code"].isin([
            "CROSS_SHEET_VALUE_MISMATCH", "CROSS_SHEET_FIELD_CONFLICT", "DUPLICATE_SOURCE_G_CONFLICT", "RECONSTRUCTION_INPUT_FIELD_CONFLICT", "INVALID_SAMPLE_SIZE", "INVALID_STANDARD_DEVIATION", "DUPLICATE_KEY_WITHIN_WORKSHEET"
        ]).any())
    )
    classification_ready = not has_structural_blocker
    shared_control = bool(
        not issue_df.empty and issue_df["code"].eq("POSSIBLE_SHARED_CONTROL_GROUP").any()
    )
    h_summary = harmonization["summary"]
    effect_calculation_ready = classification_ready and bool(h_summary["harmonization_ready"])
    discrepancy_count = int(h_summary["recalculation_disagreement_count"]) + int(h_summary["duplicate_source_conflict_count"])
    pooling_ready = effect_calculation_ready and not shared_control and discrepancy_count == 0

    summary = {
        "worksheet_count": len(sheets),
        "nonempty_rows_across_sheets": total_rows,
        "recognized_record_appearances": len(records),
        "unique_record_keys": len(unique_records),
        "cross_worksheet_record_pairs": cross_sheet_pairs,
        "cross_worksheet_matched_pairs": matched_pairs,
        "cross_worksheet_mismatched_pairs": mismatched_pairs,
        "warning_count": warnings,
        "critical_count": critical,
        "information_count": info,
        "classification_ready": classification_ready,
        "effect_calculation_ready": effect_calculation_ready,
        "pooling_ready": pooling_ready,
        "harmonized_records": h_summary["canonical_record_count"],
        "effect_estimates_available": h_summary["effect_estimate_available_count"],
        "effect_estimates_unresolved": h_summary["effect_estimate_unresolved_count"],
        "recalculated_records": h_summary["independently_recalculated_count"],
        "reported_effects_retained": h_summary["reported_g_retained_count"],
        "records_with_variance": h_summary["variance_available_count"],
        "variance_reconstruction_pending": h_summary.get("reconstruction_pending_count", h_summary["variance_unresolved_count"]),
        "recalculation_disagreements": discrepancy_count,
        "estimand": estimand,
        "design_assumption": design,
        "scope_note": "Internal workbook checks only. No external study or publication verification was performed.",
    }

    return {
        "summary": summary,
        "inventory": inventory,
        "issues": issue_df,
        "records": records,
        "harmonization": harmonization,
        "readiness": {
            "classification_ready": classification_ready,
            "effect_calculation_ready": effect_calculation_ready,
            "effect_size_harmonization_ready": bool(h_summary["harmonization_ready"]),
            "pooling_ready": pooling_ready,
            "researcher_review_required": (warnings > 0 or discrepancy_count > 0),
        },
    }


# ---------------------------------------------------------------------------
# Correction helpers: write only changed cells to the working workbook copy.
# ---------------------------------------------------------------------------


def _dataframe_for_editor(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result.insert(0, "MetaPy Excel Row", [i + 2 for i in range(len(result))])
    return result


def _strip_editor_marker(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "MetaPy Excel Row" not in frame.columns:
        raise ValueError("The Excel row reference column is missing. Do not remove or rename it.")
    expected = list(range(2, len(frame) + 2))
    observed = [int(x) for x in frame["MetaPy Excel Row"].tolist()]
    if observed != expected:
        raise ValueError("Please edit cell values only. Do not add, delete, reorder or renumber rows.")
    return frame.drop(columns=["MetaPy Excel Row"]).reset_index(drop=True)


def _write_sheet_edits(
    workbook_path: str | Path,
    sheet_name: str,
    old_frame: pd.DataFrame,
    edited_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Update changed body cells only; keep workbook layout and original file intact."""
    edited = _strip_editor_marker(edited_frame)
    old = old_frame.copy().reset_index(drop=True)

    if list(edited.columns) != list(old.columns):
        raise ValueError("Do not rename, add, remove or reorder worksheet columns in this correction panel.")
    if edited.shape != old.shape:
        raise ValueError("Do not add or delete rows or columns. Edit existing cell values only.")

    changes = []
    wb = load_workbook(workbook_path, data_only=False)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Worksheet '{sheet_name}' is not present in the working workbook.")
    ws = wb[sheet_name]

    for row_idx in range(len(old)):
        for col_idx, column in enumerate(old.columns):
            before = old.iat[row_idx, col_idx]
            after = edited.iat[row_idx, col_idx]
            if _same_value(before, after):
                continue
            # Excel row 1 is the header; DataFrame row 0 maps to Excel row 2.
            cell = ws.cell(row=row_idx + 2, column=col_idx + 1)
            cell.value = _python_value(after)
            changes.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "worksheet": sheet_name,
                "Excel row": row_idx + 2,
                "column": str(column),
                "previous value": None if _is_missing(before) else str(before),
                "new value": None if _is_missing(after) else str(after),
            })

    if not changes:
        wb.close()
        return []

    wb.save(workbook_path)
    wb.close()
    return changes


def _readability_markdown(review: dict[str, Any]) -> str:
    issues = review["issues"]
    if issues.empty:
        return "## Workbook findings\n\nNo internal inconsistencies or data-quality findings were detected by the current checks. This is not proof of source authenticity or scientific correctness."

    severity_order = {"Critical": 0, "Warning": 1, "Information": 2}
    ordered = issues.assign(_order=issues["severity"].map(severity_order).fillna(9)).sort_values("_order")
    lines = ["## Findings from your workbook", ""]
    number = 0
    for _, row in ordered.iterrows():
        if row["severity"] == "Information":
            label = "Information"
        elif row["severity"] == "Critical":
            label = "Critical issue"
        else:
            label = "Warning"
        number += 1
        lines.extend([
            f"### {number}. {row['finding']}",
            f"**{label} · {row['issue_id']} · {row['status']}**",
            f"**What MetaPy found:** {row['what_it_means']}",
            f"**Where:** {row['worksheet(s)'] or 'Workbook-wide'}; Excel row(s): {row['Excel row(s)'] or 'Not applicable'}; study ID(s): {row['study ID(s)'] or 'Not identified'}.",
            f"**What to do:** {row['recommended_action']}",
            f"**Why MetaPy flagged it:** {row['rule / method']}",
            f"**Effect on analysis:** {row['analysis impact']}",
            "",
        ])
    return "\n".join(lines)


def _readiness_markdown(review: dict[str, Any], history: list[dict[str, Any]]) -> str:
    readiness = review["readiness"]
    summary = review["summary"]
    lines = ["## MetaPy — Final readiness", ""]
    if readiness["classification_ready"]:
        lines.extend([
            "### ✅ READY TO PROCEED TO FOUR-DIMENSIONAL CLASSIFICATION",
            "The current workbook passed the structural and internal-consistency checks that block classification.",
            "",
        ])
    else:
        lines.extend([
            "### ⚠️ NOT READY FOR CLASSIFICATION",
            "Resolve the blocking structural or internal-consistency findings listed above, then re-check.",
            "",
        ])

    lines.extend([
        "| Stage | Current status |",
        "|---|---|",
        f"| Four-dimensional classification | {'✅ Ready' if readiness['classification_ready'] else '⛔ Blocked'} |",
        f"| Effect-size estimates | {'✅ Available for all recognized records' if summary.get('harmonized_records', 0) == summary.get('effect_estimates_available', -1) else '⚠️ Some estimates remain unresolved'} |",
        f"| Sampling variances | {'✅ Available for all recognized records' if readiness['effect_size_harmonization_ready'] else '⚠️ Some sampling variances remain unresolved'} |",
        f"| Researcher review of warnings / discrepancies | {'⚠️ Required before final clearance' if readiness['researcher_review_required'] else '✅ No unresolved warnings'} |",
        f"| Statistical pooling | {'✅ Eligible for downstream checks' if readiness['pooling_ready'] else '⛔ Not yet cleared'} |",
        "",
        f"**Workbook:** {summary['worksheet_count']} worksheets; {summary['nonempty_rows_across_sheets']} populated rows; {summary['unique_record_keys']} recognized unique study/arm/outcome/duration keys.",
        f"**Findings:** {summary['critical_count']} critical; {summary['warning_count']} warnings; {summary['information_count']} information items.",
        f"**Corrections recorded this session:** {len(history)} cell change(s).",
        f"**Effect-size records:** {summary.get('harmonized_records', 0)} canonical records; {summary.get('recalculated_records', 0)} independently recalculated; {summary.get('reported_effects_retained', 0)} reported Hedges' g values retained; {summary.get('records_with_variance', 0)} with sampling variance.",
        f"**Uncertainty reconstruction:** {summary.get('harmonized_records', 0) - summary.get('variance_reconstruction_pending', 0)} resolved; {summary.get('variance_reconstruction_pending', 0)} pending after supported routes were attempted.",
        f"**Recalculation disagreements requiring review:** {summary.get('recalculation_disagreements', 0)}.",
        f"**Selected estimand:** {summary.get('estimand', 'Not selected')}; **design assumption:** {summary.get('design_assumption', 'Not selected')}.",
        "",
        "### Transparency and limitations",
        "- MetaPy checked only the information inside the uploaded workbook. It did not search or verify studies on journal websites, PubMed, DOI pages or external databases.",
        "- A mathematical pass means the tested values satisfy the stated calculation rule; it does not prove that the extracted values are scientifically true.",
        "- For Morris PPC2, MetaPy can calculate Hedges' g from pre/post means, baseline SDs and group Ns. Its sampling variance requires a pre/post correlation; if the workbook does not supply one, MetaPy leaves variance unresolved rather than inventing a correlation.",
        "- Missing values are not invented. If an independently valid raw-data calculation is unavailable, a reported Hedges' g is retained when present. A sampling variance is derived from g and group sample sizes only under the selected parallel independent-group assumption, unless a valid effect-specific SE, variance or CI is supplied.",
        "- When recalculated Hedges' g differs from the reported value, the independently calculated value is selected under the researcher's policy, and the discrepancy is listed for review. Values are compared only under the selected estimand; post-treatment and change-score effects must not be confused.",
        "- Incomplete alternative raw data are excluded from the calculation route, not deleted from the workbook or audit trail.",
        "- Original workbook remains unchanged. Corrections are saved to a separate working copy in this Colab runtime.",
        "- Passing classification readiness does not authorize pooling. Effect-size method, multi-arm dependence, compatibility and analysis-specific checks remain necessary.",
    ])
    return "\n".join(lines)


def _inventory_text(review: dict[str, Any]) -> str:
    s = review["summary"]
    return (
        "META PY SINGLE-WORKBOOK INSPECTION\n"
        "=================================\n"
        f"Worksheets read: {s['worksheet_count']}\n"
        f"Populated rows across worksheets: {s['nonempty_rows_across_sheets']}\n"
        f"Recognized record appearances: {s['recognized_record_appearances']}\n"
        f"Unique study/arm/outcome/duration keys: {s['unique_record_keys']}\n"
        f"Cross-worksheet matching pairs: {s['cross_worksheet_record_pairs']}\n"
        f"Matching pairs: {s['cross_worksheet_matched_pairs']}\n"
        f"Conflicting pairs: {s['cross_worksheet_mismatched_pairs']}\n\n"
        "SCOPE\n"
        "---------------------------------\n"
        "Internal workbook analysis only. No external study verification was performed.\n"
        "Consistent records across worksheets are silently combined into one canonical record; all source locations are retained in the calculation log. Only conflicting values are shown for researcher review. No source rows are deleted."
    )


def _make_session_from_upload(upload_path: str, estimand: str, design: str) -> dict[str, Any]:
    source = Path(upload_path)
    if not source.exists():
        raise FileNotFoundError("The uploaded Excel workbook could not be found.")
    if source.suffix.lower() != ".xlsx":
        raise ValueError("Please upload one .xlsx workbook. CSV cannot contain multiple worksheets.")

    work_dir = Path.cwd() / "metapy" / "working_corrections"
    work_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    working_path = work_dir / f"{source.stem}_MetaPy_working_{stamp}.xlsx"
    shutil.copy2(source, working_path)

    sheets = read_workbook(working_path)
    review = analyze_workbook(sheets, estimand=estimand, design=design)
    return {
        "source_filename": source.name,
        "working_path": str(working_path),
        "sheets": sheets,
        "review": review,
        "estimand": estimand,
        "design": design,
        "correction_history": [],
    }


# ---------------------------------------------------------------------------
# Final readiness report export (openpyxl only; avoids pandas ExcelWriter issue)
# ---------------------------------------------------------------------------


def _write_frame_to_worksheet(ws, frame: pd.DataFrame):
    from openpyxl.styles import Font, PatternFill, Alignment

    frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    for col_idx, column in enumerate(frame.columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=str(column))
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="24476B")
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row_idx, values in enumerate(frame.itertuples(index=False, name=None), start=2):
        for col_idx, value in enumerate(values, start=1):
            if _is_missing(value):
                value = None
            elif hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    value = str(value)
            if isinstance(value, (dict, list, tuple, set)):
                value = str(value)
            if not isinstance(value, (str, int, float, bool, type(None))):
                value = str(value)
            ws.cell(row=row_idx, column=col_idx, value=value)
    ws.freeze_panes = "A2"
    if frame.shape[1] > 0:
        ws.auto_filter.ref = ws.dimensions
    for col_idx, column in enumerate(frame.columns, start=1):
        values = [str(column)] + [str(x) for x in frame.iloc[:, col_idx - 1].dropna().head(100).tolist()]
        width = min(55, max(12, max((len(x) for x in values), default=12) + 2))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width


def export_readiness_report(state: dict[str, Any]) -> str:
    """Export the current validation and harmonization findings to a new xlsx."""
    if not isinstance(state, dict) or not state.get("review"):
        raise ValueError("Run the workbook inspection before exporting the readiness report.")
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    review = state["review"]
    summary = review["summary"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Final Readiness"
    ws.append(["MetaPy — Final Data Readiness Assessment", "Value"])
    ws["A1"].font = Font(bold=True, color="FFFFFF")
    ws["B1"].font = Font(bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor="24476B")
    ws["B1"].fill = PatternFill("solid", fgColor="24476B")
    for key, value in summary.items():
        ws.append([str(key), value])
    ws.freeze_panes = "A2"
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 105

    for sheet_name, frame in (
        ("Worksheet Inventory", review["inventory"]),
        ("Validation Issues", review["issues"]),
        ("Harmonized Effects", review["harmonization"]["records"]),
        ("Calculation Log", review["harmonization"]["calculation_log"]),
        ("Correction History", pd.DataFrame(state.get("correction_history", []))),
    ):
        target = wb.create_sheet(sheet_name)
        _write_frame_to_worksheet(target, frame)

    output_dir = Path(state["working_path"]).parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{Path(state['source_filename']).stem}_MetaPy_Readiness_{stamp}.xlsx"
    wb.save(output_path)
    wb.close()
    return str(output_path)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------


def build_app():
    """Build the single-workbook Gradio application."""
    import gradio as gr

    def start_review(file_path, estimand, design):
        if not file_path:
            raise gr.Error("Upload one Excel workbook first.")
        if not estimand:
            raise gr.Error("Select the effect-size estimand that matches your protocol before running MetaPy.")
        if not design:
            raise gr.Error("Select the study-design assumption before running MetaPy.")
        try:
            state = _make_session_from_upload(file_path, estimand, design)
        except Exception as exc:
            raise gr.Error(f"MetaPy could not read the workbook: {exc}")

        review = state["review"]
        sheet_names = list(state["sheets"].keys())
        default_sheet = sheet_names[0] if sheet_names else None
        editor_value = _dataframe_for_editor(state["sheets"][default_sheet]) if default_sheet else pd.DataFrame()
        return (
            state,
            _inventory_text(review),
            review["inventory"],
            review["issues"],
            _readability_markdown(review),
            _readability_markdown(review),
            gr.update(choices=sheet_names, value=default_sheet),
            editor_value,
            f"Selected worksheet: {default_sheet}. Edit cell values only; do not add/delete/reorder rows or columns.",
            _readiness_markdown(review, state["correction_history"]),
            pd.DataFrame(columns=["timestamp", "worksheet", "Excel row", "column", "previous value", "new value"]),
            review["harmonization"]["records"],
            review["harmonization"]["calculation_log"],
        )

    def load_selected_sheet(state, sheet_name):
        if not isinstance(state, dict) or not state.get("sheets"):
            raise gr.Error("Run the workbook inspection first.")
        if sheet_name not in state["sheets"]:
            raise gr.Error("Select a worksheet from the uploaded workbook.")
        return (
            _dataframe_for_editor(state["sheets"][sheet_name]),
            f"Selected worksheet: {sheet_name}. Excel row numbers are shown in the first column. Edit existing values only.",
        )

    def apply_and_recheck(state, sheet_name, editor_frame):
        if not isinstance(state, dict) or not state.get("sheets"):
            raise gr.Error("Upload and inspect the workbook before correcting values.")
        if sheet_name not in state["sheets"]:
            raise gr.Error("Select a valid worksheet.")
        if editor_frame is None:
            raise gr.Error("No worksheet table was supplied for correction.")

        try:
            edited = editor_frame if isinstance(editor_frame, pd.DataFrame) else pd.DataFrame(editor_frame)
            old_frame = state["sheets"][sheet_name]
            changes = _write_sheet_edits(
                state["working_path"], sheet_name, old_frame, edited
            )
            if not changes:
                raise ValueError("No cell changes were detected. Edit a value before re-checking.")

            # Reload every worksheet after editing one sheet; checks are workbook-wide.
            new_sheets = read_workbook(state["working_path"])
            new_review = analyze_workbook(new_sheets, estimand=state["estimand"], design=state["design"])
        except Exception as exc:
            raise gr.Error(f"MetaPy could not apply/re-check the correction: {exc}")

        state["sheets"] = new_sheets
        state["review"] = new_review
        state["correction_history"] = list(state.get("correction_history", [])) + changes

        new_editor = _dataframe_for_editor(new_sheets[sheet_name])
        history = pd.DataFrame(state["correction_history"])
        return (
            state,
            _inventory_text(new_review),
            new_review["inventory"],
            new_review["issues"],
            _readability_markdown(new_review),
            new_editor,
            f"Re-checked all {len(new_sheets)} worksheets after {len(changes)} correction(s) in '{sheet_name}'.",
            _readiness_markdown(new_review, state["correction_history"]),
            history,
            new_review["harmonization"]["records"],
            new_review["harmonization"]["calculation_log"],
        )

    with gr.Blocks(title="MetaPy — Single Workbook Review") as demo:
        session_state = gr.State(value=None)

        gr.Markdown(
            """
            # MetaPy — Single Workbook Review

            **One research workbook. All worksheets read together. One connected set of findings.**

            MetaPy checks the researcher's own study IDs, values, worksheet relationships and calculations.
            It does not search for or externally verify individual studies.

            The uploaded original is never modified. Researcher corrections are made in a separate working copy.
            """
        )

        with gr.Tab("1. Upload & Inspect"):
            workbook_file = gr.File(
                label="Upload one research workbook (.xlsx)",
                file_types=[".xlsx"],
                type="filepath",
            )
            estimand_selector = gr.Dropdown(
                label="Continuous-outcome estimand for independent recalculation",
                choices=[
                    ("Post-treatment between-group SMD (Hedges' g)", "posttest_smd"),
                    ("Morris (2008) PPC2: difference in pre/post change / pooled baseline SD", "morris_ppc2"),
                    ("Between-group change-score SMD (requires pre/post correlations)", "change_smd"),
                    ("Do not recalculate from raw means; retain reported g or convert reported Cohen's d", "reported_only"),
                ],
                value=None,
            )
            design_selector = gr.Dropdown(
                label="Study design assumption for variance calculation",
                choices=[
                    ("Parallel independent groups (confirm this matches the study design)", "parallel_independent"),
                    ("Paired/crossover/repeated or other dependent design — do not use independent-group variance", "dependent_or_unknown"),
                ],
                value=None,
            )
            gr.Markdown(
                "**Researcher confirmation required:** choose the estimand that matches your protocol and the method used for any already-reported Hedges' g. Morris PPC2 can calculate g from pre/post means, baseline SDs and group Ns; its variance additionally requires a pre/post correlation. MetaPy will not invent that correlation. Post-treatment, Morris PPC2 and change-score-SD effects are not interchangeable. Shared-control dependence is flagged separately for later pooling."
            )
            run_button = gr.Button("Read all worksheets, harmonize effects & inspect readiness", variant="primary")
            workbook_summary = gr.Textbox(label="Workbook inspection summary", lines=12, interactive=False)
            inventory = gr.Dataframe(label="Worksheet inventory", interactive=False, wrap=True)
            issue_table = gr.Dataframe(label="All findings — one workbook-wide register", interactive=False, wrap=True)
            initial_findings = gr.Markdown()

        with gr.Tab("2. Correct & Re-check"):
            findings_to_fix = gr.Markdown(
                "Run the workbook inspection first. MetaPy will display all findings here in plain language."
            )
            gr.Markdown(
                """
                ### Correct the actual workbook values

                1. Read the findings above; each finding gives worksheet names, Excel row numbers and columns.
                2. Select the worksheet containing the value to correct.
                3. Edit the value directly in the table below.
                4. Click **Apply correction & re-check all worksheets**.

                **Edit values only.** Do not add, delete, rename or reorder rows/columns. MetaPy will re-read every worksheet after each correction and rebuild the complete findings register.
                """
            )
            sheet_selector = gr.Dropdown(label="Worksheet to correct", choices=[], value=None)
            sheet_info = gr.Textbox(label="Selected worksheet instructions", interactive=False)
            sheet_editor = gr.Dataframe(
                label="Editable worksheet — working copy only",
                value=pd.DataFrame(),
                type="pandas",
                interactive=True,
                wrap=True,
                max_height=420,
            )
            apply_button = gr.Button("Apply correction & re-check all worksheets", variant="primary")
            refreshed_findings = gr.Markdown(label="Updated workbook findings")

        with gr.Tab("3. Effect-Size Harmonization"):
            gr.Markdown(
                "MetaPy creates one canonical record per study/arm/outcome/timepoint. Consistent data repeated across worksheets are merged silently; complementary values are combined field-by-field. Only conflicting repeated values require review. MetaPy prefers a valid independent raw-data reconstruction when the selected estimand can be calculated; otherwise it retains reported Hedges' g."
            )
            harmonized_table = gr.Dataframe(label="Canonical harmonized effect-size records", interactive=False, wrap=True)
            calculation_log_table = gr.Dataframe(label="Calculation and source-provenance log", interactive=False, wrap=True)

        with gr.Tab("4. Final Readiness"):
            final_status = gr.Markdown(
                "Upload and inspect one workbook to see whether it is ready for classification and effect-size harmonization."
            )
            correction_history = gr.Dataframe(
                label="Cell-level correction history",
                value=pd.DataFrame(columns=["timestamp", "worksheet", "Excel row", "column", "previous value", "new value"]),
                interactive=False,
                wrap=True,
            )
            export_button = gr.Button("Export final readiness report (.xlsx)", variant="primary")
            export_file = gr.File(label="Generated readiness report", interactive=False)

        run_button.click(
            fn=start_review,
            inputs=[workbook_file, estimand_selector, design_selector],
            outputs=[
                session_state, workbook_summary, inventory, issue_table,
                initial_findings, findings_to_fix, sheet_selector, sheet_editor, sheet_info,
                final_status, correction_history, harmonized_table, calculation_log_table,
            ],
        )

        sheet_selector.change(
            fn=load_selected_sheet,
            inputs=[session_state, sheet_selector],
            outputs=[sheet_editor, sheet_info],
        )

        apply_button.click(
            fn=apply_and_recheck,
            inputs=[session_state, sheet_selector, sheet_editor],
            outputs=[
                session_state, workbook_summary, inventory, issue_table,
                refreshed_findings, sheet_editor, sheet_info,
                final_status, correction_history, harmonized_table, calculation_log_table,
            ],
        )

        export_button.click(
            fn=export_readiness_report,
            inputs=[session_state],
            outputs=[export_file],
        )

        gr.Markdown(
            """
            ---
            **Transparency:** Consistent records repeated across worksheets are silently merged into one canonical analytical record, with all source cells retained in the calculation provenance. Complementary fields from different worksheets can be combined. Only conflicting repeated values are sent for review; conflicting fields are excluded from calculations until resolved. A mathematically consistent value is not proof that the underlying extraction is scientifically true.

            **Readiness:** Internal validation, effect-size harmonization and pooling are separate gates. Passing the first two does not automatically approve pooling.
            """
        )

    return demo


__all__ = [
    "read_workbook",
    "analyze_workbook",
    "build_app",
]
