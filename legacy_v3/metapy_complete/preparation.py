from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .schema import as_float, is_missing, normalize_direction


def _is_parallel_design(value: Any) -> bool:
    if is_missing(value):
        return False
    text = str(value).strip().lower()
    if "crossover" in text or "cluster" in text or "within-subject" in text or "within subject" in text:
        return False
    return ("parallel" in text and ("independent" in text or "group" in text or "random" in text)) or text in {
        "parallel", "parallel-group", "parallel group", "parallel randomized controlled trial",
        "parallel rct", "parallel controlled trial", "parallel controlled study",
    }


def _is_known_unsupported_design(value: Any) -> bool:
    if is_missing(value):
        return False
    text = str(value).strip().lower()
    return any(token in text for token in (
        "crossover", "cross-over", "cluster", "within-subject", "within subject",
        "single-arm", "single arm", "one-group", "one group", "before-after", "before after",
    ))


def prepare_records(
    canonical: pd.DataFrame,
    *,
    global_decisions: dict[str, Any] | None = None,
    record_approvals: dict[str, bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create transparent prep statuses, record-level approvals and audit logs.

    Global decisions and row-level approvals are deliberately separate.
    """
    global_decisions = global_decisions or {}
    record_approvals = record_approvals or {}
    if canonical.empty:
        empty_summary = pd.DataFrame([{
            "canonical_comparisons": 0,
            "global_design_confirmation": bool(global_decisions.get("parallel_independent_groups_confirmed")),
            "record_level_approvals": 0,
            "record_level_reviews_pending": 0,
            "records_with_actual_both_group_correlations": 0,
            "records_with_source_unit": 0,
            "records_with_field_conflicts": 0,
            "preparation_status": "No canonical comparisons created",
            "calculation_status": "Not yet executed",
            "pooling_status": "Not yet executed",
        }])
        return canonical.copy(), pd.DataFrame(), pd.DataFrame(), empty_summary

    prepared = canonical.copy()
    findings: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for idx, row in prepared.iterrows():
        rid = row["record_id"]
        study = row.get("study_id")
        outcome = row.get("outcome")
        arm = row.get("intervention_arm")
        issues: list[str] = []
        # A missing arm label is only blocking when there are multiple canonical
        # comparisons within the same study/outcome/timepoint and the arm cannot
        # be distinguished. A single comparison is retained as one unspecified arm.
        group_mask = (
            (prepared["study_id"].astype(str) == str(study))
            & (prepared["outcome"].astype(str) == str(outcome))
        )
        if "duration" in prepared:
            dur = row.get("duration")
            group_mask = group_mask & (prepared["duration"].fillna("__NA__").astype(str) == ("__NA__" if is_missing(dur) else str(dur)))
        comparison_count = int(group_mask.sum())
        if bool(row.get("arm_ambiguity_detected", False)):
            arm_status = "Ambiguous — multiple unlabelled comparisons retained separately"
            issues.append("intervention_arm_not_specified")
        elif is_missing(arm) or str(arm).casefold() in {"not specified", "nan"}:
            if comparison_count <= 1:
                arm_status = "Single comparison only; arm label not supplied"
            else:
                arm_status = "Not supplied — arm structure requires review"
                issues.append("intervention_arm_not_specified")
        else:
            arm_status = "Recognized — supplied field or explicitly embedded label"

        # Apply an explicit per-record override only when the researcher enters it.
        method_override = global_decisions.get("record_effect_method_overrides", {}).get(rid)
        if not is_missing(method_override):
            prepared.at[idx, "reported_effect_method"] = str(method_override).strip()
            log_rows.append({
                "record_id": rid,
                "study_id": study,
                "outcome": outcome,
                "action": f"Researcher confirmed reported effect method as {str(method_override).strip()}",
                "source_locations": "Researcher review table",
            })

        # Units are treated as declared metadata, not externally verified.
        unit = row.get("unit")
        if not is_missing(unit):
            unit_status = "Single unit declared for this comparison; no within-record mismatch detected"
            unit_note = "This does not independently verify that the unit is correct for both groups or source publication."
        else:
            unit_status = "No unit supplied"
            unit_note = "A unit is not required for standardized mean differences, but is required for interpretable raw mean differences."

        # Actual group-specific source correlations always take priority.
        r_i = as_float(row.get("r_prepost_i"))
        r_c = as_float(row.get("r_prepost_c"))
        if (r_i is not None and not -1 <= r_i <= 1) or (r_c is not None and not -1 <= r_c <= 1):
            corr_status = "Invalid source correlation detected — value must be between -1 and +1"
            corr_source = "invalid source value"
            issues.append("prepost_correlation_out_of_range")
        elif r_i is not None and r_c is not None:
            corr_status = "Both group-specific pre/post correlations detected in source workbook"
            corr_source = "source workbook"
        elif r_i is not None or r_c is not None:
            corr_status = "Only one group-specific pre/post correlation detected; other group value unresolved"
            corr_source = "partial source workbook"
            issues.append("one_prepost_correlation_missing")
        else:
            corr_status = "No source pre/post correlations detected"
            corr_source = "none"

        # Explicit outcome direction or a researcher-provided global decision.
        raw_direction = row.get("higher_is_worse")
        direction_multiplier = normalize_direction(raw_direction)
        direction_source = "record field" if direction_multiplier is not None else None
        record_direction = global_decisions.get("record_direction_multipliers", {}).get(rid)
        if direction_multiplier is None and record_direction in (-1, 1):
            direction_multiplier = int(record_direction)
            direction_source = "researcher record-level direction decision"
        global_direction = global_decisions.get("outcome_direction", {})
        if direction_multiplier is None and isinstance(global_direction, dict):
            candidate = global_direction.get(str(outcome)) or global_direction.get(str(outcome).casefold())
            if not is_missing(candidate):
                direction_multiplier = int(candidate)
                direction_source = "researcher global outcome decision"
        if direction_multiplier is None:
            direction_status = "Pending — outcome direction not confirmed"
            issues.append("outcome_direction_pending")
        else:
            direction_status = "Confirmed — benefit-oriented sign convention recorded"

        # Explicit study design takes priority; a researcher confirmation is a
        # separately recorded global choice and is not presented as source data.
        design_value = row.get("study_design")
        design_confirmed = _is_parallel_design(design_value)
        known_unsupported_design = _is_known_unsupported_design(design_value)
        design_source = "source workbook" if design_confirmed else None
        record_design = global_decisions.get("record_design_confirmations", {}).get(rid)
        # A researcher can classify missing or vague design metadata. Explicit
        # crossover/cluster/single-arm designs cannot be overwritten by a
        # generic parallel checkbox; correct the source field only if it is wrong.
        if not design_confirmed and not known_unsupported_design and bool(record_design):
            design_confirmed = True
            design_source = "researcher record-level design decision"
        if not design_confirmed and not known_unsupported_design and bool(global_decisions.get("parallel_independent_groups_confirmed")):
            design_confirmed = True
            design_source = "researcher global project decision"
        if design_confirmed:
            design_status = "Parallel independent groups — confirmed"
        elif known_unsupported_design:
            design_status = "Unsupported design — not analysed in v3.3.0"
            issues.append("study_design_unsupported")
        else:
            design_status = "Pending — design needs researcher classification"
            issues.append("study_design_pending")

        # Conflict-free data only. Row-level review is separate from global forms.
        if bool(row.get("has_field_conflict")):
            conflict_status = "Blocked — conflicting repeated source values"
            issues.append("source_field_conflict")
        else:
            conflict_status = "No cross-sheet field conflict detected"

        record_review = bool(record_approvals.get(rid, False))
        record_review_status = "Approved by researcher at record level" if record_review else "Pending — individual comparison not yet approved"
        if not record_review:
            issues.append("record_level_review_pending")

        raw_effect_present = all(not is_missing(row.get(k)) for k in ["mean_pre_i", "mean_post_i", "mean_pre_c", "mean_post_c", "sd_pre_i", "sd_pre_c", "n_i", "n_c"])
        post_smd_present = all(not is_missing(row.get(k)) for k in ["mean_post_i", "mean_post_c", "sd_post_i", "sd_post_c", "n_i", "n_c"])
        binary_present = all(not is_missing(row.get(k)) for k in ["events_i", "total_i", "events_c", "total_c"])

        prepared.at[idx, "intervention_arm_status"] = arm_status
        prepared.at[idx, "unit_status"] = unit_status
        prepared.at[idx, "unit_note"] = unit_note
        prepared.at[idx, "r_prepost_i_source"] = r_i
        prepared.at[idx, "r_prepost_c_source"] = r_c
        prepared.at[idx, "correlation_source_status"] = corr_status
        prepared.at[idx, "correlation_source_type"] = corr_source
        prepared.at[idx, "study_design_status"] = design_status
        prepared.at[idx, "study_design_confirmation_source"] = design_source or "not confirmed"
        prepared.at[idx, "effect_direction_multiplier"] = direction_multiplier
        prepared.at[idx, "effect_direction_status"] = direction_status
        prepared.at[idx, "effect_direction_source"] = direction_source or "not confirmed"
        prepared.at[idx, "record_review_status"] = record_review_status
        prepared.at[idx, "record_review_approved"] = record_review
        prepared.at[idx, "raw_morris_ppc2_candidate"] = raw_effect_present
        prepared.at[idx, "raw_posttest_smd_candidate"] = post_smd_present
        prepared.at[idx, "binary_pairwise_candidate"] = binary_present
        prepared.at[idx, "preparation_issues"] = "; ".join(issues)
        prepared.at[idx, "preparation_status"] = (
            "Prepared for analysis-stage review" if not issues else "Prepared with outstanding researcher review items"
        )

        if issues:
            for issue in issues:
                finding_type = "Blocker" if issue in {"source_field_conflict", "study_design_pending", "study_design_unsupported", "outcome_direction_pending", "intervention_arm_not_specified"} else "Review item"
                findings.append({
                    "record_id": rid,
                    "study_id": study,
                    "outcome": outcome,
                    "intervention_arm": arm,
                    "finding_type": finding_type,
                    "finding_code": issue,
                    "description": _issue_description(issue),
                    "effect_on_analysis": _issue_effect(issue),
                    "status": "Pending" if not record_review else "Needs resolution before pooling",
                })

        # Exact transformation log, only for actions that actually happened.
        prov = row.get("field_provenance_json")
        try:
            source_prov = json.loads(prov) if isinstance(prov, str) else {}
        except json.JSONDecodeError:
            source_prov = {}
        actions = []
        if len(row.get("source_sheets", [])) > 1:
            actions.append(f"Merged complementary source fields from {len(row.get('source_sheets', []))} worksheets")
        if str(arm_status).startswith("Recognized"):
            actions.append(f"Recorded intervention arm: {arm}")
        if r_i is not None or r_c is not None:
            actions.append(f"Carried source correlation values: r_I={r_i}, r_C={r_c}")
        if not is_missing(unit):
            actions.append(f"Carried declared unit: {unit}")
        if record_review:
            actions.append("Researcher approved this canonical comparison at record level")
        for action in actions:
            log_rows.append({
                "record_id": rid,
                "study_id": study,
                "outcome": outcome,
                "action": action,
                "source_locations": "; ".join(sorted({x for vals in source_prov.values() for x in vals})),
            })

    approvals_pending = int((~prepared["record_review_approved"].astype(bool)).sum())
    nonapproval_issues = prepared["preparation_issues"].fillna("").apply(
        lambda value: any(code.strip() and code.strip() != "record_level_review_pending" for code in str(value).split(";"))
    )
    records_with_other_review_items = int(nonapproval_issues.sum())
    if approvals_pending:
        prep_status = f"Prepared — {approvals_pending} individual comparison approval(s) pending"
        if records_with_other_review_items:
            prep_status += f"; {records_with_other_review_items} comparison(s) also have other review items"
    elif records_with_other_review_items:
        prep_status = f"Prepared — all individual approvals complete; {records_with_other_review_items} comparison(s) still have other preparation review items"
    else:
        prep_status = "Prepared — all preparation review items complete; calculation and pooling gates remain separate"

    summary = pd.DataFrame([{
        "canonical_comparisons": len(prepared),
        "global_design_confirmation": bool(global_decisions.get("parallel_independent_groups_confirmed")),
        "record_level_approvals": int(prepared["record_review_approved"].sum()),
        "record_level_reviews_pending": approvals_pending,
        "records_with_other_preparation_review_items": records_with_other_review_items,
        "records_with_actual_both_group_correlations": int((prepared["r_prepost_i_source"].notna() & prepared["r_prepost_c_source"].notna()).sum()),
        "records_with_source_unit": int(prepared["unit"].notna().sum()) if "unit" in prepared else 0,
        "records_with_field_conflicts": int(prepared.get("has_field_conflict", pd.Series(dtype=bool)).fillna(False).sum()),
        "preparation_status": prep_status,
        "calculation_status": "Not yet executed",
        "pooling_status": "Not yet executed",
    }])
    return prepared, pd.DataFrame(findings), pd.DataFrame(log_rows), summary


def _issue_description(code: str) -> str:
    return {
        "intervention_arm_not_specified": "No explicit intervention-arm label was mapped. This is especially important for multi-arm studies.",
        "one_prepost_correlation_missing": "Only one group's pre/post correlation is present. The missing group value is not imputed.",
        "prepost_correlation_out_of_range": "A source correlation is outside the mathematically valid range -1 to +1. It must be checked against the supplied extraction.",
        "outcome_direction_pending": "The researcher has not confirmed which direction represents improvement for this outcome.",
        "study_design_pending": "A parallel independent-group design has not been confirmed for this comparison.",
        "source_field_conflict": "Repeated source rows contain conflicting values for one or more fields. Conflicting fields were excluded from calculations.",
        "record_level_review_pending": "Global project decisions do not approve each comparison. This row still needs its own researcher review.",
    }.get(code, code)


def _issue_effect(code: str) -> str:
    return {
        "intervention_arm_not_specified": "The comparison may be ambiguous if more than one intervention arm is present.",
        "one_prepost_correlation_missing": "Morris PPC2 point estimation may remain possible, but a correlation-based sampling variance is not fully reconstructed.",
        "prepost_correlation_out_of_range": "A correlation-based variance cannot be calculated from this value until corrected.",
        "outcome_direction_pending": "The effect sign cannot yet be reliably oriented to the selected benefit convention.",
        "study_design_pending": "Independent-group formula selection is blocked until design is confirmed.",
        "study_design_unsupported": "No independent parallel-group effect-size formula or primary pooling is applied to this design in v3.3.0.",
        "source_field_conflict": "The affected field cannot be used until the researcher resolves the conflict.",
        "record_level_review_pending": "The comparison cannot enter primary pooling until individually approved.",
    }.get(code, "Requires researcher review.")
