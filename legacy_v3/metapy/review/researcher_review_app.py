"""Guided, selection-based Gradio workbench for MetaPy researcher decisions.

This interface deliberately avoids free-text researcher response fields. The
researcher selects from predefined options and structured form controls.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import tempfile
from typing import Any

import pandas as pd

from .researcher_decision_engine import (
    VERSION as DECISION_VERSION,
    EXCLUSION_REASON_OPTIONS,
    build_decision_queue,
    validate_researcher_decisions,
    decision_summary,
    export_decision_audit,
    build_approved_analysis_plan,
)


def _safe(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _details(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(_safe(value) or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}



def _queue_to_state(queue: pd.DataFrame) -> str:
    """Serialize the decision queue for reliable per-session Gradio state."""
    if queue is None or queue.empty:
        return "[]"
    return queue.to_json(orient="records", force_ascii=False)


def _queue_from_state(value: Any) -> pd.DataFrame:
    """Restore a decision queue from serialized state (or accept legacy frames)."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, str) and value.strip():
        try:
            records = json.loads(value)
            return pd.DataFrame(records if isinstance(records, list) else [])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _single_choice(value: Any) -> str:
    """Normalize a single-select control value without accepting multiple choices."""
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _safe(value[0])
        if len(value) == 0:
            return ""
        raise ValueError("Please select only one option in this field.")
    return _safe(value)


# Gradio validates choice components against their configured choices even when
# they are hidden. Returning an empty string for an unselected Radio/Dropdown can
# therefore abort a later event before its callback runs. Use None for no choice.
def _ui_choice_value(value: Any, allowed_values: set[str], default: Any = None) -> Any:
    normalized = _single_choice(value)
    if not normalized:
        return default
    return normalized if normalized in allowed_values else default


_CORRELATION_SOURCE_VALUES = {
    "reported_per_study", "calculated_per_study", "author_provided_per_study"
}
_REPORTED_METHOD_VALUES = {
    "not_checked", "confirmed_ppc2", "confirmed_change_smd", "confirmed_post_smd",
    "confirmed_ancova", "unclear"
}
_CHANGE_TYPE_VALUES = {"sd_change", "se_change", "change_ci", "paired_t", "paired_p"}
_CHANGE_SOURCE_VALUES = {"original_report", "author_provided", "documented_calculation"}
_PROTOCOL_ALIGNMENT_VALUES = {"registered_protocol", "documented_amendment", "not_sure"}
_ARM_SIMILARITY_VALUES = {"similar", "not_similar", "uncertain"}
_SENSITIVITY_SCOPE_VALUES = {"common_grid", "study_specific_grid"}
_SENSITIVITY_VALUE_VALUES = {0.1, 0.3, 0.5, 0.7, 0.9}
_EXCLUSION_REASON_VALUES = {value for _, value in EXCLUSION_REASON_OPTIONS}


def _ui_multi_choice_values(value: Any, allowed_values: set[Any]) -> list[Any]:
    """Return only valid checkbox-group values, in the allowed-option order."""
    if value is None:
        selected = set()
    elif isinstance(value, (list, tuple, set)):
        selected = set(value)
    else:
        selected = {value}
    return [item for item in (0.1, 0.3, 0.5, 0.7, 0.9) if item in allowed_values and item in selected]


def _save_decision(
    queue: pd.DataFrame, decision_id: str, choice: Any, status: Any,
    reported_method_status: Any, ppc2_ack: Any, correlation_source: Any,
    change_type: Any, change_source: Any, sensitivity_values: Any,
    sensitivity_scope: Any, sensitivity_ack: Any, exclusion_reason: Any,
    protocol_alignment: Any, arm_similarity: Any,
) -> tuple[pd.DataFrame, str]:
    """Update one decision row while preserving all decisions already saved."""
    if queue is None or queue.empty:
        raise ValueError("Inspect a workbook first.")
    decision_id = _single_choice(decision_id)
    if not decision_id:
        raise ValueError("Select a decision first.")

    updated = queue.copy(deep=True)
    matches = updated.index[updated["Decision ID"].astype(str).eq(decision_id)].tolist()
    if not matches:
        raise ValueError("The selected decision was not found. Please inspect the workbook again.")
    idx = matches[0]
    row = updated.loc[idx]
    choice_value = _single_choice(choice)
    status_value = _single_choice(status) or "Pending"

    try:
        available = json.loads(_safe(row.get("Available options")) or "[]")
    except Exception:
        available = []
    allowed = {str(item.get("value")) for item in available if isinstance(item, dict)}
    if choice_value and choice_value not in allowed:
        raise ValueError("The selected option is not available for this decision. Please choose again.")
    if status_value not in {"Pending", "Approved", "Rejected", "Deferred"}:
        raise ValueError("Choose Pending, Approved, Rejected or Deferred.")
    if status_value == "Approved" and not choice_value:
        raise ValueError("Select an option before approving this decision.")

    category = _safe(row.get("Category"))
    updated.at[idx, "Your choice"] = choice_value
    updated.at[idx, "Status"] = status_value
    updated.at[idx, "Exclusion reason"] = _single_choice(exclusion_reason)

    # Preserve existing structured fields, then replace only the fields governed
    # by this decision category. Other decision rows are never reconstructed.
    details = _details(row.get("Decision details (structured)"))
    if category == "Reported effect-size method":
        for key in ("reported_method_status", "acknowledge_unverified_ppc2",
                    "reported_g_method_status", "assumption_warning"):
            details.pop(key, None)
        if choice_value == "confirm_from_source":
            details["reported_method_status"] = _single_choice(reported_method_status) or "not_checked"
        elif choice_value == "assume_ppc2_unverified":
            details["reported_method_status"] = "not_checked"
            details["acknowledge_unverified_ppc2"] = bool(ppc2_ack)
            if status_value == "Approved":
                details["reported_g_method_status"] = "assumed_morris_ppc2_unverified"
                details["assumption_warning"] = "The workbook does not independently confirm the reported Hedges' g calculation method."

    elif category == "Sampling variance plan":
        for key in ("correlation_source", "change_information_type", "change_information_source",
                    "correlation_values", "sensitivity_scope", "acknowledge_assumed_correlations"):
            details.pop(key, None)
        if choice_value == "provide_actual_correlation":
            details["correlation_source"] = _single_choice(correlation_source)
        elif choice_value == "provide_change_sd_or_se":
            details["change_information_type"] = _single_choice(change_type)
            details["change_information_source"] = _single_choice(change_source)
        elif choice_value == "approved_correlation_sensitivity":
            raw_values = sensitivity_values or []
            if not isinstance(raw_values, (list, tuple)):
                raw_values = [raw_values]
            details["correlation_values"] = list(raw_values)
            details["sensitivity_scope"] = _single_choice(sensitivity_scope)
            details["acknowledge_assumed_correlations"] = bool(sensitivity_ack)

    elif category == "Primary analysis method":
        details["protocol_alignment"] = _single_choice(protocol_alignment) or "registered_protocol"

    elif category == "Multi-arm study dependence":
        details["arm_similarity"] = _single_choice(arm_similarity)

    updated.at[idx, "Decision details (structured)"] = json.dumps(details, ensure_ascii=False, sort_keys=True)
    if "Decision recorded (UTC)" in updated.columns:
        updated.at[idx, "Decision recorded (UTC)"] = datetime.now(timezone.utc).isoformat()

    category_label = category or "Decision"
    message = f"Saved: **{category_label}** — status **{status_value}**. This decision is retained while you review the others."
    return updated, message


def _decision_rows(queue: pd.DataFrame) -> list[tuple[str, str]]:
    if queue is None or queue.empty:
        return []
    choices = []
    for _, row in queue.iterrows():
        label = f"{_safe(row.get('Category'))} — {_safe(row.get('Subject'))}"
        choices.append((label, _safe(row.get("Decision ID"))))
    return choices


def _summary_view(queue: pd.DataFrame) -> pd.DataFrame:
    if queue is None or queue.empty:
        return pd.DataFrame(columns=["Decision", "Selected option", "Status"])
    labels = {}
    for _, row in queue.iterrows():
        try:
            options = json.loads(_safe(row.get("Available options")) or "[]")
            labels.update({str(o.get("value")): str(o.get("label")) for o in options if isinstance(o, dict)})
        except Exception:
            pass
    out = pd.DataFrame({
        "Decision": queue.get("Category", pd.Series(dtype=str)).astype(str) + " — " + queue.get("Subject", pd.Series(dtype=str)).astype(str),
        "Selected option": queue.get("Your choice", pd.Series(dtype=str)).map(lambda x: labels.get(_safe(x), "Not selected")),
        "Status": queue.get("Status", pd.Series(dtype=str)).fillna("Pending").astype(str),
    })
    return out


def _row_for(decision_id: str, queue: pd.DataFrame):
    if queue is None or queue.empty or not decision_id:
        return None
    matches = queue.index[queue["Decision ID"].astype(str).eq(str(decision_id))].tolist()
    return queue.loc[matches[0]] if matches else None


def _option_choices(row) -> list[tuple[str, str]]:
    try:
        options = json.loads(_safe(row.get("Available options")) or "[]")
    except Exception:
        options = []
    return [(str(o.get("label", o.get("value", ""))), str(o.get("value", "")))
            for o in options if isinstance(o, dict)]


def _detail_markdown(row) -> str:
    if row is None:
        return "Select a decision above to see what MetaPy found and what each choice means."
    return (
        f"### {row.get('Category', 'Decision')}\n\n"
        f"**Item:** {row.get('Subject', '')}\n\n"
        f"**What MetaPy found**\n\n{row.get('What MetaPy found (simple words)', '')}\n\n"
        f"**Why it matters**\n\n{row.get('Why it matters', '')}\n\n"
        f"**MetaPy's suggestion**\n\n{row.get('MetaPy suggestion', '')}"
    )


def _visibility(category: str, choice: str) -> dict[str, bool]:
    return {
        "reported_method_status": category == "Reported effect-size method" and choice == "confirm_from_source",
        "ppc2_ack": category == "Reported effect-size method" and choice == "assume_ppc2_unverified",
        "correlation_source": category == "Sampling variance plan" and choice == "provide_actual_correlation",
        "change_type": category == "Sampling variance plan" and choice == "provide_change_sd_or_se",
        "change_source": category == "Sampling variance plan" and choice == "provide_change_sd_or_se",
        "sensitivity_values": category == "Sampling variance plan" and choice == "approved_correlation_sensitivity",
        "sensitivity_scope": category == "Sampling variance plan" and choice == "approved_correlation_sensitivity",
        "sensitivity_ack": category == "Sampling variance plan" and choice == "approved_correlation_sensitivity",
        "exclusion_reason": category == "Study eligibility" and choice == "exclude_with_reason",
        "protocol_alignment": category == "Primary analysis method" and choice in {"morris_ppc2", "change_smd", "ancova_giv"},
        "arm_similarity": category == "Multi-arm study dependence" and choice == "combine_arms",
    }


def _choice_explanation(row, choice: str) -> str:
    if row is None or not choice:
        return "Select an option to see its explanation."
    try:
        options = json.loads(_safe(row.get("Available options")) or "[]")
    except Exception:
        options = []
    for option in options:
        if isinstance(option, dict) and str(option.get("value")) == str(choice):
            return f"**Selected option:** {option.get('label', choice)}\n\n{option.get('explanation', '')}"
    return "Select one of the listed options."


def build_researcher_review_app():
    """Build an upload -> researcher decisions -> preparation review app."""
    import gradio as gr
    from .single_workbook_workflow import _make_session_from_upload
    from ..preparation.data_preparation_engine import (
        build_preparation_bundle, save_preparation_review,
        validate_preparation_bundle, export_preparation_workbook,
        bundle_to_state, bundle_from_state,
        correction_source_options, correction_row_column_options,
        apply_approved_cell_correction,
        VERSION as PREPARATION_VERSION,
    )

    def inspect_workbook(file_path, design, provisional_ppc2):
        if not file_path:
            raise gr.Error("Please upload one Excel workbook (.xlsx).")
        if not design:
            raise gr.Error("Please select the study-design context.")
        try:
            state = _make_session_from_upload(file_path, "auto", design)
            review = state["review"]
            queue = build_decision_queue(review, assume_reported_g_is_ppc2=bool(provisional_ppc2))
            overview = (
                "## Workbook screening complete\n\n"
                f"- Worksheets read: **{len(state.get('sheets', {}))}**\n"
                f"- Decision items prepared: **{len(queue)}**\n"
                f"- Provisional reported-g-as-PPC2 assumption: **{'ON (unverified)' if provisional_ppc2 else 'OFF'}**\n\n"
                "Now open **2. Researcher decision forms**. Choose each decision, select an option and set its status. "
                "You do not need to write explanations."
            )
            options = _decision_rows(queue)
            first = options[0][1] if options else None
            return state, _queue_to_state(queue), overview, gr.update(choices=options, value=first), _summary_view(queue)
        except Exception as exc:
            raise gr.Error(f"MetaPy could not inspect this workbook: {exc}")

    def load_decision(decision_id, queue_state_value):
        queue = _queue_from_state(queue_state_value)
        row = _row_for(decision_id, queue)
        if row is None:
            return (
                "Select a decision above.", gr.update(choices=[], value=None), gr.update(value="Pending"),
                "Select an option to see its explanation.",
                gr.update(visible=False, value="not_checked"), gr.update(visible=False, value=False),
                gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None),
                gr.update(visible=False, value=[]), gr.update(visible=False, value="common_grid"), gr.update(visible=False, value=False),
                gr.update(visible=False, value=None), gr.update(visible=False, value="registered_protocol"), gr.update(visible=False, value=None),
            )
        option_choices = _option_choices(row)
        allowed_option_values = {value for _, value in option_choices}
        selected = _ui_choice_value(row.get("Your choice"), allowed_option_values)
        status = _safe(row.get("Status")) or "Pending"
        if status not in {"Pending", "Approved", "Rejected", "Deferred"}:
            status = "Pending"
        info = _detail_markdown(row)
        explanation = _choice_explanation(row, selected or "")
        details = _details(row.get("Decision details (structured)"))
        vis = _visibility(_safe(row.get("Category")), selected or "")
        return (
            info,
            gr.update(choices=option_choices, value=selected),
            gr.update(value=status),
            explanation,
            gr.update(visible=vis["reported_method_status"], value=_ui_choice_value(details.get("reported_method_status", "not_checked"), _REPORTED_METHOD_VALUES, "not_checked")),
            gr.update(visible=vis["ppc2_ack"], value=bool(details.get("acknowledge_unverified_ppc2", False))),
            gr.update(visible=vis["correlation_source"], value=_ui_choice_value(details.get("correlation_source"), _CORRELATION_SOURCE_VALUES)),
            gr.update(visible=vis["change_type"], value=_ui_choice_value(details.get("change_information_type"), _CHANGE_TYPE_VALUES)),
            gr.update(visible=vis["change_source"], value=_ui_choice_value(details.get("change_information_source"), _CHANGE_SOURCE_VALUES)),
            gr.update(visible=vis["sensitivity_values"], value=_ui_multi_choice_values(details.get("correlation_values", [0.1, 0.3, 0.5, 0.7, 0.9]), _SENSITIVITY_VALUE_VALUES)),
            gr.update(visible=vis["sensitivity_scope"], value=_ui_choice_value(details.get("sensitivity_scope", "common_grid"), _SENSITIVITY_SCOPE_VALUES, "common_grid")),
            gr.update(visible=vis["sensitivity_ack"], value=bool(details.get("acknowledge_assumed_correlations", False))),
            gr.update(visible=vis["exclusion_reason"], value=_ui_choice_value(row.get("Exclusion reason"), _EXCLUSION_REASON_VALUES)),
            gr.update(visible=vis["protocol_alignment"], value=_ui_choice_value(details.get("protocol_alignment", "registered_protocol"), _PROTOCOL_ALIGNMENT_VALUES, "registered_protocol")),
            gr.update(visible=vis["arm_similarity"], value=_ui_choice_value(details.get("arm_similarity"), _ARM_SIMILARITY_VALUES)),
        )

    def update_choice_help(decision_id, choice, queue_state_value):
        queue = _queue_from_state(queue_state_value)
        row = _row_for(decision_id, queue)
        if row is None:
            return ("Select an option to see its explanation.",) + tuple(gr.update(visible=False) for _ in range(11))
        vis = _visibility(_safe(row.get("Category")), _safe(choice))
        return (
            _choice_explanation(row, choice),
            gr.update(visible=vis["reported_method_status"]),
            gr.update(visible=vis["ppc2_ack"]),
            gr.update(visible=vis["correlation_source"]),
            gr.update(visible=vis["change_type"]),
            gr.update(visible=vis["change_source"]),
            gr.update(visible=vis["sensitivity_values"]),
            gr.update(visible=vis["sensitivity_scope"]),
            gr.update(visible=vis["sensitivity_ack"]),
            gr.update(visible=vis["exclusion_reason"]),
            gr.update(visible=vis["protocol_alignment"]),
            gr.update(visible=vis["arm_similarity"]),
        )

    def save_form(
        decision_id, choice, status, reported_method_status, ppc2_ack,
        correlation_source, change_type, change_source, sensitivity_values,
        sensitivity_scope, sensitivity_ack, exclusion_reason, protocol_alignment,
        arm_similarity, queue_state_value,
    ):
        queue = _queue_from_state(queue_state_value)
        try:
            updated, message = _save_decision(
                queue, decision_id, choice, status, reported_method_status, ppc2_ack,
                correlation_source, change_type, change_source, sensitivity_values,
                sensitivity_scope, sensitivity_ack, exclusion_reason, protocol_alignment,
                arm_similarity,
            )
        except Exception as exc:
            raise gr.Error(str(exc))
        return _queue_to_state(updated), _summary_view(updated), message

    def validate(queue_state_value):
        queue = _queue_from_state(queue_state_value)
        if queue is None or queue.empty:
            raise gr.Error("Inspect a workbook first.")
        audit, findings = validate_researcher_decisions(queue)
        summary = decision_summary(audit)
        plan = build_approved_analysis_plan(audit)
        status = (
            f"### Decision summary\n\n"
            f"- Total decisions: **{summary['total']}**\n"
            f"- Approved: **{summary['approved']}**\n"
            f"- Pending: **{summary['pending']}**\n"
            f"- Rejected: **{summary['rejected']}**\n"
            f"- Deferred: **{summary['deferred']}**\n\n"
            f"- Primary method approved: **{plan.get('primary_estimand_approved', False)}**\n"
            f"- Variance route approved: **{plan.get('variance_plan_approved', False)}**\n"
            f"- Shared-control handling selected: **{plan.get('multiarm_handling_selected', True)}**\n\n"
            "**Pooling is not enabled in this version.** These forms record and validate decisions; the next integration step connects approved choices to the calculation modules."
        )
        return _queue_to_state(audit), findings, status, _summary_view(audit)

    def export(queue_state_value):
        queue = _queue_from_state(queue_state_value)
        if queue is None or queue.empty:
            raise gr.Error("There is no decision queue to export.")
        audit, findings = validate_researcher_decisions(queue)
        if not findings.empty and any(str(x) == "Error" for x in findings.get("Severity", [])):
            raise gr.Error("Fix the form validation errors before exporting the decision audit.")
        out = Path(tempfile.gettempdir()) / "MetaPy_Researcher_Decision_Audit.xlsx"
        return export_decision_audit(audit, str(out))

    def _prep_preview(bundle):
        if not bundle or not isinstance(bundle.get("prepared_records"), pd.DataFrame):
            return pd.DataFrame()
        columns = [
            "Preparation record ID", "Study ID", "Intervention arm", "Outcome label",
            "Original outcome / instrument", "Outcome data type", "Reporting format",
            "Study design", "Analysis structure", "Follow-up / duration",
            "N intervention (source)", "N control (source)",
            "N control for selected primary variance route", "Selected primary estimand",
            "Primary effect route status", "Reported g method status", "Preparation status",
        ]
        frame = bundle["prepared_records"]
        return frame[[c for c in columns if c in frame.columns]].copy()

    def _prep_summary_markdown(bundle):
        if not bundle:
            return "Build the prepared dataset after saving your researcher decisions."
        summary = bundle.get("summary", {})
        return (
            "## Data preparation overview\n\n"
            f"- Canonical records: **{summary.get('canonical_records', 0)}**\n"
            f"- Outcome groups: **{summary.get('outcome_groups', 0)}**\n"
            f"- Selected primary method: **{summary.get('primary_estimand_approved') or 'Not approved'}**\n"
            f"- Records with a compatible route candidate: **{summary.get('records_with_primary_route_candidate', 0)}**\n"
            f"- Reported g values provisionally assumed PPC2 (unverified): **{summary.get('reported_g_assumed_ppc2_unverified', 0)}**\n"
            f"- Shared-control records with a split prepared for sample-size-based variance routes: **{summary.get('shared_control_records_prepared_for_variance_weighting', 0)}**\n"
            f"- Preparation review items: **{summary.get('preparation_review_items', 0)}** active; **{summary.get('preparation_reviews_approved', 0)}** approved; **{summary.get('preparation_reviews_pending', 0)}** pending\n"
            f"- Cascading follow-up tasks: **{summary.get('dynamic_dependent_review_items', 0)}**\n"
            f"- Blocking findings: **{summary.get('blocking_findings', 0)}**\n\n"
            f"**Status: {summary.get('data_preparation_status', 'Not started')}**\n\n"
            "The source workbook is unchanged. This stage creates a separate prepared dataset; it does not calculate final effect sizes, sampling variances or pooled results."
        )

    def _prep_review_choices(bundle):
        if not bundle or bundle.get("review_items") is None or bundle["review_items"].empty:
            return []
        out = []
        for _, row in bundle["review_items"].iterrows():
            if _safe(row.get("Status")) == "Not applicable" or _safe(row.get("Dependency state")) == "Inactive":
                continue
            label = f"{_safe(row.get('Category'))} — {_safe(row.get('Subject'))} [{_safe(row.get('Status'))}]"
            if _safe(row.get("Dependency state")).startswith("Re-review required"):
                label = "REVIEW AGAIN — " + label
            out.append((label, _safe(row.get("Review ID"))))
        return out

    def _prep_item_info(review_row):
        if review_row is None:
            return "Select a preparation review item."
        return (
            f"### {review_row.get('Category', 'Preparation review')}\n\n"
            f"**Item:** {review_row.get('Subject', '')}\n\n"
            f"**What MetaPy found**\n\n{review_row.get('What MetaPy found', '')}\n\n"
            f"**Why it matters**\n\n{review_row.get('Why it matters', '')}\n\n"
            f"**MetaPy's suggestion**\n\n{review_row.get('MetaPy suggestion', '')}\n\n"
            f"**What this decision depends on**\n\n{review_row.get('Depends on', 'Source data and approved decisions')}\n\n"
            f"**What this decision can change**\n\n{review_row.get('Will affect', 'The affected preparation records')}\n\n"
            f"**Dependency status:** {review_row.get('Dependency state', 'Active')}\n\n"
            f"{('**Why MetaPy reopened this item:** ' + str(review_row.get('Re-review reason'))) if _safe(review_row.get('Re-review reason')) else ''}"
        )

    def build_prepared_data(session_value, queue_value, old_prep_state):
        if not session_value:
            raise gr.Error("Inspect your workbook first in tab 1.")
        queue = _queue_from_state(queue_value)
        if queue.empty:
            raise gr.Error("Save your researcher decisions before preparing the dataset.")
        try:
            previous = bundle_from_state(old_prep_state) if old_prep_state else None
            bundle = build_preparation_bundle(session_value, queue, previous_state=previous)
            state_value = bundle_to_state(bundle)
            review_options = _prep_review_choices(bundle)
            first_id = review_options[0][1] if review_options else None
            return (
                state_value, _prep_summary_markdown(bundle), _prep_preview(bundle),
                bundle.get("classification", pd.DataFrame()), bundle.get("findings", pd.DataFrame()),
                bundle.get("transformations", pd.DataFrame()), bundle.get("review_items", pd.DataFrame()),
                gr.update(choices=review_options, value=first_id),
            )
        except Exception as exc:
            raise gr.Error(f"MetaPy could not prepare the dataset: {exc}")

    def refresh_preparation_after_global_decision(queue_value, session_value, old_prep_state):
        """Recompute downstream preparation reviews after any upstream decision changes."""
        if not old_prep_state or not session_value:
            return (
                old_prep_state or "", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
            )
        try:
            queue = _queue_from_state(queue_value)
            previous = bundle_from_state(old_prep_state)
            refreshed = build_preparation_bundle(session_value, queue, previous_state=previous)
            state_value = bundle_to_state(refreshed)
            options = _prep_review_choices(refreshed)
            current_id = options[0][1] if options else None
            return (
                state_value, _prep_summary_markdown(refreshed), _prep_preview(refreshed),
                refreshed.get("classification", pd.DataFrame()), refreshed.get("findings", pd.DataFrame()),
                refreshed.get("transformations", pd.DataFrame()), refreshed.get("review_items", pd.DataFrame()),
                gr.update(choices=options, value=current_id),
            )
        except Exception as exc:
            # Do not undo the saved upstream decision. Surface a visible refresh error.
            return (
                old_prep_state,
                f"Preparation refresh failed after the decision was saved: {exc}. Rebuild the prepared dataset to reconcile dependent reviews.",
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            )

    def load_preparation_item(review_id, prep_state_value):
        bundle = bundle_from_state(prep_state_value)
        if not bundle or bundle.get("review_items") is None or bundle["review_items"].empty:
            return "Build the prepared dataset first.", gr.update(choices=[], value=None), gr.update(value="Pending"), "Select an option to see its explanation."
        frame = bundle["review_items"]
        matches = frame.index[frame["Review ID"].astype(str).eq(_safe(review_id))].tolist()
        if not matches:
            return "Select a preparation review item.", gr.update(choices=[], value=None), gr.update(value="Pending"), "Select an option to see its explanation."
        row = frame.loc[matches[0]]
        try:
            options = json.loads(_safe(row.get("Available options")) or "[]")
        except Exception:
            options = []
        choices = [(str(o.get("label", o.get("value", ""))), str(o.get("value", ""))) for o in options if isinstance(o, dict)]
        selected = _safe(row.get("Your choice")) or None
        status = _safe(row.get("Status")) or "Pending"
        selected_option = next((o for o in options if str(o.get("value")) == str(selected)), None)
        explanation = "Select an option to see its explanation."
        if selected_option:
            explanation = f"**Selected:** {selected_option.get('label', selected)}\n\n{selected_option.get('explanation', '')}"
        return _prep_item_info(row), gr.update(choices=choices, value=selected), gr.update(value=status), explanation

    def explain_preparation_choice(review_id, choice, prep_state_value):
        bundle = bundle_from_state(prep_state_value)
        if not bundle or bundle.get("review_items") is None:
            return "Select an option to see its explanation."
        frame = bundle["review_items"]
        matches = frame.index[frame["Review ID"].astype(str).eq(_safe(review_id))].tolist()
        if not matches:
            return "Select an option to see its explanation."
        try:
            options = json.loads(_safe(frame.loc[matches[0]].get("Available options")) or "[]")
        except Exception:
            options = []
        option = next((o for o in options if isinstance(o, dict) and str(o.get("value")) == str(choice)), None)
        return f"**{option.get('label', choice)}**\n\n{option.get('explanation', '')}" if option else "Select one of the listed options."

    def save_preparation_form(review_id, choice, status, prep_state_value, queue_state_value, session_value):
        bundle = bundle_from_state(prep_state_value)
        if not bundle:
            raise gr.Error("Build the prepared dataset before saving a preparation decision.")
        try:
            updated = save_preparation_review(bundle, review_id, choice, status)
        except Exception as exc:
            raise gr.Error(str(exc))

        # Some preparation findings may request reconsideration of a previously
        # approved upstream choice. Reset that upstream choice to Pending rather
        # than silently approving a new method or dependence strategy.
        queue = _queue_from_state(queue_state_value)
        changed_upstream = ""
        if status == "Approved" and _safe(review_id) == "PREP-METHOD-COMPATIBILITY" and _safe(choice) == "request_morris_reconsideration":
            rows = queue.index[queue["Category"].astype(str).eq("Primary analysis method")].tolist() if not queue.empty else []
            if rows:
                idx = rows[0]
                queue.at[idx, "Your choice"] = "morris_ppc2"
                queue.at[idx, "Status"] = "Pending"
                if "Decision recorded (UTC)" in queue.columns:
                    queue.at[idx, "Decision recorded (UTC)"] = ""
                try:
                    details = json.loads(_safe(queue.at[idx, "Decision details (structured)"]) or "{}")
                except Exception:
                    details = {}
                details["protocol_alignment"] = "not_sure"
                queue.at[idx, "Decision details (structured)"] = json.dumps(details, ensure_ascii=False, sort_keys=True)
                changed_upstream = " The primary-method choice is now Morris PPC2 but has been reset to Pending; return to tab 2 and approve it after checking your protocol."
        if status == "Approved" and _safe(review_id) == "PREP-MULTIARM-ANCOVA-COMPATIBILITY" and _safe(choice) in {"request_robust_multiarm", "request_combine_arms"}:
            rows = queue.index[queue["Category"].astype(str).eq("Multi-arm study dependence")].tolist() if not queue.empty else []
            if rows:
                idx = rows[0]
                new_choice = "multivariate_model" if choice == "request_robust_multiarm" else "combine_arms"
                queue.at[idx, "Your choice"] = new_choice
                queue.at[idx, "Status"] = "Pending"
                if "Decision recorded (UTC)" in queue.columns:
                    queue.at[idx, "Decision recorded (UTC)"] = ""
                try:
                    details = json.loads(_safe(queue.at[idx, "Decision details (structured)"]) or "{}")
                except Exception:
                    details = {}
                if new_choice == "combine_arms":
                    details.pop("arm_similarity", None)
                queue.at[idx, "Decision details (structured)"] = json.dumps(details, ensure_ascii=False, sort_keys=True)
                changed_upstream = " The multi-arm decision has been changed to a new Pending choice; return to tab 2 to review and approve it."
        if status == "Approved" and _safe(review_id) == "PREP-MULTIARM-ANCOVA-COMPATIBILITY" and _safe(choice) == "revisit_primary_method":
            rows = queue.index[queue["Category"].astype(str).eq("Primary analysis method")].tolist() if not queue.empty else []
            if rows:
                idx = rows[0]
                queue.at[idx, "Your choice"] = ""
                queue.at[idx, "Status"] = "Pending"
                if "Decision recorded (UTC)" in queue.columns:
                    queue.at[idx, "Decision recorded (UTC)"] = ""
                try:
                    details = json.loads(_safe(queue.at[idx, "Decision details (structured)"]) or "{}")
                except Exception:
                    details = {}
                details["protocol_alignment"] = "not_sure"
                queue.at[idx, "Decision details (structured)"] = json.dumps(details, ensure_ascii=False, sort_keys=True)
                changed_upstream = " The primary-method decision has been reset to Pending; return to tab 2 and select a compatible method."
        if isinstance(session_value, dict):
            # Rebuild after every saved choice. This refreshes dependent tasks,
            # applies approved metadata to the prepared copy, and recomputes findings.
            updated = build_preparation_bundle(session_value, queue, previous_state=updated)
        updated["transformations"] = updated.get("transformations", pd.DataFrame())
        return (
            bundle_to_state(updated), _queue_to_state(queue), _prep_summary_markdown(updated),
            _prep_preview(updated), updated.get("classification", pd.DataFrame()),
            updated.get("findings", pd.DataFrame()), updated.get("transformations", pd.DataFrame()),
            updated.get("review_items", pd.DataFrame()),
            f"Saved preparation choice: **{choice}** — **{status}**.{changed_upstream}",
            gr.update(choices=_prep_review_choices(updated), value=review_id),
            _summary_view(queue),
        )

    def load_correction_sheets(session_value):
        if not isinstance(session_value, dict) or not session_value.get("sheets"):
            return gr.update(choices=[], value=None)
        opts = correction_source_options(session_value)
        choices = [(name, name) for name in opts.get("sheets", [])]
        first = choices[0][1] if choices else None
        return gr.update(choices=choices, value=first)

    def load_correction_fields(sheet_name, session_value):
        if not isinstance(session_value, dict) or not sheet_name:
            return gr.update(choices=[], value=None), gr.update(choices=[], value=None), "Select a workbook and worksheet first."
        opts = correction_row_column_options(session_value, sheet_name)
        row_choices = opts.get("rows", [])
        column_choices = opts.get("columns", [])
        return (
            gr.update(choices=row_choices, value=row_choices[0][1] if row_choices else None),
            gr.update(choices=column_choices, value=column_choices[0][1] if column_choices else None),
            f"Worksheet **{sheet_name}** has {len(row_choices)} data rows and {len(column_choices)} columns. Row numbers refer to the Excel file (header is row 1).",
        )

    def apply_cell_correction(
        session_value, queue_value, prep_state_value, sheet_name, excel_row, column_name,
        new_value, reason, evidence_source, source_checked,
    ):
        bundle = bundle_from_state(prep_state_value)
        if not bundle:
            raise gr.Error("Build the prepared dataset first and approve the cell-correction review item.")
        queue = _queue_from_state(queue_value)
        try:
            new_session, new_queue, entry = apply_approved_cell_correction(
                session_value, queue, bundle, sheet_name, excel_row, column_name,
                new_value, reason, evidence_source, bool(source_checked),
            )
        except Exception as exc:
            raise gr.Error(f"MetaPy did not apply the correction: {exc}")
        summary = (
            "## Correction applied to the MetaPy working copy\n\n"
            f"- Worksheet: **{entry['Worksheet']}**\n"
            f"- Excel row: **{entry['Excel row']}**\n"
            f"- Column: **{entry['Column']}**\n"
            f"- Previous value: **{entry['Previous value']}**\n"
            f"- New value: **{entry['Corrected value']}**\n\n"
            "MetaPy re-read the workbook and repeated its internal checks. The original upload was not changed. "
            "All upstream researcher decisions have been reset to Pending because the data changed. Re-review them, then rebuild the prepared dataset."
        )
        sheet_choices = [(name, name) for name in correction_source_options(new_session).get("sheets", [])]
        row_col = correction_row_column_options(new_session, sheet_name)
        row_choices, col_choices = row_col.get("rows", []), row_col.get("columns", [])
        return (
            new_session, _queue_to_state(new_queue), "", summary,
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            _summary_view(new_queue),
            gr.update(choices=sheet_choices, value=sheet_name),
            gr.update(choices=row_choices, value=excel_row if any(v == excel_row for _, v in row_choices) else (row_choices[0][1] if row_choices else None)),
            gr.update(choices=col_choices, value=column_name if any(v == column_name for _, v in col_choices) else (col_choices[0][1] if col_choices else None)),
            "Correction saved. Prior approvals were reset and must be reviewed again.",
        )

    def validate_prepared_data(prep_state_value):
        bundle = bundle_from_state(prep_state_value)
        if not bundle:
            raise gr.Error("Build the prepared dataset first.")
        try:
            validated, findings = validate_preparation_bundle(bundle)
        except Exception as exc:
            raise gr.Error(f"Preparation validation failed: {exc}")
        status = validated.get("summary", {}).get("data_preparation_status", "Needs review")
        report = (
            f"## Data preparation validation\n\n"
            f"**Status: {status}**\n\n"
            f"- Review items approved: **{validated.get('summary', {}).get('preparation_reviews_approved', 0)}**\n"
            f"- Review items pending/deferred: **{validated.get('summary', {}).get('preparation_reviews_pending', 0)}**\n"
            f"- Blocking findings: **{validated.get('summary', {}).get('blocking_findings', 0)}**\n"
            f"- Critical/error findings: **{validated.get('summary', {}).get('critical_or_error_findings', 0)}**\n\n"
            "**No effect sizes, sampling variances or pooled estimates were calculated in this step.**"
        )
        return bundle_to_state(validated), report, findings, _prep_summary_markdown(validated), _prep_preview(validated)

    def export_prepared_data(prep_state_value):
        bundle = bundle_from_state(prep_state_value)
        if not bundle:
            raise gr.Error("Build the prepared dataset first.")
        try:
            out = Path(tempfile.gettempdir()) / "MetaPy_Prepared_Data_v3_2.xlsx"
            export_preparation_workbook(bundle, out)
            return str(out)
        except Exception as exc:
            raise gr.Error(f"Could not export the prepared dataset: {exc}")

    with gr.Blocks(title=f"MetaPy {DECISION_VERSION} + Data Preparation {PREPARATION_VERSION}") as app:
        gr.Markdown(
            f"# MetaPy — Researcher Decisions + Data Preparation\n\n"
            f"Decision engine {DECISION_VERSION} · Data-preparation engine {PREPARATION_VERSION}\n\n"
            "**Choose from the forms. You do not need to write a response or explain your choice in a text box.**\n\n"
            "MetaPy explains each finding in simple words, shows the available choices and records the selection you approve."
        )
        session_state = gr.State(None)
        queue_state = gr.State("[]")
        preparation_state = gr.State("")

        with gr.Tab("1. Inspect workbook"):
            file = gr.File(label="Upload Excel workbook (.xlsx)", file_types=[".xlsx"], type="filepath")
            design = gr.Dropdown(
                label="What type of study design is mainly included?",
                choices=[("Parallel groups (separate treatment and control participants)", "parallel_independent"),
                         ("Paired, crossover, repeated or not yet sure", "dependent_or_unknown")],
                value="parallel_independent", multiselect=False, type="value",
            )
            provisional = gr.Checkbox(
                label="For this provisional analysis, treat reported Hedges' g values as Morris PPC2 (not verified from the papers)",
                value=False,
                info="This is a working assumption only. MetaPy will keep it visibly marked as unverified.",
            )
            run = gr.Button("Inspect workbook and prepare decision forms", variant="primary")
            overview = gr.Markdown()

        with gr.Tab("2. Researcher decision forms"):
            gr.Markdown("### Step A — Choose which decision to review")
            decision_selector = gr.Dropdown(label="Decision to review", choices=[], value=None, interactive=True, multiselect=False, type="value")
            decision_info = gr.Markdown("Inspect a workbook first, then select a decision.")
            gr.Markdown("### Step B — Select your decision")
            choice_dropdown = gr.Dropdown(label="Select one option", choices=[], value=None, interactive=True, multiselect=False, type="value")
            choice_explanation = gr.Markdown("Select an option to see its explanation.")
            status_dropdown = gr.Dropdown(
                label="Decision status",
                choices=["Pending", "Approved", "Rejected", "Deferred"],
                value="Pending",
                interactive=True,
                multiselect=False, type="value",
                info="Approved means you accept this selected pathway. Pending means more review is needed.",
            )

            with gr.Group():
                gr.Markdown("### Step C — Complete the relevant fields (only the fields for your selected option will appear)")
                reported_method_status = gr.Dropdown(
                    label="What does the original paper say about the reported Hedges' g method?",
                    choices=[("Not checked yet", "not_checked"),
                             ("The paper confirms Morris PPC2", "confirmed_ppc2"),
                             ("The paper confirms change-score SMD", "confirmed_change_smd"),
                             ("The paper confirms post-treatment SMD", "confirmed_post_smd"),
                             ("The paper reports an ANCOVA-adjusted effect", "confirmed_ancova"),
                             ("The method is not stated or cannot be determined", "unclear")],
                    value="not_checked", visible=False, multiselect=False, type="value",
                )
                ppc2_ack = gr.Checkbox(
                    label="I understand that treating these reported values as PPC2 is an unverified working assumption.",
                    value=False, visible=False,
                )
                correlation_source = gr.Radio(
                    label="How will the actual pre–post correlations be obtained? (study-specific values are needed)",
                    choices=[("Reported separately for each study", "reported_per_study"),
                             ("Calculated separately for each study from participant-level data", "calculated_per_study"),
                             ("Provided by study authors separately for each study", "author_provided_per_study")],
                    value=None, visible=False,
                )
                change_type = gr.Dropdown(
                    label="Which change-score information is available?",
                    choices=[("Standard deviation of change (SD change)", "sd_change"),
                             ("Standard error of change (SE change)", "se_change"),
                             ("95% confidence interval for the change", "change_ci"),
                             ("Paired t statistic and degrees of freedom", "paired_t"),
                             ("Exact paired p-value", "paired_p")],
                    value=None, visible=False, multiselect=False, type="value",
                )
                change_source = gr.Dropdown(
                    label="Where will this change-score information come from?",
                    choices=[("Original study report", "original_report"),
                             ("Study authors", "author_provided"),
                             ("A documented calculation from reported study data", "documented_calculation")],
                    value=None, visible=False, multiselect=False, type="value",
                )
                sensitivity_values = gr.CheckboxGroup(
                    label="Select the assumed pre–post correlations to test",
                    choices=[("0.10", 0.1), ("0.30", 0.3), ("0.50", 0.5), ("0.70", 0.7), ("0.90", 0.9)],
                    value=[0.1, 0.3, 0.5, 0.7, 0.9], visible=False,
                    info="These are assumptions, not observed study data. Select at least two values.",
                )
                sensitivity_scope = gr.Radio(
                    label="How should this sensitivity grid be applied later?",
                    choices=[("Use the same selected grid for all studies missing correlations", "common_grid"),
                             ("Set a separate grid for each study in the next study-level step", "study_specific_grid")],
                    value="common_grid", visible=False,
                )
                sensitivity_ack = gr.Checkbox(
                    label="I understand these correlation values are assumed and will be labelled as such.",
                    value=False, visible=False,
                )
                exclusion_reason = gr.Dropdown(
                    label="Why is this record being excluded?",
                    choices=EXCLUSION_REASON_OPTIONS, value=None, visible=False, multiselect=False, type="value",
                )
                protocol_alignment = gr.Dropdown(
                    label="How does the selected method relate to your protocol?",
                    choices=[("It matches the registered protocol", "registered_protocol"),
                             ("The protocol was amended and the change is documented", "documented_amendment"),
                             ("I am not sure and need to check", "not_sure")],
                    value="registered_protocol", visible=False, multiselect=False, type="value",
                )
                arm_similarity = gr.Dropdown(
                    label="Are the intervention arms sufficiently similar to combine?",
                    choices=[("Yes — sufficiently similar", "similar"),
                             ("No — meaningfully different", "not_similar"),
                             ("Not sure — needs review", "uncertain")],
                    value=None, visible=False, multiselect=False, type="value",
                )

            save = gr.Button("Save this decision", variant="primary")
            save_message = gr.Markdown()
            decision_summary_table = gr.Dataframe(label="Decision overview", interactive=False, wrap=True)

        with gr.Tab("3. Validate and export"):
            validate_btn = gr.Button("Validate all saved decisions", variant="primary")
            validation = gr.Dataframe(label="Validation findings", interactive=False, wrap=True)
            decision_status = gr.Markdown()
            export_btn = gr.Button("Export decision audit (.xlsx)", variant="primary")
            export_file = gr.File(label="Download decision audit", interactive=False)

        with gr.Tab("4. Data preparation"):
            gr.Markdown(
                "### Build the separate, analysis-oriented dataset\n\n"
                "MetaPy will apply only approved decisions, preserve the original workbook, classify records in four dimensions, identify remaining problems and ask for structured review. It will not calculate final effect sizes, sampling variances or pooled results yet."
            )
            prepare_btn = gr.Button("Build prepared dataset from saved decisions", variant="primary")
            preparation_summary = gr.Markdown("Inspect the workbook and save researcher decisions first.")
            prepared_preview = gr.Dataframe(label="Prepared dataset preview", interactive=False, wrap=True)
            with gr.Row():
                classification_table = gr.Dataframe(label="Four-dimensional classification", interactive=False, wrap=True)
                transformation_table = gr.Dataframe(label="Approved / automatic transformation log", interactive=False, wrap=True)
            preparation_findings = gr.Dataframe(label="Preparation checks and findings", interactive=False, wrap=True)
            gr.Markdown("### Preparation review forms\nChoose a finding, select one of the offered actions, and save it. No written response is required.")
            preparation_item_selector = gr.Dropdown(label="Preparation item to review", choices=[], value=None, interactive=True, multiselect=False, type="value")
            preparation_item_info = gr.Markdown("Build the prepared dataset to see review items.")
            preparation_choice = gr.Dropdown(label="Select one option", choices=[], value=None, interactive=True, multiselect=False, type="value")
            preparation_choice_explanation = gr.Markdown("Select an option to see what it means.")
            preparation_status = gr.Dropdown(label="Review status", choices=["Pending", "Approved", "Rejected", "Deferred"], value="Pending", interactive=True, multiselect=False, type="value")
            save_preparation_btn = gr.Button("Save preparation decision", variant="primary")
            preparation_save_message = gr.Markdown()
            preparation_review_table = gr.Dataframe(label="Preparation review overview", interactive=False, wrap=True)

            gr.Markdown("### Researcher-approved cell correction (only when specifically approved above)")
            gr.Markdown("Use this form only after approving **Specific cell-level corrections are needed**. MetaPy changes one cell in its separate working copy, reruns workbook checks, and resets earlier decisions to Pending. The original uploaded workbook is never overwritten.")
            refresh_correction_sources_btn = gr.Button("Load worksheet correction choices")
            correction_sheet = gr.Dropdown(label="Worksheet", choices=[], value=None, interactive=True, multiselect=False, type="value")
            correction_row = gr.Dropdown(label="Excel data row", choices=[], value=None, interactive=True, multiselect=False, type="value")
            correction_column = gr.Dropdown(label="Column", choices=[], value=None, interactive=True, multiselect=False, type="value")
            correction_cell_hint = gr.Markdown("Load worksheet choices to select a cell.")
            correction_new_value = gr.Textbox(label="Verified replacement cell value", placeholder="Enter the corrected cell value (not an explanation)")
            correction_reason = gr.Dropdown(
                label="Why is this cell being corrected?",
                choices=[("Transcription / typing error", "transcription_error"),
                         ("Clarification from the original report", "source_report_clarification"),
                         ("Sample-size correction", "sample_size_correction"),
                         ("Outcome or treatment-arm label correction", "outcome_or_arm_label"),
                         ("Other source-verified correction", "other_verified_correction")],
                value=None, interactive=True, multiselect=False, type="value",
            )
            correction_evidence = gr.Dropdown(
                label="Which source did you check?",
                choices=[("Original published report", "original_publication"),
                         ("Supplementary material", "supplementary_material"),
                         ("Author correspondence", "author_correspondence"),
                         ("Researcher-verified extraction record", "researcher_verified_extraction")],
                value=None, interactive=True, multiselect=False, type="value",
            )
            correction_source_checked = gr.Checkbox(label="I checked this replacement value against the selected source.", value=False)
            apply_correction_btn = gr.Button("Apply correction and rerun workbook checks", variant="stop")
            correction_message = gr.Markdown()

            validate_preparation_btn = gr.Button("Validate prepared dataset", variant="primary")
            preparation_validation_report = gr.Markdown()
            preparation_validation_table = gr.Dataframe(label="Validation findings", interactive=False, wrap=True)
            export_preparation_btn = gr.Button("Export prepared dataset and audit (.xlsx)", variant="primary")
            preparation_export_file = gr.File(label="Download prepared dataset workbook", interactive=False)

        run_event = run.click(
            inspect_workbook,
            [file, design, provisional],
            [session_state, queue_state, overview, decision_selector, decision_summary_table],
        )
        run_event.then(
            load_decision,
            [decision_selector, queue_state],
            [decision_info, choice_dropdown, status_dropdown, choice_explanation,
             reported_method_status, ppc2_ack, correlation_source, change_type, change_source,
             sensitivity_values, sensitivity_scope, sensitivity_ack, exclusion_reason,
             protocol_alignment, arm_similarity],
        )
        decision_selector.change(
            load_decision,
            [decision_selector, queue_state],
            [decision_info, choice_dropdown, status_dropdown, choice_explanation,
             reported_method_status, ppc2_ack, correlation_source, change_type, change_source,
             sensitivity_values, sensitivity_scope, sensitivity_ack, exclusion_reason,
             protocol_alignment, arm_similarity],
        )
        choice_dropdown.change(
            update_choice_help,
            [decision_selector, choice_dropdown, queue_state],
            [choice_explanation, reported_method_status, ppc2_ack, correlation_source,
             change_type, change_source, sensitivity_values, sensitivity_scope, sensitivity_ack,
             exclusion_reason, protocol_alignment, arm_similarity],
        )
        save_event = save.click(
            save_form,
            [decision_selector, choice_dropdown, status_dropdown, reported_method_status, ppc2_ack,
             correlation_source, change_type, change_source, sensitivity_values, sensitivity_scope,
             sensitivity_ack, exclusion_reason, protocol_alignment, arm_similarity, queue_state],
            [queue_state, decision_summary_table, save_message],
        )
        save_event.then(
            load_decision,
            [decision_selector, queue_state],
            [decision_info, choice_dropdown, status_dropdown, choice_explanation,
             reported_method_status, ppc2_ack, correlation_source, change_type, change_source,
             sensitivity_values, sensitivity_scope, sensitivity_ack, exclusion_reason,
             protocol_alignment, arm_similarity],
        ).then(
            refresh_preparation_after_global_decision,
            [queue_state, session_state, preparation_state],
            [preparation_state, preparation_summary, prepared_preview, classification_table,
             preparation_findings, transformation_table, preparation_review_table, preparation_item_selector],
        ).then(
            load_preparation_item,
            [preparation_item_selector, preparation_state],
            [preparation_item_info, preparation_choice, preparation_status, preparation_choice_explanation],
        )
        validate_btn.click(validate, [queue_state], [queue_state, validation, decision_status, decision_summary_table])
        export_btn.click(export, [queue_state], [export_file])

        prepare_event = prepare_btn.click(
            build_prepared_data,
            [session_state, queue_state, preparation_state],
            [preparation_state, preparation_summary, prepared_preview, classification_table,
             preparation_findings, transformation_table, preparation_review_table, preparation_item_selector],
        )
        prepare_event.then(
            load_preparation_item,
            [preparation_item_selector, preparation_state],
            [preparation_item_info, preparation_choice, preparation_status, preparation_choice_explanation],
        )
        preparation_item_selector.change(
            load_preparation_item,
            [preparation_item_selector, preparation_state],
            [preparation_item_info, preparation_choice, preparation_status, preparation_choice_explanation],
        )
        preparation_choice.change(
            explain_preparation_choice,
            [preparation_item_selector, preparation_choice, preparation_state],
            [preparation_choice_explanation],
        )
        save_preparation_btn.click(
            save_preparation_form,
            [preparation_item_selector, preparation_choice, preparation_status, preparation_state, queue_state, session_state],
            [preparation_state, queue_state, preparation_summary, prepared_preview, classification_table,
             preparation_findings, transformation_table, preparation_review_table,
             preparation_save_message, preparation_item_selector, decision_summary_table],
        ).then(
            load_preparation_item,
            [preparation_item_selector, preparation_state],
            [preparation_item_info, preparation_choice, preparation_status, preparation_choice_explanation],
        )
        refresh_correction_sources_btn.click(
            load_correction_sheets, [session_state], [correction_sheet],
        ).then(
            load_correction_fields, [correction_sheet, session_state],
            [correction_row, correction_column, correction_cell_hint],
        )
        correction_sheet.change(
            load_correction_fields, [correction_sheet, session_state],
            [correction_row, correction_column, correction_cell_hint],
        )
        apply_correction_btn.click(
            apply_cell_correction,
            [session_state, queue_state, preparation_state, correction_sheet, correction_row,
             correction_column, correction_new_value, correction_reason, correction_evidence,
             correction_source_checked],
            [session_state, queue_state, preparation_state, preparation_summary, prepared_preview,
             classification_table, preparation_findings, transformation_table, preparation_review_table,
             decision_summary_table, correction_sheet, correction_row, correction_column, correction_message],
        )

        validate_preparation_btn.click(
            validate_prepared_data,
            [preparation_state],
            [preparation_state, preparation_validation_report, preparation_validation_table,
             preparation_summary, prepared_preview],
        )
        export_preparation_btn.click(export_prepared_data, [preparation_state], [preparation_export_file])

    return app
