"""Researcher-controlled decision queue for MetaPy.

MetaPy proposes options in plain language; the researcher records the decision.
This module never executes a statistical assumption merely because it was suggested.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import re
from typing import Any, Iterable

import pandas as pd

VERSION = "3.1.3"

EXCLUSION_REASON_OPTIONS = [
    ("Wrong population", "wrong_population"),
    ("Wrong intervention or comparator", "wrong_intervention_comparator"),
    ("Wrong outcome", "wrong_outcome"),
    ("Wrong study design", "wrong_study_design"),
    ("Duplicate or overlapping report", "duplicate_report"),
    ("No usable outcome data", "no_usable_data"),
    ("Outside the prespecified time point", "wrong_timepoint"),
    ("Other protocol-based reason", "other_protocol_reason"),
]



@dataclass(frozen=True)
class Option:
    value: str
    label: str
    explanation: str
    requires_value: bool = False


@dataclass(frozen=True)
class Decision:
    decision_id: str
    category: str
    priority: str
    subject: str
    plain_language_finding: str
    why_it_matters: str
    recommendation: str
    options: tuple[Option, ...]
    default_status: str = "Pending"
    source: str = "MetaPy internal workbook analysis"


def _safe(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _records_from_review(review: dict[str, Any]) -> pd.DataFrame:
    """Return canonical records regardless of whether review is a dict of tables."""
    for path in (("harmonization", "records"), ("harmonized_records",)):
        node: Any = review
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, pd.DataFrame):
            return node.copy()
    return pd.DataFrame()


def _decision_id(category: str, subject: str, suffix: str = "") -> str:
    raw = "|".join((category, subject, suffix))
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()
    return f"D-{slug[:72]}"


def _option(value: str, label: str, explanation: str, requires_value: bool = False) -> Option:
    return Option(value, label, explanation, requires_value)


def _decision(category: str, priority: str, subject: str, finding: str,
              why: str, recommendation: str, options: Iterable[Option], suffix: str = "") -> Decision:
    return Decision(
        decision_id=_decision_id(category, subject, suffix), category=category,
        priority=priority, subject=subject, plain_language_finding=finding,
        why_it_matters=why, recommendation=recommendation,
        options=tuple(options),
    )


def build_decision_queue(review: dict[str, Any], *, assume_reported_g_is_ppc2: bool = False) -> pd.DataFrame:
    """Create researcher decisions from v3 review outputs.

    `assume_reported_g_is_ppc2` records a provisional working assumption only;
    it is never presented as external verification.
    """
    decisions: list[Decision] = []
    records = _records_from_review(review)

    # Reported-only records: do not mistake a reported value duplicated beside
    # raw measurements for a summary-only study. Provenance identifies whether
    # the workbook contains the raw pre/post inputs for this canonical record.
    reported_only = []
    if not records.empty:
        for _, row in records.iterrows():
            reported_g = _safe(row.get("reported_hedges_g"))
            if not reported_g:
                continue
            provenance = _safe(row.get("source_value_provenance")).lower()
            has_raw_prepost = "pre_m1:" in provenance and "post_m1:" in provenance
            if not has_raw_prepost:
                study = _safe(row.get("study_id")) or "Study (ID not detected)"
                outcome = _safe(row.get("outcome")) or "Outcome not detected"
                arm = _safe(row.get("arm"))
                subject = " / ".join(x for x in (study, arm, outcome) if x)
                reported_only.append(subject)
    if reported_only:
        assumption_text = (
            "Your provisional instruction is to treat the reported-only Hedges' g values as PPC2. "
            "MetaPy will record this as an unverified researcher assumption."
            if assume_reported_g_is_ppc2 else
            "Some records contain a reported Hedges' g but do not contain the raw pre/post means in the harmonised source record. "
            "The calculation method is not established by the workbook alone."
        )
        decisions.append(_decision(
            "Reported effect-size method", "High", f"{len(reported_only)} reported-only record(s)",
            assumption_text,
            "A reported Hedges' g may describe a post-treatment difference, a change-score difference, or another analysis. The label alone does not tell us which one.",
            "If you want to provisionally treat these reported-only values as PPC2, record that as an unverified assumption. Otherwise keep them separate until the original method is checked.",
            [
                _option("assume_ppc2_unverified", "Treat as Morris PPC2 for this working analysis (unverified)", "Keeps the values available but clearly marks the method as an assumption."),
                _option("confirm_from_source", "I will check the original study reports", "Keeps these values pending until their calculation method is established."),
                _option("keep_unclassified", "Keep reported values separate for now", "Preserves the values but does not include them in the PPC2 pool."),
                _option("exclude_reported", "Exclude reported-only values from this analysis", "Excludes these records from the selected analysis, while retaining them in the audit."),
            ], suffix="reported-g",
        ))

    # Variance plan: in auto-screening mode the final estimand has not yet been
    # selected. Explain the likely PPC2 variance requirement without claiming
    # that a final variance calculation has already been attempted.
    has_raw_prepost = False
    if not records.empty and "source_value_provenance" in records.columns:
        prov = records["source_value_provenance"].fillna("").astype(str).str.lower()
        has_raw_prepost = bool((prov.str.contains("pre_m1:") & prov.str.contains("post_m1:")).any())
    if has_raw_prepost:
        decisions.append(_decision(
            "Sampling variance plan", "High", "Pre–post studies before pooling",
            "Your workbook contains pre-treatment and post-treatment measurements. These can support a PPC2 effect estimate, but the exact precision usually also needs the pre–post relationship or equivalent change-score information.",
            "A meta-analysis needs to know not only how large each study's effect is, but also how precise it is. That precision determines how much weight the study receives.",
            "First look for actual pre–post correlations, SD(change), SE(change), paired test results, or a suitable confidence interval. Only if those are unavailable should you consider an explicitly labelled correlation sensitivity analysis.",
            [
                _option("provide_actual_correlation", "I can provide the actual pre–post correlation", "Best when reported by the study or calculated from available participant-level data.", True),
                _option("provide_change_sd_or_se", "I can provide SD(change) or SE(change)", "May allow the variance to be calculated without assuming a correlation.", True),
                _option("approved_correlation_sensitivity", "Consider several assumed correlations in sensitivity analysis", "Shows how conclusions change under assumptions; assumed values are not observed study data."),
                _option("hold_variance", "Keep unresolved records pending", "Do not invent precision information. These records wait until enough information is available."),
            ], suffix="variance-plan",
        ))

    # Shared-control / multi-arm flags are taken from the structured issue table.
    issues = review.get("issues", pd.DataFrame()) if isinstance(review, dict) else pd.DataFrame()
    shared_rows = pd.DataFrame()
    if isinstance(issues, pd.DataFrame) and not issues.empty and "code" in issues.columns:
        shared_rows = issues[issues["code"].astype(str).str.upper().eq("POSSIBLE_SHARED_CONTROL_GROUP")]
    if not shared_rows.empty:
        studies = sorted(set(shared_rows.get("study ID(s)", pd.Series(dtype=str)).fillna("").astype(str)))
        studies = [x for x in studies if x]
        subject = f"{len(shared_rows)} possible shared-control finding(s)"
        if studies:
            subject += " — " + ", ".join(studies)
        explanation = " ".join(shared_rows.get("what_it_means", pd.Series(dtype=str)).fillna("").astype(str).unique().tolist())
        decisions.append(_decision(
            "Multi-arm study dependence", "High", subject,
            explanation or "At least one study may compare multiple treatment groups with the same control group.",
            "If the same control participants are counted more than once as if they were separate people, that study can receive too much weight.",
            "Do not pool these comparisons as independent by default. Choose a method that handles the shared control, after checking the study design and analysis plan.",
            [
                _option("combine_arms", "Combine eligible intervention arms", "Can be suitable when the arms are sufficiently similar and combining them makes scientific sense."),
                _option("split_control", "Split the shared control group across comparisons", "Can avoid double-counting in some pairwise analyses; the calculation must adjust control sample size correctly."),
                _option("multivariate_model", "Use a multivariate or robust-variance method", "Keeps multiple comparisons while accounting for their dependence, if suitable data and methods are available."),
                _option("hold_multiarm", "Keep this study pending until dependence is handled", "Prevents an invalid assumption that comparisons are independent."),
            ], suffix="shared-control",
        ))

    # Primary estimand is always researcher-selected.
    decisions.append(_decision(
        "Primary analysis method", "High", "Choose the effect measure required by your research question and protocol",
        "MetaPy can calculate more than one kind of effect, but these methods do not all answer exactly the same question.",
        "Pooling different estimands together can make the overall result scientifically misleading.",
        "If your protocol is based on improvement from baseline in controlled pre–post trials, Morris PPC2 is a reasonable candidate. Confirm it against your protocol before proceeding.",
        [
            _option("morris_ppc2", "Morris PPC2 — compare improvement between groups", "Uses baseline and follow-up measurements and standardises the difference in change using baseline SD."),
            _option("change_smd", "Change-score SMD — compare standardised changes", "Uses the SD of change as the standardiser; requires change-score variability or an approved way to derive it."),
            _option("ancova_giv", "ANCOVA-adjusted estimate using generic inverse variance", "Often useful when trials report adjusted treatment effects with SEs or confidence intervals. It is a different source estimate and needs compatibility review."),
            _option("not_decided", "Do not choose yet — review the protocol and studies first", "Keeps the workflow open and prevents pooling until the estimand is confirmed."),
        ], suffix="primary-estimand",
    ))

    # Data readiness and no silent exclusions.
    decisions.append(_decision(
        "Study eligibility", "Medium", "Records with incomplete or incompatible information",
        "Some records may have missing values, unclear outcome timing, multiple arms, or an unclear relationship to the selected effect measure.",
        "A study should not be deleted just because MetaPy cannot calculate it automatically. The reason and researcher decision need to be recorded.",
        "Review each flagged record individually; correct extraction errors, request missing data, retain as pending, or exclude with a reason.",
        [
            _option("correct_extraction", "Correct the extracted workbook data", "Use only when the source supports the correction; preserve the original and record the change."),
            _option("request_missing_data", "Seek missing information from the study authors/source", "Keeps the study potentially eligible while more information is sought."),
            _option("keep_pending", "Keep the record pending", "No forced decision; it will not enter the pool until resolved."),
            _option("exclude_with_reason", "Exclude with a documented reason", "Exclusion is recorded and the source data remains preserved."),
        ], suffix="eligibility",
    ))

    rows = []
    for d in decisions:
        rows.append({
            "Decision ID": d.decision_id,
            "Category": d.category,
            "Priority": d.priority,
            "Subject": d.subject,
            "What MetaPy found (simple words)": d.plain_language_finding,
            "Why it matters": d.why_it_matters,
            "MetaPy suggestion": d.recommendation,
            "Available options": json.dumps([asdict(o) for o in d.options], ensure_ascii=False),
            "Your choice": ("assume_ppc2_unverified" if assume_reported_g_is_ppc2 and d.category == "Reported effect-size method" else ""),
            "Status": d.default_status,
            "Exclusion reason": "",
            "Decision details (structured)": "{}",
            "Decision recorded (UTC)": "",
            "Source": d.source,
        })
    return pd.DataFrame(rows)


def _parse_structured_details(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(_safe(value) or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _as_float(value: Any) -> float | None:
    try:
        if value is None or _safe(value) == "":
            return None
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


def validate_researcher_decisions(queue: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a decision queue populated through fixed-choice form controls.

    No free-text explanation is required. Additional evidence and assumptions
    are captured as structured form selections and numeric fields.
    """
    if queue is None or queue.empty:
        return pd.DataFrame(), pd.DataFrame([{"Severity": "Error", "Finding": "There are no decisions to review."}])
    out = queue.copy()
    findings: list[dict[str, str]] = []
    required = {"Decision ID", "Available options", "Your choice", "Status"}
    missing = required - set(out.columns)
    if missing:
        return out, pd.DataFrame([{"Severity": "Error", "Finding": f"Missing required columns: {', '.join(sorted(missing))}"}])

    for idx, row in out.iterrows():
        choice = _safe(row.get("Your choice"))
        status = _safe(row.get("Status")) or "Pending"
        try:
            options = json.loads(_safe(row.get("Available options")) or "[]")
        except Exception:
            options = []
        value_to_label = {str(o.get("value")): str(o.get("label")) for o in options if isinstance(o, dict)}
        label_to_value = {label: value for value, label in value_to_label.items()}
        allowed = set(value_to_label)
        if choice in label_to_value:
            choice = label_to_value[choice]
            out.at[idx, "Your choice"] = choice
        if choice and choice not in allowed:
            findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: the selected option is not in this decision's option list."})
        if status not in {"Pending", "Approved", "Rejected", "Deferred"}:
            findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: invalid decision status."})
        if status == "Approved" and not choice:
            findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select an option before approving this decision."})

        details = _parse_structured_details(row.get("Decision details (structured)"))
        category = _safe(row.get("Category"))
        subject = _safe(row.get("Subject"))
        exclusion_reason = _safe(row.get("Exclusion reason"))

        if status == "Approved" and choice == "exclude_with_reason":
            valid_reasons = {value for _, value in EXCLUSION_REASON_OPTIONS}
            if exclusion_reason not in valid_reasons:
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select an exclusion reason from the form."})

        if status == "Approved" and choice == "not_decided":
            findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: this option means the analysis method is not selected yet. Keep it Pending or Deferred."})

        if status == "Approved" and choice == "confirm_from_source":
            method_status = _safe(details.get("reported_method_status"))
            valid_method_status = {"confirmed_ppc2", "confirmed_change_smd", "confirmed_post_smd", "confirmed_ancova", "unclear"}
            if method_status not in valid_method_status:
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select what the original study report says about the Hedges' g method."})

        if status == "Approved" and choice == "assume_ppc2_unverified":
            if not bool(details.get("acknowledge_unverified_ppc2", False)):
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: tick the confirmation box to acknowledge that the PPC2 classification is unverified."})
            details["reported_g_method_status"] = "assumed_morris_ppc2_unverified"
            details["assumption_warning"] = "The workbook does not independently confirm the reported Hedges' g calculation method."

        if status == "Approved" and choice == "provide_actual_correlation":
            source = _safe(details.get("correlation_source"))
            valid_sources = {"reported_per_study", "calculated_per_study", "author_provided_per_study"}
            if source not in valid_sources:
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select how the actual pre–post correlations will be obtained for each study."})

        if status == "Approved" and choice == "provide_change_sd_or_se":
            kind = _safe(details.get("change_information_type"))
            source = _safe(details.get("change_information_source"))
            valid_kinds = {"sd_change", "se_change", "change_ci", "paired_t", "paired_p"}
            valid_change_sources = {"original_report", "author_provided", "documented_calculation"}
            if kind not in valid_kinds:
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select which type of change-score information is available."})
            if source not in valid_change_sources:
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select where the change-score information will come from."})

        if status == "Approved" and choice == "approved_correlation_sensitivity":
            values = details.get("correlation_values", [])
            try:
                values = sorted({float(x) for x in values})
            except (TypeError, ValueError):
                values = []
            if len(values) < 2 or any(x < 0 or x > 1 for x in values):
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select at least two correlation values between 0 and 1 for sensitivity analysis."})
            if not bool(details.get("acknowledge_assumed_correlations", False)):
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: confirm that these correlation values are assumptions, not observed data."})
            if _safe(details.get("sensitivity_scope")) not in {"common_grid", "study_specific_grid"}:
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select how the assumed correlation values will be applied in sensitivity analysis."})

        if status == "Approved" and category == "Multi-arm study dependence" and choice == "combine_arms":
            similarity = _safe(details.get("arm_similarity"))
            if similarity not in {"similar", "not_similar", "uncertain"}:
                findings.append({"Severity": "Error", "Finding": f"Row {idx + 1}: select whether the intervention arms are similar enough to combine."})
            elif similarity != "similar":
                findings.append({"Severity": "Warning", "Finding": f"Row {idx + 1}: combining arms was selected, but the form records that similarity is not established. Reconsider before execution."})

        if status == "Approved" and category == "Primary analysis method":
            alignment = _safe(details.get("protocol_alignment"))
            if choice in {"morris_ppc2", "change_smd", "ancova_giv"} and alignment not in {"registered_protocol", "documented_amendment", "not_sure"}:
                findings.append({"Severity": "Warning", "Finding": f"Row {idx + 1}: indicate whether the selected method matches the registered protocol."})
            elif alignment == "not_sure":
                findings.append({"Severity": "Warning", "Finding": f"Row {idx + 1}: the selected method still needs protocol confirmation."})

        if "Decision recorded (UTC)" in out.columns and status == "Approved" and not _safe(row.get("Decision recorded (UTC)")):
            out.at[idx, "Decision recorded (UTC)"] = datetime.now(timezone.utc).isoformat()
        if "Decision details (structured)" in out.columns:
            out.at[idx, "Decision details (structured)"] = json.dumps(details, ensure_ascii=False, sort_keys=True)

    if not findings:
        findings.append({"Severity": "Info", "Finding": "All selected decisions passed the form checks. This records decisions; it does not itself run statistical calculations."})
    return out, pd.DataFrame(findings)

def decision_summary(audit: pd.DataFrame) -> dict[str, int]:
    if audit is None or audit.empty or "Status" not in audit:
        return {"total": 0, "approved": 0, "pending": 0, "rejected": 0, "deferred": 0}
    status = audit["Status"].fillna("Pending").astype(str).str.lower()
    return {
        "total": int(len(audit)),
        "approved": int((status == "approved").sum()),
        "pending": int((status == "pending").sum()),
        "rejected": int((status == "rejected").sum()),
        "deferred": int((status == "deferred").sum()),
    }


def build_approved_analysis_plan(queue: pd.DataFrame) -> dict[str, Any]:
    """Create a conservative execution plan from approved decisions.

    This is a gatekeeper, not a pooling engine. It deliberately keeps pooling
    blocked until effect variances and multi-arm dependence are independently
    resolved by their validated calculation modules.
    """
    audit, findings = validate_researcher_decisions(queue)
    errors = findings[findings["Severity"].astype(str).eq("Error")] if not findings.empty else pd.DataFrame()
    if not errors.empty:
        return {"valid": False, "pooling_authorized": False, "errors": findings, "actions": []}
    approved = audit[audit["Status"].astype(str).eq("Approved")].copy()
    actions = []
    for _, row in approved.iterrows():
        actions.append({"decision_id": row["Decision ID"], "category": row["Category"],
                        "choice": row["Your choice"],
                        "details": _parse_structured_details(row.get("Decision details (structured)")),
                        "exclusion_reason": _safe(row.get("Exclusion reason"))})
    primary = approved[approved["Category"].eq("Primary analysis method")]
    primary_choice = _safe(primary.iloc[0]["Your choice"]) if not primary.empty else ""
    primary_ready = bool(primary_choice and primary_choice != "not_decided")
    variance = approved[approved["Category"].eq("Sampling variance plan")]
    variance_choice = _safe(variance.iloc[0]["Your choice"]) if not variance.empty else ""
    variance_plan_approved = variance_choice in {"provide_actual_correlation", "provide_change_sd_or_se", "approved_correlation_sensitivity"}
    multi = audit[audit["Category"].eq("Multi-arm study dependence")]
    multi_ok = multi.empty or (not multi[multi["Status"].eq("Approved")].empty and
                               _safe(multi[multi["Status"].eq("Approved")].iloc[0]["Your choice"]) in
                               {"combine_arms", "split_control", "multivariate_model"})
    reported = approved[approved["Category"].eq("Reported effect-size method")]
    reported_assumption = (not reported.empty and
                           _safe(reported.iloc[0]["Your choice"]) == "assume_ppc2_unverified")
    return {
        "valid": True, "pooling_authorized": False,
        "primary_estimand": primary_choice or None,
        "primary_estimand_approved": primary_ready,
        "variance_plan": variance_choice or None,
        "variance_plan_approved": variance_plan_approved,
        "reported_g_assumed_ppc2_unverified": reported_assumption,
        "multiarm_decision_required": not multi.empty,
        "multiarm_handling_selected": multi_ok,
        "actions": actions,
        "blocking_reasons": [
            "Pooling remains blocked until each included study has a valid effect estimate and sampling variance.",
            "Pooling remains blocked until shared-control or other dependent comparisons are handled appropriately.",
            "Approved decisions are recorded but this version does not yet execute the corresponding statistical calculations.",
        ],
    }


def export_decision_audit(queue: pd.DataFrame, path: str) -> str:
    """Export decision queue and audit metadata to an Excel workbook."""
    validated, findings = validate_researcher_decisions(queue)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        validated.to_excel(writer, index=False, sheet_name="Researcher Decisions")
        findings.to_excel(writer, index=False, sheet_name="Validation Findings")
        pd.DataFrame([{"MetaPy decision engine version": VERSION,
                       "Exported UTC": datetime.now(timezone.utc).isoformat(),
                       **decision_summary(validated)}]).to_excel(writer, index=False, sheet_name="Audit Summary")
    return path
