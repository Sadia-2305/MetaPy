"""Rule-based, auditable cascading decision logic for MetaPy preparation reviews.

A parent decision can activate, deactivate or invalidate downstream review tasks.
No statistical assumptions or numerical corrections are made here.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pandas as pd


def _s(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _s(value).lower()) or "unknown"


def _options(items: list[dict[str, str]]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _item(rid: str, category: str, subject: str, finding: str, why: str,
          suggestion: str, options: list[dict[str, str]], affected: list[str],
          parent: str, severity: str = "Warning") -> dict[str, Any]:
    return {
        "Review ID": rid, "Category": category, "Priority": "High" if severity == "Critical" else "Medium",
        "Subject": subject, "What MetaPy found": finding, "Why it matters": why,
        "MetaPy suggestion": suggestion, "Available options": _options(options),
        "Your choice": "", "Status": "Pending", "Affected record IDs": "; ".join(affected),
        "Decision recorded (UTC)": "", "Severity": severity, "Is dynamic": True,
        "Triggered by": parent, "Depends on": parent,
        "Will affect": "The listed study records and their later calculation eligibility.",
        "Dependency state": "Active — parent decision approved",
        "Re-review reason": "",
    }


def _review_lookup(items: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if items is None or items.empty:
        return {}
    return { _s(r.get("Review ID")): r.to_dict() for _, r in items.iterrows() }


def _decision_state(lookup: dict[str, dict[str, Any]], rid: str) -> tuple[str, str]:
    row = lookup.get(rid, {})
    if _s(row.get("Status")) != "Approved":
        return "", _s(row.get("Status")) or "Missing"
    return _s(row.get("Your choice")), "Approved"


def _fingerprint(row: dict[str, Any], lookup: dict[str, dict[str, Any]], decision_map: dict[str, dict[str, Any]] | None = None) -> str:
    parent_ids = [x.strip() for x in _s(row.get("Depends on")).split(";") if x.strip()]
    payload = []
    decision_map = decision_map or {}
    for parent in parent_ids:
        p = lookup.get(parent, {})
        d = decision_map.get(parent, {})
        payload.append((parent, _s(p.get("Your choice")) or _s(d.get("choice")), _s(p.get("Status")) or _s(d.get("status"))))
    payload.append(("affected", _s(row.get("Affected record IDs"))))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _dynamic_items(items: pd.DataFrame, prepared: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = _review_lookup(items)
    generated: list[dict[str, Any]] = []

    # Mixed project design activates study-specific design decisions.
    design_choice, design_status = _decision_state(lookup, "PREP-DESIGN-CONFIRM")
    if design_status == "Approved" and design_choice == "mixed_designs" and not prepared.empty:
        design_options = [
            {"value": "parallel_independent", "label": "Parallel groups — separate intervention and control participants", "explanation": "Use only when each participant belongs to one study group."},
            {"value": "crossover", "label": "Crossover — participants receive more than one condition", "explanation": "Requires paired/crossover-specific effect and variance handling."},
            {"value": "cluster", "label": "Cluster-randomised — groups or sites were randomised", "explanation": "Requires attention to clustering and any reported/design effect adjustment."},
            {"value": "paired_matched", "label": "Paired or matched groups", "explanation": "Requires paired/matched dependence information."},
            {"value": "longitudinal_other", "label": "Other repeated-measures or longitudinal design", "explanation": "Requires study-specific structure and dependency review."},
            {"value": "uncertain", "label": "Not sure — keep this study pending", "explanation": "Do not apply an independent-group formula until clarified."},
        ]
        for study in sorted(prepared["Study ID"].dropna().astype(str).unique()):
            records = prepared[prepared["Study ID"].astype(str).eq(study)]
            generated.append(_item(
                f"PREP-STUDY-DESIGN-{_slug(study).upper()}", "Study-specific design",
                f"Confirm the design for Study {study}",
                f"The project contains mixed designs. This study has {len(records)} canonical record(s), but its design must be confirmed separately.",
                "Different designs require different effect-size and variance methods. A project-level label is not enough.",
                "Select the design described in this study's methods section.", design_options,
                records["Preparation record ID"].astype(str).tolist(), "PREP-DESIGN-CONFIRM", "Critical",
            ))

    # A unit decision that indicates uncertainty or conversion needs activates a follow-up.
    unit_options = [
        {"value": "same_units_verified", "label": "I checked the source: treatment and control use the same unit within each study", "explanation": "No numeric conversion is needed for the affected study records."},
        {"value": "conversion_factor_known", "label": "A conversion factor is available and must be entered before calculations", "explanation": "This confirms a factor exists, but MetaPy will keep the affected records blocked until the factor is entered and checked."},
        {"value": "unit_unresolved", "label": "I cannot verify the units yet", "explanation": "Keep the affected records pending; do not guess units or conversion factors."},
    ]
    for _, parent in items.iterrows():
        if _s(parent.get("Category")) != "Measurement unit check" or _s(parent.get("Status")) != "Approved":
            continue
        choice = _s(parent.get("Your choice"))
        if choice not in {"units_need_harmonisation", "unit_not_reported"}:
            continue
        outcome = _s(parent.get("Subject")).replace("Check measurement units for ", "")
        ids = [x for x in _s(parent.get("Affected record IDs")).split(";") if x.strip()]
        generated.append(_item(
            f"PREP-UNIT-FOLLOWUP-{_slug(outcome).upper()}", "Unit verification follow-up",
            f"Resolve unit compatibility for {outcome}",
            "The previous decision indicates that units need conversion or are not confirmed.",
            "A standardised mean difference can accommodate different units between studies, but treatment and control values within one study must be comparable.",
            "Confirm within-study units or identify a conversion factor. A factor must be entered and checked before it is applied.",
            unit_options, ids, _s(parent.get("Review ID")), "Critical",
        ))

    # Context-dependent cortisol interpretation activates per-study sampling-time review.
    cortisol_direction, cortisol_status = _decision_state(lookup, "PREP-DIRECTION-CORTISOL")
    if cortisol_status == "Approved" and cortisol_direction == "context_dependent" and not prepared.empty:
        cortisol = prepared[prepared["Outcome label"].astype(str).str.casefold().eq("cortisol")]
        timing_options = [
            {"value": "same_sampling_context", "label": "Sampling time/context is comparable within this study", "explanation": "The study's cortisol measurements were collected under comparable timing and conditions across groups."},
            {"value": "different_sampling_context", "label": "Sampling time/context differs and needs separate handling", "explanation": "Do not treat the direction as directly comparable until the difference is understood."},
            {"value": "timing_not_reported", "label": "Sampling time/context is not reported", "explanation": "Keep interpretation flagged; MetaPy cannot infer collection timing."},
        ]
        for study in sorted(cortisol["Study ID"].dropna().astype(str).unique()):
            records = cortisol[cortisol["Study ID"].astype(str).eq(study)]
            generated.append(_item(
                f"PREP-CORTISOL-TIMING-{_slug(study).upper()}", "Cortisol sampling context",
                f"Check cortisol sampling context for Study {study}",
                "Cortisol direction was marked context-dependent, so study-specific sampling context needs review.",
                "Cortisol values can depend on collection time and protocol. Direction alone does not make measurements comparable.",
                "Select the option supported by the study report.", timing_options,
                records["Preparation record ID"].astype(str).tolist(), "PREP-DIRECTION-CORTISOL", "Warning",
            ))

    # If the researcher chooses to derive missing baseline SDs, require a second explicit evidence check.
    parent_choice, parent_status = _decision_state(lookup, "PREP-PPC2-MISSING-BASELINE-SD")
    if parent_status == "Approved" and parent_choice == "derive_from_valid_statistic":
        parent = lookup["PREP-PPC2-MISSING-BASELINE-SD"]
        options = [
            {"value": "valid_source_statistic_found", "label": "I found a statistic that can validly recover the baseline SD; enter its verified value/source through the correction workflow", "explanation": "This does not calculate or insert the SD. The exact value and source must still be entered and checked."},
            {"value": "no_valid_statistic_found", "label": "No valid statistic is available — keep these records pending", "explanation": "Do not substitute an unrelated post-treatment confidence interval."},
            {"value": "source_review_pending", "label": "I have not finished checking the original report", "explanation": "Keep the records pending until the source is checked."},
        ]
        generated.append(_item(
            "PREP-PPC2-SD-SOURCE-CHECK", "Baseline SD source verification",
            "Verify a valid source for the missing baseline SDs",
            "You selected the option to look for another statistic that could recover missing baseline SD information.",
            "The source statistic must mathematically identify the baseline SD. A post-treatment between-group MD confidence interval is not sufficient.",
            "Confirm whether a valid statistic was found. If found, enter the actual verified value using the cell-correction workflow; this approval alone does not insert it.",
            options, [_s(x) for x in _s(parent.get("Affected record IDs")).split(";") if _s(x)],
            "PREP-PPC2-MISSING-BASELINE-SD", "Critical",
        ))

    return generated


def reconcile_dependency_reviews(items: pd.DataFrame, prepared: pd.DataFrame,
                                 previous_items: pd.DataFrame | None = None,
                                 decision_map: dict[str, dict[str, Any]] | None = None,
                                 changed_review_id: str | None = None) -> pd.DataFrame:
    """Generate cascading tasks and preserve decisions only when dependencies match."""
    current = items.copy(deep=True) if isinstance(items, pd.DataFrame) else pd.DataFrame()
    if current.empty:
        return current
    previous = previous_items.copy(deep=True) if isinstance(previous_items, pd.DataFrame) else pd.DataFrame()

    def is_dynamic(value: Any) -> bool:
        return _s(value).lower() in {"true", "1", "yes"}

    # Separate the stable parent review forms from previously generated child forms.
    if "Is dynamic" in current.columns:
        static = current[~current["Is dynamic"].apply(is_dynamic)].copy()
        current_dynamic = current[current["Is dynamic"].apply(is_dynamic)].copy()
    else:
        static = current.copy()
        current_dynamic = pd.DataFrame()
    if "Is dynamic" not in static.columns:
        static["Is dynamic"] = False
    static["Is dynamic"] = False

    generated = _dynamic_items(static, prepared)
    dynamic = pd.DataFrame(generated)
    if not dynamic.empty:
        # A just-submitted child choice is retained for this exact event only.
        if changed_review_id:
            submitted = current_dynamic[current_dynamic["Review ID"].astype(str).eq(_s(changed_review_id))] if not current_dynamic.empty else pd.DataFrame()
            if not submitted.empty:
                rid = _s(changed_review_id)
                target = dynamic.index[dynamic["Review ID"].astype(str).eq(rid)].tolist()
                if target:
                    j = target[0]
                    for col in ("Your choice", "Status", "Decision recorded (UTC)"):
                        if col in submitted.columns:
                            dynamic.at[j, col] = submitted.iloc[0].get(col)
        base = pd.concat([static, dynamic], ignore_index=True, sort=False)
    else:
        base = static.reset_index(drop=True)

    active_ids = set(base["Review ID"].astype(str))
    # Keep inactive child rows for audit visibility, but never carry an old approval forward.
    source_for_inactive = previous if not previous.empty else current_dynamic
    if not source_for_inactive.empty and "Is dynamic" in source_for_inactive.columns:
        for _, old in source_for_inactive.iterrows():
            rid = _s(old.get("Review ID"))
            if not rid or rid in active_ids or not is_dynamic(old.get("Is dynamic")):
                continue
            inactive = old.to_dict()
            inactive["Status"] = "Not applicable"
            inactive["Dependency state"] = "Inactive — parent choice no longer triggers this task"
            inactive["Re-review reason"] = "The upstream selection changed; this task is not currently applicable."
            inactive["Your choice"] = ""
            inactive["Decision recorded (UTC)"] = ""
            base = pd.concat([base, pd.DataFrame([inactive])], ignore_index=True, sort=False)
            active_ids.add(rid)

    # Add dependency metadata and fingerprints.
    if "Is dynamic" not in base.columns:
        base["Is dynamic"] = False
    base["Is dynamic"] = base["Is dynamic"].apply(is_dynamic)
    if "Dependency state" not in base.columns:
        base["Dependency state"] = "Active — review available"
    parent_map = {
        "PREP-METHOD-COMPATIBILITY": ("Primary analysis method", "May change the selected estimand and all downstream route checks."),
        "PREP-MULTIARM-ANCOVA-COMPATIBILITY": ("Primary analysis method; Multi-arm study dependence", "May reopen the primary-method or shared-control decision."),
        "PREP-DESIGN-CONFIRM": ("Project design setting", "May create one design review per study."),
        "PREP-PPC2-MISSING-BASELINE-SD": ("Primary analysis method; source record SD fields", "May create a source-statistic verification task."),
        "PREP-REPORTED-G-ASSUMPTION": ("Primary analysis method; Reported effect-size method", "Changes which reported effects are provisionally eligible."),
        "PREP-CELL-CORRECTION": ("Study eligibility; verified source correction", "A correction triggers re-extraction and resets prior approvals."),
    }
    for idx, row in base.iterrows():
        rid = _s(row.get("Review ID"))
        if rid.startswith("PREP-DIRECTION-"):
            depends, affects = "Outcome/instrument metadata", "Changes sign interpretation for this outcome; it may activate context-specific checks."
        elif rid.startswith("PREP-UNIT-") and "FOLLOWUP" not in rid:
            depends, affects = "Source unit metadata", "May activate a conversion or unit-verification follow-up."
        elif rid.startswith("PREP-STUDY-DESIGN-"):
            depends, affects = "PREP-DESIGN-CONFIRM", "Sets the study-specific design and restricts compatible calculation routes."
        elif rid.startswith("PREP-UNIT-FOLLOWUP-"):
            depends, affects = _s(row.get("Triggered by")), "Determines whether affected records need numeric conversion or remain blocked."
        elif rid.startswith("PREP-CORTISOL-TIMING-"):
            depends, affects = "PREP-DIRECTION-CORTISOL", "Changes cortisol interpretation status for the named study."
        elif rid == "PREP-PPC2-SD-SOURCE-CHECK":
            depends, affects = "PREP-PPC2-MISSING-BASELINE-SD", "Determines whether verified SD entry is still required or records remain pending."
        else:
            depends, affects = parent_map.get(rid, ("Source data and approved project decisions", "Updates the preparation status of affected records."))
        base.at[idx, "Depends on"] = depends
        base.at[idx, "Will affect"] = affects
        if bool(row.get("Is dynamic", False)):
            if _s(row.get("Status")) == "Not applicable":
                base.at[idx, "Dependency state"] = "Inactive — parent choice no longer triggers this task"
            else:
                base.at[idx, "Dependency state"] = "Active — parent decision approved"
        else:
            base.at[idx, "Dependency state"] = "Active — review available"

    lookup = _review_lookup(base)
    for idx, row in base.iterrows():
        base.at[idx, "Decision fingerprint"] = _fingerprint(base.loc[idx].to_dict(), lookup, decision_map)

    # Preserve older decisions only if their relevant dependency fingerprint is unchanged.
    if not previous.empty:
        old_map = _review_lookup(previous)
        for idx, row in base.iterrows():
            rid = _s(row.get("Review ID"))
            old = old_map.get(rid)
            if not old or rid == _s(changed_review_id):
                continue
            if _s(row.get("Dependency state")).startswith("Inactive"):
                base.at[idx, "Status"] = "Not applicable"
                base.at[idx, "Your choice"] = ""
                base.at[idx, "Decision recorded (UTC)"] = ""
                continue
            if _s(old.get("Status")) == "Not applicable":
                base.at[idx, "Your choice"] = ""
                base.at[idx, "Status"] = "Pending"
                base.at[idx, "Decision recorded (UTC)"] = ""
                base.at[idx, "Re-review reason"] = "This dependent task has become active again and needs a fresh review."
                continue
            old_fp = _s(old.get("Decision fingerprint"))
            new_fp = _s(row.get("Decision fingerprint"))
            if not old_fp or old_fp == new_fp:
                for col in ("Your choice", "Status", "Decision recorded (UTC)"):
                    if col in old:
                        base.at[idx, col] = old.get(col)
            else:
                base.at[idx, "Your choice"] = ""
                base.at[idx, "Status"] = "Pending"
                base.at[idx, "Decision recorded (UTC)"] = ""
                base.at[idx, "Re-review reason"] = "An upstream decision or affected record set changed. Please review this item again."
                base.at[idx, "Dependency state"] = "Re-review required — upstream change detected"
    return base

def dependency_impact_table(items: pd.DataFrame) -> pd.DataFrame:
    """Compact dependency map for the interactive UI and exported audit."""
    if items is None or items.empty:
        return pd.DataFrame(columns=["Review ID", "Decision", "Status", "Depends on", "Will affect", "Dependency state", "Re-review reason"])
    cols = ["Review ID", "Subject", "Your choice", "Status", "Depends on", "Will affect", "Dependency state", "Re-review reason"]
    for col in cols:
        if col not in items:
            items[col] = ""
    return items[cols].copy()
