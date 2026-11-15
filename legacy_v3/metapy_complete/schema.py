from __future__ import annotations

import re
import math
from typing import Any

VERSION = "3.3.0"

FIELD_ALIASES = {
    "study_id": ["studyid", "study_id", "study", "study_no", "trial_id", "id"],
    "outcome": ["outcome", "scale", "anxiety_scale", "biomarker", "measure", "endpoint", "variable"],
    "duration": ["duration", "timepoint", "time_point", "followup", "follow_up", "weeks", "month"],
    "intervention_arm": ["intervention_arm", "interventionarm", "arm", "treatment_arm", "intervention_group", "treatment_group", "intervention", "treatment"],
    "study_design": ["study_design", "design", "trial_design", "allocation_design"],
    "unit": ["unit", "units", "measurement_unit", "measurement_units", "outcome_unit"],
    "higher_is_worse": ["higher_is_worse", "higher_more", "outcome_direction", "direction", "effect_direction"],
    "mean_pre_i": ["mean_pre_i", "mean_pre_intervention", "pre_mean_i", "baseline_mean_i", "mean_baseline_i", "mean_pre_t"],
    "sd_pre_i": ["sd_pre_i", "sd_pre_intervention", "pre_sd_i", "baseline_sd_i", "sd_baseline_i", "sd_pre_t"],
    "mean_post_i": ["mean_post_i", "mean_post_intervention", "post_mean_i", "followup_mean_i", "mean_followup_i", "mean_post_t"],
    "sd_post_i": ["sd_post_i", "sd_post_intervention", "post_sd_i", "followup_sd_i", "sd_followup_i", "sd_post_t"],
    "n_i": ["n_i", "n_intervention", "n_treatment", "n_exp", "n_experimental", "sample_size_i", "total_i", "n1"],
    "mean_pre_c": ["mean_pre_c", "mean_pre_control", "pre_mean_c", "baseline_mean_c", "mean_baseline_c", "mean_pre_c"],
    "sd_pre_c": ["sd_pre_c", "sd_pre_control", "pre_sd_c", "baseline_sd_c", "sd_baseline_c"],
    "mean_post_c": ["mean_post_c", "mean_post_control", "post_mean_c", "followup_mean_c", "mean_followup_c"],
    "sd_post_c": ["sd_post_c", "sd_post_control", "post_sd_c", "followup_sd_c", "sd_followup_c"],
    "n_c": ["n_c", "n_control", "n_comparison", "n_comparator", "sample_size_c", "total_c", "n0"],
    "reported_effect": ["reported_effect", "reported_hedges_g", "hedges_g", "hedgesg", "hedges_g_value", "smd_hedges_g", "smd", "effect_size", "effect", "standardized_mean_difference", "standardised_mean_difference"],
    "reported_effect_method": ["effect_method", "reported_effect_method", "standardization_method", "standardisation_method", "estimand", "effect_type"],
    "effect_se": ["effect_se", "se_effect", "standard_error", "se"],
    "effect_variance": ["effect_variance", "variance", "var_effect", "sampling_variance"],
    "ci_lower": ["ci_lower", "lower_ci", "lower_95_ci", "ci_low", "lower_limit"],
    "ci_upper": ["ci_upper", "upper_ci", "upper_95_ci", "ci_high", "upper_limit"],
    "ci_level": ["ci_level", "confidence_level", "confidence_interval_level"],
    "reported_md_ci_text": ["md_95_ci", "md_95_ci_text", "mean_difference_95_ci", "mean_difference_ci", "md_ci"],
    "r_prepost_i": ["r_prepost_i", "r_pre_post_i", "correlation_i", "prepost_correlation_i", "r_intervention", "r_treatment"],
    "r_prepost_c": ["r_prepost_c", "r_pre_post_c", "correlation_c", "prepost_correlation_c", "r_control", "r_comparator"],
    "control_id": ["control_id", "shared_control_id", "comparator_id"],
    "events_i": ["events_i", "event_i", "events_intervention", "events_treatment", "event_treatment", "events_exp"],
    "total_i": ["total_i", "total_intervention", "total_treatment", "n_intervention", "n_treatment"],
    "events_c": ["events_c", "event_c", "events_control", "event_control", "events_comparator"],
    "total_c": ["total_c", "total_control", "total_comparator", "n_control", "n_comparison"],
}


def normalize_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

ALIAS_LOOKUP = {
    normalize_header(alias): field
    for field, aliases in FIELD_ALIASES.items()
    for alias in aliases
}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a"}


def as_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(str(value).replace(",", "").strip())
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def canonical_study_and_arm(raw_study: Any, explicit_arm: Any = None, outcome: Any = None) -> tuple[str, str, str]:
    """Normalize study IDs and recover an arm label only from explicit labels.

    No study identity is externally verified. A trailing arm name is parsed only
    when it is explicitly embedded in the supplied StudyID or outcome label.
    """
    study_text = "" if is_missing(raw_study) else str(raw_study).strip()
    arm = "" if is_missing(explicit_arm) else str(explicit_arm).strip()
    out = "" if is_missing(outcome) else str(outcome).strip()

    # StudyID values such as '06-Meditation' or '06 Meditation'.
    match = re.match(r"^0*(\d+)\s*[-_/ ]\s*(.+?)\s*$", study_text)
    if match:
        study = str(int(match.group(1)))
        if not arm:
            arm = match.group(2).strip()
    elif re.fullmatch(r"0*\d+", study_text):
        study = str(int(study_text))
    else:
        study = study_text.casefold().strip()

    # Outcomes such as 'HADS-A Meditation' explicitly carry the arm label.
    if not arm and out:
        m = re.match(r"^(.*?)[\s\-_]+(meditation|yoga|mindfulness|cbt|cognitive behavioural therapy|cognitive behavioral therapy|exercise|usual care|education|relaxation)\s*$", out, flags=re.I)
        if m and m.group(1).strip():
            out = m.group(1).strip()
            arm = m.group(2).strip()

    return study, arm or "Not specified", out


def normalize_direction(value: Any) -> int | None:
    if is_missing(value):
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    positive_higher = {"higher worse", "higher is worse", "higher more", "higher means worse", "higher means more severe", "higher is worse", "1", "+1"}
    negative_higher = {"higher better", "higher is better", "lower is worse", "higher means better", "-1"}
    if text in positive_higher:
        return 1  # higher-worse: a beneficial decrease already has a negative I-minus-C effect
    if text in negative_higher:
        return -1  # higher-better: reverse sign so benefit is negative
    return None
