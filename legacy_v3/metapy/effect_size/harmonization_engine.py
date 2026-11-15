"""MetaPy continuous-outcome effect-size harmonization engine.

The engine operates only on the researcher's uploaded workbook. It does not
perform external study verification or modify source files.

First implementation supports:
- Hedges' g retained from source and variance reconstructed from group Ns
  (parallel independent-group designs only).
- Cohen's d converted to Hedges' g using the small-sample correction J(df).
- Hedges' g independently recalculated from post-treatment means, SDs and Ns.
- Morris (2008) PPC2 Hedges' g: between-group difference in pre/post change,
  standardized by pooled baseline SD; variance requires pre/post correlation.
- Between-group change-score Hedges' g when pre/post correlations for both
  groups are explicitly present in the workbook.
- Independent-groups pooled-variance t statistic converted to Hedges' g when
  the test type/design and direction are sufficiently clear.
- Reported effect-specific SE/variance/95% CI used to derive uncertainty.

The engine deliberately does not infer missing SDs, assume pre/post
correlations, use p-value thresholds, or compare estimates from different
estimands. Other transformations (e.g. median/IQR, Welch t, partial eta²,
ordinal or binary outcomes) require additional validated pathways.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

import pandas as pd


# ----------------------------- helpers ------------------------------------

def _norm(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def _missing(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except Exception:
        return x is None


def _num(x: Any) -> float | None:
    if _missing(x):
        return None
    if isinstance(x, (int, float)):
        y = float(x)
        return y if math.isfinite(y) else None
    # Accept a leading numeric value, e.g. "-0.805 (calculated via...)".
    text = str(x).strip().replace(",", "")
    match = re.match(r"^[+\-−]?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?", text)
    if not match:
        return None
    try:
        y = float(match.group(0).replace("−", "-"))
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _first_col(columns, aliases=(), contains_all=(), excludes=()):
    items = [(c, _norm(c)) for c in columns]
    aliases_n = {_norm(a) for a in aliases}
    for col, name in items:
        if name in aliases_n:
            return col
    must = [_norm(x) for x in contains_all]
    no = [_norm(x) for x in excludes]
    for col, name in items:
        if must and all(t in name for t in must) and not any(t in name for t in no):
            return col
    return None


def _study_arm(study_value: Any, outcome_value: Any) -> tuple[str, str]:
    raw = "" if _missing(study_value) else str(study_value).strip()
    out = "" if _missing(outcome_value) else str(outcome_value).strip()
    match = re.match(r"^0*(\d+)(?:\s*[-_: ]\s*(.*))?$", raw)
    if match:
        sid = str(int(match.group(1)))
        tail = match.group(2) or ""
    else:
        sid = _norm(raw)
        tail = ""
    both = f"{tail} {out}".lower()
    arm = ""
    for candidate in ("meditation", "yoga", "exercise", "intervention", "treatment"):
        if re.search(rf"\b{re.escape(candidate)}\b", both):
            arm = candidate
            break
    return sid, arm


def _duration_key(value: Any) -> str:
    n = _num(value)
    if n is not None:
        return str(int(n)) if float(n).is_integer() else str(n)
    return "" if _missing(value) else str(value).strip().lower()


def _outcome_key(value: Any, arm: str) -> str:
    text = "" if _missing(value) else str(value).strip().lower()
    if arm:
        text = re.sub(rf"\b{re.escape(arm)}\b", " ", text, flags=re.I)
    return _norm(text)


def _j_correction(df: float) -> float:
    """Exact Hedges small-sample correction J(df), evaluated via log-gamma."""
    if df <= 1:
        raise ValueError("Hedges' correction requires degrees of freedom > 1.")
    return math.exp(math.lgamma(df / 2.0) - 0.5 * math.log(df / 2.0) - math.lgamma((df - 1.0) / 2.0))


def _g_from_independent_means(m1, sd1, n1, m2, sd2, n2):
    vals = [m1, sd1, n1, m2, sd2, n2]
    if any(v is None for v in vals):
        return None
    if n1 <= 1 or n2 <= 1 or not float(n1).is_integer() or not float(n2).is_integer() or sd1 <= 0 or sd2 <= 0:
        return None
    df = n1 + n2 - 2
    pooled_var = (((n1 - 1) * sd1 ** 2) + ((n2 - 1) * sd2 ** 2)) / df
    if pooled_var <= 0:
        return None
    d = (m1 - m2) / math.sqrt(pooled_var)
    j = _j_correction(df)
    g = j * d
    return {"d": d, "g": g, "df": df, "j": j, "pooled_sd": math.sqrt(pooled_var)}


def _variance_g_ls3(g: float, n1: float, n2: float) -> float | None:
    """Large-sample SMD variance approximation used by metafor LS3/RevMan.

    vi = 1/n1 + 1/n2 + g^2/[2*(n1+n2-3.94)].
    Appropriate here only for a conventional independent parallel-group SMD.
    """
    if n1 <= 1 or n2 <= 1 or not float(n1).is_integer() or not float(n2).is_integer():
        return None
    denom = n1 + n2 - 3.94
    if denom <= 0:
        return None
    value = 1.0 / n1 + 1.0 / n2 + (g * g) / (2.0 * denom)
    return value if math.isfinite(value) and value > 0 else None


def _columns_for_frame(df: pd.DataFrame) -> dict[str, Any]:
    c = df.columns
    out = {}
    out["study"] = _first_col(c, ("studyid", "study", "studyidentifier", "studycode"))
    out["outcome"] = _first_col(c, ("scale", "biomarker", "outcome", "outcomename", "marker", "anxietyscale"))
    # Explicit treatment-arm column aliases; retain the researcher's arm label.
    out["arm"] = _first_col(c, (
        "interventionarm", "treatmentarm", "experimentalarm", "interventiongroup",
        "treatmentgroup", "armname", "activearm", "interventionname",
    ))
    out["duration"] = _first_col(c, ("duration", "followupduration", "timepointduration", "timepoint", "followup"))
    out["n1"] = _first_col(c, ("ni", "nintervention", "nexperimental", "nactive", "n1", "ninterventiongroup", "nexperimentalgroup"))
    out["n2"] = _first_col(c, ("nc", "ncontrol", "ncomparator", "nplacebo", "n2", "ncontrolgroup"))

    # Standardized effect columns.
    out["hedges_g"] = None
    out["cohen_d"] = None
    for col in c:
        name = _norm(col)
        if ("hedges" in name and "g" in name) or name in {"hedgesg", "smdhedgesg"}:
            if not any(t in name for t in ("se", "variance", "var", "cilower", "ciupper", "lower", "upper")):
                out["hedges_g"] = col
                break
    for col in c:
        name = _norm(col)
        if name in {"cohend", "cohensd", "cohendvalue", "standardizedmeandifferencecohend"} or ("cohen" in name and "d" in name and not any(t in name for t in ("se", "variance", "var"))):
            out["cohen_d"] = col
            break

    # Group-specific means/SDs. Exact aliases first, then conservative token matching.
    aliases = {
        "post_m1": ("meanposti", "postmeani", "postinterventionmean", "interventionpostmean", "meanpostintervention", "meanposttreatmenti", "posttreatmentmeani"),
        "post_s1": ("sdposti", "post sdi", "postinterventionsd", "interventionpostsd", "sdpostintervention", "sdposttreatmenti", "posttreatmentsdi"),
        "post_m2": ("meanpostc", "postmeanc", "postcontrolmean", "controlpostmean", "meanpostcontrol", "meanposttreatmentc", "posttreatmentmeanc"),
        "post_s2": ("sdpostc", "post sdc", "postcontrolsd", "controlpostsd", "sdpostcontrol", "sdposttreatmentc", "posttreatmentsdc"),
        "pre_m1": ("meanprei", "premeani", "preinterventionmean", "interventionpremean", "meanpreintervention", "pretreatmentmeani"),
        "pre_s1": ("sdprei", "preinterventionsd", "interventionpresd", "sdpreintervention", "pretreatmentsdi"),
        "pre_m2": ("meanprec", "premeanc", "precontrolmean", "controlpremean", "meanprecontrol", "pretreatmentmeanc"),
        "pre_s2": ("sdprec", "precontrolsd", "controlpresd", "sdprecontrol", "pretreatmentsdc"),
    }
    for key, a in aliases.items():
        out[key] = _first_col(c, a)

    # Explicit pre/post correlations only; avoid guessing a generic "r" column.
    out["r1"] = _first_col(c, ("rpreposti", "prepostcorrelationi", "prepostcorrelationintervention", "correlationprepostintervention", "interventionprepostcorrelation", "rinterventionprepost"))
    out["r2"] = _first_col(c, ("rprepostc", "prepostcorrelationc", "prepostcorrelationcontrol", "correlationprepostcontrol", "controlprepostcorrelation", "rcontrolprepost"))
    out["r_common"] = _first_col(c, ("rprepost", "prepostcorrelation", "commonprepostcorrelation", "withinpersoncorrelation"))

    # Effect-specific uncertainty. Generic SE/variance are only accepted if they are clearly labelled.
    out["g_se"] = _first_col(c, ("hedgesgse", "sehedgesg", "smdse", "sesmd"))
    out["g_var"] = _first_col(c, ("hedgesgvariance", "variancehedgesg", "hedgesgvar", "varhedgesg", "smdvariance", "variancesmd"))
    out["g_ci_low"] = _first_col(c, ("hedgesgcilower", "hedgesglowerci", "cilowerhedgesg", "smdcilower", "smdlowerci"))
    out["g_ci_high"] = _first_col(c, ("hedgesgciupper", "hedgesgupperci", "ciupperhedgesg", "smdciupper", "smdupperci"))

    # Explicit test-statistic fields for an independent pooled-variance t test.
    out["t"] = _first_col(c, ("t", "tvalue", "tstat", "tstatistic", "independentt", "independenttteststatistic"))
    out["f"] = _first_col(c, ("f", "fvalue", "fstat", "fstatistic", "anova_f"))
    out["df"] = _first_col(c, ("df", "degreesoffreedom", "tdegreesoffreedom", "dferror"))
    out["df1"] = _first_col(c, ("df1", "numeratordf", "dfnumerator"))
    out["df2"] = _first_col(c, ("df2", "denominatordf", "dfdenominator", "dferror"))
    out["eta2"] = _first_col(c, ("partialeta2", "partialetasquared", "etap2"))
    out["test_type"] = _first_col(c, ("testtype", "statisticaltest", "ttesttype", "analysismethod"))
    return out


def _entry_value(row: pd.Series, col):
    if col is None:
        return None
    return _num(row.get(col))


def _source_loc(entry: dict[str, Any]) -> str:
    return f"{entry['sheet']}!row {entry['excel_row']}"


def _source_values_disagree(values, *, exact: bool = False, abs_tol: float = 0.005, rel_tol: float = 1e-6) -> bool:
    """Return True when duplicate representations disagree beyond tolerance.

    Sample sizes and degrees of freedom are compared exactly. Other numeric
    extraction values allow a 0.005 absolute rounding tolerance by default.
    This tolerance is only for comparing repeated source representations; it
    is not the tolerance used to compare an independently calculated effect.
    """
    numbers = []
    for item in values:
        if isinstance(item, dict):
            value = item.get("value")
        else:
            value = item[0]
        if value is not None:
            numbers.append(float(value))
    if len(numbers) < 2:
        return False
    reference = numbers[0]
    if exact:
        return any(value != reference for value in numbers[1:])
    return any(not math.isclose(value, reference, rel_tol=rel_tol, abs_tol=abs_tol) for value in numbers[1:])


# --------------------------- main engine ----------------------------------

def harmonize_workbook(
    sheets: dict[str, pd.DataFrame],
    *,
    estimand: str = "posttest_smd",
    design: str = "parallel_independent",
    comparison_abs_tolerance: float = 0.02,
    comparison_rel_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Create one harmonized Hedges-g record per study/arm/outcome/timepoint.

    Parameters
    ----------
    estimand:
      - posttest_smd: between-group SMD using post-treatment scores.
      - change_smd: between-group SMD using change scores; requires both
        groups' explicitly reported pre/post correlations.
      - reported_only: do not recalculate from raw values; retain reported g, or
        convert a reported Cohen's d if a reported g is absent.
    design:
      - parallel_independent: permits conventional independent-group variance.
      - other values: estimates may be retained, but no independent-group
        variance is inferred.
    """
    # Backward-compatible alias used by earlier review-workbench sessions.
    # "auto" means preserve all eligible routes for researcher review; the
    # harmonisation pass uses posttest SMD as its provisional comparison route.
    if estimand == "auto":
        estimand = "posttest_smd"
    if estimand not in {"posttest_smd", "morris_ppc2", "change_smd", "reported_only"}:
        raise ValueError("estimand must be posttest_smd, morris_ppc2, change_smd, or reported_only")

    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for sheet_name, frame in sheets.items():
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        cols = _columns_for_frame(frame)
        if cols["study"] is None or cols["outcome"] is None:
            continue
        for idx, row in frame.iterrows():
            if row.isna().all() or _missing(row.get(cols["study"])):
                continue
            sid, arm = _study_arm(row.get(cols["study"]), row.get(cols["outcome"]))
            if cols.get("arm") is not None and not _missing(row.get(cols["arm"])):
                explicit_arm = str(row.get(cols["arm"])).strip()
                if explicit_arm:
                    arm = explicit_arm
            outcome_key = _outcome_key(row.get(cols["outcome"]), arm)
            if not sid or not outcome_key:
                continue
            duration = row.get(cols["duration"]) if cols["duration"] is not None else None
            key = (sid, arm, outcome_key, _duration_key(duration))
            entry = {
                "sheet": sheet_name,
                "excel_row": int(idx) + 2,
                "row": row,
                "cols": cols,
                "study_id": sid,
                "arm": arm,
                "outcome": str(row.get(cols["outcome"])),
                "duration": duration,
                "key": key,
            }
            grouped[key].append(entry)

    records = []
    calculation_log = []
    issue_rows = []

    for key, entries in grouped.items():
        # Keep only entries that actually contain an effect or raw/alternative stats.
        reported_g_items = []
        reported_d_items = []
        for e in entries:
            row, c = e["row"], e["cols"]
            g = _entry_value(row, c["hedges_g"])
            d = _entry_value(row, c["cohen_d"])
            if g is not None:
                reported_g_items.append({"value": g, "sheet": e["sheet"], "row": e["excel_row"], "column": str(c["hedges_g"])})
            if d is not None:
                reported_d_items.append({"value": d, "sheet": e["sheet"], "row": e["excel_row"], "column": str(c["cohen_d"])})

        # Merge values across the detailed and summary worksheet representations.
        field_values: dict[str, list[tuple[float, str, str, int]]] = defaultdict(list)
        uncertainty: dict[str, list[tuple[float, str, str, int]]] = defaultdict(list)
        for e in entries:
            row, c = e["row"], e["cols"]
            g = _entry_value(row, c["hedges_g"])
            d = _entry_value(row, c["cohen_d"])
            for field in ("n1", "n2", "post_m1", "post_s1", "post_m2", "post_s2", "pre_m1", "pre_s1", "pre_m2", "pre_s2", "r1", "r2", "r_common", "t", "f", "df", "df1", "df2", "eta2"):
                val = _entry_value(row, c[field])
                if val is not None:
                    field_values[field].append((val, e["sheet"], str(c[field]), e["excel_row"]))
            for field in ("g_se", "g_var", "g_ci_low", "g_ci_high"):
                val = _entry_value(row, c[field])
                if val is not None:
                    uncertainty[field].append((val, e["sheet"], str(c[field]), e["excel_row"]))

            if g is not None:
                field_values["reported_g"].append((g, e["sheet"], str(c["hedges_g"]), e["excel_row"]))
            if d is not None:
                field_values["reported_d"].append((d, e["sheet"], str(c["cohen_d"]), e["excel_row"]))

        exact_fields = {"n1", "n2", "df", "df1", "df2"}
        field_conflicts = {}
        for field, values in field_values.items():
            # Repeated source values are only considered duplicates when
            # entries occur in more than one worksheet for this canonical key.
            if len({item[1] for item in values}) > 1 and _source_values_disagree(
                values, exact=field in exact_fields
            ):
                field_conflicts[field] = values
        for field, values in uncertainty.items():
            if len({item[1] for item in values}) > 1 and _source_values_disagree(values):
                field_conflicts[field] = values

        # Hedges' g has a dedicated conflict finding below so that the
        # recalculation-precedence policy can still select a valid raw result.
        field_conflicts.pop("reported_g", None)

        def first(field):
            if field in field_conflicts:
                return None
            values = field_values.get(field, [])
            return values[0][0] if values else None

        def first_uncertainty(field):
            if field in field_conflicts:
                return None
            values = uncertainty.get(field, [])
            return values[0] if values else None

        def consistent_reported(items):
            return not _source_values_disagree(items)

        # Build a transparent map of which worksheet cells supplied each field.
        provenance_parts = []
        for field, values in sorted(field_values.items()):
            locs = ", ".join(f"{sheet}!{col} (row {row})" for _, sheet, col, row in values)
            provenance_parts.append(f"{field}: {locs}")
        for field, values in sorted(uncertainty.items()):
            locs = ", ".join(f"{sheet}!{col} (row {row})" for _, sheet, col, row in values)
            provenance_parts.append(f"{field}: {locs}")
        source_value_provenance = "; ".join(provenance_parts)

        # Any non-effect field disagreement is excluded from reconstruction
        # rather than arbitrarily taking the first worksheet value.
        if field_conflicts:
            details = []
            conflict_sources = []
            for field, values in field_conflicts.items():
                label = field.replace("_", " ")
                shown = []
                for value, sheet, column, row_no in values:
                    shown.append(f"{sheet}!{column} row {row_no}={value}")
                    conflict_sources.append(sheet)
                details.append(f"{label}: " + " | ".join(shown))
            issue_rows.append({
                "severity": "Warning", "code": "CROSS_SHEET_FIELD_CONFLICT",
                "finding": "The same study comparison has conflicting values across worksheets",
                "study_id": key[0], "arm": key[1], "outcome": key[2],
                "source_locations": "; ".join(sorted(set(conflict_sources))),
                "reported_g": None, "calculated_g": None, "difference": None,
                "message": (
                    f"For Study {key[0]} ({key[1] or 'unspecified arm'}; {key[2]}), "
                    "the same canonical record contains conflicting repeated values: "
                    + "; ".join(details)
                    + ". MetaPy excluded the conflicting field(s) from calculations rather than choosing one worksheet value. "
                    "Check the original extraction and correct only a confirmed error."
                ),
            })

        n1, n2 = first("n1"), first("n2")
        raw_candidates = []
        raw_note = ""
        if estimand == "posttest_smd":
            raw = _g_from_independent_means(first("post_m1"), first("post_s1"), n1,
                                           first("post_m2"), first("post_s2"), n2)
            if raw:
                raw_candidates.append(("post-treatment raw means/SDs", raw))
            else:
                raw_note = "Post-treatment raw-data route unavailable: both group post means, SDs and sample sizes are required."
        elif estimand == "morris_ppc2":
            pre_m1, pre_s1, post_m1 = first("pre_m1"), first("pre_s1"), first("post_m1")
            pre_m2, pre_s2, post_m2 = first("pre_m2"), first("pre_s2"), first("post_m2")
            if all(v is not None for v in (pre_m1, pre_s1, post_m1, pre_m2, pre_s2, post_m2, n1, n2)) and n1 > 1 and n2 > 1 and float(n1).is_integer() and float(n2).is_integer() and pre_s1 > 0 and pre_s2 > 0:
                df = n1 + n2 - 2
                pooled_pre_var = (((n1 - 1) * pre_s1 ** 2) + ((n2 - 1) * pre_s2 ** 2)) / df
                if pooled_pre_var > 0:
                    change_difference = (post_m1 - pre_m1) - (post_m2 - pre_m2)
                    d_ppc2 = change_difference / math.sqrt(pooled_pre_var)
                    j = _j_correction(df)
                    raw_candidates.append(("Morris (2008) PPC2: difference in mean change / pooled baseline SD", {
                        "d": d_ppc2, "g": j * d_ppc2, "df": df, "j": j,
                        "pooled_sd": math.sqrt(pooled_pre_var),
                    }))
                else:
                    raw_note = "Morris PPC2 calculation unavailable because the pooled baseline variance is not positive."
            else:
                raw_note = "Morris PPC2 calculation requires pre/post means, baseline SDs and group sample sizes. Post-treatment SDs and pre/post correlations are not required to calculate the effect estimate."
        elif estimand == "change_smd":
            pre_m1, pre_s1, post_m1, post_s1 = first("pre_m1"), first("pre_s1"), first("post_m1"), first("post_s1")
            pre_m2, pre_s2, post_m2, post_s2 = first("pre_m2"), first("pre_s2"), first("post_m2"), first("post_s2")
            r1, r2 = first("r1"), first("r2")
            if all(v is not None for v in (pre_m1, pre_s1, post_m1, post_s1, pre_m2, pre_s2, post_m2, post_s2, r1, r2, n1, n2)) and -1 <= r1 <= 1 and -1 <= r2 <= 1:
                sd_change_1_sq = pre_s1**2 + post_s1**2 - 2*r1*pre_s1*post_s1
                sd_change_2_sq = pre_s2**2 + post_s2**2 - 2*r2*pre_s2*post_s2
                if sd_change_1_sq > 0 and sd_change_2_sq > 0:
                    raw = _g_from_independent_means(post_m1-pre_m1, math.sqrt(sd_change_1_sq), n1,
                                                   post_m2-pre_m2, math.sqrt(sd_change_2_sq), n2)
                    if raw:
                        raw_candidates.append(("change-score raw data + reported pre/post correlations", raw))
                if not raw_candidates:
                    raw_note = "Change-score calculation was not possible because the reconstructed change SDs were invalid."
            else:
                raw_note = "Change-score calculation requires pre/post means and SDs plus explicit within-group pre/post correlations for both groups. No correlation was assumed."

        # Alternative Cohen's d route.
        converted_d = None
        reported_d_is_consistent = consistent_reported(reported_d_items)
        if reported_d_items and reported_d_is_consistent and n1 is not None and n2 is not None and n1 > 1 and n2 > 1:
            df = n1 + n2 - 2
            d_val = reported_d_items[0]["value"]
            converted_d = {"d": d_val, "g": _j_correction(df) * d_val, "df": df, "j": _j_correction(df), "pooled_sd": None}

        # Alternative independent pooled-variance t/F/eta-squared routes.
        # Signed direction is mandatory; F and eta-squared alone do not contain it.
        converted_t = None
        t_value = first("t")
        f_value = first("f")
        eta2_value = first("eta2")
        test_type_text = " ".join(str(e["row"].get(e["cols"]["test_type"])) for e in entries if e["cols"]["test_type"] is not None).lower()
        design_ok = design == "parallel_independent" and "welch" not in test_type_text and "paired" not in test_type_text and "repeated" not in test_type_text
        independent_t_test_confirmed = any(token in test_type_text for token in ("independent", "two sample", "two-sample", "between group", "pooled variance"))
        direction = None
        if first("post_m1") is not None and first("post_m2") is not None:
            direction = 1.0 if first("post_m1") > first("post_m2") else (-1.0 if first("post_m1") < first("post_m2") else 0.0)
        elif first("pre_m1") is not None and first("pre_m2") is not None and first("post_m1") is not None and first("post_m2") is not None:
            change_diff = (first("post_m1") - first("pre_m1")) - (first("post_m2") - first("pre_m2"))
            direction = 1.0 if change_diff > 0 else (-1.0 if change_diff < 0 else 0.0)
        if design_ok and n1 and n2 and n1 > 1 and n2 > 1:
            if t_value is not None and independent_t_test_confirmed:
                # Convert only when the workbook explicitly identifies an independent-groups test.
                df_val = first("df") or (n1 + n2 - 2)
                if df_val > 1:
                    d_val = t_value * math.sqrt(1.0/n1 + 1.0/n2)
                    converted_t = {"d": d_val, "g": _j_correction(df_val) * d_val, "df": df_val, "j": _j_correction(df_val), "pooled_sd": None}
            elif f_value is not None and f_value >= 0 and direction in (-1.0, 1.0):
                # A one-degree-of-freedom between-group F test is equivalent to t^2.
                df1 = first("df1") or 1.0
                df_val = first("df2") or first("df") or (n1 + n2 - 2)
                if math.isclose(df1, 1.0) and df_val > 1:
                    d_val = direction * math.sqrt(f_value) * math.sqrt(1.0/n1 + 1.0/n2)
                    converted_t = {"d": d_val, "g": _j_correction(df_val) * d_val, "df": df_val, "j": _j_correction(df_val), "pooled_sd": None}
            elif eta2_value is not None and 0 < eta2_value < 1 and direction in (-1.0, 1.0):
                # Partial eta-squared from a two-group one-way between-subjects model.
                df_val = first("df2") or first("df") or (n1 + n2 - 2)
                if df_val > 1:
                    f_equiv = eta2_value / (1.0 - eta2_value) * df_val
                    d_val = direction * math.sqrt(f_equiv) * math.sqrt(1.0/n1 + 1.0/n2)
                    converted_t = {"d": d_val, "g": _j_correction(df_val) * d_val, "df": df_val, "j": _j_correction(df_val), "pooled_sd": None}

        # Select estimate under the agreed precedence.
        reported_g_is_consistent = consistent_reported(reported_g_items)
        reported_g = reported_g_items[0]["value"] if reported_g_items and reported_g_is_consistent else None
        selected_g = None
        recalculated_g = raw_candidates[0][1]["g"] if raw_candidates else None
        method = ""
        selection_reason = ""
        status = "unresolved"

        if raw_candidates and estimand != "reported_only":
            selected_g = raw_candidates[0][1]["g"]
            method = raw_candidates[0][0]
            selection_reason = "Valid raw-data reconstruction takes precedence over the reported effect under the selected estimand."
            if reported_g is not None:
                diff = selected_g - reported_g
                tol = max(comparison_abs_tolerance, comparison_rel_tolerance * max(abs(selected_g), abs(reported_g)))
                if abs(diff) <= tol:
                    status = "recalculated; agrees with reported g"
                else:
                    status = "recalculated; differs from reported g"
                    issue_rows.append({
                        "severity": "Warning", "code": "G_RECALCULATION_DISAGREEMENT",
                        "finding": "Independently calculated Hedges' g differs from the reported value",
                        "study_id": key[0], "arm": key[1], "outcome": key[2],
                        "source_locations": "; ".join(_source_loc(e) for e in entries),
                        "reported_g": reported_g, "calculated_g": selected_g,
                        "difference": diff,
                        "message": (f"For Study {key[0]} ({key[1] or 'unspecified arm'}; {key[2]}), the selected {estimand} calculation produced g={selected_g:.6g}, versus reported g={reported_g:.6g}. MetaPy selected the independently calculated value. Confirm that the reported and recalculated values represent the same estimand and direction before pooling."),
                    })
            else:
                status = "calculated from raw data"
        elif reported_g is not None:
            selected_g = reported_g
            method = "reported Hedges' g retained"
            selection_reason = "No valid independent raw-data reconstruction was available for the selected estimand; reported g retained."
            status = "reported g retained"
        elif converted_d is not None:
            selected_g = converted_d["g"]
            method = "Cohen's d converted to Hedges' g"
            selection_reason = "No raw-data reconstruction or reported Hedges' g was available; Cohen's d was converted using J(df)."
            status = "converted from Cohen's d"
        elif converted_t is not None:
            selected_g = converted_t["g"]
            method = "independent pooled-variance t statistic converted to Hedges' g"
            selection_reason = "Signed independent-groups pooled-variance t statistic converted using group Ns and Hedges' correction."
            status = "converted from t statistic"

        # If raw data unavailable but reported Cohen d + reported g both exist,
        # retain g under explicit source priority and log converted d as a check.
        if reported_g is not None and converted_d is not None and not raw_candidates:
            converted_diff = converted_d["g"] - reported_g
            selection_reason += f" Cohen's d also implies g={converted_d['g']:.6g}; reported g retained because there was no independent raw-data reconstruction."
            if abs(converted_diff) > max(comparison_abs_tolerance, comparison_rel_tolerance * max(abs(converted_d["g"]), abs(reported_g))):
                status += "; Cohen-d conversion differs"

        # Obtain sampling variance: source uncertainty first, then a design- and
        # estimand-specific approximation. Morris PPC2 requires a pre/post
        # correlation; it must never fall through to the ordinary posttest-SMD
        # variance formula.
        variance = None
        variance_method = ""
        rho_used = None
        se = None
        ci_low = None
        ci_high = None
        # Accept reported variance/SE only from clearly effect-specific columns.
        if selected_g is not None:
            u_var = first_uncertainty("g_var")
            u_se = first_uncertainty("g_se")
            u_low = first_uncertainty("g_ci_low")
            u_high = first_uncertainty("g_ci_high")
            if u_var is not None:
                variance = u_var[0]
                variance_method = f"reported effect-specific variance ({u_var[1]}:{u_var[2]})"
            elif u_se is not None:
                se0 = u_se[0]
                if se0 > 0:
                    variance = se0 ** 2
                    variance_method = f"squared reported effect-specific SE ({u_se[1]}:{u_se[2]})"
            elif u_low is not None and u_high is not None:
                lo = u_low[0]
                hi = u_high[0]
                if hi > lo:
                    variance = ((hi - lo) / 3.92) ** 2
                    variance_method = "derived from effect-specific 95% CI using normal approximation"
            if variance is None and estimand == "morris_ppc2" and design == "parallel_independent" and n1 is not None and n2 is not None:
                r1, r2, r_common = first("r1"), first("r2"), first("r_common")
                rho = None
                if r1 is not None and r2 is not None and -1 <= r1 <= 1 and -1 <= r2 <= 1:
                    rho = (n1 * r1 + n2 * r2) / (n1 + n2)
                elif r_common is not None and -1 <= r_common <= 1:
                    rho = r_common
                if rho is not None:
                    rho_used = rho
                    variance = 2.0 * (1.0 - rho) * (1.0 / n1 + 1.0 / n2) + (selected_g ** 2) / (2.0 * (n1 + n2))
                    if variance > 0 and math.isfinite(variance):
                        variance_method = f"Morris PPC2 approximate variance using pre/post correlation rho={rho:.4g} (metafor PPC2 implementation)"
                    else:
                        variance = None
                else:
                    raw_note = (raw_note + " " if raw_note else "") + "Hedges' g was calculable, but its Morris PPC2 sampling variance requires a pre/post correlation. MetaPy did not assume one. Supply group-specific or common pre/post correlation, or document an approved assumption and sensitivity analysis."
            if variance is None and estimand != "morris_ppc2" and design == "parallel_independent" and n1 is not None and n2 is not None:
                variance = _variance_g_ls3(selected_g, n1, n2)
                if variance is not None:
                    variance_method = "calculated from Hedges' g and group Ns (LS3/RevMan large-sample approximation)"

        if variance is not None and variance > 0:
            se = math.sqrt(variance)
            ci_low = selected_g - 1.96 * se
            ci_high = selected_g + 1.96 * se
        if selected_g is not None and variance is not None:
            status += "; variance available"
        elif selected_g is not None:
            status += "; variance unresolved"

        # If multiple duplicate source g values disagree, report this too.
        distinct_g = [x["value"] for x in reported_g_items]
        if len(distinct_g) > 1 and _source_values_disagree(reported_g_items):
            spread = max(distinct_g) - min(distinct_g)
            issue_rows.append({
                "severity": "Warning", "code": "DUPLICATE_SOURCE_G_CONFLICT",
                "finding": "Reported Hedges' g values differ across source worksheets",
                "study_id": key[0], "arm": key[1], "outcome": key[2],
                "source_locations": "; ".join(_source_loc(e) for e in entries),
                "reported_g": "; ".join(f"{x['value']:.6g} ({x['sheet']} row {x['row']})" for x in reported_g_items),
                "calculated_g": recalculated_g, "difference": spread,
                "message": f"Multiple workbook representations of Study {key[0]} report different Hedges' g values. MetaPy does not choose or average these source values. A valid independent raw-data reconstruction, when available, takes precedence; otherwise the reported effect remains unresolved until the source conflict is checked.",
            })

        locations = "; ".join(_source_loc(e) for e in entries)
        incomplete_note = raw_note
        if selected_g is None:
            incomplete_note = incomplete_note or "No reported Hedges' g, convertible Cohen's d, or supported raw/test-statistic route was found."
            issue_rows.append({
                "severity": "Warning", "code": "G_ESTIMATE_UNRESOLVED",
                "finding": "No valid Hedges' g calculation route remains",
                "study_id": key[0], "arm": key[1], "outcome": key[2],
                "source_locations": "; ".join(_source_loc(e) for e in entries),
                "reported_g": reported_g, "calculated_g": None, "difference": None,
                "message": f"For Study {key[0]} ({key[1] or 'unspecified arm'}; {key[2]}), MetaPy could not retain a reported Hedges' g or reconstruct one from the available compatible statistics. {incomplete_note}",
            })
        elif variance is None:
            issue_rows.append({
                "severity": "Warning", "code": "G_VARIANCE_UNRESOLVED",
                "finding": "Hedges' g is available, but its sampling variance is unresolved",
                "study_id": key[0], "arm": key[1], "outcome": key[2],
                "source_locations": "; ".join(_source_loc(e) for e in entries),
                "reported_g": reported_g, "calculated_g": selected_g, "difference": None,
                "message": f"For Study {key[0]} ({key[1] or 'unspecified arm'}; {key[2]}), MetaPy has an effect estimate but could not establish a valid sampling variance. {incomplete_note or 'The required design-specific uncertainty information is not available.'}",
            })

        rec = {
            "study_id": key[0], "arm": key[1], "outcome": key[2], "duration": key[3],
            "reported_hedges_g": reported_g,
            "reported_cohens_d": reported_d_items[0]["value"] if reported_d_items and reported_d_is_consistent else None,
            "independently_calculated_hedges_g": recalculated_g,
            "selected_hedges_g": selected_g,
            "difference_calculated_minus_reported": (recalculated_g - reported_g) if recalculated_g is not None and reported_g is not None else None,
            "variance": variance, "standard_error": se,
            "prepost_correlation_used": rho_used,
            "r_prepost_intervention_source": _num(entries[0]["row"].get(entries[0]["cols"].get("r1"))) if entries and entries[0]["cols"].get("r1") else None,
            "r_prepost_control_source": _num(entries[0]["row"].get(entries[0]["cols"].get("r2"))) if entries and entries[0]["cols"].get("r2") else None,
            "ci_lower_95_normal": ci_low, "ci_upper_95_normal": ci_high,
            "n_intervention": n1, "n_control": n2,
            "estimate_method": method, "variance_method": variance_method,
            "status": status, "selection_reason": selection_reason,
            "alternative_route_note": incomplete_note,
            "source_locations": locations,
            "source_value_provenance": source_value_provenance,
        }
        records.append(rec)
        calculation_log.append({
            "study_id": key[0], "arm": key[1], "outcome": key[2], "duration": key[3],
            "estimand_selected": estimand, "design_assumption": design,
            "reported_hedges_g": reported_g,
            "reported_cohens_d": reported_d_items[0]["value"] if reported_d_items and reported_d_is_consistent else None,
            "raw_route_calculated_g": recalculated_g,
            "selected_hedges_g": selected_g,
            "sampling_variance": variance, "standard_error": se,
            "prepost_correlation_used": rho_used,
            "estimate_method": method, "variance_method": variance_method,
            "sample_size_intervention": n1, "sample_size_control": n2,
            "source_locations": locations,
            "source_value_provenance": source_value_provenance,
            "selection_reason": selection_reason,
            "other_route_or_limitation": incomplete_note,
            "calculation_status": status,
        })

    result_df = pd.DataFrame(records)
    log_df = pd.DataFrame(calculation_log)
    issue_df = pd.DataFrame(issue_rows)
    total = len(records)
    estimable = int(result_df["selected_hedges_g"].notna().sum()) if total else 0
    with_variance = int(result_df["variance"].notna().sum()) if total else 0
    calculated = int(result_df["independently_calculated_hedges_g"].notna().sum()) if total else 0
    retained = int(result_df["status"].astype(str).str.startswith("reported g retained").sum()) if total else 0
    unresolved_effect = total - estimable
    unresolved_variance = total - with_variance
    summary = {
        "canonical_record_count": total,
        "independently_recalculated_count": calculated,
        "reported_g_retained_count": retained,
        "effect_estimate_available_count": estimable,
        "variance_available_count": with_variance,
        "effect_estimate_unresolved_count": unresolved_effect,
        "variance_unresolved_count": unresolved_variance,
        "recalculation_disagreement_count": int((issue_df["code"] == "G_RECALCULATION_DISAGREEMENT").sum()) if not issue_df.empty else 0,
        "duplicate_source_conflict_count": int((issue_df["code"] == "DUPLICATE_SOURCE_G_CONFLICT").sum()) if not issue_df.empty else 0,
        "estimand": estimand,
        "design_assumption": design,
        "harmonization_ready": (total > 0 and estimable == total and with_variance == total),
        "scope_note": "Internal workbook calculations only; no external source verification. Compatible records across worksheets are silently merged into one canonical record. Consistent repeated values require no researcher action; conflicting source values are flagged and excluded from calculations. Variance is estimand-specific; Morris PPC2 requires a reported or researcher-specified pre/post correlation.",
    }
    return {"records": result_df, "calculation_log": log_df, "issues": issue_df, "summary": summary}


__all__ = ["harmonize_workbook"]
