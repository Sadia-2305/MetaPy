from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any

import pandas as pd

from .schema import ALIAS_LOOKUP, as_float, canonical_study_and_arm, is_missing, normalize_header

NUMERIC_FIELDS = {
    "duration", "mean_pre_i", "sd_pre_i", "mean_post_i", "sd_post_i", "n_i",
    "mean_pre_c", "sd_pre_c", "mean_post_c", "sd_post_c", "n_c",
    "reported_effect", "effect_se", "effect_variance", "ci_lower", "ci_upper", "ci_level",
    "r_prepost_i", "r_prepost_c", "events_i", "total_i", "events_c", "total_c",
}


def _clean_scalar(value: Any, field: str) -> Any:
    if is_missing(value):
        return None
    if field in NUMERIC_FIELDS:
        return as_float(value)
    return str(value).strip()


def _duration_key(value: Any) -> str:
    x = as_float(value)
    if x is not None:
        return f"{x:g}"
    return "unspecified" if is_missing(value) else str(value).strip().casefold()


def _stable_id(key: tuple[str, str, str, str]) -> str:
    raw = "|".join(key)
    return "MP-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12].upper()


def load_excel_workbook(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("MetaPy v3.3.0 accepts .xlsx and .xlsm workbooks in this release.")
    sheets: dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(path, engine="openpyxl") as xls:
        sheet_names = list(xls.sheet_names)
    for name in sheet_names:
        df = pd.read_excel(path, sheet_name=name, engine="openpyxl")
        if not df.empty:
            df = df.dropna(how="all").copy()
        sheets[name] = df
    if not any(not frame.empty for frame in sheets.values()):
        raise ValueError("No non-empty data rows were found in the workbook.")
    return sheets


def _parse_md_ci_text(value: Any) -> tuple[float | None, float | None, float | None]:
    if is_missing(value):
        return None, None, None
    text = str(value).replace("−", "-").replace(",", "")
    # Remove interval labels before number extraction so a text prefix such as
    # "95% CI:" is not mistaken for the MD estimate.
    text = re.sub(r"95\s*%\s*(?:confidence\s+interval|CI)?", " ", text, flags=re.I)
    text = re.sub(r"\bconfidence\s+interval\b|\bCI\b", " ", text, flags=re.I)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(nums) < 3:
        return None, None, None
    # Expected source shape: estimate (lower to upper). Keep as extracted text
    # unless the researcher confirms that this is the same raw-MD estimand.
    estimate, lower, upper = map(float, nums[:3])
    if upper <= lower:
        return estimate, None, None
    return estimate, lower, upper


def _map_sheet_row(sheet: str, row_number: int, row: pd.Series) -> tuple[dict[str, Any], dict[str, list[str]], list[str]]:
    values: dict[str, Any] = {}
    provenance: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for column in row.index:
        field = ALIAS_LOOKUP.get(normalize_header(column))
        value = row[column]
        if field is None and str(column).strip().lower().startswith("unnamed") and not is_missing(value):
            field = "source_note"
        if field is None:
            if not is_missing(value):
                unmapped.append(str(column))
            continue
        clean = _clean_scalar(value, field)
        if clean is None:
            continue
        values.setdefault(field, clean)
        provenance.setdefault(field, []).append(f"{sheet}!{row_number}:{column}")

    # Some legacy extraction files use a single total_i/total_c alias.
    if values.get("n_i") is None and values.get("total_i") is not None:
        values["n_i"] = values["total_i"]
        provenance["n_i"] = provenance.get("total_i", []).copy()
    if values.get("n_c") is None and values.get("total_c") is not None:
        values["n_c"] = values["total_c"]
        provenance["n_c"] = provenance.get("total_c", []).copy()

    # Explicit arm values take priority. Otherwise a named arm embedded in the
    # supplied StudyID or outcome label is parsed, never guessed from the study.
    md_text = values.get("reported_md_ci_text")
    if md_text is not None:
        estimate, lower, upper = _parse_md_ci_text(md_text)
        if estimate is not None:
            values["reported_md_estimate"] = estimate
        if lower is not None and upper is not None:
            values["md_ci_lower"] = lower
            values["md_ci_upper"] = upper
            values["md_ci_level"] = 0.95
        provenance.setdefault("reported_md_ci_text", []).append(f"{sheet}!{row_number}:MD CI text (unparsed source field)")

    study, arm, outcome = canonical_study_and_arm(
        values.get("study_id"), values.get("intervention_arm"), values.get("outcome")
    )
    values["study_id"] = study
    values["intervention_arm"] = arm
    values["outcome"] = outcome
    return values, provenance, unmapped


def canonicalize_workbook(sheets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge repeated source records into comparison-level canonical records.

    Complementary fields are combined. Contradictory repeated fields are set to
    null and explicitly listed in the conflict register; no arbitrary source is
    silently preferred.
    """
    staged: list[dict[str, Any]] = []
    unmapped_rows: list[dict[str, Any]] = []
    for sheet, df in sheets.items():
        if df is None or df.empty:
            continue
        for idx, (_, row) in enumerate(df.iterrows(), start=2):
            vals, provenance, unmapped = _map_sheet_row(sheet, idx, row)
            if not vals.get("study_id") or not vals.get("outcome"):
                continue
            duration = vals.get("duration")
            key = (
                str(vals.get("study_id", "")).casefold(),
                str(vals.get("outcome", "")).strip().casefold(),
                _duration_key(duration),
                str(vals.get("intervention_arm", "Not specified")).strip().casefold(),
            )
            entry = {
                "key": key,
                "values": vals,
                "provenance": provenance,
                "sheet": sheet,
                "row": idx,
                "unmapped_columns": unmapped,
            }
            staged.append(entry)
            if unmapped:
                unmapped_rows.append({"sheet": sheet, "excel_row": idx, "unmapped_columns": ", ".join(unmapped)})

    # Prevent silent merging when one source worksheet contains multiple rows
    # for the same study/outcome/timepoint but no intervention-arm label.
    # Such rows are retained separately and flagged for researcher classification.
    no_arm_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for entry in staged:
        vals = entry["values"]
        arm = str(vals.get("intervention_arm", "Not specified")).strip().casefold()
        if arm in {"not specified", "", "nan"}:
            base = entry["key"][:3]
            no_arm_counts[(base[0], base[1], base[2], entry["sheet"])] += 1
    ambiguous_bases = {
        (study, outcome, duration)
        for (study, outcome, duration, sheet), count in no_arm_counts.items()
        if count > 1
    }

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for entry in staged:
        key = entry["key"]
        arm = str(entry["values"].get("intervention_arm", "Not specified")).strip().casefold()
        base = key[:3]
        if arm in {"not specified", "", "nan"} and base in ambiguous_bases:
            entry["arm_ambiguity_detected"] = True
            entry["values"]["intervention_arm"] = f"Unlabelled comparison — {entry['sheet']} row {entry['row']}"
            key = (*base, f"__unlabelled__{entry['sheet']}__{entry['row']}")
            entry["key"] = key
        else:
            entry["arm_ambiguity_detected"] = False
        grouped.setdefault(key, []).append(entry)

    canonical_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    canonical_fieldnames = sorted({field for e in staged for field in e["values"]})
    for key, entries in grouped.items():
        record: dict[str, Any] = {field: None for field in canonical_fieldnames}
        record.update({
            "record_id": _stable_id(key),
            "study_id": entries[0]["values"].get("study_id"),
            "outcome": entries[0]["values"].get("outcome"),
            "duration": entries[0]["values"].get("duration"),
            "intervention_arm": entries[0]["values"].get("intervention_arm"),
            "source_sheets": sorted({e["sheet"] for e in entries}),
            "source_row_count": len(entries),
            "arm_ambiguity_detected": any(e.get("arm_ambiguity_detected", False) for e in entries),
        })
        record_prov: dict[str, list[str]] = {}
        conflicts: list[str] = []
        for field in canonical_fieldnames:
            observations: list[tuple[Any, str]] = []
            for e in entries:
                val = e["values"].get(field)
                if val is None:
                    continue
                locations = e["provenance"].get(field, [f"{e['sheet']}!{e['row']}:{field}"])
                observations.extend((val, loc) for loc in locations)
            if not observations:
                continue
            unique: list[Any] = []
            for val, _ in observations:
                if not any(_equivalent(val, old) for old in unique):
                    unique.append(val)
            record_prov[field] = sorted({loc for _, loc in observations})
            if len(unique) == 1:
                record[field] = unique[0]
            else:
                record[field] = None
                conflicts.append(field)
                conflict_rows.append({
                    "record_id": record["record_id"],
                    "study_id": record["study_id"],
                    "outcome": record["outcome"],
                    "duration": record["duration"],
                    "intervention_arm": record["intervention_arm"],
                    "field": field,
                    "conflicting_values": json.dumps(unique, ensure_ascii=False),
                    "source_locations": "; ".join(sorted({loc for _, loc in observations})),
                    "status": "Conflict — researcher resolution required",
                })
        record["field_provenance_json"] = json.dumps(record_prov, ensure_ascii=False, sort_keys=True)
        record["conflict_fields"] = ", ".join(conflicts)
        record["has_field_conflict"] = bool(conflicts)
        canonical_rows.append(record)

    canonical = pd.DataFrame(canonical_rows)
    conflicts_df = pd.DataFrame(conflict_rows, columns=[
        "record_id", "study_id", "outcome", "duration", "intervention_arm", "field",
        "conflicting_values", "source_locations", "status"
    ])
    unmapped_df = pd.DataFrame(unmapped_rows, columns=["sheet", "excel_row", "unmapped_columns"])

    if not canonical.empty:
        canonical = canonical.sort_values(["outcome", "study_id", "duration", "intervention_arm"], kind="stable").reset_index(drop=True)
    return canonical, conflicts_df, unmapped_df


def _equivalent(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(1e-8, 1e-6 * max(abs(float(a)), abs(float(b)), 1.0))
    return str(a).strip().casefold() == str(b).strip().casefold()
