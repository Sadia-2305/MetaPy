"""Read CSV / TSV / Excel files into canonical MetaPy records.

* every worksheet is read; sheet names become the outcome when no outcome column exists
* the header row is detected automatically (title rows above the table are skipped)
* column headers are matched to canonical fields (see ``fields.py``)
* text cells are parsed: "12.3 (4.1)", "10 [8-14]", "1.2 (0.8 to 1.6)", "45%", "p<0.05"
* unrecognised columns are kept as study characteristics / candidate moderators
The source file is never modified.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fields import (SYNONYM, TEXT_FIELDS, canonical_outcome, concept_key, construct_of, match_column, norm,
                     split_instrument_arm)

NUM = r"[-+−]?\d*\.?\d+(?:[eE][-+]?\d+)?"


@dataclass
class Dataset:
    records: pd.DataFrame
    column_map: pd.DataFrame
    characteristics: list[str]
    parse_notes: pd.DataFrame
    inexact: dict[str, set[str]] = field(default_factory=dict)
    source_name: str = ""
    sheets: list[str] = field(default_factory=list)
    conflicts: pd.DataFrame = field(default_factory=pd.DataFrame)
    merge_summary: dict[str, Any] = field(default_factory=dict)
    outcome_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    fingerprint: str = ""          # SHA-256 of the exact file read


def _to_float(text: Any) -> float | None:
    if text is None:
        return None
    if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, bool):
        f = float(text)
        return f if math.isfinite(f) else None
    s = str(text).strip().replace("−", "-").replace(",", "")
    if not s or s.lower() in {"na", "n/a", "nr", "-", "--", "nan", "none", "not reported", "."}:
        return None
    s = s.rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _numbers(text: str) -> list[float]:
    t = str(text).replace("−", "-").replace("–", " to ").replace("—", " to ")
    # "8-14" style ranges: a hyphen between two numbers is a separator, not a minus sign
    t = re.sub(r"(\d)\s*-\s*(?=\d)", r"\1 to ", t)
    t = t.replace(",", " ")
    return [float(x) for x in re.findall(NUM, t)]


def _parse_p(text: Any) -> tuple[float | None, bool]:
    if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, bool):
        f = float(text)
        return (f, False) if math.isfinite(f) else (None, False)
    s = str(text).strip().lower().replace(" ", "")
    if s in {"ns", "n.s.", "nonsignificant"}:
        return None, False
    m = re.match(r"^p?([<>≤≥]=?|=)?(" + NUM + ")$", s)
    if not m:
        return _to_float(s), False
    op, val = m.group(1) or "", float(m.group(2))
    return val, op.startswith("<") or op.startswith("≤")


def _parse_direction(text: Any) -> float | None:
    f = _to_float(text)
    if f is not None:
        return 1.0 if f >= 0 else -1.0
    s = norm(text)
    if s in {"", "nan"}:
        return None
    if any(k in s for k in ("lower", "less", "decrease", "ic", "ilt", "negative", "minus")):
        return -1.0
    if any(k in s for k in ("higher", "greater", "increase", "igt", "positive", "plus", "intervention")):
        return 1.0
    return None


def read_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".csv", ".txt"):
        raw = {"Data": pd.read_csv(path, header=None, dtype=object, sep=None, engine="python", encoding_errors="replace")}
    elif ext == ".tsv":
        raw = {"Data": pd.read_csv(path, header=None, dtype=object, sep="\t", encoding_errors="replace")}
    elif ext in (".xlsx", ".xlsm", ".xls"):
        raw = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Upload a .csv, .tsv, .xlsx or .xlsm file.")
    tables = {}
    for name, df in raw.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue
        hdr = _detect_header(df)
        cols = [str(c).strip() if not pd.isna(c) else "Notes" for i, c in enumerate(df.iloc[hdr])]
        body = df.iloc[hdr + 1:].copy()
        body.columns = _dedupe(cols)
        body = body.dropna(how="all")
        # instruction / README sheets (no recognisable data column, or an instruction-like name) are not data
        if INSTRUCTION_SHEET.search(str(name)) or not any(match_column(c) for c in cols):
            continue
        if not body.empty:
            tables[str(name)] = body.reset_index(drop=True)
    if not tables:
        raise ValueError("No data table found in the file.")
    return tables


def _dedupe(cols):
    seen, out = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _detect_header(df: pd.DataFrame) -> int:
    best, best_score = 0, -1
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        matched = sum(1 for v in row if not pd.isna(v) and match_column(v))
        texty = sum(1 for v in row if not pd.isna(v) and _to_float(v) is None)
        score = matched * 3 + texty
        if score > best_score:
            best, best_score = i, score
    return best


INSTRUCTION_SHEET = re.compile(r"^\s*(read\s*me|instructions?|how to|help|guide|about|notes?|codebook|dictionary)\b", re.I)
GENERIC_SHEET = re.compile(r"^(sheet|tabelle|feuil|hoja|data|table|blad)\s*\d*$", re.I)


def _study_and_arm(raw: Any) -> tuple[str, str | None]:
    """'06-Meditation' -> ('6', 'Meditation'); 6.0 -> ('6', None); 'Smith 2019' unchanged."""
    if isinstance(raw, (int, float, np.integer, np.floating)) and not isinstance(raw, bool) and float(raw).is_integer():
        return str(int(raw)), None
    t = str(raw).strip()
    if re.fullmatch(r"0*\d+(\.0+)?", t):
        return str(int(float(t))), None
    m = re.fullmatch(r"0*(\d+)\s*[-_:/,]?\s*([A-Za-z][\w\s\-+&]*)", t)
    if m and len(m.group(2)) > 1:
        return m.group(1), m.group(2).strip()
    return t, None


def _num_or_text(v):
    try:
        f = float(v)
        return round(f, 6) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return norm(v) or None


def load_dataset(path: str | Path, group_constructs: bool = True) -> Dataset:
    path = Path(path)
    tables = read_tables(path)
    multi = len(tables) > 1
    rows: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    inexact: dict[str, set[str]] = {}
    characteristics: list[str] = []

    for sheet, df in tables.items():
        colmap: dict[str, str] = {}
        used: set[str] = set()
        for col in df.columns:
            f = match_column(col)
            if f and f not in used:
                colmap[col] = f
                used.add(f)
            maps.append({"sheet": sheet, "column": col, "mapped_to": f if f and colmap.get(col) == f else "",
                         "status": "Recognised" if colmap.get(col) else ("Duplicate of another column" if f else "Kept as study characteristic")})
        extra = [c for c in df.columns if c not in colmap]
        for c in extra:
            if c not in characteristics:
                characteristics.append(c)
        for idx, src in df.iterrows():
            rec: dict[str, Any] = {"_sheet": sheet, "_row": int(idx) + 2}
            for col, f in colmap.items():
                val = src[col]
                if pd.isna(val) if not isinstance(val, str) else not val.strip():
                    continue
                _assign(rec, f, val, notes, sheet, int(idx) + 2, col, inexact)
            for c in extra:
                v = src[c]
                if not (isinstance(v, float) and math.isnan(v)) and v is not None:
                    rec[f"char::{c}"] = v
            if "study_id" not in rec:
                if "author" in rec:
                    yr = str(int(rec["year"])) if "year" in rec else ""
                    rec["study_id"] = rec["author"] if (not yr or yr in str(rec["author"])) else f"{rec['author']} {yr}"
                else:
                    continue
            sid, arm = _study_and_arm(src[[c for c, f in colmap.items() if f == "study_id"][0]] if "study_id" in colmap.values() else rec["study_id"])
            rec["study_id"] = sid
            if arm and not rec.get("intervention_arm"):
                rec["intervention_arm"] = arm
            if rec.get("instrument"):
                inst, arm2 = split_instrument_arm(rec["instrument"])
                rec["instrument"] = inst
                if arm2 and not rec.get("intervention_arm"):
                    rec["intervention_arm"] = arm2
            if rec.get("outcome"):
                head, arm3 = split_instrument_arm(rec["outcome"])
                if arm3 and not rec.get("intervention_arm"):
                    rec["intervention_arm"] = arm3
                canon = canonical_outcome(head)
                if construct_of(head) and group_constructs:      # an instrument placed in an outcome column
                    rec.setdefault("instrument", head)
                    canon = construct_of(head)
                elif canon != head or concept_key(head) in SYNONYM:
                    rec.setdefault("instrument", head)          # keep e.g. 'hsCRP' vs 'CRP' as the assay label
                rec["outcome"] = canon
            elif rec.get("instrument"):
                c = construct_of(rec["instrument"]) if group_constructs else None
                rec["outcome"] = c or canonical_outcome(rec["instrument"])
            else:
                rec["outcome"] = sheet if (multi and not GENERIC_SHEET.match(sheet)) else "Primary outcome"
            rows.append(rec)
    if not rows:
        raise ValueError("No usable rows: MetaPy needs at least a study identifier column (e.g. 'Study', 'StudyID' or 'Author').")
    rows, conflicts, merge_summary = _merge_duplicates(rows, inexact)
    records = pd.DataFrame(rows)
    for col in ("instrument", "timepoint", "intervention_arm", "control_arm", "control_id", "unit", "study_design", "outcome_direction", "timepoint_unit"):
        if col not in records:
            records[col] = None
    _add_followup_moderator(records, characteristics)
    records["study_id"] = records["study_id"].astype(str).str.strip()
    records["outcome"] = records["outcome"].astype(str).str.strip()
    records.insert(0, "record_id", [f"R{i + 1:03d}" for i in range(len(records))])
    # re-key inexact by record id
    inexact_by_id: dict[str, set[str]] = {}
    for rid, (sheet, row) in zip(records["record_id"], zip(records["_sheet"], records["_row"])):
        k = f"{sheet}:{row}"
        if k in inexact:
            inexact_by_id[rid] = inexact[k]
    if records["instrument"].notna().any():
        records["char::Scale / instrument"] = records["instrument"]
        if "Scale / instrument" not in characteristics:
            characteristics.insert(0, "Scale / instrument")
    omap = (records.assign(instrument=records["instrument"].fillna("—"))
            .groupby("outcome").agg(scales=("instrument", lambda x: ", ".join(sorted(set(map(str, x))))),
                                    rows=("record_id", "size"), studies=("study_id", "nunique")).reset_index())
    import hashlib
    fp = hashlib.sha256(path.read_bytes()).hexdigest()
    return Dataset(fingerprint=fp, records=records, column_map=pd.DataFrame(maps), characteristics=characteristics,
                   parse_notes=pd.DataFrame(notes, columns=["sheet", "row", "column", "original", "interpreted_as"]),
                   inexact=inexact_by_id, source_name=path.name, sheets=list(tables),
                   conflicts=pd.DataFrame(conflicts, columns=["study", "outcome", "field", "kept value", "kept from", "other value", "other source"]),
                   merge_summary=merge_summary, outcome_map=omap)


def _merge_duplicates(rows: list[dict], inexact: dict) -> tuple[list[dict], list[dict], dict]:
    """Rows describing the same comparison (study, outcome, time point, arm) in several sheets are merged.

    Complementary values are combined; agreeing duplicates are kept once; disagreeing values keep the
    first (the row with more raw statistics) and are listed as conflicts for the researcher.
    """
    def key(r):
        tp = _num_or_text(r.get("timepoint"))
        arm = concept_key(r.get("intervention_arm")) if r.get("intervention_arm") else ""
        return (str(r["study_id"]), concept_key(r["outcome"]), tp, arm)

    def richness(r):
        return sum(1 for k, v in r.items() if not k.startswith(("_", "char::")) and isinstance(v, (int, float)))

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    merged, conflicts = [], []
    n_dup = 0
    for k, grp in groups.items():
        grp = sorted(grp, key=richness, reverse=True)
        base = dict(grp[0])
        srcs = [f"{grp[0]['_sheet']} row {grp[0]['_row']}"]
        base_inexact = set(inexact.get(f"{grp[0]['_sheet']}:{grp[0]['_row']}", set()))
        for other in grp[1:]:
            n_dup += 1
            src = f"{other['_sheet']} row {other['_row']}"
            srcs.append(src)
            base_inexact |= inexact.get(f"{other['_sheet']}:{other['_row']}", set())
            for f, v in other.items():
                if f.startswith("_"):
                    continue
                if f not in base or base[f] is None or (isinstance(base[f], float) and math.isnan(base[f])):
                    base[f] = v
                elif isinstance(v, (int, float)) and isinstance(base[f], (int, float)) and not isinstance(v, bool):
                    a, b = float(base[f]), float(v)
                    if abs(a - b) > max(1e-4, 1e-3 * max(abs(a), abs(b))):
                        conflicts.append({"study": base["study_id"], "outcome": base["outcome"], "field": f, "kept value": a,
                                          "kept from": srcs[0], "other value": b, "other source": src})
        base["_sources"] = "; ".join(srcs)
        if base_inexact:
            inexact[f"{base['_sheet']}:{base['_row']}"] = base_inexact
        merged.append(base)
    summary = {"rows_read": len(rows), "comparisons_after_merge": len(merged), "duplicate_rows_merged": n_dup,
               "conflicting_values": len(conflicts)}
    return merged, conflicts, summary


WEEKS = {"day": 1 / 7, "days": 1 / 7, "d": 1 / 7, "week": 1, "weeks": 1, "wk": 1, "wks": 1, "w": 1,
         "month": 4.345, "months": 4.345, "mo": 4.345, "m": 4.345, "year": 52.18, "years": 52.18, "yr": 52.18, "y": 52.18}


def _add_followup_moderator(records: pd.DataFrame, characteristics: list[str]) -> None:
    """Turn a numeric time point / duration into a numeric study characteristic (usable in meta-regression)."""
    tp = records["timepoint"]
    vals = []
    units = set()
    for t, u in zip(tp, records["timepoint_unit"]):
        if t is None or (isinstance(t, float) and math.isnan(t)):
            vals.append(None)
            continue
        txt = f"{t} {u if isinstance(u, str) else ''}".strip().lower()
        nums = _numbers(txt)
        unit = next((w for w in re.findall(r"[a-z]+", txt) if w in WEEKS), None)
        units.add(unit)
        vals.append(nums[0] * WEEKS[unit] if nums and unit else (nums[0] if nums else None))
    if sum(v is not None for v in vals) >= 2 and len({v for v in vals if v is not None}) >= 2:
        name = "Follow-up (weeks)" if units - {None} else "Follow-up"
        records[f"char::{name}"] = vals
        if name not in characteristics:
            characteristics.append(name)
    # readable time-point label, e.g. "8 weeks"
    records["timepoint"] = [None if t is None or (isinstance(t, float) and math.isnan(t)) else
                            (f"{t} {u}" if isinstance(u, str) and u and not re.search(r"[a-z]", str(t).lower()) else str(t))
                            for t, u in zip(tp, records["timepoint_unit"])]


def direction_from_text(values) -> bool | None:
    """Higher-is-better (True) / higher-is-worse (False) from an outcome-direction column, if consistent."""
    verdicts = set()
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        t = norm(v)
        if not t:
            continue
        if any(k in t for k in ("lowerbetter", "lowerisbetter", "higherworse", "higherisworse", "highermore", "higherharm",
                                "highersevere", "decreasebenefit", "lowerimprove", "negative")):
            verdicts.add(False)
        elif any(k in t for k in ("higherbetter", "higherisbetter", "lowerworse", "higherimprove", "increasebenefit", "positive")):
            verdicts.add(True)
        elif t in ("yes", "true", "1"):
            verdicts.add(True)
        elif t in ("no", "false", "0"):
            verdicts.add(False)
    return verdicts.pop() if len(verdicts) == 1 else None


def _note(notes, sheet, row, col, orig, interp):
    notes.append({"sheet": sheet, "row": row, "column": col, "original": str(orig), "interpreted_as": interp})


def _assign(rec, f, val, notes, sheet, row, col, inexact):
    if f in ("study_id", "author", "outcome", "instrument", "timepoint", "intervention_arm", "control_arm", "effect_type",
             "timepoint_unit", "control_id", "outcome_direction", "unit", "study_design"):
        rec[f] = str(val).strip() if not isinstance(val, float) else (str(int(val)) if float(val).is_integer() else str(val))
        return
    if f == "direction_sign":
        d = _parse_direction(val)
        if d is not None:
            rec[f] = d
            _note(notes, sheet, row, col, val, f"direction = {int(d):+d}")
        return
    if f.startswith("p_"):
        p, ineq = _parse_p(val)
        if p is not None:
            rec[f] = p
            if ineq:
                inexact.setdefault(f"{sheet}:{row}", set()).add(f)
                _note(notes, sheet, row, col, val, f"{f} = {p} (threshold, not exact)")
        return
    if f.startswith("text_"):
        parts = f.split("_")
        arm, time = parts[-1], parts[-2]
        kind = "_".join(parts[1:-2])
        nums = _numbers(val)
        if kind == "mean_sd" and len(nums) >= 2:
            rec[f"mean_{time}_{arm}"], rec[f"sd_{time}_{arm}"] = nums[0], nums[1]
            _note(notes, sheet, row, col, val, f"mean = {nums[0]}, SD = {nums[1]}")
        elif kind == "median_iqr" and len(nums) >= 3:
            rec[f"median_{time}_{arm}"], rec[f"q1_{time}_{arm}"], rec[f"q3_{time}_{arm}"] = nums[:3]
            _note(notes, sheet, row, col, val, f"median = {nums[0]}, Q1 = {nums[1]}, Q3 = {nums[2]}")
        elif kind == "median_range" and len(nums) >= 3:
            rec[f"median_{time}_{arm}"], rec[f"min_{time}_{arm}"], rec[f"max_{time}_{arm}"] = nums[:3]
            _note(notes, sheet, row, col, val, f"median = {nums[0]}, min = {nums[1]}, max = {nums[2]}")
        return
    if f.endswith("ci_text"):
        nums = _numbers(val)
        prefix = {"md_ci_text": ("md_post", "md_ci_lo_post", "md_ci_hi_post"), "g_ci_text": ("g_post", "g_ci_lo_post", "g_ci_hi_post"),
                  "d_ci_text": ("d_post", "d_ci_lo_post", "d_ci_hi_post"), "or_ci_text": ("or", "or_ci_lo", "or_ci_hi"),
                  "rr_ci_text": ("rr", "rr_ci_lo", "rr_ci_hi"), "hr_ci_text": ("hr", "hr_ci_lo", "hr_ci_hi"),
                  "ci_text": ("effect", "effect_ci_lo", "effect_ci_hi")}[f]
        if len(nums) >= 3:
            for k, x in zip(prefix, nums[:3]):
                rec.setdefault(k, x)
            _note(notes, sheet, row, col, val, f"estimate = {nums[0]}, 95% CI {nums[1]} to {nums[2]}")
        elif len(nums) == 2:
            rec.setdefault(prefix[1], nums[0])
            rec.setdefault(prefix[2], nums[1])
            _note(notes, sheet, row, col, val, f"95% CI {nums[0]} to {nums[1]}")
        return
    x = _to_float(val)
    if x is not None:
        rec[f] = x
        if isinstance(val, str) and val.strip().endswith("%"):
            _note(notes, sheet, row, col, val, f"{x} (percent)")
        return
    nums = _numbers(val)
    if f.startswith("mean_") and len(nums) == 2:
        parts = f.split("_")
        rec[f], rec[f"sd_{parts[1]}_{parts[2]}"] = nums
        _note(notes, sheet, row, col, val, f"mean = {nums[0]}, SD = {nums[1]} (\"mean (SD)\" text)")
    elif f in ("or", "rr", "hr", "md_post", "g_post", "d_post", "effect") and len(nums) == 3:
        lo, hi = {"or": ("or_ci_lo", "or_ci_hi"), "rr": ("rr_ci_lo", "rr_ci_hi"), "hr": ("hr_ci_lo", "hr_ci_hi"),
                  "md_post": ("md_ci_lo_post", "md_ci_hi_post"), "g_post": ("g_ci_lo_post", "g_ci_hi_post"),
                  "d_post": ("d_ci_lo_post", "d_ci_hi_post"), "effect": ("effect_ci_lo", "effect_ci_hi")}[f]
        rec[f], rec[lo], rec[hi] = nums
        _note(notes, sheet, row, col, val, f"estimate = {nums[0]}, 95% CI {nums[1]} to {nums[2]}")
    elif len(nums) == 1:
        rec[f] = nums[0]
        _note(notes, sheet, row, col, val, f"{nums[0]}")
    else:
        _note(notes, sheet, row, col, val, "could not be read as a number; ignored")


def numeric_source(record: pd.Series | dict) -> dict[str, float]:
    """Numeric canonical values of a record (inputs for the solver)."""
    out = {}
    for k, v in dict(record).items():
        if k.startswith("_") or k.startswith("char::") or k in TEXT_FIELDS or k == "record_id":
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[k] = float(v)
    return out
