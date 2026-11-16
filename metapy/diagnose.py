"""Plain-language explanations when a data file cannot be read or analysed.

Used by the apps instead of showing a raw error. It looks at the file directly (independently of MetaPy's
reader), says what it found, what is missing, and what a usable sheet looks like. When the file does contain
everything MetaPy needs, the problem is MetaPy's, and the researcher is asked to send the file to the developer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .fields import label, match_column

USABLE_LAYOUT = ("A usable sheet has **one row per study (or per comparison)** and, as columns: a **study identifier** "
                 "(e.g. *Study*, *StudyID* or *Author*), the **outcome** (if there are several), and for each group either "
                 "**means, SDs and sample sizes** (e.g. *Mean_Post_I, SD_Post_I, N_I* and the same for the control group), "
                 "**events and totals** (binary outcomes), or a **reported effect size with its SE or 95% CI** (e.g. *Hedges' g*, "
                 "*MD (95% CI)*). The first row (or one of the first few rows) must contain the column names.")


def _family(field: str) -> str | None:
    f = field.lower()
    if f == "study_id" or f == "author":
        return "id"
    if f.startswith(("n_i", "n_c", "n_total")):
        return "n"
    if f.startswith(("mean_", "sd_", "se_mean", "median", "iqr", "range", "min_", "max_")):
        return "raw"
    if f.startswith("events"):
        return "binary"
    if any(k in f for k in ("g_", "d_", "smd", "md_", "log_", "or_", "rr_", "hr_", "cor", "fisher", "effect", "estimate", "_ci_", "_se")):
        return "effect"
    if f in ("p_value", "t_stat", "f_stat") or f.startswith(("p_", "t_")):
        return "test"
    return None


def _read_raw(path: str) -> dict[str, pd.DataFrame]:
    p = str(path).lower()
    if p.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, sheet_name=None, header=None)
    sep = "\t" if p.endswith((".tsv", ".txt")) else None
    return {"data": pd.read_csv(path, header=None, sep=sep, engine="python")}


def diagnose(path: str | Path, error: Exception | None = None) -> dict[str, Any]:
    """Return {'title', 'found', 'missing', 'advice', 'internal', 'detail'} describing why a file can't be analysed."""
    name = Path(path).name
    out: dict[str, Any] = {"title": "", "found": [], "missing": [], "advice": USABLE_LAYOUT, "internal": False,
                           "detail": f"{type(error).__name__}: {error}" if error is not None else ""}
    try:
        sheets = _read_raw(str(path))
    except Exception as exc:  # the file itself can't be opened
        out["title"] = f"MetaPy could not open `{name}`."
        out["missing"].append("The file could not be opened as a spreadsheet. It may be password-protected, damaged, or in a "
                              "different format with the wrong extension. Open it in Excel and save it again as **.xlsx** "
                              "(or as **.csv**), then upload that copy.")
        out["detail"] = out["detail"] or f"{type(exc).__name__}: {exc}"
        return out
    fams: dict[str, set[str]] = {}
    for sname, raw in sheets.items():
        raw = raw.dropna(how="all").dropna(axis=1, how="all")
        if raw.empty:
            out["found"].append(f"Sheet **{sname}**: empty.")
            continue
        best_row, best = 0, []
        for i in range(min(10, len(raw))):                       # header may not be the first row
            rec = [(c, match_column(c)) for c in raw.iloc[i].tolist() if not pd.isna(c)]
            rec = [(c, f) for c, f in rec if f]
            if len(rec) > len(best):
                best_row, best = i, rec
        n_rows = max(0, len(raw) - best_row - 1)
        if not best:
            out["found"].append(f"Sheet **{sname}**: {len(raw)} rows, but none of its column names were recognised "
                                f"(first row: {', '.join(map(str, raw.iloc[0].dropna().tolist()[:6]))}…).")
            continue
        fields = sorted({f for _, f in best})
        for f in fields:
            fam = _family(f)
            if fam:
                fams.setdefault(fam, set()).add(label(f))
        out["found"].append(f"Sheet **{sname}**: {n_rows} data rows; recognised columns: "
                            + ", ".join(f"*{c}* → {label(f)}" for c, f in best[:12]) + (" …" if len(best) > 12 else "") + ".")
    has_stats = bool(fams.get("raw") or fams.get("binary") or fams.get("effect") or fams.get("test"))
    if not fams.get("id"):
        out["missing"].append("No column identifying the studies was found (name it e.g. **Study**, **StudyID** or **Author**).")
    if not has_stats:
        out["missing"].append("No columns with statistics MetaPy can use were found: means and SDs, events, or a reported "
                              "effect size with its SE / 95% CI.")
    if fams.get("raw") and not fams.get("n"):
        out["missing"].append("Means and SDs were found, but no sample sizes (name them e.g. **N_I** and **N_C**).")
    if out["missing"]:
        out["title"] = f"MetaPy cannot analyse `{name}` yet: some information it needs is missing or not recognised."
    elif error is not None:
        out["internal"] = True
        out["title"] = (f"`{name}` contains the information MetaPy needs, but MetaPy ran into a problem while reading it. "
                        "This is a problem in MetaPy, not in your data.")
        out["advice"] = ("Please send this file (or a copy with the numbers changed) to the developer so it can be fixed. "
                         "In the meantime, check that each statistics column contains only numbers (e.g. *12.5*, not "
                         "*12.5 ± 3*, which MetaPy reads in most cases but not all) and try again.")
    else:
        out["title"] = (f"MetaPy read `{name}` but found fewer than 2 studies with usable statistics for any one outcome, "
                        "so no meta-analysis is possible yet.")
        out["advice"] = ("Check the column mapping below: every statistics column should be recognised. If a column is "
                         "listed as a study characteristic instead, rename it (e.g. *Mean post intervention* → *Mean_Post_I*). "
                         + USABLE_LAYOUT)
    return out


def diagnosis_md(d: dict[str, Any], with_detail: bool = False) -> str:
    lines = [f"**{d['title']}**"]
    if d["missing"]:
        lines.append("\n**What is missing**")
        lines += [f"- {m}" for m in d["missing"]]
    if d["found"]:
        lines.append("\n**What MetaPy found in the file**")
        lines += [f"- {f}" for f in d["found"]]
    if d.get("advice"):
        lines.append("\n**What to do**\n\n" + d["advice"])
    if with_detail and d.get("detail"):
        lines.append(f"\n*Technical detail (for the developer):* `{d['detail']}`")
    return "\n".join(lines)
