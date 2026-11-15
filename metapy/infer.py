"""Evidence-based inference of what a reported Hedges' g actually is.

A column headed 'Hedges' g' may hold endpoint g, change-score g or Morris PPC2 g.
For every row that has BOTH a reported g and raw statistics, MetaPy recalculates
each design exactly and checks which one the reported value matches. Rows that
report only g then get the design that the matching rows of the same outcome
(or the whole file) show, and rows whose raw data contradict every design are
flagged. The researcher confirms or changes every suggestion.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from .derive.solver import solve
from .io import Dataset, numeric_source

DESIGNS = {"ppc2": ("g_ppc2", "Morris PPC2 (pre-post-control, baseline-adjusted)"),
           "change": ("g_change", "change-score g"),
           "post": ("g_post", "endpoint (post-test) g")}
DESIGN_LABEL = {k: v[1] for k, v in DESIGNS.items()}
STRICT = {"vetoed_groups": ["median_to_mean", "or_to_smd", "r_to_smd", "spearman", "pct_to_events", "p_inequality"]}


def _tol(x: float) -> float:
    return max(0.015, 0.03 * abs(x))


def infer_g_types(ds: Dataset) -> dict[str, Any]:
    rec = ds.records
    if "g_reported" not in rec or rec["g_reported"].isna().all():
        return {"evidence": pd.DataFrame(), "suggested": {}, "summary": [], "by_outcome": {}}
    rows = []
    for _, r in rec[rec["g_reported"].notna()].iterrows():
        src = {k: v for k, v in numeric_source(r).items() if k not in ("g_reported", "se_g_reported", "g_reported_ci_lo", "g_reported_ci_hi")}
        sol = solve(src, STRICT, ds.inexact.get(r["record_id"], set()))
        rep = float(r["g_reported"])
        cand = {k: sol.get(f) for k, (f, _) in DESIGNS.items()}
        diffs = {k: abs(v - rep) for k, v in cand.items() if v is not None}
        match = None
        if diffs:
            k_best = min(diffs, key=diffs.get)
            if diffs[k_best] <= _tol(rep):
                match = k_best
        rows.append({"record_id": r["record_id"], "study": r["study_id"] + (f" ({r['intervention_arm']})" if isinstance(r.get("intervention_arm"), str) else ""),
                     "outcome": r["outcome"], "reported g": rep,
                     "PPC2 from raw data": cand["ppc2"], "change g from raw data": cand["change"], "endpoint g from raw data": cand["post"],
                     "matches": DESIGN_LABEL.get(match, "—") if diffs else "no raw data to check",
                     "_match": match, "_checkable": bool(diffs),
                     "difference": round(diffs[match], 4) if match else (round(min(diffs.values()), 4) if diffs else None)})
    ev = pd.DataFrame(rows)
    # majority design per outcome, then overall
    by_outcome: dict[str, str | None] = {}
    overall = Counter(ev.loc[ev["_match"].notna(), "_match"])
    overall_type = None
    if overall:
        t, n = overall.most_common(1)[0]
        if n >= 2 and n / sum(overall.values()) >= 0.7:
            overall_type = t
    for out, g in ev.groupby("outcome"):
        c = Counter(g.loc[g["_match"].notna(), "_match"])
        if c:
            t, n = c.most_common(1)[0]
            by_outcome[out] = t if n / sum(c.values()) >= 0.7 else None
        else:
            by_outcome[out] = overall_type
    suggested, basis = {}, []
    for _, e in ev.iterrows():
        if isinstance(e["_match"], str):
            suggested[e["record_id"]] = e["_match"]
            basis.append("verified: matches this row's own raw data")
        elif bool(e["_checkable"]):
            suggested[e["record_id"]] = None
            basis.append("CHECK: the reported g matches no design calculable from this row's own data; not used until you decide")
        else:
            t = by_outcome.get(e["outcome"]) or overall_type
            suggested[e["record_id"]] = t
            basis.append(f"inferred from matching rows ({DESIGN_LABEL[t]})" if t else "unknown: please choose")
    ev["suggested design"] = [DESIGN_LABEL.get(suggested.get(r), "unknown") for r in ev["record_id"]]
    ev["basis"] = basis
    n_check = int(ev["_checkable"].sum())
    n_match = int(ev["_match"].map(lambda x: isinstance(x, str)).sum())
    summary = []
    if n_check:
        dom = Counter(ev.loc[ev["_match"].notna(), "_match"]).most_common(1)
        summary.append(f"Reported Hedges' g was checked against raw data in {n_check} row(s): {n_match} match exactly"
                       + (f", mostly {DESIGN_LABEL[dom[0][0]]}" if dom else "")
                       + f" (largest difference {ev.loc[ev['_match'].notna(), 'difference'].max():.3f})." if n_match else ".")
    n_only = int((~ev["_checkable"]).sum())
    if n_only:
        if overall_type or any(by_outcome.values()):
            summary.append(f"{n_only} row(s) report g without raw data; MetaPy suggests the design shown by the matching rows. Please confirm.")
        else:
            summary.append(f"{n_only} row(s) report a Hedges' g without stating its design (endpoint, change or PPC2) and there are no rows "
                           "to infer it from: you will be asked to choose in Step 2.")
    bad = ev[ev["_checkable"] & ev["_match"].isna()]
    if len(bad):
        summary.append(f"{len(bad)} row(s) have a reported g that matches none of the designs calculable from their own data "
                       f"({', '.join(f'{s} / {o}' for s, o in zip(bad['study'], bad['outcome']))}). Check how it was calculated.")
    return {"evidence": ev.drop(columns=["_match", "_checkable"]), "suggested": suggested, "summary": summary, "by_outcome": by_outcome}
