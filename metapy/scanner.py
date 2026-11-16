"""Scan a dataset and work out which meta-analyses it can support.

For every row and every effect measure the solver is run under three policies:
  * strict      – exact derivations only
  * standard    – exact + published approximations
  * permissive  – also researcher assumptions (imputed r, equal N split, SMD type)
so MetaPy can tell the researcher, per outcome, how many studies are usable
directly, how many need approximations, how many need an assumption, and how
many cannot be used without more data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .derive.registry import GROUP_INFO
from .derive.solver import add_best_g, solve
from .infer import infer_g_types
from .io import Dataset, numeric_source
from .measures import MEASURES

STRICT = {"vetoed_groups": list(GROUP_INFO)}
STANDARD: dict[str, Any] = {}
PERMISSIVE = {"assumed_r": 0.5, "equal_split": True, "smd_as_g": True}

STATUS_ORDER = ["Reported directly", "Exact calculation", "Approximation", "Needs an assumption", "Not possible"]


@dataclass
class ScanReport:
    overview: dict[str, Any]
    row_status: pd.DataFrame            # record x measure status
    plan_table: pd.DataFrame            # outcome x measure feasibility counts
    plans: list[dict[str, Any]]         # selectable analysis plans, best first
    characteristics: pd.DataFrame       # candidate moderators with their type
    extra_analyses: list[dict[str, Any]] = field(default_factory=list)
    g_inference: dict[str, Any] = field(default_factory=dict)


def _status(sol_strict, sol_std, sol_perm, m) -> str:
    fields = [m.est, m.var]
    if sol_strict.has(*fields):
        est_reported = sol_strict.values[m.est].kind == "source"
        if est_reported and sol_strict.overall_kind(fields) in ("source", "exact"):
            return "Reported directly"
        return "Exact calculation"
    if sol_std.has(*fields):
        return "Approximation"
    if sol_perm.has(*fields):
        return "Needs an assumption"
    return "Not possible"


def classify_characteristics(records: pd.DataFrame, characteristics: list[str]) -> pd.DataFrame:
    rows = []
    for c in characteristics:
        col = f"char::{c}"
        if col not in records:
            continue
        vals = records[col].dropna()
        if vals.empty:
            continue
        numeric = pd.to_numeric(vals, errors="coerce")
        n_studies = records.loc[vals.index, "study_id"].nunique()
        nuniq = vals.astype(str).nunique()
        if nuniq == 1:
            kind = "constant"
        elif nuniq == len(vals) and numeric.notna().mean() < 0.8:
            kind = "identifier"
        elif numeric.notna().mean() >= 0.8 and numeric.nunique() >= 3:
            kind = "numeric"
        elif vals.astype(str).nunique() <= max(2, int(0.6 * len(vals))) and vals.astype(str).nunique() >= 2:
            kind = "categorical"
        else:
            kind = "text"
        rows.append({"characteristic": c, "type": kind, "studies_with_value": int(n_studies),
                     "distinct_values": int(vals.astype(str).nunique()),
                     "example_values": ", ".join(map(str, vals.astype(str).unique()[:4]))})
    if "year" in records and records["year"].notna().any():
        rows.append({"characteristic": "year", "type": "numeric", "studies_with_value": int(records.loc[records["year"].notna(), "study_id"].nunique()),
                     "distinct_values": int(records["year"].nunique()), "example_values": ", ".join(map(str, records["year"].dropna().astype(int).unique()[:4]))})
    return pd.DataFrame(rows, columns=["characteristic", "type", "studies_with_value", "distinct_values", "example_values"])


def scan(ds: Dataset) -> ScanReport:
    rec = ds.records
    inf = infer_g_types(ds)
    ds.g_evidence = inf["evidence"]
    status_rows = []
    for _, r in rec.iterrows():
        src = numeric_source(r)
        inex = ds.inexact.get(r["record_id"], set())
        perm = dict(PERMISSIVE)
        gt = inf["suggested"].get(r["record_id"])
        if gt:
            perm[f"g_type_{gt}"] = True
        s1, s2, s3 = (solve(src, pol, inex) for pol in (STRICT, STANDARD, perm))
        for sol in (s1, s2, s3):
            add_best_g(sol)
        row = {"record_id": r["record_id"], "study_id": r["study_id"], "outcome": r["outcome"], "timepoint": r.get("timepoint")}
        for key, m in MEASURES.items():
            row[key] = _status(s1, s2, s3, m)
        status_rows.append(row)
    status = pd.DataFrame(status_rows)

    plan_rows = []
    for (outcome,), grp in status.groupby(["outcome"], sort=False):
        for key, m in MEASURES.items():
            counts = grp[key].value_counts()
            usable_rows = grp[grp[key] != "Not possible"]
            row = {"outcome": outcome, "measure": key, "measure_label": m.label, "family": m.family,
                   "rows": len(grp), "studies_total": grp["study_id"].nunique(),
                   "studies_usable": usable_rows["study_id"].nunique(),
                   "studies_without_assumptions": grp[grp[key].isin(STATUS_ORDER[:3])]["study_id"].nunique()}
            for s in STATUS_ORDER:
                row[s] = int(counts.get(s, 0))
            plan_rows.append(row)
    plan_table = pd.DataFrame(plan_rows)

    plans = []
    if not plan_table.empty:
        feasible = plan_table[plan_table["studies_usable"] >= 2].copy()
        # rank: most studies usable without assumptions, then exact share, then family-native measures first
        native_bonus = {"smd_g": 3, "log_or": 2, "log_rr": 2, "z_r": 2, "logit_prop": 2, "log_hr": 2, "md": 1, "ppc2": 1}
        feasible["score"] = ((feasible["Reported directly"] + feasible["Exact calculation"]) * 100 + feasible["Approximation"] * 40
                             + feasible["Needs an assumption"] * 10 + feasible["measure"].map(native_bonus).fillna(0))
        # Pre-post designs: prefer baseline-adjusted PPC2 (Morris 2008) when it covers as many studies as the endpoint SMD
        # and either the raw data support it exactly or the reported g values were shown to be PPC2.
        # When designs are mixed so that only the composite 'best available g' covers every study, recommend that.
        reasons: dict[str, str] = {}
        top = feasible["score"].max() + 1000
        for outcome in feasible["outcome"].unique():
            sub = feasible[feasible["outcome"] == outcome].set_index("measure")
            u = {k: int(sub.loc[k, "studies_usable"]) if k in sub.index else 0 for k in ("ppc2", "smd_g", "g_best")}
            ppc_exact = (sub.loc["ppc2", "Reported directly"] + sub.loc["ppc2", "Exact calculation"]) if "ppc2" in sub.index else 0
            evidence_ppc2 = inf["by_outcome"].get(outcome) == "ppc2"
            ppc2_ok = u["ppc2"] >= 2 and u["ppc2"] >= u["smd_g"] and (
                evidence_ppc2 or (ppc_exact >= 2 and ppc_exact >= 0.8 * (sub.loc["smd_g", "studies_without_assumptions"] if "smd_g" in sub.index else 0)))
            mask = feasible["outcome"] == outcome
            n_st = int(sub["studies_total"].iloc[0]) if len(sub) else 0
            baseline_norm = evidence_ppc2 or (n_st and ppc_exact >= 0.5 * n_st)
            no_conversions = ("g_best" in sub.index and "smd_g" in sub.index
                              and sub.loc["g_best", "Approximation"] <= sub.loc["smd_g", "Approximation"])
            if (u["g_best"] >= 2 and u["g_best"] > u["ppc2"] and u["g_best"] >= u["smd_g"] and baseline_norm and no_conversions):
                feasible.loc[mask & (feasible["measure"] == "g_best"), "score"] = top
                reasons[outcome] = "g_best"
            elif ppc2_ok:
                feasible.loc[mask & (feasible["measure"] == "ppc2"), "score"] = top
                reasons[outcome] = "ppc2"
            elif "g_best" in sub.index:   # composite measure is only shown when it adds something
                if u["g_best"] < max(u["ppc2"], u["smd_g"]):
                    feasible = feasible[~(mask & (feasible["measure"] == "g_best"))]
        feasible = feasible.sort_values(["outcome", "score"], ascending=[True, False])
        for outcome, grp in feasible.groupby("outcome", sort=False):
            for i, (_, p) in enumerate(grp.iterrows()):
                bits = []
                for s in STATUS_ORDER[:4]:
                    if p[s]:
                        bits.append(f"{p[s]} {s.lower()}")
                miss = p["Not possible"]
                plans.append({
                    "plan_id": f"{outcome} :: {p['measure']}",
                    "outcome": outcome, "measure": p["measure"], "measure_label": p["measure_label"],
                    "studies_usable": int(p["studies_usable"]), "studies_total": int(p["studies_total"]),
                    "recommended": i == 0,
                    "summary": f"{outcome} — {p['measure_label']}: {int(p['studies_usable'])}/{int(p['studies_total'])} studies usable"
                               f" ({', '.join(bits)}{'; ' + str(miss) + ' rows need more data' if miss else ''})"
                               f"{'  [recommended' + _why(p['measure'], reasons.get(outcome), inf['by_outcome'].get(outcome)) + ']' if i == 0 else ''}",
                })

    chars = classify_characteristics(rec, ds.characteristics)
    fam_present = []
    cols = set(rec.columns)
    if cols & {"mean_post_i", "sd_post_i", "median_post_i", "md_post", "g_post", "d_post", "t_post", "smd_unspecified"}:
        fam_present.append("continuous")
    if cols & {"mean_pre_i", "mean_change_i"}:
        fam_present.append("pre-post continuous")
    if cols & {"events_i", "pct_i", "or", "rr", "log_or"}:
        fam_present.append("binary")
    if cols & {"hr", "log_hr"}:
        fam_present.append("time-to-event")
    if cols & {"r", "r_spearman"}:
        fam_present.append("correlation")
    if cols & {"prop", "events"} and not cols & {"events_i"}:
        fam_present.append("single-group proportion")
    overview = {
        "file": ds.source_name, "worksheets": len(ds.sheets), "rows": len(rec), "studies": int(rec["study_id"].nunique()),
        "outcomes": int(rec["outcome"].nunique()), "outcome_names": list(rec["outcome"].unique()),
        "timepoints": [t for t in rec["timepoint"].dropna().unique()] if "timepoint" in rec else [],
        "data_types_detected": fam_present,
        "columns_recognised": int((ds.column_map["status"] == "Recognised").sum()),
        "columns_kept_as_characteristics": len(ds.characteristics),
        "text_cells_interpreted": len(ds.parse_notes),
        "p_values_reported_as_thresholds": sum(len(v) for v in ds.inexact.values()),
        "multi_arm_candidates": _multi_arm_count(rec),
        "studies_with_several_timepoints": int(rec.assign(_tp=rec["timepoint"].fillna("").astype(str)).groupby(["study_id", "outcome"])["_tp"].nunique().gt(1).sum()),
    }
    overview["rows_read"] = ds.merge_summary.get("rows_read", len(rec))
    overview["duplicate_rows_merged"] = ds.merge_summary.get("duplicate_rows_merged", 0)
    overview["conflicting_values"] = ds.merge_summary.get("conflicting_values", 0)
    return ScanReport(overview, status, plan_table, plans, chars, extra_analyses_for(chars, overview, rec), inf)


def _why(measure: str, reason: str | None, evidence: str | None) -> str:
    if measure == "g_best":
        return ": studies report different designs; PPC2 where possible, otherwise change or endpoint g"
    if measure == "ppc2":
        return (": your reported g values match PPC2, and it uses baseline data" if evidence == "ppc2"
                else ": uses baseline data and pre-post information")
    return ""


def _multi_arm_count(rec: pd.DataFrame) -> int:
    r = rec.assign(_tp=rec["timepoint"].fillna("").astype(str))
    if "control_id" in r and r["control_id"].notna().any():
        key = r["control_id"].fillna(r["study_id"].astype(str) + "|" + r["_tp"]).astype(str) + "|" + r["outcome"].astype(str)
        return int(key.value_counts().gt(1).sum())
    return int(r.groupby(["study_id", "outcome", "_tp"], dropna=False).size().gt(1).sum())


def usable_moderator(records: pd.DataFrame, char: str, kind: str) -> bool:
    """Offer a moderator only if it varies within at least one outcome."""
    col = f"char::{char}" if f"char::{char}" in records else char
    if col not in records:
        return False
    for _, g in records.groupby("outcome"):
        v = g[[col, "study_id"]].dropna()
        if kind == "categorical":
            counts = v.groupby(v[col].astype(str))["study_id"].nunique()
            if (counts >= 2).sum() >= 2:
                return True
        elif kind == "numeric":
            if v["study_id"].nunique() >= 6 and pd.to_numeric(v[col], errors="coerce").nunique() >= 2:
                return True
    return False


def extra_analyses_for(chars: pd.DataFrame, overview: dict[str, Any], records: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    out = [
        {"key": "loo", "label": "Leave-one-out sensitivity analysis", "needs": "3+ studies"},
        {"key": "influence", "label": "Influence diagnostics (studentised residuals, Cook's distance)", "needs": "3+ studies"},
        {"key": "exact_only", "label": "Sensitivity: only exactly-derived effect sizes", "needs": "approximations used"},
        {"key": "no_assumptions", "label": "Sensitivity: exclude effect sizes that rely on assumptions", "needs": "assumptions used"},
        {"key": "r_grid", "label": "Sensitivity: vary the imputed pre-post correlation (0.3/0.5/0.7)", "needs": "imputed r used"},
        {"key": "model_compare", "label": "Compare fixed-effect and random-effects models", "needs": "2+ studies"},
        {"key": "egger", "label": "Egger's regression test for small-study effects", "needs": "10+ studies"},
        {"key": "begg", "label": "Begg's rank correlation test", "needs": "10+ studies"},
        {"key": "trimfill", "label": "Trim-and-fill adjustment", "needs": "10+ studies recommended"},
        {"key": "cumulative", "label": "Cumulative meta-analysis (by year)", "needs": "year column"},
        {"key": "tau_ci", "label": "Confidence intervals for tau², I² and H² (Q-profile)", "needs": "2+ studies"},
        {"key": "bayes", "label": "Bayesian random-effects model (weakly informative priors)", "needs": "2+ studies"},
        {"key": "pet_peese", "label": "PET-PEESE small-study adjusted estimate", "needs": "10+ studies recommended"},
        {"key": "selection", "label": "Three-parameter selection model (3PSM) for publication bias", "needs": "20+ studies recommended"},
        {"key": "peters", "label": "Peters' test (binary outcomes)", "needs": "binary outcome, 10+ studies"},
        {"key": "failsafe", "label": "Fail-safe N (Rosenthal, Orwin; descriptive only)", "needs": "any"},
        {"key": "baujat", "label": "Baujat plot (contribution to heterogeneity vs influence)", "needs": "3+ studies"},
        {"key": "multilevel", "label": "Three-level model for several effects per study", "needs": "studies with several effects"},
        {"key": "che_rve", "label": "CHE model with robust variance estimation (CR2, small-sample corrected)", "needs": "4+ studies"},
        {"key": "tau2_compare", "label": "Heterogeneity under every τ² estimator (sensitivity)", "needs": "3+ studies"},
        {"key": "influence_full", "label": "Full influence diagnostics (DFFITS, Cook's D, covariance ratio, hat values)", "needs": "3+ studies"},
        {"key": "selection_multi", "label": "Step-function selection model with several cut-points (p = .025, .05, .5)", "needs": "30+ studies recommended"},
        {"key": "tes", "label": "Test of excess significance", "needs": "5+ studies"},
        {"key": "mh", "label": "Mantel-Haenszel common-effect estimate (binary outcomes)", "needs": "binary outcome with event counts"},
        {"key": "peto", "label": "Peto odds ratio (rare events, balanced groups)", "needs": "odds ratio with event counts"},
        {"key": "glmm", "label": "Generalized linear mixed model (exact binomial likelihood; rare events)", "needs": "odds ratio or proportion with counts"},
    ]
    for _, c in chars.iterrows():
        if records is not None and not usable_moderator(records, c["characteristic"], c["type"]):
            continue
        if c["type"] == "categorical":
            out.append({"key": f"subgroup::{c['characteristic']}", "label": f"Subgroup analysis by {c['characteristic']}", "needs": "2+ studies per subgroup"})
        if c["type"] in ("numeric", "categorical") and (records is None or usable_moderator(records, c["characteristic"], "numeric")
                                                          or (c["type"] == "categorical" and records["study_id"].nunique() >= 6)):
            out.append({"key": f"metareg::{c['characteristic']}", "label": f"Meta-regression on {c['characteristic']}", "needs": "10+ studies recommended"})
    mods = [o_["key"].split("::", 1)[1] for o_ in out if o_["key"].startswith("metareg::")]
    ctype = dict(zip(chars["characteristic"], chars["type"])) if len(chars) else {}
    numeric_first = sorted(mods, key=lambda m_: (ctype.get(m_) != "numeric", m_.lower() == "year"))
    if mods:
        mods_n = numeric_first
        cat = [m_ for m_ in mods_n if ctype.get(m_) == "categorical"]
        if records is not None:                     # fewest categories first (simplest interaction)
            nlev = lambda m_: records[f"char::{m_}" if f"char::{m_}" in records else m_].nunique() if (f"char::{m_}" in records or m_ in records) else 99
            cat = sorted(cat, key=nlev)
        pair = [mods_n[0]] + ([c for c in cat if c != mods_n[0]][:1] or [m_ for m_ in mods_n[1:2]])
        out.append({"key": "metareg_adv::" + "|".join(pair), "label": f"Meta-regression on {' × '.join(pair)}"
                    + (" with interaction" if len(pair) >= 2 else "") + ", predictions and robust (CR2) inference", "needs": "10+ studies"})
        out.append({"key": f"location_scale::{mods_n[0]}", "label": f"Location-scale model: does heterogeneity depend on {mods_n[0]}?",
                    "needs": "10+ studies"})
    if False:
        out.append({"key": "metareg_adv::" + "|".join(mods[:2]), "label": f"Meta-regression on {' × '.join(mods[:2])}"
                    + (" with interaction" if len(mods) >= 2 else "") + ", predictions and robust (CR2) inference", "needs": "10+ studies"})
        out.append({"key": f"location_scale::{mods[0]}", "label": f"Location-scale model: does heterogeneity depend on {mods[0]}?",
                    "needs": "10+ studies"})
    for _, c in chars.iterrows():
        if c["type"] == "categorical" and records is not None:
            col = f"char::{c['characteristic']}" if f"char::{c['characteristic']}" in records else c["characteristic"]
            if col in records:
                g = records[[col, "study_id"]].dropna()
                if len(g) and g.groupby("study_id")[col].nunique().max() == 1 and g[col].nunique() >= 3 and g[col].nunique() < g["study_id"].nunique():
                    out.append({"key": f"multilevel_nested::{c['characteristic']}", "label": f"Multilevel model: studies nested in {c['characteristic']}",
                                "needs": "3+ groups"})
    if len(mods) >= 2:
        out.append({"key": "metareg_multi::" + "|".join(mods[:3]), "label": f"Multiple meta-regression ({', '.join(mods[:3])}) with permutation test",
                    "needs": "10+ studies per moderator recommended"})
    return out
