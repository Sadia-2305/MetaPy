"""Data-preparation phase: derive what each chosen analysis needs, transparently.

Produces
  * ``data``      – one row per comparison with yi / vi and how they were obtained
  * ``log``       – every calculated value, its formula, inputs and citation
  * ``actions``   – plain-language summary of what MetaPy calculated
  * ``questions`` – decisions MetaPy needs, each with a list of options
  * ``requests``  – rows that cannot be derived and exactly what to extract
  * ``checks``    – reported values that disagree with their own raw data
Nothing is written back to the source file; nothing is invented.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .derive.registry import GROUP_INFO
from .derive.solver import Solution, add_best_g, missing_recipes, solve
from .infer import DESIGN_LABEL
from .fields import label
from .io import Dataset, _numbers, direction_from_text, numeric_source
from .measures import MEASURES, Measure

DEFAULT_SETTINGS: dict[str, Any] = {
    "vetoed_groups": [],        # approximation / assumption groups the researcher rejects
    "assumed_r": None,          # imputed pre-post correlation (assumption)
    "equal_split": False,       # split total N equally (assumption)
    "smd_type": None,           # "g" | "d" | "exclude" for SMDs of unstated type
    "zero_cell": True,          # 0.5 continuity correction
    "multi_arm": "split",       # split | combine | keep | first
    "consistency": "reported",  # reported | recalculated
    "higher_is_better": {},     # {outcome: True/False}
    "harmonise_direction": True,
    "excluded_records": [],
    "timepoint_policy": "latest",
    "g_types": {},              # {record_id: 'ppc2' | 'change' | 'post' | None}: design of a reported Hedges' g
    "ppc2_var_simple": False,   # approximate PPC2 variance without a pre-post correlation
    "best_policy": "design",    # composite g: 'design' (PPC2 > change > endpoint) or 'assumptions' (fewest assumptions)   # latest | earliest | all  (studies reporting several time points)
    "manual_values": {},        # {record_id: {field: value}}
}


@dataclass
class PrepResult:
    plan: dict[str, Any]
    measure: Measure
    data: pd.DataFrame
    log: pd.DataFrame
    actions: pd.DataFrame
    questions: list[dict[str, Any]]
    requests: pd.DataFrame
    checks: pd.DataFrame
    counts: dict[str, int]
    settings: dict[str, Any]
    solutions: dict[str, Solution] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def solver_settings(s: dict[str, Any]) -> dict[str, Any]:
    out = {"vetoed_groups": list(s.get("vetoed_groups", [])), "zero_cell": s.get("zero_cell", True)}
    if s.get("assumed_r") is not None:
        out["assumed_r"] = float(s["assumed_r"])
    if s.get("equal_split"):
        out["equal_split"] = True
    if s.get("ppc2_var_simple"):
        out["ppc2_var_simple"] = True
    if s.get("smd_type") == "g":
        out["smd_as_g"] = True
    elif s.get("smd_type") == "d":
        out["smd_as_d"] = True
    return out


def _tp_value(t) -> float:
    nums = _numbers(t) if t is not None and not (isinstance(t, float) and math.isnan(t)) else []
    return nums[0] if nums else float("nan")


def row_settings(base: dict[str, Any], s: dict[str, Any], rid: str) -> dict[str, Any]:
    t = (s.get("g_types") or {}).get(rid)
    return {**base, f"g_type_{t}": True} if t in ("ppc2", "change", "post") else base


def _label(r: pd.Series, multi_time: bool, multi_arm: bool) -> str:
    lab = str(r["study_id"])
    if multi_arm and isinstance(r.get("intervention_arm"), str) and r.get("intervention_arm"):
        lab += f" ({r['intervention_arm']})"
    if multi_time and r.get("timepoint") not in (None, "") and not pd.isna(r.get("timepoint")):
        lab += f" [{r['timepoint']}]"
    return lab


def _combine_continuous(rows: list[dict], t: str = "post") -> dict | None:
    need = [f"mean_{t}_i", f"sd_{t}_i", "n_i"]
    if not all(all(k in r and not pd.isna(r[k]) for k in need) for r in rows):
        return None
    n = sum(r["n_i"] for r in rows)
    m = sum(r["n_i"] * r[f"mean_{t}_i"] for r in rows) / n
    # Cochrane Handbook 6.5.2.10, applied sequentially
    acc_n, acc_m, acc_sd = rows[0]["n_i"], rows[0][f"mean_{t}_i"], rows[0][f"sd_{t}_i"]
    for r in rows[1:]:
        n2, m2, sd2 = r["n_i"], r[f"mean_{t}_i"], r[f"sd_{t}_i"]
        tot = acc_n + n2
        sd = math.sqrt(((acc_n - 1) * acc_sd ** 2 + (n2 - 1) * sd2 ** 2 + acc_n * n2 / tot * (acc_m - m2) ** 2) / (tot - 1))
        acc_m = (acc_n * acc_m + n2 * m2) / tot
        acc_n, acc_sd = tot, sd
    return {"n_i": n, f"mean_{t}_i": m, f"sd_{t}_i": acc_sd}


def prepare(ds: Dataset, plan: dict[str, Any], settings: dict[str, Any] | None = None) -> PrepResult:
    s = copy.deepcopy(DEFAULT_SETTINGS)
    s.update(settings or {})
    m = MEASURES[plan["measure"]]
    recs = ds.records[ds.records["outcome"] == plan["outcome"]].copy()
    ssets = solver_settings(s)
    warnings: list[str] = []
    tp = recs["timepoint"].fillna("").astype(str)
    # ---- studies reporting the outcome at several time points (dependent effects)
    tp_counts = recs.assign(_tp=tp).groupby("study_id")["_tp"].nunique()
    several_tp = set(tp_counts[tp_counts > 1].index)
    multi_time = bool(several_tp)
    tp_dropped: set[str] = set()
    if several_tp and s.get("timepoint_policy", "latest") in ("latest", "earliest"):
        for sid in several_tp:
            g = recs[recs["study_id"] == sid]
            vals = g["timepoint"].map(_tp_value)
            if vals.notna().any():
                target = vals.max() if s["timepoint_policy"] == "latest" else vals.min()
                tp_dropped |= set(g.loc[vals != target, "record_id"])
    if "control_id" in recs and recs["control_id"].notna().any():
        group_key = recs["control_id"].fillna(recs["study_id"].astype(str) + "||" + tp).astype(str)
    else:
        group_key = recs["study_id"].fillna("").astype(str) + "||" + tp
    group_key = group_key[~recs["record_id"].isin(tp_dropped)]
    group_size = group_key.map(group_key.value_counts())
    multi_arm_any = bool((group_size > 1).any())
    # ---- outcome direction from the data file (used as default; researcher can change it)
    hib_map = dict(s.get("higher_is_better") or {})
    from .fields import known_direction
    d_known = known_direction(plan["outcome"])
    chosen0 = hib_map.get(plan["outcome"], "unset")
    if d_known is not None and chosen0 not in ("unset", d_known) and not (
            "outcome_direction" in recs and direction_from_text(recs["outcome_direction"].tolist()) is not None):
        what = "'leave signs as they are'" if chosen0 is None else ("higher is better" if chosen0 else "higher is worse")
        warnings.append(f"Direction check: for {plan['outcome']} higher values usually mean a worse state, but the answer chosen was {what}. "
                        "The 'Favours intervention / control' plot labels follow your answer: please make sure it is right."
                        if d_known is False else
                        f"Direction check: for {plan['outcome']} higher values usually mean a better state, but the answer chosen was {what}.")
    if "outcome_direction" in recs:
        d_file = direction_from_text(recs["outcome_direction"].tolist())
        chosen = hib_map.get(plan["outcome"], "unset")
        if d_file is not None and chosen != "unset" and chosen != d_file:
            what = "'leave signs as they are'" if chosen is None else ("higher is better" if chosen else "higher is worse")
            warnings.append(f"Direction conflict: your data file says higher scores are {'better' if d_file else 'worse'} for this outcome, "
                            f"but the answer chosen was {what}. The 'Favours intervention / control' labels in the plots follow your answer, "
                            "so please make sure it is right.")
    if plan["outcome"] not in hib_map and "outcome_direction" in recs:
        d = direction_from_text(recs["outcome_direction"].tolist())
        if d is not None:
            hib_map[plan["outcome"]] = d
            s["higher_is_better"] = hib_map
            s.setdefault("_direction_from_file", {})[plan["outcome"]] = True
            vals = ", ".join(sorted(set(map(str, recs["outcome_direction"].dropna()))))
            warnings.append(f"Outcome direction taken from the data file ('{vals}'): higher scores are "
                            f"{'better' if d else 'worse'}" + ("." if d else ", so signs were reversed and positive effects favour the intervention."))
    # ---- units and design checks
    if "unit" in recs:
        units = sorted(set(str(u).strip() for u in recs["unit"].dropna()))
        if len(units) > 1 and m.scale == "Raw units":
            warnings.append(f"Studies use different units ({', '.join(units)}): a raw mean difference is not valid. Choose a standardised measure (SMD) instead.")
        elif len(units) > 1:
            warnings.append(f"Studies use different units ({', '.join(units)}); fine for a standardised measure, but check they measure the same construct.")
    if "study_design" in recs:
        bad = recs[recs["study_design"].astype(str).str.contains("cross|cluster", case=False, na=False)]
        if len(bad):
            warnings.append(f"{len(bad)} row(s) are crossover or cluster designs ({', '.join(bad['study_id'].astype(str))}). "
                            "MetaPy treats every study as parallel-group with individual randomisation; adjust or exclude these.")

    sources: dict[str, dict[str, float]] = {}
    extra_notes: dict[str, list[str]] = {}
    manual = s.get("manual_values", {}) or {}
    for _, r in recs.iterrows():
        src = numeric_source(r)
        for f, val in (manual.get(r["record_id"], {}) or {}).items():
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                src[f] = float(val)
                extra_notes.setdefault(r["record_id"], []).append(f"{label(f)} = {val} added by researcher")
        sources[r["record_id"]] = src

    # ---- multi-arm handling
    multi_rows = []
    dropped_multi: set[str] = set()
    split_for_variance: dict[str, int] = {}
    for key, grp in recs.loc[group_key.index].groupby(group_key, sort=False):
        if len(grp) < 2:
            continue
        ids = list(grp["record_id"])
        multi_rows.append({"study": grp["study_id"].iloc[0], "timepoint": grp["timepoint"].iloc[0], "records": ids,
                           "arms": [str(a) for a in grp["intervention_arm"].fillna("unlabelled arm")]})
        policy = s.get("multi_arm", "split")
        if policy == "split":
            # Cochrane 23.3.4: the split control N is used ONLY for the sampling variance (weights);
            # the effect estimate keeps the real control-group statistics.
            k = len(ids)
            for rid in ids:
                split_for_variance[rid] = k
                extra_notes.setdefault(rid, []).append(f"shared control: its N was divided by {k} for the variance only (effect estimate unchanged)")
        elif policy == "combine":
            rows = [sources[i] for i in ids]
            combined = {}
            for t in ("post", "pre", "change"):
                c = _combine_continuous(rows, t)
                if c:
                    combined.update(c)
            if all("events_i" in r and "n_i" in r for r in rows):
                combined.update({"events_i": sum(r["events_i"] for r in rows), "n_i": sum(r["n_i"] for r in rows)})
            if combined:
                keep = dict(rows[0])
                for f in list(keep):
                    if f.endswith("_i") or f in ("n_total",) or f.startswith(("md_", "g_", "d_", "t_", "p_", "se_", "var_", "or", "rr", "log_")):
                        keep.pop(f, None)
                keep.update(combined)
                sources[ids[0]] = keep
                extra_notes.setdefault(ids[0], []).append(f"{len(ids)} intervention arms combined (Cochrane Handbook 6.5.2.10)")
                dropped_multi.update(ids[1:])
            else:
                k = len(ids)
                for rid in ids:
                    split_for_variance[rid] = k
                    extra_notes.setdefault(rid, []).append(f"arms could not be combined (group-level data missing); shared-control N divided by {k} for the variance only")
        elif policy == "first":
            dropped_multi.update(ids[1:])

    rows_out, log_rows, req_rows, check_rows = [], [], [], []
    sols: dict[str, Solution] = {}
    excluded = set(s.get("excluded_records", []))
    hib = s.get("higher_is_better", {}) or {}
    for _, r in recs.iterrows():
        rid = r["record_id"]
        src = sources[rid]
        inex = ds.inexact.get(rid, set())
        rs = row_settings(ssets, s, rid)
        sol = solve(src, rs, inex)
        if s.get("consistency") == "recalculated" and sol.checks:
            src2 = {k: v for k, v in src.items() if k not in {c["field"] for c in sol.checks}}
            sol2 = solve(src2, rs, inex)
            if sol2.has(m.est, m.var):
                extra_notes.setdefault(rid, []).append("reported value(s) that disagreed with raw data were replaced by recalculation")
                sol2.checks = sol.checks
                sol = sol2
        design = None
        if m.key == "g_best":
            design = add_best_g(sol, s.get("best_policy", "design"))
        elif m.key in ("ppc2", "smd_g", "smd_change_g"):
            design = {"ppc2": "Morris PPC2", "smd_g": "endpoint g", "smd_change_g": "change-score g"}[m.key]
        sols[rid] = sol
        for c in sol.checks:
            check_rows.append({"record_id": rid, "study": r["study_id"], **c})
        y, v = sol.get(m.est), sol.get(m.var)
        if rid in split_for_variance and y is not None:
            k = split_for_variance[rid]
            src_v = dict(src)
            for f in ("n_c", "events_c"):
                if f in src_v:
                    src_v[f] = src_v[f] / k
            # pin the effect estimates themselves (they must not change when only the weight is adjusted)
            for f in ("g_post", "d_post", "g_change", "d_change", "g_ppc2", "md_post", "md_change"):
                if sol.get(f) is not None:
                    src_v[f] = sol.get(f)
            if "n_total" in src_v and "n_c" in src:
                src_v["n_total"] = src_v["n_total"] - src["n_c"] + src_v["n_c"]
            sol_v = solve(src_v, rs, inex)
            if m.key == "g_best":
                add_best_g(sol_v, s.get("best_policy", "design"))
            v_split = sol_v.get(m.var)
            if v_split is not None and (v is None or v_split >= v):
                v = v_split
        flip = 1.0
        hb = hib.get(plan["outcome"])
        if s.get("harmonise_direction") and hb is False and m.family in ("continuous",):
            flip = -1.0
        row = {"record_id": rid, "study_id": r["study_id"], "label": _label(r, multi_time, multi_arm_any),
               "outcome": r["outcome"], "timepoint": r.get("timepoint"), "intervention_arm": r.get("intervention_arm"),
               "year": r.get("year"), "n_i": sol.get("n_i"), "n_c": sol.get("n_c"), "n_total": sol.get("n_total") or sol.get("n_i"),
               "yi": None if y is None else y * flip, "vi": v, "direction_flipped": flip < 0,
               "Effect-size design": design if (y is not None and v is not None) else None,
               "reported g used as": DESIGN_LABEL.get((s.get("g_types") or {}).get(rid)) if "g_reported" in src else None}
        for c in ds.characteristics:
            row[c] = r.get(f"char::{c}")
        basis = "—"
        if y is not None and v is not None:
            kind = sol.overall_kind([m.est, m.var])
            basis = {"source": "Reported directly", "exact": "Exact calculation", "approximation": "Approximation",
                     "assumption": "Assumption"}[kind]
            groups = sorted(sol.groups_used([m.est, m.var]))
            row["derivation_basis"] = basis
            row["approximations_or_assumptions"] = "; ".join(GROUP_INFO[g][0] for g in groups if g in GROUP_INFO)
            chain = sol.chain([m.est, m.var])
            row["derivation_steps"] = " -> ".join(step["rule"] for step in chain) if chain else "Values used as reported"
            est_chain = [st["rule"] for st in sol.chain([m.est])]
            row["estimate from"] = ("reported g (design confirmed)" if any(r_.startswith("Reported g used") for r_ in est_chain)
                                    else "raw data" if est_chain else "reported value")
            for step in chain:
                log_rows.append({"record_id": rid, "study": row["label"], "calculated": label(step["field"]), "value": step["value"],
                                 "type": step["kind"], "rule": step["rule"], "formula": step["formula"],
                                 "inputs": "; ".join(f"{label(k)} = {val:.4g}" for k, val in step["input_values"].items()),
                                 "citation": step["citation"], "note": step["note"]})
        else:
            row["derivation_basis"] = "Not derivable"
            row["approximations_or_assumptions"] = ""
            row["derivation_steps"] = ""
            recipes = missing_recipes(sol, [m.est, m.var], rs)
            if "g_reported" in src and not (s.get("g_types") or {}).get(rid) and m.key in ("ppc2", "smd_g", "smd_change_g", "g_best"):
                recipes = [["__confirm_g_design__"]] + recipes
            req_rows.append({"record_id": rid, "study": row["label"], "outcome": r["outcome"],
                             "have": ", ".join(label(k) for k, vv in sol.values.items() if vv.kind == "source"),
                             "option_1": (" + ".join(label(x) for x in recipes[0]) if recipes[0] != ["__confirm_g_design__"] else
                                          "Confirm the design of the reported Hedges' g (table 'Reported Hedges' g in this plan' in Step 2)") if recipes else "Re-check the paper",
                             "option_2": " + ".join(label(x) for x in recipes[1]) if len(recipes) > 1 else "",
                             "option_3": " + ".join(label(x) for x in recipes[2]) if len(recipes) > 2 else "",
                             "fields_option_1": ",".join(recipes[0]) if recipes and recipes[0] != ["__confirm_g_design__"] else
                                                (",".join(recipes[1]) if len(recipes) > 1 else "")})
        row["notes"] = "; ".join(extra_notes.get(rid, []))
        reasons = []
        if rid in excluded:
            reasons.append("Excluded by researcher")
        if rid in dropped_multi:
            reasons.append("Merged into another arm / not selected (multi-arm handling)")
        if rid in tp_dropped:
            reasons.append(f"Other time point used for this study ({s.get('timepoint_policy')} follow-up)")
        if row["yi"] is None or row["vi"] is None or not (row["vi"] or 0) > 0:
            reasons.append("Effect size or variance not derivable from the data")
        row["included"] = not reasons
        row["exclusion_reason"] = "; ".join(reasons)
        if row["yi"] is not None and row["vi"]:
            se = math.sqrt(row["vi"])
            row["se"] = se
            row["ci_lo"], row["ci_hi"] = row["yi"] - 1.959964 * se, row["yi"] + 1.959964 * se
        rows_out.append(row)

    data = pd.DataFrame(rows_out)
    for col in ("se", "ci_lo", "ci_hi", "estimate from", "derivation_steps", "approximations_or_assumptions"):
        if col not in data:
            data[col] = None
    log = pd.DataFrame(log_rows, columns=["record_id", "study", "calculated", "value", "type", "rule", "formula", "inputs", "citation", "note"])
    actions = _actions(log, data)
    questions = _questions(ds, recs, data, sols, m, s, multi_rows, check_rows, plan)
    counts = {
        "rows": len(data), "included": int(data["included"].sum()) if len(data) else 0,
        "reported_directly": int((data["derivation_basis"] == "Reported directly").sum()),
        "exact": int((data["derivation_basis"] == "Exact calculation").sum()),
        "approximation": int((data["derivation_basis"] == "Approximation").sum()),
        "assumption": int((data["derivation_basis"] == "Assumption").sum()),
        "not_derivable": int((data["derivation_basis"] == "Not derivable").sum()),
        "consistency_warnings": len(check_rows),
    }
    if m.key == "g_best" and not data.empty:
        mix = data.loc[data["included"], "Effect-size design"].value_counts()
        if len(mix) > 1:
            warnings.append("Designs are mixed in this outcome (" + ", ".join(f"{k}: {v}" for k, v in mix.items()) +
                            "). Cochrane Handbook 10.5.2 advises caution when mixing change-based and endpoint SMDs; "
                            "add the subgroup analysis by 'Effect-size design' and the single-design sensitivity analysis.")
    if "instrument" in recs and recs["instrument"].nunique() > 1:
        insts = sorted(set(map(str, recs["instrument"].dropna())))
        if m.scale == "Raw units":
            warnings.append(f"Different scales/instruments ({', '.join(insts)}): a raw mean difference is not valid across scales. Use an SMD.")
        else:
            warnings.append(f"Scales/instruments combined in this outcome: {', '.join(insts)} (standardised measure, so this is allowed).")
    if several_tp:
        questions.insert(0, {"id": "timepoint_policy",
            "question": f"{len(several_tp)} stud{'y reports' if len(several_tp) == 1 else 'ies report'} this outcome at several time points. "
                        "Effects from the same participants are not independent. Which time point should MetaPy use?",
            "options": [("Latest follow-up of each study", "latest"), ("Earliest follow-up of each study", "earliest"),
                        ("Keep all time points (ignores dependence; not recommended)", "all")],
            "default": s.get("timepoint_policy", "latest"), "multi": False})
        warnings.append(f"Time point rule: {s.get('timepoint_policy')} ({len(tp_dropped)} row(s) set aside).")
    ev = getattr(ds, "g_evidence", None)
    if isinstance(ev, pd.DataFrame) and not ev.empty:
        bad = ev[(ev["outcome"] == plan["outcome"]) & ev["basis"].str.startswith("CHECK")]
        for _, b in bad.iterrows():
            calc = [f"{k.replace(' from raw data', '')} = {b[k]:.3f}" for k in ("PPC2 from raw data", "change g from raw data", "endpoint g from raw data")
                    if pd.notna(b[k])]
            check_rows.append({"record_id": b["record_id"], "study": b["study"], "field": "g_reported", "reported": b["reported g"],
                               "recalculated": None, "rule": "design check",
                               "message": f"Reported g = {b['reported g']:.3f} does not match what this row's own data give ({'; '.join(calc)}). "
                                          "Check how the study's g was calculated."})
        counts["consistency_warnings"] = len(check_rows)
        checks_df = pd.DataFrame(check_rows)
    else:
        checks_df = pd.DataFrame(check_rows)
    if counts["included"] < 2:
        if questions:
            preview_s = copy.deepcopy(s)
            for q in questions:
                if q["id"] in ("ppc2_variance", "g_type_all", "smd_type", "equal_split"):
                    preview_s = apply_answers(preview_s, {q["id"]: q["default"]}, plan["outcome"], questions)
            n_prev = 0 if preview_s == s else prepare(ds, plan, {**preview_s, "_preview": True}).counts["included"] if not s.get("_preview") else 0
            warnings.insert(0, f"{counts['included']} comparison(s) are usable until you answer the questions below; "
                               + (f"MetaPy's suggested answers would make {n_prev} usable." if n_prev else "a meta-analysis needs at least 2."))
        else:
            warnings.insert(0, f"Only {counts['included']} comparison(s) can be analysed for this plan: a meta-analysis needs at least 2. "
                               "See the 'not included' reasons in the table.")
    return PrepResult(plan, m, data, log, actions, questions,
                      pd.DataFrame(req_rows, columns=["record_id", "study", "outcome", "have", "option_1", "option_2", "option_3", "fields_option_1"]),
                      checks_df, counts, s, sols, warnings)


def _actions(log: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame(columns=["what MetaPy did", "type", "studies", "formula", "citation"])
    g = log.groupby(["rule", "type", "formula", "citation"], dropna=False)["record_id"].nunique().reset_index()
    g = g.rename(columns={"rule": "what MetaPy did", "record_id": "studies"})
    order = {"exact": 0, "approximation": 1, "assumption": 2}
    g["_o"] = g["type"].map(order)
    return g.sort_values(["_o", "studies"], ascending=[True, False]).drop(columns="_o")[["what MetaPy did", "type", "studies", "formula", "citation"]]


def _questions(ds, recs, data, sols, m, s, multi_rows, check_rows, plan) -> list[dict[str, Any]]:
    qs: list[dict[str, Any]] = []
    blocked = data[data["derivation_basis"] == "Not derivable"]["record_id"].tolist()

    # what would assumptions unlock?
    def unlock(extra):
        s2 = {**s, **extra}
        ss = solver_settings(s2)
        n = 0
        for rid in blocked:
            src = numeric_source(recs[recs["record_id"] == rid].iloc[0])
            sol = solve(src, row_settings(ss, s2, rid), ds.inexact.get(rid, set()))
            if m.key == "g_best":
                add_best_g(sol, s2.get("best_policy", "design"))
            if sol.has(m.est, m.var):
                n += 1
        return n

    has_rep = recs["g_reported"].notna() if "g_reported" in recs else pd.Series(False, index=recs.index)
    if has_rep.any() and m.key in ("ppc2", "smd_g", "smd_change_g", "g_best"):
        gt = s.get("g_types") or {}
        rids = list(recs.loc[has_rep, "record_id"])
        n_set = sum(1 for r in rids if gt.get(r))
        n_unset = len(rids) - n_set
        qs.append({"id": "g_type_all", "question": f"{len(rids)} row(s) report a Hedges' g whose design is not stated in the column header. "
                   f"MetaPy checked them against the raw data (see 'Reported g: evidence'): {n_set} have a design suggested, {n_unset} need your decision. "
                   "How should MetaPy use reported g values where it cannot calculate the effect itself?",
                   "options": [("Use MetaPy's evidence-based suggestion for each row (recommended)", "suggested"),
                               ("Treat every reported g as Morris PPC2", "ppc2"), ("Treat every reported g as endpoint g", "post"),
                               ("Treat every reported g as change-score g", "change"), ("Do not use reported g values (raw data only)", "none")],
                   "default": s.get("_g_type_choice", "suggested"), "multi": False})
    needs_r = m.key in ("ppc2", "g_best", "smd_change_g", "md_change") and s.get("assumed_r") is None and not s.get("ppc2_var_simple")
    if needs_r:
        n = unlock({"assumed_r": 0.5})
        if m.key == "g_best":      # rows that would move from endpoint g to the baseline-adjusted PPC2 design
            s2 = {**s, "assumed_r": 0.5}
            ss2 = solver_settings(s2)
            for _, r in recs.iterrows():
                rid = r["record_id"]
                if rid in blocked or rid not in sols or getattr(sols[rid], "best_design", None) == "Morris PPC2":
                    continue
                sol2 = solve(numeric_source(r), row_settings(ss2, s2, rid), ds.inexact.get(rid, set()))
                if add_best_g(sol2, s2.get("best_policy", "design")) == "Morris PPC2":
                    n += 1
        if n:
            qs.append({"id": "ppc2_variance", "question": f"{n} comparison(s) need a pre-post correlation for their variance, but none is reported. How should MetaPy proceed? "
                       "(Whichever you choose, add the imputed-correlation sensitivity analysis in Step 3.)",
                       "options": [("Impute r = 0.5 and use the exact Morris (2008) variance (common default)", 0.5),
                                   ("Impute r = 0.7 (typical for stable traits / repeated scales)", 0.7),
                                   ("Impute r = 0.3 (conservative for wide variance)", 0.3),
                                   ("Use the simple SMD variance formula (approximation, no r needed)", "simple"),
                                   ("Do not impute: exclude these comparisons", None)],
                       "default": 0.5, "multi": False})

    if "smd_unspecified" in recs and recs["smd_unspecified"].notna().any() and s.get("smd_type") is None:
        k = int(recs["smd_unspecified"].notna().sum())
        qs.append({"id": "smd_type", "question": f"{k} row(s) report an SMD without saying whether it is Cohen's d or Hedges' g. How should MetaPy treat it?",
                   "options": [("Treat as Hedges' g", "g"), ("Treat as Cohen's d (MetaPy will apply the J correction)", "d"), ("Do not use these values", "exclude")],
                   "default": "g", "multi": False})
    if False:
        n = 0
        if n:
            qs.append({"id": "assumed_r", "question": f"{n} row(s) need a pre-post correlation that the paper does not report. Impute one? (MetaPy will add a sensitivity analysis.)",
                       "options": [("Do not impute (exclude these rows)", None), ("r = 0.3", 0.3), ("r = 0.5 (common default)", 0.5), ("r = 0.7", 0.7)],
                       "default": None, "multi": False})
    if not s.get("equal_split"):
        n = unlock({"equal_split": True})
        if n:
            qs.append({"id": "equal_split", "question": f"{n} row(s) give only a total sample size. Split it equally between the two arms?",
                       "options": [("No, exclude these rows", False), ("Yes, assume equal arms", True)], "default": False, "multi": False})
    used_groups = sorted({g for rid, sol in sols.items() for g in sol.groups_used([m.est, m.var])
                          if g in GROUP_INFO and g not in ("assumed_r", "equal_split", "smd_type", "g_type", "ppc2_var_simple")})
    if used_groups or s.get("vetoed_groups"):
        opts = [(f"{GROUP_INFO[g][0]} ({GROUP_INFO[g][1]})", g) for g in sorted(set(used_groups) | set(s.get("vetoed_groups", [])))]
        qs.append({"id": "allowed_groups", "question": "MetaPy used these approximations. Untick any you do not accept:",
                   "options": opts, "default": [g for _, g in opts if g not in s.get("vetoed_groups", [])], "multi": True})
    if check_rows:
        qs.append({"id": "consistency", "question": f"{len(check_rows)} reported value(s) disagree with what the study's own raw data give (see the consistency checks). Which should MetaPy use?",
                   "options": [("Keep the reported value", "reported"), ("Use the value recalculated from raw data", "recalculated")],
                   "default": s.get("consistency", "reported"), "multi": False})
    if multi_rows:
        desc = "; ".join(f"{r['study']} ({len(r['records'])} arms)" for r in multi_rows[:5])
        qs.append({"id": "multi_arm", "question": f"Multi-arm studies share a control group: {desc}. How should MetaPy avoid double-counting the control?",
                   "options": [("Split the shared control N across comparisons (Cochrane 23.3.4)", "split"),
                               ("Combine intervention arms into one group (Cochrane 6.5.2.10)", "combine"),
                               ("Keep only the first listed arm", "first"),
                               ("Keep all as they are (ignores the dependence)", "keep")],
                   "default": s.get("multi_arm", "split"), "multi": False})
    from .fields import known_direction
    hb_set = s.get("higher_is_better") or {}
    cur = hb_set.get(plan["outcome"]) if plan["outcome"] in hb_set else known_direction(plan["outcome"])
    from_file = (s.get("_direction_from_file") or {}).get(plan["outcome"])
    src_note = (" (pre-filled from your outcome-direction column)" if from_file else
                " (pre-selected: higher values of this outcome usually mean a worse state; please confirm)" if plan["outcome"] not in hb_set and cur is False else
                " (pre-selected: higher values usually mean a better state; please confirm)" if plan["outcome"] not in hb_set and cur is True else "")
    if m.family == "continuous":
        qs.append({"id": "higher_is_better", "question": f"For '{plan['outcome']}', are higher scores better or worse for participants?{src_note}",
                   "options": [("Higher is better", True), ("Higher is worse (e.g. symptoms, inflammation); flip signs so positive = benefit", False),
                               ("Not applicable / leave signs as they are", None)], "default": cur, "multi": False})
    elif m.family in ("binary", "time-to-event"):
        qs.append({"id": "higher_is_better", "question": f"For '{plan['outcome']}', is the event desirable (e.g. remission) or harmful (e.g. death)? (Used for plot labels.){src_note}",
                   "options": [("Desirable event", True), ("Harmful event", False), ("Not applicable", None)], "default": cur, "multi": False})
    return qs


def apply_answers(settings: dict[str, Any], answers: dict[str, Any], outcome: str,
                  questions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Turn the researcher's option choices into preparation settings."""
    s = copy.deepcopy(settings)
    offered = {}
    for q in questions or []:
        offered[q["id"]] = [v for _, v in q["options"]]
    for qid, val in answers.items():
        if qid == "smd_type":
            s["smd_type"] = val
        elif qid == "assumed_r":
            s["assumed_r"] = None if val in (None, "", "None") else float(val)
        elif qid == "equal_split":
            s["equal_split"] = bool(val)
        elif qid == "allowed_groups":
            allowed = set(val or [])
            groups = set(offered.get("allowed_groups", [])) | set(s.get("vetoed_groups", []))
            s["vetoed_groups"] = sorted(groups - allowed)
        elif qid == "consistency":
            s["consistency"] = val
        elif qid == "multi_arm":
            s["multi_arm"] = val
        elif qid == "ppc2_variance":
            if val == "simple":
                s["ppc2_var_simple"], s["assumed_r"] = True, None
            else:
                s["ppc2_var_simple"] = False
                s["assumed_r"] = None if val in (None, "", "None") else float(val)
        elif qid == "g_type_all":
            s["_g_type_choice"] = val
            sugg = s.get("_g_suggested", {})
            if val == "suggested":
                s["g_types"] = dict(sugg)
            elif val == "none":
                s["g_types"] = {}
            else:
                s["g_types"] = {rid: val for rid in sugg}
        elif qid == "g_type_rows":   # per-row overrides {record_id: design or None}
            gt = dict(s.get("g_types") or {})
            gt.update(val or {})
            s["g_types"] = gt
        elif qid == "best_policy":
            s["best_policy"] = val
        elif qid == "timepoint_policy":
            s["timepoint_policy"] = val
        elif qid == "higher_is_better":
            hib = dict(s.get("higher_is_better") or {})
            hib[outcome] = val
            s["higher_is_better"] = hib
    return s
