"""Analysis phase: pool the prepared data and run any extra analyses chosen."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import stats
from .io import Dataset
from .measures import Measure
from .prepare import PrepResult, prepare
from . import advanced as AD
from . import dependent as D
from . import extended as EXT
from . import metareg as MR
from . import glmm as GL
from . import advmodels as AM

DEFAULT_OPTIONS = {"model": "random", "tau_method": "REML", "ci_method": "HK", "extras": []}

EXTRA_LABELS = {
    "loo": "Leave-one-out", "influence": "Influence diagnostics", "exact_only": "Exact-only sensitivity",
    "no_assumptions": "No-assumption sensitivity", "r_grid": "Imputed-correlation sensitivity",
    "model_compare": "Fixed vs random effects", "egger": "Egger's test", "begg": "Begg's test",
    "trimfill": "Trim-and-fill", "cumulative": "Cumulative meta-analysis",
    "raw_only": "Sensitivity: effects calculated from raw data only", "design_only": "Sensitivity: most common design only",
    "tau_ci": "Heterogeneity: Q-profile confidence intervals", "pet_peese": "PET-PEESE (small-study adjusted estimates)",
    "selection": "Three-parameter selection model (3PSM)", "peters": "Peters' test (binary outcomes)",
    "failsafe": "Fail-safe N", "baujat": "Baujat diagnostics", "bayes": "Bayesian random-effects model",
    "multilevel": "Three-level model (effects nested in studies)", "che_rve": "CHE working model with robust variance estimation (CR2)",
    "metareg_multi": "Multiple meta-regression with permutation test",
    "tau2_compare": "Heterogeneity: estimate under every τ² estimator", "mh": "Mantel-Haenszel common-effect estimate",
    "peto": "Peto odds ratio", "glmm": "Generalized linear mixed model (exact binomial likelihood)",
    "selection_multi": "Step-function selection model with several cut-points", "influence_full": "Influence diagnostics (full set)",
    "tes": "Test of excess significance", "metareg_adv": "Meta-regression with interaction, predictions and robust (CR2) inference",
    "location_scale": "Location-scale model (moderators of heterogeneity)", "multilevel_nested": "Multilevel model with an extra nesting level",
}


@dataclass
class AnalysisResult:
    prep: PrepResult
    options: dict[str, Any]
    studies: pd.DataFrame
    main: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    @property
    def measure(self) -> Measure:
        return self.prep.measure


def _kw(o):
    return {"model": o["model"], "tau_method": o["tau_method"], "ci_method": o["ci_method"]}


def run(prep: PrepResult, options: dict[str, Any] | None = None, ds: Dataset | None = None) -> AnalysisResult:
    o = {**DEFAULT_OPTIONS, **(options or {})}
    studies = prep.data[prep.data["included"]].copy().reset_index(drop=True)
    main = stats.pool(studies["yi"], studies["vi"], **_kw(o))
    if main["status"] == "Pooled":
        studies["weight_percent"] = main["weights"]
    res = AnalysisResult(prep, o, studies, main)
    for key in o.get("extras", []):
        try:
            _extra(res, key, ds)
        except Exception as exc:  # keep the rest of the analysis going
            res.extras[key] = {"status": f"Could not be run: {exc}"}
    return res


def _extra(res: AnalysisResult, key: str, ds: Dataset | None):
    d, o, kw = res.studies, res.options, _kw(res.options)
    k = len(d)
    if key.split("::")[0] in ADVANCED:
        res.extras[key] = ADVANCED[key.split("::")[0]](res, key)
        return
    if key == "loo":
        res.extras[key] = {"table": stats.leave_one_out(d, **kw)} if k >= 3 else {"status": "Needs 3+ studies"}
    elif key == "influence":
        res.extras[key] = {"table": stats.influence(d, **kw)} if k >= 3 else {"status": "Needs 3+ studies"}
    elif key == "model_compare":
        rows = []
        for model, tm, ci in [("fixed", "—", "Wald"), ("random", "DL", "Wald"), ("random", "REML", "Wald"),
                              ("random", "REML", "HK"), ("random", "PM", "HK")]:
            r = stats.pool(d["yi"], d["vi"], model=model, tau_method=tm if tm != "—" else "REML", ci_method=ci)
            rows.append({"model": model, "tau2 estimator": tm, "CI method": ci, "estimate": r.get("estimate"),
                         "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi"), "p": r.get("p"), "tau2": r.get("tau2"), "I2": r.get("I2")})
        res.extras[key] = {"table": pd.DataFrame(rows)}
    elif key == "egger":
        res.extras[key] = stats.egger(d["yi"], d["vi"])
    elif key == "begg":
        res.extras[key] = stats.begg(d["yi"], d["vi"])
    elif key == "trimfill":
        res.extras[key] = stats.trim_and_fill(d["yi"], d["vi"], **kw)
    elif key == "cumulative":
        if "year" in d and d["year"].notna().sum() >= 2:
            res.extras[key] = {"table": stats.cumulative(d, "year", **kw)}
        else:
            res.extras[key] = {"status": "Needs a year column"}
    elif key == "exact_only":
        sub = d[d["derivation_basis"].isin(["Reported directly", "Exact calculation"])]
        res.extras[key] = {"result": stats.pool(sub["yi"], sub["vi"], **kw), "k": len(sub),
                           "note": f"{k - len(sub)} approximated/assumed effect(s) removed"}
    elif key == "no_assumptions":
        sub = d[d["derivation_basis"] != "Assumption"]
        res.extras[key] = {"result": stats.pool(sub["yi"], sub["vi"], **kw), "k": len(sub),
                           "note": f"{k - len(sub)} effect(s) relying on assumptions removed"}
    elif key == "raw_only":
        col = d.get("estimate from", pd.Series(index=d.index, dtype=object))
        sub = d[col.astype(str) == "raw data"]
        res.extras[key] = {"result": stats.pool(sub["yi"], sub["vi"], **kw), "k": len(sub),
                           "note": f"{k - len(sub)} effect(s) taken from reported g values removed"}
    elif key == "design_only":
        if "Effect-size design" not in d or d["Effect-size design"].nunique() < 2:
            res.extras[key] = {"status": "All included effects already share one design"}
            return
        top = d["Effect-size design"].value_counts().index[0]
        sub = d[d["Effect-size design"] == top]
        res.extras[key] = {"result": stats.pool(sub["yi"], sub["vi"], **kw), "k": len(sub),
                           "note": f"only {top}; {k - len(sub)} effect(s) of other designs removed"}
    elif key == "r_grid":
        if ds is None:
            res.extras[key] = {"status": "Dataset not available"}
            return
        rows = []
        for r_ in (0.3, 0.5, 0.7, 0.9):
            s2 = copy.deepcopy(res.prep.settings)
            s2["assumed_r"] = r_
            s2["ppc2_var_simple"] = False
            p2 = prepare(ds, res.prep.plan, s2)
            dd = p2.data[p2.data["included"]]
            pr = stats.pool(dd["yi"], dd["vi"], **kw)
            rows.append({"assumed r": r_, "k": len(dd), "estimate": pr.get("estimate"), "ci_lo": pr.get("ci_lo"),
                         "ci_hi": pr.get("ci_hi"), "p": pr.get("p"), "I2": pr.get("I2"), "tau2": pr.get("tau2")})
        res.extras[key] = {"table": pd.DataFrame(rows)}
    elif key.startswith("subgroup::"):
        col = key.split("::", 1)[1]
        if col not in d or d[col].isna().all():
            res.extras[key] = {"status": f"No values for {col}"}
            return
        sub = d[d[col].notna()]
        res.extras[key] = stats.subgroup(sub, col, **kw)
    elif key.startswith("metareg::"):
        col = key.split("::", 1)[1]
        if col not in d:
            res.extras[key] = {"status": f"No values for {col}"}
            return
        kk = int(d[col].notna().sum())
        if kk < 6:
            res.extras[key] = {"status": f"Not run: only {kk} studies with {col}. Meta-regression needs at least 6 studies "
                                         "(ideally 10 or more per moderator; Cochrane Handbook 10.11.5)."}
            return
        r = stats.meta_regression(d, [col], tau_method=o["tau_method"] if o["tau_method"] != "PM" else "REML",
                                  knha=o["ci_method"] == "HK")
        if r.get("k", 0) < 10 and r.get("status") == "Estimated":
            r["warning"] = f"Only {r['k']} studies: meta-regression has low power and high risk of spurious findings (Cochrane Handbook 10.11.5)."
        res.extras[key] = r


def participants(d: pd.DataFrame) -> int | None:
    """Participants counted once: a control group shared by several arms of one study is counted once."""
    if d is None or d.empty or "n_i" not in d:
        return None
    ni = pd.to_numeric(d["n_i"], errors="coerce").fillna(0).sum()
    nc = pd.to_numeric(d["n_c"], errors="coerce")
    key = d["study_id"].astype(str) + "|" + d.get("timepoint", pd.Series("", index=d.index)).astype(str)
    shared = pd.DataFrame({"k": key, "nc": nc}).groupby("k")["nc"].max().fillna(0).sum()
    return int(round(ni + shared))


def summary_table(res: AnalysisResult) -> pd.DataFrame:
    m, r = res.measure, res.main
    if r.get("status") != "Pooled":
        return pd.DataFrame([{"outcome": res.prep.plan["outcome"], "measure": m.reported_label, "status": r.get("status")}])
    return pd.DataFrame([{
        "outcome": res.prep.plan["outcome"], "measure": m.reported_label, "comparisons (k)": r["k"],
        "studies": int(res.studies["study_id"].nunique()),
        "participants": participants(res.studies),
        "model": f"{r['model']}" + (f" ({r['tau_method']}, {r['ci_method']})" if r["model"] == "random" else ""),
        "pooled": m.transform(r["estimate"]), "95% CI lower": m.transform(r["ci_lo"]), "95% CI upper": m.transform(r["ci_hi"]),
        "p": r["p"], "prediction interval": (f"{m.transform(r['pi_lo']):.2f} to {m.transform(r['pi_hi']):.2f}" if "pi_lo" in r else "n/a (k<3)"),
        "I2 (%)": r["I2"], "tau2": r["tau2"], "Q": r["Q"], "Q p": r["Q_p"],
    }])


def interpret(res: AnalysisResult) -> list[str]:
    """Plain-language reading of the result (for the review step)."""
    m, r = res.measure, res.main
    if r.get("status") != "Pooled":
        return [r.get("status", "Not pooled")]
    est, lo, hi = (m.transform(r[k]) for k in ("estimate", "ci_lo", "ci_hi"))
    sig = r["p"] < 0.05
    n_st = res.studies["study_id"].nunique()
    src = f"{r['k']} studies" if n_st == r["k"] else f"{r['k']} comparisons from {n_st} studies"
    lines = [f"Pooled {m.reported_label} = {est:.2f} (95% CI {lo:.2f} to {hi:.2f}; p = {r['p']:.3g}) from {src}."]
    null = "1" if m.back == "exp" else "0"
    lines.append(f"The 95% CI does not include {null} (no effect)." if sig else f"The 95% CI includes {null} (no effect), so the result is not statistically significant.")
    i2 = r["I2"]
    het = "low" if i2 < 40 else "moderate" if i2 < 60 else "substantial" if i2 < 75 else "considerable"
    lines.append(f"Heterogeneity is {het} (I² = {i2:.0f}%, Q p = {r['Q_p']:.3g}); Cochrane Handbook 10.10.2 bands are overlapping, so judge in context.")
    if "pi_lo" in r:
        lines.append(f"95% prediction interval: {m.transform(r['pi_lo']):.2f} to {m.transform(r['pi_hi']):.2f} (where a new study's true effect may lie).")
    if m.scale == "SMD":
        a = abs(r["estimate"])
        size = "very small" if a < 0.2 else "small" if a < 0.5 else "moderate" if a < 0.8 else "large"
        lines.append(f"By Cohen's rough benchmarks this is a {size} standardised effect; clinical relevance depends on the outcome.")
    c = res.prep.counts
    if c.get("approximation") or c.get("assumption"):
        bits = []
        if c.get("approximation"):
            bits.append(f"{c['approximation']} used published approximations")
        if c.get("assumption"):
            bits.append(f"{c['assumption']} relied on a researcher-confirmed assumption (e.g. imputed pre-post r, or the design of a reported g)")
        lines.append("Of the included effects, " + " and ".join(bits) + ": see the sensitivity analyses.")
    if r["k"] < 10:
        lines.append("Fewer than 10 studies: tests for funnel-plot asymmetry have low power.")
    return lines


# ------------------------------------------------------------------ v5 methods (advanced.py, dependent.py)
def _x_tau_ci(res, key):
    d = res.studies
    if len(d) < 2:
        return {"status": "Needs 2+ studies"}
    c = AD.tau2_qprofile(d["yi"], d["vi"])
    r = res.main
    rows = [{"statistic": "tau²", "value": r.get("tau2"), "ci_lo": c["tau2_lo"], "ci_hi": c["tau2_hi"]},
            {"statistic": "tau", "value": r.get("tau"), "ci_lo": c["tau_lo"], "ci_hi": c["tau_hi"]},
            {"statistic": "I² (%)", "value": r.get("I2"), "ci_lo": c["I2_lo"], "ci_hi": c["I2_hi"]},
            {"statistic": "H²", "value": r.get("H2"), "ci_lo": c["H2_lo"], "ci_hi": c["H2_hi"]}]
    return {"table": pd.DataFrame(rows), "ci": c, "status": "Performed"}


def _x_pet_peese(res, key):
    d = res.studies
    r = AD.pet_peese(d["yi"], d["vi"])
    if "PET" not in r:
        return r
    rows = [{"method": n, "estimate": r[n]["estimate"], "ci_lo": r[n]["ci_lo"], "ci_hi": r[n]["ci_hi"], "p": r[n]["p"],
             "slope p": r[n]["slope_p"], "selected": "yes" if r["conditional"] == n else ""} for n in ("PET", "PEESE")]
    return {**r, "table": pd.DataFrame(rows)}


def _x_selection(res, key):
    d = res.studies
    r = AD.selection_3psm(d["yi"], d["vi"])
    if "adjusted_estimate" not in r:
        return r
    t = pd.DataFrame([{"model": "Random effects (ML, no selection)", "estimate": r["unadjusted_ml_estimate"], "ci_lo": None, "ci_hi": None,
                       "note": ""},
                      {"model": "3PSM (selection at p < .05 two-sided)", "estimate": r["adjusted_estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                       "note": f"relative publication weight of non-significant studies = {r['delta']:.2f}; LRT = {r['LRT']:.2f}, p = {r['p_selection']:.3f}"}])
    return {**r, "table": t}


def _x_peters(res, key):
    d = res.studies
    if res.measure.family != "binary":
        return {"status": "Not applicable: Peters' test is for binary outcomes"}
    n = pd.to_numeric(d.get("n_i"), errors="coerce") + pd.to_numeric(d.get("n_c"), errors="coerce")
    if n.isna().any():
        return {"status": "Needs group sizes for every study"}
    return AD.peters(d["yi"], d["vi"], n)


def _x_failsafe(res, key):
    return AD.failsafe_n(res.studies["yi"], res.studies["vi"])


def _x_baujat(res, key):
    d = res.studies
    if len(d) < 3:
        return {"status": "Needs 3+ studies"}
    return {"table": AD.baujat(d["yi"], d["vi"], list(d["label"] if "label" in d else d["study_id"])), "status": "Performed"}


def _x_bayes(res, key):
    d = res.studies
    if len(d) < 2:
        return {"status": "Needs 2+ studies"}
    raw = res.measure.scale in ("Raw units", "Risk difference")
    scale = max(0.5, float(np.std(d["yi"]))) if raw else 0.5
    r = AD.bayes_re(d["yi"], d["vi"], tau_prior_scale=scale, mu_prior_sd=max(10.0, 10 * float(np.std(d["yi"]))) if raw else 10.0)
    t = pd.DataFrame([{"parameter": "mu (pooled effect)", "estimate": r["mu_median"], "ci_lo": r["mu_lo"], "ci_hi": r["mu_hi"]},
                      {"parameter": "new study (prediction)", "estimate": None, "ci_lo": r["pred_lo"], "ci_hi": r["pred_hi"]}])
    t2 = pd.DataFrame([{"tau (median)": r["tau_median"], "tau 95% CrI": f"{r['tau_lo']:.3f} to {r['tau_hi']:.3f}",
                        "P(mu > 0)": r["P(mu > 0)"], "priors": r["priors"]}])
    return {**r, "table": t, "table2": t2, "status": "Performed (posterior median and 95% credible intervals)"}


def _dep_data(res):
    d = res.studies.copy()
    return d, D.dependency_summary(d, "study_id")


def _dep_table(fit_coef, robust=None):
    t = fit_coef.copy()
    if robust is not None:
        t = robust.rename(columns={"se (CR2)": "se"})
    return t


def _x_multilevel(res, key):
    d, info = _dep_data(res)
    if info["clusters"] < 3:
        return {"status": "Needs 3+ studies"}
    r = D.multilevel(d, "study_id")
    f = r["fit"]
    note = ("" if info["clusters_with_several_effects"] else
            "Every study contributes one effect, so the within-study variance is not identified; the model equals the standard random-effects model. "
            "To use it, keep multi-arm comparisons separate and/or include all time points.")
    var = pd.DataFrame([{"level": "Between studies (level 3)", "variance": f.sigma2_between, "I² (%)": f.extra.get("I2_between")},
                        {"level": "Within studies (level 2)", "variance": f.sigma2_within, "I² (%)": f.extra.get("I2_within")},
                        {"level": "Total", "variance": f.sigma2_between + f.sigma2_within, "I² (%)": f.extra.get("I2_total")}])
    return {"table": f.coef, "variance_table": var, "level_tests": r["level_tests"], "info": info, "note": note,
            "estimate": f.estimate, "status": "Performed"}


def _x_che_rve(res, key):
    d, info = _dep_data(res)
    if info["clusters"] < 4:
        return {"status": "Needs 4+ studies for robust variance estimation"}
    rho = float(res.options.get("rve_rho", 0.6))
    r = D.che_rve(d, "study_id", None, rho)
    f = r["fit"]
    sens = []
    for rr in (0.0, 0.2, 0.4, 0.6, 0.8):
        ff = D.che_rve(d, "study_id", None, rr)
        rb = ff["robust"].iloc[0]
        sens.append({"rho": rr, "estimate": rb["estimate"], "ci_lo": rb["ci_lo"], "ci_hi": rb["ci_hi"], "p": rb["p"],
                     "df": rb["df (Satterthwaite)"], "sigma² between": ff["fit"].sigma2_between, "sigma² within": ff["fit"].sigma2_within})
    return {"table": r["robust"], "rho_sensitivity": pd.DataFrame(sens), "info": info, "rho": rho,
            "estimate": float(r["robust"]["estimate"].iloc[0]), "status": "Performed"}


def _x_metareg_multi(res, key):
    cols = key.split("::", 1)[1].split("|") if "::" in key else res.options.get("metareg_vars", [])
    d = res.studies
    cols = [c for c in cols if c in d]
    if len(cols) < 1:
        return {"status": "No moderators selected"}
    kk = len(d.dropna(subset=cols))
    if kk < 10:
        return {"status": f"Not run: only {kk} studies with complete moderator data; multiple meta-regression needs at least 10 "
                          "(Cochrane Handbook 10.11.5)."}
    o = res.options
    r = AD.metareg_permutation(d, cols, n_perm=int(o.get("n_perm", 1000)),
                               tau_method=o["tau_method"] if o["tau_method"] != "PM" else "REML", knha=o["ci_method"] == "HK")
    if r.get("status") == "Estimated" and r["k"] < 10 * len(cols):
        r["warning"] = f"{r['k']} studies for {len(cols)} moderators (fewer than 10 per moderator): results are exploratory."
    return r


# ------------------------------------------------------------------ v5.1 methods (extended.py, metareg.py, glmm.py, advmodels.py)
def _counts(d, prop=False):
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    if prop:
        for e_, n_ in (("events", "n_total"), ("events_i", "n_i")):
            x, n = num(e_), num(n_)
            if x.notna().all() and n.notna().all():
                return x.to_numpy(float), n.to_numpy(float)
        return None
    cols = [num(c) for c in ("events_i", "n_i", "events_c", "n_c")]
    if all(c.notna().all() for c in cols):
        return [c.to_numpy(float) for c in cols]
    return None


def _x_tau2_compare(res, key):
    d = res.studies
    if len(d) < 3:
        return {"status": "Needs 3+ studies"}
    t = EXT.compare_tau2(d["yi"].to_numpy(float), d["vi"].to_numpy(float))
    return {"table": t[["estimator", "tau2", "I2", "estimate", "ci_lo", "ci_hi"]], "status": "Performed",
            "note": "Wald-type intervals; the main analysis uses the estimator chosen in the analysis settings."}


def _binary_measure(res):
    return {"log_or": "OR", "log_rr": "RR", "rd": "RD"}.get(res.measure.key)


def _x_mh(res, key):
    meas = _binary_measure(res)
    if meas is None:
        return {"status": "Not applicable: Mantel-Haenszel needs a binary outcome (OR, RR or RD)"}
    c = _counts(res.studies)
    if c is None:
        return {"status": "Needs event counts and group sizes for every study"}
    a, n1, cc, n2 = c
    r = EXT.mantel_haenszel(a, n1 - a, cc, n2 - cc, meas)
    if r.get("status") != "Pooled":
        return r
    t = pd.DataFrame([{"method": r["method"], "k": r["k"], "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                       "p": r["p"], "Q": r["QE"], "I2": r["I2"]}])
    return {**r, "table": t}


def _x_peto(res, key):
    if res.measure.key != "log_or":
        return {"status": "Not applicable: Peto's method gives odds ratios"}
    c = _counts(res.studies)
    if c is None:
        return {"status": "Needs event counts and group sizes for every study"}
    a, n1, cc, n2 = c
    r = EXT.peto(a, n1 - a, cc, n2 - cc)
    if r.get("status") != "Pooled":
        return r
    t = pd.DataFrame([{"method": r["method"], "k": r["k"], "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                       "p": r["p"], "Q": r["QE"], "I2": r["I2"]}])
    note = "Peto's method can be biased when effects are large or groups are unbalanced (Cochrane Handbook 10.4.4.2)."
    return {**r, "table": t, "note": note}


def _x_glmm(res, key):
    d = res.studies
    if res.measure.key == "log_or":
        c = _counts(d)
        if c is None:
            return {"status": "Needs event counts and group sizes for every study"}
        r = GL.logistic_fixed_study(*c)
    elif res.measure.key == "logit_prop":
        c = _counts(d, prop=True)
        if c is None:
            return {"status": "Needs event counts and sample sizes for every study"}
        r = GL.binomial_normal(*c)
    else:
        return {"status": "Not applicable: available for odds ratios and single proportions"}
    if r.get("status") != "Pooled":
        return r
    t = pd.DataFrame([{"model": r["method"], "k": r["k"], "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                       "p": r["p"], "tau2": r["tau2"], "LRT for tau² = 0": r["LRT_tau2"], "p (LRT)": r["p_tau2"]}])
    return {**{k_: v_ for k_, v_ in r.items() if not isinstance(v_, (list, np.ndarray))}, "table": t, "status": "Performed"}


def _x_selection_multi(res, key):
    d = res.studies
    steps = res.options.get("selection_steps", [0.025, 0.05, 0.5])
    y, v = d["yi"].to_numpy(float), d["vi"].to_numpy(float)
    alt = "greater" if np.average(y, weights=1 / v) >= 0 else "less"
    r = EXT.selection_stepfun(y, v, steps, alternative=alt)
    if "adjusted_estimate" not in r:
        return {"status": r["status"], "counts": r.get("counts")}
    lab = ["p ≤ " + str(r["steps"][0])] + [f"{lo} < p ≤ {hi}" for lo, hi in zip(r["steps"][:-1], r["steps"][1:])]
    w = pd.DataFrame({"one-sided p interval": lab, "studies": r["counts"], "selection weight": r["delta"], "se": r["se_delta"]})
    t = pd.DataFrame([{"model": f"Step-function selection model ({len(r['steps']) - 1} cut-points)", "estimate": r["adjusted_estimate"],
                       "ci_lo": r["ci_lo0"], "ci_hi": r["ci_hi0"], "tau2": r["tau2"],
                       "note": f"LRT of no selection = {r['LRT']:.2f} (df = {r['LRT_df']}), p = {r['p_selection']:.3f}"}])
    return {**{k_: v_ for k_, v_ in r.items() if k_ not in ("beta", "se_beta", "ci_lo", "ci_hi")}, "table": t, "table2": w}


def _x_influence_full(res, key):
    d = res.studies
    if len(d) < 3:
        return {"status": "Needs 3+ studies"}
    lab = list(d["label"] if "label" in d else d["study_id"])
    t = EXT.influence_full(d["yi"].to_numpy(float), d["vi"].to_numpy(float), lab,
                           res.options["tau_method"] if res.options["model"] == "random" else "REML")
    return {"table": t, "status": "Performed", "note": "Cut-offs as in metafor (Viechtbauer & Cheung 2010)."}


def _x_tes(res, key):
    d = res.studies
    if len(d) < 5:
        return {"status": "Needs 5+ studies"}
    r = EXT.tes(d["yi"].to_numpy(float), d["vi"].to_numpy(float))
    t = pd.DataFrame([{"observed significant (O)": r["O"], "expected significant (E)": r["E"], "O/E": r["OE_ratio"],
                       "statistic": r.get("statistic"), "p (O > E)": r["p"]}])
    return {k_: v_ for k_, v_ in r.items() if k_ != "power"} | {"table": t, "status": "Performed"}


def _moderators_from_key(res, key):
    cols = key.split("::", 1)[1].split("|") if "::" in key else res.options.get("metareg_vars", [])
    return [c for c in cols if c in res.studies]


def _x_metareg_adv(res, key):
    d = res.studies
    cols = _moderators_from_key(res, key)
    if not cols:
        return {"status": "No moderators selected"}
    terms = [cols[0] + "*" + cols[1]] if len(cols) >= 2 else cols
    dd = d.dropna(subset=cols).reset_index(drop=True)
    o = res.options
    try:
        f = MR.fit_df(dd, terms, tau_method=o["tau_method"], test="knha" if o["ci_method"] == "HK" else "z")
    except ValueError as exc:
        return {"status": str(exc)}
    if f.k < 10:
        return {"status": f"Not run: only {f.k} studies with complete moderator data (at least 10 needed)"}
    t = f.table()
    om = MR.omnibus(f)
    tests = [{"hypothesis": "All moderators = 0", "test": om["test"], "statistic": om["statistic"], "p": om["p"]}]
    for c in cols[:2]:
        tt = MR.test_terms(f, c)
        tests.append({"hypothesis": f"All terms involving {c} = 0", "test": tt["test"], "statistic": tt["statistic"], "p": tt["p"]})
    # predictions: numeric moderators at their quartiles (others at their first level), categorical at every level
    grid = []
    base = {}
    for c in cols[:2]:
        num = pd.to_numeric(dd[c], errors="coerce")
        base[c] = float(num.median()) if num.notna().all() else str(sorted(dd[c].astype(str).unique())[0])
    first = cols[0]
    num = pd.to_numeric(dd[first], errors="coerce")
    vals = [float(q) for q in num.quantile([0.25, 0.5, 0.75])] if num.notna().all() else sorted(dd[first].astype(str).unique())
    for vv in vals:
        grid.append({**base, first: vv})
    pr = MR.predict(f, grid)
    rob = MR.robust(f, dd["study_id"]) if dd["study_id"].nunique() >= 4 else None
    return {"table": t.rename(columns={"statistic": "z / t"}), "table2": pd.DataFrame(tests), "table3": pr,
            "robust_table": rob, "tau2_residual": f.tau2, "R2_percent": MR.r2(f), "k": f.k, "terms": terms, "status": "Estimated"}


def _x_location_scale(res, key):
    d = res.studies
    cols = _moderators_from_key(res, key)
    if not cols:
        return {"status": "No moderator selected"}
    dd = d.dropna(subset=cols).reset_index(drop=True)
    if len(dd) < 10:
        return {"status": f"Not run: only {len(dd)} studies (at least 10 needed)"}
    Z, zn, _ = MR.design(dd, cols)
    r = AM.location_scale(dd["yi"], dd["vi"], np.ones((len(dd), 1)), Z, ["Intercept"], zn, "REML")
    if r.get("status") != "Estimated":
        return r
    sc = r["scale"].copy()
    sc.insert(1, "model part", "log τ²")
    return {"table": r["location"], "scale_table": sc, "LRT_scale": r["LRT_scale"], "p_scale": r["p_scale"], "k": r["k"],
            "status": "Estimated", "note": f"LRT that heterogeneity depends on the moderator(s): {r['LRT_scale']:.2f}, p = {r['p_scale']:.3f}"}


def _x_multilevel_nested(res, key):
    d = res.studies.copy()
    outer = key.split("::", 1)[1] if "::" in key else None
    if not outer or outer not in d:
        return {"status": "No grouping variable selected"}
    d["_es_id"] = np.arange(len(d))
    levels = [outer, "study_id", "_es_id"]
    if d.groupby("study_id")[outer].nunique().max() > 1:
        return {"status": f"Not run: some studies have more than one value of {outer}, so studies are not nested in {outer}"}
    if d[outer].nunique() < 3:
        return {"status": f"Not run: {outer} needs at least 3 groups"}
    if (d.groupby("study_id").size() > 1).sum() == 0:
        levels = [outer, "study_id"]
    r = AM.multilevel(d, levels)
    if r.get("status") != "Estimated":
        return r
    comp = r["components"].copy()
    comp["level"] = comp["level"].replace({"study_id": "Studies", "_es_id": "Effect sizes within studies"})
    return {"table": r["coefficients"], "variance_table": comp, "status": "Performed", "I2_total": r["I2_total"]}


ADVANCED = {"tau_ci": _x_tau_ci, "pet_peese": _x_pet_peese, "selection": _x_selection, "peters": _x_peters, "failsafe": _x_failsafe,
            "baujat": _x_baujat, "bayes": _x_bayes, "multilevel": _x_multilevel, "che_rve": _x_che_rve, "metareg_multi": _x_metareg_multi,
            "tau2_compare": _x_tau2_compare, "mh": _x_mh, "peto": _x_peto, "glmm": _x_glmm, "selection_multi": _x_selection_multi,
            "influence_full": _x_influence_full, "tes": _x_tes, "metareg_adv": _x_metareg_adv, "location_scale": _x_location_scale,
            "multilevel_nested": _x_multilevel_nested}
