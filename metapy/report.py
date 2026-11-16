"""Assemble the researcher-selected tables, figures and text into one report."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from . import figures as F
from .analysis import EXTRA_LABELS, AnalysisResult, interpret, summary_table
from .derive.registry import GROUP_INFO
from .io import Dataset

TABLE_CHOICES = {
    "characteristics": "Characteristics of included studies",
    "effects": "Study-level effect sizes",
    "pooled": "Pooled results and heterogeneity",
    "sof": "Summary of findings (GRADE columns to complete)",
    "extras": "Tables for every extra analysis (subgroup, meta-regression, sensitivity, bias)",
    "derivations": "Derivation log: every value MetaPy calculated, with formula and citation",
    "actions": "Summary of data-preparation actions",
    "requests": "Studies excluded for missing data, with what to extract",
    "checks": "Consistency checks (reported vs recalculated)",
    "parsing": "Data import: merged duplicates, outcome grouping, column mapping, text cells",
    "evidence": "Reported Hedges' g: design check against raw data",
}
FIGURE_CHOICES = {
    "summary_forest": "Summary forest plot across outcomes (one row per outcome)",
    "forest": "Forest plot", "subgroup_forest": "Subgroup forest plot(s)", "funnel": "Funnel plot (contour-enhanced)",
    "bubble": "Meta-regression bubble plot(s)", "loo": "Leave-one-out plot", "cumulative": "Cumulative meta-analysis plot",
    "provenance": "Data-provenance chart (how each effect was obtained)", "prisma": "PRISMA 2020 flow diagram",
    "baujat": "Baujat plot (needs the Baujat analysis)", "rho": "CHE-RVE sensitivity to ρ (needs the CHE-RVE analysis)",
    "network": "Network graph and forest plot (network meta-analysis)",
    "radial": "Radial (Galbraith) plot", "labbe": "L'Abbé plot (binary outcomes with event counts)",
}
TEXT_CHOICES = {"methods": "Auto-written methods paragraph (with citations)", "results": "Auto-written results paragraph",
                "plain": "Plain-language summary and cautions (for readers new to meta-analysis)"}


@dataclass
class Section:
    heading: str
    text: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    figures: list[tuple[str, Any]] = field(default_factory=list)


@dataclass
class Report:
    title: str
    sections: list[Section]
    meta: dict[str, Any]


def _r(x, d=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return round(float(x), d)


def characteristics_table(res: AnalysisResult, ds: Dataset) -> pd.DataFrame:
    d = res.studies
    cols = {"label": "Study", "year": "Year", "intervention_arm": "Intervention", "n_i": "N intervention", "n_c": "N control"}
    t = d[[c for c in cols if c in d]].rename(columns=cols)
    for c in ds.characteristics:
        if c in d and d[c].notna().any():
            t[c] = d[c]
    for c in ("N intervention", "N control", "Year"):
        if c in t:
            t[c] = pd.to_numeric(t[c], errors="coerce").round(0).astype("Int64")
    # drop empty columns and characteristics that are identical for every study
    keep = [c for c in t.columns if c == "Study" or (t[c].notna().any() and (c in cols.values() or t[c].astype(str).nunique() > 1))]
    return t[keep]


def effects_table(res: AnalysisResult) -> pd.DataFrame:
    m, d = res.measure, res.studies
    return pd.DataFrame({
        "Study": d["label"], m.short: [_r(m.transform(x)) for x in d["yi"]],
        "95% CI": [f"{m.transform(a):.2f} to {m.transform(b):.2f}" for a, b in zip(d["ci_lo"], d["ci_hi"])],
        "SE (analysis scale)": [_r(x, 3) for x in d["se"]],
        "Weight (%)": [_r(x, 1) for x in d.get("weight_percent", pd.Series([None] * len(d)))],
        "How obtained": d["derivation_basis"], "Approximations / assumptions": d["approximations_or_assumptions"],
        "Calculation steps": d["derivation_steps"], "Notes": d["notes"],
    })


def sof_table(results: list[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        r, m = res.main, res.measure
        if r.get("status") != "Pooled":
            continue
        from .analysis import participants
        rows.append({"Outcome": res.prep.plan["outcome"], "Studies (participants)": f"{res.studies['study_id'].nunique()} studies, {r['k']} comparisons ({participants(res.studies)} participants)",
                     "Effect (95% CI)": f"{m.short} {m.transform(r['estimate']):.2f} ({m.transform(r['ci_lo']):.2f} to {m.transform(r['ci_hi']):.2f})",
                     "I²": f"{r['I2']:.0f}%", "Risk of bias": "", "Inconsistency": "", "Indirectness": "", "Imprecision": "",
                     "Publication bias": "", "Certainty (GRADE)": "to be judged by reviewers", "Comments": ""})
    return pd.DataFrame(rows)


def extra_tables(res: AnalysisResult) -> list[tuple[str, pd.DataFrame]]:
    out = []
    m = res.measure
    for key, val in res.extras.items():
        name = EXTRA_LABELS.get(key) or key.replace("subgroup::", "Subgroup analysis: ").replace("metareg::", "Meta-regression: ")
        if not isinstance(val, dict):
            continue
        if "table" in val and isinstance(val["table"], pd.DataFrame) and key.startswith("subgroup::"):
            t = val["table"][["subgroup", "k", "estimate", "ci_lo", "ci_hi", "p", "I2", "tau2"]].copy()
            for c in ("estimate", "ci_lo", "ci_hi"):
                t[c] = t[c].map(lambda x: _r(m.transform(x)) if pd.notna(x) else "")
            if "Q_between" in val:
                t.loc[len(t)] = ["Test for subgroup differences", "", f"Q = {val['Q_between']:.2f}", f"df = {val['df']}", "", val["p_between"], "", ""]
            out.append((name, t))
        elif "coefficients" in val:
            t = val["coefficients"].copy().astype(object)
            nc = len(t.columns)
            pad = lambda first: first + [""] * (nc - len(first))
            pcol = list(t.columns).index("p") if "p" in t.columns else nc - 1
            row = pad([f"QM ({val['test']})", val["QM"]]); row[pcol] = val["QM_p"]
            if "permutation" in val and "p_permutation" in t.columns:
                row[list(t.columns).index("p_permutation")] = val["permutation"]["QM_p"]
            t.loc[len(t)] = row
            row = pad(["Residual heterogeneity QE", val["QE"]]); row[pcol] = val["QE_p"]; t.loc[len(t)] = row
            t.loc[len(t)] = pad(["R² (% heterogeneity explained)", val["R2_percent"]])
            out.append((name + (" — " + val["warning"] if val.get("warning") else ""), t))
        elif "table" in val and isinstance(val["table"], pd.DataFrame):
            t = val["table"].copy()
            if m.back and not key.startswith(("tau_ci", "baujat", "metareg_adv", "location_scale", "multilevel_nested", "influence_full", "tes")):
                for c in ("estimate", "ci_lo", "ci_hi"):
                    if c in t:
                        t[c] = t[c].map(lambda x: m.transform(x) if pd.notna(x) else x)
            title = name + (f" — {val['warning']}" if val.get("warning") else "") + (f" — {val['status']}" if str(val.get("status", "")).startswith("Performed (") else "")
            out.append((title, t))
            for sub, lab in (("variance_table", "variance components"), ("level_tests", "likelihood-ratio tests for each level"),
                             ("rho_sensitivity", "sensitivity to the assumed within-study correlation"),
                             ("table2", "heterogeneity and priors" if key == "bayes" else ("estimated selection weights" if key == "selection_multi" else "tests of moderators")),
                             ("table3", "predicted effects (with 95% confidence and prediction intervals)"),
                             ("robust_table", "cluster-robust (CR2) inference, clustered by study"),
                             ("scale_table", "scale part: moderators of log τ²")):
                if isinstance(val.get(sub), pd.DataFrame):
                    st = val[sub].copy()
                    if m.back and sub == "rho_sensitivity":
                        for c in ("estimate", "ci_lo", "ci_hi"):
                            st[c] = st[c].map(m.transform)
                    out.append((f"{name}: {lab}", st))
            if val.get("note"):
                out.append((f"{name}: note", pd.DataFrame([{"note": val["note"]}])))
        elif "result" in val:
            rr = val["result"]
            out.append((name, pd.DataFrame([{"studies": val.get("k"), "estimate": _r(m.transform(rr.get("estimate"))) if rr.get("estimate") is not None else "",
                                              "ci_lo": _r(m.transform(rr.get("ci_lo"))) if rr.get("ci_lo") is not None else "",
                                              "ci_hi": _r(m.transform(rr.get("ci_hi"))) if rr.get("ci_hi") is not None else "",
                                              "p": rr.get("p"), "I2": rr.get("I2"), "status": rr.get("status"), "note": val.get("note")}])))
        elif key == "trimfill":
            adj = val.get("adjusted", {})
            out.append((name, pd.DataFrame([{"imputed studies (k0)": val.get("k0"), "side": val.get("side"),
                                              "adjusted estimate": _r(m.transform(adj.get("estimate"))) if adj.get("estimate") is not None else "",
                                              "ci_lo": _r(m.transform(adj.get("ci_lo"))) if adj.get("ci_lo") is not None else "",
                                              "ci_hi": _r(m.transform(adj.get("ci_hi"))) if adj.get("ci_hi") is not None else "",
                                              "status": val.get("status")}])))
        else:
            out.append((name, pd.DataFrame([{k: v for k, v in val.items() if not isinstance(v, (list, dict, pd.DataFrame, np.ndarray))}])))
    return out


def methods_text(results: list[AnalysisResult]) -> list[str]:
    if not results:
        return []
    r0 = results[0]
    o = r0.options
    measures = sorted({r.measure.label for r in results})
    paras = []
    per = "; ".join(f"{r.prep.plan['outcome']}: {r.measure.label}" for r in results)
    p = (f"Meta-analyses were conducted in MetaPy {__version__} (Python). Effect measures by outcome: {per}. ")
    if o["model"] == "random":
        tau = {"REML": "restricted maximum likelihood (REML)", "DL": "the DerSimonian-Laird method", "PM": "the Paule-Mandel method",
               "ML": "maximum likelihood", "EB": "the empirical Bayes method (Morris 1983)", "HE": "the Hedges method",
               "HS": "the Hunter-Schmidt method", "HSk": "the Hunter-Schmidt method with small-sample correction",
               "SJ": "the Sidik-Jonkman method"}.get(o["tau_method"], o["tau_method"])
        p += f"A random-effects model was used, with between-study variance (τ²) estimated by {tau}"
        p += (" and confidence intervals computed with the Hartung-Knapp-Sidik-Jonkman adjustment (modified so it is never narrower than the Wald interval)."
              if o["ci_method"] == "HK" else " and Wald-type confidence intervals.")
        p += " 95% prediction intervals were calculated where at least three studies were available."
    else:
        p += "A fixed-effect (inverse-variance) model was used."
    p += (" Heterogeneity was assessed with Cochran's Q and the I² statistic (Higgins & Thompson 2002)"
          + (", with I² computed from the estimated τ²." if o["model"] == "random" else "."))
    paras.append(p)
    # Describe only the calculations that were actually performed
    rule_cites: dict[str, str] = {}
    for r in results:
        if r.prep.log.empty:
            continue
        inc = set(r.studies["record_id"])
        for _, row in r.prep.log[r.prep.log["record_id"].isin(inc)].iterrows():
            name = re.sub(r"\s*\((post|pre|change)(, [ic])?\)|\s*\([ic]\)", "", str(row["rule"])).strip()
            rule_cites.setdefault(name, row["citation"] or "")
    direct = all(r.prep.log.empty for r in results)
    if direct:
        dp = "All effect sizes and variances were used as reported; no values were derived."
    else:
        items = [f"{n}{f' ({c})' if c else ''}" for n, c in rule_cites.items()]
        dp = ("Effect sizes and their sampling variances were calculated from the statistics reported in each study with the following formulas: "
              + "; ".join(items) + ". Every calculated value, its inputs and its formula are listed in the derivation log.")
    used = set()
    for r in results:
        inc = set(r.studies["record_id"])
        for rid, sol in r.prep.solutions.items():
            if rid in inc:
                used |= sol.groups_used([r.measure.est, r.measure.var])
    extra = []
    for g in sorted(used):
        if g in GROUP_INFO:
            name = GROUP_INFO[g][0]
            extra.append(f"{name[0].lower() + name[1:]} ({GROUP_INFO[g][1]})")
    if extra:
        dp += " The following approximations or researcher-approved assumptions were used: " + "; ".join(extra) + "."
    flips = [r.prep.plan["outcome"] for r in results if r.studies.get("direction_flipped", pd.Series(dtype=bool)).any()]
    kept = [r.prep.plan["outcome"] for r in results
            if (r.prep.settings.get("higher_is_better") or {}).get(r.prep.plan["outcome"]) == "worse_keep"]
    if kept:
        dp += (f" For {', '.join(kept)}, higher scores indicate a worse state; the original signs were kept, so negative values "
               "indicate a benefit of the intervention.")
    if flips:
        dp += (f" For outcomes where higher scores indicate a worse state ({', '.join(flips)}), signs were reversed so that positive "
               "effects favour the intervention.")
    s0 = r0.prep.settings
    if any(r.prep.data["notes"].astype(str).str.contains("shared control|combined").any() for r in results):
        dp += {"split": " For multi-arm studies the shared control group was split across comparisons (Cochrane Handbook 23.3.4).",
               "combine": " For multi-arm studies intervention arms were combined (Cochrane Handbook 6.5.2.10).",
               "first": " For multi-arm studies only one comparison was retained.", "keep": ""}.get(s0.get("multi_arm"), "")
    if any(r.prep.data["exclusion_reason"].astype(str).str.contains("Other time point").any() for r in results):
        dp += f" Where a study reported several follow-up times, the {s0.get('timepoint_policy', 'latest')} was used."
    paras.append(dp)
    ex = set(k.split("::")[0] for r in results for k in r.extras)
    sp = []
    if "subgroup" in ex:
        sp.append("subgroup analyses with a test for subgroup differences")
    if "metareg" in ex:
        sp.append("mixed-effects meta-regression" + (" with the Knapp-Hartung adjustment" if o["ci_method"] == "HK" else ""))
    if "loo" in ex:
        sp.append("leave-one-out sensitivity analysis")
    if "influence" in ex:
        sp.append("influence diagnostics (studentised residuals and Cook's distances; Viechtbauer & Cheung 2010)")
    if ex & {"exact_only", "no_assumptions", "r_grid"}:
        sp.append("sensitivity analyses restricted to exactly derived effect sizes and/or varying imputed correlations")
    if "cumulative" in ex:
        sp.append("cumulative meta-analysis by publication year")
    if ex & {"egger", "begg", "trimfill"}:
        b = []
        if "egger" in ex:
            b.append("Egger's regression test (Egger et al. 1997)")
        if "begg" in ex:
            b.append("Begg's rank correlation test (Begg & Mazumdar 1994)")
        if "trimfill" in ex:
            b.append("trim-and-fill (Duval & Tweedie 2000)")
        sp.append("small-study effects were explored with contour-enhanced funnel plots, " + ", ".join(b))
    if "tau_ci" in ex:
        sp.append("confidence intervals for τ², I² and H² by the Q-profile method (Viechtbauer 2007)")
    if "bayes" in ex:
        sp.append("a Bayesian normal-normal random-effects model with a half-normal prior on τ and a vague normal prior on the mean, "
                  "computed by exact numerical integration (Röver 2020)")
    if ex & {"pet_peese", "selection", "peters", "failsafe"}:
        b = []
        if "pet_peese" in ex:
            b.append("PET-PEESE regression estimates (Stanley & Doucouliagos 2014)")
        if "selection" in ex:
            b.append("a three-parameter selection model with one cut-point at two-sided p = .05 (Vevea & Hedges 1995)")
        if "peters" in ex:
            b.append("Peters' regression test (Peters et al. 2006)")
        if "failsafe" in ex:
            b.append("fail-safe N (Rosenthal 1979; Orwin 1983), reported descriptively only")
        sp.append("publication bias was further examined with " + ", ".join(b))
    if "baujat" in ex:
        sp.append("Baujat plots to identify studies that contribute most to heterogeneity (Baujat et al. 2002)")
    if "multilevel" in ex:
        sp.append("a three-level random-effects model (effect sizes nested within studies; REML) with likelihood-ratio tests of each variance "
                  "component using a 50:50 chi-square mixture, and level-specific I² (Cheung 2014)")
    if "che_rve" in ex:
        rho = next((v.get("rho") for r in results for k_, v in r.extras.items() if k_ == "che_rve" and "rho" in v), 0.6)
        sp.append(f"a correlated-hierarchical-effects working model (assumed within-study correlation ρ = {rho}, varied from 0 to 0.8 in "
                  "sensitivity analysis) with cluster-robust CR2 standard errors and Satterthwaite degrees of freedom (Pustejovsky & Tipton 2022)")
    if "metareg_multi" in ex:
        sp.append("multiple meta-regression with permutation-based p-values (1,000 permutations; Higgins & Thompson 2004)")
    if "tau2_compare" in ex:
        sp.append("a sensitivity analysis of the pooled estimate across nine τ² estimators (Viechtbauer 2005; Veroniki et al. 2016)")
    if "influence_full" in ex:
        sp.append("case diagnostics including externally standardized residuals, DFFITS, Cook's distances, covariance ratios, leave-one-out τ² "
                  "and hat values (Viechtbauer & Cheung 2010)")
    if "mh" in ex:
        sp.append("a Mantel-Haenszel common-effect analysis of the 2×2 tables (Mantel & Haenszel 1959; variances of Robins, Breslow & Greenland "
                  "1986, Greenland & Robins 1985 and Sato, Greenland & Robins 1989)")
    if "peto" in ex:
        sp.append("Peto's one-step odds ratio (Yusuf et al. 1985)")
    if "glmm" in ex:
        sp.append("a generalized linear mixed model with the exact binomial likelihood (fixed study effects and a random treatment effect for "
                  "odds ratios, or the binomial-normal model for proportions), fitted by maximum likelihood with adaptive Gauss-Hermite "
                  "quadrature (Stijnen et al. 2010)")
    if "selection_multi" in ex:
        sp.append("a step-function selection model with cut-points at one-sided p = .025, .05 and .5 (Vevea & Hedges 1995; Vevea & Woods 2005)")
    if "tes" in ex:
        sp.append("the test of excess significance (Ioannidis & Trikalinos 2007)")
    if "metareg_adv" in ex:
        sp.append("meta-regression including an interaction term, with tests of linear hypotheses, predicted effects with prediction intervals, "
                  "and cluster-robust CR2 standard errors with Satterthwaite degrees of freedom (Tipton 2015; Pustejovsky & Tipton 2018)")
    if "location_scale" in ex:
        sp.append("a location-scale model in which log τ² depends on moderators, with a likelihood-ratio test (Viechtbauer & López-López 2022)")
    if "multilevel_nested" in ex:
        sp.append("a multilevel model with an additional nesting level above studies (REML; Konstantopoulos 2011), with likelihood-ratio tests "
                  "of each variance component")
    if sp:
        paras.append("Additional analyses included " + "; ".join(sp) + ". Tests for funnel-plot asymmetry were interpreted cautiously when fewer than 10 studies were available.")
    paras.append("Reporting follows the PRISMA 2020 statement (Page et al. 2021).")
    return paras


def results_text(results: list[AnalysisResult]) -> list[str]:
    out = []
    for r in results:
        lines = interpret(r)
        out.append(f"{r.prep.plan['outcome']}: " + " ".join(lines))
        for key, val in r.extras.items():
            if key.startswith("subgroup::") and "p_between" in val:
                out.append(f"Subgroup analysis by {val['variable']}: test for subgroup differences Q = {val['Q_between']:.2f}, df = {val['df']}, p = {val['p_between']:.3f}.")
            if key.startswith("metareg::") and val.get("status") == "Estimated":
                c = val["coefficients"].iloc[1:]
                terms = "; ".join(f"{t}: b = {b:.3f} (95% CI {lo:.3f} to {hi:.3f}), p = {p:.3f}" for t, b, lo, hi, p in zip(c["term"], c["estimate"], c["ci_lo"], c["ci_hi"], c["p"]))
                out.append(f"Meta-regression ({', '.join(val['moderators'])}): {terms}; R² = {val['R2_percent']:.0f}%.")
            if key == "egger" and "p" in val:
                out.append(f"Egger's test: intercept = {val['intercept']:.2f}, p = {val['p']:.3f}"
                           + (" (fewer than 10 studies: low power, interpret with caution)." if r.main.get("k", 0) < 10 else "."))
            m_ = r.measure
            f_ = lambda x: f"{m_.transform(x):.2f}"
            if key == "tau_ci" and "ci" in val:
                c = val["ci"]
                out.append(f"Heterogeneity: τ² = {r.main.get('tau2', float('nan')):.3f} (95% CI {c['tau2_lo']:.3f} to {c['tau2_hi']:.3f}); "
                           f"I² = {r.main.get('I2', float('nan')):.0f}% (95% CI {c['I2_lo']:.0f}% to {c['I2_hi']:.0f}%).")
            if key == "bayes" and "mu_median" in val:
                out.append(f"Bayesian model: {m_.short} = {f_(val['mu_median'])} (95% credible interval {f_(val['mu_lo'])} to {f_(val['mu_hi'])}); "
                           f"τ = {val['tau_median']:.2f} ({val['tau_lo']:.2f} to {val['tau_hi']:.2f}); posterior probability that the effect "
                           f"is above zero (on the analysis scale) = {val['P(mu > 0)']:.3f}.")
            if key == "pet_peese" and "conditional" in val:
                n_ = val["conditional"]
                out.append(f"PET-PEESE: the conditional procedure selected {n_}; adjusted {m_.short} = {f_(val[n_]['estimate'])} "
                           f"(95% CI {f_(val[n_]['ci_lo'])} to {f_(val[n_]['ci_hi'])}).")
            if key == "selection" and "adjusted_estimate" in val:
                out.append(f"Selection model (3PSM): adjusted {m_.short} = {f_(val['adjusted_estimate'])} (95% CI {f_(val['ci_lo'])} to "
                           f"{f_(val['ci_hi'])}); test for selection LRT = {val['LRT']:.2f}, p = {val['p_selection']:.3f}.")
            if key == "peters" and "p" in val:
                out.append(f"Peters' test: slope = {val['slope']:.2f}, p = {val['p']:.3f}.")
            if key == "multilevel" and val.get("status") == "Performed":
                vt = val["variance_table"]
                out.append(f"Three-level model ({val['info']['effects']} effects in {val['info']['clusters']} studies): {m_.short} = "
                           f"{f_(val['table']['estimate'].iloc[0])} (95% CI {f_(val['table']['ci_lo'].iloc[0])} to {f_(val['table']['ci_hi'].iloc[0])}); "
                           f"between-study variance {vt['variance'].iloc[0]:.3f}, within-study variance {vt['variance'].iloc[1]:.3f}.")
            if key == "che_rve" and val.get("status") == "Performed":
                t0 = val["table"].iloc[0]
                out.append(f"CHE-RVE (ρ = {val['rho']}): {m_.short} = {f_(t0['estimate'])} (95% CI {f_(t0['ci_lo'])} to {f_(t0['ci_hi'])}), "
                           f"t({t0['df (Satterthwaite)']:.1f}) = {t0['t']:.2f}, p = {t0['p']:.3f}" + (f". {t0['note']}." if t0["note"] else "."))
            if key.startswith("metareg_multi") and val.get("status") == "Estimated":
                out.append(f"Multiple meta-regression ({', '.join(val['moderators'])}): QM p = {val['QM_p']:.3f}, permutation p = "
                           f"{val['permutation']['QM_p']:.3f}; R² = {val['R2_percent']:.0f}%.")
            if key == "trimfill" and val.get("status") == "Performed":
                adj = val["adjusted"]
                m = r.measure
                out.append(f"Trim-and-fill imputed {val['k0']} stud{'y' if val['k0'] == 1 else 'ies'}; adjusted {m.short} = {m.transform(adj['estimate']):.2f} "
                           f"(95% CI {m.transform(adj['ci_lo']):.2f} to {m.transform(adj['ci_hi']):.2f}).")
    return out


def build_report(results: list[AnalysisResult], ds: Dataset, tables: list[str], figures: list[str], texts: list[str],
                 prisma_counts: dict[str, Any] | None = None, theme: str = "navy", title: str = "Meta-analysis report",
                 forest_sort: str | None = None, nma_results: list[dict[str, Any]] | None = None) -> Report:
    secs: list[Section] = []
    foreign = [r.prep.plan["outcome"] for r in results if getattr(r.prep, "fingerprint", ds.fingerprint) != ds.fingerprint
               or not set(r.prep.data["record_id"]) <= set(ds.records["record_id"])]
    if foreign:
        raise ValueError("Results for " + ", ".join(foreign) + " were not produced from the current data file. "
                         "Re-run the analysis for this file; results from another file are never mixed in.")
    from .guide import CITATION
    meta = {"MetaPy version": __version__, "How to cite MetaPy": CITATION, "Source file": ds.source_name,
            "Data file fingerprint (SHA-256, first 16)": ds.fingerprint[:16] or "n/a",
            "Studies in data file": int(ds.records["study_id"].nunique()), "Source file modified": "No",
            "Analyses": "; ".join(f"{r.prep.plan['outcome']} — {r.measure.label}" for r in results)}
    if prisma_counts is not None and F.prisma_given(prisma_counts):      # optional output: only when the researcher switched it on
        chk = F.prisma_checks(prisma_counts)
        secs.append(Section("Study selection", figures=[("PRISMA 2020 flow diagram", F.prisma_diagram(prisma_counts, "PRISMA 2020 flow diagram of study selection"))],
                            text=[f"PRISMA numbers to check: {x}" for x in chk]))
    if "methods" in texts:
        secs.append(Section("Methods", text=methods_text(results)))
    if "results" in texts:
        secs.append(Section("Results summary", text=results_text(results)))
    if "plain" in texts:
        from . import guide as GD
        txt = []
        for r in results:
            txt.append(f"{r.prep.plan['outcome']} ({r.measure.label}):")
            txt.extend(GD.interpret(r))
            txt.extend("Caution: " + w for w in GD.warnings_for(r))
        txt.append("These sentences describe the numbers only. Judge the certainty of the evidence (for example with GRADE), "
                   "the risk of bias of the included studies and the practical importance of the effect before drawing conclusions.")
        secs.append(Section("Plain-language summary", text=txt))
    if "sof" in tables:
        secs.append(Section("Summary of findings", tables=[("Summary of findings table", sof_table(results))],
                            text=["GRADE judgements are left blank for the review team to complete."]))
    pooled_res = [r for r in results if r.main.get("status") == "Pooled"]
    if "summary_forest" in figures and len(pooled_res) >= 2:
        secs.append(Section("All outcomes", figures=[("Summary of pooled effects across outcomes", F.summary_forest(pooled_res, theme))]))
    for r in results:
        out = r.prep.plan["outcome"]
        s = Section(f"{out}: {r.measure.label}")
        if r.prep.warnings:
            s.text += [f"Note: {w}" for w in r.prep.warnings]
        if r.main.get("status") != "Pooled":
            s.text.append(f"Not pooled: {r.main.get('status')}")
        if "characteristics" in tables:
            s.tables.append(("Characteristics of included studies", characteristics_table(r, ds)))
        if "pooled" in tables:
            s.tables.append(("Pooled result", summary_table(r)))
        if "effects" in tables:
            s.tables.append(("Study-level effect sizes", effects_table(r)))
        if "forest" in figures and r.main.get("status") == "Pooled":
            s.figures.append(("Forest plot", F.forest_plot(r, theme, sort_by=forest_sort)))
        if "subgroup_forest" in figures:
            for k, v in r.extras.items():
                if k.startswith("subgroup::") and "table" in v:
                    s.figures.append((f"Subgroup forest plot: {v['variable']}", F.subgroup_forest(r, k, theme)))
        if "funnel" in figures and r.main.get("status") == "Pooled" and len(r.studies) < 10:
            s.text.append(f"Funnel plot not drawn: {len(r.studies)} studies (at least 10 are needed for a meaningful funnel plot; Cochrane Handbook 13.3.5.4).")
        if "funnel" in figures and r.main.get("status") == "Pooled" and len(r.studies) >= 10:
            s.figures.append(("Funnel plot", F.funnel_plot(r, theme, trimfill=r.extras.get("trimfill") if "trimfill" in r.extras else None)))
        if "bubble" in figures:
            for k, v in r.extras.items():
                if k.startswith("metareg::") and v.get("status") == "Estimated":
                    s.figures.append((f"Meta-regression: {v['moderators'][0]}", F.bubble_plot(r, k, theme)))
        if "loo" in figures and "table" in r.extras.get("loo", {}):
            s.figures.append(("Leave-one-out", F.sensitivity_forest(r, r.extras["loo"]["table"], "omitted", f"Leave-one-out: {out}", theme)))
        if "cumulative" in figures and "table" in r.extras.get("cumulative", {}):
            t = r.extras["cumulative"]["table"]
            s.figures.append(("Cumulative meta-analysis", F.sensitivity_forest(r, t.assign(lbl=[f"+ {a}" for a in t["added"]]), "lbl", f"Cumulative meta-analysis: {out}", theme)))
        if "provenance" in figures:
            s.figures.append(("Data provenance", F.provenance_chart(r.prep, theme)))
        if "radial" in figures and len(r.studies) >= 3:
            s.figures.append(("Radial (Galbraith) plot", F.radial_plot(r, theme)))
        if "labbe" in figures and r.measure.family == "binary":
            fig = F.labbe_plot(r, theme)
            if fig is not None:
                s.figures.append(("L'Abbé plot", fig))
        if "baujat" in figures and "table" in r.extras.get("baujat", {}):
            s.figures.append(("Baujat plot", F.baujat_plot(r, theme)))
        if "rho" in figures and "rho_sensitivity" in r.extras.get("che_rve", {}):
            s.figures.append(("CHE-RVE sensitivity to ρ", F.dependent_rho_plot(r, theme)))
        if "extras" in tables:
            s.tables.extend(extra_tables(r))
        if "actions" in tables and not r.prep.actions.empty:
            s.tables.append(("Data-preparation actions", r.prep.actions))
        if "derivations" in tables and not r.prep.log.empty:
            lg = r.prep.log[r.prep.log["record_id"].isin(set(r.studies["record_id"]) | set(r.prep.data["record_id"]))]
            s.tables.append(("Derivation log", lg[["study", "calculated", "value", "type", "formula", "inputs", "citation"]]))
        notinc = r.prep.data[~r.prep.data["included"]]
        if not notinc.empty:
            s.tables.append(("Comparisons not included, with reasons", notinc[["label", "derivation_basis", "exclusion_reason"]]
                             .rename(columns={"label": "Study", "derivation_basis": "Effect size status", "exclusion_reason": "Reason"})))
        if "requests" in tables and not r.prep.requests.empty:
            s.tables.append(("Studies not analysed: data still needed", r.prep.requests.drop(columns=["fields_option_1"])))
        if "checks" in tables and not r.prep.checks.empty:
            s.tables.append(("Consistency checks", r.prep.checks[["study", "message"]]))
        secs.append(s)
    ev = getattr(ds, "g_evidence", None)
    if "evidence" in tables and isinstance(ev, pd.DataFrame) and not ev.empty:
        secs.append(Section("Reported Hedges' g: design check", text=[
            "Each reported Hedges' g was compared with the effect sizes MetaPy calculated from the same row's raw data. "
            "Where a study reported only g, the design shown by the matching rows was used after researcher confirmation."],
            tables=[("Evidence", ev.drop(columns=["record_id"]))]))
    if "parsing" in tables:
        s = Section("Data import")
        ms = getattr(ds, "merge_summary", {}) or {}
        if ms.get("duplicate_rows_merged"):
            s.text.append(f"{ms['rows_read']} rows were read; {ms['duplicate_rows_merged']} duplicate rows describing the same comparison in several "
                          f"worksheets were merged into {ms['comparisons_after_merge']} unique comparisons ({ms['conflicting_values']} conflicting values).")
        if not ds.outcome_map.empty:
            s.tables.append(("Outcomes and the scales grouped into them", ds.outcome_map))
        if not ds.conflicts.empty:
            s.tables.append(("Conflicting values between worksheets", ds.conflicts))
        s.tables.append(("Column mapping", ds.column_map))
        if not ds.parse_notes.empty:
            s.tables.append(("Text cells interpreted", ds.parse_notes))
        secs.append(s)
    for nr in nma_results or []:
        secs.append(nma_section(nr, figures, theme))
    return Report(title, secs, meta)


def nma_section(nr: dict[str, Any], figures: list[str], theme: str = "navy") -> Section:
    from . import nma as N
    res = nr["result"]
    con = res.contrasts
    lab = N.MEASURES_NMA.get(con.measure, con.measure)
    s = Section(f"Network meta-analysis: {nr.get('outcome', lab)}")
    Q = res.Q
    s.text.append(
        f"A frequentist network meta-analysis ({'random' if res.model == 'random' else 'common'}-effects, contrast-based generalised least squares; "
        f"Rücker & Schwarzer framework) compared {len(res.treatments)} treatments in {res.extra['n_studies']} studies "
        f"({res.extra['multi_arm_studies']} multi-arm). Measure: {lab}. Reference: {res.reference}. "
        f"Common between-study variance τ² = {res.tau2:.3f} (REML). Total Q = {Q['Q_total']:.2f} (df = {Q['df_total']}, p = {Q['p_total']:.3f}); "
        f"within-design heterogeneity Q = {Q['Q_heterogeneity']:.2f} (df = {Q['df_heterogeneity']}); between-design inconsistency "
        f"Q = {Q['Q_inconsistency']:.2f} (df = {Q['df_inconsistency']}" + (f", p = {Q['p_inconsistency']:.3f})." if Q['df_inconsistency'] > 0 else ")."))
    s.text.append("P-scores rank treatments by the mean certainty of being better than the others (Rücker & Schwarzer 2015); they ignore the "
                  "size of the differences and should be read alongside the league table. "
                  + ("Smaller values were treated as better." if res.small_values == "good" else "Larger values were treated as better."))
    if nr.get("notes"):
        s.text.append("Data notes: " + "; ".join(nr["notes"]))
    s.tables.append(("Network estimates against the reference", nr["vs_reference"].drop(columns=["_y", "_se"])))
    s.tables.append(("League table (row treatment vs column treatment)", nr["league"].reset_index(drop=True)))
    s.tables.append(("Treatment ranking (P-scores)", nr["pscores"]))
    if not nr["split"].empty:
        s.tables.append(("Direct vs indirect evidence (back-calculation)", nr["split"]))
    if "network" in figures:
        s.figures.append(("Network graph", N.network_plot(res)))
        s.figures.append(("Network forest plot", N.forest_vs_reference(res, nr.get("transform"), lab)))
    return s
