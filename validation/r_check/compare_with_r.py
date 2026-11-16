"""Compare MetaPy 5.1 with R (output of metafor_check.R).

Usage:  python validation/r_check/compare_with_r.py metafor_results.json
Writes validation/results/r_comparison.csv and r_comparison_summary.json and prints a summary.
Without an argument it only computes the MetaPy side (useful to check that everything runs).
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from metapy import advanced as AD, advmodels as AM, dependent as D, extended as E, glmm as G, metareg as MR  # noqa: E402
from metapy import nma as N, stats  # noqa: E402

DATA = ROOT / "validation" / "data"


def bcg():
    d = pd.read_csv(DATA / "bcg.csv")
    a, n1, c, n2 = d.Events_I.values, d.N_I.values, d.Events_C.values, d.N_C.values
    d["yi"] = np.log((a / n1) / (c / n2))
    d["vi"] = 1 / a - 1 / n1 + 1 / c - 1 / n2
    return d


def metapy_side() -> dict:
    d = bcg()
    y, v = d.yi.to_numpy(float), d.vi.to_numpy(float)
    a, n1, c, n2 = d.Events_I.values, d.N_I.values, d.Events_C.values, d.N_C.values
    m: dict = {}
    m["tau2"] = {}
    for meth in ["REML", "ML", "DL", "PM", "EB", "HE", "HS", "HSk", "SJ"]:
        t2 = E.estimate_tau2(y, v, meth)
        w = 1 / (v + t2)
        m["tau2"][meth] = {"tau2": t2, "beta": float(np.sum(w * y) / np.sum(w)), "se": math.sqrt(1 / np.sum(w))}
    q = AD.tau2_qprofile(y, v)
    m["qprofile"] = {"tau2_lo": q["tau2_lo"], "tau2_hi": q["tau2_hi"], "I2_lo": q["I2_lo"], "I2_hi": q["I2_hi"]}
    m["mh"] = {k: {kk: E.mantel_haenszel(a, n1 - a, c, n2 - c, k)[src] for kk, src in (("beta", "estimate"), ("se", "se"), ("QE", "QE"))}
               for k in ("OR", "RR", "RD")}
    p = E.peto(a, n1 - a, c, n2 - c)
    m["peto"] = {"beta": p["estimate"], "se": p["se"], "QE": p["QE"]}
    g = G.logistic_fixed_study(a, n1, c, n2)
    m["glmm_or"] = {"beta": g["estimate"], "se": g["se"], "tau2": g["tau2"]}
    g = G.binomial_normal(a, n1)
    m["glmm_plo"] = {"beta": g["estimate"], "se": g["se"], "tau2": g["tau2"]}
    inf = E.influence_full(y, v, d.Study)
    m["influence"] = {c_: inf[c_].tolist() for c_ in ("rstudent", "dffits", "cook_d", "cov_r", "tau2_del", "QE_del", "hat", "dfbetas")}
    t = E.tes(y, v)
    m["tes"] = {"O": t["O"], "E": t["E"], "pval": t["p"], "theta": t["theta"]}
    tf = stats.trim_and_fill(y, v, "random", "REML", "Wald")
    m["trimfill"] = {"k0": tf["k0"], "beta": tf["adjusted"]["estimate"], "se": tf["adjusted"]["se"]}
    e = stats.egger(y, v)
    m["egger"] = {"t": e["t"], "p": e["p"]}
    b = stats.begg(y, v)
    m["begg"] = {"tau": b.get("tau", b.get("kendall_tau")), "p": b["p"]}
    f = MR.fit_df(d, ["Latitude*Year"], test="z")
    m["metareg_interaction"] = {"beta": f.beta.tolist(), "se": np.sqrt(np.diag(f.vcov)).tolist(), "tau2": f.tau2,
                                "QM": MR.omnibus(f)["statistic"], "QM_lat": MR.test_terms(f, "Latitude")["statistic"]}
    f = MR.fit_df(d, ["Latitude"], test="knha")
    pr = MR.predict(f, [{"Latitude": 10}, {"Latitude": 50}])
    m["predict"] = {"pred": pr.pred.tolist(), "se": pr.se.tolist(), "ci_lb": pr.ci_lo.tolist(), "ci_ub": pr.ci_hi.tolist(),
                    "pi_lb": pr.pi_lo.tolist(), "pi_ub": pr.pi_hi.tolist()}
    f = MR.fit_df(d, ["Latitude"], test="z")
    rb = MR.robust(f, d.Study)
    m["robust_cr2"] = {"beta": rb.estimate.tolist(), "se": rb["se (CR2)"].tolist(), "df": rb["df (Satterthwaite)"].tolist()}
    s = pd.read_csv(DATA / "selection_example.csv")
    s1 = E.selection_stepfun(s.yi, s.vi, [0.025])
    s3 = E.selection_stepfun(s.yi, s.vi, [0.025, 0.05, 0.5])
    m["selection"] = {"one": {"beta": s1["beta"], "se": s1["se_beta"], "tau2": s1["tau2"], "delta": s1["delta"], "LRT": s1["LRT"]},
                      "three": {"beta": s3["beta"], "se": s3["se_beta"], "tau2": s3["tau2"], "delta": s3["delta"], "LRT": s3["LRT"]}}
    l_ = pd.read_csv(DATA / "locscale_example.csv")
    ls = AM.location_scale(l_.yi, l_.vi, np.ones((len(l_), 1)), np.column_stack([np.ones(len(l_)), l_.z]), ["Intercept"], ["Intercept", "z"])
    m["location_scale"] = {"beta": ls["location"].estimate.tolist(), "se": ls["location"].se.tolist(),
                           "alpha": ls["scale"].estimate.tolist(), "se_alpha": ls["scale"].se.tolist()}
    n_ = pd.read_csv(DATA / "nested_example.csv")
    ml = AM.multilevel(n_, ["country", "study", "es"])
    m["nested"] = {"sigma2": ml["components"].sigma2.tolist(), "beta": ml["coefficients"].estimate.tolist(),
                   "se": ml["coefficients"].se.tolist()}
    dd = pd.read_csv(DATA / "dependent_example.csv").rename(columns={"study": "study_id"})
    tl = AM.multilevel(dd, ["study_id", "es"])
    m["threelevel"] = {"sigma2": tl["components"].sigma2.tolist(), "beta": tl["coefficients"].estimate.tolist(),
                       "se": tl["coefficients"].se.tolist()}
    ch = D.che_rve(dd, "study_id", ["dose"], 0.6)
    m["che_cr2"] = {"sigma2": [ch["fit"].sigma2_between, ch["fit"].sigma2_within], "beta": ch["robust"].estimate.tolist(),
                    "se": ch["robust"]["se (CR2)"].tolist(), "df": ch["robust"]["df (Satterthwaite)"].tolist()}
    arm = pd.read_csv(DATA / "nma_arms.csv")
    con = N.contrasts_from_arms(arm, "MD")
    fe, re_ = N.fit_nma(con, "A", "fixed"), N.fit_nma(con, "A", "random")
    tr = ["B", "C", "D"]
    m["nma"] = {"TE_common": [fe.effect(t_, "A")[0] for t_ in tr], "se_common": [fe.effect(t_, "A")[1] for t_ in tr],
                "TE_random": [re_.effect(t_, "A")[0] for t_ in tr], "se_random": [re_.effect(t_, "A")[1] for t_ in tr],
                "tau2": re_.tau2, "Q": fe.Q.get("Q_total"), "Q_heterogeneity": fe.Q.get("Q_heterogeneity"), "Q_inconsistency": fe.Q.get("Q_inconsistency")}
    bz = AD.bayes_re(y, v, tau_prior_scale=0.5, mu_prior_sd=10.0)
    # bayesmeta reports the shortest 95% interval, so the shortest MetaPy interval is compared
    m["bayes"] = {"mu_median": bz["mu_median"], "mu_lo": bz["mu_lo_shortest"], "mu_hi": bz["mu_hi_shortest"], "tau_median": bz["tau_median"]}
    return m


def _flat(prefix, x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            _flat(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _flat(f"{prefix}[{i + 1}]", v, out)
    elif isinstance(x, (int, float, np.floating, np.integer)) and x is not None:
        out[prefix] = float(x)


def compare(mp: dict, r: dict) -> pd.DataFrame:
    fm, fr = {}, {}
    _flat("", mp, fm)
    _flat("", {k: v for k, v in r.items() if k != "versions"}, fr)
    for k in list(fr):                      # R returns length-1 vectors as scalars
        if k not in fm and f"{k}[1]" in fm and f"{k}[2]" not in fm:
            fm[k] = fm.pop(f"{k}[1]")
    rows = []
    for k in sorted(set(fm) & set(fr)):
        a, b = fm[k], fr[k]
        diff = abs(a - b)
        rel = diff / max(abs(b), 1e-8)
        rows.append({"block": k.split(".")[0], "quantity": k, "MetaPy": a, "R": b, "abs_diff": diff, "rel_diff": rel})
    t = pd.DataFrame(rows)
    missing = sorted(set(fr) - set(fm))
    if missing:
        print("In R output but not compared:", ", ".join(missing[:20]), "..." if len(missing) > 20 else "")
    return t


def main():
    mp = metapy_side()
    out_dir = ROOT / "validation" / "results"
    (out_dir / "metapy_side_for_r.json").write_text(json.dumps(mp, indent=1, default=float))
    if len(sys.argv) < 2:
        print("MetaPy side computed for", len(mp), "blocks; pass metafor_results.json to compare.")
        return
    r = json.loads(Path(sys.argv[1]).read_text())
    t = compare(mp, r)
    errors = {k: v["error"] for k, v in r.items() if isinstance(v, dict) and "error" in v}
    t.to_csv(out_dir / "r_comparison.csv", index=False)
    summ = t.groupby("block").agg(n=("quantity", "size"), max_abs_diff=("abs_diff", "max"), max_rel_diff=("rel_diff", "max")).reset_index()
    (out_dir / "r_comparison_summary.json").write_text(json.dumps({"versions": r.get("versions"), "errors_in_R": errors,
                                                                   "summary": summ.to_dict("records")}, indent=1, default=float))
    pd.set_option("display.width", 160)
    print(summ.to_string(index=False))
    if errors:
        print("R blocks that failed:", errors)


if __name__ == "__main__":
    main()
