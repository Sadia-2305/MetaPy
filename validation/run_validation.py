import sys, json, math, time, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys as _sys
_sys.path[:0] = [str(REPO), str(REPO / "tests"), str(HERE)]
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from metapy import stats, advanced as AD, dependent as D, nma as N
from test_metapy import bcg_yv, BCG
import independent as I
out = {}
y, v = bcg_yv()
# --- 1. BCG vs metafor reference values (rma(..., measure="RR") examples in the metafor documentation)
re = stats.pool(y, v, "random", "REML", "Wald"); dl = stats.pool(y, v, "random", "DL", "Wald"); fe = stats.pool(y, v, "fixed", "REML", "Wald")
d = pd.DataFrame({"yi": y, "vi": v, "ablat": BCG["ablat"].to_numpy()})
mr = stats.meta_regression(d, ["ablat"], tau_method="REML", knha=False)
qp = AD.tau2_qprofile(y, v)
out["bcg"] = {"re_est": re["estimate"], "re_se": re["se"], "re_tau2": re["tau2"], "re_I2": re["I2"], "re_Q": re["Q"],
              "dl_tau2": dl["tau2"], "fe_est": fe["estimate"], "fe_se": fe["se"],
              "mr_b0": float(mr["coefficients"]["estimate"].iloc[0]), "mr_b1": float(mr["coefficients"]["estimate"].iloc[1]),
              "mr_tau2": mr["tau2_residual"], "qp_lo": qp["tau2_lo"], "qp_hi": qp["tau2_hi"]}
# --- 2. independent re-implementation (separate code base) on the BCG data
t2 = I.reml(y, v); p = I.pool(y, v, t2, hk=False)
out["indep_bcg"] = {"tau2": t2, "est": p["mu"], "diff_est": abs(p["mu"] - re["estimate"]), "diff_tau2": abs(t2 - re["tau2"])}
# --- 3. synthetic demo: MetaPy PPC2 vs independent PPC2 for every comparison with raw pre/post data
from metapy import MetaPySession
S = MetaPySession(); S.load(str(REPO / "examples" / "MetaPy_SYNTHETIC_demo.xlsx"))
pid = next(q["plan_id"] for q in S.scan_report.plans if q["outcome"].startswith("Anxiety") and q["measure"] == "ppc2")
S.select_plans([pid]); prep = S.prepare(pid)
raw = pd.read_excel(str(REPO / "examples" / "MetaPy_SYNTHETIC_demo.xlsx"), sheet_name="Anxiety")
diffs = []
dd = prep.data[prep.data["included"]]
for _, r in raw.iterrows():
    if any(pd.isna(r.get(c)) for c in ("Mean_Pre_I", "Mean_Post_I", "SD_Pre_I", "Mean_Pre_C", "Mean_Post_C", "SD_Pre_C")):
        continue
    g = I.ppc2(r["Mean_Pre_I"], r["Mean_Post_I"], r["SD_Pre_I"], r["N_I"], r["Mean_Pre_C"], r["Mean_Post_C"], r["SD_Pre_C"], r["N_C"])
    m = dd[dd["label"].astype(str).str.startswith(str(r["Study"]))]
    if len(m) == 1:
        sign = -1 if bool(m["direction_flipped"].iloc[0]) else 1
        diffs.append(abs(sign * float(m["yi"].iloc[0]) - g))
out["indep_ppc2"] = {"n": len(diffs), "max_abs_diff": max(diffs) if diffs else None}
# --- 4. multilevel model reduces to RE with one effect per cluster
f = D.fit_dependent(pd.DataFrame({"study_id": range(len(y)), "yi": y, "vi": v}))
out["multilevel_vs_re"] = {"est_diff": abs(f.estimate - re["estimate"]), "var_diff": abs(f.sigma2_between + f.sigma2_within - re["tau2"])}
# --- 5. NMA checks

import test_v5 as T
arm = T._arm_data()
fe_n = N.fit_nma(N.contrasts_from_arms(arm, "MD"), "A", "fixed")
studies, trts = sorted(arm["study"].unique()), ["B", "C", "D"]
X = np.zeros((len(arm), len(studies) + 3))
for i_, (s_, t_) in enumerate(zip(arm["study"], arm["treatment"])):
    X[i_, studies.index(s_)] = 1
    if t_ != "A": X[i_, len(studies) + trts.index(t_)] = 1
W = np.diag(1 / (arm["sd"] ** 2 / arm["n"]).to_numpy()); b = np.linalg.solve(X.T @ W @ X, X.T @ W @ arm["mean"].to_numpy())
se_a = np.sqrt(np.diag(np.linalg.inv(X.T @ W @ X)))[-3:]
star = N.fit_nma(N.contrasts_from_table(pd.DataFrame({"study": [f"T{i}" for i in range(len(y))], "treat1": "V", "treat2": "C", "TE": y, "seTE": np.sqrt(v)})), "C", "random")
tri = N.fit_nma(N.contrasts_from_table(pd.DataFrame({"study": ["a", "b", "c"], "treat1": ["B", "C", "C"], "treat2": ["A", "A", "B"], "TE": [-0.5, -0.8, -0.3], "seTE": [0.2] * 3})), "A", "fixed")
out["nma"] = {"armbased_max_diff_est": float(np.max(np.abs(fe_n.d - b[-3:]))), "armbased_max_diff_se": float(np.max(np.abs(np.sqrt(np.diag(fe_n.cov)) - se_a))),
              "star_est_diff": abs(float(star.d[0]) - re["estimate"]), "star_tau2_diff": abs(star.tau2 - re["tau2"]), "triangle_Qinc": tri.Q["Q_inconsistency"],
              "n_studies": int(arm["study"].nunique()), "n_treat": 4}
# --- 6. Bayesian model vs brute-force 2D grid
from scipy.stats import halfnorm, norm
rb = AD.bayes_re(y, v, 0.5, 10)
mus, taus = np.linspace(-2, 0.6, 700), np.linspace(1e-4, 2.5, 700); M, Tt = np.meshgrid(mus, taus, indexing="ij")
lp = sum(norm.logpdf(yi, M, np.sqrt(vi + Tt ** 2)) for yi, vi in zip(y, v)) + norm.logpdf(M, 0, 10) + halfnorm.logpdf(Tt, scale=0.5)
pp = np.exp(lp - lp.max()); pm = pp.sum(1); pm /= pm.sum(); pt = pp.sum(0); pt /= pt.sum()
out["bayes"] = {"mu_med": rb["mu_median"], "mu_med_grid": float(np.interp(0.5, np.cumsum(pm), mus)), "tau_med": rb["tau_median"],
                "tau_med_grid": float(np.interp(0.5, np.cumsum(pt), taus)), "mu_lo": rb["mu_lo"], "mu_hi": rb["mu_hi"]}
# --- 7. 3PSM without selection = ML random effects (independent ML)
from scipy.optimize import minimize_scalar
s3 = AD.selection_3psm(y, v)
def nll(t2_):
    w = 1 / (v + t2_); mu = np.sum(w * y) / w.sum(); return 0.5 * np.sum(np.log(v + t2_) + w * (y - mu) ** 2)
t2ml = minimize_scalar(nll, bounds=(0, 5), method="bounded").x; w = 1 / (v + t2ml)
out["psm"] = {"unadj": s3["unadjusted_ml_estimate"], "ml_indep": float(np.sum(w * y) / w.sum()), "adj": s3["adjusted_estimate"], "p": s3["p_selection"]}
# --- 8. PET equals independent weighted least squares
pet = AD.pet_peese(y, v); Xp = np.column_stack([np.ones(len(y)), np.sqrt(v)]); wsq = np.sqrt(1 / v)
bp = np.linalg.lstsq(Xp * wsq[:, None], y * wsq, rcond=None)[0]
out["pet"] = {"diff": abs(pet["PET"]["estimate"] - bp[0])}
# --- 9. CR2 coverage simulation
t0 = time.time(); reps = 300; hits = 0; dfs = []
for s_ in range(reps):
    df = T._three_level(seed=1000 + s_, n_st=15)
    rob = D.che_rve(df, rho=0.0)["robust"].iloc[0]
    hits += rob["ci_lo"] <= 0.3 <= rob["ci_hi"]; dfs.append(rob["df (Satterthwaite)"])
cov = hits / reps
out["cr2"] = {"reps": reps, "coverage": cov, "mcse": math.sqrt(cov * (1 - cov) / reps), "median_df": float(np.median(dfs)), "seconds": time.time() - t0}
json.dump(out, open(str(HERE / "results" / "validation.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
