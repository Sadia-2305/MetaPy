"""BCG data: MetaPy against the outputs printed in the metafor documentation (README, test = "knha"), and against
the independent implementation for quantities not shown there. Writes results/reference_checks.json."""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, t as tdist

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import independent as I  # noqa: E402
from metapy import advanced as AD, stats  # noqa: E402

bcg = pd.read_csv(HERE / "data" / "bcg.csv")
a, n1, c, n2 = bcg["Events_I"], bcg["N_I"], bcg["Events_C"], bcg["N_C"]
y = np.log((a / n1) / (c / n2)).to_numpy()
v = (1 / a - 1 / n1 + 1 / c - 1 / n2).to_numpy()
k = len(y)

# --- metafor README outputs (rma(yi, vi, data = dat, test = "knha"); predict(res, transf = exp); mods = ~ ablat)
METAFOR = {"est": -0.7145, "se": 0.1808, "ci_lo": -1.1084, "ci_hi": -0.3206, "p": 0.0019, "tau2": 0.3132, "tau": 0.5597,
           "I2": 92.22, "H2": 12.86, "Q": 152.2330, "pred": 0.49, "pred_lo": 0.33, "pred_hi": 0.73, "pi_lo": 0.14, "pi_hi": 1.76,
           "mr_tau2": 0.0764, "mr_R2": 75.62, "mr_QE": 30.7331, "mr_F": 12.5905, "mr_F_p": 0.0046,
           "mr_b0": 0.2515, "mr_b0_se": 0.2839, "mr_b0_lo": -0.3735, "mr_b0_hi": 0.8764,
           "mr_b1": -0.0291, "mr_b1_se": 0.0082, "mr_b1_lo": -0.0472, "mr_b1_hi": -0.0111}
re = stats.pool(y, v, "random", "REML", "HK")
mr = stats.meta_regression(pd.DataFrame({"yi": y, "vi": v, "ablat": bcg["Latitude"]}), ["ablat"], tau_method="REML", knha=True)
cf = mr["coefficients"]
w0 = 1 / v
s2 = (k - 1) * w0.sum() / (w0.sum() ** 2 - np.sum(w0 ** 2))
pse = math.sqrt(re["tau2"] + re["se"] ** 2)
META = {"est": re["estimate"], "se": re["se"], "ci_lo": re["ci_lo"], "ci_hi": re["ci_hi"], "p": re["p"], "tau2": re["tau2"],
        "tau": math.sqrt(re["tau2"]), "I2": re["I2"], "H2": (re["tau2"] + s2) / s2, "Q": re["Q"],
        "pred": math.exp(re["estimate"]), "pred_lo": math.exp(re["ci_lo"]), "pred_hi": math.exp(re["ci_hi"]),
        "pi_lo": math.exp(re["pi_lo"]), "pi_hi": math.exp(re["pi_hi"]),
        "pi_lo_k1": math.exp(re["estimate"] - tdist.ppf(0.975, k - 1) * pse), "pi_hi_k1": math.exp(re["estimate"] + tdist.ppf(0.975, k - 1) * pse),
        "mr_tau2": mr["tau2_residual"], "mr_R2": mr["R2_percent"], "mr_QE": mr["QE"], "mr_F": mr["QM"], "mr_F_p": mr["QM_p"],
        "mr_b0": cf["estimate"].iloc[0], "mr_b0_se": cf["se"].iloc[0], "mr_b0_lo": cf["ci_lo"].iloc[0], "mr_b0_hi": cf["ci_hi"].iloc[0],
        "mr_b1": cf["estimate"].iloc[1], "mr_b1_se": cf["se"].iloc[1], "mr_b1_lo": cf["ci_lo"].iloc[1], "mr_b1_hi": cf["ci_hi"].iloc[1]}

# exact REML optimum of the meta-regression residual tau^2 (independent, tolerance 1e-14)
X = np.column_stack([np.ones(k), bcg["Latitude"].to_numpy(float)])
def nreml(t2):
    W = np.diag(1 / (v + t2)); XtWX = X.T @ W @ X; b = np.linalg.solve(XtWX, X.T @ W @ y); r = y - X @ b
    return 0.5 * (np.sum(np.log(v + t2)) + np.linalg.slogdet(XtWX)[1] + r @ W @ r)
exact_mr_tau2 = float(minimize_scalar(nreml, bounds=(0, 1), method="bounded", options={"xatol": 1e-14}).x)

# --- independent implementation for quantities not printed in the metafor README
dl_i = I.dl(y, v)
fe_i = I.pool(y, v, 0.0, hk=False)
def genq(t2):
    w = 1 / (v + t2); mu = np.sum(w * y) / w.sum(); return np.sum(w * (y - mu) ** 2)
qp_i = (brentq(lambda t: genq(t) - chi2.ppf(0.975, k - 1), 0, 10, xtol=1e-14), brentq(lambda t: genq(t) - chi2.ppf(0.025, k - 1), 0, 50, xtol=1e-14))
dl_m = stats.pool(y, v, "random", "DL", "Wald")["tau2"]
fe_m = stats.pool(y, v, "fixed", "REML", "Wald")
qp_m = AD.tau2_qprofile(y, v)
INDEP = {"dl_tau2": (dl_m, dl_i), "fe_est": (fe_m["estimate"], fe_i["mu"]), "fe_se": (fe_m["se"], fe_i["se"]),
         "qp_lo": (qp_m["tau2_lo"], qp_i[0]), "qp_hi": (qp_m["tau2_hi"], qp_i[1])}
out = {"metafor": METAFOR, "metapy": META, "exact_mr_tau2": exact_mr_tau2, "independent": INDEP}
json.dump(out, open(HERE / "results" / "reference_checks.json", "w"), indent=1, default=float)
for key, ref in METAFOR.items():
    print(f"{key:9s} metafor {ref:>9}  MetaPy {META[key]:.5f}")
print("PI with k-1 df:", META["pi_lo_k1"], META["pi_hi_k1"], "| exact MR tau2:", exact_mr_tau2)
print({k_: (round(a_, 6), round(b_, 6)) for k_, (a_, b_) in INDEP.items()})
