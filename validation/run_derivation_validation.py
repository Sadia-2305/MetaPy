import sys, json, math, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys as _sys
_sys.path[:0] = [str(REPO), str(REPO / "tests"), str(HERE)]
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import independent as I
from metapy import MetaPySession
rng = np.random.default_rng(20261116)
res = {}

def run(df, measure):
    p = str(HERE / "results" / "_tmp.csv"); df.to_csv(p, index=False)
    s = MetaPySession(); s.load(p)
    pid = next(q["plan_id"] for q in s.scan_report.plans if q["measure"] == measure)
    s.select_plans([pid]); s.prepare(pid); s.apply_pending_answers([pid])
    d = s.preps[pid].data
    return d[d["included"]].set_index("study_id")[["yi", "vi"]], s.preps[pid]

# A) pre/post data -> Morris PPC2 (r imputed as 0.5)
rows = []
for k in range(40):
    ni, nc = int(rng.integers(15, 120)), int(rng.integers(15, 120)); base = rng.normal(20, 5)
    rows.append(dict(Study=f"P{k+1}", N_I=ni, N_C=nc, Mean_Pre_I=base + rng.normal(0, 1), SD_Pre_I=abs(rng.normal(6, 1)),
                     Mean_Post_I=base - rng.normal(2, 2), SD_Post_I=abs(rng.normal(6, 1)), Mean_Pre_C=base + rng.normal(0, 1),
                     SD_Pre_C=abs(rng.normal(6, 1)), Mean_Post_C=base - rng.normal(0.5, 2), SD_Post_C=abs(rng.normal(6, 1))))
A = pd.DataFrame(rows).round(3)
got, prep = run(A, "ppc2")
dg, dv = [], []
for _, r in A.iterrows():
    g = I.ppc2(r.Mean_Pre_I, r.Mean_Post_I, r.SD_Pre_I, r.N_I, r.Mean_Pre_C, r.Mean_Post_C, r.SD_Pre_C, r.N_C)
    v = I.ppc2_var(g, r.N_I, r.N_C, 0.5)
    m = got.loc[r.Study]; dg.append(abs(m.yi - g)); dv.append(abs(m.vi - v))
res["ppc2"] = {"k": len(A), "included": int(len(got)), "max_diff_g": max(dg), "max_diff_v": max(dv)}

# B) endpoint means/SDs -> Hedges' g (Borenstein variance)
rows = []
for k in range(40):
    ni, nc = int(rng.integers(10, 150)), int(rng.integers(10, 150))
    rows.append(dict(Study=f"E{k+1}", N_I=ni, N_C=nc, Mean_Post_I=rng.normal(18, 3), SD_Post_I=abs(rng.normal(5, 1)),
                     Mean_Post_C=rng.normal(20, 3), SD_Post_C=abs(rng.normal(5, 1))))
B = pd.DataFrame(rows).round(3)
got, _ = run(B, "smd_g")
dg, dv = [], []
for _, r in B.iterrows():
    g = I.endpoint_g(r.Mean_Post_I, r.SD_Post_I, r.N_I, r.Mean_Post_C, r.SD_Post_C, r.N_C); v = I.var_g_ind(g, r.N_I, r.N_C)
    m = got.loc[r.Study]; dg.append(abs(m.yi - g)); dv.append(abs(m.vi - v))
res["endpoint"] = {"k": len(B), "included": int(len(got)), "max_diff_g": max(dg), "max_diff_v": max(dv)}

# C) only MD with 95% CI (text cell) + group sizes -> SE -> pooled SD -> d -> g (multi-step chain)
rows, truth = [], []
for k in range(30):
    ni, nc = int(rng.integers(15, 120)), int(rng.integers(15, 120)); md = rng.normal(-3, 2); sdp = abs(rng.normal(8, 1.5))
    se = sdp * math.sqrt(1 / ni + 1 / nc); lo, hi = md - 1.96 * se, md + 1.96 * se
    rows.append({"Study": f"M{k+1}", "N_I": ni, "N_C": nc, "MD (95% CI)": f"{md:.2f} ({lo:.2f} to {hi:.2f})"})
C = pd.DataFrame(rows)
got, prepC = run(C, "smd_g")
dg, dv = [], []
for _, r in C.iterrows():
    md, lo, hi = [float(x) for x in r["MD (95% CI)"].replace("(", "").replace(")", "").replace(" to ", " ").split()]
    se = (hi - lo) / (2 * 1.959963984540054); sdp = se / math.sqrt(1 / r.N_I + 1 / r.N_C); g = I.J(r.N_I + r.N_C - 2) * md / sdp; v = I.var_g_ind(g, r.N_I, r.N_C)
    m = got.loc[r.Study]; dg.append(abs(m.yi - g)); dv.append(abs(m.vi - v))
chain = prepC.log[prepC.log["record_id"] == prepC.log["record_id"].iloc[0]]
res["md_ci_chain"] = {"k": len(C), "included": int(len(got)), "max_diff_g": max(dg), "max_diff_v": max(dv), "steps_logged_first_study": int(len(chain)),
                      "steps": list(chain["calculated"]) if "calculated" in chain else list(chain.columns)}
json.dump(res, open(str(HERE / "results" / "validation_derivation.json"), "w"), indent=1, default=float)
A.to_csv(str(HERE / "data" / "sim_prepost.csv"), index=False); B.to_csv(str(HERE / "data" / "sim_endpoint.csv"), index=False); C.to_csv(str(HERE / "data" / "sim_md_ci.csv"), index=False)
print(json.dumps(res, indent=1, default=float))
