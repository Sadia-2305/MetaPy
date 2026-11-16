import sys, warnings, json; warnings.filterwarnings("ignore"); 
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys as _sys
_sys.path[:0] = [str(REPO), str(REPO / "tests"), str(HERE)]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from metapy import MetaPySession, figures as F
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
INK, MUTED, LINE, FILL = "#1B2A4A", "#5F6B7A", "#1B2A4A", "#F2F4F7"

def save(fig, stem, tiff):
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

def box(ax, x, y, w, h, title, body, fc="white", bold=True, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=LINE, lw=1.0))
    ax.text(x + 0.12, y + h - 0.16, title, ha="left", va="top", fontsize=fs + 0.6, fontweight="bold" if bold else "normal", color=INK)
    ax.text(x + 0.12, y + h - 0.48, body, ha="left", va="top", fontsize=fs, color=INK, linespacing=1.35)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.1, mutation_scale=11))

# ---------------- Figure 1: workflow
fig, ax = plt.subplots(figsize=(11.5, 3.9)); ax.set_xlim(0, 23); ax.set_ylim(2.6, 10); ax.axis("off")
steps = [
    ("1 Read & scan", "All sheets; header\ndetection; text cells\nsuch as '12.3 (4.1)';\noutcome grouping;\ndesign check of\nreported g"),
    ("2 Choose plans", "Every outcome x\neffect measure,\nranked by usable\nstudies and fewest\napproximations"),
    ("3 Derive", "180 cited rules;\nexact, then\napproximation, then\nassumption; logged\nprovenance"),
    ("4 Decide", "Option questions;\nveto groups of\napproximations;\nverified data\nrequests; approval"),
    ("5 Analyse", "RE/FE models,\nmeta-regression,\ndependent effects,\nNMA, bias and\nBayesian models"),
    ("6 Review", "Accept, add\nanalyses, change\nmodel, or return\nto step 4"),
    ("7 Report", "Tables, figures,\nPRISMA 2020,\nmethods text;\nWord, Excel, PDF,\nHTML, JSON record"),
]
w, h, gap, x0, ytop = 2.95, 3.55, 0.28, 0.2, 9.7
xs = [x0 + i * (w + gap) for i in range(len(steps))]
for i, (t, b_) in enumerate(steps):
    box(ax, xs[i], ytop - h, w, h, t, b_, fc=FILL if i in (3, 5) else "white", fs=7.4)
    if i < len(steps) - 1:
        arrow(ax, xs[i] + w, ytop - h / 2, xs[i + 1], ytop - h / 2)
ax.text(xs[3] + w / 2, ytop - h - 0.25, "shaded: researcher decisions", ha="center", va="top", fontsize=7.4, color=MUTED, style="italic")
yb = 4.85
ax.text(0.2, yb + 0.2, "Step 3 in detail: forward-chaining solver with evidence tiers", fontsize=9, fontweight="bold", color=INK, va="bottom")
tiers = [("1  Exact rules", "applied until nothing new can be derived:\nSE from 95% CI, SD from SE, d from t or p,\nMorris PPC2, log OR from a 2x2 table"),
         ("2  Approximations", "one at a time, then exact rules again:\nmean/SD from median and IQR, OR <-> SMD,\nSpearman -> Pearson; each group can be vetoed"),
         ("3  Assumptions", "only when the researcher switches them on:\nimputed pre-post r, equal split of a total N,\ntype of an unlabelled SMD")]
tw = 7.25
for i, (t, b_) in enumerate(tiers):
    xx = 0.2 + i * (tw + 0.35)
    box(ax, xx, yb - 1.95, tw, 1.95, t, b_, fc=["white", FILL, "#E1E5EB"][i], fs=7.6)
    if i < 2:
        arrow(ax, xx + tw, yb - 0.97, xx + tw + 0.35, yb - 0.97)
save(fig, str(HERE / "figures" / "fig1_workflow"), str(HERE / "figures" / "Figure1.tiff"))

# ---------------- Figure 2: one logged derivation chain (worked example, study with MD and 95% CI only)
s = MetaPySession(); s.load(str(HERE / "data" / "MetaPy_worked_example_mixed.xlsx"))
pid = "Anxiety :: g_best"; s.select_plans([pid]); p = s.prepare(pid)
raw = pd.read_excel(str(HERE / "data" / "MetaPy_worked_example_mixed.xlsx"))
mrows = p.data[p.data["label"].astype(str).str.startswith("M")].copy()
pick = mrows.loc[mrows["yi"].abs().idxmax()]
rid, lab = pick["record_id"], pick["label"]
log = p.log[p.log["record_id"] == rid].set_index("calculated")
src = raw[raw["Study"] == lab].iloc[0]
def L(name): return log.loc[name]
fig, ax = plt.subplots(figsize=(11.5, 2.3)); ax.set_xlim(0, 23); ax.set_ylim(2.2, 6.5); ax.axis("off")
cells = [
    ("Data file", f"cell 'MD (95% CI)':\n{src['MD (95% CI)']}\nN_I = {src['N_I']}, N_C = {src['N_C']}\n\nparsed into MD,\nlower and upper limit"),
    ("SE of MD", f"= {L('SE of mean difference (endpoint)')['value']:.3f}\n\n(upper - lower)\n/ (2 x 1.959964)\nCochrane Handbook\n6.3.1"),
    ("Pooled SD", f"= {L('Pooled SD (endpoint)')['value']:.3f}\n\nSE / sqrt(1/n_I\n+ 1/n_C)\nCochrane Handbook\n6.5.2.3"),
    ("Cohen's d", f"= {L(chr(67)+'ohen' + chr(39) + 's d (endpoint)')['value']:.4f}\n\nMD / pooled SD\n\nBorenstein et al.\n2009, eq. 4.18"),
    ("Hedges' g", f"= {L(chr(72)+'edges' + chr(39) + ' g (endpoint)')['value']:.4f}\n\nJ x d,\nJ = 1 - 3/(4df - 1)\nHedges 1981"),
    ("Variance of g", f"= {L('Variance of g (endpoint)')['value']:.4f}\n\nJ\u00b2 x V(d)\n\nBorenstein et al.\n2009, eqs. 4.20, 4.24"),
]
w, h, gap = 3.45, 3.9, 0.4
for i, (t, b) in enumerate(cells):
    x = 0.2 + i * (w + gap)
    box(ax, x, 6.3 - h, w, h, t, b, fc=FILL if i == 0 else "white", fs=7.6)
    if i < len(cells) - 1:
        arrow(ax, x + w, 6.3 - h / 2, x + w + gap, 6.3 - h / 2)
save(fig, str(HERE / "figures" / "fig2_derivation_chain"), str(HERE / "figures" / "Figure2.tiff"))
json.dump({"study": lab, "values": {k: float(v) for k, v in log["value"].items()}, "cell": src["MD (95% CI)"], "n": [int(src["N_I"]), int(src["N_C"])]},
          open(str(HERE / "results" / "figure2_values.json"), "w"), indent=1)

# ---------------- Figure 3: BCG forest plot drawn by MetaPy
bcg = pd.DataFrame({
    "Study": ["Aronson", "Ferguson & Simes", "Rosenthal et al", "Hart & Sutherland", "Frimodt-Moller et al", "Stein & Aronson", "Vandiviere et al",
              "TPT Madras", "Coetzee & Berjak", "Rosenthal et al (2)", "Comstock et al", "Comstock & Webster", "Comstock et al (2)"],
    "Year": [1948, 1949, 1960, 1977, 1973, 1953, 1973, 1980, 1968, 1961, 1974, 1969, 1976],
    "tpos": [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27], "tneg": [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886],
    "cpos": [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29], "cneg": [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825],
    "ablat": [44, 55, 42, 52, 13, 44, 19, 13, 27, 42, 18, 33, 33]})
from test_metapy import BCG
assert (BCG[["tpos", "tneg", "cpos", "cneg", "ablat"]].to_numpy() == bcg[["tpos", "tneg", "cpos", "cneg", "ablat"]].to_numpy()).all()
out = pd.DataFrame({"Study": bcg["Study"] + " " + bcg["Year"].astype(str), "Year": bcg["Year"], "Outcome": "Tuberculosis",
                    "Events_I": bcg["tpos"], "N_I": bcg["tpos"] + bcg["tneg"], "Events_C": bcg["cpos"], "N_C": bcg["cpos"] + bcg["cneg"], "Latitude": bcg["ablat"]})
out.to_csv(str(HERE / "data" / "bcg.csv"), index=False)
s = MetaPySession(); s.load(str(HERE / "data" / "bcg.csv"))
pid = next(q["plan_id"] for q in s.scan_report.plans if q["measure"] == "log_rr"); s.select_plans([pid]); s.approve_many([pid])
r = s.analyse({"model": "random", "tau_method": "REML", "ci_method": "HK", "extras": []})[pid]
json.dump({k: r.main[k] for k in ("estimate", "ci_lo", "ci_hi", "tau2", "I2", "pi_lo", "pi_hi", "Q", "p")}, open(str(HERE / "results" / "bcg_hk.json"), "w"), indent=1, default=float)
fig = F.forest_plot(r, "monochrome"); save(fig, str(HERE / "figures" / "fig3_bcg_forest"), str(HERE / "figures" / "Figure3.tiff"))
print("ok", lab, json.load(open(str(HERE / "results" / "bcg_hk.json"))))

# ---------------- numbers for the BCG worked example (Section 5.1)
import math
from metapy import stats
mr = stats.meta_regression(r.studies.assign(Latitude=out["Latitude"].to_numpy()), ["Latitude"], tau_method="REML", knha=True)
c = mr["coefficients"]
hk = json.load(open(str(HERE / "results" / "bcg_hk.json")))
json.dump({"hk": hk, "hk_exp": {k: math.exp(hk[k]) for k in ("estimate", "ci_lo", "ci_hi", "pi_lo", "pi_hi")},
           "mr_knha": {"b1": float(c["estimate"].iloc[1]), "lo": float(c["ci_lo"].iloc[1]), "hi": float(c["ci_hi"].iloc[1]),
                       "p": float(c["p"].iloc[1]), "tau2": mr["tau2_residual"], "R2": mr["R2_percent"], "QE": mr["QE"], "QE_p": mr["QE_p"]}},
          open(str(HERE / "results" / "bcg_numbers.json"), "w"), indent=1, default=float)
