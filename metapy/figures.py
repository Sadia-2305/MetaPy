"""Publication figures: forest, subgroup forest, funnel, bubble, leave-one-out,
cumulative, influence, data-provenance chart and the PRISMA 2020 flow diagram."""
from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon

THEMES = {
    "navy": ("#1F3A5F", "#C0392B", "#7F8C8D"), "teal": ("#0F6E56", "#B9480F", "#7F8C8D"),
    "purple": ("#4B3B8F", "#B03060", "#7F8C8D"), "monochrome": ("#222222", "#222222", "#888888"),
    "orange": ("#B85C00", "#1F3A5F", "#7F8C8D"),
}
BASIS_MARK = {"Approximation": "†", "Assumption": "‡"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def favours_labels(res) -> tuple[str, str]:
    m = res.measure
    hib = (res.prep.settings.get("higher_is_better") or {}).get(res.prep.plan["outcome"])
    if m.family == "continuous":
        if hib is None:
            return "Lower in intervention", "Higher in intervention"
        if hib == "worse_keep":                            # original signs kept: negative = benefit
            return "Favours intervention", "Favours control"
        return "Favours control", "Favours intervention"  # higher is better, or signs flipped so positive = benefit
    if m.family in ("binary", "time-to-event"):
        if hib is True:
            return "Favours control", "Favours intervention"
        if hib is False:
            return "Favours intervention", "Favours control"
        return "Fewer events (intervention)", "More events (intervention)"
    return "", ""


def _fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _pfmt(p) -> str:
    """'p = 0.034' or 'p < 0.001' (never 'p = 0.000')."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "p = n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _forest_core(rows: pd.DataFrame, pooled: list[dict[str, Any]], m, title: str, theme: str, left_right=("", ""),
                 width=9.0, row_h=0.32, show_weights=True, xlabel=None):
    """rows: label, yi, ci_lo, ci_hi, weight, basis, kind ('study'|'header'|'spacer'). pooled: dicts drawn as diamonds."""
    c_main, c_pool, c_grey = THEMES.get(theme, THEMES["navy"])
    log = m.back == "exp"
    tr = (lambda x: math.exp(x)) if log else (lambda x: x)
    items = list(rows.to_dict("records")) + [{"kind": "spacer"}] + [{"kind": "pooled", **p} for p in pooled]
    n = len(items)
    top_in, bottom_in = 0.95, 1.55            # fixed margins in inches, so short plots do not collide
    fig_h = row_h * (n + 1) + top_in + bottom_in
    fig = plt.figure(figsize=(width, fig_h))
    ax = fig.add_axes([0.36, bottom_in / fig_h, 0.38, row_h * (n + 1) / fig_h])
    ax.set_ylim(-0.5, n + 0.5)
    ax.invert_yaxis()
    vals = []
    for it in items:
        for k in ("ci_lo", "ci_hi", "pi_lo", "pi_hi"):
            v = it.get(k)
            if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
                vals.append(v)
    lo, hi = (min(vals), max(vals)) if vals else (-1, 1)
    null = 0.0
    lo, hi = min(lo, null), max(hi, null)
    pad = (hi - lo) * 0.08 or 0.5
    ax.set_xlim(lo - pad, hi + pad)
    ax.axvline(null, color=c_grey, lw=0.8, ls="-")
    wmax = max([it["weight"] for it in items if _ok(it.get("weight"))] + [1])
    for i, it in enumerate(items):
        y = i
        kind = it.get("kind", "study")
        if kind == "header":
            fig.text(0.01, _yfig(ax, fig, y), str(it["label"]), fontsize=9, fontweight="bold", va="center")
            continue
        if kind == "spacer":
            continue
        if kind == "study":
            ax.plot([it["ci_lo"], it["ci_hi"]], [y, y], color=c_main, lw=1)
            size = 4 + 10 * math.sqrt((it["weight"] if _ok(it.get("weight")) else wmax / 4) / wmax)
            ax.plot(it["yi"], y, marker="s", ms=size, color=c_main, mec=c_main, ls="none")
            lab = str(it["label"]) + BASIS_MARK.get(it.get("basis"), "")
            fig.text(0.01, _yfig(ax, fig, y), lab, fontsize=8.5, va="center")
            fig.text(0.76, _yfig(ax, fig, y), f"{_fmt(tr(it['yi']))} [{_fmt(tr(it['ci_lo']))}, {_fmt(tr(it['ci_hi']))}]", fontsize=8.5, va="center")
            if show_weights and _ok(it.get("weight")):
                fig.text(0.93, _yfig(ax, fig, y), f"{it['weight']:.1f}%", fontsize=8.5, va="center")
        elif kind == "pooled" and _ok(it.get("estimate")):
            e, a, b = it["estimate"], it["ci_lo"], it["ci_hi"]
            ax.add_patch(Polygon([[a, y], [e, y - 0.35], [b, y], [e, y + 0.35]], closed=True, color=c_pool))
            if _ok(it.get("pi_lo")) and _ok(it.get("pi_hi")):
                ax.plot([it["pi_lo"], it["pi_hi"]], [y + 0.45, y + 0.45], color=c_pool, lw=2, alpha=0.45, solid_capstyle="butt")
            fig.text(0.01, _yfig(ax, fig, y), it.get("label", "Pooled"), fontsize=8.5, fontweight="bold", va="center")
            fig.text(0.76, _yfig(ax, fig, y), f"{_fmt(tr(e))} [{_fmt(tr(a))}, {_fmt(tr(b))}]", fontsize=8.5, fontweight="bold", va="center")
            if isinstance(it.get("het"), str) and it.get("het"):
                fig.text(0.01, _yfig(ax, fig, y + 0.75), it["het"], fontsize=7.5, color="#555555", va="center")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    if log:                                        # ratio scale: ticks at round ratios (0.1, 0.2, 0.5, 1, 2, 5, ...)
        cands = [b * 10.0 ** e for e in range(-4, 5) for b in (1, 2, 5)]
        ticks = [math.log(c) for c in cands if lo - pad <= math.log(c) <= hi + pad]
        if len(ticks) < 3:
            cands = [b * 10.0 ** e for e in range(-4, 5) for b in (1, 1.5, 2, 3, 5, 7)]
            ticks = [math.log(c) for c in cands if lo - pad <= math.log(c) <= hi + pad]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{math.exp(t):g}" for t in ticks])
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel or m.reported_label)
    top = _yfig(ax, fig, -1.0)
    fig.text(0.01, top, "Study", fontweight="bold")
    fig.text(0.76, top, f"{m.short} [95% CI]", fontweight="bold", fontsize=8.5)
    if show_weights:
        fig.text(0.93, top, "Weight", fontweight="bold", fontsize=8.5)
    l, r = left_right
    if l or r:
        fig.text(0.36, 0.72 / fig_h, "← " + l, fontsize=8, color="#444444", ha="left", va="center")
        fig.text(0.74, 0.72 / fig_h, r + " →", fontsize=8, color="#444444", ha="right", va="center")
    fig.text(0.01, 1 - 0.3 / fig_h, title, ha="left", va="center", fontsize=11, fontweight="bold")
    fig._metapy_h = fig_h
    return fig


def _ok(x) -> bool:
    return isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def _yfig(ax, fig, y):
    disp = ax.transData.transform((ax.get_xlim()[0], y))
    return fig.transFigure.inverted().transform(disp)[1]


def _het_text(r):
    if r.get("status") != "Pooled":
        return ""
    t = f"Heterogeneity: I² = {r['I2']:.0f}%, τ² = {r['tau2']:.3f}, Q = {r['Q']:.2f} (df = {r['Q_df']}, {_pfmt(r['Q_p'])})"
    return t + f"; test for effect: {r['stat_label']} = {r['stat']:.2f}, {_pfmt(r['p'])}"


def forest_plot(res, theme="navy", width=9.0, row_h=0.32, sort_by: str | None = None):
    d = res.studies.copy()
    if sort_by == "effect":
        d = d.sort_values("yi")
    elif sort_by == "year" and "year" in d:
        d = d.sort_values("year")
    rows = pd.DataFrame({"label": d["label"], "yi": d["yi"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                         "weight": d.get("weight_percent"), "basis": d["derivation_basis"], "kind": "study"})
    r = res.main
    model = "Random-effects" if r.get("model") == "random" else "Fixed-effect"
    pooled = [{"label": f"{model} model (k = {r.get('k')})", "estimate": r.get("estimate"), "ci_lo": r.get("ci_lo"),
               "ci_hi": r.get("ci_hi"), "pi_lo": r.get("pi_lo"), "pi_hi": r.get("pi_hi"), "het": _het_text(r)}]
    fig = _forest_core(rows, pooled, res.measure, f"{res.prep.plan['outcome']}", theme, favours_labels(res), width, row_h)
    _footnote(fig, res)
    return fig


def _footnote(fig, res):
    notes = []
    if (res.studies["derivation_basis"] == "Approximation").any():
        notes.append("† effect size derived with a published approximation")
    if (res.studies["derivation_basis"] == "Assumption").any():
        notes.append("‡ effect size relies on a researcher assumption")
    if res.main.get("pi_lo") is not None:
        notes.append("bar under diamond = 95% prediction interval")
    if notes:
        fig.text(0.01, 0.2 / getattr(fig, "_metapy_h", 4), "; ".join(notes), fontsize=7, color="#555555", va="center")


def summary_forest(results, theme="navy"):
    """One row per outcome: pooled estimate with CI (all on the same measure scale when possible)."""
    c_main, c_pool, c_grey = THEMES.get(theme, THEMES["navy"])
    rows = [r for r in results if r.main.get("status") == "Pooled"]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(rows) + 1.8))
    for i, r in enumerate(rows):
        m, mm = r.measure, r.main
        e, a, b = mm["estimate"], mm["ci_lo"], mm["ci_hi"]
        ax.plot([a, b], [i, i], color=c_main, lw=1.4)
        ax.add_patch(Polygon([[a, i], [e, i - 0.25], [b, i], [e, i + 0.25]], closed=True, color=c_pool))
        ax.text(1.02, i, f"{m.short} {m.transform(e):.2f} [{m.transform(a):.2f}, {m.transform(b):.2f}]   k = {mm['k']}, I\u00b2 = {mm['I2']:.0f}%",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
    ax.axvline(0, color=c_grey, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.prep.plan["outcome"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Pooled effect (analysis scale; log scale for ratio measures)")
    l, rlab = favours_labels(rows[0]) if rows else ("", "")
    if all(favours_labels(r) == (l, rlab) for r in rows) and l:
        ax.set_title(f"\u2190 {l}        {rlab} \u2192", fontsize=8.5, color="#444444")
    fig.suptitle("Pooled effects across outcomes", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.2, right=0.62, top=0.85)
    return fig


def subgroup_forest(res, key, theme="navy", width=9.0, row_h=0.32):
    sg = res.extras[key]
    col = sg["variable"]
    d = res.studies[res.studies[col].notna()].copy()
    rows, pooled_inline = [], []
    items = []
    for level, g in d.groupby(d[col].astype(str), sort=True):
        items.append({"kind": "header", "label": f"{col}: {level}"})
        for _, s in g.iterrows():
            items.append({"kind": "study", "label": s["label"], "yi": s["yi"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                          "weight": None, "basis": s["derivation_basis"]})
        t = sg["table"]
        rr = t[t["subgroup"] == level].iloc[0].to_dict()
        if rr.get("status") == "Pooled":
            items.append({"kind": "pooled", "label": f"Subtotal (k = {rr['k']})", "estimate": rr["estimate"], "ci_lo": rr["ci_lo"],
                          "ci_hi": rr["ci_hi"], "pi_lo": None, "het": f"I² = {rr['I2']:.0f}%, τ² = {rr['tau2']:.3f}"})
        items.append({"kind": "spacer"})
    r = res.main
    test = f"Test for subgroup differences: Q = {sg['Q_between']:.2f}, df = {sg['df']}, {_pfmt(sg['p_between'])}" if "Q_between" in sg else sg.get("note", "")
    overall = [{"label": f"Overall (k = {r['k']})", "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                "pi_lo": r.get("pi_lo"), "pi_hi": r.get("pi_hi"), "het": test}]
    fig = _forest_core(pd.DataFrame(items), overall, res.measure, f"{res.prep.plan['outcome']} by {col}", theme,
                       favours_labels(res), width, row_h, show_weights=False)
    return fig


def funnel_plot(res, theme="navy", trimfill: dict | None = None, contour=True):
    c_main, c_pool, c_grey = THEMES.get(theme, THEMES["navy"])
    d = res.studies
    se = np.sqrt(d["vi"].to_numpy(float))
    y = d["yi"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    mu = res.main["estimate"]
    smax = max(se.max(), *(np.sqrt(trimfill["filled_v"]) if trimfill and trimfill.get("filled_v") else [0])) * 1.1
    ss = np.linspace(0, smax, 50)
    if contour:
        for z, alpha in ((1.645, 0.10), (1.96, 0.18), (2.576, 0.28)):
            ax.fill_betweenx(ss, -z * ss, z * ss, color="#D9D9D9", alpha=alpha, lw=0)
    ax.plot(mu - 1.96 * ss, ss, color=c_grey, lw=0.8, ls="--")
    ax.plot(mu + 1.96 * ss, ss, color=c_grey, lw=0.8, ls="--")
    ax.axvline(mu, color=c_pool, lw=1)
    ax.axvline(0, color=c_grey, lw=0.6)
    ax.scatter(y, se, s=28, color=c_main, zorder=3, label="Studies")
    if trimfill and trimfill.get("filled_y"):
        ax.scatter(trimfill["filled_y"], np.sqrt(trimfill["filled_v"]), s=28, facecolors="white", edgecolors=c_main, zorder=3,
                   label=f"Imputed by trim-and-fill (k0 = {trimfill['k0']})")
        ax.axvline(trimfill["adjusted"]["estimate"], color=c_pool, lw=1, ls=":", label="Adjusted estimate")
    ax.set_ylim(smax, 0)
    ax.set_ylabel("Standard error")
    ax.set_xlabel(res.measure.label)
    if res.measure.back == "exp":
        ax.set_xlabel(res.measure.label + " (log scale)")
    ax.set_title(f"Funnel plot: {res.prep.plan['outcome']}" + (" (contours: p = 0.10, 0.05, 0.01)" if contour else ""), loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    if len(d) < 10:
        ax.text(0.01, 0.01, "Fewer than 10 studies: interpret asymmetry with great caution", transform=ax.transAxes, fontsize=7, color="#a00")
    fig.tight_layout()
    return fig


def bubble_plot(res, key, theme="navy"):
    c_main, c_pool, _ = THEMES.get(theme, THEMES["navy"])
    mr = res.extras[key]
    col = mr["moderators"][0]
    d = res.studies.dropna(subset=[col]).copy()
    x = pd.to_numeric(d[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    if x.notna().all():
        w = 1 / (d["vi"] + mr["tau2_residual"])
        ax.scatter(x, d["yi"], s=30 + 400 * w / w.max(), color=c_main, alpha=0.55, edgecolors=c_main)
        coef = mr["coefficients"]
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, coef["estimate"].iloc[0] + coef["estimate"].iloc[1] * xs, color=c_pool, lw=1.5)
        ax.set_xlabel(col)
    else:
        cats = d[col].astype(str)
        levels = sorted(cats.unique())
        xi = cats.map({l: i for i, l in enumerate(levels)})
        w = 1 / (d["vi"] + mr["tau2_residual"])
        ax.scatter(xi + np.random.default_rng(1).uniform(-0.08, 0.08, len(xi)), d["yi"], s=30 + 400 * w / w.max(), color=c_main, alpha=0.55)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_xlabel(col)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_ylabel(res.measure.label)
    ax.set_title(f"Meta-regression on {col} (QM {_pfmt(mr['QM_p'])}, R² = {mr['R2_percent']:.0f}%)", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def sensitivity_forest(res, table: pd.DataFrame, label_col: str, title: str, theme="navy"):
    rows = pd.DataFrame({"label": table[label_col].astype(str), "yi": table["estimate"], "ci_lo": table["ci_lo"],
                         "ci_hi": table["ci_hi"], "weight": None, "basis": None, "kind": "study"})
    r = res.main
    pooled = [{"label": "All studies", "estimate": r["estimate"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "pi_lo": None}]
    return _forest_core(rows, pooled, res.measure, title, theme, favours_labels(res), show_weights=False)


def provenance_chart(prep, theme="navy"):
    c_main, c_pool, c_grey = THEMES.get(theme, THEMES["navy"])
    order = ["Reported directly", "Exact calculation", "Approximation", "Assumption", "Not derivable"]
    counts = prep.data["derivation_basis"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    colors = [c_main, c_main, "#C98A1B", "#B03A2E", "#9E9E9E"]
    alphas = [1, 0.6, 1, 1, 1]
    for i, (k, v) in enumerate(counts.items()):
        ax.barh(i, v, color=colors[i], alpha=alphas[i])
        ax.text(v + 0.1, i, f"{int(v)}", va="center", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("Comparisons")
    ax.set_title(f"How each effect size was obtained ({prep.plan['outcome']})", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ PRISMA 2020 (optional output)
# Input model (all optional): lists for databases / registers / exclusion reasons, numbers for every box,
# a label for the included studies (e.g. "RCTs"), comparisons and participants, a footnote for the
# records-excluded box and a free-text Note printed under the diagram.
PRISMA_NUMBER_FIELDS = [
    ("duplicates_removed", "Duplicate records removed"), ("automation_removed", "Records marked ineligible by automation tools"),
    ("other_removed", "Records removed for other reasons"), ("records_screened", "Records screened (blank = calculated)"),
    ("records_excluded", "Records excluded"), ("reports_sought", "Reports sought for retrieval (blank = calculated)"),
    ("reports_not_retrieved", "Reports not retrieved"), ("reports_assessed", "Reports assessed for eligibility (blank = calculated)"),
    ("studies_included", "Studies included in review"), ("reports_included", "Reports of included studies (optional)"),
    ("comparisons_included", "Comparisons included in meta-analysis (optional)"), ("participants", "Total sample size, participants (optional)"),
]
PRISMA_TEXT_FIELDS = [("study_label", "Label for included studies, e.g. RCTs (optional)"),
                      ("excluded_footnote", "Footnote for 'Records excluded', e.g. 'after title/abstract screening' (optional)"),
                      ("note", "Note printed under the diagram (optional)")]
COMMON_DATABASES = ["PubMed", "Embase", "Scopus", "CINAHL", "Web of Science", "Cochrane Central Register of Controlled Trials (CENTRAL)", "PsycINFO"]
COMMON_REGISTERS = ["ClinicalTrials.gov", "WHO ICTRP"]
COMMON_REASONS = ["Not a randomized controlled trial", "Wrong population", "Wrong intervention", "Wrong outcomes", "Wrong study design",
                  "Duplicate publication/secondary data", "Insufficient data for meta-analysis"]


def _num(x):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return None
        f = float(str(x).replace(",", "").strip())
        return int(round(f)) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _pairs(obj) -> list[tuple[str, int]]:
    """Rows of (name, n) from a DataFrame / list of pairs / 'name = n' text; rows without a number are dropped."""
    import re as _re
    out = []
    if obj is None:
        return out
    if isinstance(obj, pd.DataFrame):
        rows = obj.values.tolist()
    elif isinstance(obj, str):
        rows = []
        for line in obj.replace(";", "\n").splitlines():
            if not line.strip():
                continue
            if _re.match(r"^\s*(total|reports?\s+excluded|excluded|records?\s+excluded)\b", line, _re.I):
                continue                                   # a total line is not a reason
            nums = _re.findall(r"\d[\d,]*", line)
            name = _re.sub(r"\(?\s*n\s*=\s*\d[\d,]*\s*\)?|[:=]\s*\d[\d,]*\s*$", "", line).strip(" :=-()")
            rows.append([name, nums[-1].replace(",", "") if nums else None])
    else:
        rows = list(obj)
    for r in rows:
        if r is None or len(r) < 2:
            continue
        name, n = str(r[0] if r[0] is not None else "").strip(), _num(r[1])
        if name and name.lower() != "nan" and n is not None:
            out.append((name, n))
    return out


def prisma_normalise(c: dict[str, Any] | None) -> dict[str, Any]:
    """Accept the new structured input or the older flat fields (db_records, register_records, reasons text)."""
    c = dict(c or {})
    out: dict[str, Any] = {"databases": _pairs(c.get("databases")), "registers": _pairs(c.get("registers")),
                           "reasons": _pairs(c.get("reasons")) or _pairs(c.get("reports_excluded_reasons")),
                           "other_sources": _pairs(c.get("other_sources"))}
    if not out["databases"] and _num(c.get("db_records")) is not None:
        out["databases"] = [("Databases", _num(c.get("db_records")))]
        out["_db_flat"] = True
    if not out["registers"] and _num(c.get("register_records")) is not None:
        out["registers"] = [("Registers", _num(c.get("register_records")))]
        out["_reg_flat"] = True
    if not out["other_sources"] and _num(c.get("other_records")) is not None:
        out["other_sources"] = [("Other methods", _num(c.get("other_records")))]
    for k, _ in PRISMA_NUMBER_FIELDS:
        out[k] = _num(c.get(k))
    for k in ("other_sought", "other_not_retrieved", "other_assessed"):
        out[k] = _num(c.get(k))
    out["other_reasons"] = _pairs(c.get("other_reasons")) or _pairs(c.get("other_excluded_reasons"))
    for k, _ in PRISMA_TEXT_FIELDS:
        v = c.get(k)
        out[k] = str(v).strip() if v is not None and str(v).strip() and str(v).strip().lower() != "nan" else ""
    out["_stated_excluded_total"] = None
    for line in str(c.get("reports_excluded_reasons") or "").splitlines():
        import re as _re
        if _re.match(r"^\s*(total|reports?\s+excluded|excluded)\b", line, _re.I) and _re.findall(r"\d+", line):
            out["_stated_excluded_total"] = int(_re.findall(r"\d+", line)[-1])
    return out


def prisma_given(c: dict[str, Any] | None) -> bool:
    if prisma_is_text(c):
        return any(_txt(c.get(k)) for k in PRISMA_BOX_KEYS)
    p = prisma_normalise(c)
    return bool(p["databases"] or p["registers"] or any(p[k] is not None for k, _ in PRISMA_NUMBER_FIELDS
                                                         if k not in ("studies_included", "comparisons_included", "participants")))


def prisma_complete(c: dict[str, Any]) -> dict[str, Any]:
    """Totals, arithmetic gaps filled, and a list of consistency problems ('_notes')."""
    p = prisma_normalise(c)
    notes = []
    ident = sum(n for _, n in p["databases"]) + sum(n for _, n in p["registers"])
    p["records_identified"] = ident if (p["databases"] or p["registers"]) else None
    removed_parts = [p[k] for k in ("duplicates_removed", "automation_removed", "other_removed")]
    p["records_removed"] = sum(x or 0 for x in removed_parts) if any(x is not None for x in removed_parts) else None
    if p["records_identified"] is not None:
        exp = p["records_identified"] - (p["records_removed"] or 0)
        if p["records_screened"] is None:
            p["records_screened"] = exp
        elif p["records_screened"] != exp:
            notes.append(f"Check: identified ({p['records_identified']:,}) minus removed ({p['records_removed'] or 0:,}) = {exp:,}, but screened = {p['records_screened']:,}.")
    else:
        notes.append("Records identified (databases / registers) not entered.")
    if p["records_screened"] is not None and p["records_excluded"] is not None:
        exp = p["records_screened"] - p["records_excluded"]
        if p["reports_sought"] is None:
            p["reports_sought"] = exp
        elif p["reports_sought"] != exp:
            notes.append(f"Check: screened ({p['records_screened']:,}) minus excluded ({p['records_excluded']:,}) = {exp:,}, but reports sought = {p['reports_sought']:,}.")
    if p["reports_sought"] is not None:
        exp = p["reports_sought"] - (p["reports_not_retrieved"] or 0)
        if p["reports_assessed"] is None:
            p["reports_assessed"] = exp
        elif p["reports_assessed"] != exp:
            notes.append(f"Check: sought ({p['reports_sought']:,}) minus not retrieved ({p['reports_not_retrieved'] or 0:,}) = {exp:,}, but assessed = {p['reports_assessed']:,}.")
    for child, parent in (("records_excluded", "records_screened"), ("reports_not_retrieved", "reports_sought"),
                          ("reports_assessed", "reports_sought"), ("records_screened", "records_identified")):
        if p.get(child) is not None and p.get(parent) is not None and p[child] > p[parent]:
            notes.append(f"Impossible: {child.replace('_', ' ')} ({p[child]:,}) is larger than {parent.replace('_', ' ')} ({p[parent]:,}).")
    p["reports_excluded"] = sum(n for _, n in p["reasons"]) if p["reasons"] else None
    if p["_stated_excluded_total"] is not None and p["reports_excluded"] is not None and p["_stated_excluded_total"] != p["reports_excluded"]:
        notes.append(f"Check: the exclusion reasons add up to {p['reports_excluded']}, but the total written is {p['_stated_excluded_total']}.")
    if p["reports_assessed"] is not None:
        if p["reports_excluded"] is None:
            notes.append("Reports excluded (with reasons) not entered.")
        else:
            inc = p["reports_assessed"] - p["reports_excluded"]
            target = p["reports_included"] if p["reports_included"] is not None else p["studies_included"]
            if target is not None and inc != target and p["reports_included"] is not None:
                notes.append(f"Check: assessed ({p['reports_assessed']}) minus excluded ({p['reports_excluded']}) = {inc}, but reports included = {target}.")
            elif target is not None and inc != target and not p["other_sources"]:
                notes.append(f"Check: assessed ({p['reports_assessed']}) minus excluded ({p['reports_excluded']}) = {inc}, but studies included = {target} "
                             "(fine if some studies have several reports; enter 'Reports of included studies' to make this explicit).")
    if p["studies_included"] is None:
        notes.append("Studies included not entered.")
    p["_notes"] = notes
    return p


def _fmt_n(n):
    return "…" if n is None else f"{n:,}"


# ---- free-text boxes (MetaPy 5): the researcher types or pastes anything; it is printed as written
PRISMA_BOXES = [  # key, printed heading, position (row, column 0 = main, 1 = side)
    ("box_identified", "Records identified from:", 0, 0), ("box_removed", "Records removed before screening:", 0, 1),
    ("box_screened", "Records screened", 1, 0), ("box_excluded", "Records excluded", 1, 1),
    ("box_sought", "Reports sought for retrieval", 2, 0), ("box_not_retrieved", "Reports not retrieved", 2, 1),
    ("box_assessed", "Reports assessed for eligibility", 3, 0), ("box_reports_excluded", "Reports excluded:", 3, 1),
    ("box_included", "Studies included in review", 4, 0),
]
PRISMA_OTHER_BOXES = [  # optional third column: identification via other methods
    ("box_other_identified", "Records identified from:"), ("box_other_sought", "Reports sought for retrieval"),
    ("box_other_not_retrieved", "Reports not retrieved"), ("box_other_assessed", "Reports assessed for eligibility"),
    ("box_other_excluded", "Reports excluded:"),
]
PRISMA_BOX_KEYS = [k for k, *_ in PRISMA_BOXES] + [k for k, _ in PRISMA_OTHER_BOXES]


def _txt(v) -> str:
    if v is None:
        return ""
    v = str(v).replace("\r\n", "\n").replace("\r", "\n").strip("\n").rstrip()
    return "" if v.strip().lower() in ("", "nan", "none") else v


def prisma_is_text(c: dict[str, Any] | None) -> bool:
    return bool(c) and any(k in c for k in PRISMA_BOX_KEYS)


def prisma_boxes(c: dict[str, Any] | None) -> dict[str, Any]:
    """Text for every box. Free-text input is used exactly as typed; older number input is turned into text."""
    c = dict(c or {})
    if prisma_is_text(c):
        out = {k: _txt(c.get(k)) for k in PRISMA_BOX_KEYS}
        out["note"] = _txt(c.get("note"))
        out["_notes"] = []
        return out
    p = prisma_complete(c)
    n = _fmt_n
    ident = []
    if p["databases"]:
        ident.append(f"Databases (n = {len(p['databases'])})" if not p.get("_db_flat") else f"Databases (n = {n(p['databases'][0][1])})")
        if not p.get("_db_flat"):
            ident += [f"  {nm} (n = {v:,})" for nm, v in p["databases"]]
    if p["registers"]:
        ident.append(f"Registers (n = {len(p['registers'])})" if not p.get("_reg_flat") else f"Registers (n = {n(p['registers'][0][1])})")
        if not p.get("_reg_flat"):
            ident += [f"  {nm} (n = {v:,})" for nm, v in p["registers"]]
    ident.append(f"Total records identified (n = {n(p['records_identified'])})")
    lab = f" {p['study_label']}" if p["study_label"] else ""
    inc = [f"(n = {n(p['studies_included'])}{lab})"]
    if p["reports_included"] is not None:
        inc.append(f"Reports of included studies (n = {p['reports_included']:,})")
    if p["comparisons_included"] is not None:
        inc.append(f"Comparisons included in meta-analysis (n = {p['comparisons_included']:,})")
    if p["participants"] is not None:
        inc.append(f"Total sample size (N = {p['participants']:,} participants)")
    note = "\n".join(x for x in (("** " + p["excluded_footnote"]) if p["excluded_footnote"] else "", p["note"]) if x)
    out = {"box_identified": "\n".join(ident),
           "box_removed": "\n".join([f"Duplicate records removed (n = {n(p['duplicates_removed'])})",
                                     f"Records marked as ineligible by automation tools (n = {n(p['automation_removed'])})",
                                     f"Records removed for other reasons (n = {n(p['other_removed'])})",
                                     f"Total records removed (n = {n(p['records_removed'])})"]),
           "box_screened": f"(n = {n(p['records_screened'])})", "box_excluded": f"(n = {n(p['records_excluded'])})" + ("**" if p["excluded_footnote"] else ""),
           "box_sought": f"(n = {n(p['reports_sought'])})", "box_not_retrieved": f"(n = {n(p['reports_not_retrieved'])})",
           "box_assessed": f"(n = {n(p['reports_assessed'])})",
           "box_reports_excluded": "\n".join([f"(n = {n(p['reports_excluded'])})"] + [f"{r} (n = {v:,})" for r, v in p["reasons"]]),
           "box_included": "\n".join(inc), "note": note, "_notes": p["_notes"]}
    if p["other_sources"] or p["other_sought"] is not None or p["other_assessed"] is not None:
        out.update({"box_other_identified": "\n".join(f"{nm} (n = {v:,})" for nm, v in p["other_sources"]),
                    "box_other_sought": f"(n = {n(p['other_sought'])})", "box_other_not_retrieved": f"(n = {n(p['other_not_retrieved'])})",
                    "box_other_assessed": f"(n = {n(p['other_assessed'])})",
                    "box_other_excluded": "\n".join(f"{r} (n = {v:,})" for r, v in p["other_reasons"])})
    return out


def prisma_checks(c: dict[str, Any] | None) -> list[str]:
    """Arithmetic checks are only possible for number input; free text is printed as written."""
    return [] if prisma_is_text(c) else prisma_complete(c).get("_notes", [])


def prisma_diagram(counts: dict[str, Any], title: str = "PRISMA 2020 flow diagram"):
    """Draw the PRISMA 2020 flow diagram from the text of each box (databases and registers; optional 'other methods' column)."""
    import re
    import textwrap
    t = prisma_boxes(counts)
    other = any(t.get(k) for k, _ in PRISMA_OTHER_BOXES)
    LH, PAD, GAP = 0.215, 0.14, 0.42                     # inches: line height, box padding, vertical gap
    xL, wL = 0.75, 4.45
    xR, wR = xL + wL + 0.55, 4.45
    xO, wO = xR + wR + 0.45, 3.1
    W = (xO + wO + 0.25) if other else 10.4
    CHAR = 0.074                                          # inches per character at 9.2 pt (approx.)
    BOLD_HEAD = {"box_identified", "box_removed", "box_reports_excluded", "box_other_identified", "box_other_excluded"}

    def body(text, w):
        out = []
        for raw in text.split("\n"):
            stripped = raw.lstrip(" \t•-")
            indent = min(3, (len(raw) - len(raw.lstrip(" \t"))) // 2 + (1 if raw.lstrip(" \t")[:1] in "•-" and raw.strip() else 0))
            width = max(10, int((w - 0.3 - 0.17 * indent) / CHAR))
            keep = re.sub(r"\((n|N) = ", lambda m_: f"({m_.group(1)}\u00a0=\u00a0", stripped)      # keep "(n = 12)" together
            keep = re.sub(r"(?<=\S) \((n|N)\u00a0", lambda m_: f"\u00a0({m_.group(1)}\u00a0", keep)
            parts = [x.replace("\u00a0", " ") for x in (textwrap.wrap(keep, width) or [""])]
            out.append((parts[0], False, indent))
            out += [(pt, False, indent + 1) for pt in parts[1:]]
        return out

    def lines_for(key, heading, w):
        txt = t.get(key, "")
        return [(heading, key in BOLD_HEAD, 0)] + (body(txt, w) if txt else [("(n = …)", False, 0)] if key not in BOLD_HEAD else [])

    pos = {(r, c): (k, h) for k, h, r, c in PRISMA_BOXES}
    rows = [(lines_for(*pos[(r, 0)], wL), lines_for(*pos[(r, 1)], wR) if (r, 1) in pos else None) for r in range(5)]
    heights = [max(len(l), len(r) if r else 0) * LH + 2 * PAD for l, r in rows]
    heights[0] = max(heights[0], 1.45)                    # room for the rotated 'Identification' label
    heights[4] = max(heights[4], 0.95)                    # and for 'Included'
    oth = {k: lines_for(k, h, wO) for k, h in PRISMA_OTHER_BOXES} if other else {}
    if other:   # the other-methods boxes sit in rows 0, 2, 3; 'not retrieved' and 'excluded' are printed inside them
        o_rows = {0: oth["box_other_identified"], 2: oth["box_other_sought"] + oth["box_other_not_retrieved"],
                  3: oth["box_other_assessed"] + oth["box_other_excluded"]}
        heights = [max(h, len(o_rows[i]) * LH + 2 * PAD) if i in o_rows else h for i, h in enumerate(heights)]
    wrap_w = int((W - 0.7) / 0.066)
    note_lines = []
    if t.get("note"):
        for i, para in enumerate(t["note"].split("\n")):
            txt = ("Note: " + para) if i == 0 else para
            note_lines += textwrap.wrap(txt, wrap_w) or [""]
    check_lines = textwrap.wrap(" ".join("⚠ " + x for x in t["_notes"]), wrap_w) if t["_notes"] else []
    top_banner = 0.55
    H = top_banner + 0.35 + sum(heights) + GAP * (len(rows) - 1) + 0.3
    H += (len(note_lines) * LH * 0.95 + 2 * PAD + 0.25) if note_lines else 0
    H += 0.45 + len(check_lines) * 0.17 if check_lines else 0.4
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    navy, blue, yellow = "#1B2A4A", "#CFE2F3", "#FFD54F"

    def box(x, ytop, w, h, lines, fc="white", ec=navy):
        ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=1.4))
        y = ytop - PAD - LH / 2
        for text, bold, indent in lines:
            ax.text(x + 0.14 + 0.17 * indent, y, text, ha="left", va="center", fontsize=9.2, fontweight="bold" if bold else "normal", color=navy)
            y -= LH

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=navy, lw=1.4, mutation_scale=14))

    yb = H - 0.15
    ax.add_patch(FancyBboxPatch((xL, yb - top_banner), xR + wR - xL, top_banner, boxstyle="round,pad=0,rounding_size=0.08", fc=yellow, ec="#C9A227", lw=1.2))
    ax.text((xL + xR + wR) / 2, yb - top_banner / 2, "Identification of studies via databases and registers", ha="center", va="center",
            fontsize=12, fontweight="bold", color=navy)
    if other:
        ax.add_patch(FancyBboxPatch((xO, yb - top_banner), wO, top_banner, boxstyle="round,pad=0,rounding_size=0.08", fc="#E8E8E8", ec="#999999", lw=1.2))
        ax.text(xO + wO / 2, yb - top_banner / 2, "Identification via other methods", ha="center", va="center", fontsize=10.5, fontweight="bold", color=navy)
    y = yb - top_banner - 0.35
    tops = []
    for (l, r), h in zip(rows, heights):
        tops.append(y)
        box(xL, y, wL, h, l)
        if r:
            box(xR, y, wR, h, r)
            arrow(xL + wL, y - h / 2, xR, y - h / 2)
        y -= h + GAP
    for i in range(len(rows) - 1):
        arrow(xL + wL / 2, tops[i] - heights[i], xL + wL / 2, tops[i + 1])
    for name, a, b in (("Identification", 0, 0), ("Screening", 1, 3), ("Included", 4, 4)):
        top, bot = tops[a], tops[b] - heights[b]
        ax.add_patch(FancyBboxPatch((0.12, bot), 0.42, top - bot, boxstyle="round,pad=0,rounding_size=0.1", fc=blue, ec="#5B8DBE", lw=1.2))
        ax.text(0.33, (top + bot) / 2, name, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold", color=navy)
    if other:
        for i, lines in o_rows.items():
            box(xO, tops[i], wO, heights[i], lines)
        arrow(xO + wO / 2, tops[0] - heights[0], xO + wO / 2, tops[2])
        arrow(xO + wO / 2, tops[2] - heights[2], xO + wO / 2, tops[3])
        ax.annotate("", xy=(xL + wL, tops[4] - heights[4] / 2), xytext=(xO + wO / 2, tops[3] - heights[3]),
                    arrowprops=dict(arrowstyle="-|>", color=navy, lw=1.4, connectionstyle="angle,angleA=-90,angleB=0"))
    yb2 = tops[-1] - heights[-1] - 0.3
    if note_lines:
        nh = len(note_lines) * LH * 0.95 + 2 * PAD
        ax.add_patch(FancyBboxPatch((0.12, yb2 - nh), W - 0.24, nh, boxstyle="round,pad=0,rounding_size=0.1", fc="#EEF5FC", ec="#9DBEDF", lw=1.1))
        yy = yb2 - PAD - LH * 0.95 / 2
        for i, text in enumerate(note_lines):
            if i == 0 and text.startswith("Note: "):
                ax.text(0.3, yy, "Note:", ha="left", va="center", fontsize=8.6, fontweight="bold", style="italic", color=navy)
                ax.text(0.3 + 0.42, yy, text[6:], ha="left", va="center", fontsize=8.6, style="italic", color=navy)
            else:
                ax.text(0.3, yy, text, ha="left", va="center", fontsize=8.6, style="italic", color=navy)
            yy -= LH * 0.95
        yb2 -= nh + 0.25
    ax.text(W / 2, yb2 - 0.1, title, ha="center", va="top", fontsize=11, fontweight="bold", color=navy)
    if check_lines:
        ax.text(0.15, yb2 - 0.42, "\n".join(check_lines), ha="left", va="top", fontsize=7.5, color="#A32D2D")
    return fig


def prisma_defaults(n_studies: int | None = None) -> dict[str, Any]:
    return {"databases": pd.DataFrame({"Database": COMMON_DATABASES, "Records": [None] * len(COMMON_DATABASES)}),
            "registers": pd.DataFrame({"Register": COMMON_REGISTERS, "Records": [None] * len(COMMON_REGISTERS)}),
            "reasons": pd.DataFrame({"Reason": COMMON_REASONS, "Reports": [None] * len(COMMON_REASONS)}),
            "studies_included": n_studies}


# kept for older callers
PRISMA_FIELDS = PRISMA_NUMBER_FIELDS + PRISMA_TEXT_FIELDS


def _reasons_text(text: Any) -> tuple[str, int | None]:
    pairs = _pairs(text)
    return "\n".join(f"{r} (n = {n})" for r, n in pairs), (sum(n for _, n in pairs) if pairs else None)


def baujat_plot(res, theme="navy"):
    c_main, c_pool, _ = THEMES.get(theme, THEMES["navy"])
    t = res.extras["baujat"]["table"]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(t["contribution_to_Q"], t["influence_on_estimate"], s=40, color=c_main, alpha=0.7)
    for _, r in t.iterrows():
        ax.annotate(str(r["study"]), (r["contribution_to_Q"], r["influence_on_estimate"]), fontsize=7.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Contribution to overall heterogeneity (Q)")
    ax.set_ylabel("Influence on the pooled estimate")
    ax.set_title("Baujat plot", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def dependent_rho_plot(res, theme="navy"):
    c_main, c_pool, _ = THEMES.get(theme, THEMES["navy"])
    t = res.extras["che_rve"]["rho_sensitivity"]
    m = res.measure
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.errorbar(t["rho"], [m.transform(x) for x in t["estimate"]],
                yerr=[[m.transform(e) - m.transform(l) for e, l in zip(t["estimate"], t["ci_lo"])],
                      [m.transform(h) - m.transform(e) for e, h in zip(t["estimate"], t["ci_hi"])]],
                fmt="s", color=c_main, capsize=4)
    ax.axhline(1 if m.back == "exp" else 0, color="#999", lw=0.6)
    ax.set_xlabel("Assumed correlation between effects from the same study (ρ)")
    ax.set_ylabel(m.short + " (95% CI, CR2)")
    ax.set_title("CHE-RVE: sensitivity to ρ", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def radial_plot(res, theme="navy"):
    """Radial (Galbraith) plot: y_i / sqrt(v_i + tau^2) against 1 / sqrt(v_i + tau^2), with the pooled-effect line and
    +/- 1.96 bands (Galbraith 1988; as in metafor::radial for random-effects models)."""
    c_main, c_pool, _ = THEMES.get(theme, THEMES["navy"])
    d = res.studies
    y, v = d["yi"].to_numpy(float), d["vi"].to_numpy(float)
    t2 = float(res.main.get("tau2") or 0.0)
    mu = float(res.main.get("estimate"))
    x = 1 / np.sqrt(v + t2)
    z = y * x
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    xs = np.linspace(0, x.max() * 1.08, 50)
    ax.plot(xs, mu * xs, color=c_pool, lw=1.2, label="pooled effect")
    ax.plot(xs, mu * xs + 1.96, color="#999", lw=0.8, ls="--")
    ax.plot(xs, mu * xs - 1.96, color="#999", lw=0.8, ls="--", label="±1.96")
    ax.axhline(0, color="#ccc", lw=0.6)
    ax.scatter(x, z, s=36, color=c_main, alpha=0.8, zorder=3)
    out = np.abs(z - mu * x) > 1.96
    lab = list(d["label"] if "label" in d else d["study_id"])
    for xi, zi, o_, l_ in zip(x, z, out, lab):
        if o_:
            ax.annotate(str(l_), (xi, zi), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(0, None)
    ax.set_xlabel("Precision, 1 / √(vᵢ + τ²)")
    ax.set_ylabel("Standardized effect, yᵢ / √(vᵢ + τ²)")
    ax.set_title("Radial (Galbraith) plot", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def labbe_plot(res, theme="navy"):
    """L'Abbé plot: event proportion in the intervention group against the control group; point size by total sample size."""
    c_main, c_pool, _ = THEMES.get(theme, THEMES["navy"])
    d = res.studies
    need = ["events_i", "n_i", "events_c", "n_c"]
    if not all(c in d for c in need):
        return None
    a, n1, c, n2 = (pd.to_numeric(d[k], errors="coerce").to_numpy(float) for k in need)
    ok = np.isfinite(a) & np.isfinite(n1) & np.isfinite(c) & np.isfinite(n2) & (n1 > 0) & (n2 > 0)
    if ok.sum() < 2:
        return None
    p1, p2, n = a[ok] / n1[ok], c[ok] / n2[ok], (n1 + n2)[ok]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(p2, p1, s=20 + 280 * n / n.max(), color=c_main, alpha=0.45, edgecolor=c_main)
    ax.plot([0, 1], [0, 1], color="#999", lw=0.8, ls="--", label="no effect")
    est = res.main.get("estimate")
    xs = np.linspace(0.0005, 0.9995, 200)
    if est is not None:
        k_ = res.measure.key
        if k_ == "log_or":
            ys = 1 / (1 + (1 - xs) / (xs * math.exp(est)))
        elif k_ == "log_rr":
            ys = np.clip(xs * math.exp(est), 0, 1)
        else:
            ys = np.clip(xs + est, 0, 1)
        ax.plot(xs, ys, color=c_pool, lw=1.2, label="pooled effect")
    hi = float(max(p1.max(), p2.max())) * 1.1 + 0.01
    ax.set_xlim(0, min(1, hi)); ax.set_ylim(0, min(1, hi))
    ax.set_xlabel("Event proportion, control group")
    ax.set_ylabel("Event proportion, intervention group")
    ax.set_title("L'Abbé plot", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
