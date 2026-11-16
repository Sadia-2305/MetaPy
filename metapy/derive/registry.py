"""MetaPy formula registry — the knowledge base of the derivation brain.

Every rule declares
  * the fields it needs (inputs) and produces (outputs),
  * its kind:  ``exact``         – algebraically exact or standard identity,
               ``approximation`` – a published estimator that is not exact,
               ``assumption``    – needs a researcher-supplied value/decision,
  * a group (used for researcher-facing allow / veto toggles),
  * a formula string and a citation, written into the derivation report.

Rules never invent data: every output is a deterministic function of values
that are present in the row (or of an explicit researcher setting).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from scipy.stats import norm, t as tdist

Z975 = float(norm.ppf(0.975))


@dataclass
class Rule:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    fn: Callable[[dict[str, float], dict[str, Any]], dict[str, float] | None]
    kind: str = "exact"
    group: str = "core"
    formula: str = ""
    citation: str = ""
    requires_setting: str | None = None  # rule only fires if settings[requires_setting] is truthy
    priority: int = 50  # lower runs first within a tier

    def label(self) -> str:
        return self.name


# Groups the researcher can veto (shown as options in the preparation phase)
GROUP_INFO = {
    "median_to_mean": ("Estimate mean/SD from median, IQR or range", "Wan et al. 2014; Luo et al. 2018"),
    "or_to_smd": ("Convert between log odds ratio and SMD", "Chinn 2000; Cochrane Handbook 10.6"),
    "r_to_smd": ("Convert between correlation r and SMD", "Borenstein et al. 2009, ch. 7"),
    "spearman": ("Convert Spearman rho to Pearson r", "Rupinski & Dunlap 1996"),
    "pct_to_events": ("Reconstruct event counts from percentages", "Rounded to nearest integer"),
    "p_inequality": ("Use p-value thresholds such as p<0.05 as exact", "Conservative; Cochrane Handbook 6.5.2.3"),
    "assumed_r": ("Impute a pre-post correlation", "Researcher-chosen value; Cochrane Handbook 6.5.2.8"),
    "equal_split": ("Split a total N equally between two arms", "Researcher assumption"),
    "smd_type": ("Interpret an SMD of unstated type", "Researcher decision"),
    "zero_cell": ("Add 0.5 to all cells when a cell is zero", "Cochrane Handbook 10.4.4.2"),
    "g_type": ("Use a reported Hedges' g whose design (endpoint / change / PPC2) was confirmed by the researcher", "Researcher decision, supported by matching rows"),
    "ppc2_var_simple": ("Approximate the PPC2 variance with the independent-groups SMD formula (no pre-post r needed)",
                        "Borenstein et al. 2009, eq. 4.20; equivalent to Morris 2008 with r \u2248 0.5"),
}

RULES: list[Rule] = []


def rule(name, inputs, outputs, kind="exact", group="core", formula="", citation="", requires_setting=None, priority=50):
    def deco(fn):
        RULES.append(Rule(name, tuple(inputs), tuple(outputs), fn, kind, group, formula, citation, requires_setting, priority))
        return fn
    return deco


def _pos(*vals):
    return all(v is not None and math.isfinite(v) and v > 0 for v in vals)


def j_correction(df: float) -> float:
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def _xi(n):   # for range-based SD (Wan 2014)
    return float(norm.ppf((n - 0.375) / (n + 0.25)))


def _eta(n):  # for IQR-based SD (Wan 2014)
    return float(norm.ppf((0.75 * n - 0.125) / (n + 0.25)))


# =====================================================================
# Sample sizes
# =====================================================================
@rule("Total N from arm sizes", ["n_i", "n_c"], ["n_total"], formula="N = n_i + n_c")
def _(v, s):
    return {"n_total": v["n_i"] + v["n_c"]}


@rule("Control N from total", ["n_total", "n_i"], ["n_c"], formula="n_c = N - n_i")
def _(v, s):
    n = v["n_total"] - v["n_i"]
    return {"n_c": n} if n > 0 else None


@rule("Intervention N from total", ["n_total", "n_c"], ["n_i"], formula="n_i = N - n_c")
def _(v, s):
    n = v["n_total"] - v["n_c"]
    return {"n_i": n} if n > 0 else None


@rule("Equal split of total N", ["n_total"], ["n_i", "n_c"], kind="assumption", group="equal_split",
      formula="n_i = n_c = N / 2", citation="Researcher assumption", requires_setting="equal_split", priority=90)
def _(v, s):
    half = v["n_total"] / 2.0
    return {"n_i": math.floor(half), "n_c": math.ceil(half)}


# =====================================================================
# Group-level descriptive statistics, per arm and time point
# =====================================================================
for _a in ("i", "c"):
    for _t in ("post", "pre", "change"):
        n_ = f"n_{_a}"
        m, sd, se, lo, hi = (f"{x}_{_t}_{_a}" for x in ("mean", "sd", "se", "ci_lo", "ci_hi"))
        med, q1, q3, mn, mx = (f"{x}_{_t}_{_a}" for x in ("median", "q1", "q3", "min", "max"))

        rule(f"SD from SE ({_t}, {_a})", [se, n_], [sd], formula="SD = SE x sqrt(n)",
             citation="Cochrane Handbook 6.5.2.2")(lambda v, s, se=se, n_=n_, sd=sd: {sd: v[se] * math.sqrt(v[n_])} if _pos(v[se], v[n_]) else None)
        rule(f"SE from SD ({_t}, {_a})", [sd, n_], [se], formula="SE = SD / sqrt(n)",
             citation="Cochrane Handbook 6.5.2.2", priority=60)(lambda v, s, se=se, n_=n_, sd=sd: {se: v[sd] / math.sqrt(v[n_])} if _pos(v[sd], v[n_]) else None)
        rule(f"Mean from 95% CI ({_t}, {_a})", [lo, hi], [m], formula="mean = (lower + upper) / 2",
             citation="Cochrane Handbook 6.5.2.2")(lambda v, s, lo=lo, hi=hi, m=m: {m: (v[lo] + v[hi]) / 2.0} if v[hi] >= v[lo] else None)

        def _sd_from_ci(v, s, lo=lo, hi=hi, n_=n_, sd=sd):
            n = v[n_]
            if not _pos(n) or n < 2 or v[hi] <= v[lo]:
                return None
            crit = float(tdist.ppf(0.975, n - 1))
            return {sd: math.sqrt(n) * (v[hi] - v[lo]) / (2.0 * crit)}
        rule(f"SD from 95% CI of mean ({_t}, {_a})", [lo, hi, n_], [sd],
             formula="SD = sqrt(n) x (upper - lower) / (2 x t(0.975, n-1))", citation="Cochrane Handbook 6.5.2.2")(_sd_from_ci)

        # median-based estimators (approximations)
        def _mean_iqr(v, s, med=med, q1=q1, q3=q3, n_=n_, m=m):
            n = v[n_]
            if not _pos(n):
                return None
            w = 0.7 + 0.39 / n
            return {m: w * (v[q1] + v[q3]) / 2.0 + (1 - w) * v[med]}
        rule(f"Mean from median and IQR ({_t}, {_a})", [med, q1, q3, n_], [m], kind="approximation", group="median_to_mean",
             formula="mean = w(q1+q3)/2 + (1-w)median, w = 0.7 + 0.39/n", citation="Luo et al. 2018, Stat Methods Med Res", priority=40)(_mean_iqr)

        def _sd_iqr(v, s, q1=q1, q3=q3, n_=n_, sd=sd):
            n = v[n_]
            if not _pos(n) or v[q3] < v[q1]:
                return None
            return {sd: (v[q3] - v[q1]) / (2.0 * _eta(n))}
        rule(f"SD from IQR ({_t}, {_a})", [q1, q3, n_], [sd], kind="approximation", group="median_to_mean",
             formula="SD = (q3 - q1) / (2 x Phi^-1((0.75n - 0.125)/(n + 0.25)))", citation="Wan et al. 2014, BMC Med Res Methodol", priority=40)(_sd_iqr)

        def _mean_range(v, s, med=med, mn=mn, mx=mx, n_=n_, m=m):
            n = v[n_]
            if not _pos(n):
                return None
            w = 4.0 / (4.0 + n ** 0.75)
            return {m: w * (v[mn] + v[mx]) / 2.0 + (1 - w) * v[med]}
        rule(f"Mean from median and range ({_t}, {_a})", [med, mn, mx, n_], [m], kind="approximation", group="median_to_mean",
             formula="mean = w(min+max)/2 + (1-w)median, w = 4/(4 + n^0.75)", citation="Luo et al. 2018", priority=45)(_mean_range)

        def _sd_range(v, s, mn=mn, mx=mx, n_=n_, sd=sd):
            n = v[n_]
            if not _pos(n) or v[mx] < v[mn]:
                return None
            return {sd: (v[mx] - v[mn]) / (2.0 * _xi(n))}
        rule(f"SD from range ({_t}, {_a})", [mn, mx, n_], [sd], kind="approximation", group="median_to_mean",
             formula="SD = (max - min) / (2 x Phi^-1((n - 0.375)/(n + 0.25)))", citation="Wan et al. 2014", priority=45)(_sd_range)

        def _mean_five(v, s, med=med, q1=q1, q3=q3, mn=mn, mx=mx, n_=n_, m=m):
            n = v[n_]
            if not _pos(n):
                return None
            w1 = 2.2 / (2.2 + n ** 0.75)
            w2 = 0.7 - 0.72 / n ** 0.55
            return {m: w1 * (v[mn] + v[mx]) / 2.0 + w2 * (v[q1] + v[q3]) / 2.0 + (1 - w1 - w2) * v[med]}
        rule(f"Mean from five-number summary ({_t}, {_a})", [med, q1, q3, mn, mx, n_], [m], kind="approximation", group="median_to_mean",
             formula="mean = w1(min+max)/2 + w2(q1+q3)/2 + (1-w1-w2)median", citation="Luo et al. 2018", priority=30)(_mean_five)

        def _sd_five(v, s, q1=q1, q3=q3, mn=mn, mx=mx, n_=n_, sd=sd):
            n = v[n_]
            if not _pos(n):
                return None
            return {sd: (v[mx] - v[mn]) / (4.0 * _xi(n)) + (v[q3] - v[q1]) / (4.0 * _eta(n))}
        rule(f"SD from five-number summary ({_t}, {_a})", [q1, q3, mn, mx, n_], [sd], kind="approximation", group="median_to_mean",
             formula="SD = (max-min)/(4 xi) + (q3-q1)/(4 eta)", citation="Wan et al. 2014", priority=30)(_sd_five)

    # change scores
    mpre, mpost, mchg = f"mean_pre_{_a}", f"mean_post_{_a}", f"mean_change_{_a}"
    sdpre, sdpost, sdchg, r_ = f"sd_pre_{_a}", f"sd_post_{_a}", f"sd_change_{_a}", f"r_prepost_{_a}"
    rule(f"Mean change ({_a})", [mpre, mpost], [mchg], formula="mean change = mean post - mean pre",
         citation="Cochrane Handbook 6.5.2.8")(lambda v, s, a=mpre, b=mpost, c=mchg: {c: v[b] - v[a]})
    rule(f"Post mean from baseline and change ({_a})", [mpre, mchg], [mpost], formula="mean post = mean pre + mean change",
         priority=70)(lambda v, s, a=mpre, b=mpost, c=mchg: {b: v[a] + v[c]})

    def _sd_change(v, s, a=sdpre, b=sdpost, r_=r_, c=sdchg):
        val = v[a] ** 2 + v[b] ** 2 - 2 * v[r_] * v[a] * v[b]
        return {c: math.sqrt(val)} if val > 0 else None
    rule(f"SD of change from correlation ({_a})", [sdpre, sdpost, r_], [sdchg],
         formula="SD change = sqrt(SDpre^2 + SDpost^2 - 2 r SDpre SDpost)", citation="Cochrane Handbook 6.5.2.8")(_sd_change)

    def _r_from_sds(v, s, a=sdpre, b=sdpost, r_=r_, c=sdchg):
        if not _pos(v[a], v[b]):
            return None
        r = (v[a] ** 2 + v[b] ** 2 - v[c] ** 2) / (2 * v[a] * v[b])
        return {r_: r} if -1 <= r <= 1 else None
    rule(f"Pre-post correlation from SDs ({_a})", [sdpre, sdpost, sdchg], [r_],
         formula="r = (SDpre^2 + SDpost^2 - SDchange^2) / (2 SDpre SDpost)", citation="Cochrane Handbook 6.5.2.8")(_r_from_sds)

    rule(f"Imputed pre-post correlation ({_a})", [], [r_], kind="assumption", group="assumed_r",
         formula="r = researcher-chosen value", citation="Cochrane Handbook 6.5.2.8; run sensitivity analysis",
         requires_setting="assumed_r", priority=95)(lambda v, s, r_=r_: {r_: float(s["assumed_r"])})


# =====================================================================
# Between-group continuous effects (for endpoint and change bases)
# =====================================================================
for _b in ("post", "change"):
    mi, mc, sdi, sdc = f"mean_{_b}_i", f"mean_{_b}_c", f"sd_{_b}_i", f"sd_{_b}_c"
    md, mdse, mdlo, mdhi = f"md_{_b}", f"md_se_{_b}", f"md_ci_lo_{_b}", f"md_ci_hi_{_b}"
    sdp, tt, tab, pp, ff = f"sdp_{_b}", f"t_{_b}", f"tabs_{_b}", f"p_{_b}", f"f_{_b}"
    d, g, vd, vg = f"d_{_b}", f"g_{_b}", f"var_d_{_b}", f"var_g_{_b}"
    sed, seg = f"se_d_{_b}", f"se_g_{_b}"
    dlo, dhi, glo, ghi = f"d_ci_lo_{_b}", f"d_ci_hi_{_b}", f"g_ci_lo_{_b}", f"g_ci_hi_{_b}"

    rule(f"Mean difference ({_b})", [mi, mc], [md], formula="MD = mean_i - mean_c")(lambda v, s, mi=mi, mc=mc, md=md: {md: v[mi] - v[mc]})
    rule(f"SE of MD from SDs ({_b})", [sdi, sdc, "n_i", "n_c"], [mdse], formula="SE = sqrt(SD_i^2/n_i + SD_c^2/n_c)",
         citation="Borenstein et al. 2009, eq. 4.5")(
        lambda v, s, sdi=sdi, sdc=sdc, mdse=mdse: {mdse: math.sqrt(v[sdi] ** 2 / v["n_i"] + v[sdc] ** 2 / v["n_c"])} if _pos(v["n_i"], v["n_c"]) else None)

    def _pooled_sd(v, s, sdi=sdi, sdc=sdc, sdp=sdp):
        ni, nc = v["n_i"], v["n_c"]
        if not _pos(ni, nc) or ni + nc <= 2:
            return None
        return {sdp: math.sqrt(((ni - 1) * v[sdi] ** 2 + (nc - 1) * v[sdc] ** 2) / (ni + nc - 2))}
    rule(f"Pooled SD ({_b})", [sdi, sdc, "n_i", "n_c"], [sdp], formula="SDp = sqrt(((n_i-1)SD_i^2 + (n_c-1)SD_c^2)/(n_i+n_c-2))",
         citation="Borenstein et al. 2009, eq. 4.19")(_pooled_sd)

    rule(f"MD from its 95% CI ({_b})", [mdlo, mdhi], [md], formula="MD = (lower + upper)/2", citation="Cochrane Handbook 6.3.1")(
        lambda v, s, lo=mdlo, hi=mdhi, md=md: {md: (v[lo] + v[hi]) / 2.0} if v[hi] >= v[lo] else None)
    rule(f"SE of MD from its 95% CI ({_b})", [mdlo, mdhi], [mdse], formula="SE = (upper - lower) / (2 x 1.959964)", citation="Cochrane Handbook 6.3.1")(
        lambda v, s, lo=mdlo, hi=mdhi, se=mdse: {se: (v[hi] - v[lo]) / (2 * Z975)} if v[hi] > v[lo] else None)

    def _tabs_from_p(v, s, pp=pp, tab=tab):
        p = v[pp]
        ni, nc = v["n_i"], v["n_c"]
        if not (0 < p < 1) or not _pos(ni, nc) or ni + nc <= 2:
            return None
        return {tab: float(tdist.isf(p / 2.0, ni + nc - 2))}
    rule(f"|t| from two-sided p value ({_b})", [pp, "n_i", "n_c"], [tab], formula="|t| = t^-1(1 - p/2, n_i + n_c - 2)",
         citation="Cochrane Handbook 6.5.2.3")(_tabs_from_p)
    rule(f"|t| from F with 1 df ({_b})", [ff], [tab], formula="|t| = sqrt(F)", citation="Borenstein et al. 2009, ch. 7", priority=55)(
        lambda v, s, ff=ff, tab=tab: {tab: math.sqrt(v[ff])} if v[ff] > 0 else None)
    rule(f"|t| from t ({_b})", [tt], [tab], formula="|t| = abs(t)", priority=10)(lambda v, s, tt=tt, tab=tab: {tab: abs(v[tt])})
    rule(f"Signed t from |t| and MD sign ({_b})", [tab, md], [tt], formula="t = sign(MD) x |t|")(
        lambda v, s, tab=tab, md=md, tt=tt: {tt: math.copysign(v[tab], v[md])} if v[md] != 0 else None)
    rule(f"Signed t from |t| and stated direction ({_b})", [tab, "direction_sign"], [tt], formula="t = direction x |t|", priority=60)(
        lambda v, s, tab=tab, tt=tt: {tt: v[tab] * (1 if v["direction_sign"] >= 0 else -1)})
    rule(f"SE of MD from MD and t ({_b})", [md, tab], [mdse], formula="SE = |MD| / |t|", citation="Cochrane Handbook 6.5.2.3")(
        lambda v, s, md=md, tab=tab, se=mdse: {se: abs(v[md]) / v[tab]} if v[tab] > 0 and v[md] != 0 else None)

    def _sdp_from_se(v, s, mdse=mdse, sdp=sdp):
        ni, nc = v["n_i"], v["n_c"]
        if not _pos(ni, nc, v[mdse]):
            return None
        return {sdp: v[mdse] / math.sqrt(1.0 / ni + 1.0 / nc)}
    rule(f"Pooled SD from SE of MD ({_b})", [mdse, "n_i", "n_c"], [sdp], formula="SDp = SE / sqrt(1/n_i + 1/n_c)",
         citation="Cochrane Handbook 6.5.2.3", priority=65)(_sdp_from_se)

    rule(f"Cohen's d from MD and pooled SD ({_b})", [md, sdp], [d], formula="d = MD / SDp", citation="Borenstein et al. 2009, eq. 4.18")(
        lambda v, s, md=md, sdp=sdp, d=d: {d: v[md] / v[sdp]} if v[sdp] > 0 else None)
    rule(f"Cohen's d from t ({_b})", [tt, "n_i", "n_c"], [d], formula="d = t x sqrt(1/n_i + 1/n_c)", citation="Borenstein et al. 2009, eq. 7.1", priority=55)(
        lambda v, s, tt=tt, d=d: {d: v[tt] * math.sqrt(1 / v["n_i"] + 1 / v["n_c"])} if _pos(v["n_i"], v["n_c"]) else None)

    rule(f"Hedges' g from d ({_b})", [d, "n_i", "n_c"], [g], formula="g = J x d, J = 1 - 3/(4(n_i+n_c-2) - 1)",
         citation="Hedges 1981; Borenstein et al. 2009, eq. 4.22")(
        lambda v, s, d=d, g=g: {g: j_correction(v["n_i"] + v["n_c"] - 2) * v[d]} if v["n_i"] + v["n_c"] > 3 else None)
    rule(f"Cohen's d from g ({_b})", [g, "n_i", "n_c"], [d], formula="d = g / J", citation="Borenstein et al. 2009, eq. 4.22")(
        lambda v, s, d=d, g=g: {d: v[g] / j_correction(v["n_i"] + v["n_c"] - 2)} if v["n_i"] + v["n_c"] > 3 else None)

    def _var_d(v, s, d=d, vd=vd):
        ni, nc = v["n_i"], v["n_c"]
        if not _pos(ni, nc):
            return None
        return {vd: (ni + nc) / (ni * nc) + v[d] ** 2 / (2.0 * (ni + nc))}
    rule(f"Variance of d ({_b})", [d, "n_i", "n_c"], [vd], formula="V(d) = (n_i+n_c)/(n_i n_c) + d^2 / (2(n_i+n_c))",
         citation="Borenstein et al. 2009, eq. 4.20", priority=60)(_var_d)
    rule(f"Variance of g from variance of d ({_b})", [vd, "n_i", "n_c"], [vg], formula="V(g) = J^2 x V(d)",
         citation="Borenstein et al. 2009, eq. 4.24", priority=60)(
        lambda v, s, vd=vd, vg=vg: {vg: j_correction(v["n_i"] + v["n_c"] - 2) ** 2 * v[vd]} if v["n_i"] + v["n_c"] > 3 else None)
    rule(f"Variance of d from variance of g ({_b})", [vg, "n_i", "n_c"], [vd], formula="V(d) = V(g) / J^2", priority=61)(
        lambda v, s, vd=vd, vg=vg: {vd: v[vg] / j_correction(v["n_i"] + v["n_c"] - 2) ** 2} if v["n_i"] + v["n_c"] > 3 else None)
    rule(f"Variance of g from its SE ({_b})", [seg], [vg], formula="V = SE^2", priority=20)(lambda v, s, seg=seg, vg=vg: {vg: v[seg] ** 2})
    rule(f"Variance of d from its SE ({_b})", [sed], [vd], formula="V = SE^2", priority=20)(lambda v, s, sed=sed, vd=vd: {vd: v[sed] ** 2})
    rule(f"g from its 95% CI ({_b})", [glo, ghi], [g], formula="g = (lower + upper)/2", priority=70)(
        lambda v, s, lo=glo, hi=ghi, g=g: {g: (v[lo] + v[hi]) / 2} if v[hi] >= v[lo] else None)
    rule(f"SE of g from its 95% CI ({_b})", [glo, ghi], [seg], formula="SE = (upper - lower) / (2 x 1.959964)", citation="Cochrane Handbook 6.3.1", priority=20)(
        lambda v, s, lo=glo, hi=ghi, se=seg: {se: (v[hi] - v[lo]) / (2 * Z975)} if v[hi] > v[lo] else None)
    rule(f"d from its 95% CI ({_b})", [dlo, dhi], [d], formula="d = (lower + upper)/2", priority=70)(
        lambda v, s, lo=dlo, hi=dhi, d=d: {d: (v[lo] + v[hi]) / 2} if v[hi] >= v[lo] else None)
    rule(f"SE of d from its 95% CI ({_b})", [dlo, dhi], [sed], formula="SE = (upper - lower) / (2 x 1.959964)", citation="Cochrane Handbook 6.3.1", priority=20)(
        lambda v, s, lo=dlo, hi=dhi, se=sed: {se: (v[hi] - v[lo]) / (2 * Z975)} if v[hi] > v[lo] else None)

# Unspecified SMD: researcher states whether it is d or g
rule("SMD of unstated type treated as Hedges' g", ["smd_unspecified"], ["g_post"], kind="assumption", group="smd_type",
     formula="g = reported SMD", citation="Researcher decision", requires_setting="smd_as_g", priority=80)(lambda v, s: {"g_post": v["smd_unspecified"]})
rule("SMD of unstated type treated as Cohen's d", ["smd_unspecified"], ["d_post"], kind="assumption", group="smd_type",
     formula="d = reported SMD", citation="Researcher decision", requires_setting="smd_as_d", priority=80)(lambda v, s: {"d_post": v["smd_unspecified"]})
rule("SE of unstated SMD treated as SE of g", ["smd_unspecified_se"], ["se_g_post"], kind="assumption", group="smd_type",
     formula="SE(g) = reported SE", requires_setting="smd_as_g", priority=80)(lambda v, s: {"se_g_post": v["smd_unspecified_se"]})
rule("SE of unstated SMD treated as SE of d", ["smd_unspecified_se"], ["se_d_post"], kind="assumption", group="smd_type",
     formula="SE(d) = reported SE", requires_setting="smd_as_d", priority=80)(lambda v, s: {"se_d_post": v["smd_unspecified_se"]})


# Morris (2008) pretest-posttest-control design
def _ppc2(v, s):
    ni, nc = v["n_i"], v["n_c"]
    if not _pos(ni, nc, v["sd_pre_i"], v["sd_pre_c"]) or ni + nc <= 4:
        return None
    sd_pre = math.sqrt(((ni - 1) * v["sd_pre_i"] ** 2 + (nc - 1) * v["sd_pre_c"] ** 2) / (ni + nc - 2))
    delta = ((v["mean_post_i"] - v["mean_pre_i"]) - (v["mean_post_c"] - v["mean_pre_c"])) / sd_pre
    return {"g_ppc2": j_correction(ni + nc - 2) * delta}


rule("Morris PPC2 effect size", ["mean_pre_i", "mean_post_i", "mean_pre_c", "mean_post_c", "sd_pre_i", "sd_pre_c", "n_i", "n_c"], ["g_ppc2"],
     formula="g = J x [(Mpost_i - Mpre_i) - (Mpost_c - Mpre_c)] / SDpre_pooled", citation="Morris 2008, Organ Res Methods, eq. 8")(_ppc2)


def _ppc2_var(v, s):
    ni, nc = v["n_i"], v["n_c"]
    n = ni + nc
    r = (ni * v["r_prepost_i"] + nc * v["r_prepost_c"]) / n
    j = j_correction(n - 2)
    d = v["g_ppc2"] / j
    a = n / (ni * nc)
    if r >= 1 or n <= 4:
        return None
    var_d = 2 * (1 - r) * a * ((n - 2) / (n - 4)) * (1 + d ** 2 / (2 * (1 - r) * a)) - d ** 2 / j ** 2
    var_g = j ** 2 * var_d
    return {"var_g_ppc2": var_g} if var_g > 0 else None


rule("Variance of Morris PPC2", ["g_ppc2", "n_i", "n_c", "r_prepost_i", "r_prepost_c"], ["var_g_ppc2"],
     formula="V = J^2[2(1-r)(n_i+n_c)/(n_i n_c) x (N-2)/(N-4) x (1 + d^2/(2(1-r)(n_i+n_c)/(n_i n_c))) - d^2/J^2]",
     citation="Morris 2008, eq. 25; r = N-weighted mean of group correlations")(_ppc2_var)


# =====================================================================
# Binary outcomes
# =====================================================================
for _a in ("i", "c"):
    rule(f"Events from percentage ({_a})", [f"pct_{_a}", f"n_{_a}"], [f"events_{_a}"], kind="approximation", group="pct_to_events",
         formula="events = round(% / 100 x n)", citation="Reconstructed count; rounding error possible", priority=60)(
        lambda v, s, a=_a: {f"events_{a}": float(round(v[f"pct_{a}"] / (100.0 if v[f"pct_{a}"] > 1 else 1.0) * v[f"n_{a}"]))})


def _cells(v, s):
    a, n1, c, n2 = v["events_i"], v["n_i"], v["events_c"], v["n_c"]
    b, d = n1 - a, n2 - c
    if min(a, b, c, d) < 0 or not _pos(n1, n2):
        return None
    note = ""
    if min(a, b, c, d) == 0:
        if not s.get("zero_cell", True):
            return None
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        note = "0.5 added to every cell because of a zero cell"
    return a, b, c, d, note


def _log_or_cells(v, s):
    cells = _cells(v, s)
    if cells is None:
        return None
    a, b, c, d, note = cells
    out = {"log_or": math.log(a * d / (b * c)), "var_log_or": 1 / a + 1 / b + 1 / c + 1 / d}
    if note:
        out["_note"] = note
    return out


rule("log OR from 2x2 table", ["events_i", "n_i", "events_c", "n_c"], ["log_or", "var_log_or"],
     formula="lnOR = ln(ad/bc); V = 1/a + 1/b + 1/c + 1/d", citation="Borenstein et al. 2009, eq. 5.8-5.9")(_log_or_cells)


def _log_rr_cells(v, s):
    cells = _cells(v, s)
    if cells is None:
        return None
    a, b, c, d, note = cells
    out = {"log_rr": math.log((a / (a + b)) / (c / (c + d))), "var_log_rr": 1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)}
    if note:
        out["_note"] = note
    return out


rule("log RR from 2x2 table", ["events_i", "n_i", "events_c", "n_c"], ["log_rr", "var_log_rr"],
     formula="lnRR = ln[(a/n_i)/(c/n_c)]; V = 1/a - 1/n_i + 1/c - 1/n_c", citation="Borenstein et al. 2009, eq. 5.1-5.2")(_log_rr_cells)


def _rd_cells(v, s):
    n1, n2 = v["n_i"], v["n_c"]
    if not _pos(n1, n2):
        return None
    p1, p2 = v["events_i"] / n1, v["events_c"] / n2
    var = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    return {"rd": p1 - p2, "var_rd": var} if var > 0 else None


rule("Risk difference from 2x2 table", ["events_i", "n_i", "events_c", "n_c"], ["rd", "var_rd"],
     formula="RD = a/n_i - c/n_c; V = p_i(1-p_i)/n_i + p_c(1-p_c)/n_c", citation="Borenstein et al. 2009, eq. 5.13-5.14")(_rd_cells)

for _m in ("or", "rr", "hr"):
    lm, se, var, lo, hi, p_ = f"log_{_m}", f"se_log_{_m}", f"var_log_{_m}", f"{_m}_ci_lo", f"{_m}_ci_hi", f"p_{_m}"
    up = _m.upper()
    rule(f"log {up} from reported {up}", [_m], [lm], formula=f"ln{up} = ln({up})", priority=10)(
        lambda v, s, m=_m, lm=lm: {lm: math.log(v[m])} if v[m] > 0 else None)
    rule(f"log {up} from its 95% CI", [lo, hi], [lm], formula=f"ln{up} = (ln lower + ln upper)/2", priority=70)(
        lambda v, s, lo=lo, hi=hi, lm=lm: {lm: (math.log(v[lo]) + math.log(v[hi])) / 2} if _pos(v[lo], v[hi]) else None)
    rule(f"SE of log {up} from 95% CI", [lo, hi], [se], formula=f"SE = (ln upper - ln lower) / (2 x 1.959964)", citation="Cochrane Handbook 6.3.2")(
        lambda v, s, lo=lo, hi=hi, se=se: {se: (math.log(v[hi]) - math.log(v[lo])) / (2 * Z975)} if _pos(v[lo], v[hi]) and v[hi] > v[lo] else None)
    rule(f"SE of log {up} from p value", [lm, p_], [se], formula=f"SE = |ln{up}| / z(1 - p/2)", citation="Altman & Bland 2011, BMJ")(
        lambda v, s, lm=lm, p_=p_, se=se: {se: abs(v[lm]) / float(norm.isf(v[p_] / 2))} if 0 < v[p_] < 1 and v[lm] != 0 else None)
    rule(f"Variance of log {up} from SE", [se], [var], formula="V = SE^2", priority=20)(lambda v, s, se=se, var=var: {var: v[se] ** 2})
    rule(f"SE of log {up} from variance", [var], [se], formula="SE = sqrt(V)", priority=80)(
        lambda v, s, se=se, var=var: {se: math.sqrt(v[var])} if v[var] > 0 else None)

# OR <-> SMD (Chinn 2000)
for _b in ("post",):
    rule("Cohen's d from log OR", ["log_or", "var_log_or"], [f"d_{_b}", f"var_d_{_b}"], kind="approximation", group="or_to_smd",
         formula="d = lnOR x sqrt(3)/pi; V(d) = V(lnOR) x 3/pi^2", citation="Chinn 2000, Stat Med; Cochrane Handbook 10.6", priority=85)(
        lambda v, s: {"d_post": v["log_or"] * math.sqrt(3) / math.pi, "var_d_post": v["var_log_or"] * 3 / math.pi ** 2})
    rule("log OR from Cohen's d", ["d_post", "var_d_post"], ["log_or", "var_log_or"], kind="approximation", group="or_to_smd",
         formula="lnOR = d x pi/sqrt(3); V = V(d) x pi^2/3", citation="Chinn 2000; Borenstein et al. 2009, eq. 7.1", priority=85)(
        lambda v, s: {"log_or": v["d_post"] * math.pi / math.sqrt(3), "var_log_or": v["var_d_post"] * math.pi ** 2 / 3})

# =====================================================================
# Correlations and r <-> d
# =====================================================================
rule("Pearson r from Spearman rho", ["r_spearman"], ["r"], kind="approximation", group="spearman",
     formula="r = 2 sin(pi rho / 6)", citation="Rupinski & Dunlap 1996", priority=60)(lambda v, s: {"r": 2 * math.sin(math.pi * v["r_spearman"] / 6)})
rule("Fisher z from r", ["r"], ["z_r"], formula="z = atanh(r)", citation="Borenstein et al. 2009, eq. 6.2")(
    lambda v, s: {"z_r": math.atanh(v["r"])} if -1 < v["r"] < 1 else None)
rule("Variance of Fisher z", ["n_total"], ["var_z_r"], formula="V(z) = 1/(N - 3)", citation="Borenstein et al. 2009, eq. 6.3", priority=60)(
    lambda v, s: {"var_z_r": 1 / (v["n_total"] - 3)} if v["n_total"] > 3 else None)


def _d_from_r(v, s):
    r, n = v["r"], v["n_total"]
    if not (-1 < r < 1) or n <= 3:
        return None
    var_r = (1 - r * r) ** 2 / (n - 1)
    return {"d_post": 2 * r / math.sqrt(1 - r * r), "var_d_post": 4 * var_r / (1 - r * r) ** 3}


rule("Cohen's d from correlation r", ["r", "n_total"], ["d_post", "var_d_post"], kind="approximation", group="r_to_smd",
     formula="d = 2r/sqrt(1-r^2); V(d) = 4V(r)/(1-r^2)^3", citation="Borenstein et al. 2009, eq. 7.5-7.6", priority=85)(_d_from_r)


def _r_from_d(v, s):
    ni, nc = v["n_i"], v["n_c"]
    if not _pos(ni, nc):
        return None
    a = (ni + nc) ** 2 / (ni * nc)
    return {"r": v["d_post"] / math.sqrt(v["d_post"] ** 2 + a)}


rule("Correlation r from Cohen's d", ["d_post", "n_i", "n_c"], ["r"], kind="approximation", group="r_to_smd",
     formula="r = d / sqrt(d^2 + a), a = (n_i+n_c)^2/(n_i n_c)", citation="Borenstein et al. 2009, eq. 7.7", priority=85)(_r_from_d)

# =====================================================================
# Single-group proportions
# =====================================================================
rule("Events from proportion", ["prop", "n_total"], ["events"], kind="approximation", group="pct_to_events",
     formula="events = round(p x N)", priority=60)(
    lambda v, s: {"events": float(round(v["prop"] / (100.0 if v["prop"] > 1 else 1.0) * v["n_total"]))})


def _logit(v, s):
    e, n = v["events"], v["n_total"]
    if not _pos(n) or e < 0 or e > n:
        return None
    out = {}
    if e == 0 or e == n:
        if not s.get("zero_cell", True):
            return None
        e, n = e + 0.5, n + 1.0
        out["_note"] = "0.5 continuity correction because events = 0 or N"
    p = e / n
    out.update({"logit_prop": math.log(p / (1 - p)), "var_logit_prop": 1 / e + 1 / (n - e)})
    return out


rule("Logit proportion", ["events", "n_total"], ["logit_prop", "var_logit_prop"],
     formula="logit = ln(p/(1-p)); V = 1/e + 1/(N-e)", citation="Lipsey & Wilson 2001")(_logit)

# =====================================================================
# Generic inverse-variance
# =====================================================================
rule("Generic effect from 95% CI", ["effect_ci_lo", "effect_ci_hi"], ["effect"], formula="estimate = (lower + upper)/2", priority=70)(
    lambda v, s: {"effect": (v["effect_ci_lo"] + v["effect_ci_hi"]) / 2})
rule("SE of generic effect from 95% CI", ["effect_ci_lo", "effect_ci_hi"], ["effect_se"], formula="SE = (upper - lower) / (2 x 1.959964)")(
    lambda v, s: {"effect_se": (v["effect_ci_hi"] - v["effect_ci_lo"]) / (2 * Z975)} if v["effect_ci_hi"] > v["effect_ci_lo"] else None)
rule("Variance of generic effect", ["effect_se"], ["var_effect"], formula="V = SE^2", priority=20)(lambda v, s: {"var_effect": v["effect_se"] ** 2})


for _b in ("post", "change"):
    rule(f"Variance of MD from its SE ({_b})", [f"md_se_{_b}"], [f"md_var_{_b}"], formula="V = SE^2", priority=20)(
        lambda v, s, b=_b: {f"md_var_{b}": v[f"md_se_{b}"] ** 2})


# Reported Hedges' g of unstated design -> typed g (only with a researcher decision for that row)
for _t, _g, _se, _lo, _hi in (("post", "g_post", "se_g_post", "g_ci_lo_post", "g_ci_hi_post"),
                              ("change", "g_change", "se_g_change", "g_ci_lo_change", "g_ci_hi_change"),
                              ("ppc2", "g_ppc2", "se_g_ppc2", "g_ppc2_ci_lo", "g_ppc2_ci_hi")):
    _lab = {"post": "endpoint g", "change": "change-score g", "ppc2": "Morris PPC2 g"}[_t]
    rule(f"Reported g used as {_lab}", ["g_reported"], [_g], kind="assumption", group="g_type",
         formula=f"{_g} = reported g", citation="Researcher-confirmed design", requires_setting=f"g_type_{_t}", priority=75)(
        lambda v, s, g=_g: {g: v["g_reported"]})
    rule(f"Reported SE of g used for {_lab}", ["se_g_reported"], [_se], kind="assumption", group="g_type",
         formula=f"{_se} = reported SE", requires_setting=f"g_type_{_t}", priority=75)(lambda v, s, se=_se: {se: v["se_g_reported"]})
    rule(f"Reported CI of g used for {_lab}", ["g_reported_ci_lo", "g_reported_ci_hi"], [_lo, _hi], kind="assumption", group="g_type",
         formula="CI = reported CI", requires_setting=f"g_type_{_t}", priority=75)(
        lambda v, s, lo=_lo, hi=_hi: {lo: v["g_reported_ci_lo"], hi: v["g_reported_ci_hi"]})

rule("Variance of PPC2 g from its SE", ["se_g_ppc2"], ["var_g_ppc2"], formula="V = SE^2", priority=20)(lambda v, s: {"var_g_ppc2": v["se_g_ppc2"] ** 2})
rule("SE of PPC2 g from its 95% CI", ["g_ppc2_ci_lo", "g_ppc2_ci_hi"], ["se_g_ppc2"], formula="SE = (upper - lower) / (2 x 1.959964)",
     citation="Cochrane Handbook 6.3.1", priority=20)(
    lambda v, s: {"se_g_ppc2": (v["g_ppc2_ci_hi"] - v["g_ppc2_ci_lo"]) / (2 * Z975)} if v["g_ppc2_ci_hi"] > v["g_ppc2_ci_lo"] else None)
rule("Variance of change-score g from its SE", ["se_g_change"], ["var_g_change"], formula="V = SE^2", priority=20)(
    lambda v, s: {"var_g_change": v["se_g_change"] ** 2})


def _ppc2_var_simple(v, s):
    ni, nc = v["n_i"], v["n_c"]
    if not _pos(ni, nc):
        return None
    return {"var_g_ppc2": (ni + nc) / (ni * nc) + v["g_ppc2"] ** 2 / (2 * (ni + nc))}


rule("Approximate PPC2 variance (independent-groups formula)", ["g_ppc2", "n_i", "n_c"], ["var_g_ppc2"], kind="approximation",
     group="ppc2_var_simple", formula="V \u2248 (n_i+n_c)/(n_i n_c) + g^2/(2(n_i+n_c))",
     citation="Borenstein et al. 2009, eq. 4.20; approximates Morris 2008 with r = 0.5", requires_setting="ppc2_var_simple", priority=96)(_ppc2_var_simple)


def rules_producing(field_name: str) -> list[Rule]:
    return [r for r in RULES if field_name in r.outputs]


ALL_FIELDS = sorted({f for r in RULES for f in r.inputs + r.outputs})
