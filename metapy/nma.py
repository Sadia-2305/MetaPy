"""Frequentist network meta-analysis (contrast-based, generalised least squares).

Input
  * arm-level data (one row per study arm: study, treatment, mean/sd/n or events/n), or
  * contrast-level data (study, treat1, treat2, TE, seTE), optionally with arm sizes for multi-arm studies.
Multi-arm studies: all contrasts of a study share arms, so their sampling errors are correlated.
  - arm-level MD / log OR / log RR: exact covariances (variance of the shared arm)
  - arm-level SMD: Gleser & Olkin (2009) approximation, Cov(d_ab, d_ac) = 1/n_a + d_ab d_ac / (2N)
  - contrast-level with a shared comparator: same approximations when arm sizes are given.
Model
  y = X d + e,  e ~ N(0, S + T), T = tau^2 on the diagonal and tau^2/2 between contrasts of the same
  multi-arm study (common heterogeneity under consistency; Higgins & Whitehead 1996).
  tau^2 by REML; fixed (common) effect when tau^2 = 0.
Outputs: estimates against a reference, league table, P-scores (Ruecker & Schwarzer 2015), Q total /
within-design heterogeneity / between-design inconsistency (Krahn et al. 2013), direct vs indirect
evidence by back-calculation (node-splitting style; Dias et al. 2010), network graph and forest plot.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from .fields import norm as _norm

MEASURES_NMA = {"MD": "Mean difference", "SMD": "Standardised mean difference (Hedges' g)",
                "OR": "Odds ratio (log scale)", "RR": "Risk ratio (log scale)", "generic": "Generic effect (as given)"}


# ------------------------------------------------------------------ input
ARM_ALIASES = {
    "study": ["study", "studyid", "study_id", "trial", "id", "author"],
    "treatment": ["treatment", "treat", "arm", "intervention", "group", "condition"],
    "mean": ["mean", "m", "meanpost", "mean_post", "postmean"],
    "sd": ["sd", "std", "sdpost", "sd_post"],
    "n": ["n", "size", "samplesize", "nrandomised", "nanalysed"],
    "events": ["events", "event", "cases", "responders", "r"],
}
CONTRAST_ALIASES = {
    "study": ARM_ALIASES["study"], "treat1": ["treat1", "t1", "treatment1", "intervention"], "treat2": ["treat2", "t2", "treatment2", "comparator", "control"],
    "TE": ["te", "effect", "estimate", "yi", "g", "smd", "md", "logor", "lnor"], "seTE": ["sete", "se", "sei", "stderr"],
    "n1": ["n1", "ni", "n_i"], "n2": ["n2", "nc", "n_c"],
}


def _map(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str]:
    out = {}
    for col in df.columns:
        key = _norm(col)
        for f, al in aliases.items():
            if f not in out and key in {_norm(a) for a in al}:
                out[f] = col
    return out


def detect_format(df: pd.DataFrame) -> str | None:
    a = _map(df, ARM_ALIASES)
    if {"study", "treatment", "n"} <= set(a) and ({"mean", "sd"} <= set(a) or "events" in a):
        return "arm"
    c = _map(df, CONTRAST_ALIASES)
    if {"study", "treat1", "treat2", "TE", "seTE"} <= set(c):
        return "contrast"
    return None


@dataclass
class Contrasts:
    data: pd.DataFrame                    # study, t1, t2, y, v
    cov: dict[str, np.ndarray]            # study -> covariance matrix of its contrasts (in data order)
    measure: str
    notes: list[str] = field(default_factory=list)


def contrasts_from_arms(df: pd.DataFrame, measure: str = "SMD") -> Contrasts:
    """Contrasts of every arm against the study's first arm with their (co)variances. Treatment order: first listed arm as baseline."""
    m = _map(df, ARM_ALIASES)
    d = df.rename(columns={v: k for k, v in m.items()})
    rows, covs, notes = [], {}, []
    for sid, g in d.groupby("study", sort=False):
        g = g.dropna(subset=["treatment", "n"]).reset_index(drop=True)
        if len(g) < 2:
            notes.append(f"{sid}: fewer than two arms, skipped")
            continue
        n = g["n"].astype(float).to_numpy()
        if measure in ("MD", "SMD"):
            mu, sd = g["mean"].astype(float).to_numpy(), g["sd"].astype(float).to_numpy()
            arm_var = sd ** 2 / n
            if measure == "SMD":
                sp = math.sqrt(np.sum((n - 1) * sd ** 2) / (np.sum(n) - len(n)))
                J = 1 - 3 / (4 * (np.sum(n) - len(n)) - 1)
        else:
            ev = g["events"].astype(float).to_numpy()
            if np.any((ev == 0) | (ev == n)):
                ev, n = ev + 0.5, n + 1.0
                notes.append(f"{sid}: 0.5 added to every arm (zero cell)")
            p = ev / n
            if measure == "OR":
                theta = np.log(p / (1 - p))
                arm_var = 1 / ev + 1 / (n - ev)
            else:
                theta = np.log(p)
                arm_var = 1 / ev - 1 / n
        pairs = [(0, b) for b in range(1, len(g))]          # basic contrasts vs the study baseline arm (non-redundant)
        ys, specs = [], []
        for a, b in pairs:
            if measure == "MD":
                yv = mu[b] - mu[a]
            elif measure == "SMD":
                yv = J * (mu[b] - mu[a]) / sp
            else:
                yv = theta[b] - theta[a]
            ys.append(yv)
            specs.append((a, b))
            rows.append({"study": str(sid), "t1": str(g.loc[b, "treatment"]), "t2": str(g.loc[a, "treatment"]), "y": yv})
        P = len(pairs)
        C = np.zeros((P, P))
        N = np.sum(n)
        for i, (a1, b1) in enumerate(specs):
            for j, (a2, b2) in enumerate(specs):
                if measure == "SMD":
                    d1, d2 = ys[i], ys[j]
                    C[i, j] = (1 / n[a1] + 1 / n[b1] + d1 ** 2 / (2 * N)) if i == j else (1 / n[a1] + d1 * d2 / (2 * N))
                else:
                    # contrast = arm b - arm a ; Cov = sum over shared arms with sign
                    s_ = 0.0
                    for arm, sg1 in ((b1, 1), (a1, -1)):
                        for arm2, sg2 in ((b2, 1), (a2, -1)):
                            if arm == arm2:
                                s_ += sg1 * sg2 * arm_var[arm]
                    C[i, j] = s_
        covs[str(sid)] = C
    out = pd.DataFrame(rows)
    out["v"] = [covs[s][i, i] for s, i in zip(out["study"], out.groupby("study").cumcount())]
    return Contrasts(out, covs, measure, notes)


def contrasts_from_table(df: pd.DataFrame, measure: str = "generic") -> Contrasts:
    """Contrast-level input. For a multi-arm study, give every comparison against the same comparator
    (e.g. A vs C and B vs C); the shared-arm covariance is approximated from the comparator arm size when
    available (n2), otherwise half of the smaller variance (a common default; flagged)."""
    m = _map(df, CONTRAST_ALIASES)
    d = df.rename(columns={v: k for k, v in m.items()}).dropna(subset=["TE", "seTE"])
    rows, covs, notes = [], {}, []
    for sid, g in d.groupby("study", sort=False):
        g = g.reset_index(drop=True)
        P = len(g)
        C = np.diag(g["seTE"].astype(float).to_numpy() ** 2)
        for i in range(P):
            for j in range(i + 1, P):
                shared = None
                same_cmp = g.loc[i, "treat2"] == g.loc[j, "treat2"]
                if same_cmp:
                    if "n2" in g and pd.notna(g.loc[i, "n2"]):
                        nc = float(g.loc[i, "n2"])
                        if measure == "SMD":
                            N = float(g["n1"].sum() + nc) if "n1" in g else 2 * nc
                            shared = 1 / nc + float(g.loc[i, "TE"]) * float(g.loc[j, "TE"]) / (2 * N)
                        else:
                            shared = min(C[i, i], C[j, j]) / 2
                    else:
                        shared = min(C[i, i], C[j, j]) / 2
                        notes.append(f"{sid}: shared-comparator covariance approximated (no comparator N given)")
                C[i, j] = C[j, i] = shared or 0.0
        for i in range(P):
            rows.append({"study": str(sid), "t1": str(g.loc[i, "treat1"]), "t2": str(g.loc[i, "treat2"]),
                         "y": float(g.loc[i, "TE"]), "v": float(C[i, i])})
        covs[str(sid)] = C
    return Contrasts(pd.DataFrame(rows), covs, measure, notes)


def contrasts_from_prepared(data: pd.DataFrame, control_label: str = "Control", default_label: str = "Intervention") -> Contrasts:
    """Use MetaPy's prepared comparisons (yi, vi) as contrasts: intervention arm vs control arm.
    Arms that share a control in the same study get the Gleser-Olkin covariance 1/n_c + d1 d2 / (2N)."""
    d = data[data["included"]].copy()
    d["t1"] = d.get("intervention_arm").fillna(default_label).astype(str) if "intervention_arm" in d else default_label
    d["t2"] = d["control_arm"].fillna(control_label).astype(str) if "control_arm" in d else control_label
    notes = []
    unnamed = d.loc[d["t1"] == default_label, "study_id"].astype(str).unique().tolist()
    if unnamed:
        notes.append(f"Arms without a name were labelled '{default_label}' ({', '.join(unnamed)}); name them (column 'intervention_arm') "
                     "if they are different treatments, otherwise the network treats them as one node")
    rows, covs = [], {}
    for sid, g in d.groupby("study_id", sort=False):
        g = g.reset_index(drop=True)
        C = np.diag(g["vi"].astype(float).to_numpy())
        N = float(pd.to_numeric(g["n_i"], errors="coerce").sum() + pd.to_numeric(g["n_c"], errors="coerce").max())
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if g.loc[i, "t2"] == g.loc[j, "t2"] and pd.notna(g.loc[i, "n_c"]):
                    C[i, j] = C[j, i] = 1 / float(g.loc[i, "n_c"]) + g.loc[i, "yi"] * g.loc[j, "yi"] / (2 * N)
        covs[str(sid)] = C
        for i in range(len(g)):
            rows.append({"study": str(sid), "t1": g.loc[i, "t1"], "t2": g.loc[i, "t2"], "y": float(g.loc[i, "yi"]), "v": float(C[i, i])})
    return Contrasts(pd.DataFrame(rows), covs, "SMD", notes)


# ------------------------------------------------------------------ model
@dataclass
class NMAResult:
    treatments: list[str]
    reference: str
    model: str
    tau2: float
    d: np.ndarray                       # basic parameters vs reference (reference = 0)
    cov: np.ndarray
    contrasts: Contrasts
    Q: dict[str, Any]
    small_values: str
    extra: dict[str, Any] = field(default_factory=dict)

    def effect(self, a: str, b: str) -> tuple[float, float]:
        """Estimate and SE of a vs b."""
        ia, ib = self.treatments.index(a), self.treatments.index(b)
        c = np.zeros(len(self.treatments))
        c[ia] += 1
        c[ib] -= 1
        full = np.zeros((len(self.treatments), len(self.treatments) - 1))
        k = 0
        for i, t in enumerate(self.treatments):
            if t != self.reference:
                full[i, k] = 1
                k += 1
        L = c @ full
        return float(L @ self.d), float(math.sqrt(max(L @ self.cov @ L, 0)))


def _build(con: Contrasts, treatments: list[str], reference: str):
    others = [t for t in treatments if t != reference]
    col = {t: i for i, t in enumerate(others)}
    data = con.data.reset_index(drop=True)
    X = np.zeros((len(data), len(others)))
    for r, (t1, t2) in enumerate(zip(data["t1"], data["t2"])):
        if t1 != reference:
            X[r, col[t1]] += 1
        if t2 != reference:
            X[r, col[t2]] -= 1
    blocks = []
    for sid, g in data.groupby("study", sort=False):
        blocks.append((g.index.to_numpy(), con.cov[sid]))
    return data["y"].to_numpy(float), X, blocks


def _gls(y, X, blocks, tau2):
    p = X.shape[1]
    XtWX = np.zeros((p, p))
    XtWy = np.zeros(p)
    logdet = 0.0
    parts = []
    for ix, S in blocks:
        n = len(ix)
        T = tau2 * (np.eye(n) * 0.5 + 0.5)            # tau^2 on diagonal, tau^2/2 off-diagonal
        V = S + T
        Vi = np.linalg.inv(V)
        logdet += np.linalg.slogdet(V)[1]
        XtWX += X[ix].T @ Vi @ X[ix]
        XtWy += X[ix].T @ Vi @ y[ix]
        parts.append((ix, Vi))
    cov = np.linalg.inv(XtWX)
    beta = cov @ XtWy
    q = sum((y[ix] - X[ix] @ beta) @ Vi @ (y[ix] - X[ix] @ beta) for ix, Vi in parts)
    reml = logdet + np.linalg.slogdet(XtWX)[1] + q
    return beta, cov, q, reml


def _connected(con: Contrasts) -> list[set[str]]:
    adj: dict[str, set[str]] = {}
    for a, b in zip(con.data["t1"], con.data["t2"]):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        stack, comp = [t], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(adj[u] - comp)
        seen |= comp
        comps.append(comp)
    return comps


def fit_nma(con: Contrasts, reference: str | None = None, model: str = "random", small_values: str = "good") -> NMAResult:
    comps = _connected(con)
    if len(comps) > 1:
        raise ValueError("The network is disconnected: " + " | ".join(", ".join(sorted(c)) for c in comps))
    treatments = sorted(set(con.data["t1"]) | set(con.data["t2"]))
    if reference is None or reference not in treatments:
        counts = pd.concat([con.data["t1"], con.data["t2"]]).value_counts()
        reference = str(counts.index[0])
    treatments = [reference] + [t for t in treatments if t != reference]
    y, X, blocks = _build(con, treatments, reference)
    beta_f, cov_f, q_total, _ = _gls(y, X, blocks, 0.0)
    # Q decomposition (Krahn et al. 2013): within-design heterogeneity + between-design inconsistency
    data = con.data.reset_index(drop=True)
    design = data.groupby("study")[["t1", "t2"]].apply(lambda g: ":".join(sorted(set(g["t1"]) | set(g["t2"]))))
    data["design"] = data["study"].map(design)
    q_het = 0.0
    for dsg, g in data.groupby("design"):
        sub = Contrasts(g.drop(columns="design"), {st: con.cov[st] for st in g["study"].unique()}, con.measure)
        tr = sorted(set(g["t1"]) | set(g["t2"]))
        yy, XX, bb = _build(sub, tr, tr[0])
        q_het += _gls(yy, XX, bb, 0.0)[2]
    df_total = len(y) - X.shape[1]
    # within-design df: per design, contrasts minus (treatments in design - 1)
    df_het = sum(len(g) - (len(dsg.split(":")) - 1) for dsg, g in data.groupby("design"))
    df_inc = df_total - df_het
    q_inc = q_total - q_het
    Q = {"Q_total": q_total, "df_total": df_total, "p_total": float(chi2.sf(q_total, df_total)) if df_total > 0 else float("nan"),
         "Q_heterogeneity": q_het, "df_heterogeneity": df_het, "p_heterogeneity": float(chi2.sf(q_het, df_het)) if df_het > 0 else float("nan"),
         "Q_inconsistency": q_inc, "df_inconsistency": df_inc, "p_inconsistency": float(chi2.sf(q_inc, df_inc)) if df_inc > 0 else float("nan")}
    tau2 = 0.0
    if model == "random" and df_total > 0:
        r = minimize_scalar(lambda t: _gls(y, X, blocks, t)[3], bounds=(0, max(1.0, float(np.var(y)) * 10)), method="bounded",
                            options={"xatol": 1e-10})
        tau2 = float(r.x) if _gls(y, X, blocks, r.x)[3] < _gls(y, X, blocks, 0.0)[3] else 0.0
    beta, cov, _, _ = _gls(y, X, blocks, tau2)
    Q["tau2"] = tau2
    k = len(y)
    s2 = (k - 1) * np.sum(1 / data["v"]) / (np.sum(1 / data["v"]) ** 2 - np.sum(1 / data["v"] ** 2)) if k > 1 else float("nan")
    Q["I2"] = max(0.0, (q_total - df_total) / q_total * 100) if q_total > 0 else 0.0
    res = NMAResult(treatments, reference, model, tau2, beta, cov, con, Q, small_values)
    res.extra["n_studies"] = data["study"].nunique()
    res.extra["n_contrasts"] = k
    res.extra["multi_arm_studies"] = int((data.groupby("study").size() > 1).sum())
    return res


# ------------------------------------------------------------------ outputs
def league_table(res: NMAResult, transform=None) -> pd.DataFrame:
    tr = transform or (lambda x: x)
    T = res.treatments
    M = pd.DataFrame("", index=T, columns=T)
    for a in T:
        for b in T:
            if a == b:
                M.loc[a, b] = a
            else:
                e, s = res.effect(a, b)
                M.loc[a, b] = f"{tr(e):.2f} ({tr(e - 1.96 * s):.2f} to {tr(e + 1.96 * s):.2f})"
    return M


def p_scores(res: NMAResult) -> pd.DataFrame:
    """P-score: mean certainty that a treatment is better than each other treatment (Ruecker & Schwarzer 2015)."""
    T = res.treatments
    sign = -1 if res.small_values == "good" else 1
    rows = []
    for a in T:
        ps = []
        for b in T:
            if a == b:
                continue
            e, s = res.effect(a, b)
            ps.append(norm.cdf(sign * e / s) if s > 0 else 0.5)
        rows.append({"treatment": a, "P-score": float(np.mean(ps))})
    return pd.DataFrame(rows).sort_values("P-score", ascending=False).reset_index(drop=True)


def vs_reference(res: NMAResult, transform=None) -> pd.DataFrame:
    tr = transform or (lambda x: x)
    rows = []
    for t in res.treatments:
        if t == res.reference:
            continue
        e, s = res.effect(t, res.reference)
        rows.append({"treatment": t, "vs": res.reference, "estimate": tr(e), "ci_lo": tr(e - 1.96 * s), "ci_hi": tr(e + 1.96 * s),
                     "p": float(2 * norm.sf(abs(e / s))), "_y": e, "_se": s})
    return pd.DataFrame(rows)


def direct_indirect(res: NMAResult) -> pd.DataFrame:
    """Back-calculation split of each compared pair into direct and indirect evidence (Koenig et al. 2013 style)."""
    con = res.contrasts
    data = con.data
    rows = []
    for (a, b), g in data.groupby(["t1", "t2"]):
        two_arm = g[g["study"].map(data["study"].value_counts()) == 1]
        if two_arm.empty:
            continue
        w = 1 / (two_arm["v"] + res.tau2)
        d_dir = float(np.sum(w * two_arm["y"]) / w.sum())
        v_dir = float(1 / w.sum())
        e_net, s_net = res.effect(a, b)
        v_net = s_net ** 2
        if v_dir <= v_net + 1e-12:
            rows.append({"comparison": f"{a} vs {b}", "studies": len(two_arm), "direct": d_dir, "indirect": None,
                         "network": e_net, "difference": None, "p (difference)": None, "note": "no indirect evidence"})
            continue
        v_ind = 1 / (1 / v_net - 1 / v_dir)
        d_ind = v_ind * (e_net / v_net - d_dir / v_dir)
        z = (d_dir - d_ind) / math.sqrt(v_dir + v_ind)
        rows.append({"comparison": f"{a} vs {b}", "studies": len(two_arm), "direct": d_dir, "indirect": d_ind, "network": e_net,
                     "difference": d_dir - d_ind, "p (difference)": float(2 * norm.sf(abs(z))), "note": ""})
    return pd.DataFrame(rows)


def network_plot(res: NMAResult, theme_color="#1F3A5F"):
    import matplotlib.pyplot as plt
    T = res.treatments
    n = len(T)
    ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    pos = {t: (math.cos(a), math.sin(a)) for t, a in zip(T, ang)}
    data = res.contrasts.data
    edges = data.groupby(["t1", "t2"])["study"].nunique()
    und: dict[tuple, int] = {}
    for (a, b), k in edges.items():
        key = tuple(sorted((a, b)))
        und[key] = und.get(key, 0) + k
    arms = pd.concat([data[["study", "t1"]].rename(columns={"t1": "t"}), data[["study", "t2"]].rename(columns={"t2": "t"})]).drop_duplicates()
    nstud = arms["t"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    mx = max(und.values())
    for (a, b), k in und.items():
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color="#7F8C8D", lw=1 + 6 * k / mx, alpha=0.7, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(k), fontsize=8, color="#333", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for t in T:
        x, y = pos[t]
        ax.scatter([x], [y], s=300 + 120 * nstud.get(t, 1), color=theme_color, zorder=2)
        ax.text(x * 1.22, y * 1.22, f"{t}\n({nstud.get(t, 0)} studies)", ha="center", va="center", fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title("Network of treatment comparisons (edge label = number of studies)", fontsize=10)
    fig.tight_layout()
    return fig


def forest_vs_reference(res: NMAResult, transform=None, label="Effect", theme_color="#1F3A5F"):
    import matplotlib.pyplot as plt
    tab = vs_reference(res).sort_values("_y")
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(tab) + 1.6))
    for i, (_, r) in enumerate(tab.iterrows()):
        ax.plot([r["_y"] - 1.96 * r["_se"], r["_y"] + 1.96 * r["_se"]], [i, i], color=theme_color)
        ax.plot(r["_y"], i, "s", color=theme_color, ms=8)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(range(len(tab)))
    ax.set_yticklabels(tab["treatment"])
    ax.set_xlabel(f"{label} vs {res.reference} ({'random' if res.tau2 > 0 or res.model == 'random' else 'common'} effects)")
    if transform:
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{transform(t):.2g}" for t in ticks])
    ax.set_title(f"Network estimates against {res.reference}", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def run_nma(con: Contrasts, reference=None, model="random", small_values="good") -> dict[str, Any]:
    res = fit_nma(con, reference, model, small_values)
    tr = math.exp if con.measure in ("OR", "RR") else None
    return {"result": res, "league": league_table(res, tr), "pscores": p_scores(res), "vs_reference": vs_reference(res, tr),
            "split": direct_indirect(res), "notes": con.notes, "transform": tr}
