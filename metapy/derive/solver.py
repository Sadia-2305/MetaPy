"""Forward-chaining derivation solver with full provenance.

For one study row the solver
  1. starts from the values present in the data (``source``),
  2. repeatedly applies *exact* rules until nothing new can be derived,
  3. then fires the best *approximation* rule (if the researcher allows that
     group) and returns to step 2,
  4. finally fires *assumption* rules that the researcher has switched on.

Exact chains are therefore always preferred over approximations, and
approximations over assumptions. Every derived value remembers the rule and the
input values used, so the complete chain behind any effect size can be shown.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .registry import RULES, Rule, rules_producing
from ..fields import label

KIND_RANK = {"source": 0, "exact": 1, "approximation": 2, "assumption": 3}


@dataclass
class Value:
    value: float
    kind: str                     # source | exact | approximation | assumption
    rule: str | None = None
    inputs: tuple[str, ...] = ()
    formula: str = ""
    citation: str = ""
    group: str = "core"
    note: str = ""
    inexact_source: bool = False  # e.g. p reported as "<0.05"

    def as_dict(self) -> dict[str, Any]:
        return {"value": self.value, "kind": self.kind, "rule": self.rule, "inputs": list(self.inputs),
                "formula": self.formula, "citation": self.citation, "group": self.group, "note": self.note}


@dataclass
class Solution:
    values: dict[str, Value] = field(default_factory=dict)
    fired: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        return all(n in self.values for n in names)

    def get(self, name: str) -> float | None:
        v = self.values.get(name)
        return None if v is None else v.value

    def overall_kind(self, names: Iterable[str]) -> str:
        """Worst kind along the whole ancestry of the given fields."""
        worst = 0
        for f in self.ancestry(names):
            worst = max(worst, KIND_RANK[self.values[f].kind])
        return [k for k, r in KIND_RANK.items() if r == worst][0]

    def ancestry(self, names: Iterable[str]) -> list[str]:
        seen: list[str] = []
        stack = [n for n in names if n in self.values]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.append(n)
            stack.extend(i for i in self.values[n].inputs if i in self.values)
        return seen

    def chain(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """Derived steps (in dependency order) needed for the given fields."""
        anc = self.ancestry(names)
        derived = [n for n in anc if self.values[n].kind != "source"]
        ordered: list[str] = []

        def visit(n):
            if n in ordered or n not in derived:
                return
            for i in self.values[n].inputs:
                visit(i)
            ordered.append(n)
        for n in derived:
            visit(n)
        steps = []
        for n in ordered:
            v = self.values[n]
            steps.append({"field": n, **v.as_dict(),
                          "input_values": {i: self.values[i].value for i in v.inputs if i in self.values}})
        return steps

    def groups_used(self, names: Iterable[str]) -> set[str]:
        return {self.values[n].group for n in self.ancestry(names)
                if self.values[n].kind in ("approximation", "assumption")}


def _allowed(rule: Rule, settings: dict[str, Any]) -> bool:
    if rule.requires_setting and not settings.get(rule.requires_setting):
        return False
    if rule.kind != "exact" and rule.group in set(settings.get("vetoed_groups", ())):
        return False
    return True


def _try(rule: Rule, sol: Solution, settings: dict[str, Any], kinds: set[str]) -> bool:
    if rule.kind not in kinds or not _allowed(rule, settings):
        return False
    if all(o in sol.values for o in rule.outputs):
        return False
    if not all(i in sol.values for i in rule.inputs):
        return False
    inexact = any(sol.values[i].inexact_source for i in rule.inputs)
    if inexact and "p_inequality" in set(settings.get("vetoed_groups", ())):
        return False
    vals = {i: sol.values[i].value for i in rule.inputs}
    try:
        out = rule.fn(vals, settings)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return False
    if not out:
        return False
    note = out.pop("_note", "") if isinstance(out, dict) else ""
    kind, group = rule.kind, rule.group
    if inexact and kind == "exact":
        kind, group = "approximation", "p_inequality"
        note = (note + "; " if note else "") + "uses a p value reported as a threshold (e.g. p<0.05); conservative"
    added = False
    for k, val in out.items():
        if k in sol.values or val is None:
            continue
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            continue
        sol.values[k] = Value(float(val), kind, rule.name, rule.inputs, rule.formula, rule.citation, group, note,
                              inexact_source=inexact)
        added = True
    if added:
        sol.fired.append(rule.name)
    return added


def _sorted(kind: str) -> list[Rule]:
    return sorted([r for r in RULES if r.kind == kind], key=lambda r: r.priority)


EXACT, APPROX, ASSUME = _sorted("exact"), _sorted("approximation"), _sorted("assumption")


def solve(source: dict[str, Any], settings: dict[str, Any] | None = None,
          inexact_fields: Iterable[str] = ()) -> Solution:
    settings = settings or {}
    sol = Solution()
    inexact_fields = set(inexact_fields)
    for k, v in source.items():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            sol.values[k] = Value(fv, "source", note="reported in data", inexact_source=k in inexact_fields)

    def exact_fixpoint():
        changed = True
        while changed:
            changed = False
            for r in EXACT:
                if _try(r, sol, settings, {"exact"}):
                    changed = True

    exact_fixpoint()
    # One approximation (or, failing that, one assumption) at a time, then back to exact rules,
    # so cheaper evidence is always preferred and later approximations can build on earlier steps.
    while True:
        fired = False
        for tier in (APPROX, ASSUME):
            for r in tier:
                if _try(r, sol, settings, {r.kind}):
                    exact_fixpoint()
                    fired = True
                    break
            if fired:
                break
        if not fired:
            break
    _consistency_checks(sol, settings)
    return sol


def _consistency_checks(sol: Solution, settings: dict[str, Any]) -> None:
    """Re-derive reported values with exact rules and flag disagreements."""
    for r in EXACT:
        if not all(i in sol.values for i in r.inputs):
            continue
        reported_outputs = [o for o in r.outputs if o in sol.values and sol.values[o].kind == "source"]
        if not reported_outputs:
            continue
        if not all(sol.values[i].kind in ("source", "exact") for i in r.inputs):
            continue
        if any(o in r.inputs for o in reported_outputs):
            continue
        try:
            out = r.fn({i: sol.values[i].value for i in r.inputs}, settings) or {}
        except Exception:
            continue
        for o in reported_outputs:
            if o not in out or not isinstance(out[o], (int, float)):
                continue
            rep, calc = sol.values[o].value, float(out[o])
            tol = max(0.02, 0.05 * abs(rep))
            if abs(rep - calc) > tol:
                sol.checks.append({"field": o, "reported": rep, "recalculated": calc, "rule": r.name,
                                   "message": f"Reported {label(o)} = {rep:.4g}, but '{r.name}' gives {calc:.4g} from the other reported values. Check the extraction."})


UNREPORTABLE = {"sdp_post", "sdp_change", "tabs_post", "tabs_change", "var_d_post", "var_g_post", "var_d_change",
                "var_g_change", "var_log_or", "var_log_rr", "var_log_hr", "var_rd", "var_z_r", "var_logit_prop", "z_r",
                "logit_prop", "var_g_ppc2", "g_ppc2", "var_effect", "smd_unspecified", "smd_unspecified_se"}
PENALTY = {"exact": 0, "approximation": 4, "assumption": 6}
# Prefer asking for raw group statistics over asking for an already-computed effect
FIELD_COST = {f: 3 for f in ("g_post", "d_post", "g_change", "d_change", "t_post", "t_change", "md_post", "md_change",
                              "smd_unspecified", "log_or", "log_rr", "log_hr", "effect", "se_g_post", "se_d_post")}


def _search_recipes(sol: Solution, targets: Iterable[str], settings: dict[str, Any] | None = None,
                    depth: int = 3, max_recipes: int = 8) -> list[list[str]]:
    """Cheapest sets of extra source values that would make the targets derivable.

    Backward search through the registry. Cost = number of values to collect,
    plus a penalty for routes that rely on approximations or assumptions, so
    the suggestion is the most natural thing to extract from the paper.
    """
    settings = settings or {}
    known = set(sol.values)
    memo: dict[tuple, list[tuple[int, frozenset]]] = {}

    def need(f: str, d: int, visiting: frozenset) -> list[tuple[int, frozenset]]:
        if f in known:
            return [(0, frozenset())]
        key = (f, d)
        if key in memo and not (visiting & {f}):
            return memo[key]
        options: list[tuple[int, frozenset]] = []
        if f not in UNREPORTABLE:
            options.append((FIELD_COST.get(f, 1), frozenset([f])))
        if d > 0 and f not in visiting:
            for r in rules_producing(f):
                if r.requires_setting and not settings.get(r.requires_setting):
                    continue
                if r.kind != "exact" and r.group in set(settings.get("vetoed_groups", ())):
                    continue
                combos = [(PENALTY[r.kind], frozenset())]
                for i in r.inputs:
                    sub = need(i, d - 1, visiting | {f})
                    if not sub:
                        combos = []
                        break
                    combos = [(c1 + c2, s1 | s2) for c1, s1 in combos for c2, s2 in sub[:4]]
                    combos = sorted(combos, key=lambda x: (x[0], len(x[1])))[:6]
                options.extend(combos)
        best: dict[frozenset, int] = {}
        for c, st in options:
            best[st] = min(c, best.get(st, 10 ** 6))
        res = sorted(((c, st) for st, c in best.items()), key=lambda x: (x[0], len(x[1])))[:6]
        memo[key] = res
        return res

    combos = [(0, frozenset())]
    for t in targets:
        sub = need(t, depth, frozenset())
        combos = [(c1 + c2, s1 | s2) for c1, s1 in combos for c2, s2 in sub]
        combos = sorted(combos, key=lambda x: (x[0], len(x[1])))[:10]
    out: list[list[str]] = []
    for c, st in sorted(combos, key=lambda x: (len(x[1]) + x[0], len(x[1]))):
        lst = sorted(st)
        if lst and lst not in out and not any(set(o) <= st for o in out):
            out.append(lst)
    return out[:max_recipes]


# Standard ways papers report data, used to phrase data requests naturally
_CONT = lambda b: [
    [f"mean_{b}_i", f"sd_{b}_i", f"mean_{b}_c", f"sd_{b}_c", "n_i", "n_c"],
    [f"mean_{b}_i", f"se_{b}_i", f"mean_{b}_c", f"se_{b}_c", "n_i", "n_c"],
    [f"mean_{b}_i", f"ci_lo_{b}_i", f"ci_hi_{b}_i", f"mean_{b}_c", f"ci_lo_{b}_c", f"ci_hi_{b}_c", "n_i", "n_c"],
    [f"median_{b}_i", f"q1_{b}_i", f"q3_{b}_i", f"median_{b}_c", f"q1_{b}_c", f"q3_{b}_c", "n_i", "n_c"],
    [f"md_{b}", f"md_ci_lo_{b}", f"md_ci_hi_{b}", "n_i", "n_c"],
    [f"md_{b}", f"p_{b}", "n_i", "n_c"],
    [f"t_{b}", "n_i", "n_c"],
    [f"g_{b}", f"g_ci_lo_{b}", f"g_ci_hi_{b}"],
    [f"g_{b}", "n_i", "n_c"],
    [f"d_{b}", "n_i", "n_c"],
]
STANDARD_RECIPES = {
    "g_post": _CONT("post"), "d_post": _CONT("post"), "md_post": _CONT("post")[:6],
    "g_change": _CONT("change") + [["mean_pre_i", "sd_pre_i", "mean_post_i", "sd_post_i", "r_prepost_i",
                                    "mean_pre_c", "sd_pre_c", "mean_post_c", "sd_post_c", "r_prepost_c", "n_i", "n_c"]],
    "md_change": _CONT("change")[:6],
    "g_ppc2": [["mean_pre_i", "sd_pre_i", "mean_post_i", "mean_pre_c", "sd_pre_c", "mean_post_c", "r_prepost_i", "r_prepost_c", "n_i", "n_c"]],
    "log_or": [["events_i", "n_i", "events_c", "n_c"], ["or", "or_ci_lo", "or_ci_hi"], ["pct_i", "n_i", "pct_c", "n_c"], ["or", "p_or"]],
    "log_rr": [["events_i", "n_i", "events_c", "n_c"], ["rr", "rr_ci_lo", "rr_ci_hi"], ["pct_i", "n_i", "pct_c", "n_c"]],
    "rd": [["events_i", "n_i", "events_c", "n_c"], ["pct_i", "n_i", "pct_c", "n_c"]],
    "log_hr": [["hr", "hr_ci_lo", "hr_ci_hi"], ["log_hr", "se_log_hr"], ["hr", "p_hr"]],
    "z_r": [["r", "n_total"], ["r_spearman", "n_total"]],
    "logit_prop": [["events", "n_total"], ["prop", "n_total"]],
    "effect": [["effect", "effect_se"], ["effect", "effect_ci_lo", "effect_ci_hi"]],
}
DUMMY = {"mean": 10.0, "median": 10.0, "sd": 3.0, "se": 0.6, "ci_lo": 8.5, "ci_hi": 11.5, "q1": 8.0, "q3": 12.0,
         "min": 2.0, "max": 20.0, "n": 30.0, "events": 8.0, "pct": 30.0, "r": 0.5, "p": 0.03, "t": 2.1, "md": -1.5,
         "g": -0.4, "d": -0.4, "or": 1.6, "rr": 1.4, "hr": 0.8, "log": 0.3, "prop": 0.3, "effect": 0.3, "direction": 1.0}


def _dummy(field: str) -> float:
    if field.endswith("_ci_lo") or "_ci_lo_" in field or field.startswith("ci_lo"):
        base = field.split("_ci_lo")[0] if "_ci_lo" in field else "mean"
        return {"or": 1.1, "rr": 1.05, "hr": 0.6, "g": -0.8, "d": -0.8, "md": -3.0, "effect": 0.1}.get(base, 8.5)
    if field.endswith("_ci_hi") or "_ci_hi_" in field or field.startswith("ci_hi"):
        base = field.split("_ci_hi")[0] if "_ci_hi" in field else "mean"
        return {"or": 2.3, "rr": 1.9, "hr": 0.95, "g": -0.05, "d": -0.05, "md": -0.2, "effect": 0.5}.get(base, 11.5)
    if field.startswith("n_"):
        return 30.0
    if field.startswith("se_") or field.endswith("_se") or "_se_" in field:
        return 0.3 if "log" in field or "_g_" in field or "_d_" in field else 0.6
    if field.startswith("r_prepost"):
        return 0.5
    for key in sorted(DUMMY, key=len, reverse=True):
        if field.startswith(key):
            return DUMMY[key]
    return 1.0


def missing_recipes(sol: Solution, targets: Iterable[str], settings: dict[str, Any] | None = None,
                    max_recipes: int = 3) -> list[list[str]]:
    """Smallest sets of extra values that would make the targets derivable.

    Candidates come from standard reporting recipes and from a backward search of
    the registry; each candidate is verified by running the solver with
    placeholder values, so every suggestion is guaranteed to work.
    """
    settings = settings or {}
    targets = list(targets)
    known = {k: v.value for k, v in sol.values.items() if v.kind == "source"}
    cands: list[tuple[int, list[str]]] = []
    for recipe in STANDARD_RECIPES.get(targets[0], []):
        miss = [f for f in recipe if f not in known]
        if miss:
            cands.append((0, miss))
    for c in _search_recipes(sol, targets, settings):
        cands.append((1, c))
    valid: list[tuple[int, int, int, list[str]]] = []
    seen = set()
    for origin, miss in cands:
        key = tuple(sorted(miss))
        if key in seen:
            continue
        seen.add(key)
        trial = dict(known)
        trial.update({f: _dummy(f) for f in miss})
        s2 = solve(trial, settings)
        if s2.has(*targets):
            valid.append((sum(FIELD_COST.get(f, 1) for f in miss), KIND_RANK[s2.overall_kind(targets)], origin, sorted(miss)))
    valid.sort()
    out: list[list[str]] = []
    for _, _, _, miss in valid:
        if not any(set(o) <= set(miss) for o in out):
            out.append(miss)
    return out[:max_recipes]


BEST_DESIGNS = [("ppc2", "g_ppc2", "var_g_ppc2", "Morris PPC2"), ("change", "g_change", "var_g_change", "change-score g"),
                ("post", "g_post", "var_g_post", "endpoint g")]


def add_best_g(sol: Solution, policy: str = "design") -> str | None:
    """Composite 'best available' Hedges' g for mixed reporting.

    policy 'design'      : baseline-adjusted first (PPC2 > change > endpoint), whatever it costs in assumptions
    policy 'assumptions' : fewest approximations/assumptions first, then the design order
    """
    cands = [(t, g, v, lab) for t, g, v, lab in BEST_DESIGNS if sol.has(g, v)]
    if not cands:
        return None
    if policy == "assumptions":
        cands.sort(key=lambda c: KIND_RANK[sol.overall_kind([c[1], c[2]])])
    t, g, v, lab = cands[0]
    sol.values["g_best"] = Value(sol.values[g].value, "exact", f"Best available design: {lab}", (g,), f"g = {lab}",
                                 "Design chosen per study; see the 'Effect-size design' column", "core")
    sol.values["var_g_best"] = Value(sol.values[v].value, "exact", f"Best available design: {lab}", (v,), f"V = V({lab})", "", "core")
    sol.best_design = lab
    return lab
