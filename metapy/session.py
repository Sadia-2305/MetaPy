"""MetaPySession — drives the whole workflow and records every decision.

    s = MetaPySession()
    scan = s.load("my_data.xlsx")              # 1-2  read + what is possible
    s.select_plans([scan.plans[0]["plan_id"]]) # researcher picks plan(s)
    prep = s.prepare(plan_id)                  # 3   derive, then show questions
    s.answer(plan_id, {"smd_type": "g"})       #     researcher answers with options
    s.approve(plan_id)                         # 4   permission to move on
    s.analyse({"model": "random"})             # 5   analysis
    s.extend(["loo", "subgroup::Region"])      # 6   more analyses on request
    s.export(["docx", "xlsx"], tables=[...], figures=[...])   # 7-8
"""
from __future__ import annotations

import copy
import datetime as _dt
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from . import analysis as A
from .export import export as _export
from .fields import ALIAS_TABLE, label, norm
from .io import Dataset, load_dataset
from .prepare import DEFAULT_SETTINGS, PrepResult, apply_answers, prepare
from .report import build_report
from .scanner import ScanReport, scan


class MetaPySession:
    def __init__(self):
        self.ds: Dataset | None = None
        self.scan_report: ScanReport | None = None
        self.plans: dict[str, dict[str, Any]] = {}
        self.settings: dict[str, dict[str, Any]] = {}
        self.preps: dict[str, PrepResult] = {}
        self.approved: dict[str, bool] = {}
        self.options: dict[str, Any] = dict(A.DEFAULT_OPTIONS)
        self.results: dict[str, A.AnalysisResult] = {}
        self.accepted = False
        self.history: list[dict[str, Any]] = []
        self.withdrawn: list[str] = []      # outcomes whose approval was withdrawn because their data changed
        self.save_path: str | None = None
        self.nma_results: list[dict[str, Any]] = []     # network meta-analyses (cleared with every new file)
        self.answered: set[str] = set()                   # plans whose questions the researcher has answered
        self.source_path: str | None = None

    # ---------------------------------------------------------- helpers
    def _log(self, step: str, detail: Any):
        self.history.append({"time": _dt.datetime.now().isoformat(timespec="seconds"), "step": step, "detail": detail})

    # ---------------------------------------------------------- phase 1-2
    def load(self, path: str | Path, group_constructs: bool = True) -> ScanReport:
        self.__init__()
        self.ds = load_dataset(path, group_constructs=group_constructs)
        self.source_path = str(path)
        self.scan_report = scan(self.ds)
        self._log("load", {"file": self.ds.source_name, **{k: v for k, v in self.scan_report.overview.items() if k != "outcome_names"}})
        return self.scan_report

    def plan_choices(self, show_all: bool = False) -> list[tuple[str, str]]:
        plans = self.scan_report.plans if self.scan_report else []
        if not show_all:
            best: dict[str, int] = {}
            keep = []
            for p in plans:
                best.setdefault(p["outcome"], 0)
                if best[p["outcome"]] < 4:
                    keep.append(p)
                    best[p["outcome"]] += 1
            plans = keep
        return [(p["summary"], p["plan_id"]) for p in plans]

    def select_plans(self, plan_ids: list[str]) -> None:
        by_id = {p["plan_id"]: p for p in self.scan_report.plans}
        chosen = [by_id[i] for i in plan_ids if i in by_id]
        outcomes = [p["outcome"] for p in chosen]
        if len(outcomes) != len(set(outcomes)):
            dup = sorted({o for o in outcomes if outcomes.count(o) > 1})
            raise ValueError(f"Choose one effect measure per outcome (more than one chosen for: {', '.join(dup)}). "
                             "Alternative measures can be added later as sensitivity analyses.")
        self.plans = {p["plan_id"]: p for p in chosen}
        sugg = (self.scan_report.g_inference or {}).get("suggested", {}) if self.scan_report else {}
        self.settings = {}
        for pid in self.plans:
            st = copy.deepcopy(DEFAULT_SETTINGS)
            st["g_types"] = dict(sugg)
            st["_g_suggested"] = dict(sugg)
            self.settings[pid] = st
        self.preps, self.approved, self.results = {}, {pid: False for pid in self.plans}, {}
        self._log("select_plans", plan_ids)

    # ---------------------------------------------------------- phase 3-4
    @staticmethod
    def _fingerprint(p: PrepResult) -> tuple:
        d = p.data
        return tuple((r, None if y is None or y != y else round(float(y), 10), None if v is None or v != v else round(float(v), 12), bool(i))
                     for r, y, v, i in zip(d["record_id"], d["yi"], d["vi"], d["included"]))

    def prepare(self, plan_id: str) -> PrepResult:
        """(Re)prepare a plan. An existing approval is kept unless the prepared numbers changed."""
        old = self.preps.get(plan_id)
        p = prepare(self.ds, self.plans[plan_id], self.settings[plan_id])
        p.fingerprint = self.ds.fingerprint
        self.preps[plan_id] = p
        if self.approved.get(plan_id):
            if old is None or self._fingerprint(old) != self._fingerprint(p):
                self.approved[plan_id] = False
                self.withdrawn.append(self.plans[plan_id]["outcome"])
                self._log("approval_withdrawn", {"plan": plan_id, "reason": "prepared data changed"})
        else:
            self.approved.setdefault(plan_id, False)
        return p

    def prepare_all(self) -> dict[str, PrepResult]:
        return {pid: self.prepare(pid) for pid in self.plans}

    def answer(self, plan_id: str, answers: dict[str, Any]) -> PrepResult:
        prep = self.preps.get(plan_id) or self.prepare(plan_id)
        self.settings[plan_id] = apply_answers(self.settings[plan_id], answers, self.plans[plan_id]["outcome"], prep.questions)
        self._log("answers", {"plan": plan_id, "answers": answers})
        if not hasattr(self, "answered"):
            self.answered = set()
        self.answered.add(plan_id)
        return self.prepare(plan_id)

    def default_answers(self, plan_id: str) -> dict[str, Any]:
        """The answers pre-selected on screen for a plan's open questions (MetaPy's recommendations)."""
        prep = self.preps.get(plan_id) or self.prepare(plan_id)
        return {q["id"]: q.get("default") for q in prep.questions if "default" in q}

    def apply_pending_answers(self, plan_ids: list[str]) -> list[str]:
        """Apply the pre-selected answers to plans whose questions were never answered, so that what the
        researcher saw on screen is what is used. Returns the plans that were updated."""
        done = []
        for pid in plan_ids:
            if pid in getattr(self, "answered", set()):
                continue
            prep = self.preps.get(pid) or self.prepare(pid)
            if prep.questions:
                self.answer(pid, self.default_answers(pid))
                done.append(pid)
        return done

    def exclude(self, plan_id: str, record_ids: list[str]) -> PrepResult:
        self.settings[plan_id]["excluded_records"] = list(record_ids)
        self._log("exclude", {"plan": plan_id, "records": record_ids})
        return self.prepare(plan_id)

    def add_values(self, plan_id: str, values: dict[str, dict[str, float]]) -> PrepResult:
        """Researcher-supplied extra statistics: {record_id: {field or column name: value}}."""
        mv = self.settings[plan_id].setdefault("manual_values", {})
        for rid, kv in values.items():
            for k, v in kv.items():
                f = k if k in _known_fields() else ALIAS_TABLE.get(norm(k), k)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                mv.setdefault(rid, {})[f] = fv
        self._log("add_values", {"plan": plan_id, "values": values})
        return self.prepare(plan_id)

    def extra_choices(self) -> list[dict[str, Any]]:
        """Additional analyses that make sense for the prepared plans (adds mixed-data options when relevant)."""
        out = list(self.scan_report.extra_analyses) if self.scan_report else []
        keys = {a["key"] for a in out}
        preps = list(self.preps.values())
        if any(p.data.loc[p.data["included"], "Effect-size design"].nunique() > 1 for p in preps if "Effect-size design" in p.data):
            for k, lab in (("subgroup::Effect-size design", "Subgroup analysis by effect-size design (PPC2 / change / endpoint)"),
                           ("design_only", "Sensitivity: most common design only")):
                if k not in keys:
                    out.insert(0, {"key": k, "label": lab, "needs": "mixed designs"})
        if any((p.data.get("estimate from", pd.Series(dtype=object)) == "reported g (design confirmed)").any() for p in preps):
            out.insert(0, {"key": "raw_only", "label": "Sensitivity: effects calculated from raw data only (drop reported-g rows)", "needs": "reported g used"})
        if any(p.settings.get("assumed_r") is not None or p.settings.get("ppc2_var_simple") for p in preps):
            for a in out:
                if a["key"] == "r_grid":
                    a["label"] += "  [recommended: you imputed r]"
        return out

    def approve_many(self, plan_ids: list[str], apply_pending: bool = True) -> tuple[list[str], list[str]]:
        """Approve exactly the given plans (others become unapproved). Returns (approved, refused).
        Plans whose questions were never answered get the pre-selected (recommended) answers first."""
        if apply_pending:
            self.last_auto_answered = self.apply_pending_answers([p for p in plan_ids if p in self.plans])
        ok, refused = [], []
        for pid in self.plans:
            want = pid in plan_ids
            if want and pid not in self.preps:
                self.prepare(pid)
            if want and self.preps[pid].counts["included"] < 2:
                refused.append(pid)
                want = False
            if want != bool(self.approved.get(pid)):
                self.approve(pid, want)
            if want:
                ok.append(pid)
        return ok, refused

    def approve(self, plan_id: str, approved: bool = True) -> None:
        if approved:
            self.apply_pending_answers([plan_id])
        self.approved[plan_id] = approved
        p = self.preps[plan_id]
        self._log("approve_preparation", {"plan": plan_id, "approved": approved, "counts": p.counts,
                                          "settings": {k: v for k, v in self.settings[plan_id].items() if k != "manual_values"}})

    # ---------------------------------------------------------- phase 5-6
    def analyse(self, options: dict[str, Any] | None = None, plan_ids: list[str] | None = None) -> dict[str, A.AnalysisResult]:
        if options:
            self.options.update(options)
        todo = [pid for pid, ok in self.approved.items() if ok and (plan_ids is None or pid in plan_ids)]
        if not todo:
            raise ValueError("Approve the data preparation for at least one analysis plan first.")
        self.results = {pid: A.run(self.preps[pid], self.options, self.ds) for pid in todo}
        self.analysed_plans = todo
        self.accepted = False
        self._log("analyse", {"options": self.options, "plans": todo})
        return self.results

    def extend(self, extras: list[str], options: dict[str, Any] | None = None) -> dict[str, A.AnalysisResult]:
        opts = dict(self.options)
        opts.update(options or {})
        opts["extras"] = sorted(set(opts.get("extras", [])) | set(extras))
        return self.analyse(opts, list(self.results) or None)

    # ---------------------------------------------------------- saving / resuming
    def save(self, path: str | Path | None = None) -> str:
        import pickle
        path = Path(path or self.save_path or default_save_path(self.ds.source_name if self.ds else "session",
                                                                 self.ds.fingerprint if self.ds else ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path = str(path)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(self, fh)
        tmp.replace(path)
        return str(path)

    @staticmethod
    def restore(path: str | Path) -> "MetaPySession":
        import pickle
        with open(path, "rb") as fh:
            s = pickle.load(fh)
        s.save_path = str(path)
        return s

    def prisma_autofill(self) -> dict[str, Any]:
        """Studies, comparisons and unique participants of the analysed plans (shared controls counted once)."""
        res = list(self.results.values())
        if not res:
            return {}
        d = pd.concat([r.studies[["study_id", "intervention_arm", "timepoint", "n_i", "n_c"]] for r in res], ignore_index=True)
        d["arm"] = d["intervention_arm"].fillna("").astype(str)
        studies = d["study_id"].nunique()
        comparisons = d.drop_duplicates(["study_id", "arm"]).shape[0]
        ni = pd.to_numeric(d["n_i"], errors="coerce")
        nc = pd.to_numeric(d["n_c"], errors="coerce")
        people = 0.0
        for sid, g in d.assign(ni=ni, nc=nc).groupby("study_id"):
            people += g.groupby("arm")["ni"].max().sum() + g["nc"].max()
        return {"studies_included": int(studies), "comparisons_included": int(comparisons), "participants": int(round(people))}

    def accept(self):
        self.accepted = True
        self._log("accept_results", True)

    # ---------------------------------------------------------- phase 7-8
    def nma_sources(self) -> list[tuple[str, str]]:
        """Where a network can come from: sheets of the loaded file in arm or contrast format, and prepared outcomes with several arms."""
        from . import nma as N
        out = []
        if self.source_path:
            for sheet, df in _read_all_sheets(self.source_path).items():
                fmt = N.detect_format(df)
                if fmt:
                    out.append((f"Sheet '{sheet}' ({fmt}-level data)", f"sheet::{sheet}"))
        for pid, p in self.preps.items():
            d = p.data[p.data["included"]]
            if "intervention_arm" in d and d["intervention_arm"].fillna("").astype(str).str.len().gt(0).any():
                arms = set(d["intervention_arm"].dropna().astype(str))
                if len(arms) >= 2:
                    out.append((f"Prepared outcome {p.plan['outcome']} ({len(arms)} intervention arms vs control)", f"plan::{pid}"))
        return out

    def nma(self, source: str, measure: str = "SMD", reference: str | None = None, model: str = "random",
            small_values: str = "good", outcome: str | None = None) -> dict[str, Any]:
        from . import nma as N
        kind, name = source.split("::", 1)
        if kind == "sheet":
            df = _read_all_sheets(self.source_path)[name]
            fmt = N.detect_format(df)
            con = N.contrasts_from_arms(df, measure) if fmt == "arm" else N.contrasts_from_table(df, measure if measure in N.MEASURES_NMA else "generic")
        else:
            con = N.contrasts_from_prepared(self.preps[name].data)
            outcome = outcome or self.preps[name].plan["outcome"]
        r = N.run_nma(con, reference, model, small_values)
        r["outcome"] = outcome or name
        r["fingerprint"] = self.ds.fingerprint if self.ds else ""
        self.nma_results = [x for x in self.nma_results if x["outcome"] != r["outcome"]] + [r]
        self._log("nma", {"source": source, "measure": measure, "reference": r["result"].reference, "model": model})
        return r

    def report(self, tables, figures, texts, prisma=None, theme="navy", title="Meta-analysis report", forest_sort=None):
        nm = [r for r in self.nma_results if not self.ds or r.get("fingerprint") == self.ds.fingerprint]
        return build_report(list(self.results.values()), self.ds, tables, figures, texts, prisma, theme, title, forest_sort, nm)

    def export(self, formats: list[str], out_dir: str | Path = "metapy_output", tables=None, figures=None, texts=None,
               prisma=None, theme="navy", title="Meta-analysis report", stem="MetaPy_report", forest_sort=None) -> list[str]:
        from .report import FIGURE_CHOICES, TABLE_CHOICES, TEXT_CHOICES
        rep = self.report(tables if tables is not None else list(TABLE_CHOICES),
                          figures if figures is not None else list(FIGURE_CHOICES),
                          texts if texts is not None else list(TEXT_CHOICES), prisma, theme, title, forest_sort)
        self._log("export", {"formats": formats, "tables": tables, "figures": figures, "texts": texts, "prisma": prisma})
        paths = _export(rep, formats, out_dir, stem, self.record())
        import matplotlib.pyplot as plt
        plt.close("all")
        return paths

    def record(self) -> dict[str, Any]:
        """Everything needed to reproduce the analysis."""
        return {
            "metapy_version": __version__, "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "source_file": self.ds.source_name if self.ds else None,
            "plans": list(self.plans.values()), "settings": self.settings, "analysis_options": self.options,
            "approved": self.approved, "results_accepted": self.accepted,
            "pooled": {pid: {k: v for k, v in r.main.items() if k != "weights"} for pid, r in self.results.items()},
            "decision_history": self.history,
        }


def save_dir() -> Path:
    import os
    d = os.environ.get("METAPY_SAVE_DIR")
    if d:
        return Path(d)
    return Path("/content/metapy_sessions") if Path("/content").is_dir() else Path.home() / ".metapy_sessions"


def default_save_path(source_name: str, fingerprint: str = "") -> Path:
    """One save file per distinct data file: same name but different content -> different save file."""
    import re
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_name).stem)[:60] or "session"
    return save_dir() / (f"{stem}_{fingerprint[:8]}.metapy" if fingerprint else f"{stem}.metapy")


def saved_sessions() -> list[tuple[str, str]]:
    d = save_dir()
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.metapy"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [(f"{p.stem}  (saved {_dt.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M})", str(p)) for p in files]


def _known_fields() -> set[str]:
    from .derive.registry import ALL_FIELDS
    return set(ALL_FIELDS)


def _read_all_sheets(path: str) -> dict[str, pd.DataFrame]:
    p = str(path).lower()
    if p.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, sheet_name=None)
    return {"data": pd.read_csv(path)}
