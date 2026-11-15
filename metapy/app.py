"""MetaPy step-by-step app (Gradio). Every decision is an option, not free text.

Launch:  python -m metapy            or   from metapy.app import launch; launch()
"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import pandas as pd

from . import __version__
from .analysis import EXTRA_LABELS, interpret, summary_table
from .export import FORMATS, display_frame
from .fields import label
from .figures import (PRISMA_BOXES, PRISMA_OTHER_BOXES, funnel_plot, forest_plot, prisma_checks, prisma_given)
from .report import FIGURE_CHOICES, TABLE_CHOICES, TEXT_CHOICES
from .session import MetaPySession, saved_sessions

# Website mode (Hugging Face Spaces or METAPY_WEB=1): many people share one server, so nothing is saved on it
# and no saved-session list is shown (one user could otherwise open another user's data).
WEB_MODE = os.environ.get("METAPY_WEB", "").lower() in ("1", "true", "yes") or bool(os.environ.get("SPACE_ID"))
WEB_NOTICE = ("**Online version.** Your file is processed in memory on this server only while you use the page; MetaPy does not "
              "store your data or results here, and nothing is shared with other users. Closing or reloading the page ends your session, "
              "so export your report before you leave. Use de-identified, aggregate data only. To save and resume work, use the Colab or "
              "local version.")

N_RADIO, N_CHECK = 8, 2
DEFAULT_TABLES = ["characteristics", "effects", "pooled", "sof", "extras", "derivations", "actions", "requests", "checks", "evidence", "parsing"]
DEFAULT_FIGURES = ["summary_forest", "forest", "subgroup_forest", "funnel", "bubble", "loo", "provenance", "baujat", "rho", "network"]
# PRISMA form inputs, in this order: switch, the main boxes, the optional 'other methods' boxes, the note
PRISMA_KEYS = ["_on"] + [k for k, *_ in PRISMA_BOXES] + [k for k, _ in PRISMA_OTHER_BOXES] + ["note"]
PRISMA_HINTS = {  # neutral examples only: the researcher types or pastes whatever their review reports
    "box_identified": "Type or paste, one item per line, e.g.\nDatabases (n = …)\n  first source (n = …)\nRegisters (n = …)",
    "box_removed": "e.g.\nDuplicate records removed (n = …)\nRecords removed for other reasons (n = …)",
    "box_screened": "e.g. (n = …)", "box_excluded": "e.g. (n = …)", "box_sought": "e.g. (n = …)", "box_not_retrieved": "e.g. (n = …)",
    "box_assessed": "e.g. (n = …)", "box_reports_excluded": "e.g.\n(n = …)\nreason (n = …)\nreason (n = …)",
    "box_included": "e.g.\n(n = … studies)\nComparisons (n = …)\nParticipants (N = …)",
}


def _gr():
    import gradio as gr
    return gr


def upd(**kw):
    return _gr().update(**kw)


def _df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"": ["Nothing to show"]})
    return display_frame(df)


# ======================================================================== step 1
def scan_overview_md(s) -> str:
    """Plain-text (markdown) summary of what MetaPy found in the file. Used by the Gradio and Streamlit apps."""
    rep = s.scan_report
    o = rep.overview
    md = [f"### What MetaPy {__version__} found in `{o['file']}`",
          f"- **{o['studies']} studies**, {o['rows']} unique comparisons, {o['outcomes']} outcome(s): {', '.join(map(str, o['outcome_names']))}"]
    if o.get("duplicate_rows_merged"):
        md.append(f"- {o['rows_read']} rows were read from {o['worksheets']} worksheet(s); **{o['duplicate_rows_merged']} duplicate rows** (the same study, "
                  f"outcome, time point and arm in several sheets) were merged"
                  + (f"; **{o['conflicting_values']} conflicting values** are listed below" if o.get("conflicting_values") else "; all duplicated values agreed") + ".")
    om = s.ds.outcome_map
    if not om.empty and (om["scales"] != "\u2014").any():
        md.append("- Outcomes and the scales/assays grouped into them: " + "; ".join(f"**{r['outcome']}**: {r['scales']}" for _, r in om.iterrows()))
    for line in (rep.g_inference or {}).get("summary", []):
        md.append(f"- {line}")
    md += [
          f"- Data types detected: {', '.join(o['data_types_detected']) or 'none recognised'}",
          f"- {o['columns_recognised']} columns recognised automatically; {o['columns_kept_as_characteristics']} kept as study characteristics (possible moderators)",
          f"- {o['text_cells_interpreted']} text cells interpreted (e.g. \"mean (SD)\", \"estimate (95% CI)\", percentages)"]
    if o["p_values_reported_as_thresholds"]:
        md.append(f"- {o['p_values_reported_as_thresholds']} p value(s) reported as thresholds (e.g. p<0.05); these can only give approximate effects")
    if o["multi_arm_candidates"]:
        md.append(f"- {o['multi_arm_candidates']} multi-arm study/outcome group(s) found; MetaPy will ask how to handle the shared control")
    if not rep.plans:
        md.append("\n**No analysis is possible yet:** fewer than 2 studies per outcome have usable statistics. See the column mapping below.")
    else:
        md.append("\n### Possible analyses\nPick **one effect measure per outcome** below. MetaPy has ranked them: the first for each outcome uses the most studies with the fewest approximations.")
    chars = rep.characteristics
    if not chars.empty:
        usable = chars[chars["type"].isin(["categorical", "numeric"])]
        if not usable.empty:
            md.append("\n**Study characteristics available for subgroup analysis / meta-regression:** " +
                      ", ".join(f"{r['characteristic']} ({r['type']})" for _, r in usable.iterrows()))
    return "\n".join(md)


def on_scan(file, show_all, group_constructs=True):
    if file is None:
        return (None, "Upload a .csv, .tsv, .xlsx or .xlsm file first.", _df(None), _df(None), upd(choices=[], value=[]), _df(None), _df(None))
    s = MetaPySession()
    path = file if isinstance(file, str) else getattr(file, "name", file)
    try:
        rep = s.load(path, group_constructs=bool(group_constructs))
    except Exception as exc:
        return (None, f"**Could not read the file:** {exc}", _df(None), _df(None), upd(choices=[], value=[]), _df(None), _df(None))
    md = scan_overview_md(s)
    choices = s.plan_choices(show_all)
    rec = [p["plan_id"] for p in rep.plans if p["recommended"]]
    ev = (rep.g_inference or {}).get("evidence")
    return (s, md, _df(s.ds.column_map), _df(s.ds.parse_notes), upd(choices=choices, value=rec),
            _df(ev.drop(columns=["record_id"]) if isinstance(ev, pd.DataFrame) and not ev.empty else None),
            _df(s.ds.conflicts if not s.ds.conflicts.empty else None))


def on_resume(path):
    if WEB_MODE:
        return None, "Saved sessions are not available in the online version.", upd(choices=[], value=None), upd(), upd(), plan_status(None)
    if not path:
        return None, "Choose a saved session first.", upd(choices=[], value=None), upd(), upd(), plan_status(None)
    try:
        s = MetaPySession.restore(path)
    except Exception as exc:
        return None, f"**Could not resume:** {exc}", upd(choices=[], value=None), upd(), upd(), plan_status(None)
    choices = [(f"{p['outcome']} — {p['measure_label']}", pid) for pid, p in s.plans.items()]
    n_ok = len(_approved_ids(s))
    msg = (f"Resumed session for data file **`{s.ds.source_name}`**: {len(s.plans)} plan(s), {n_ok} approved, "
           f"{len(s.results)} analysed. Continue in Step 2, 3 or 4.")
    return (s, msg, upd(choices=choices, value=choices[0][1] if choices else None),
            upd(choices=plan_choices_status(s), value=_approved_ids(s)), upd(choices=plan_choices_status(s), value=_approved_ids(s)),
            plan_status(s))


def on_show_all(s, show_all):
    if s is None:
        return upd()
    return upd(choices=s.plan_choices(show_all))


def on_select(s, plan_ids):
    if s is None:
        return s, "Scan a file first.", upd(choices=[], value=None)
    if not plan_ids:
        return s, "Tick at least one analysis plan.", upd(choices=[], value=None)
    try:
        s.select_plans(plan_ids)
    except ValueError as exc:
        return s, f"**{exc}**", upd(choices=[], value=None)
    choices = [(f"{p['outcome']} — {p['measure_label']}", pid) for pid, p in s.plans.items()]
    s.prepare_all()
    _autosave(s)
    return s, (f"{len(plan_ids)} plan(s) selected and prepared. Go to **Step 2 · Prepare data**: review each plan with the plan selector, "
               "answer its questions, then approve the plans in the checklist at the bottom."), upd(choices=choices, value=choices[0][1])


def _autosave(s) -> str:
    if WEB_MODE or s is None or s.ds is None:
        return ""
    try:
        return s.save()
    except Exception as exc:  # saving must never break the workflow
        return f"(autosave failed: {exc})"


def plan_choices_status(s) -> list[tuple[str, str]]:
    out = []
    for pid, p in (s.plans if s else {}).items():
        prep = s.preps.get(pid)
        st = "approved" if s.approved.get(pid) else ("prepared, not approved" if prep else "not prepared")
        inc = f", {prep.counts['included']} comparisons" if prep else ""
        out.append((f"{p['outcome']} — {p['measure_label']}  [{st}{inc}]", pid))
    return out


def _approved_ids(s) -> list[str]:
    return [p for p, ok in (s.approved if s else {}).items() if ok]


# ======================================================================== step 2
def plan_status(s) -> pd.DataFrame:
    rows = []
    for pid, p in (s.plans if s else {}).items():
        prep = s.preps.get(pid)
        rows.append({"outcome": p["outcome"], "effect measure": p["measure_label"],
                     "prepared": "yes" if prep else "not yet",
                     "comparisons included": f"{prep.counts['included']} of {prep.counts['rows']}" if prep else "",
                     "open questions": len(prep.questions) if prep else "",
                     "approved": "yes" if s.approved.get(pid) else "no"})
    return pd.DataFrame(rows)


def prep_summary_md(s, pid, note="") -> str:
    """Markdown summary of one plan's data preparation. Used by the Gradio and Streamlit apps."""
    p = s.preps[pid]
    c = p.counts
    md = [f"### Data preparation: {p.plan['outcome']} — {p.measure.label}", f"*Data file: `{s.ds.source_name}`*"]
    if note:
        md.append(f"> {note}")
    if s.withdrawn:
        md.append(f"> **Approval withdrawn for {', '.join(sorted(set(s.withdrawn)))}** because its prepared data changed. Re-approve it below.")
        s.withdrawn = []
    for w in p.warnings:
        md.append(f"> **Note:** {w}" + (" If that is wrong, change the direction question below." if "data file" in w else ""))
    md.append(f"**{c['included']} of {c['rows']} comparisons are included.** Reported directly: {c['reported_directly']} · "
              f"exact calculation: {c['exact']} · approximation: {c['approximation']} · assumption: {c['assumption']} · "
              f"not derivable: {c['not_derivable']}")
    if c["consistency_warnings"]:
        md.append(f"\n**{c['consistency_warnings']} consistency warning(s):**")
        md += [f"- {r['study']}: {r['message']}" for _, r in p.checks.iterrows()]
    if not p.actions.empty:
        md.append("\n**What MetaPy calculated** (full details with every input value are in the derivation log below):")
        for _, a in p.actions.iterrows():
            md.append(f"- {a['what MetaPy did']}: {a['studies']} stud{'y' if a['studies'] == 1 else 'ies'} *({a['type']})*")
    if p.questions:
        md.append("\n**MetaPy needs your decision on the questions below** before you approve.")
    else:
        md.append("\nNo open decisions. Review the table and the derivation log, then approve.")
    return "\n".join(md)


def _prep_outputs(s, pid, note=""):
    p = s.preps[pid]
    md = [prep_summary_md(s, pid, note)]
    studies = p.data[["record_id", "label", "derivation_basis", "approximations_or_assumptions", "yi", "se", "included", "exclusion_reason", "notes"]].rename(
        columns={"label": "study", "yi": f"{p.measure.short} (analysis scale)", "derivation_basis": "how obtained"})
    radios, checks = _question_updates(p.questions)
    req = p.requests
    entry_rows = []
    for _, r in req.iterrows():
        for f in [x for x in str(r["fields_option_1"]).split(",") if x]:
            entry_rows.append({"record_id": r["record_id"], "study": r["study"], "statistic": f, "description": label(f), "value": None})
    entry = pd.DataFrame(entry_rows, columns=["record_id", "study", "statistic", "description", "value"])
    req_md = "All comparisons can be derived." if req.empty else (
        f"**{len(req)} comparison(s) cannot be derived from the data.** MetaPy only asks for more data here. Each row lists the "
        "smallest sets of statistics that would work (any one option is enough). Enter values from the paper in the table "
        "below and click *Add these values*, or leave them excluded.")
    incl_choices = [(f"{r['label']} ({r['record_id']})", r["record_id"]) for _, r in p.data.iterrows()]
    excluded_now = set(s.settings[pid].get("excluded_records", []))
    _autosave(s)
    return ([s, "\n".join(md), _df(studies)] + radios + checks +
            [req_md, _df(req.drop(columns=["fields_option_1"]) if not req.empty else req), entry,
             upd(choices=incl_choices, value=[v for _, v in incl_choices if v not in excluded_now]),
             _df(p.log.drop(columns=["record_id"]) if not p.log.empty else p.log),
             upd(value=False),
             plan_status(s), _g_rows(s, pid),
             upd(choices=plan_choices_status(s), value=_approved_ids(s)),
             upd(choices=plan_choices_status(s), value=_approved_ids(s))])


def _question_updates(questions):
    radios, checks = [], []
    single = [q for q in questions if not q.get("multi")]
    multi = [q for q in questions if q.get("multi")]
    for i in range(N_RADIO):
        if i < len(single):
            q = single[i]
            opts = [(lab, str(j)) for j, (lab, _) in enumerate(q["options"])]
            default = next((str(j) for j, (_, v) in enumerate(q["options"]) if v == q.get("default")), None)
            radios.append(upd(visible=True, label=q["question"], choices=opts, value=default))
        else:
            radios.append(upd(visible=False, choices=[], value=None))
    for i in range(N_CHECK):
        if i < len(multi):
            q = multi[i]
            opts = [(lab, str(j)) for j, (lab, _) in enumerate(q["options"])]
            default = [str(j) for j, (_, v) in enumerate(q["options"]) if v in (q.get("default") or [])]
            checks.append(upd(visible=True, label=q["question"], choices=opts, value=default))
        else:
            checks.append(upd(visible=False, choices=[], value=[]))
    return radios, checks


def _decode_answers(questions, values):
    single = [q for q in questions if not q.get("multi")]
    multi = [q for q in questions if q.get("multi")]
    rv, cv = values[:N_RADIO], values[N_RADIO:N_RADIO + N_CHECK]
    ans = {}
    for q, v in zip(single, rv):
        if v not in (None, ""):
            ans[q["id"]] = q["options"][int(v)][1]
    for q, v in zip(multi, cv):
        ans[q["id"]] = [q["options"][int(x)][1] for x in (v or [])]
    return ans


def _empty_prep(s, msg):
    return [s, msg, _df(None)] + [upd(visible=False)] * (N_RADIO + N_CHECK) + ["", _df(None), pd.DataFrame(), upd(), _df(None), upd(),
                                                                              plan_status(s), pd.DataFrame(), upd(), upd()]


G_CODES = {"ppc2": "ppc2", "post": "endpoint", "change": "change", None: "none"}
G_FROM = {"ppc2": "ppc2", "endpoint": "post", "post": "post", "change": "change", "none": None, "": None}


def _g_rows(s, pid) -> pd.DataFrame:
    """Editable table: how each reported Hedges' g in this plan is used (ppc2 / endpoint / change / none)."""
    p = s.preps.get(pid)
    ev = getattr(s.ds, "g_evidence", None)
    if p is None or not isinstance(ev, pd.DataFrame) or ev.empty:
        return pd.DataFrame(columns=["record_id", "study", "reported g", "evidence", "use reported g as"])
    ev = ev[ev["record_id"].isin(p.data["record_id"])]
    gt = s.settings[pid].get("g_types") or {}
    return pd.DataFrame({"record_id": ev["record_id"], "study": ev["study"], "reported g": ev["reported g"].round(4),
                         "evidence": ev["basis"], "use reported g as": [G_CODES.get(gt.get(r)) for r in ev["record_id"]]})


def on_g_rows(s, pid, table):
    if s is None or not pid or pid not in s.preps:
        return _empty_prep(s, "Run data preparation first.")
    vals = {}
    if isinstance(table, pd.DataFrame):
        for _, r in table.iterrows():
            code = str(r.get("use reported g as", "")).strip().lower()
            if code in G_FROM:
                vals[str(r["record_id"])] = G_FROM[code]
    s.answer(pid, {"g_type_rows": vals})
    return _prep_outputs(s, pid, "Design choices for reported g applied (ppc2 / endpoint / change / none).")


def on_prepare(s, pid):
    if s is None or not pid:
        return _empty_prep(s, "Select a plan in Step 1 first.")
    s.prepare(pid)
    return _prep_outputs(s, pid)


def on_answers(s, pid, *values):
    if s is None or not pid or pid not in s.preps:
        return _empty_prep(s, "Run data preparation first.")
    ans = _decode_answers(s.preps[pid].questions, list(values))
    s.answer(pid, ans)
    return _prep_outputs(s, pid, "Your answers were applied and everything was recalculated.")


def on_add_values(s, pid, table):
    if s is None or not pid or pid not in s.preps:
        return _empty_prep(s, "Run data preparation first.")
    vals: dict[str, dict[str, float]] = {}
    if isinstance(table, pd.DataFrame):
        for _, r in table.iterrows():
            try:
                v = float(r["value"])
            except (TypeError, ValueError):
                continue
            vals.setdefault(str(r["record_id"]), {})[str(r["statistic"])] = v
    if not vals:
        return _prep_outputs(s, pid, "No values were entered.")
    s.add_values(pid, vals)
    return _prep_outputs(s, pid, f"Added researcher-supplied values for {len(vals)} comparison(s); they are recorded as such in the log.")


def on_prepare_all(s, pid):
    if s is None or not s.plans:
        return _empty_prep(s, "Select plan(s) in Step 1 first.")
    s.prepare_all()
    first = pid if pid in s.plans else next(iter(s.plans))
    return _prep_outputs(s, first, f"All {len(s.plans)} plan(s) prepared. Use the plan selector to review each one; the status table shows progress.")


def on_exclude(s, pid, included):
    """The UI lists comparisons to INCLUDE (all ticked by default); unticked ones are excluded."""
    if s is None or not pid or pid not in s.preps:
        return _empty_prep(s, "Run data preparation first.")
    all_ids = list(s.preps[pid].data["record_id"])
    keep = set(included or [])
    excluded = [r for r in all_ids if r not in keep]
    if len(all_ids) - len(excluded) < 2:
        return _prep_outputs(s, pid, "Not applied: at least 2 comparisons must stay included for a meta-analysis. "
                                     "Tick the comparisons you want to KEEP.")
    s.exclude(pid, excluded)
    return _prep_outputs(s, pid, f"{len(excluded)} comparison(s) excluded by you." if excluded else "All comparisons included.")


def on_approve_many(s, ticked, confirmed, pid=None, *qvals):
    """Approve exactly the ticked plans. Returns state, message, extras, more, status, analyse-group, approve-group.
    Choices shown on screen for the current plan are applied first if the researcher did not click 'Apply'."""
    if s is None or not s.plans:
        return s, "Select plans in Step 1 first.", upd(), upd(), plan_status(s), upd(), upd()
    if confirmed and pid in s.preps and pid not in getattr(s, "answered", set()) and qvals and s.preps[pid].questions:
        ans = _decode_answers(s.preps[pid].questions, list(qvals))
        if ans:
            s.answer(pid, ans)
    if not confirmed:
        return (s, "Tick **I have reviewed ...** to confirm, then click again.", upd(), upd(), plan_status(s), upd(),
                upd(choices=plan_choices_status(s), value=_approved_ids(s)))
    ok, refused = s.approve_many(list(ticked or []))
    msg = []
    auto = getattr(s, "last_auto_answered", [])
    if auto:
        msg.append("The answers shown on screen (MetaPy's recommendations) were applied to " +
                   ", ".join(f"**{s.plans[p]['outcome']}**" for p in auto) + " before approval.")
    if ok:
        msg.append("Approved: " + ", ".join(f"**{s.plans[p]['outcome']}**" for p in ok) + ".")
    if refused:
        msg.append("Not approved (fewer than 2 comparisons included): " + ", ".join(s.plans[p]["outcome"] for p in refused) + ".")
    waiting = [s.plans[p]["outcome"] for p in s.plans if not s.approved.get(p)]
    msg.append(f"Not approved yet: {', '.join(waiting)}." if waiting else "All plans approved. Continue to **Step 3 · Analyse & review**.")
    path = _autosave(s)
    if path:
        msg.append(f"Session saved ({path}).")
    choices = [(a["label"] + f"  (needs {a['needs']})", a["key"]) for a in s.extra_choices()]
    return (s, " ".join(msg), upd(choices=choices), upd(choices=choices), plan_status(s),
            upd(choices=plan_choices_status(s), value=_approved_ids(s)),
            upd(choices=plan_choices_status(s), value=_approved_ids(s)))


def on_approve(s, pid, confirmed):
    if s is None or not pid or pid not in s.preps:
        return s, "Run data preparation first.", upd()
    if not confirmed:
        return s, "Tick the confirmation box to approve.", upd()
    if s.preps[pid].counts["included"] < 2:
        return s, (f"**Cannot approve {s.plans[pid]['outcome']}:** only {s.preps[pid].counts['included']} comparison(s) are included; "
                   "at least 2 are needed. Check the 'comparisons included' list and the reasons in the table."), upd()
    s.approve(pid)
    waiting = [p for p, ok in s.approved.items() if not ok]
    msg = f"Approved: **{s.plans[pid]['outcome']}**."
    msg += (f" Still to prepare/approve: {', '.join(s.plans[p]['outcome'] for p in waiting)}." if waiting
            else " All plans approved. Continue to **Step 3 · Analyse**.")
    choices = [(a["label"] + f"  (needs {a['needs']})", a["key"]) for a in s.extra_choices()]
    return s, msg, upd(choices=choices)


# ======================================================================== step 3-4
def results_md(s, which=None, note="") -> str:
    """Markdown summary of the results (all plans) and the additional analyses of one plan."""
    pid = which if which in s.results else next(iter(s.results))
    r = s.results[pid]
    md = []
    if note:
        md.append(f"> {note}")
    for p, res in s.results.items():
        md.append(f"### {res.prep.plan['outcome']} — {res.measure.label}")
        md += [f"- {x}" for x in interpret(res)]
        notinc = res.prep.data[~res.prep.data["included"]]
        if len(notinc):
            reasons = notinc["exclusion_reason"].value_counts()
            md.append(f"- {len(notinc)} comparison(s) not included: " + "; ".join(f"{r} ({n})" for r, n in reasons.items()))
    ex = []
    for key, val in r.extras.items():
        name = EXTRA_LABELS.get(key, key.replace("subgroup::", "Subgroup: ").replace("metareg::", "Meta-regression: "))
        if "status" in val and len(val) == 1:
            ex.append(f"- **{name}:** {val['status']}")
        elif key.startswith("subgroup::") and "p_between" in val:
            ex.append(f"- **{name}:** test for subgroup differences Q = {val['Q_between']:.2f}, p = {val['p_between']:.3f}")
        elif key.startswith("metareg::") and val.get("status") == "Estimated":
            c = val["coefficients"].iloc[1:]
            ex.append(f"- **{name}:** " + "; ".join(f"{t}: b = {b:.3f}, p = {pv:.3f}" for t, b, pv in zip(c["term"], c["estimate"], c["p"]))
                      + f"; R² = {val['R2_percent']:.0f}%" + (f" ⚠ {val['warning']}" if val.get("warning") else ""))
        elif key in ("egger", "begg") and "p" in val:
            ex.append(f"- **{name}:** p = {val['p']:.3f} ({val['status']})")
        elif key == "trimfill" and "k0" in val:
            a = val["adjusted"]
            ex.append(f"- **{name}:** {val['k0']} imputed; adjusted {r.measure.short} = {r.measure.transform(a['estimate']):.2f} "
                      f"({r.measure.transform(a['ci_lo']):.2f} to {r.measure.transform(a['ci_hi']):.2f})")
        elif "result" in val:
            rr = val["result"]
            if rr.get("estimate") is not None:
                ex.append(f"- **{name}:** {r.measure.short} = {r.measure.transform(rr['estimate']):.2f} "
                          f"({r.measure.transform(rr['ci_lo']):.2f} to {r.measure.transform(rr['ci_hi']):.2f}), k = {val['k']}; {val.get('note', '')}")
            else:
                ex.append(f"- **{name}:** {rr.get('status')}")
        elif "table" in val:
            ex.append(f"- **{name}:** computed (table in the report)")
    if ex:
        md.append("\n### Additional analyses\n" + "\n".join(ex))
    md.append("\n**Are these results OK?** Use the review options below.")
    return "\n".join(md)


def _results_outputs(s, which=None, note=""):
    if not s.results:
        return s, "No results yet.", _df(None), None, None, "", upd()
    pid = which if which in s.results else next(iter(s.results))
    r = s.results[pid]
    md = [results_md(s, pid, note)]
    forest = forest_plot(r) if r.main.get("status") == "Pooled" else None
    funnel = funnel_plot(r, trimfill=r.extras.get("trimfill") if "trimfill" in r.extras and "filled_y" in r.extras["trimfill"] else None) if len(r.studies) >= 10 and r.main.get("status") == "Pooled" else None
    choices = [(f"{res.prep.plan['outcome']} — {res.measure.label}", p) for p, res in s.results.items()]
    table = pd.concat([summary_table(res) for res in s.results.values()], ignore_index=True)
    _autosave(s)
    return s, "\n".join(md), _df(table), forest, funnel, "", upd(choices=choices, value=pid)


def on_analyse(s, model, tau, ci, extras, ticked=None, confirm_new=False):
    if s is None or not s.plans:
        return s, "Select and prepare plans first (Steps 1 and 2).", _df(None), None, None, "", upd()
    ticked = list(ticked) if ticked else _approved_ids(s)
    if not ticked:
        return s, "Tick at least one plan to analyse.", _df(None), None, None, "", upd()
    new = [p for p in ticked if not s.approved.get(p)]
    if new:
        if not confirm_new:
            return (s, "These ticked plans are not approved yet: " + ", ".join(f"**{s.plans[p]['outcome']}**" for p in new) +
                    ". Review them in Step 2, or tick *I approve the preparation of the ticked plans* here, then run again.",
                    _df(plan_status(s)), None, None, "", upd())
        _, refused = s.approve_many(sorted(set(_approved_ids(s)) | set(new)))
        ticked = [p for p in ticked if p not in refused]
    opts = {"model": "random" if model.startswith("Random") else "fixed", "tau_method": tau.split()[0],
            "ci_method": "HK" if ci.startswith("Hartung") else "Wald", "extras": list(extras or [])}
    s.analyse(opts, ticked)
    left = [s.plans[p]["outcome"] for p in s.plans if p not in s.results]
    return _results_outputs(s, note=(f"Analysed {len(s.results)} plan(s)." +
                                     (f" Not included in this run: {', '.join(left)}." if left else "")))


def on_show_result(s, pid):
    if s is None or not s.results:
        return s, "", _df(None), None, None, "", upd()
    return _results_outputs(s, pid)


def on_review(s, decision, more, model, tau, ci, pid):
    if s is None or not s.results:
        return (s, "Run the analysis first.", _df(None), None, None, "", upd())
    if decision.startswith("Yes"):
        s.accept()
        out = list(_results_outputs(s, pid, "Results accepted. Continue to **Step 4 · Tables & figures**."))
        return tuple(out)
    if decision.startswith("Add"):
        if not more:
            return _results_outputs(s, pid, "Tick the additional analyses you want, then click again.")
        s.extend(list(more))
        return _results_outputs(s, pid, f"Added {len(more)} analysis/analyses.")
    if decision.startswith("Change"):
        opts = {"model": "random" if model.startswith("Random") else "fixed", "tau_method": tau.split()[0],
                "ci_method": "HK" if ci.startswith("Hartung") else "Wald"}
        s.analyse({**opts, "extras": s.options.get("extras", [])})
        return _results_outputs(s, pid, "Re-analysed with the new model settings.")
    return _results_outputs(s, pid, "Go back to **Step 2 · Prepare data**, change decisions, re-approve, then run the analysis again.")


# ======================================================================== step 5-6
def _prisma_dict(values):
    """PRISMA inputs -> dict, or None when the researcher has not switched the (optional) diagram on."""
    d = dict(zip(PRISMA_KEYS, values))
    if not d.pop("_on", False):
        return None
    return d


def prisma_reset_values(n_studies=None):
    """Blank PRISMA form (used for every newly read data file)."""
    return [False] + ["" for _ in PRISMA_KEYS[1:]]


def on_prisma_fill(s):
    """Write studies / comparisons / participants from the analysis into the 'Studies included' box."""
    if s is None or not s.results:
        return upd(), "Run the analysis first (Step 3); then MetaPy can fill this box."
    a = s.prisma_autofill()
    txt = (f"(n = {a['studies_included']} studies)\nComparisons included in meta-analysis (n = {a['comparisons_included']})\n"
           f"Total sample size (N = {a['participants']} participants)")
    return upd(value=txt), ("Filled from your analysis (shared control groups counted once). Edit the text freely; it is printed as written.")


def preview_md(s, rep, prisma, n_figures) -> str:
    """Markdown outline of a report before export."""
    md = ["### Report preview", f"**Data file: `{s.ds.source_name}`**: if this is not the file you meant, go back to Step 1 and read the right file.",
          f"**{n_figures} figure(s)** and **{sum(len(sec.tables) for sec in rep.sections)} table(s)** selected."]
    if prisma is None:
        md.append("PRISMA flow diagram: not included (optional; switch it on in the PRISMA section if you want it).")
    elif not prisma_given(prisma):
        md.insert(1, "> **PRISMA diagram is switched on but every box is empty**, so it is left out. Type in the PRISMA boxes and preview again.")
    else:
        md += [f"> **PRISMA check:** {x}" for x in prisma_checks(prisma)]
    for sec in rep.sections:
        md.append(f"\n**{sec.heading}**")
        md += [f"\n{t}" for t in sec.text]
        md += [f"- table: {n} ({len(df)} rows)" for n, df in sec.tables]
    return "\n".join(md)


def on_preview(s, tables, figures, texts, theme, sort, title, *prisma_vals):
    if s is None or not s.results:
        return [], "Run and accept the analysis first."
    prisma = _prisma_dict(prisma_vals)
    rep = s.report(tables or [], figures or [], texts or [], prisma, theme, title or "Meta-analysis report",
                   None if sort == "As in data" else sort.lower().split()[-1])
    tmp = tempfile.mkdtemp(prefix="metapy_preview_")
    gallery = []
    i = 0
    for sec in rep.sections:
        for name, fig in sec.figures:
            i += 1
            p = os.path.join(tmp, f"fig{i:02d}.png")
            fig.savefig(p, dpi=110, bbox_inches="tight")
            gallery.append((p, f"{sec.heading}: {name}"))
    import matplotlib.pyplot as plt
    plt.close("all")
    md = [preview_md(s, rep, prisma, len(gallery))]
    return gallery, "\n".join(md)


def on_export(s, formats, tables, figures, texts, theme, sort, title, *prisma_vals):
    if s is None or not s.results:
        return None, "Run and accept the analysis first."
    if not formats:
        return None, "Choose at least one file format."
    prisma = _prisma_dict(prisma_vals)
    out = tempfile.mkdtemp(prefix="metapy_export_")
    paths = s.export(list(formats), out, tables or [], figures or [], texts or [], prisma, theme,
                     title or "Meta-analysis report", forest_sort=None if sort == "As in data" else sort.lower().split()[-1])
    msg = (f"Exported {len(paths)} file(s), built from data file **`{s.ds.source_name}`** "
           f"({', '.join(r.prep.plan['outcome'] for r in s.results.values())}). The original data file was not modified.")
    if not s.accepted:
        msg += " Note: results were exported before you accepted them in Step 3."
    return paths, msg


# ======================================================================== UI
def on_nma_sources(s):
    if s is None or s.ds is None:
        return upd(choices=[]), "Load a file first (Step 1)."
    src = s.nma_sources()
    if not src:
        return upd(choices=[]), ("No network data found. Add a sheet in arm format (study, treatment, mean, sd, n) or contrast format "
                                 "(study, treat1, treat2, TE, seTE), or prepare an outcome whose arms are named in an 'intervention_arm' column.")
    return upd(choices=src, value=src[0][1]), f"{len(src)} possible network data source(s) found."


def on_nma(s, src, measure, model, small, ref):
    empty = (s, "", _df(None), _df(None), _df(None), None, None)
    if s is None or not src:
        return (s, "Choose the data for the network first (press *Find network data*).") + empty[2:]
    try:
        r = s.nma(src, measure, ref.strip() or None, "random" if model.startswith("Random") else "fixed",
                  "good" if small.startswith("Smaller") else "bad")
    except Exception as exc:
        return (s, f"Network meta-analysis could not be run: {exc}") + empty[2:]
    from . import nma as N
    res, Q = r["result"], r["result"].Q
    msg = (f"**{len(res.treatments)} treatments, {res.extra['n_studies']} studies ({res.extra['multi_arm_studies']} multi-arm)**, reference "
           f"**{res.reference}**, τ² = {res.tau2:.3f}. Inconsistency between designs: Q = {Q['Q_inconsistency']:.2f}, df = {Q['df_inconsistency']}"
           + (f", p = {Q['p_inconsistency']:.3f}." if Q["df_inconsistency"] > 0 else " (no closed loops: consistency cannot be checked).")
           + ("\n\n" + "\n".join("- " + n for n in r["notes"]) if r["notes"] else ""))
    return (s, msg, _df(r["pscores"].round(3)), _df(r["league"].reset_index(drop=True)), _df(r["split"].round(3)),
            N.network_plot(res), N.forest_vs_reference(res, r.get("transform"), N.MEASURES_NMA.get(res.contrasts.measure, "")))


def build_app():
    gr = _gr()
    with gr.Blocks(title=f"MetaPy {__version__}") as app:
        state = gr.State(None)
        gr.Markdown(f"# MetaPy {__version__}\nAn intelligent, transparent meta-analysis assistant. Upload your extraction sheet; "
                    "MetaPy works out what is possible, derives missing statistics (and tells you exactly how), and builds your report. "
                    "**Your original file is never modified.** Use de-identified, aggregate data only.")
        with gr.Tabs():
            # ---------------- step 1
            with gr.Tab("1 · Upload & choose analysis"):
                f = gr.File(label="Data file (.csv, .tsv, .xlsx, .xlsm)", file_types=[".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"], type="filepath")
                scan_btn = gr.Button("Read file and find possible analyses", variant="primary")
                if WEB_MODE:
                    gr.Markdown(WEB_NOTICE)
                with gr.Accordion("Or resume a saved session (MetaPy saves your work automatically)", open=False, visible=not WEB_MODE):
                    with gr.Row():
                        saved_dd = gr.Dropdown(label="Saved sessions", choices=[] if WEB_MODE else saved_sessions())
                        refresh_btn = gr.Button("Refresh list")
                    resume_btn = gr.Button("Resume selected session")
                overview = gr.Markdown()
                group_cb = gr.Checkbox(label="Combine different scales of the same construct into one outcome (e.g. HAM-A, BAI, GAD-7 \u2192 Anxiety; hsCRP + CRP \u2192 CRP)", value=True)
                show_all = gr.Checkbox(label="Show every possible effect measure (including cross-family conversions)", value=False)
                plans = gr.CheckboxGroup(label="Analysis plans (one per outcome)", choices=[])
                select_btn = gr.Button("Use the selected plan(s)", variant="primary")
                select_msg = gr.Markdown()
                with gr.Accordion("Reported Hedges' g: what design is it? (checked against your raw data)", open=False):
                    gev = gr.Dataframe(interactive=False)
                with gr.Accordion("Conflicting values between sheets", open=False):
                    conf = gr.Dataframe(interactive=False)
                with gr.Accordion("How MetaPy read your columns", open=False):
                    colmap = gr.Dataframe(label="Column mapping", interactive=False)
                    parsed = gr.Dataframe(label="Text cells interpreted", interactive=False)
            # ---------------- step 2
            with gr.Tab("2 · Prepare data"):
                plan_dd = gr.Dropdown(label="Analysis plan", choices=[])
                with gr.Row():
                    prep_all_btn = gr.Button("Run data preparation for ALL plans", variant="primary")
                    prep_btn = gr.Button("Re-run for the selected plan")
                status_df = gr.Dataframe(label="Progress of every plan", interactive=False)
                prep_md = gr.Markdown()
                studies_df = gr.Dataframe(label="Comparisons and how each effect size was obtained", interactive=False)
                with gr.Group():
                    radios = [gr.Radio(visible=False, choices=[], label=f"Question {i + 1}") for i in range(N_RADIO)]
                    checks = [gr.CheckboxGroup(visible=False, choices=[], label=f"Options {i + 1}") for i in range(N_CHECK)]
                    ans_btn = gr.Button("Apply my answers and recalculate")
                with gr.Accordion("Reported Hedges' g in this plan: how each one is used (edit the last column: ppc2 / endpoint / change / none)", open=False):
                    g_rows = gr.Dataframe(interactive=True)
                    g_btn = gr.Button("Apply design choices")
                req_md = gr.Markdown()
                req_df = gr.Dataframe(label="Data still needed (any one option per study is enough)", interactive=False)
                entry_df = gr.Dataframe(label="Enter extra statistics from the papers (fill the value column)", interactive=True,
                                        headers=["record_id", "study", "statistic", "description", "value"])
                add_btn = gr.Button("Add these values and recalculate")
                excl = gr.CheckboxGroup(label="Comparisons INCLUDED in this analysis (untick to exclude; recorded in the audit trail)", choices=[])
                excl_btn = gr.Button("Apply inclusion choices")
                with gr.Accordion("Derivation log: every value MetaPy calculated, with formula, inputs and citation", open=False):
                    log_df = gr.Dataframe(interactive=False)
                gr.Markdown("### Approve plans\nTick every plan whose preparation you have reviewed. Approvals are kept when you switch plans "
                            "or re-run preparation; a plan is only un-approved if its prepared data change (MetaPy tells you).")
                approve_group = gr.CheckboxGroup(label="Plans approved for analysis", choices=[])
                confirm = gr.Checkbox(label="I have reviewed everything MetaPy calculated for the ticked plans and approve it", value=False)
                approve_btn = gr.Button("Save approvals and continue", variant="primary")
                approve_md = gr.Markdown()
            # ---------------- step 3
            with gr.Tab("3 · Analyse & review"):
                with gr.Row():
                    model = gr.Radio(["Random effects", "Fixed effect"], value="Random effects", label="Model")
                    tau = gr.Radio(["REML (recommended)", "DL (DerSimonian-Laird)", "PM (Paule-Mandel)"], value="REML (recommended)", label="τ² estimator")
                    ci = gr.Radio(["Hartung-Knapp (recommended)", "Wald (normal)"], value="Hartung-Knapp (recommended)", label="Confidence interval")
                analyse_group = gr.CheckboxGroup(label="Plans to analyse (all approved plans are ticked)", choices=[])
                confirm3 = gr.Checkbox(label="I approve the preparation of the ticked plans that are not yet approved", value=False)
                extras = gr.CheckboxGroup(label="Additional analyses (optional; you can add more after seeing the results)", choices=[])
                run_btn = gr.Button("Run analysis", variant="primary")
                res_md = gr.Markdown()
                res_tbl = gr.Dataframe(label="Pooled results", interactive=False)
                res_pick = gr.Dropdown(label="Show plots for", choices=[])
                with gr.Row():
                    forest = gr.Plot(label="Forest plot")
                    funnel = gr.Plot(label="Funnel plot")
                spare = gr.Markdown(visible=False)
                gr.Markdown("### Review")
                decision = gr.Radio(["Yes, the results are OK; accept them", "Add more analyses (tick them below)",
                                     "Change model settings (set them above)", "Go back and change data preparation"],
                                    value="Yes, the results are OK; accept them", label="Are these results OK?")
                more = gr.CheckboxGroup(label="More analyses", choices=[])
                review_btn = gr.Button("Continue", variant="primary")
                with gr.Accordion("Network meta-analysis (optional: three or more treatments compared across studies)", open=False):
                    gr.Markdown("Use a sheet of your file in **arm format** (study, treatment, mean, sd, n — or events, n) or **contrast format** "
                                "(study, treat1, treat2, TE, seTE), or a prepared outcome whose studies compare several named intervention arms "
                                "with a control. The network result is added to the report.")
                    with gr.Row():
                        nma_src = gr.Dropdown(label="Data for the network", choices=[])
                        nma_refresh = gr.Button("Find network data")
                    with gr.Row():
                        nma_measure = gr.Radio(["SMD", "MD", "OR", "RR", "generic"], value="SMD", label="Effect measure (arm-format data)")
                        nma_model = gr.Radio(["Random effects", "Common effect"], value="Random effects", label="Model")
                        nma_small = gr.Radio(["Smaller values are better", "Larger values are better"], value="Smaller values are better", label="Ranking direction")
                        nma_ref = gr.Textbox(label="Reference treatment (blank = most connected)", value="")
                    nma_btn = gr.Button("Run network meta-analysis", variant="primary")
                    nma_md = gr.Markdown()
                    nma_rank = gr.Dataframe(label="Ranking (P-scores)", interactive=False)
                    nma_league = gr.Dataframe(label="League table (row vs column)", interactive=False)
                    nma_split = gr.Dataframe(label="Direct vs indirect evidence", interactive=False)
                    with gr.Row():
                        nma_graph = gr.Plot(label="Network graph")
                        nma_forest = gr.Plot(label="Estimates vs reference")
            # ---------------- step 4
            with gr.Tab("4 · Tables & figures"):
                tbls = gr.CheckboxGroup(label="Tables", choices=[(v, k) for k, v in TABLE_CHOICES.items()], value=DEFAULT_TABLES)
                figs = gr.CheckboxGroup(label="Figures", choices=[(v, k) for k, v in FIGURE_CHOICES.items()], value=DEFAULT_FIGURES)
                txts = gr.CheckboxGroup(label="Text", choices=[(v, k) for k, v in TEXT_CHOICES.items()], value=list(TEXT_CHOICES))
                with gr.Row():
                    theme = gr.Dropdown(["navy", "teal", "purple", "monochrome", "orange"], value="navy", label="Figure colour theme")
                    sort = gr.Dropdown(["As in data", "Sort by effect", "Sort by year"], value="As in data", label="Forest plot order")
                    title = gr.Textbox(value="Meta-analysis report", label="Report title")
                gr.Markdown("### PRISMA 2020 flow diagram (optional)\nSwitch it on only if you want the diagram in your report. "
                            "Each box below is one box of the diagram: type or paste anything, on as many lines as you like. "
                            "The heading is printed for you and your text is printed exactly as written (start a line with two spaces to indent it). "
                            "Empty boxes show '(n = …)'.")
                with gr.Accordion("PRISMA 2020 flow diagram", open=False):
                    prisma_on = gr.Checkbox(label="Include a PRISMA 2020 flow diagram in the report", value=False)
                    pbox = {}
                    gr.Markdown("**Identification**")
                    with gr.Row():
                        for k in ("box_identified", "box_removed"):
                            pbox[k] = gr.Textbox(label=dict((kk, h) for kk, h, *_ in PRISMA_BOXES)[k], lines=6, placeholder=PRISMA_HINTS[k])
                    gr.Markdown("**Screening**")
                    for left, right in (("box_screened", "box_excluded"), ("box_sought", "box_not_retrieved"), ("box_assessed", "box_reports_excluded")):
                        with gr.Row():
                            for k in (left, right):
                                pbox[k] = gr.Textbox(label=dict((kk, h) for kk, h, *_ in PRISMA_BOXES)[k], lines=5 if k == "box_reports_excluded" else 2,
                                                     placeholder=PRISMA_HINTS[k])
                    gr.Markdown("**Included**")
                    pbox["box_included"] = gr.Textbox(label="Studies included in review", lines=3, placeholder=PRISMA_HINTS["box_included"])
                    fill_btn = gr.Button("Fill 'Studies included' from my analysis")
                    fill_md = gr.Markdown()
                    with gr.Accordion("Identification via other methods (optional third column: websites, citation searching, …)", open=False):
                        for k, h in PRISMA_OTHER_BOXES:
                            pbox[k] = gr.Textbox(label=f"Other methods: {h}", lines=2, placeholder="Type or paste anything")
                    pbox["note"] = gr.Textbox(label="Note:", lines=3, placeholder="Optional. Printed under the diagram, e.g. how comparisons or reports were counted.")
                    prisma_inputs = [prisma_on] + [pbox[k] for k in PRISMA_KEYS[1:]]
                prev_btn = gr.Button("Preview report", variant="primary")
                gallery = gr.Gallery(label="Figures", columns=2, height="auto")
                prev_md = gr.Markdown()
            # ---------------- step 5
            with gr.Tab("5 · Export"):
                fmts = gr.CheckboxGroup(label="File formats", choices=[(v, k) for k, v in FORMATS.items()], value=["docx", "xlsx"])
                exp_btn = gr.Button("Export", variant="primary")
                exp_files = gr.File(label="Download", file_count="multiple")
                exp_md = gr.Markdown()

        prep_outputs = [state, prep_md, studies_df] + radios + checks + [req_md, req_df, entry_df, excl, log_df, confirm, status_df, g_rows,
                                                                          approve_group, analyse_group]
        res_outputs = [state, res_md, res_tbl, forest, funnel, spare, res_pick]

        # Everything below the scan is cleared when a file is read, so nothing from a previous file can carry over.
        reset_targets = ([select_msg, plan_dd, prep_md, studies_df, req_md, req_df, entry_df, excl, log_df, confirm, status_df, g_rows,
                          approve_group, approve_md, analyse_group, confirm3, extras, more, res_md, res_tbl, res_pick, forest, funnel,
                          gallery, prev_md, exp_files, exp_md, fill_md] + radios + checks + prisma_inputs)

        def _scan(file, sa, grp):
            out = list(on_scan(file, sa, grp))
            s = out[0]
            n = s.scan_report.overview["studies"] if s is not None and s.scan_report else None
            fp = f" Data fingerprint: `{s.ds.fingerprint[:12]}`. Everything from any previously loaded file has been cleared." if s is not None else ""
            resets = [fp, gr.update(choices=[], value=None), "", None, "", None, None, gr.update(choices=[], value=[]), None,
                      gr.update(value=False), None, None, gr.update(choices=[], value=[]), "", gr.update(choices=[], value=[]),
                      gr.update(value=False), gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), "", None,
                      gr.update(choices=[], value=None), None, None, [], "", None, "", ""]
            resets += [gr.update(visible=False, choices=[], value=None)] * len(radios) + [gr.update(visible=False, choices=[], value=[])] * len(checks)
            resets += prisma_reset_values(n)
            return out + resets
        scan_btn.click(_scan, [f, show_all, group_cb], [state, overview, colmap, parsed, plans, gev, conf] + reset_targets)
        fill_btn.click(on_prisma_fill, [state], [pbox["box_included"], fill_md])
        g_btn.click(on_g_rows, [state, plan_dd, g_rows], prep_outputs)
        show_all.change(on_show_all, [state, show_all], [plans])
        select_btn.click(on_select, [state, plans], [state, select_msg, plan_dd])
        prep_btn.click(on_prepare, [state, plan_dd], prep_outputs)
        prep_all_btn.click(on_prepare_all, [state, plan_dd], prep_outputs)
        def _switch(s, p):
            if s is None or not p or p not in s.plans:
                return _empty_prep(s, "")
            return _prep_outputs(s, p) if p in s.preps else on_prepare(s, p)
        plan_dd.change(_switch, [state, plan_dd], prep_outputs)
        ans_btn.click(on_answers, [state, plan_dd] + radios + checks, prep_outputs)
        add_btn.click(on_add_values, [state, plan_dd, entry_df], prep_outputs)
        excl_btn.click(on_exclude, [state, plan_dd, excl], prep_outputs)

        def _approve(s, p, c):
            s2, msg, ch = on_approve(s, p, c)
            return s2, msg, ch, ch, plan_status(s2)
        approve_btn.click(on_approve_many, [state, approve_group, confirm, plan_dd] + radios + checks, [state, approve_md, extras, more, status_df, analyse_group, approve_group])
        run_btn.click(on_analyse, [state, model, tau, ci, extras, analyse_group, confirm3], res_outputs)
        refresh_btn.click(lambda: gr.update(choices=[] if WEB_MODE else saved_sessions()), [], [saved_dd])
        resume_btn.click(on_resume, [saved_dd], [state, select_msg, plan_dd, approve_group, analyse_group, status_df])
        res_pick.change(on_show_result, [state, res_pick], res_outputs)
        nma_refresh.click(on_nma_sources, [state], [nma_src, nma_md])
        nma_btn.click(on_nma, [state, nma_src, nma_measure, nma_model, nma_small, nma_ref],
                      [state, nma_md, nma_rank, nma_league, nma_split, nma_graph, nma_forest])
        review_btn.click(on_review, [state, decision, more, model, tau, ci, res_pick], res_outputs)
        prev_btn.click(on_preview, [state, tbls, figs, txts, theme, sort, title] + prisma_inputs, [gallery, prev_md])
        exp_btn.click(on_export, [state, fmts, tbls, figs, txts, theme, sort, title] + prisma_inputs, [exp_files, exp_md])
    return app


def launch(share: bool | None = None, **kw):
    if WEB_MODE:          # hosted website: the host provides the public address
        kw.setdefault("server_name", "0.0.0.0")
        kw.setdefault("max_file_size", "25mb")
        share = False
    elif share is None:
        share = os.environ.get("METAPY_SHARE", "true").lower() not in ("0", "false", "no")
    app = build_app()
    app.queue(default_concurrency_limit=4)
    return app.launch(share=share, **kw)
