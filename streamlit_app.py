"""MetaPy website (Streamlit). Run locally with:  streamlit run streamlit_app.py

The same engine and explanations as the Colab / Gradio app, in five steps. Website mode: the uploaded file and all
results live only in this browser session's memory; nothing is saved on the server or shared with other users.
"""
from __future__ import annotations

import mimetypes
import os
import shutil
import tempfile
import zlib
from pathlib import Path

os.environ.setdefault("METAPY_WEB", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from metapy import __version__  # noqa: E402
from metapy.analysis import summary_table  # noqa: E402
from metapy.app import (DEFAULT_FIGURES, DEFAULT_TABLES, G_FROM, PRISMA_HINTS, PRISMA_KEYS, WEB_NOTICE, _g_rows,  # noqa: E402
                        _prisma_dict, plan_status, prep_summary_md, preview_md, results_md, scan_overview_md)
from metapy.export import FORMATS, display_frame  # noqa: E402
from metapy.fields import label as field_label  # noqa: E402
from metapy.figures import PRISMA_BOXES, PRISMA_OTHER_BOXES, forest_plot, funnel_plot  # noqa: E402
from metapy.report import FIGURE_CHOICES, TABLE_CHOICES, TEXT_CHOICES  # noqa: E402
from metapy.session import MetaPySession  # noqa: E402

st.set_page_config(page_title="MetaPy: transparent meta-analysis", page_icon="📊", layout="wide")
ss = st.session_state
for _k, _v in (("s", None), ("gen", 0), ("tmpdir", None), ("flash", {}), ("nma_src", None), ("nma_last", None), ("exports", [])):
    if _k not in ss:
        ss[_k] = _v


# ------------------------------------------------------------------ helpers
def K(name: str) -> str:
    """Widget key that is new for every data file, so nothing from a previous file carries over."""
    return f"{name}__{ss.gen}"


def flash(tab: str, kind: str, text: str) -> None:
    ss.flash[tab] = (kind, text)


def show_flash(tab: str) -> None:
    item = ss.flash.pop(tab, None)
    if item:
        getattr(st, item[0])(item[1])


def table(df: pd.DataFrame | None, **kw) -> None:
    if df is None or len(df) == 0:
        st.caption("Nothing to show.")
    else:
        st.dataframe(display_frame(df), width="stretch", hide_index=True, **kw)


def show_fig(fig) -> None:
    if fig is not None:
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def plan_label(s, pid: str) -> str:
    p = s.plans[pid]
    return f"{p['outcome']} — {p['measure_label']}"


def approved_ids(s) -> list[str]:
    return [p for p, ok in s.approved.items() if ok]


def new_file(upload, group: bool) -> None:
    """Read a newly uploaded file into a brand-new session (previous data are discarded)."""
    if ss.tmpdir:
        shutil.rmtree(ss.tmpdir, ignore_errors=True)
    ss.tmpdir = tempfile.mkdtemp(prefix="metapy_upload_")
    path = Path(ss.tmpdir) / Path(upload.name).name
    path.write_bytes(upload.getvalue())
    s = MetaPySession()
    s.load(str(path), group_constructs=group)
    ss.s, ss.gen = s, ss.gen + 1
    ss.flash, ss.nma_src, ss.nma_last, ss.exports = {}, None, None, []


s = ss.s

# ------------------------------------------------------------------ page
st.title("📊 MetaPy")
st.caption(f"Transparent meta-analysis · version {__version__} · Sadia Nadeem · MIT License")
st.info(WEB_NOTICE.replace("To save and resume work, use the Colab or local version.",
                           "To save and resume work, use the Colab or local version of MetaPy."))
if s is not None and s.ds is not None:
    st.markdown(f"**Data file:** `{s.ds.source_name}` · fingerprint `{s.ds.fingerprint[:12]}` · "
                f"{len(s.plans)} plan(s) selected · {len(approved_ids(s))} approved · {len(s.results)} analysed")

tabs = st.tabs(["1 · Upload & choose", "2 · Prepare data", "3 · Analyse & review", "4 · Tables & figures", "5 · Export"])

# ================================================================== 1 upload
with tabs[0]:
    show_flash("1")
    up = st.file_uploader("Data file (.csv, .tsv, .xlsx, .xlsm)", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"], key="uploader")
    group = st.checkbox("Combine different scales of the same construct into one outcome "
                        "(e.g. HAM-A, BAI, GAD-7 → Anxiety; hsCRP + CRP → CRP)", value=True, key="group_constructs")
    if st.button("Read file and find possible analyses", type="primary", key="btn_read", disabled=up is None):
        try:
            with st.spinner("Reading the file…"):
                new_file(up, group)
            flash("1", "success", "File read. Everything from any previously loaded file has been cleared.")
        except Exception as exc:  # show the reason, keep the page usable
            ss.s = None
            flash("1", "error", f"Could not read the file: {exc}")
        st.rerun()

    if s is not None and s.scan_report is not None:
        st.markdown(scan_overview_md(s))
        show_all = st.checkbox("Show every possible effect measure (including cross-family conversions)", key=K("show_all"))
        choices = s.plan_choices(show_all)
        labels = dict((pid, lab) for lab, pid in choices)
        rec = [p["plan_id"] for p in s.scan_report.plans if p["recommended"] and p["plan_id"] in labels]
        picked = st.multiselect("Analysis plans (one effect measure per outcome)", list(labels), default=rec,
                                format_func=lambda pid: labels[pid], key=K(f"plans_{int(show_all)}"))
        if st.button("Use the selected plan(s)", type="primary", key=K("btn_select"), disabled=not picked):
            try:
                with st.spinner("Preparing the data…"):
                    s.select_plans(picked)
                    s.prepare_all()
                flash("2", "success", f"{len(picked)} plan(s) selected and prepared. Review each plan below, answer its questions, "
                                      "then approve the plans at the bottom.")
                flash("1", "success", "Plans selected. Continue in **2 · Prepare data**.")
            except ValueError as exc:
                flash("1", "error", str(exc))
            st.rerun()
        with st.expander("How MetaPy read your columns"):
            table(s.ds.column_map)
        with st.expander("Text cells interpreted (e.g. \"mean (SD)\", \"estimate (95% CI)\")"):
            table(s.ds.parse_notes)
        ev = (s.scan_report.g_inference or {}).get("evidence")
        if isinstance(ev, pd.DataFrame) and not ev.empty:
            with st.expander("Reported Hedges' g: what design is it? (checked against your raw data)"):
                table(ev.drop(columns=["record_id"], errors="ignore"))
        if not s.ds.conflicts.empty:
            with st.expander("Conflicting values between sheets"):
                table(s.ds.conflicts)

# ================================================================== 2 prepare
with tabs[1]:
    show_flash("2")
    if s is None or not s.plans:
        st.info("Read a file and select plan(s) in step 1 first.")
    else:
        st.markdown("**Status of your plans**")
        table(plan_status(s))
        pid = st.selectbox("Plan to review", list(s.plans), format_func=lambda p: plan_label(s, p), key=K("plan_dd"))
        p = s.preps.get(pid) or s.prepare(pid)
        st.markdown(prep_summary_md(s, pid))

        studies = p.data[["record_id", "label", "derivation_basis", "approximations_or_assumptions", "yi", "se", "included",
                          "exclusion_reason", "notes"]].rename(columns={"label": "study", "derivation_basis": "how obtained",
                                                                         "yi": f"{p.measure.short} (analysis scale)"})
        table(studies)

        if p.questions:
            st.markdown("#### Your decisions")
            answers = {}
            for q in p.questions:
                opts = q["options"]
                key = K(f"q::{pid}::{q['id']}::{len(opts)}")          # new key if the choices change
                if q.get("multi"):
                    dflt = [j for j, (_, v) in enumerate(opts) if v in (q.get("default") or [])]
                    sel = st.multiselect(q["question"], list(range(len(opts))), default=dflt, format_func=lambda j, o=opts: o[j][0], key=key)
                    answers[q["id"]] = [opts[j][1] for j in sel]
                else:
                    dflt = next((j for j, (_, v) in enumerate(opts) if v == q.get("default")), 0)
                    j = st.radio(q["question"], list(range(len(opts))), index=dflt, format_func=lambda j, o=opts: o[j][0], key=key)
                    answers[q["id"]] = opts[j][1]
            if st.button("Apply my answers", type="primary", key=K(f"btn_ans::{pid}")):
                s.answer(pid, answers)
                flash("2", "success", "Your answers were applied and everything was recalculated.")
                st.rerun()

        g_df = _g_rows(s, pid)
        if not g_df.empty:
            with st.expander("Reported Hedges' g in this plan: how each one is used (ppc2 / endpoint / change / none)"):
                edited = st.data_editor(g_df, hide_index=True, width="stretch", key=K(f"g::{pid}"),
                                        disabled=[c for c in g_df.columns if c != "use reported g as"],
                                        column_config={"use reported g as": st.column_config.SelectboxColumn(
                                            "use reported g as", options=["ppc2", "endpoint", "change", "none"])})
                if st.button("Apply these design choices", key=K(f"btn_g::{pid}")):
                    vals = {}
                    for _, r in edited.iterrows():
                        code = str(r.get("use reported g as", "") or "").strip().lower()
                        if code in G_FROM:
                            vals[str(r["record_id"])] = G_FROM[code]
                    s.answer(pid, {"g_type_rows": vals})
                    flash("2", "success", "Design choices for reported g applied.")
                    st.rerun()

        req = p.requests
        if not req.empty:
            st.markdown("#### Missing data")
            st.markdown(f"**{len(req)} comparison(s) cannot be derived from the data.** Each row lists the smallest sets of statistics that "
                        "would work (any one option is enough). Enter values from the paper below, or leave them excluded.")
            table(req.drop(columns=["fields_option_1"], errors="ignore"))
            rows = [{"record_id": r["record_id"], "study": r["study"], "statistic": f, "description": field_label(f), "value": None}
                    for _, r in req.iterrows() for f in [x for x in str(r["fields_option_1"]).split(",") if x]]
            entry = pd.DataFrame(rows, columns=["record_id", "study", "statistic", "description", "value"]).astype({"value": float})
            filled = st.data_editor(entry, hide_index=True, width="stretch", key=K(f"entry::{pid}"),
                                    disabled=["record_id", "study", "statistic", "description"])
            if st.button("Add these values", key=K(f"btn_add::{pid}")):
                vals: dict[str, dict[str, float]] = {}
                for _, r in filled.iterrows():
                    try:
                        v = float(r["value"])
                    except (TypeError, ValueError):
                        continue
                    if pd.notna(v):
                        vals.setdefault(str(r["record_id"]), {})[str(r["statistic"])] = v
                if vals:
                    s.add_values(pid, vals)
                    flash("2", "success", f"Added your values for {len(vals)} comparison(s); they are recorded as researcher-supplied in the log.")
                else:
                    flash("2", "warning", "No values were entered.")
                st.rerun()

        with st.expander("Include or exclude comparisons"):
            ids = list(p.data["record_id"])
            names = {r["record_id"]: f"{r['label']} ({r['record_id']})" for _, r in p.data.iterrows()}
            excluded_now = set(s.settings[pid].get("excluded_records", []))
            keep = st.multiselect("Comparisons to include (untick to exclude)", ids, default=[i for i in ids if i not in excluded_now],
                                  format_func=lambda i: names[i], key=K(f"incl::{pid}"))
            if st.button("Apply inclusion", key=K(f"btn_incl::{pid}")):
                excl = [i for i in ids if i not in set(keep)]
                if len(ids) - len(excl) < 2:
                    flash("2", "error", "Not applied: at least 2 comparisons must stay included for a meta-analysis.")
                else:
                    s.exclude(pid, excl)
                    flash("2", "success", f"{len(excl)} comparison(s) excluded by you." if excl else "All comparisons included.")
                st.rerun()

        with st.expander("Derivation log: every value MetaPy calculated, with formula, inputs and citation"):
            table(p.log.drop(columns=["record_id"], errors="ignore") if not p.log.empty else p.log)

        st.markdown("#### Approve")
        to_approve = st.multiselect("Plans to approve", list(s.plans), default=approved_ids(s), format_func=lambda x: plan_label(s, x),
                                    key=K("approve_group"))
        confirm = st.checkbox("I have reviewed the preparation of the ticked plans (calculations, approximations and exclusions)",
                              key=K("approve_confirm"))
        if st.button("Approve the ticked plans", type="primary", key=K("btn_approve")):
            if not confirm:
                flash("2", "warning", "Tick the confirmation box first.")
            else:
                ok, refused = s.approve_many(list(to_approve))
                msg = []
                if ok:
                    msg.append("Approved: " + ", ".join(f"**{s.plans[x]['outcome']}**" for x in ok) + ".")
                if refused:
                    msg.append("Not approved (fewer than 2 comparisons included): " + ", ".join(s.plans[x]["outcome"] for x in refused) + ".")
                waiting = [s.plans[x]["outcome"] for x in s.plans if not s.approved.get(x)]
                msg.append(f"Not approved yet: {', '.join(waiting)}." if waiting else "All plans approved. Continue in **3 · Analyse & review**.")
                flash("2", "success" if ok else "warning", " ".join(msg))
            st.rerun()

# ================================================================== 3 analyse
with tabs[2]:
    show_flash("3")
    if s is None or not s.plans:
        st.info("Select and prepare plans first (steps 1 and 2).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            model = st.radio("Model", ["Random effects", "Fixed effect"], key=K("model"))
        with c2:
            tau = st.radio("τ² estimator", ["REML (recommended)", "DL (DerSimonian-Laird)", "PM (Paule-Mandel)"], key=K("tau"))
        with c3:
            ci = st.radio("Confidence interval", ["Hartung-Knapp (recommended)", "Wald (normal)"], key=K("ci"))
        opts = {"model": "random" if model.startswith("Random") else "fixed", "tau_method": tau.split()[0],
                "ci_method": "HK" if ci.startswith("Hartung") else "Wald"}
        appr_sig = zlib.crc32("|".join(sorted(approved_ids(s))).encode())     # refresh the ticks when approvals change
        ticked = st.multiselect("Plans to analyse", list(s.plans), default=approved_ids(s), format_func=lambda x: plan_label(s, x),
                                key=K(f"analyse_group_{appr_sig}"))
        unapproved = [x for x in ticked if not s.approved.get(x)]
        confirm3 = st.checkbox("I approve the preparation of the ticked plans that are not yet approved", key=K("confirm3")) if unapproved else False
        extra = {a["key"]: f"{a['label']}  (needs {a['needs']})" for a in s.extra_choices()}
        extras = st.multiselect("Additional analyses (optional; you can add more after seeing the results)", list(extra),
                                format_func=lambda k: extra[k], key=K(f"extras_{len(extra)}"))
        if st.button("Run analysis", type="primary", key=K("btn_run"), disabled=not ticked):
            if unapproved and not confirm3:
                flash("3", "warning", "These ticked plans are not approved yet: " + ", ".join(s.plans[x]["outcome"] for x in unapproved) +
                      ". Review them in step 2, or tick the approval box here.")
            else:
                if unapproved:
                    _, refused = s.approve_many(sorted(set(approved_ids(s)) | set(unapproved)))
                    ticked = [x for x in ticked if x not in refused]
                with st.spinner("Analysing…"):
                    s.analyse({**opts, "extras": list(extras)}, ticked)
                left = [s.plans[x]["outcome"] for x in s.plans if x not in s.results]
                flash("3", "success", f"Analysed {len(s.results)} plan(s)." + (f" Not included in this run: {', '.join(left)}." if left else ""))
            st.rerun()

        if s.results:
            st.divider()
            pick = st.selectbox("Show details and plots for", list(s.results), format_func=lambda x: plan_label(s, x), key=K("res_pick"))
            st.markdown(results_md(s, pick))
            table(pd.concat([summary_table(r) for r in s.results.values()], ignore_index=True))
            r = s.results[pick]
            if r.main.get("status") == "Pooled":
                show_fig(forest_plot(r))
                if len(r.studies) >= 10:
                    tf = r.extras.get("trimfill")
                    show_fig(funnel_plot(r, trimfill=tf if tf and "filled_y" in tf else None))
                else:
                    st.caption(f"Funnel plot not drawn: {len(r.studies)} studies (at least 10 are needed).")

            st.markdown("#### Review")
            decision = st.radio("Are these results OK?", ["Yes, the results are OK; accept them", "Add more analyses (tick them below)",
                                                          "Change model settings (set them above)", "Go back and change data preparation"],
                                key=K("decision"))
            more = st.multiselect("More analyses", list(extra), format_func=lambda k: extra[k], key=K(f"more_{len(extra)}"))
            if st.button("Continue", type="primary", key=K("btn_review")):
                if decision.startswith("Yes"):
                    s.accept()
                    flash("3", "success", "Results accepted. Continue in **4 · Tables & figures**.")
                elif decision.startswith("Add"):
                    if more:
                        with st.spinner("Running the additional analyses…"):
                            s.extend(list(more))
                        flash("3", "success", f"Added {len(more)} analysis/analyses.")
                    else:
                        flash("3", "warning", "Tick the additional analyses you want, then click again.")
                elif decision.startswith("Change"):
                    with st.spinner("Re-analysing…"):
                        s.analyse({**opts, "extras": s.options.get("extras", [])}, list(s.results))
                    flash("3", "success", "Re-analysed with the new model settings.")
                else:
                    flash("3", "info", "Go back to **2 · Prepare data**, change decisions, re-approve, then run the analysis again.")
                st.rerun()

        with st.expander("Network meta-analysis (optional: three or more treatments compared across studies)"):
            st.markdown("Use a sheet of your file in **arm format** (study, treatment, mean, sd, n, or events, n) or **contrast format** "
                        "(study, treat1, treat2, TE, seTE), or a prepared outcome whose studies compare several named intervention arms "
                        "with a control. The network result is added to the report.")
            if st.button("Find network data", key=K("btn_nma_src")):
                ss.nma_src = s.nma_sources()
                if not ss.nma_src:
                    flash("3", "info", "No network data found. Add a sheet in arm or contrast format, or name the arms in an "
                                       "'intervention_arm' column.")
                st.rerun()
            if ss.nma_src:
                src_labels = dict((v, lab) for lab, v in ss.nma_src)
                src = st.selectbox("Data for the network", list(src_labels), format_func=lambda v: src_labels[v], key=K("nma_src"))
                n1, n2, n3, n4 = st.columns(4)
                with n1:
                    measure = st.radio("Effect measure (arm data)", ["SMD", "MD", "OR", "RR", "generic"], key=K("nma_measure"))
                with n2:
                    nmodel = st.radio("Model", ["Random effects", "Common effect"], key=K("nma_model"))
                with n3:
                    small = st.radio("Ranking direction", ["Smaller values are better", "Larger values are better"], key=K("nma_small"))
                with n4:
                    ref = st.text_input("Reference treatment (blank = most connected)", key=K("nma_ref"))
                if st.button("Run network meta-analysis", type="primary", key=K("btn_nma")):
                    try:
                        ss.nma_last = s.nma(src, measure, (ref or "").strip() or None, "random" if nmodel.startswith("Random") else "fixed",
                                            "good" if small.startswith("Smaller") else "bad")
                    except Exception as exc:
                        ss.nma_last = None
                        flash("3", "error", f"Network meta-analysis could not be run: {exc}")
                    st.rerun()
            if ss.nma_last:
                from metapy import nma as N
                r_ = ss.nma_last["result"]
                Q = r_.Q
                st.markdown(f"**{len(r_.treatments)} treatments, {r_.extra['n_studies']} studies ({r_.extra['multi_arm_studies']} multi-arm)**, "
                            f"reference **{r_.reference}**, τ² = {r_.tau2:.3f}. Inconsistency between designs: Q = {Q['Q_inconsistency']:.2f}, "
                            f"df = {Q['df_inconsistency']}" + (f", p = {Q['p_inconsistency']:.3f}." if Q["df_inconsistency"] > 0
                                                               else " (no closed loops: consistency cannot be checked)."))
                for note in ss.nma_last.get("notes") or []:
                    st.caption("• " + note)
                st.markdown("**Ranking (P-scores)**")
                table(ss.nma_last["pscores"])
                st.markdown("**League table (row treatment vs column treatment)**")
                table(ss.nma_last["league"].reset_index(drop=True))
                if not ss.nma_last["split"].empty:
                    st.markdown("**Direct vs indirect evidence**")
                    table(ss.nma_last["split"])
                g1, g2 = st.columns(2)
                with g1:
                    show_fig(N.network_plot(r_))
                with g2:
                    show_fig(N.forest_vs_reference(r_, ss.nma_last.get("transform"), N.MEASURES_NMA.get(r_.contrasts.measure, "")))

# ================================================================== 4 tables & figures
with tabs[3]:
    show_flash("4")
    if s is None or not s.results:
        st.info("Run the analysis first (step 3).")
    tbls = st.multiselect("Tables", list(TABLE_CHOICES), default=DEFAULT_TABLES, format_func=lambda k: TABLE_CHOICES[k], key=K("tbls"))
    fig_keys = [k for k in FIGURE_CHOICES if k != "prisma"]
    figs = st.multiselect("Figures", fig_keys, default=[k for k in DEFAULT_FIGURES if k in fig_keys],
                          format_func=lambda k: FIGURE_CHOICES[k], key=K("figs"))
    txts = st.multiselect("Text", list(TEXT_CHOICES), default=list(TEXT_CHOICES), format_func=lambda k: TEXT_CHOICES[k], key=K("txts"))
    a1, a2, a3 = st.columns(3)
    with a1:
        theme = st.selectbox("Figure colour theme", ["navy", "teal", "purple", "monochrome", "orange"], key=K("theme"))
    with a2:
        sort = st.selectbox("Forest plot order", ["As in data", "Sort by effect", "Sort by year"], key=K("sort"))
    with a3:
        title = st.text_input("Report title", value="Meta-analysis report", key=K("title"))
    forest_sort = None if sort == "As in data" else sort.lower().split()[-1]

    st.markdown("#### PRISMA 2020 flow diagram (optional)")
    st.caption("Each box below is one box of the diagram: type or paste anything, on as many lines as you like. The heading is printed "
               "for you and your text is printed exactly as written (start a line with two spaces to indent it).")
    for k in PRISMA_KEYS[1:]:                     # initialise once, so the 'fill' button can write into a box
        if K("p_" + k) not in ss:
            ss[K("p_" + k)] = ""

    def _fill_included():
        if s is not None and s.results:
            a = s.prisma_autofill()
            ss[K("p_box_included")] = (f"(n = {a['studies_included']} studies)\nComparisons included in meta-analysis "
                                       f"(n = {a['comparisons_included']})\nTotal sample size (N = {a['participants']} participants)")

    with st.expander("PRISMA 2020 flow diagram"):
        prisma_on = st.checkbox("Include a PRISMA 2020 flow diagram in the report", key=K("p_on"))
        heads = {k: h for k, h, *_ in PRISMA_BOXES}
        st.markdown("**Identification**")
        b1, b2 = st.columns(2)
        with b1:
            st.text_area(heads["box_identified"], height=150, placeholder=PRISMA_HINTS["box_identified"], key=K("p_box_identified"))
        with b2:
            st.text_area(heads["box_removed"], height=150, placeholder=PRISMA_HINTS["box_removed"], key=K("p_box_removed"))
        st.markdown("**Screening**")
        for left, right in (("box_screened", "box_excluded"), ("box_sought", "box_not_retrieved"), ("box_assessed", "box_reports_excluded")):
            c_l, c_r = st.columns(2)
            with c_l:
                st.text_area(heads[left], height=80, placeholder=PRISMA_HINTS[left], key=K("p_" + left))
            with c_r:
                st.text_area(heads[right], height=130 if right == "box_reports_excluded" else 80, placeholder=PRISMA_HINTS[right],
                             key=K("p_" + right))
        st.markdown("**Included**")
        st.text_area(heads["box_included"], height=100, placeholder=PRISMA_HINTS["box_included"], key=K("p_box_included"))
        st.button("Fill 'Studies included' from my analysis", key=K("btn_fill"), on_click=_fill_included, disabled=s is None or not s.results)
        st.markdown("**Identification via other methods** (optional third column: websites, citation searching, …)")
        for k, h in PRISMA_OTHER_BOXES:
            st.text_area(f"Other methods: {h}", height=70, placeholder="Type or paste anything", key=K("p_" + k))
        st.text_area("Note:", height=90, placeholder="Optional. Printed under the diagram.", key=K("p_note"))
    prisma = _prisma_dict([prisma_on] + [ss.get(K("p_" + k), "") for k in PRISMA_KEYS[1:]])

    if st.button("Preview report", type="primary", key=K("btn_preview"), disabled=s is None or not s.results):
        with st.spinner("Building the report…"):
            rep = s.report(tbls, figs, txts, prisma, theme, title or "Meta-analysis report", forest_sort)
        n_fig = sum(len(sec.figures) for sec in rep.sections)
        st.markdown(preview_md(s, rep, prisma, n_fig))
        for sec in rep.sections:
            for name, fig in sec.figures:
                st.markdown(f"**{sec.heading}: {name}**")
                show_fig(fig)
        plt.close("all")

# ================================================================== 5 export
with tabs[4]:
    show_flash("5")
    if s is None or not s.results:
        st.info("Run the analysis first (step 3).")
    else:
        st.markdown(f"The report uses the tables, figures, text and PRISMA settings chosen in step 4, and is built from data file "
                    f"**`{s.ds.source_name}`**. Your original file is never modified.")
        fmts = st.multiselect("File formats", list(FORMATS), default=["docx", "xlsx"], format_func=lambda k: FORMATS[k], key=K("fmts"))
        if st.button("Export", type="primary", key=K("btn_export"), disabled=not fmts):
            out = tempfile.mkdtemp(prefix="metapy_export_")
            try:
                with st.spinner("Writing the files…"):
                    paths = s.export(list(fmts), out, tbls, figs, txts, prisma, theme, title or "Meta-analysis report", forest_sort=forest_sort)
                ss.exports = [(Path(p).name, Path(p).read_bytes()) for p in paths]
                note = "" if s.accepted else " Note: you have not accepted the results in step 3 yet."
                flash("5", "success", f"Exported {len(paths)} file(s). Download them below.{note}")
            except Exception as exc:
                ss.exports = []
                flash("5", "error", f"Export failed: {exc}")
            finally:
                shutil.rmtree(out, ignore_errors=True)
            st.rerun()
        for i, (name, data) in enumerate(ss.exports):
            st.download_button(f"⬇ Download {name}", data=data, file_name=name,
                               mime=mimetypes.guess_type(name)[0] or "application/octet-stream", key=K(f"dl_{i}"))
