from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr
import pandas as pd

from .effect_sizes import ESTIMANDS
from .preparation import prepare_records
from .workbook import canonicalize_workbook, load_excel_workbook
from .workflow import build_researcher_review_table, run_analysis
from .templates import create_default_templates


def _inspect(path):
    if path is None:
        return "Upload a workbook to begin.", pd.DataFrame(), None
    try:
        sheets = load_excel_workbook(path)
        canonical, conflicts, unmapped = canonicalize_workbook(sheets)
        prepared, findings, log, summary = prepare_records(canonical)
        review = build_researcher_review_table(canonical, prepared)
        if canonical.empty:
            return (
                "**No comparison rows found.** This may be a blank MetaPy template. Fill the required fields (especially `StudyID` and `Outcome`) in the appropriate input sheet, save the workbook, and inspect it again.",
                review,
                path,
            )
        text = (
            f"**Workbook:** {Path(path).name}\n\n"
            f"**Worksheets:** {len(sheets)}  |  **Canonical comparisons:** {len(canonical)}  |  "
            f"**Field conflicts:** {len(conflicts)}  |  **Unmapped non-empty columns:** {len(unmapped)}\n\n"
            "Review each row below. Set the direction multiplier to **+1 when higher values are worse** "
            "and **−1 when higher values are better**. Confirm the design for each comparison and tick "
            "`approve_record` only after checking that row. Global settings do not replace row-level approval."
        )
        if len(conflicts):
            text += f"\n\n**Important:** {len(conflicts)} conflicting field(s) were found. Conflicting values are excluded from calculations."
        if len(unmapped):
            text += f"\n\n**Note:** {len(unmapped)} source row(s) contain non-empty columns that were not mapped. Review the `Unmapped Columns` output sheet."
        return text, review, path
    except Exception as exc:
        return f"**Import error:** {type(exc).__name__}: {exc}", pd.DataFrame(), None


def _parse_sensitivity(enabled, raw):
    if not enabled:
        return None
    values = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        number = float(part)
        if not -1 <= number <= 1:
            raise ValueError("Correlation sensitivity values must be between -1 and +1.")
        values.append(number)
    if not values:
        raise ValueError("Enter at least one correlation value for the sensitivity grid.")
    return sorted(set(values))


def _run(path, estimand, review_table, confirm_all_parallel, shared_policy, model, tau_method, ci_method,
         run_sensitivity, sensitivity_values, plot_theme, forest_width, forest_row_height, output_name):
    if path is None:
        return "Upload a workbook first.", pd.DataFrame(), pd.DataFrame(), None
    try:
        grid = _parse_sensitivity(run_sensitivity, sensitivity_values)
        out = Path("/content") / (output_name.strip() or "MetaPy_Analysis_Report.xlsx")
        if not out.name.lower().endswith(".xlsx"):
            out = out.with_suffix(".xlsx")
        result = run_analysis(
            path,
            estimand=estimand,
            review_table=review_table,
            global_decisions={"parallel_independent_groups_confirmed": bool(confirm_all_parallel)},
            correlation_sensitivity=grid,
            shared_control_policy=shared_policy,
            model=model,
            tau_method=tau_method,
            ci_method=ci_method,
            plot_settings={
                "theme": plot_theme,
                "forest_width_inches": forest_width,
                "forest_row_height_inches": forest_row_height,
            },
            output_path=out,
        )
        s = result["summary"]
        msg = (
            f"### Run complete — MetaPy {s['metapy_version']}\n\n"
            f"**Comparisons:** {s['canonical_comparisons']}  |  **Record-level approvals:** {s['record_level_approvals']}/{s['canonical_comparisons']}  |  "
            f"**Effect estimates:** {s['records_with_effect_estimate']}  |  **Variances:** {s['records_with_sampling_variance']}  |  "
            f"**Eligible for primary pooling:** {s['records_eligible_for_primary_pool']}\n\n"
            f"**Preparation:** {s['preparation_status']}\n\n"
            f"**Pooling:** {s['pooling_status']}\n\n"
            "MetaPy has not verified the authenticity of extracted study data. Assumed correlations, when requested, "
            "are reported only in a separate sensitivity analysis and are not treated as source values."
        )
        return msg, result["effects"], result["pooled"], str(result["report_path"])
    except Exception as exc:
        return f"**Analysis error:** {type(exc).__name__}: {exc}\n\n```\n{traceback.format_exc()}\n```", pd.DataFrame(), pd.DataFrame(), None


def build_app():
    estimand_choices = [(label, key) for key, label in ESTIMANDS.items()]
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    template_paths = create_default_templates(examples_dir)
    with gr.Blocks(title="MetaPy 3.3.0 — Complete Pairwise Analysis") as app:
        gr.Markdown("# MetaPy 3.3.0")
        gr.Markdown("### Workbook preparation, effect-size reconstruction and pairwise meta-analysis")
        gr.Markdown(
            "MetaPy works only from the workbook you upload. It checks internal structure and calculation compatibility; "
            "it does not verify study authenticity or retrieve missing values from external sources. This release supports "
            "pairwise continuous and binary outcomes in parallel independent groups."
        )
        with gr.Row():
            gr.DownloadButton("Download continuous pairwise input template", value=template_paths["continuous"])
            gr.DownloadButton("Download binary pairwise input template", value=template_paths["binary"])
        path_state = gr.State(None)
        with gr.Row():
            file_input = gr.File(label="Upload research workbook (.xlsx)", file_types=[".xlsx", ".xlsm"], type="filepath")
            inspect_button = gr.Button("Inspect workbook", variant="primary")
        inspect_summary = gr.Markdown("Upload a workbook, then click **Inspect workbook**.")
        review_table = gr.Dataframe(
            label="Record-level review and confirmation table",
            headers=["record_id", "study_id", "outcome", "duration", "intervention_arm", "study_design_source", "direction_multiplier", "parallel_independent_groups_confirmed", "reported_effect_method_source", "reported_effect_method_override", "approve_record"],
            interactive=True,
            wrap=True,
            row_count=(0, "dynamic"),
            column_count=(11, "fixed"),
        )
        inspect_button.click(_inspect, inputs=[file_input], outputs=[inspect_summary, review_table, path_state])
        with gr.Accordion("Analysis settings", open=True):
            estimand = gr.Dropdown(choices=estimand_choices, value="morris_ppc2", label="Primary effect measure / estimand")
            confirm_all_parallel = gr.Checkbox(value=False, label="Confirm all comparisons are parallel independent groups (only tick if this applies to every row)")
            shared_policy = gr.Dropdown(
                choices=[
                    ("Block possible shared controls until handled", "block"),
                    ("Split shared control sample size for variance weights only", "split_control_variance"),
                    ("Confirm these are separate independent controls", "reviewer_confirmed_independent"),
                ], value="block", label="Multi-arm / shared-control handling"
            )
            model = gr.Radio(choices=[("Random effects", "random"), ("Fixed effect", "fixed")], value="random", label="Pooling model")
            tau_method = gr.Dropdown(choices=["REML", "DL"], value="REML", label="Random-effects tau-squared estimator")
            ci_method = gr.Dropdown(choices=["Hartung-Knapp", "Normal"], value="Hartung-Knapp", label="Random-effects confidence interval method")
            run_sensitivity = gr.Checkbox(value=False, label="Run a separate assumed-correlation sensitivity analysis if source correlations are missing")
            sensitivity_values = gr.Textbox(value="0.1, 0.3, 0.5, 0.7, 0.9", label="Correlation sensitivity grid", interactive=True)
            with gr.Accordion("Visualisation settings", open=False):
                plot_theme = gr.Dropdown(
                    choices=[("Navy", "navy"), ("Teal", "teal"), ("Purple", "purple"), ("Monochrome", "monochrome"), ("Orange", "orange")],
                    value="navy", label="Forest/funnel plot colour theme",
                )
                forest_width = gr.Slider(minimum=8, maximum=16, value=10.5, step=0.5, label="Forest plot width (inches)")
                forest_row_height = gr.Slider(minimum=0.3, maximum=0.8, value=0.42, step=0.02, label="Forest plot row height (inches per comparison)")
            output_name = gr.Textbox(value="MetaPy_Analysis_Report.xlsx", label="Output report filename")
        run_button = gr.Button("Calculate effects and pool eligible comparisons", variant="primary")
        run_status = gr.Markdown()
        with gr.Tab("Effect calculations"):
            effect_table = gr.Dataframe(label="Row-level effects and variance status", interactive=False, wrap=True)
        with gr.Tab("Pooled results"):
            pooled_table = gr.Dataframe(label="Primary pooled results", interactive=False, wrap=True)
        report_file = gr.File(label="Download the MetaPy analysis report")
        run_button.click(
            _run,
            inputs=[path_state, estimand, review_table, confirm_all_parallel, shared_policy, model, tau_method,
                    ci_method, run_sensitivity, sensitivity_values, plot_theme, forest_width, forest_row_height, output_name],
            outputs=[run_status, effect_table, pooled_table, report_file],
        )
        gr.Markdown(
            "**Interpretation safeguard:** a calculated point estimate is not automatically poolable. A comparison "
            "must also have compatible uncertainty, confirmed design and outcome direction, no unresolved source conflict, "
            "individual researcher approval, and appropriate handling of possible shared controls."
        )
    return app
