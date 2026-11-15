from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils import get_column_letter


def _safe_sheet_name(name: str) -> str:
    return name[:31]


PLOT_THEMES = {
    "navy": {"point": "#1F4E79", "ci": "#6FA8DC", "reference": "#687386"},
    "teal": {"point": "#0F766E", "ci": "#5EB8AA", "reference": "#687386"},
    "purple": {"point": "#6B4FA1", "ci": "#A78BFA", "reference": "#687386"},
    "monochrome": {"point": "#303030", "ci": "#777777", "reference": "#A0A0A0"},
    "orange": {"point": "#B45309", "ci": "#F3B66D", "reference": "#687386"},
}


def normalize_plot_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = settings or {}
    theme = str(settings.get("theme", "navy")).strip().lower()
    if theme not in PLOT_THEMES:
        theme = "navy"
    try:
        width = float(settings.get("forest_width_inches", 10.5))
    except (TypeError, ValueError):
        width = 10.5
    try:
        row_height = float(settings.get("forest_row_height_inches", 0.42))
    except (TypeError, ValueError):
        row_height = 0.42
    return {
        "theme": theme,
        "forest_width_inches": min(16.0, max(8.0, width)),
        "forest_row_height_inches": min(0.8, max(0.3, row_height)),
    }


def make_forest_plot(
    group: pd.DataFrame, output: str | Path, title: str,
    plot_settings: dict[str, Any] | None = None,
) -> str | None:
    valid = group.copy()
    valid["effect_estimate"] = pd.to_numeric(valid["effect_estimate"], errors="coerce")
    valid["standard_error"] = pd.to_numeric(valid["standard_error"], errors="coerce")
    valid = valid[np.isfinite(valid["effect_estimate"]) & np.isfinite(valid["standard_error"]) & (valid["standard_error"] > 0)]
    if valid.empty:
        return None
    valid = valid.reset_index(drop=True)
    y = np.arange(len(valid))[::-1]
    est = valid["effect_estimate"].to_numpy(float)
    se = valid["standard_error"].to_numpy(float)
    lo, hi = est - 1.96*se, est + 1.96*se
    settings = normalize_plot_settings(plot_settings)
    palette = PLOT_THEMES[settings["theme"]]
    fig_height = max(4.0, settings["forest_row_height_inches"]*len(valid)+1.8)
    fig, ax = plt.subplots(figsize=(settings["forest_width_inches"], fig_height))
    ax.errorbar(
        est, y, xerr=[est-lo, hi-est], fmt="o", capsize=3,
        color=palette["point"], ecolor=palette["ci"],
        markerfacecolor=palette["point"], markeredgecolor=palette["point"],
    )
    labels = [f"{r.study_id} — {r.intervention_arm}" for r in valid.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, linestyle="--", linewidth=1, color=palette["reference"])
    ax.set_xlabel("Effect estimate (95% normal-approximation CI)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    output = str(output)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def make_funnel_plot(
    group: pd.DataFrame, output: str | Path, title: str,
    plot_settings: dict[str, Any] | None = None,
) -> str | None:
    valid = group.copy()
    valid["effect_estimate"] = pd.to_numeric(valid["effect_estimate"], errors="coerce")
    valid["standard_error"] = pd.to_numeric(valid["standard_error"], errors="coerce")
    valid = valid[np.isfinite(valid["effect_estimate"]) & np.isfinite(valid["standard_error"]) & (valid["standard_error"] > 0)]
    if valid["study_id"].nunique() < 10:
        return None
    settings = normalize_plot_settings(plot_settings)
    palette = PLOT_THEMES[settings["theme"]]
    funnel_width = max(6.0, settings["forest_width_inches"] * 0.66)
    fig, ax = plt.subplots(figsize=(funnel_width, funnel_width * 0.86))
    ax.scatter(valid["effect_estimate"], valid["standard_error"], color=palette["point"], alpha=0.85)
    # Centre the visual guide on the inverse-variance weighted fixed-effect
    # estimate, rather than the unweighted arithmetic mean.
    weights = 1.0 / np.square(valid["standard_error"].to_numpy(float))
    pooled = float(np.sum(weights * valid["effect_estimate"].to_numpy(float)) / np.sum(weights))
    ymax = float(valid["standard_error"].max())
    ax.axvline(pooled, linestyle="--", linewidth=1, color=palette["reference"])
    ax.plot([pooled-1.96*ymax, pooled, pooled+1.96*ymax], [ymax, 0, ymax], linestyle=":", color=palette["ci"])
    ax.set_ylim(ymax*1.05, 0)
    ax.set_xlabel("Effect estimate")
    ax.set_ylabel("Standard error (reversed)")
    ax.set_title(title + " — visual assessment only")
    fig.tight_layout()
    output = str(output)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def export_report(
    result: dict[str, Any], output_path: str | Path,
    plot_settings: dict[str, Any] | None = None,
) -> str:
    """Export an auditable multi-sheet workbook and embedded forest plots."""
    plot_settings = normalize_plot_settings(plot_settings)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path.parent / (output_path.stem + "_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    prepared = result.get("prepared", pd.DataFrame())
    conflicts = result.get("conflicts", pd.DataFrame())
    findings = result.get("quality_findings", result.get("findings", pd.DataFrame()))
    if findings.empty:
        findings = pd.DataFrame([{
            "finding_type": "Information",
            "finding_code": "NO_FINDINGS_DETECTED",
            "description": "No automated data-quality findings were detected in this run. This is not source authenticity verification.",
            "status": "No findings detected",
        }])
    review_items = result.get("researcher_review_items", pd.DataFrame())
    if review_items.empty:
        review_items = pd.DataFrame([{
            "finding_type": "Information",
            "finding_code": "NO_PENDING_RESEARCHER_REVIEW_ITEMS",
            "description": "No outstanding row-level researcher review items were generated.",
            "status": "No pending items",
        }])
    if conflicts.empty:
        conflicts = pd.DataFrame(columns=["record_id", "study_id", "outcome", "duration", "intervention_arm", "field", "conflicting_values", "source_locations", "status"])

    effect_results = result.get("effects", pd.DataFrame())
    pooled = result.get("pooled", pd.DataFrame())
    sensitivity_records = result.get("sensitivity_records", pd.DataFrame())
    sensitivity_pooled = result.get("sensitivity_pooled", pd.DataFrame())
    log = result.get("transformation_log", pd.DataFrame())
    summary = result.get("summary", {})
    decisions = result.get("decision_audit", pd.DataFrame())
    unmapped = result.get("unmapped_columns", pd.DataFrame())

    sheets: dict[str, pd.DataFrame] = {
        "Run Summary": pd.DataFrame([summary]),
        "Prepared Dataset": prepared,
        "Effect Calculations": effect_results,
        "Primary Pooled Results": pooled,
        "Data Quality Findings": findings,
        "Researcher Review Items": review_items,
        "Analysis Blockers": effect_results[~effect_results.get("include_in_primary_pool", pd.Series(dtype=bool)).fillna(False).astype(bool)].copy() if not effect_results.empty else pd.DataFrame(),
        "Source Conflicts": conflicts,
        "Unmapped Columns": unmapped,
        "Correlation Sensitivity": sensitivity_records,
        "Sensitivity Pooled": sensitivity_pooled,
        "Leave-One-Out": result.get("leave_one_out", pd.DataFrame()),
        "Transformation Log": log,
        "Decision Audit": decisions,
    }
    canonical = result.get("canonical", pd.DataFrame())
    sheets["Canonical Source Records"] = canonical
    dictionary = pd.DataFrame([
        ["record_id", "MetaPy-generated stable comparison identifier; not a source study identifier"],
        ["reported_effect", "Effect estimate reported in the workbook; not used unless method is confirmed to match the selected estimand"],
        ["effect_estimate", "Calculated or compatible researcher-reported effect estimate, oriented using the confirmed outcome direction"],
        ["sampling_variance", "Variance derived using a method compatible with the selected estimand; blank means unresolved"],
        ["record_review_approved", "Explicit researcher approval of an individual canonical comparison"],
        ["include_in_primary_pool", "True only when effect, variance, design, direction, record approval and dependence gates pass"],
        ["r_prepost_i / r_prepost_c", "Actual intervention/control pre-post correlations mapped from the supplied workbook; never externally verified"],
        ["assumption_flag", "Marks an assumed correlation used only in a separate sensitivity scenario"],
    ], columns=["field", "definition"])
    sheets["Data Dictionary"] = dictionary

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=_safe_sheet_name(name), index=False)

    # Style each output sheet consistently and create a readable workbook.
    wb = load_workbook(output_path)
    dark = "1F3B5B"
    light = "EAF1F7"
    line = Side(style="thin", color="D9E2F0")
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions if ws.max_row >= 2 else None
        ws.row_dimensions[1].height = 30
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor=dark)
            cell.font = Font(color="FFFFFF", bold=True, size=10)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            cell.border = Border(bottom=line)
        for col_idx in range(1, ws.max_column + 1):
            letter = get_column_letter(col_idx)
            max_len = 0
            for cell in ws[letter][: min(ws.max_row, 150)]:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, min(len(value), 60))
                if cell.row > 1:
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                    if cell.row % 2 == 0:
                        cell.fill = PatternFill("solid", fgColor="F7FAFC")
                    cell.border = Border(bottom=Side(style="hair", color="E5EAF0"))
            ws.column_dimensions[letter].width = max(12, min(36, max_len + 2))
        if ws.max_column > 0 and ws.max_row > 1:
            ws.sheet_view.showGridLines = False

    # Highlight rows with calculation or review blockers.
    for name in ["Prepared Dataset", "Effect Calculations", "Researcher Review Items"]:
        if name[:31] in wb.sheetnames:
            ws = wb[name[:31]]
            headers = {c.value: c.column for c in ws[1]}
            for key in ["include_in_primary_pool", "record_review_approved"]:
                if key in headers and ws.max_row >= 2:
                    col = get_column_letter(headers[key])
                    ws.conditional_formatting.add(f"{col}2:{col}{ws.max_row}", CellIsRule(operator="equal", formula=['FALSE'], fill=PatternFill("solid", fgColor="FCE8E6")))

    # Add a dedicated visualisation sheet and embed forest plots for eligible groups.
    vis = wb.create_sheet("Visualisations")
    vis["A1"] = "MetaPy — Forest plots"
    vis["A1"].font = Font(bold=True, size=15, color="FFFFFF")
    vis["A1"].fill = PatternFill("solid", fgColor=dark)
    vis.merge_cells("A1:J1")
    vis["A2"] = "Plots use comparisons that passed the selected primary pooling gates. Error bars are normal-approximation 95% intervals."
    vis.merge_cells("A2:J2")
    vis["A2"].alignment = Alignment(wrap_text=True)
    pos_row = 4
    if not effect_results.empty and "include_in_primary_pool" in effect_results:
        eligible = effect_results[effect_results["include_in_primary_pool"].fillna(False).astype(bool)]
        group_cols = ["outcome"] + (["duration"] if "duration" in eligible and eligible["duration"].notna().any() else [])
        if not eligible.empty:
            for key, grp in eligible.groupby(group_cols, dropna=False, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                title = " — ".join(str(x) for x in key)
                plot_path = make_forest_plot(
                    grp, plots_dir / f"forest_{pos_row}.png", f"Forest plot: {title}", plot_settings
                )
                if plot_path:
                    vis.cell(pos_row, 1, f"{title}")
                    vis.cell(pos_row, 1).font = Font(bold=True, size=12, color=dark)
                    img = XLImage(plot_path)
                    img.width = int(plot_settings["forest_width_inches"] * 84)
                    fig_height = max(4.0, plot_settings["forest_row_height_inches"] * len(grp) + 1.8)
                    img.height = int(fig_height * 84)
                    vis.add_image(img, f"A{pos_row+1}")
                    pos_row += max(22, len(grp) * 2 + 8)
                    # A simple funnel display is produced only when at least 10
                    # independent studies contribute and there are no duplicate
                    # study-level effects in this outcome group.
                    if grp["study_id"].nunique() >= 10 and not grp["study_id"].astype(str).duplicated().any():
                        funnel_path = make_funnel_plot(
                            grp, plots_dir / f"funnel_{pos_row}.png", f"Funnel plot: {title}", plot_settings
                        )
                        if funnel_path:
                            vis.cell(pos_row, 1, f"Funnel plot: {title}")
                            vis.cell(pos_row, 1).font = Font(bold=True, size=12, color=dark)
                            funnel_img = XLImage(funnel_path)
                            funnel_img.width = 540
                            funnel_img.height = 450
                            vis.add_image(funnel_img, f"A{pos_row+1}")
                            pos_row += 30
        else:
            vis["A4"] = "No primary forest plots generated: no comparison passed every pooling gate."
    else:
        vis["A4"] = "No primary forest plots generated: no comparison passed every pooling gate."
    vis.column_dimensions["A"].width = 32
    vis.sheet_view.showGridLines = False
    wb.save(output_path)
    return str(output_path)
