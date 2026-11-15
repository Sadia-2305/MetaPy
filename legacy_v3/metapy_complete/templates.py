from __future__ import annotations

from pathlib import Path
from typing import Literal

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Side, Border
from openpyxl.worksheet.datavalidation import DataValidation


CONTINUOUS_COLUMNS = [
    "StudyID", "Outcome", "Duration", "Intervention_Arm", "Study_Design", "Unit", "Higher_is_worse",
    "Mean_Pre_I", "SD_Pre_I", "Mean_Post_I", "SD_Post_I", "N_I",
    "Mean_Pre_C", "SD_Pre_C", "Mean_Post_C", "SD_Post_C", "N_C",
    "r_prepost_I", "r_prepost_C", "Hedges_g", "Effect_Method", "Effect_SE", "Effect_Variance",
    "CI_Lower", "CI_Upper", "CI_Level", "MD (95% CI)", "Control_ID",
]

BINARY_COLUMNS = [
    "StudyID", "Outcome", "Duration", "Intervention_Arm", "Study_Design", "Higher_is_worse",
    "Events_I", "Total_I", "Events_C", "Total_C", "Hedges_g", "Effect_Method",
    "Effect_SE", "Effect_Variance", "CI_Lower", "CI_Upper", "CI_Level", "Control_ID",
]

INSTRUCTIONS = [
    ("Purpose", "Enter one row per intervention-versus-control comparison, outcome and timepoint."),
    ("StudyID", "Use the same study identifier for every comparison belonging to the same study."),
    ("Intervention_Arm", "Give each intervention arm an explicit label. Do not leave this blank in multi-arm studies."),
    ("Control_ID", "Use the same identifier for rows that reuse the same control participants. Leave blank only when not applicable or unknown."),
    ("Study_Design", "Primary pooling in v3.3.0 is limited to parallel independent groups. Crossover, cluster and other designs are flagged for review and are not automatically analyzed."),
    ("Higher_is_worse", "Enter 'higher is worse' when larger values mean worse symptoms/adverse outcome; enter 'higher is better' when larger values mean improvement."),
    ("Pre/post correlations", "Enter actual intervention and control correlations only when present in your extraction. Leave missing values blank; do not enter assumed sensitivity values in the source columns."),
    ("Effect_Method", "Complete this only when the reported estimate's method is known/confirmed. Use one code: morris_ppc2, posttest_smd, change_smd, raw_posttest_md, raw_change_md, log_rr, log_or, risk_difference."),
    ("Reported Hedges_g", "A reported Hedges' g is not automatically posttest SMD or Morris PPC2. The method must be explicitly confirmed before MetaPy uses it."),
    ("Raw MD confidence interval", "The 'MD (95% CI)' column may contain text like -4.2 (-6.1 to -2.3). MetaPy retains it for review and uses it only after method confirmation as raw_posttest_md."),
    ("Units", "A declared unit is needed for meaningful raw mean-difference pooling. Standardized mean differences do not require a unit label."),
    ("Missing values", "Leave unavailable values blank. Do not enter zero to represent missing data."),
    ("Original data", "MetaPy does not verify publication authenticity or retrieve missing study values. Keep the original workbook and review every row before approval."),
    ("Supported release scope", "This release supports pairwise continuous and pairwise binary outcomes in parallel independent groups. It is not a network, crossover, cluster, survival, diagnostic-accuracy or multilevel meta-analysis engine."),
]


def create_input_template(
    output_path: str | Path,
    template_type: Literal["continuous", "binary"] = "continuous",
) -> str:
    """Create a formatted, blank researcher input workbook for a supported design."""
    template_type = template_type.lower().strip()
    if template_type not in {"continuous", "binary"}:
        raise ValueError("template_type must be 'continuous' or 'binary'.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Continuous Pairwise" if template_type == "continuous" else "Binary Pairwise"
    headers = CONTINUOUS_COLUMNS if template_type == "continuous" else BINARY_COLUMNS
    ws.append(headers)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{ws.cell(row=1, column=len(headers)).coordinate}1"
    ws.sheet_view.showGridLines = False
    dark = "1F3B5B"
    border = Side(style="thin", color="D9E2F0")
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=dark)
        cell.font = Font(color="FFFFFF", bold=True, size=10)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = Border(bottom=border)
    ws.row_dimensions[1].height = 34
    for index, header in enumerate(headers, 1):
        width = 18 if len(header) < 15 else 22
        if header in {"StudyID", "Outcome", "Intervention_Arm", "Study_Design", "Effect_Method", "Control_ID"}:
            width = 26
        if header == "MD (95% CI)":
            width = 28
        ws.column_dimensions[ws.cell(row=1, column=index).column_letter].width = width

    # Add validation lists to common decision columns. These validations guide
    # entry but do not replace the researcher's review inside MetaPy.
    design_col = headers.index("Study_Design") + 1
    direction_col = headers.index("Higher_is_worse") + 1
    design_validation = DataValidation(
        type="list",
        formula1='"parallel randomized controlled trial,parallel non-randomized controlled trial,crossover,cluster randomized trial,other / unclear"',
        allow_blank=True,
    )
    direction_validation = DataValidation(
        type="list", formula1='"higher is worse,higher is better"', allow_blank=True
    )
    ws.add_data_validation(design_validation)
    ws.add_data_validation(direction_validation)
    design_letter = ws.cell(row=1, column=design_col).column_letter
    direction_letter = ws.cell(row=1, column=direction_col).column_letter
    design_validation.add(f"{design_letter}2:{design_letter}501")
    direction_validation.add(f"{direction_letter}2:{direction_letter}501")

    if "Effect_Method" in headers:
        method_col = headers.index("Effect_Method") + 1
        method_validation = DataValidation(
            type="list",
            formula1='"morris_ppc2,posttest_smd,change_smd,raw_posttest_md,raw_change_md,log_rr,log_or,risk_difference"',
            allow_blank=True,
        )
        ws.add_data_validation(method_validation)
        method_letter = ws.cell(row=1, column=method_col).column_letter
        method_validation.add(f"{method_letter}2:{method_letter}501")

    info = wb.create_sheet("Instructions")
    info.append(["Field / topic", "Instructions"])
    for row in INSTRUCTIONS:
        info.append(list(row))
    info.freeze_panes = "A2"
    info.sheet_view.showGridLines = False
    info.column_dimensions["A"].width = 28
    info.column_dimensions["B"].width = 105
    for cell in info[1]:
        cell.fill = PatternFill("solid", fgColor=dark)
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    info.row_dimensions[1].height = 28
    for row in info.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = Border(bottom=Side(style="hair", color="E5EAF0"))
        info.row_dimensions[row[0].row].height = 32

    wb.properties.title = f"MetaPy 3.3.0 {template_type.title()} Pairwise Input Template"
    wb.properties.subject = "Blank workbook template for MetaPy pairwise analysis"
    wb.properties.description = "Researcher-entered study extraction data; blank template, not clinical evidence."
    wb.save(output_path)
    return str(output_path)


def create_default_templates(directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    return {
        "continuous": create_input_template(directory / "MetaPy_Pairwise_Continuous_Input_Template.xlsx", "continuous"),
        "binary": create_input_template(directory / "MetaPy_Pairwise_Binary_Input_Template.xlsx", "binary"),
    }
