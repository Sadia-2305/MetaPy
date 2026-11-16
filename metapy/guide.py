"""Help for researchers who are new to meta-analysis (MetaPy 5.2).

make_template   : an Excel data-entry template with instructions and example rows MetaPy can read directly
interpret       : plain-language "what this means" sentences for one pooled result
warnings_for    : situations in which a result should be read with caution, in plain language
EXPLAIN         : one-line explanations of the main choices, used as help text in the apps
CITATION        : how to cite MetaPy

The wording is deliberately cautious: it describes what the numbers show and what they cannot show, and never says
that an intervention "works" or should be used.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__

VIDEO = "https://youtu.be/wqCUJPEgjXQ"
CITATION = (f"Nadeem S. MetaPy: transparent meta-analysis [computer software]. Version {__version__}. Zenodo. "
            "https://doi.org/10.5281/zenodo.24267388")

EXPLAIN = {
    "random": "Random effects: assumes the true effect differs somewhat between studies (different people, settings, "
              "doses). This is the usual choice and the one MetaPy recommends.",
    "fixed": "Fixed (common) effect: assumes every study estimates exactly the same effect. Only sensible when studies "
             "are almost identical.",
    "REML": "How the between-study variation (τ²) is estimated. REML is the recommended default; change it only if a "
            "protocol or reviewer asks for another method.",
    "HK": "Hartung-Knapp confidence intervals are wider and more honest when there are few studies. Recommended.",
    "Wald": "Wald (normal) intervals: the traditional choice, but too narrow when studies are few.",
    "extras": "Optional extra analyses. In beginner mode only the ones suited to your data are offered; each needs a "
              "minimum number of studies.",
}

BEGINNER_EXTRAS = ("loo", "influence", "subgroup::", "metareg::", "egger", "begg", "trimfill", "tau_ci", "mh", "glmm")

_TEMPLATE_ROWS = [
    # continuous outcome: different reporting formats, one per row
    {"Study": "Example A 2019", "Year": 2019, "Outcome": "Reading score (example)", "Intervention": "Programme", "Country": "Country 1",
     "N_I": 40, "Mean_Post_I": 72.4, "SD_Post_I": 10.1, "N_C": 38, "Mean_Post_C": 68.0, "SD_Post_C": 11.3},
    {"Study": "Example B 2020", "Year": 2020, "Outcome": "Reading score (example)", "Intervention": "Programme", "Country": "Country 2",
     "N_I": 55, "Mean_Pre_I": 60.2, "SD_Pre_I": 9.8, "Mean_Post_I": 70.1, "SD_Post_I": 10.4,
     "N_C": 52, "Mean_Pre_C": 61.0, "SD_Pre_C": 10.2, "Mean_Post_C": 64.9, "SD_Post_C": 10.9},
    {"Study": "Example C 2021", "Year": 2021, "Outcome": "Reading score (example)", "Intervention": "Programme", "Country": "Country 1",
     "N_I": 30, "Median_I": 71, "Q1_I": 64, "Q3_I": 79, "N_C": 31, "Median_C": 66, "Q1_C": 58, "Q3_C": 73},
    {"Study": "Example D 2021", "Year": 2021, "Outcome": "Reading score (example)", "Intervention": "Programme", "Country": "Country 3",
     "N_I": 48, "N_C": 47, "MD (95% CI)": "4.2 (0.9 to 7.5)"},
    {"Study": "Example E 2022", "Year": 2022, "Outcome": "Reading score (example)", "Intervention": "Programme", "Country": "Country 2",
     "N_I": 25, "N_C": 26, "Mean (SD) intervention": "69.8 (12.0)", "Mean (SD) control": "66.1 (11.5)"},
    {"Study": "Example F 2023", "Year": 2023, "Outcome": "Reading score (example)", "Intervention": "Programme", "Country": "Country 3",
     "N_I": 60, "N_C": 61, "Hedges g": 0.31},
    # binary outcome
    {"Study": "Example A 2019", "Year": 2019, "Outcome": "Dropped out (example)", "Intervention": "Programme", "Country": "Country 1",
     "N_I": 40, "Events_I": 4, "N_C": 38, "Events_C": 9},
    {"Study": "Example B 2020", "Year": 2020, "Outcome": "Dropped out (example)", "Intervention": "Programme", "Country": "Country 2",
     "N_I": 55, "Events_I": 6, "N_C": 52, "Events_C": 10},
    {"Study": "Example G 2022", "Year": 2022, "Outcome": "Dropped out (example)", "Intervention": "Programme", "Country": "Country 1",
     "N_I": 80, "Events_I": 0, "N_C": 78, "Events_C": 3},
]

_COLUMNS = ["Study", "Year", "Outcome", "Intervention", "Country", "N_I", "N_C",
            "Mean_Pre_I", "SD_Pre_I", "Mean_Post_I", "SD_Post_I", "Mean_Pre_C", "SD_Pre_C", "Mean_Post_C", "SD_Post_C",
            "Median_I", "Q1_I", "Q3_I", "Median_C", "Q1_C", "Q3_C", "MD (95% CI)", "Mean (SD) intervention", "Mean (SD) control",
            "Hedges g", "Events_I", "Events_C"]

_GUIDE = [
    ("Start here", ""),
    ("1", "Each row is one comparison (intervention versus control) for one outcome in one study."),
    ("2", "Replace the example rows (studies called 'Example ...') with your own studies, or delete them."),
    ("3", "Fill in whatever each study reports. You do not need every column: MetaPy works out the effect size from the "
          "columns you fill in and tells you if something is missing."),
    ("4", "Use the same Outcome name for all rows that should be pooled together (for example 'Reading score')."),
    ("5", "Leave cells empty when a value is not reported. Do not type 0 for a missing value."),
    ("6", "Columns MetaPy does not recognise (for example 'Country' or 'Risk of bias') are kept as study characteristics "
          "and can be used for subgroup analyses."),
    ("7", "Keep this instructions sheet or delete it; MetaPy ignores it."),
    ("", ""),
    ("Column", "What to enter"),
    ("Study", "A name for the study, e.g. 'Khan 2021'. Use the same name in every row from the same study."),
    ("Year", "Publication year (optional)."),
    ("Outcome", "What was measured. Rows with the same outcome name are pooled together."),
    ("Intervention", "Name of the intervention group (useful when a study has two or more intervention groups)."),
    ("N_I / N_C", "Number of participants analysed in the intervention / control group."),
    ("Mean_Post_I, SD_Post_I, Mean_Post_C, SD_Post_C", "Mean and standard deviation after the intervention."),
    ("Mean_Pre_..., SD_Pre_...", "Optional: the same values at baseline, when reported."),
    ("Median_..., Q1_..., Q3_...", "When a study reports medians and quartiles (interquartile range) instead of means."),
    ("MD (95% CI)", "A reported mean difference with its confidence interval, typed as it appears, e.g. '4.2 (0.9 to 7.5)'."),
    ("Mean (SD) intervention / control", "Mean and SD typed in one cell, e.g. '69.8 (12.0)'."),
    ("Hedges g", "A standardized mean difference reported by the study."),
    ("Events_I / Events_C", "For yes/no outcomes: number of participants with the event in each group."),
    ("", ""),
    ("Good practice", "Follow the Cochrane Handbook (training.cochrane.org/handbook) and report your review with PRISMA 2020 "
                      "(prisma-statement.org). Plan the review and register a protocol before you start."),
]


def make_template(path: str | Path) -> Path:
    """Write the data-entry template (instructions sheet + data sheet with example rows)."""
    path = Path(path)
    data = pd.DataFrame(_TEMPLATE_ROWS, columns=_COLUMNS)
    guide = pd.DataFrame(_GUIDE, columns=["", "MetaPy data template"])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        guide.to_excel(xw, sheet_name="How to fill in", index=False)
        data.to_excel(xw, sheet_name="Data", index=False)
        ws = xw.sheets["How to fill in"]
        ws.column_dimensions["A"].width = 38
        ws.column_dimensions["B"].width = 110
        wd = xw.sheets["Data"]
        for i, c in enumerate(_COLUMNS):
            wd.column_dimensions[wd.cell(row=1, column=i + 1).column_letter].width = max(10, min(26, len(c) + 3))
        wd.freeze_panes = "B2"
    return path


# ============================================================================ interpretation
def _size_smd(g: float) -> str:
    a = abs(g)
    if a < 0.2:
        return "very small"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "moderate"
    return "large"


def _fmt(x, d=2):
    return f"{x:.{d}f}"


def interpret(res) -> list[str]:
    """Plain-language sentences about one pooled result (cautious wording)."""
    from .figures import favours_labels
    m = res.main
    if m.get("status") != "Pooled":
        return [f"No pooled result: {m.get('status')}."]
    k = int(m.get("k", len(res.studies)))
    meas = res.measure
    est, lo, hi = m["estimate"], m["ci_lo"], m["ci_hi"]
    null = 0.0
    left, right = favours_labels(res)
    side = right if est > null else left
    out = []
    ci_name = "95% confidence interval"
    if meas.back is not None:                        # ratio measures, shown on the natural scale
        e, l_, h_ = math.exp(est), math.exp(lo), math.exp(hi)
        name = meas.short if hasattr(meas, "short") else meas.label
        change = (e - 1) * 100
        word = "lower" if change < 0 else "higher"
        out.append(f"Across {k} studies, the pooled {name} was {_fmt(e)} ({ci_name} {_fmt(l_)} to {_fmt(h_)}). "
                   f"In other words, the odds or risk of the outcome were on average about {abs(change):.0f}% {word} in the "
                   "intervention groups than in the comparison groups.")
        crosses = l_ <= 1 <= h_
        no_effect = "1 (no difference)"
    else:
        name = meas.label
        size = f", a {_size_smd(est)} difference by common rules of thumb (0.2 small, 0.5 moderate, 0.8 large)" \
            if meas.key.startswith("smd") else ""
        out.append(f"Across {k} studies, the pooled {name} was {_fmt(est)} ({ci_name} {_fmt(lo)} to {_fmt(hi)}){size}. "
                   f"The direction is: {side.lower()}.")
        crosses = lo <= 0 <= hi
        no_effect = "0 (no difference)"
    if crosses:
        out.append(f"The confidence interval includes {no_effect}, so the data are compatible with no average difference as "
                   "well as with a difference in either direction. This does not show that there is no effect.")
    else:
        out.append(f"The confidence interval does not include {no_effect}, so the average difference is unlikely to be due to "
                   "chance alone. It says nothing about study quality, bias or how important the difference is in practice.")
    if m.get("model") == "random" and k >= 3:
        i2 = m.get("I2") or 0.0
        level = "little" if i2 < 25 else "moderate" if i2 < 50 else "substantial" if i2 < 75 else "considerable"
        out.append(f"The results varied {level}ly between studies (I² = {i2:.0f}%)." if level != "little"
                   else f"The results were fairly consistent between studies (I² = {i2:.0f}%).")
        pl, ph = m.get("pi_lo"), m.get("pi_hi")
        if pl is not None and ph is not None and math.isfinite(pl) and math.isfinite(ph):
            if meas.back is not None:
                pl, ph = math.exp(pl), math.exp(ph)
                pin = pl <= 1 <= ph
            else:
                pin = pl <= 0 <= ph
            out.append(f"In a new study like these, the effect could plausibly lie between {_fmt(pl)} and {_fmt(ph)} (95% "
                       "prediction interval)." + (" Because this range includes no difference, the effect may be absent or "
                       "reversed in some settings." if pin else ""))
    return out


# ============================================================================ warnings
def warnings_for(res) -> list[str]:
    """Plain-language cautions for one result (empty list if none apply)."""
    m = res.main
    w: list[str] = []
    if m.get("status") != "Pooled":
        return w
    d = res.studies
    k = int(m.get("k", len(d)))
    if k < 5:
        w.append(f"Only {k} studies: the estimate of how much studies differ (τ²) is very uncertain, and the confidence "
                 "interval may be unreliable. Present the result cautiously.")
    if k < 10:
        w.append("Fewer than 10 studies: funnel plots and tests for publication bias cannot be interpreted reliably and are "
                 "not shown.")
    i2 = m.get("I2") or 0.0
    if m.get("model") == "random" and k >= 3 and i2 >= 75:
        w.append(f"Very large differences between studies (I² = {i2:.0f}%): a single average may be misleading. Look for "
                 "reasons (subgroups, meta-regression) before drawing conclusions.")
    if m.get("model") == "fixed" and k >= 3 and (m.get("Q_p") or 1) < 0.10:
        w.append("You chose a fixed-effect model, but the studies differ more than chance would explain. A random-effects "
                 "model is usually more appropriate.")
    if "weight_percent" in d and len(d) >= 3 and float(d["weight_percent"].max()) > 50:
        top = d.loc[d["weight_percent"].idxmax()]
        w.append(f"One study ({top.get('label', top.get('study_id'))}) carries more than half of the weight, so the result "
                 "depends heavily on it. Check the leave-one-out analysis.")
    if "approximations_or_assumptions" in d:
        flagged = d["approximations_or_assumptions"].astype(str).str.strip().replace({"nan": "", "None": "", "—": ""}).ne("")
        n_ap = int(flagged.sum())
        if n_ap and n_ap / max(len(d), 1) >= 0.3:
            w.append(f"{n_ap} of {len(d)} effect sizes rely on approximations or assumptions (see the derivation log). "
                     "Consider a sensitivity analysis without them.")
    eg = res.extras.get("egger") if isinstance(res.extras, dict) else None
    if isinstance(eg, dict) and eg.get("p") is not None and k >= 10 and eg["p"] < 0.10:
        w.append("Egger's test suggests small-study effects (smaller studies show different results). This can be caused by "
                 "publication bias, but also by genuine differences between small and large studies.")
    return w
