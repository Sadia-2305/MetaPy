# MetaPy 3.3.0 — Complete Pairwise Analysis Pipeline

MetaPy 3.3.0 is a self-contained, researcher-controlled Excel-to-analysis workflow. It is packaged in a separate Python namespace (`metapy_complete`) so it does not overwrite the older `metapy` modules in an existing Colab notebook.

## What this release completes

### 1. Workbook ingestion and comparison-level harmonisation

- Reads all non-empty worksheets in a supplied `.xlsx` / `.xlsm` workbook.
- Provides downloadable blank Excel templates for pairwise continuous and binary data (`examples/MetaPy_Pairwise_Continuous_Input_Template.xlsx` and `examples/MetaPy_Pairwise_Binary_Input_Template.xlsx`). The templates include field guidance and dropdowns for study design, outcome direction and effect method.
- Maps common aliases for study ID, outcome, duration, intervention arm, sample sizes, pre/post means and SDs, reported effects, correlations, units and binary event counts.
- Recognises `Intervention_Arm` and related aliases.
- Parses an arm name only when it is explicitly embedded in the supplied StudyID or outcome label (for example, `06-Meditation` or `HADS-A Yoga`). It does not invent an arm label.
- Merges repeated comparison records across worksheets and combines complementary fields.
- If repeated source values conflict, the conflicting field is set to missing and a row-specific conflict is exported. MetaPy does not silently choose one value.
- Multiple unlabelled intervention rows for the same study/outcome/timepoint in one worksheet are retained as separate ambiguous comparisons and flagged instead of being silently merged.
- Retains source worksheet/row/column provenance for mapped values.
- Retains `MD (95% CI)` text and can parse it for review, but it is not used unless the researcher explicitly confirms the matching raw-MD estimand.

### 2. Researcher decisions and data-preparation gates

- Detects source correlations and carries both intervention- and control-group values through the prepared dataset. Actual source values take priority over assumptions.
- Recognises a single unit declared for a comparison without claiming external unit verification. A missing unit blocks pooling for raw mean-difference analyses, but not for standardised effects.
- Distinguishes global project decisions from approval of each individual comparison.
- Creates an editable record-level table with direction multiplier, study-design confirmation and `approve_record` fields.
- Uses a benefit-oriented sign convention: use `+1` when higher values are worse; use `-1` when higher values are better. This makes a beneficial effect negative.
- Does not call a workbook fully approved merely because global forms were completed.
- Exports explicit data-quality findings, researcher review items, conflicts, unmapped columns, and an exact transformation log. The data-quality sheet has headers and an explicit “No findings detected” row when there are no findings.

### 3. Effect measures supported in v3.3.0

**Pairwise continuous outcomes**

- `morris_ppc2`: Hedges-corrected difference in change standardised by the pooled baseline SD. The point estimate needs pre/post means, baseline SDs and group sample sizes. The approximate variance uses group-specific pre/post SDs and actual group correlations (or a separate researcher-authorised sensitivity value where a source correlation is missing).
- `posttest_smd`: posttest Hedges' g from posttest means, SDs and sample sizes.
- `change_smd`: change-score Hedges' g; requires actual group-specific pre/post correlations to derive SD(change).
- `raw_posttest_md`: raw posttest mean difference.
- `raw_change_md`: raw between-group difference in change; requires group-specific correlations to derive change-score SDs.

**Pairwise binary outcomes**

- Log risk ratio (`log_rr`).
- Log odds ratio (`log_or`).
- Risk difference (`risk_difference`).
- For log RR/OR, a 0.5 continuity correction is applied only when a zero cell is present; the correction is shown in the calculation method.

### 4. Sampling variance and estimand safeguards

- A reported Hedges' g value is not assumed to be Morris PPC2, posttest SMD or change-score SMD. Its method must be confirmed to match the selected estimand before it can be used.
- When raw data can calculate the selected estimand, MetaPy calculates the raw-data effect and compares it with a reported value only when both are confirmed to be the same estimand.
- If a PPC2 point estimate is available but a compatible variance is not, MetaPy reports the point estimate and keeps the record out of primary pooling.
- A direct variance, SE or CI is used only when the reported effect method is explicitly confirmed to match the selected estimand.
- A 95% CI converted to a variance uses a normal critical value; the report records that method.
- PPC2 variance is labelled as an approximate delta-method calculation. It should be inspected and transparently described in a manuscript.
- Assumed correlations are not written back as source data. They are used only in a separately labelled sensitivity analysis, and any observed group-specific correlation is retained while only a missing group value is varied.

### 5. Pairwise pooling and checks

- Fixed-effect pooling.
- Random-effects pooling with REML or DerSimonian–Laird tau-squared.
- Hartung–Knapp or normal-approximation random-effects confidence intervals.
- Q, Q p-value, I-squared, tau-squared, tau and (when estimable) a prediction interval.
- Leave-one-study-out sensitivity analysis.
- Egger regression only when there are at least 10 independent studies and no duplicated study IDs in the pool. Otherwise, it is explicitly not performed.
- Forest plots for eligible outcome groups. Funnel plots are only generated when there are at least 10 independent studies without duplicate study IDs.
- User-adjustable plot colour theme (navy, teal, purple, monochrome or orange), forest-plot width, and row height.

### 6. Multi-arm / shared-control safeguards

The default is to **block** a possible shared-control group until the researcher makes a decision. If a `Control_ID` is supplied, repeated IDs are used to flag known shared controls; when IDs are absent, equal control sample sizes within the same study/outcome/timepoint are used as a conservative warning heuristic. Unique explicit control IDs prevent that equal-N heuristic from incorrectly treating two separately identified controls as one. The interface offers:

1. Block until the dependence issue is resolved (default).
2. Split the shared control sample size for variance weights only.
3. Confirm that the controls are genuinely separate and independent.

When option 2 is selected, the original control N remains unchanged for the point estimate and Hedges' small-sample correction. The split is used only in the variance-weighting approximation. This is not a multivariate meta-analysis, robust-variance estimation or a full model of within-study covariance.

## Colab installation and launch

1. Download `MetaPy_v3_3_0_Complete_Analysis_Pipeline.zip` and upload it into your Google Colab `/content` directory.
2. Run the following cells:

```python
!unzip -oq /content/MetaPy_v3_3_0_Complete_Analysis_Pipeline.zip -d /content
%cd /content/MetaPy_v3_3_0_Complete
!pip -q install -r requirements.txt
!python -m unittest discover -s tests -v
```

3. Launch the new interface:

```python
!python /content/MetaPy_v3_3_0_Complete/run_metapy_complete.py
```

4. You can download the continuous or binary input template from the app, fill it in Excel, then upload the completed workbook. Alternatively, upload an existing workbook with supported column names.
5. Click **Inspect workbook**, review every canonical comparison, complete direction/design fields, tick `approve_record` only after checking each row, select the estimand and analysis settings, then click **Calculate effects and pool eligible comparisons**.
6. The report is written to `/content/MetaPy_Analysis_Report.xlsx` by default. The source workbook is never modified.

## Output workbook sheets

- Run Summary
- Canonical Source Records
- Prepared Dataset
- Effect Calculations
- Primary Pooled Results
- Data Quality Findings
- Researcher Review Items
- Analysis Blockers
- Source Conflicts
- Unmapped Columns
- Correlation Sensitivity
- Sensitivity Pooled
- Leave-One-Out
- Transformation Log
- Decision Audit
- Data Dictionary
- Visualisations

## Important limitations

- This version is a **pairwise** meta-analysis release. It supports continuous and binary outcomes in parallel independent groups. It blocks known crossover and cluster designs. It does not implement network meta-analysis, multivariate meta-analysis, robust-variance estimation, diagnostic test accuracy, survival outcomes, rate ratios, meta-regression, or multilevel models.
- The shared-control split is a variance-weighting approximation, not a substitute for modelling covariance.
- PPC2's point estimate is calculated separately from its variance. The implemented variance is an approximate delta-method route; verify its suitability for the planned protocol and report the chosen method.
- The software performs internal structural, plausibility and mathematical checks only. It does not establish that the workbook values are authentic, correctly extracted from publications, or clinically true.
- The default launcher uses a temporary Gradio share link so the interface can be opened in a browser. Anyone with that link may be able to access the running app during the session. Use aggregate, de-identified extraction data only; do not upload patient identifiers, participant-level data or confidential files. To disable public sharing, set `METAPY_SHARE=false` before launching (for example, `import os; os.environ['METAPY_SHARE']='false'`). For confidential work, use a private/local environment.
- Synthetic test values are software test data only. They are not clinical evidence and must not be cited or published as research findings.
- Passing data preparation does not mean effect-size calculation is complete. Passing effect-size calculation does not mean pooling is permitted. The report keeps these gates separate.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```

The regression suite contains 26 tests covering column alias recognition, source provenance, PPC2 estimate/variance separation, actual and assumed correlations, ambiguous multi-arm records, method-confirmed reported effects, raw-MD confidence intervals, all supported continuous and binary effect measures, shared-control IDs and fallback handling, unsupported study designs, preparation gates, fixed/random pooling, input template generation and visualisation settings.
