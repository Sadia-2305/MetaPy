# MetaPy 5.2: transparent meta-analysis for every researcher

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24267388.svg)](https://doi.org/10.5281/zenodo.24267388)

🎬 **Video guide (2 min):** https://youtu.be/wqCUJPEgjXQ · 🌐 **Try it online:** https://metapy-23.streamlit.app

MetaPy reads your extraction sheet (CSV or Excel). It works out which meta-analyses the data can support, derives any missing statistics itself and shows exactly how, then runs the analysis and builds the report. You make every decision by picking options; there is no free text to type.

```
1 Upload  →  2 Scan & choose  →  3 Prepare (derivation brain)  →  4 Approve
          →  5 Analyse  →  6 Review / add analyses  →  7 Tables & figures (incl. PRISMA)  →  8 Export
```

## The derivation brain

`metapy/derive/` holds a registry of **180 cited formulas**; `docs/DERIVATION_RULES.md` lists all of them. A solver chains them for each study:

1. **Exact rules first.** Examples: SD from SE or 95% CI, Cohen's d from t or p, Hedges' g from d, log OR and its variance from a 2×2 table or from an OR with CI, change-score SD from the pre-post correlation, and Morris PPC2.
2. **Published approximations next.** Examples: mean/SD from median, IQR or range (Wan 2014; Luo 2018), OR↔SMD (Chinn 2000), r↔d, Spearman to Pearson, events from percentages, and p-value thresholds such as "p<0.05". The researcher can veto each group.
3. **Assumptions last**, and only when the researcher switches them on: imputed pre-post r, equal split of a total N, and the type of an unlabelled SMD.

Every calculated value records its formula, input values and citation. **Nothing is fabricated**: if no chain exists, MetaPy asks for data. It verifies that each suggested request would actually work, e.g. *"SD post-test (intervention) + SD post-test (control)"* or *"MD 95% CI"*. It also re-derives reported values from their raw data and flags extraction errors (consistency checks).

## What else it does

- **Reading**: CSV, TSV, XLSX, XLSM; all sheets; auto-detects header rows; recognises hundreds of header spellings (`Mean_Post_I`, `SD baseline control`, `N1`, `Events (treatment)`…); parses text cells such as `12.3 (4.1)`, `1.2 (0.8 to 1.6)`, `10 [8-14]`, `45%` and `p<0.05`. Unrecognised columns become study characteristics or moderators.
- **Scanning**: for every outcome × effect measure it reports how many studies are usable directly, by exact calculation, by approximation, by assumption, or not at all, then ranks the plans.
- **Effect measures**: Hedges' g, Cohen's d, MD, change-score g and MD, Morris PPC2, log OR, log RR, RD, log HR, Fisher z (r), logit proportion, generic inverse-variance.
- **Preparation decisions (as options)**: SMD type, imputed r, equal-N split, accepting or vetoing approximations, consistency conflicts, multi-arm studies (split the shared control, combine arms, keep the first, keep all), outcome direction, exclusions, and entering missing values.
- **Analysis**: fixed and random effects; τ² by REML, DL or PM; Hartung-Knapp or Wald CIs; prediction intervals; I² consistent with τ². Optional extras:
  - subgroup analysis with a test for differences, and meta-regression (Knapp-Hartung, R²)
  - leave-one-out, influence diagnostics and cumulative meta-analysis
  - fixed vs random comparison
  - exact-only and no-assumption sensitivity analyses, plus an imputed-r grid
  - Egger, Begg and trim-and-fill
  - **new in 5.0** (details below): three-level models and CHE-RVE for dependent effects, network meta-analysis, selection model (3PSM),
    PET-PEESE, Peters' test, fail-safe N, Q-profile CIs for τ²/I²/H², Baujat plot, Bayesian random effects, multiple meta-regression
    with permutation tests
- **Review loop**: accept the results, add analyses, change the model, or go back to preparation.
- **Outputs**:
  - Figures: forest, subgroup forest, contour-enhanced funnel (with trim-and-fill), bubble, leave-one-out, cumulative, data-provenance chart, PRISMA 2020 flow diagram.
  - Tables: study characteristics, effect sizes, pooled results, summary of findings (GRADE columns blank for reviewers), extra-analysis tables, derivation log, data requests.
  - Auto-written methods (with citations) and results paragraphs.
- **Export**: Word, Excel, PDF, HTML, CSV (zip), PNG 300 dpi, TIFF 600 dpi, SVG, and a JSON analysis record of every setting and decision for reproducibility.

## Quick start

**Colab**: open `MetaPy_v4.ipynb` and run the cells in order.

**Local**:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m metapy                  # opens the app (set METAPY_SHARE=false for no public link)
```

**From code**:

```python
from metapy import MetaPySession
s = MetaPySession()
scan = s.load("my_extraction.xlsx")
print([p["summary"] for p in scan.plans])
pid = scan.plans[0]["plan_id"]
s.select_plans([pid])
prep = s.prepare(pid)              # prep.actions, prep.log, prep.questions, prep.requests
s.answer(pid, {q["id"]: q["default"] for q in prep.questions})
s.approve(pid)
s.analyse({"model": "random", "tau_method": "REML", "ci_method": "HK", "extras": ["loo", "egger"]})
s.export(["docx", "xlsx"], "output")
```

## Validation

The folder `validation/` reproduces every number in the MetaPy paper (comparison with metafor's documented results, an
independent implementation, and simulations); see `validation/README.md`.


`tests/test_metapy.py` (42 tests) and `tests/test_v5.py` (17 tests, see above) check:
- pooling against **metafor** reference values for the BCG dataset (REML, DL, fixed effect, and meta-regression on latitude)
- every derivation route against hand calculations, and that d from t, p, or means agree
- the approximation veto, that assumptions apply only when enabled, and the consistency checks
- that suggested data requests are valid
- header and text-cell parsing, and the full workflow with every export format
- that the source file is never modified

`examples/MetaPy_SYNTHETIC_demo.xlsx` is **synthetic test data**, not real studies.

## New in 5.0: statistical depth

| Area | Method | Module | Reference |
|---|---|---|---|
| Dependent effects | Three-level random-effects model (effects within studies), REML; LRTs for each variance component with the 50:50 χ² boundary mixture; level-specific I² | `dependent.py` | Konstantopoulos 2011; Cheung 2014 |
| Dependent effects | Correlated-hierarchical-effects (CHE) working model + cluster-robust CR2 standard errors with Satterthwaite df; sensitivity to the assumed ρ (0–0.8); warning when df < 4 | `dependent.py` | Pustejovsky & Tipton 2018, 2022; Tipton 2015 |
| Heterogeneity | Q-profile CIs for τ², τ, I², H² | `advanced.py` | Viechtbauer 2007 |
| Small-study effects / publication bias | PET, PEESE, conditional PET-PEESE | `advanced.py` | Stanley & Doucouliagos 2014 |
| | Three-parameter selection model (one cut at two-sided p = .05), ML with LRT for selection | `advanced.py` | Vevea & Hedges 1995 |
| | Peters' regression test (binary outcomes) | `advanced.py` | Peters et al. 2006 |
| | Fail-safe N (Rosenthal, Orwin), reported as descriptive only | `advanced.py` | Rosenthal 1979; Orwin 1983 |
| Influence | Baujat plot | `advanced.py`, `figures.py` | Baujat et al. 2002 |
| Moderators | Multiple meta-regression with permutation p-values | `advanced.py` | Higgins & Thompson 2004 |
| Bayesian | Normal-normal model, half-normal prior on τ, vague normal prior on μ; exact grid integration; posterior, credible and predictive intervals, P(μ > 0) | `advanced.py` | Röver 2020; Röver et al. 2021 |
| Network meta-analysis | Contrast-based GLS, common τ² (REML), exact multi-arm covariances (MD, log OR, log RR) and Gleser–Olkin (SMD); league table; P-scores; Q decomposition into within-design heterogeneity and between-design inconsistency; direct vs indirect evidence by back-calculation; network graph and forest plot | `nma.py` | Rücker 2012; Krahn et al. 2013; Rücker & Schwarzer 2015; Gleser & Olkin 2009 |

In the app the new methods are listed under **Additional analyses** (Step 3). The network panel is in Step 3 too: it accepts a sheet in
arm format (`study, treatment, mean, sd, n` or `events, n`), a sheet in contrast format (`study, treat1, treat2, TE, seTE`), or a prepared
outcome whose arms are named. To give the three-level and CHE-RVE models something to model, keep multi-arm comparisons as separate rows
(multi-arm option *keep all*) and/or include all time points; with one effect per study they reduce to the ordinary random-effects model and MetaPy says so.

```python
s.analyse({"extras": ["multilevel", "che_rve", "tau_ci", "pet_peese", "selection", "bayes", "baujat"], "rve_rho": 0.6})
s.nma_sources()                      # sheets / prepared outcomes usable as a network
s.nma("sheet::Arms", measure="SMD", reference="Placebo", model="random", small_values="good")
```

### How the new methods were validated (`tests/test_v5.py`, 17 tests)
- **Three-level model**: with one effect per study it reproduces the metafor REML random-effects result for BCG (μ = −0.7145, τ² = 0.3132); on simulated three-level data it recovers the components, and the LRT detects the between-study level.
- **CHE-RVE (CR2)**: Monte Carlo coverage of the 95% CI (120 simulated meta-analyses of 15 studies) is checked to be at least 88%.
- **Q-profile**: BCG τ² CI 0.1197 to 1.1115, matching metafor `confint()`.
- **PET**: identical to an independent weighted least-squares fit; **3PSM**: without selection its ML estimate equals an independently maximised ML random-effects model.
- **Bayesian model**: posterior medians of μ and τ agree with a brute-force two-dimensional grid to within 0.01.
- **Network meta-analysis**: the common-effect MD network equals an independent arm-based weighted least-squares model with study intercepts to 10⁻⁸; a star network reproduces pairwise REML; a consistent triangle gives inconsistency Q = 0; multi-arm SMD covariance matrices are positive definite; disconnected networks are refused.
- **Not yet cross-checked against R** (please do this before publishing): 3PSM vs `metafor::selmodel`, CR2 vs `clubSandwich`, NMA τ², P-scores and Q decomposition vs `netmeta`, Bayesian results vs `bayesmeta`. The formulas follow those packages' published methods, but agreement has only been shown on the cases above.

## Online version (Streamlit)
`streamlit_app.py` is a website version of the same five steps, for Streamlit Community Cloud (or any server):
`pip install -r requirements.txt` then `streamlit run streamlit_app.py`. It runs in website mode: the uploaded file
and all results stay in the visitor's browser session only; nothing is saved on the server or shared between users,
and the saved-session feature is switched off. Uploads are limited to 25 MB (`.streamlit/config.toml`).
The Colab notebook and `python -m metapy` (Gradio) keep working as before.

## PRISMA 2020 diagram: free text (5.0)
The PRISMA form has one text box per box of the diagram: *Records identified from:*, *Records removed before screening:*, *Records screened*,
*Records excluded*, *Reports sought for retrieval*, *Reports not retrieved*, *Reports assessed for eligibility*, *Reports excluded:*,
*Studies included in review*, an optional *Identification via other methods* column, and a *Note:* printed under the diagram.
No database names are suggested: type or paste anything, on as many lines as needed; text is printed exactly as written
(start a line with two spaces to indent it). The diagram stays optional. In Python:
`s.export(["docx"], prisma={"box_identified": "Databases (n = 3,104)", "box_included": "(n = 10 RCTs)", "note": "…"})`.
Older number-based PRISMA input still works (it is converted to text and checked arithmetically).

## Changes in 4.2: one data file at a time, new PRISMA form

- **Clean slate for every file.**
  - Reading a file starts a new session and clears every panel: plans, answers, approvals, results, preview, export and the PRISMA form.
  - Each file gets a SHA-256 fingerprint. It's shown in Step 1 and printed in the report, and autosaves are kept per file, so even two files with the same name never share a save.
  - A report can only be built from results produced from the current file; anything else is refused.
- **PRISMA 2020 is optional.** It's off by default and drawn only when you switch it on.
- **New PRISMA form.**
  - A table of databases (name and records) and of registers. Each one is printed in the diagram with its count, and the totals are calculated.
  - Duplicates, automation and other removals are shown with their total.
  - A table of exclusion reasons; its total is calculated.
  - Blank boxes are calculated and every step is checked for consistency.
  - The included box can show the study label (e.g. "RCTs"), the number of comparisons and the total sample size. One button fills these from your analysis, counting shared controls once.
  - Optional footnote for the records-excluded box, and an optional free-text **Note** printed under the diagram.
  - The layout follows the usual journal style: banner, phase labels, database list and note box.

## Changes in 4.1.3: review of a real analysis

- **Participants** are counted once per person: a control group shared by several arms is no longer counted twice.
- **Shared control groups**: when the control N is split for the weights, the effect estimate is held fixed even when the SD itself has to be derived from N (e.g. from an MD and its 95% CI). Variances change by under 1%.
- **Direction**: for well-known outcomes, "higher is worse" is pre-selected (anxiety, depression, stress, pain, CRP, IL-6, TNF-α, cortisol…) and an opposite answer is flagged. The plot labels follow your answer.
- **PRISMA**: a "total" line inside the exclusion reasons (e.g. "Reports excluded: n = 206") is no longer added to the reasons; it is checked against their sum. Reports included = assessed − excluded is filled in, and missing boxes are listed.
- **Best-available design** is recommended over endpoint g when the reported g values (or most studies) are baseline-adjusted. This avoids sign errors caused by baseline imbalance (in the real data that prompted this, one study's endpoint g was +0.38 but its PPC2 was −0.18). MetaPy now also asks about the pre-post correlation for this measure.
- **Methods text** names the effect measure used for each outcome; an unnamed notes column is labelled "Notes".

## Changes in 4.1.2: verification fixes

- **PRISMA**: every step of the flow is checked: identified − removed = screened, screened − excluded = sought, sought − not retrieved = assessed, assessed − excluded = included, and no box can be larger than the box above it. Problems are listed in red under the diagram, in the preview and in the report.
- **Direction conflict warning**: if your answer about direction contradicts the data file's direction column, MetaPy warns you. This matters because the "Favours intervention / control" plot labels follow your answer.
- **Data file shown everywhere**: the file name appears in Step 2, the preview, the export message and the resume message, so a report can't silently come from a different file.
- **Verification**: on a real mixed workbook (36 comparisons, 5 outcomes), every study-level g and variance and every pooled statistic matched an independent implementation to within 10⁻⁶.

## Changes in 4.1.1: approvals and saving

- **Approvals are kept.** Before this fix, re-running preparation erased the approvals of every plan, so only the last plan was analysed. Now a plan is un-approved only when its prepared numbers change, and MetaPy says which plan.
- **Approve all plans at once.** Step 2 ends with a checklist of every plan.
- **Choose what to analyse.** Step 3 has a checklist of plans to analyse (all approved ones are ticked), and you can approve the remaining ones there.
- **Autosave and resume.** The session is saved after every step to `/content/metapy_sessions` in Colab, or `~/.metapy_sessions` locally. You can resume it in Step 1 after a disconnect.

## Changes in 4.1: mixed and messy data

- **Duplicate rows are merged across sheets.** Rows for the same study, outcome, time point and arm are combined. Agreeing values are kept once, conflicting values are listed, and nothing is double-counted.
- **Scales are grouped into constructs.** HAM-A, HADS-A, BAI, GAD-7, DASS-A, STAI and similar go to Anxiety. Depression, stress, sleep, QoL and pain scales are grouped the same way. Biomarker synonyms are merged (hsCRP→CRP, TNF-a→TNF-α, IL6→IL-6). The scale stays available as a subgroup, and you can turn the grouping off in Step 1.
- **Arms hidden in text are recognised**, e.g. `06-Meditation`, `06 Yoga` and `HADS-A Meditation` (study 6, arms Meditation and Yoga, sharing a control).
- **The design of a reported Hedges' g is worked out from evidence.** Each reported g is recalculated from the row's own raw data as PPC2, change-score and endpoint g to see which it matches. Rows that report only g get the design shown by the matching rows, after you confirm. Rows that match nothing are flagged and can be set row by row.
- **New "best available design" measure for mixed data.** It uses PPC2 where baseline data allow, otherwise change-score or endpoint g, and records the design of every study.
- **PPC2 without a reported pre-post r.** You choose an imputed r (0.3/0.5/0.7, Morris variance) or the simple SMD variance approximation. The r-grid sensitivity analysis is then recommended.
- **New sensitivity analyses**: raw-data only (no reported-g rows), most-common design only, and subgroup by effect-size design.
- **Shared controls**: splitting the control N now changes only the variance (weight), never the effect estimate (Cochrane 23.3.4).
- **PRISMA**: the form is open in Step 4, and the diagram is left out (with a warning) if no numbers are entered.

## Changes in 4.0.1

- **Pre/post data**: Morris PPC2 is recommended when studies report baseline and post data with pre-post correlations.
- **New columns understood**:
  - `Biomarker`/`Analyte` are read as the outcome name.
  - `Duration` + `Duration_Unit` become a time point and a numeric "Follow-up (weeks)" moderator.
  - `Outcome_Direction` sets the effect direction; you can change it.
  - `Unit` is checked for consistency, and raw MD is blocked when units differ.
  - `Study_Design` triggers a warning for crossover or cluster trials.
  - `Control_ID` is used to detect shared controls.
- **Several time points per study**: you choose latest, earliest, or all.
- **Moderators**: offered only when they vary within an outcome. Meta-regression needs at least 6 studies; funnel plots need at least 10.
- **App**:
  - Prepare all plans at once, with a progress table.
  - The include list starts with everything ticked.
  - You can't approve a plan with fewer than 2 comparisons, and plans that aren't approved are named when you run the analysis.
- **Report**:
  - The methods text describes only the formulas actually used.
  - New table of comparisons not included, with reasons.
  - New summary forest plot across outcomes.
  - Forest plot layout fixed for small k; wide tables go on landscape pages.

## Limits

- **Not included:** diagnostic test accuracy, IPD, crossover and cluster corrections, component NMA, arm-based or Bayesian NMA, multivariate models for several correlated outcomes.
- **Network meta-analysis assumes consistency and transitivity.** The inconsistency tests have low power; check that the studies are similar enough to be compared.
- **Internal checks only.** MetaPy checks the workbook's internal consistency; it cannot confirm that values were extracted correctly from the papers.
- **Judgement stays with you.** Approximations are transparent but remain approximations. Use the sensitivity analyses, and leave GRADE and risk-of-bias judgements to the review team.
- **Validate before publishing.** Check key results against an independent implementation before publication.

`legacy_v3/` contains the earlier v3.3.1 code for reference; MetaPy 4 does not import it.

## Author, license and citation

MetaPy is written by Sadia Nadeem ([ORCID 0009-0008-5510-9456](https://orcid.org/0009-0008-5510-9456)) and released under the MIT License (see `LICENSE`). To cite it, see `CITATION.cff`.
