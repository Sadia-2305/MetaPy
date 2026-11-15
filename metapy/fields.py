"""Canonical field vocabulary and smart column-name recognition for MetaPy.

Every statistic MetaPy understands has one canonical field name. Column headers
in uploaded files are normalised (lower case, separators removed) and matched
against a large, programmatically generated alias table, so headers such as
``Mean_Post_I``, ``post mean (treatment)``, ``SD_baseline_ctrl`` or ``N1`` are all
recognised without the researcher renaming anything.
"""
from __future__ import annotations

import itertools
import re
from typing import Any

ARMS = {"i": "intervention", "c": "control"}
TIMES = {"post": "post-test / endpoint", "pre": "baseline", "change": "change from baseline"}
GROUP_STATS = ["mean", "sd", "se", "ci_lo", "ci_hi", "median", "q1", "q3", "min", "max"]


def norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


# ---------------------------------------------------------------- labels
LABELS: dict[str, str] = {
    "study_id": "Study ID", "outcome": "Outcome", "timepoint": "Time point", "year": "Year",
    "author": "Author", "intervention_arm": "Intervention arm", "control_arm": "Control arm",
    "n_i": "N (intervention)", "n_c": "N (control)", "n_total": "Total N",
    "r_prepost_i": "Pre-post correlation (intervention)", "r_prepost_c": "Pre-post correlation (control)",
    "md_post": "Mean difference (endpoint)", "md_change": "Mean difference (change)",
    "md_se_post": "SE of mean difference (endpoint)", "md_se_change": "SE of mean difference (change)",
    "md_ci_lo_post": "MD 95% CI lower (endpoint)", "md_ci_hi_post": "MD 95% CI upper (endpoint)",
    "md_ci_lo_change": "MD 95% CI lower (change)", "md_ci_hi_change": "MD 95% CI upper (change)",
    "sdp_post": "Pooled SD (endpoint)", "sdp_change": "Pooled SD (change)",
    "t_post": "t statistic (endpoint)", "t_change": "t statistic (change)",
    "tabs_post": "|t| (endpoint)", "tabs_change": "|t| (change)",
    "p_post": "p value (endpoint)", "p_change": "p value (change)",
    "f_post": "F statistic (endpoint)", "f_change": "F statistic (change)",
    "d_post": "Cohen's d (endpoint)", "d_change": "Cohen's d (change)",
    "g_post": "Hedges' g (endpoint)", "g_change": "Hedges' g (change)",
    "var_d_post": "Variance of d (endpoint)", "var_d_change": "Variance of d (change)",
    "g_reported": "Hedges' g as reported (design not stated)", "se_g_reported": "SE of reported Hedges' g",
    "g_reported_ci_lo": "Reported g 95% CI lower", "g_reported_ci_hi": "Reported g 95% CI upper",
    "se_g_ppc2": "SE of Morris PPC2 g", "g_ppc2_ci_lo": "PPC2 g 95% CI lower", "g_ppc2_ci_hi": "PPC2 g 95% CI upper",
    "g_best": "Hedges' g (best available design)", "var_g_best": "Variance of best-available g", "instrument": "Scale / instrument",
    "var_g_post": "Variance of g (endpoint)", "var_g_change": "Variance of g (change)",
    "se_d_post": "SE of d (endpoint)", "se_g_post": "SE of g (endpoint)",
    "se_d_change": "SE of d (change)", "se_g_change": "SE of g (change)",
    "g_ci_lo_post": "g 95% CI lower (endpoint)", "g_ci_hi_post": "g 95% CI upper (endpoint)",
    "d_ci_lo_post": "d 95% CI lower (endpoint)", "d_ci_hi_post": "d 95% CI upper (endpoint)",
    "g_ci_lo_change": "g 95% CI lower (change)", "g_ci_hi_change": "g 95% CI upper (change)",
    "d_ci_lo_change": "d 95% CI lower (change)", "d_ci_hi_change": "d 95% CI upper (change)",
    "smd_unspecified": "SMD (type not stated)", "smd_unspecified_se": "SE of SMD (type not stated)",
    "direction_sign": "Direction (+1 intervention higher, -1 lower)",
    "g_ppc2": "Morris PPC2 g", "var_g_ppc2": "Variance of Morris PPC2 g",
    "events_i": "Events (intervention)", "events_c": "Events (control)",
    "pct_i": "Event % (intervention)", "pct_c": "Event % (control)",
    "log_or": "log odds ratio", "var_log_or": "Variance of log OR", "or": "Odds ratio",
    "or_ci_lo": "OR 95% CI lower", "or_ci_hi": "OR 95% CI upper", "se_log_or": "SE of log OR",
    "log_rr": "log risk ratio", "var_log_rr": "Variance of log RR", "rr": "Risk ratio",
    "rr_ci_lo": "RR 95% CI lower", "rr_ci_hi": "RR 95% CI upper", "se_log_rr": "SE of log RR",
    "rd": "Risk difference", "var_rd": "Variance of RD",
    "log_hr": "log hazard ratio", "var_log_hr": "Variance of log HR", "hr": "Hazard ratio",
    "hr_ci_lo": "HR 95% CI lower", "hr_ci_hi": "HR 95% CI upper", "se_log_hr": "SE of log HR",
    "p_or": "p value for OR", "p_rr": "p value for RR", "p_hr": "p value for HR",
    "r": "Correlation r", "r_spearman": "Spearman rho", "z_r": "Fisher z", "var_z_r": "Variance of Fisher z",
    "events": "Events (single group)", "prop": "Proportion (single group)",
    "logit_prop": "Logit proportion", "var_logit_prop": "Variance of logit proportion",
    "effect": "Generic effect estimate", "effect_se": "SE of generic effect",
    "effect_ci_lo": "Generic effect 95% CI lower", "effect_ci_hi": "Generic effect 95% CI upper",
    "var_effect": "Variance of generic effect",
}
for a, aname in ARMS.items():
    for t, tname in TIMES.items():
        for s in GROUP_STATS:
            LABELS[f"{s}_{t}_{a}"] = f"{s.replace('_', ' ').upper() if s in ('sd', 'se') else s.replace('_', ' ').capitalize()} {tname} ({aname})"


def label(field: str) -> str:
    return LABELS.get(field, field)


# ---------------------------------------------------------------- aliases
ARM_TOKENS = {
    "i": ["i", "int", "intervention", "treatment", "treat", "trt", "tx", "exp", "experimental", "e", "t", "1", "g1", "group1", "active", "case", "cases"],
    "c": ["c", "ctrl", "control", "comparator", "comparison", "placebo", "cont", "2", "g2", "group2", "usualcare", "uc", "pbo"],
}
TIME_TOKENS = {
    "post": ["post", "posttest", "followup", "fu", "end", "endpoint", "final", "after", "t2", "outcome"],
    "pre": ["pre", "pretest", "baseline", "bl", "before", "t1", "base"],
    "change": ["change", "chg", "diff", "delta", "difference", "gain", "cfb"],
}
STAT_TOKENS = {
    "mean": ["mean", "m", "avg", "average", "mn"],
    "sd": ["sd", "std", "stdev", "stddev", "standarddeviation", "s"],
    "se": ["se", "sem", "standarderror", "stderr"],
    "ci_lo": ["cilo", "cilow", "cilower", "lci", "lowerci", "lower95ci", "ll", "lower"],
    "ci_hi": ["cihi", "cihigh", "ciupper", "uci", "upperci", "upper95ci", "ul", "upper"],
    "median": ["median", "med", "mdn"],
    "q1": ["q1", "p25", "iqrlo", "iqrlower", "lowerquartile", "25th"],
    "q3": ["q3", "p75", "iqrhi", "iqrupper", "upperquartile", "75th"],
    "min": ["min", "minimum", "rangelo", "rangelow"],
    "max": ["max", "maximum", "rangehi", "rangehigh"],
}
TEXT_STATS = {  # combined text cells such as "12.3 (4.1)" or "10 (8-14)"
    "mean_sd": ["meansd", "meanandsd", "msd"],
    "median_iqr": ["medianiqr", "mediqr", "medianq1q3"],
    "median_range": ["medianrange", "medianminmax"],
}

STATIC_ALIASES: dict[str, list[str]] = {
    "study_id": ["studyid", "study", "studyname", "trialid", "trial", "id", "studyno", "reference", "citation", "studylabel"],
    "author": ["author", "authors", "firstauthor", "authoryear"],
    "year": ["year", "pubyear", "publicationyear", "yr"],
    "instrument": ["scale", "instrument", "questionnaire", "tool", "measurementtool", "outcomescale", "anxietyscale",
                   "depressionscale", "stressscale", "sleepscale", "painscale", "qolscale", "assessmenttool", "measure", "assay"],
    "outcome": ["outcome", "outcomename", "endpointname", "variable", "outcomemeasure", "construct", "domain",
                "biomarker", "marker", "analyte", "parameter", "indicator", "outcomevariable", "construct"],
    "timepoint_unit": ["durationunit", "timeunit", "timepointunit", "followupunit", "unitoftime"],
    "control_id": ["controlid", "sharedcontrolid", "comparatorid", "controlgroupid"],
    "outcome_direction": ["outcomedirection", "higherisbetter", "higherisworse", "directionofbenefit", "polarity",
                          "scoredirection", "desirabledirection", "higherbetter", "higherworse"],
    "unit": ["unit", "units", "measurementunit", "outcomeunit", "unitofmeasure"],
    "study_design": ["studydesign", "design", "trialdesign", "allocationdesign"],
    "timepoint": ["timepoint", "time", "duration", "followupweeks", "weeks", "followupduration", "assessmenttime"],
    "intervention_arm": ["interventionarm", "intervention", "treatmentarm", "arm", "experimentalarm", "interventionname", "treatment"],
    "control_arm": ["controlarm", "comparator", "controltype", "comparatorarm", "controlgroup"],
    "n_total": ["ntotal", "n", "totaln", "samplesize", "totalsamplesize", "nall", "participants"],
    "r_prepost_i": ["rprepost_i", "rprepostint", "rpreposti", "correlationi", "prepostcorrelationi", "ri", "rint", "rtreatment"],
    "r_prepost_c": ["rprepostc", "correlationc", "prepostcorrelationc", "rc", "rctrl", "rcontrol"],
    "md_post": ["md", "meandifference", "meandiff", "rawmd", "mdpost", "mdendpoint", "wmd"],
    "md_se_post": ["mdse", "semd", "semeandifference", "mdsepost"],
    "md_ci_lo_post": ["mdcilo", "mdlci", "mdlower", "mdcilower", "mdll"],
    "md_ci_hi_post": ["mdcihi", "mduci", "mdupper", "mdciupper", "mdul"],
    "md_change": ["mdchange", "changemd", "meandifferencechange", "mdchg"],
    "md_se_change": ["mdsechange", "semdchange"],
    "t_post": ["t", "tstat", "tstatistic", "tvalue", "ttest", "tpost"],
    "t_change": ["tchange", "tstatchange"],
    "p_post": ["p", "pvalue", "pval", "psig", "ppost", "significance"],
    "p_change": ["pchange", "pvaluechange"],
    "f_post": ["f", "fstat", "fstatistic", "fvalue", "anovaf"],
    "f_change": ["fchange"],
    "d_post": ["d", "cohend", "cohensd", "dpost", "cohendvalue"],
    "d_change": ["dchange", "cohendchange", "cohensdchange"],
    "se_d_post": ["sed", "dse", "secohend", "cohendse"],
    "var_d_post": ["vard", "dvar", "variancecohend"],
    # A Hedges' g whose design (endpoint / change / pre-post-control) is not stated in the header:
    "g_reported": ["g", "hedgesg", "hedgesgvalue", "hedgeg", "smdhedgesg", "reportedhedgesg", "hedgessg", "hedgesgsmd", "smdg"],
    "se_g_reported": ["seg", "gse", "sehedgesg", "hedgesgse"],
    "g_reported_ci_lo": ["gcilo", "glci", "glower", "gcilower", "hedgesgcilower", "hedgesglower"],
    "g_reported_ci_hi": ["gcihi", "guci", "gupper", "gciupper", "hedgesgciupper", "hedgesgupper"],
    "g_post": ["gpost", "gendpoint", "hedgesgpost", "hedgesgendpoint", "posttestg", "gposttest"],
    "g_ppc2": ["gppc2", "ppc2", "hedgesgppc2", "morrisg", "dppc2", "morrisppc2"],
    "g_change": ["gchange", "hedgesgchange"],
    "se_g_post": ["segpost", "gsepost"],
    "var_g_post": ["vargpost"],
    "d_ci_lo_post": ["dcilo", "dlci", "dlower", "dcilower"],
    "d_ci_hi_post": ["dcihi", "duci", "dupper", "dciupper"],
    "smd_unspecified": ["smd", "standardizedmeandifference", "standardisedmeandifference", "effectsizesmd", "es"],
    "smd_unspecified_se": ["smdse", "sesmd"],
    "direction_sign": ["directionsign", "direction", "favours", "favors", "higherarm"],
    "events_i": ["eventsi", "eventi", "eventsint", "eventsintervention", "eventstreatment", "eventstrt", "eventsexp", "ai", "casesi"],
    "events_c": ["eventsc", "eventc", "eventsctrl", "eventscontrol", "eventscomparator", "eventsplacebo", "ci_events", "casesc"],
    "pct_i": ["pcti", "percenti", "percentintervention", "eventratei", "riski", "proportioni", "pctint"],
    "pct_c": ["pctc", "percentc", "percentcontrol", "eventratec", "riskc", "proportionc", "pctctrl"],
    "or": ["or", "oddsratio", "reportedor"],
    "or_ci_lo": ["orcilo", "orlci", "orlower", "orcilower", "orll"],
    "or_ci_hi": ["orcihi", "oruci", "orupper", "orciupper", "orul"],
    "log_or": ["logor", "lnor"], "se_log_or": ["selogor", "selnor", "logorse"],
    "p_or": ["por", "pvalueor"],
    "rr": ["rr", "riskratio", "relativerisk"],
    "rr_ci_lo": ["rrcilo", "rrlci", "rrlower", "rrcilower"],
    "rr_ci_hi": ["rrcihi", "rruci", "rrupper", "rrciupper"],
    "log_rr": ["logrr", "lnrr"], "se_log_rr": ["selogrr", "selnrr"],
    "p_rr": ["prr"],
    "hr": ["hr", "hazardratio"],
    "hr_ci_lo": ["hrcilo", "hrlci", "hrlower", "hrcilower"],
    "hr_ci_hi": ["hrcihi", "hruci", "hrupper", "hrciupper"],
    "log_hr": ["loghr", "lnhr"], "se_log_hr": ["seloghr", "selnhr"],
    "p_hr": ["phr"],
    "rd": ["rd", "riskdifference"],
    "r": ["r", "pearsonr", "correlation", "corr", "pearson"],
    "r_spearman": ["rho", "spearman", "spearmanrho", "rs", "spearmanr"],
    "events": ["events", "cases", "nevents", "numberofevents", "positive"],
    "prop": ["prop", "proportion", "prevalence", "rate", "percent", "pct", "incidence"],
    "effect": ["effect", "estimate", "effectestimate", "yi", "effectsize"],
    "effect_se": ["effectse", "seeffect", "se", "standarderror", "sei"],
    "var_effect": ["vareffect", "vi", "variance", "samplingvariance"],
    "effect_ci_lo": ["effectcilo", "cilo", "lowerci", "lci", "cilower", "lower95"],
    "effect_ci_hi": ["effectcihi", "cihi", "upperci", "uci", "ciupper", "upper95"],
    "effect_type": ["effecttype", "effectmeasure", "measuretype", "estimand", "effectmethod", "metric"],
    "ci_text": ["ci", "95ci", "confidenceinterval", "ci95"],
    "md_ci_text": ["mdci", "md95ci", "meandifference95ci", "mdwithci"],
    "g_ci_text": ["gci", "hedgesgci", "g95ci"],
    "d_ci_text": ["dci", "cohendci", "d95ci"],
    "or_ci_text": ["orci", "or95ci", "oddsratio95ci"],
    "rr_ci_text": ["rrci", "rr95ci", "riskratio95ci"],
    "hr_ci_text": ["hrci", "hr95ci", "hazardratio95ci"],
}


def _build_alias_table() -> dict[str, str]:
    table: dict[str, str] = {}

    def add(alias: str, field: str):
        key = norm(alias)
        if key and key not in table:
            table[key] = field

    # group-level statistics: every order of stat / time / arm tokens
    for arm, arm_toks in ARM_TOKENS.items():
        for time, time_toks in TIME_TOKENS.items():
            t_opts = time_toks + ([""] if time == "post" else [])
            for stat, stat_toks in list(STAT_TOKENS.items()) + [(k, v) for k, v in TEXT_STATS.items()]:
                field = f"{stat}_{time}_{arm}" if stat in STAT_TOKENS else f"text_{stat}_{time}_{arm}"
                for s, t, a in itertools.product(stat_toks, t_opts, arm_toks):
                    for order in itertools.permutations([s, t, a]):
                        add("".join(order), field)
        for tok in arm_toks:
            for ntok in ("n", "size", "samplesize", "total", "participants", "nrandomised", "nrandomized", "nanalysed", "nanalyzed"):
                add(ntok + tok, f"n_{arm}")
                add(tok + ntok, f"n_{arm}")
            for etok in ("events", "event", "cases"):
                add(etok + tok, f"events_{arm}")
                add(tok + etok, f"events_{arm}")
            for ctok in ("r", "rprepost", "correlation", "prepostr", "prepostcorrelation", "rpp"):
                add(ctok + tok, f"r_prepost_{arm}")
                add(tok + ctok, f"r_prepost_{arm}")
    for field, aliases in STATIC_ALIASES.items():
        for alias in aliases:
            add(alias, field)
    return table


ALIAS_TABLE = _build_alias_table()
# Static aliases override generated ones where they collide (e.g. "n").
for _field, _aliases in STATIC_ALIASES.items():
    for _alias in _aliases:
        ALIAS_TABLE[norm(_alias)] = _field
# Bare intervention/control N is far more common than the generated ambiguities
ALIAS_TABLE.update({"n1": "n_i", "n2": "n_c", "ni": "n_i", "nc": "n_c", "ne": "n_i", "nt": "n_i"})

IDENTITY_FIELDS = ["study_id", "author", "year", "outcome", "instrument", "timepoint", "intervention_arm", "control_arm", "effect_type",
                   "timepoint_unit", "control_id", "outcome_direction", "unit", "study_design"]
TEXT_FIELDS = {"study_id", "author", "outcome", "instrument", "timepoint", "intervention_arm", "control_arm", "effect_type",
               "timepoint_unit", "control_id", "outcome_direction", "unit", "study_design",
               "ci_text", "md_ci_text", "g_ci_text", "d_ci_text", "or_ci_text", "rr_ci_text", "hr_ci_text"}


def match_column(header: Any) -> str | None:
    """Return the canonical field for a column header, or None."""
    key = norm(str(header).replace("\u2019", "").replace("'", ""))
    if not key:
        return None
    if key in ALIAS_TABLE:
        return ALIAS_TABLE[key]
    # tolerate common decorations: units in brackets, "95%" etc.
    stripped = re.sub(r"(95|percent|value|reported|extracted)", "", key)
    if stripped in ALIAS_TABLE:
        return ALIAS_TABLE[stripped]
    if key.endswith("scale") and len(key) > 5:      # e.g. "Anxiety_Scale", "Depression scale used"
        return "instrument"
    return None


# ---------------------------------------------------------------- constructs and synonyms
_INSTRUMENTS = {
    "Anxiety": ["hama", "hars", "hamiltonanxiety", "hadsa", "hadsanxiety", "bai", "beckanxiety", "gad7", "gad2", "dassa",
                "dassanxiety", "dass21a", "dass21anxiety", "stai", "stais", "stait", "staiy", "staiy1", "staiy2", "pswq", "zsas", "sas",
                "masq", "scared", "rcmas", "gaq", "oasis", "hadsanx", "dassanx"],
    "Depression": ["hamd", "hdrs", "hrsd", "hamiltondepression", "bdi", "bdiii", "bdi2", "phq9", "phq8", "hadsd", "hadsdepression",
                   "cesd", "madrs", "dassd", "dassdepression", "dass21d", "gds", "gds15", "epds", "zsds", "sds", "qids", "cdi", "hadsdep", "dassdep"],
    "Stress": ["pss", "pss10", "pss14", "dasss", "dassstress", "dass21s", "dass21stress", "dassstr"],
    "Sleep quality": ["psqi", "isi", "ais", "ess"],
    "Quality of life": ["sf36", "sf12", "whoqol", "whoqolbref", "eq5d", "qol"],
    "Pain": ["vas", "nrs", "bpi", "mpq", "sfmpq"],
}
INSTRUMENT_CONSTRUCT = {k: c for c, ks in _INSTRUMENTS.items() for k in ks}

_SYNONYMS = {
    "CRP": ["crp", "hscrp", "hscrpmgl", "creactiveprotein", "highsensitivitycrp", "highsensitivitycreactiveprotein", "hcrp", "ultrasensitivecrp"],
    "IL-6": ["il6", "interleukin6", "il06"], "IL-1\u03b2": ["il1b", "il1beta", "interleukin1beta", "il1\u03b2"],
    "IL-10": ["il10", "interleukin10"], "IL-8": ["il8", "interleukin8"], "IL-17": ["il17", "interleukin17"],
    "TNF-\u03b1": ["tnf", "tnfa", "tnfalpha", "tumornecrosisfactoralpha", "tumournecrosisfactoralpha", "tnf\u03b1"],
    "IFN-\u03b3": ["ifng", "ifngamma", "interferongamma"], "Cortisol": ["cortisol", "salivarycortisol", "serumcortisol", "plasmacortisol", "haircortisol"],
    "BDNF": ["bdnf"], "HbA1c": ["hba1c", "glycatedhaemoglobin", "glycatedhemoglobin"],
}
SYNONYM = {k: c for c, ks in _SYNONYMS.items() for k in ks}


def concept_key(text: Any) -> str:
    t = str(text).replace("\u03b1", "alpha").replace("\u03b2", "beta").replace("\u03b3", "gamma")
    return norm(t)


def canonical_outcome(text: Any) -> str:
    """'hsCRP' -> 'CRP', 'TNF-a' -> 'TNF-alpha'; otherwise the text unchanged."""
    return SYNONYM.get(concept_key(text), str(text).strip())


def construct_of(instrument: Any) -> str | None:
    return INSTRUMENT_CONSTRUCT.get(concept_key(instrument))


def split_instrument_arm(text: Any) -> tuple[str, str | None]:
    """'HADS-A Meditation' -> ('HADS-A', 'Meditation') when the first part is a known instrument."""
    t = str(text).strip()
    parts = re.split(r"\s+", t)
    for i in range(len(parts) - 1, 0, -1):
        head, tail = " ".join(parts[:i]), " ".join(parts[i:])
        if construct_of(head) or concept_key(head) in SYNONYM:
            return head, tail
    return t, None



# Higher score = worse for participants (used only as a pre-selected default; the researcher confirms)
HIGHER_IS_WORSE = {"Anxiety", "Depression", "Stress", "Pain", "Sleep quality", "CRP", "IL-6", "IL-1\u03b2", "IL-8", "IL-17",
                   "TNF-\u03b1", "IFN-\u03b3", "Cortisol", "HbA1c"}
HIGHER_IS_BETTER = {"Quality of life", "BDNF"}


def known_direction(outcome: str) -> bool | None:
    """True = higher is better, False = higher is worse, None = unknown."""
    if outcome in HIGHER_IS_WORSE:
        return False
    if outcome in HIGHER_IS_BETTER:
        return True
    return None
