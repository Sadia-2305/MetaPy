"""Effect measures MetaPy can pool, and the fields each one needs."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Measure:
    key: str
    label: str
    family: str
    est: str
    var: str
    back: str | None = None        # back-transformation for reporting: exp | tanh | expit
    scale: str = ""
    description: str = ""
    citation: str = ""
    short: str = "Estimate"

    def transform(self, x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return x
        if self.back == "exp":
            return math.exp(x)
        if self.back == "tanh":
            return math.tanh(x)
        if self.back == "expit":
            return 1 / (1 + math.exp(-x))
        return x

    @property
    def null(self) -> float:
        return 1.0 if self.back == "exp" else 0.0

    @property
    def reported_label(self) -> str:
        lab = {"exp": self.label.replace("log ", "").replace("Log ", ""), "tanh": "Correlation r",
               "expit": "Proportion"}.get(self.back, self.label)
        return lab[0].upper() + lab[1:]


SHORT = {"g_best": "g", "smd_g": "g", "smd_d": "d", "md": "MD", "smd_change_g": "g (change)", "md_change": "MD (change)", "ppc2": "g (PPC2)",
         "log_or": "OR", "log_rr": "RR", "rd": "RD", "log_hr": "HR", "z_r": "r", "logit_prop": "Proportion", "generic": "Estimate"}

MEASURES: dict[str, Measure] = {m.key: m for m in [
    Measure("smd_g", "Hedges' g (endpoint SMD)", "continuous", "g_post", "var_g_post", scale="SMD",
            description="Standardised mean difference between groups at follow-up, small-sample corrected.",
            citation="Hedges 1981; Borenstein et al. 2009"),
    Measure("g_best", "Hedges' g, best available design per study (mixed data)", "continuous", "g_best", "var_g_best", scale="SMD",
            description="Uses PPC2 where baseline data allow, otherwise change-score or endpoint g; the design of each study is shown and can be "
                        "tested as a subgroup.", citation="Morris 2008; Cochrane Handbook 10.5.2"),
    Measure("smd_d", "Cohen's d (endpoint SMD)", "continuous", "d_post", "var_d_post", scale="SMD",
            description="Standardised mean difference without small-sample correction.", citation="Borenstein et al. 2009"),
    Measure("md", "Mean difference (endpoint)", "continuous", "md_post", "md_var_post", scale="Raw units",
            description="Raw difference in means; all studies must use the same scale/unit.", citation="Cochrane Handbook 10.5.1"),
    Measure("smd_change_g", "Hedges' g of change scores", "continuous", "g_change", "var_g_change", scale="SMD",
            description="SMD of change from baseline; needs change SDs or pre-post correlations.", citation="Cochrane Handbook 6.5.2.8"),
    Measure("md_change", "Mean difference of change scores", "continuous", "md_change", "md_var_change", scale="Raw units",
            description="Raw difference in change from baseline.", citation="Cochrane Handbook 10.5.2"),
    Measure("ppc2", "Morris PPC2 (pre-post-control g)", "continuous", "g_ppc2", "var_g_ppc2", scale="SMD",
            description="Difference in pre-post change standardised by pooled baseline SD.", citation="Morris 2008"),
    Measure("log_or", "log odds ratio", "binary", "log_or", "var_log_or", back="exp", scale="Odds ratio",
            description="Odds ratio for a dichotomous outcome (pooled on the log scale).", citation="Borenstein et al. 2009, ch. 5"),
    Measure("log_rr", "log risk ratio", "binary", "log_rr", "var_log_rr", back="exp", scale="Risk ratio",
            description="Risk ratio for a dichotomous outcome (pooled on the log scale).", citation="Borenstein et al. 2009, ch. 5"),
    Measure("rd", "Risk difference", "binary", "rd", "var_rd", scale="Risk difference",
            description="Absolute difference in event risk.", citation="Borenstein et al. 2009, ch. 5"),
    Measure("log_hr", "log hazard ratio", "time-to-event", "log_hr", "var_log_hr", back="exp", scale="Hazard ratio",
            description="Generic inverse-variance pooling of reported hazard ratios.", citation="Tierney et al. 2007"),
    Measure("z_r", "Fisher z (correlation)", "correlation", "z_r", "var_z_r", back="tanh", scale="Correlation r",
            description="Correlation coefficients pooled on Fisher's z scale.", citation="Borenstein et al. 2009, ch. 6"),
    Measure("logit_prop", "logit proportion", "proportion", "logit_prop", "var_logit_prop", back="expit", scale="Proportion",
            description="Single-group proportion (prevalence/incidence) pooled on the logit scale.", citation="Lipsey & Wilson 2001"),
    Measure("generic", "Generic inverse-variance", "generic", "effect", "var_effect", scale="As reported",
            description="Any pre-calculated effect with its SE or CI (all studies must report the same measure).",
            citation="Cochrane Handbook 10.3"),
]}
MEASURES = {k: Measure(**{**m.__dict__, "short": SHORT[k]}) for k, m in MEASURES.items()}
