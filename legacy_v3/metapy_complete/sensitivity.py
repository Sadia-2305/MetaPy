from __future__ import annotations

import pandas as pd

from .meta_analysis import pool_effects


def pool_correlation_sensitivity(
    sensitivity_records: pd.DataFrame,
    *,
    model: str = "random",
    tau_method: str = "REML",
    ci_method: str = "Hartung-Knapp",
) -> pd.DataFrame:
    """Pool each explicitly approved assumed-correlation scenario separately."""
    if sensitivity_records is None or sensitivity_records.empty:
        return pd.DataFrame()
    rows = []
    for scenario, part in sensitivity_records.groupby("assumed_correlation_scenario", dropna=False, sort=True):
        if "sensitivity_pool_eligible" in part.columns:
            eligible = part[part["sensitivity_pool_eligible"].fillna(False).astype(bool)]
        else:
            eligible = part.iloc[0:0]
        for (outcome, duration), grp in eligible.groupby(["outcome", "duration"], dropna=False, sort=False):
            pooled = pool_effects(grp, model=model, tau_method=tau_method, ci_method=ci_method)
            rows.append({
                "assumed_correlation_scenario": scenario,
                "outcome": outcome,
                "duration": duration,
                "assumption_scope": "Missing group correlation(s) only; available source correlations retained",
                "warning": "Sensitivity analysis only. Assumed correlations are not source data and do not replace the primary analysis.",
                **pooled,
            })
    return pd.DataFrame(rows)
