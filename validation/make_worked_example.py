"""Build the synthetic mixed-reporting worked example (validation/data/MetaPy_worked_example_mixed.xlsx).

34 comparisons of one outcome reported in four formats: 10 with pre/post means and SDs, 10 with endpoint means and SDs,
8 with only a mean difference and its 95% CI (text cell), and 6 with only a reported Hedges' g and its SE.
Run run_derivation_validation.py first: it creates the three simulated source files used here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / "data"
A = pd.read_csv(D / "sim_prepost.csv").head(10)
B = pd.read_csv(D / "sim_endpoint.csv").head(10)
C = pd.read_csv(D / "sim_md_ci.csv").head(8)
rng = np.random.default_rng(7)
G = pd.DataFrame([{"Study": f"G{k + 1}", "N_I": int(n1), "N_C": int(n2), "Hedges_g": round(g, 3),
                   "SE": round(np.sqrt(1 / n1 + 1 / n2 + g * g / (2 * (n1 + n2))), 3)}
                  for k, (n1, n2, g) in enumerate(zip(rng.integers(20, 90, 6), rng.integers(20, 90, 6), rng.normal(-0.35, 0.2, 6)))])
mixed = pd.concat([A, B, C, G], ignore_index=True)
mixed.insert(1, "Outcome", "Anxiety")
mixed.to_excel(D / "MetaPy_worked_example_mixed.xlsx", index=False)
print("written", D / "MetaPy_worked_example_mixed.xlsx", len(mixed), "rows")
