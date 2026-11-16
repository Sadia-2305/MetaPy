"""MetaPy must keep agreeing with R (metafor 5.2-1, clubSandwich 0.7.0, netmeta 3.7.0, bayesmeta 3.5).

The R values were produced by validation/r_check/metafor_check.R and are stored in validation/results/metafor_results.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "r_check"))

# largest acceptable relative difference per block (most agree far more closely; see r_comparison_summary.json)
TOL = {"bayes": 5e-3, "qprofile": 1e-3, "metareg_interaction": 1e-4, "predict": 2e-4, "tau2": 2e-4, "glmm_or": 1e-4,
       "glmm_plo": 1e-4, "robust_cr2": 1e-4, "selection": 1e-4}


def test_agreement_with_r():
    import compare_with_r as C
    r = json.loads((ROOT / "validation" / "results" / "metafor_results.json").read_text())
    t = C.compare(C.metapy_side(), r)
    assert len(t) >= 240
    for block, g in t.groupby("block"):
        assert g["rel_diff"].max() <= TOL.get(block, 1e-4), (block, g.sort_values("rel_diff").tail(1).to_dict("records"))
