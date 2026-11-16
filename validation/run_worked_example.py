"""Analyse the synthetic mixed-reporting worked example exactly as described in the paper (results/worked_example.json)."""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
from metapy import MetaPySession  # noqa: E402

s = MetaPySession()
s.load(str(HERE / "data" / "MetaPy_worked_example_mixed.xlsx"))
pid = "Anxiety :: g_best"
s.select_plans([pid])
p = s.prepare(pid)
q = next(q for q in p.questions if q["id"] == "g_type_all")
endpoint = next(v for lab, v in q["options"] if "endpoint" in lab)          # researcher: reported g values are endpoint g
s.answer(pid, {"g_type_all": endpoint, "ppc2_variance": 0.5, "higher_is_better": "worse_keep"})
s.approve_many([pid])
ex = ["subgroup::Effect-size design", "design_only", "r_grid", "tau_ci", "loo", "egger"]
r = s.analyse({"model": "random", "tau_method": "REML", "ci_method": "HK", "extras": ex})[pid]
sg, rg = r.extras["subgroup::Effect-size design"], r.extras["r_grid"]
out = {"k": len(r.studies), "designs": r.studies["Effect-size design"].value_counts().to_dict(),
       "main": {k: r.main[k] for k in ("estimate", "ci_lo", "ci_hi", "tau2", "I2", "pi_lo", "pi_hi")},
       "subgroup": sg["table"][["subgroup", "k", "estimate", "ci_lo", "ci_hi"]].to_dict("records"),
       "Q_between": sg["Q_between"], "p_between": sg["p_between"],
       "design_only": {k: r.extras["design_only"]["result"][k] for k in ("estimate", "ci_lo", "ci_hi")},
       "log_rows": len(s.preps[pid].log), "counts": s.preps[pid].counts, "rgrid": rg["table"].to_dict("records")}
json.dump(out, open(HERE / "results" / "worked_example.json", "w"), indent=1, default=float)
print(json.dumps(out["main"], default=float))
