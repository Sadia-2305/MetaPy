"""Write docs/DERIVATION_RULES.md from the live registry (so documentation never drifts from code)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metapy.derive.registry import RULES, GROUP_INFO
from metapy.fields import label

out = ["# MetaPy derivation rules", "", f"{len(RULES)} rules. Generated from `metapy/derive/registry.py`.", "",
       "**exact**: algebraic identity or standard formula; applied automatically and logged.  ",
       "**approximation**: published estimator; applied automatically, logged, and listed for the researcher to accept or veto.  ",
       "**assumption**: needs a researcher decision; only applied when the researcher switches it on.", "",
       "## Researcher-controllable groups", "", "| Group | What it does | Source |", "|---|---|---|"]
out += [f"| `{g}` | {a} | {b} |" for g, (a, b) in GROUP_INFO.items()]
out += ["", "## All rules", "", "| Rule | Type | Needs | Gives | Formula | Citation |", "|---|---|---|---|---|---|"]
seen = set()
for r in RULES:
    key = (r.name.split(" (")[0], r.kind)
    out.append(f"| {r.name} | {r.kind} | {', '.join(label(i) for i in r.inputs)} | {', '.join(label(o) for o in r.outputs)} | `{r.formula}` | {r.citation} |")
Path(__file__).resolve().parents[1].joinpath("docs", "DERIVATION_RULES.md").write_text("\n".join(out) + "\n")
print(len(RULES), "rules documented")
