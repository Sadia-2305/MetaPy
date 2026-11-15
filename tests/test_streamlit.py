"""The Streamlit website, run end to end with a stand-in for Streamlit (no browser needed)."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


class _Up:
    def __init__(self, path):
        self.name, self._b = os.path.basename(path), open(path, "rb").read()

    def getvalue(self):
        return self._b


class TestStreamlitSite(unittest.TestCase):
    def test_full_workflow_and_clean_slate(self):
        import fake_streamlit as F
        st = F.install()
        app, ss = str(ROOT / "streamlit_app.py"), st.session_state

        def go(*clicks):
            st.CLICKS.update(clicks)
            F.run(app, st)
            self.assertEqual([p for k, p in st.LOG if k == "error"], [])

        go()
        widgets = [p for k, p in st.LOG if k == "widget"]
        self.assertFalse(any("Combine different scales" in str(w) for w in widgets))     # not shown before a file is chosen
        ss["uploader"] = _Up(str(ROOT / "examples" / "MetaPy_SYNTHETIC_demo.xlsx"))
        go()
        self.assertTrue(any("Combine different scales" in str(p) for k, p in st.LOG if k == "widget"))
        go("btn_read")
        K = lambda n: f"{n}__{ss['gen']}"
        s = ss["s"]
        rec = sorted(p["plan_id"] for p in s.scan_report.plans if p["recommended"])
        ticked = sorted(p["plan_id"] for p in s.scan_report.plans if ss.get(K(f"plan_cb::{p['plan_id']}")))
        self.assertEqual(ticked, rec)                                                    # recommended plans pre-ticked
        go(K("btn_select"))
        self.assertEqual(sorted(s.plans), rec)
        # a changed choice takes effect immediately (no 'Apply' button)
        pid = ss[K("plan_dd")]
        qs = s.preps[pid].questions
        if qs:
            q = next(q for q in qs if not q.get("multi"))
            st.change(K(f"q::{pid}::{q['id']}::{len(q['options'])}"), 0)
            self.assertIn(pid, s.answered)
        go()
        ss[K("approve_group")], ss[K("approve_confirm")] = list(s.plans), True
        go(K("btn_approve"))
        self.assertTrue(all(s.approved.values()))
        self.assertTrue(set(s.plans) <= s.answered)                                      # pre-selected answers applied on approval
        keys = [a["key"] for a in s.extra_choices()]
        ss[K(f"extras_{len(keys)}")] = [k for k in ("tau_ci", "multilevel") if k in keys]
        go(K("btn_run"))
        self.assertEqual(len(s.results), len(s.plans))
        ss[K("p_on")], ss[K("p_box_identified")] = True, "Databases (n = 10)"
        go(K("btn_fill"))
        self.assertIn("studies", ss[K("p_box_included")])
        ss[K("fmts")] = ["docx", "xlsx"]
        go(K("btn_export"))
        names = [n for n, _ in st.DOWNLOADS]
        self.assertEqual(sorted(names), ["MetaPy_report.docx", "MetaPy_report.xlsx"])
        self.assertTrue(all(len(b) > 1000 for _, b in st.DOWNLOADS))
        old, old_gen = s, ss["gen"]
        go("btn_read")                                        # reading a file again starts a brand-new session
        self.assertIsNot(ss["s"], old)
        self.assertEqual(ss["gen"], old_gen + 1)
        self.assertEqual(ss["exports"], [])
        self.assertEqual(ss["s"].results, {})


if __name__ == "__main__":
    unittest.main()
