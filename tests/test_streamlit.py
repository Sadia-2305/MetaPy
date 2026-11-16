"""The Streamlit website, run end to end with a stand-in for Streamlit (no browser needed)."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["METAPY_FEEDBACK"] = '{"action": ""}'          # never send real feedback from tests
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
        self.assertFalse(any("Combine different scales" in str(p) for k, p in st.LOG if k == "widget"))  # file chosen, not read
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
        reports = [(n, b) for n, b in st.DOWNLOADS if n.startswith("MetaPy_report")]
        self.assertEqual(sorted(n for n, _ in reports), ["MetaPy_report.docx", "MetaPy_report.xlsx"])
        self.assertTrue(all(len(b) > 1000 for _, b in reports))
        self.assertIn(("dialog", "How was MetaPy?"), st.LOG)                      # feedback pop-up after the first export
        self.assertTrue(any(n == "MetaPy_data_template.xlsx" for n, _ in st.DOWNLOADS))
        ss["fb_stars"] = 4
        st.LOG.clear()
        go("btn_fb_send")                                                         # not configured here: nothing is sent
        self.assertTrue(ss["fb_done"])
        st.LOG.clear()
        go()
        self.assertNotIn(("dialog", "How was MetaPy?"), st.LOG)                   # asked only once
        go("btn_example")                                     # one-click example data
        self.assertTrue(ss["s"].ds.source_name.startswith("MetaPy_example_tutoring"))
        s = ss["s"]
        old, old_gen = s, ss["gen"]
        go("btn_read")                                        # reading a file again starts a brand-new session
        self.assertIsNot(ss["s"], old)
        self.assertEqual(ss["gen"], old_gen + 1)
        self.assertEqual(ss["exports"], [])
        self.assertEqual(ss["s"].results, {})


    def test_bad_files_are_explained_and_combine_option_only_when_relevant(self):
        import tempfile
        import pandas as pd
        import fake_streamlit as F
        st = F.install()
        app, ss = str(ROOT / "streamlit_app.py"), st.session_state
        tmp = Path(tempfile.mkdtemp())

        def go(*clicks):
            st.CLICKS.update(clicks)
            F.run(app, st)

        # 1) a file that is not a spreadsheet: explained, no raw error, no crash
        (tmp / "broken.xlsx").write_bytes(b"not a spreadsheet")
        ss["uploader"] = _Up(str(tmp / "broken.xlsx"))
        go("btn_read")
        self.assertIsNone(ss["s"])
        warn = " ".join(str(p) for k, p in st.LOG if k == "warning")
        self.assertIn("could not open", warn)
        self.assertEqual([p for k, p in st.LOG if k == "error"], [])
        # 2) means and SDs without sample sizes: says exactly what is missing
        pd.DataFrame({"Study": [1, 2, 3], "Mean_Post_I": [1, 2, 3], "SD_Post_I": [1, 1, 1], "Mean_Post_C": [2, 3, 4],
                      "SD_Post_C": [1, 1, 1]}).to_csv(tmp / "no_n.csv", index=False)
        ss["uploader"] = _Up(str(tmp / "no_n.csv"))
        go("btn_read")
        text = " ".join(str(p) for k, p in st.LOG if k in ("markdown", "warning"))
        self.assertIn("no sample sizes", text)
        # 3) an internal failure is reported as MetaPy's problem, with the technical detail tucked away
        from metapy import session as S
        orig = S.MetaPySession.load
        S.MetaPySession.load = lambda self, *a, **k: (_ for _ in ()).throw(TypeError("operation 'radd' not supported"))
        try:
            ss["uploader"] = _Up(str(ROOT / "examples" / "MetaPy_SYNTHETIC_demo.xlsx"))
            go("btn_read")
        finally:
            S.MetaPySession.load = orig
        warn = " ".join(str(p) for k, p in st.LOG if k == "warning")
        self.assertIn("problem in MetaPy, not in your data", warn)
        self.assertTrue(any("radd" in str(p) for k, p in st.LOG if k == "code"))
        # 4) 'combine scales' appears only when the file has several scales of one construct
        go("btn_read")                                            # demo file: one scale per outcome
        self.assertFalse(any("Combine different scales" in str(p) for k, p in st.LOG if k == "widget"))
        rows = []
        for i, scale in enumerate(["HADS-A", "GAD-7", "HADS-A", "BAI"], start=1):
            rows.append({"Study": i, "Outcome": "Anxiety", "Scale": scale, "Mean_Post_I": 8 + i, "SD_Post_I": 3, "N_I": 30,
                         "Mean_Post_C": 10 + i, "SD_Post_C": 3, "N_C": 30})
        pd.DataFrame(rows).to_csv(tmp / "scales.csv", index=False)
        ss["uploader"] = _Up(str(tmp / "scales.csv"))
        go("btn_read")
        combine = [p for k, p in st.LOG if k == "widget" and "Combine different scales" in str(p)]
        self.assertEqual(len(combine), 1)
        self.assertIn("→ Anxiety", combine[0])
        gen = ss["gen"]
        st.change("group_cb", False)                                # untick: file is read again without combining
        go()
        self.assertEqual(ss["gen"], gen + 1)
        self.assertTrue(any("Combine different scales" in str(p) for k, p in st.LOG if k == "widget"))   # still offered


if __name__ == "__main__":
    unittest.main()
