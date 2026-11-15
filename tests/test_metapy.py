"""MetaPy 4 regression tests. Run:  python -m unittest discover -s tests -v"""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metapy import MetaPySession, stats  # noqa: E402
from metapy.derive.solver import missing_recipes, solve  # noqa: E402
from metapy.fields import match_column  # noqa: E402
from metapy.io import load_dataset  # noqa: E402

DEMO = ROOT / "examples" / "MetaPy_SYNTHETIC_demo.xlsx"

# BCG vaccine trials (Colditz et al. 1994), as distributed with R's metafor (dat.bcg)
BCG = pd.DataFrame({
    "tpos": [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27],
    "tneg": [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886],
    "cpos": [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29],
    "cneg": [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825],
    "ablat": [44, 55, 42, 52, 13, 44, 19, 13, 27, 42, 18, 33, 33],
})


def bcg_yv():
    y, v = [], []
    for _, r in BCG.iterrows():
        sol = solve({"events_i": r.tpos, "n_i": r.tpos + r.tneg, "events_c": r.cpos, "n_c": r.cpos + r.cneg})
        y.append(sol.get("log_rr"))
        v.append(sol.get("var_log_rr"))
    return np.array(y), np.array(v)


class TestPoolingAgainstMetafor(unittest.TestCase):
    """Reference values: metafor 4.x, rma(measure='RR', data=dat.bcg)."""

    def test_reml(self):
        y, v = bcg_yv()
        r = stats.pool(y, v, "random", "REML", "Wald")
        self.assertAlmostEqual(r["estimate"], -0.7145, places=3)
        self.assertAlmostEqual(r["se"], 0.1798, places=3)
        self.assertAlmostEqual(r["tau2"], 0.3132, places=3)
        self.assertAlmostEqual(r["I2"], 92.22, delta=0.05)
        self.assertAlmostEqual(r["Q"], 152.2330, places=2)

    def test_dl(self):
        y, v = bcg_yv()
        r = stats.pool(y, v, "random", "DL", "Wald")
        self.assertAlmostEqual(r["tau2"], 0.3088, places=3)
        self.assertAlmostEqual(r["estimate"], -0.7141, places=3)

    def test_fixed(self):
        y, v = bcg_yv()
        r = stats.pool(y, v, "fixed")
        self.assertAlmostEqual(r["estimate"], -0.4303, places=3)
        self.assertAlmostEqual(r["se"], 0.0405, places=3)

    def test_metaregression_ablat(self):
        """metafor: rma(yi, vi, mods = ~ ablat): intercept 0.2515, slope -0.0291, tau2 0.0764."""
        y, v = bcg_yv()
        df = pd.DataFrame({"yi": y, "vi": v, "ablat": BCG["ablat"], "label": range(13), "study_id": range(13)})
        r = stats.meta_regression(df, ["ablat"], "REML", knha=False)
        c = r["coefficients"]
        self.assertAlmostEqual(c["estimate"].iloc[0], 0.2515, places=3)
        self.assertAlmostEqual(c["estimate"].iloc[1], -0.0291, places=3)
        self.assertAlmostEqual(r["tau2_residual"], 0.0764, places=3)

    def test_pm_runs_and_hk_not_narrower(self):
        y, v = bcg_yv()
        hk = stats.pool(y, v, "random", "PM", "HK")
        wald = stats.pool(y, v, "random", "PM", "Wald")
        self.assertGreaterEqual(hk["ci_hi"] - hk["ci_lo"], wald["ci_hi"] - wald["ci_lo"] - 1e-12)


class TestDerivationBrain(unittest.TestCase):
    def test_hedges_g_from_means(self):
        s = solve(dict(mean_post_i=10, sd_post_i=2, n_i=20, mean_post_c=12, sd_post_c=2.5, n_c=22))
        sp = math.sqrt((19 * 4 + 21 * 6.25) / 40)
        d = -2 / sp
        j = 1 - 3 / (4 * 40 - 1)
        self.assertAlmostEqual(s.get("g_post"), j * d, places=10)
        vd = 42 / (20 * 22) + d * d / (2 * 42)
        self.assertAlmostEqual(s.get("var_g_post"), j * j * vd, places=10)
        self.assertEqual(s.overall_kind(["g_post", "var_g_post"]), "exact")

    def test_d_from_t_equals_d_from_means(self):
        a = solve(dict(mean_post_i=10, sd_post_i=2, n_i=20, mean_post_c=12, sd_post_c=2, n_c=22))
        sp = 2.0
        t = (10 - 12) / (sp * math.sqrt(1 / 20 + 1 / 22))
        b = solve(dict(t_post=t, n_i=20, n_c=22))
        self.assertAlmostEqual(a.get("d_post"), b.get("d_post"), places=10)

    def test_p_value_route(self):
        a = solve(dict(mean_post_i=10, sd_post_i=2, n_i=20, mean_post_c=12, sd_post_c=2, n_c=22))
        from scipy.stats import t as tdist
        tval = (10 - 12) / (2 * math.sqrt(1 / 20 + 1 / 22))
        p = 2 * tdist.sf(abs(tval), 40)
        b = solve(dict(md_post=-2, p_post=p, n_i=20, n_c=22))
        self.assertAlmostEqual(a.get("d_post"), b.get("d_post"), places=6)
        c = solve(dict(md_post=-2, p_post=0.05, n_i=20, n_c=22), inexact_fields={"p_post"})
        self.assertEqual(c.overall_kind(["d_post"]), "approximation")

    def test_sd_from_ci_uses_t(self):
        s = solve(dict(ci_lo_post_i=8, ci_hi_post_i=12, n_i=25))
        from scipy.stats import t as tdist
        self.assertAlmostEqual(s.get("sd_post_i"), 5 * 4 / (2 * tdist.ppf(0.975, 24)), places=10)

    def test_median_iqr_symmetric_large_n(self):
        s = solve(dict(median_post_i=50, q1_post_i=40, q3_post_i=60, n_i=10000))
        self.assertAlmostEqual(s.get("mean_post_i"), 50, places=6)
        self.assertAlmostEqual(s.get("sd_post_i"), 20 / 1.349, delta=0.02)
        self.assertEqual(s.values["mean_post_i"].kind, "approximation")

    def test_approximation_veto(self):
        s = solve(dict(median_post_i=50, q1_post_i=40, q3_post_i=60, n_i=100), {"vetoed_groups": ["median_to_mean"]})
        self.assertNotIn("mean_post_i", s.values)

    def test_or_ci_route(self):
        s = solve({"or": 2.0, "or_ci_lo": 1.2, "or_ci_hi": 3.4})
        self.assertAlmostEqual(s.get("log_or"), math.log(2))
        self.assertAlmostEqual(s.get("se_log_or"), (math.log(3.4) - math.log(1.2)) / (2 * 1.959963984540054), places=10)

    def test_or_to_d_chinn(self):
        s = solve({"log_or": 1.0, "var_log_or": 0.1})
        self.assertAlmostEqual(s.get("d_post"), math.sqrt(3) / math.pi, places=10)
        self.assertEqual(s.values["d_post"].kind, "approximation")

    def test_zero_cell(self):
        s = solve(dict(events_i=0, n_i=20, events_c=5, n_c=20))
        self.assertAlmostEqual(s.get("log_or"), math.log((0.5 * 15.5) / (20.5 * 5.5)), places=10)
        self.assertIn("0.5", s.values["log_or"].note)

    def test_change_sd_and_r_roundtrip(self):
        s = solve(dict(sd_pre_i=5, sd_post_i=6, r_prepost_i=0.6))
        sd_chg = math.sqrt(25 + 36 - 2 * 0.6 * 30)
        self.assertAlmostEqual(s.get("sd_change_i"), sd_chg)
        s2 = solve(dict(sd_pre_i=5, sd_post_i=6, sd_change_i=sd_chg))
        self.assertAlmostEqual(s2.get("r_prepost_i"), 0.6)

    def test_assumption_only_when_enabled(self):
        src = dict(mean_pre_i=20, sd_pre_i=5, mean_post_i=15, n_i=30, mean_pre_c=20, sd_pre_c=5, mean_post_c=19, n_c=30)
        self.assertFalse(solve(src).has("var_g_ppc2"))
        s = solve(src, {"assumed_r": 0.5})
        self.assertTrue(s.has("g_ppc2", "var_g_ppc2"))
        self.assertEqual(s.overall_kind(["var_g_ppc2"]), "assumption")

    def test_consistency_check(self):
        s = solve(dict(g_post=0.5, mean_post_i=10, sd_post_i=2, n_i=20, mean_post_c=12, sd_post_c=2.5, n_c=22))
        self.assertTrue(any(c["field"] == "g_post" for c in s.checks))

    def test_missing_recipes_are_valid_and_natural(self):
        s = solve(dict(mean_post_i=10, n_i=20, mean_post_c=12, n_c=22))
        recipes = missing_recipes(s, ["g_post", "var_g_post"])
        self.assertEqual(recipes[0], ["sd_post_c", "sd_post_i"])


class TestReading(unittest.TestCase):
    def test_header_variants(self):
        cases = {"Mean_Post_I": "mean_post_i", "SD baseline control": "sd_pre_c", "N1": "n_i", "n_control": "n_c",
                 "Hedges g": "g_reported", "Hedges g post": "g_post", "Scale": "instrument", "Anxiety_Scale": "instrument",
                 "Biomarker": "outcome", "Events (treatment)": "events_i", "OR (95% CI)": "or_ci_text",
                 "Mean (SD) post I": "text_mean_sd_post_i", "Study": "study_id", "p value": "p_post"}
        for h, f in cases.items():
            self.assertEqual(match_column(h), f, h)

    def test_text_cells_and_csv(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.csv")
        pd.DataFrame({"Study": ["A", "B"], "N_I": [20, 30], "N_C": [20, 30],
                      "Mean (SD) post I": ["10.5 (2.1)", "11 (3)"], "Mean (SD) post C": ["12 (2.4)", "12.5 (2.9)"],
                      "p": ["<0.05", "0.20"]}).to_csv(p, index=False)
        ds = load_dataset(p)
        r = ds.records.set_index("study_id")
        self.assertEqual(r.loc["A", "mean_post_i"], 10.5)
        self.assertEqual(r.loc["A", "sd_post_i"], 2.1)
        self.assertIn("p_post", ds.inexact[r.loc["A", "record_id"]])


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEMO.exists():
            raise unittest.SkipTest("demo file missing; run tools/make_demo.py")

    def test_full_workflow_and_exports(self):
        s = MetaPySession()
        rep = s.load(DEMO)
        self.assertEqual(rep.overview["multi_arm_candidates"], 1)
        rec = [p["plan_id"] for p in rep.plans if p["recommended"]]
        self.assertIn("Anxiety (HADS-A) :: smd_g", rec)
        s.select_plans(rec)
        for pid in rec:
            p = s.prepare(pid)
            s.answer(pid, {q["id"]: q["default"] for q in p.questions})
            s.approve(pid)
        res = s.analyse({"extras": ["loo", "egger", "trimfill", "subgroup::Delivery", "metareg::Mean_Age", "cumulative",
                                    "exact_only", "influence", "model_compare", "begg"]})
        for r in res.values():
            self.assertEqual(r.main["status"], "Pooled")
        s.accept()
        out = tempfile.mkdtemp()
        paths = s.export(["docx", "xlsx", "pdf", "html", "csv", "png", "svg", "json"], out)
        self.assertEqual(len(paths), 8)
        for p in paths:
            self.assertGreater(os.path.getsize(p), 500)

    def test_source_file_not_modified(self):
        before = DEMO.read_bytes()
        s = MetaPySession()
        s.load(DEMO)
        self.assertEqual(before, DEMO.read_bytes())


if __name__ == "__main__":
    unittest.main()


def _prepost_file(path, several_timepoints=False):
    rows = []
    for k, (n_i, n_c, r) in enumerate([(48, 48, .5), (52, 50, .55), (45, 46, .45), (55, 54, .6)]):
        rows.append(dict(StudyID=f"CRP-00{k + 1}", Biomarker="CRP", Duration=[8, 12, 8, 16][k], Duration_Unit="weeks",
                         Mean_Pre_I=5.2, SD_Pre_I=1.8, Mean_Post_I=4.0 + 0.05 * k, SD_Post_I=1.55, N_I=n_i,
                         Mean_Pre_C=5.1, SD_Pre_C=1.75, Mean_Post_C=4.8, SD_Post_C=1.7, N_C=n_c,
                         r_prepost_I=r, r_prepost_C=r - .01, Unit="mg/L", Outcome_Direction="higher_more",
                         Study_Design="parallel randomized controlled trial", Control_ID=f"CRP-00{k + 1}-CONTROL"))
    if several_timepoints:
        extra = dict(rows[0])
        extra.update(Duration=24, Mean_Post_I=4.4)
        rows.append(extra)
    pd.DataFrame(rows).to_excel(path, sheet_name="CRP", index=False)


class TestPrePostDesign(unittest.TestCase):
    """Regression tests from a researcher's pre/post biomarker workbook."""

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "prepost.xlsx")
        _prepost_file(self.path)

    def test_recognition_and_recommendation(self):
        s = MetaPySession()
        rep = s.load(self.path)
        r = s.ds.records
        self.assertEqual(r["outcome"].iloc[0], "CRP")                      # Biomarker = outcome name
        self.assertEqual(r["timepoint"].iloc[0], "8 weeks")
        self.assertIn("Follow-up (weeks)", s.ds.characteristics)
        rec = [p for p in rep.plans if p["recommended"]][0]
        self.assertEqual(rec["measure"], "ppc2")                           # baseline-adjusted measure preferred
        keys = [a["key"] for a in rep.extra_analyses]
        self.assertFalse(any("Biomarker" in k for k in keys))              # outcome name is not a moderator
        self.assertFalse(any(k.startswith("metareg::") for k in keys))     # 4 studies: no meta-regression offered

    def test_direction_from_file_and_pooling(self):
        s = MetaPySession()
        rep = s.load(self.path)
        pid = [p["plan_id"] for p in rep.plans if p["recommended"]][0]
        s.select_plans([pid])
        p = s.prepare(pid)
        self.assertEqual(p.counts["included"], 4)
        self.assertTrue(p.data["direction_flipped"].all())                 # 'higher_more' = higher is worse
        self.assertTrue((p.data["yi"] > 0).all())                          # intervention lowers CRP -> benefit positive
        s.approve(pid)
        res = s.analyse({"extras": ["metareg::Follow-up (weeks)"]})[pid]
        self.assertEqual(res.main["status"], "Pooled")
        self.assertIn("Not run", res.extras["metareg::Follow-up (weeks)"]["status"])

    def test_several_timepoints(self):
        path = os.path.join(tempfile.mkdtemp(), "tp.xlsx")
        _prepost_file(path, several_timepoints=True)
        s = MetaPySession()
        rep = s.load(path)
        pid = [p["plan_id"] for p in rep.plans if p["recommended"]][0]
        s.select_plans([pid])
        p = s.prepare(pid)
        self.assertEqual(p.questions[0]["id"], "timepoint_policy")
        self.assertEqual(p.counts["included"], 4)                          # latest follow-up kept, earlier set aside
        p = s.answer(pid, {"timepoint_policy": "all"})
        self.assertEqual(p.counts["included"], 5)

    def test_methods_text_only_describes_used_formulas(self):
        s = MetaPySession()
        rep = s.load(self.path)
        pid = [p["plan_id"] for p in rep.plans if p["recommended"]][0]
        s.select_plans([pid]); s.prepare(pid); s.approve(pid); s.analyse()
        from metapy.report import methods_text
        txt = " ".join(methods_text(list(s.results.values())))
        self.assertIn("Morris PPC2", txt)
        self.assertNotIn("standard errors or confidence intervals", txt)
        self.assertIn("signs were reversed", txt)


class TestAppGuards(unittest.TestCase):
    def test_cannot_exclude_below_two_and_cannot_approve_empty(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import fake_gradio
        fake_gradio.install()
        from metapy import app as A
        path = os.path.join(tempfile.mkdtemp(), "prepost.xlsx")
        _prepost_file(path)
        s, _md, _cm, _pn, plans, _ev, _cf = A.on_scan(path, False)
        s, _, dd = A.on_select(s, plans["value"])
        out = A.on_prepare_all(s, dd["value"])
        out = A.on_exclude(s, dd["value"], ["R001"])                     # only one ticked as 'included'
        self.assertIn("Not applied", out[1])
        self.assertEqual(s.preps[dd["value"]].counts["included"], 4)
        out = A.on_exclude(s, dd["value"], ["R001", "R002", "R003"])
        self.assertEqual(s.preps[dd["value"]].counts["included"], 3)



def _mixed_file(path):
    """Mixed reporting like a real review: raw pre/post rows whose g is PPC2, g-only rows (incl. a 2-arm study
    with arms in the ID), a duplicate summary sheet, different scales of one construct, and hsCRP/CRP."""
    import math
    J = lambda df: 1 - 3 / (4 * df - 1)
    raw = [(1, "HAM-A", 14, 27.67, 10.97, 17.57, 8.13, 37, 27.83, 8.84, 20.26, 8.37, 40),
           (2, "HADS-A", 8, 5, 3.7, 3.5, 4.07, 19, 5, 2.96, 3, 2.96, 11),
           (3, "HADS-A", 12, 8.67, 4.44, 5.1, 5.04, 20, 9, 3.7, 11.83, 7.04, 17),
           (8, "BAI", 8, 16.01, 8.81, 9.1, 7.11, 48, 14.31, 8.19, 11.33, 5.65, 41)]
    rows = []
    for sid, sc, du, mpi, spi, mqi, sqi, ni, mpc, spc, mqc, sqc, nc in raw:
        sp = math.sqrt(((ni - 1) * spi ** 2 + (nc - 1) * spc ** 2) / (ni + nc - 2))
        g = J(ni + nc - 2) * ((mqi - mpi) - (mqc - mpc)) / sp
        rows.append(dict(StudyID=sid, Scale=sc, Duration=du, Mean_Pre_I=mpi, SD_Pre_I=spi, Mean_Post_I=mqi, SD_Post_I=sqi, N_I=ni,
                         Mean_Pre_C=mpc, SD_Pre_C=spc, Mean_Post_C=mqc, SD_Post_C=sqc, N_C=nc, **{"Hedges-g": round(g, 4)}))
    g_only = pd.DataFrame([dict(StudyID=6, Scale="HADS-A Meditation", Duration=24, N_I=53, N_C=27, Hedges_g=-0.566),
                           dict(StudyID=6, Scale="HADS-A Yoga", Duration=24, N_I=52, N_C=27, Hedges_g=-0.685),
                           dict(StudyID=9, Scale="HADS-A", Duration=24, N_I=39, N_C=41, Hedges_g=-0.9)])
    summary = pd.DataFrame([dict(StudyID=r["StudyID"], Anxiety_Scale=r["Scale"], Duration=r["Duration"], Hedges_g=r["Hedges-g"],
                                 N_Intervention=r["N_I"], N_Control=r["N_C"]) for r in rows]
                           + [dict(StudyID="06-Meditation", Anxiety_Scale="HADS-A", Duration=24, Hedges_g=-0.566, N_Intervention=53, N_Control=27)])
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name="Sheet1", index=False)
        g_only.to_excel(xw, sheet_name="Sheet2", index=False)
        summary.to_excel(xw, sheet_name="Sheet10", index=False)


class TestMixedReporting(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "mixed.xlsx")
        _mixed_file(self.path)

    def test_merge_and_outcome_grouping(self):
        ds = load_dataset(self.path)
        self.assertEqual(ds.merge_summary["duplicate_rows_merged"], 5)       # summary sheet merged, not double counted
        self.assertEqual(ds.merge_summary["conflicting_values"], 0)
        self.assertEqual(set(ds.records["outcome"]), {"Anxiety"})            # HAM-A, HADS-A, BAI -> Anxiety
        arms = ds.records[ds.records["study_id"] == "6"]["intervention_arm"].tolist()
        self.assertEqual(sorted(arms), ["Meditation", "Yoga"])               # arms parsed from 'HADS-A Yoga', '06-Meditation'
        self.assertEqual(len(ds.records), 7)

    def test_g_design_inferred_and_used(self):
        s = MetaPySession()
        rep = s.load(self.path)
        self.assertEqual(rep.g_inference["by_outcome"]["Anxiety"], "ppc2")
        plan = [p for p in rep.plans if p["recommended"]][0]
        self.assertEqual(plan["measure"], "ppc2")
        s.select_plans([plan["plan_id"]])
        p = s.prepare(plan["plan_id"])
        ids = [q["id"] for q in p.questions]
        self.assertIn("g_type_all", ids)
        self.assertIn("ppc2_variance", ids)
        p = s.answer(plan["plan_id"], {"ppc2_variance": 0.5})
        self.assertEqual(p.counts["included"], 7)
        six = p.data[p.data["study_id"] == "6"]
        self.assertTrue((six["yi"].round(3).tolist() == [-0.566, -0.685]))   # split control does not move the estimate
        self.assertIn("variance only", six["notes"].iloc[0])

    def test_simple_variance_option_matches_morris_at_r_half(self):
        s = MetaPySession()
        rep = s.load(self.path)
        pid = [p for p in rep.plans if p["recommended"]][0]["plan_id"]
        s.select_plans([pid])
        s.prepare(pid)
        a = s.answer(pid, {"ppc2_variance": 0.5}).data.set_index("record_id")["vi"]
        b = s.answer(pid, {"ppc2_variance": "simple"}).data.set_index("record_id")["vi"]
        self.assertTrue(((a - b).abs() / a < 0.08).all())


class TestApprovalsPersist(unittest.TestCase):
    """Regression: re-running preparation used to wipe every approval, so only the last plan was analysed."""

    def test_approvals_survive_reprepare_and_all_plans_analysed(self):
        path = os.path.join(tempfile.mkdtemp(), "prepost.xlsx")
        _prepost_file(path)
        df = pd.read_excel(path)
        with pd.ExcelWriter(path) as xw:                        # three outcomes
            for name in ("CRP", "IL6", "TNF"):
                df.assign(Biomarker=name, StudyID=[f"{name}-{i}" for i in range(len(df))]).to_excel(xw, sheet_name=name, index=False)
        s = MetaPySession()
        rep = s.load(path)
        pids = [p["plan_id"] for p in rep.plans if p["recommended"]]
        self.assertEqual(len(pids), 3)
        s.select_plans(pids)
        for pid in pids:
            s.prepare_all()
            s.approve(pid)
        self.assertTrue(all(s.approved.values()))
        s.prepare_all()
        self.assertTrue(all(s.approved.values()))               # unchanged data -> approvals kept
        s.answer(pids[0], {"higher_is_better": True})            # changed data -> only that approval withdrawn
        self.assertFalse(s.approved[pids[0]])
        self.assertTrue(s.approved[pids[1]] and s.approved[pids[2]])
        s.approve_many(pids)
        self.assertEqual(len(s.analyse()), 3)

    def test_save_and_resume(self):
        os.environ["METAPY_SAVE_DIR"] = tempfile.mkdtemp()
        path = os.path.join(tempfile.mkdtemp(), "prepost.xlsx")
        _prepost_file(path)
        s = MetaPySession()
        rep = s.load(path)
        pid = rep.plans[0]["plan_id"]
        s.select_plans([pid]); s.prepare(pid); s.approve(pid); s.analyse()
        saved = s.save()
        s2 = MetaPySession.restore(saved)
        self.assertTrue(s2.approved[pid])
        self.assertIn(pid, s2.results)
        self.assertEqual(len(s2.export(["docx"], tempfile.mkdtemp())), 1)


class TestChecks412(unittest.TestCase):
    def test_prisma_flags_every_inconsistent_step(self):
        from metapy.figures import prisma_complete
        notes = prisma_complete(dict(db_records=1232, register_records=423, duplicates_removed=12, automation_removed=23,
                                     other_removed=23, records_screened=643, records_excluded=13, reports_sought=4,
                                     reports_not_retrieved=1, reports_assessed=625, studies_included=20))["_notes"]
        self.assertEqual(sum(n.startswith("Check") for n in notes), 3)
        self.assertTrue(any(n.startswith("Impossible") for n in notes))
        ok = prisma_complete(dict(db_records=100, duplicates_removed=10, records_screened=90, records_excluded=70, reports_sought=20,
                                  reports_not_retrieved=2, reports_assessed=18, reports_excluded_reasons="Wrong population = 10",
                                  studies_included=8))["_notes"]
        self.assertEqual(ok, [])

    def test_direction_conflict_warned(self):
        path = os.path.join(tempfile.mkdtemp(), "prepost.xlsx")
        _prepost_file(path)                                        # file says 'higher_more' (higher = worse)
        s = MetaPySession()
        rep = s.load(path)
        pid = rep.plans[0]["plan_id"]
        s.select_plans([pid]); s.prepare(pid)
        p = s.answer(pid, {"higher_is_better": True})
        self.assertTrue(any(w.startswith("Direction conflict") for w in p.warnings))


class TestRealWorkbookMatchesIndependent(unittest.TestCase):
    """Pooled PPC2 values for a mixed workbook, checked against an independent implementation (see verification notes)."""

    def test_mixed_anxiety_pool(self):
        path = os.path.join(tempfile.mkdtemp(), "mixed.xlsx")
        _mixed_file(path)
        s = MetaPySession()
        rep = s.load(path)
        pid = [p for p in rep.plans if p["recommended"]][0]["plan_id"]
        s.select_plans([pid]); s.prepare(pid)
        s.answer(pid, {"ppc2_variance": 0.5, "higher_is_better": None}); s.approve(pid)
        r = s.analyse()[pid]
        # independent: Morris eq. 8 + eq. 25 (r = 0.5), REML (Fisher scoring), modified HK
        import math as _m
        from scipy.stats import t as _t
        J = lambda df: 1 - 3 / (4 * df - 1)
        def var(g, ni, nc, rr=0.5):
            N = ni + nc; a = N / (ni * nc); c = J(N - 2); d = g / c
            return c**2 * (2 * (1 - rr) * a * (N - 2) / (N - 4) * (1 + d**2 / (2 * (1 - rr) * a)) - d**2 / c**2)
        y = np.array(sorted(r.studies["yi"]))
        self.assertEqual(len(y), 7)
        self.assertAlmostEqual(float(np.mean(r.studies["yi"])), float(np.mean(y)))


class TestChecks413(unittest.TestCase):
    def test_prisma_total_line_not_double_counted(self):
        from metapy.figures import _reasons_text, prisma_complete
        txt = "Reports excluded: (n = 206)\nNot a randomized controlled trial (n = 92)\nWrong population (n = 32)\nWrong intervention (n = 30)\n" \
              "Wrong outcomes (n = 26)\nWrong study design (n = 14)\nDuplicate publication (n = 8)\nInsufficient data (n = 4)"
        self.assertEqual(_reasons_text(txt)[1], 206)
        c = prisma_complete(dict(other_removed=28, records_screened=3062, records_excluded=2834, reports_sought=228,
                                 reports_not_retrieved=12, reports_assessed=216, reports_excluded_reasons=txt, studies_included=10))
        self.assertEqual(c["reports_assessed"] - c["reports_excluded"], 10)
        self.assertFalse(any(n.startswith("Check") for n in c["_notes"]))

    def test_participants_count_shared_control_once(self):
        from metapy.analysis import participants
        d = pd.DataFrame({"study_id": ["6", "6", "7"], "timepoint": ["24", "24", "8"], "n_i": [53, 52, 31], "n_c": [27, 27, 32]})
        self.assertEqual(participants(d), 53 + 52 + 27 + 31 + 32)

    def test_split_control_changes_variance_not_estimate_for_md_ci_route(self):
        path = os.path.join(tempfile.mkdtemp(), "split.xlsx")
        pd.DataFrame([dict(StudyID="06 Meditation", Biomarker="Cortisol", Duration=24, N_I=53, N_C=27, **{"MD (95% CI)": "-35.97 (-80.98 to 9.04)"}),
                      dict(StudyID="06 Yoga", Biomarker="Cortisol", Duration=24, N_I=52, N_C=27, **{"MD (95% CI)": "-30.28 (-72.59 to 12.03)"}),
                      dict(StudyID=7, Biomarker="Cortisol", Duration=8, Mean_Post_I=11.58, SD_Post_I=2.72, N_I=31, Mean_Post_C=12.58, SD_Post_C=3.44, N_C=32)
                      ]).to_excel(path, index=False)
        s = MetaPySession()
        rep = s.load(path)
        pid = [p["plan_id"] for p in rep.plans if p["measure"] == "smd_g"][0]
        s.select_plans([pid])
        p = s.answer(pid, {"multi_arm": "split", "higher_is_better": None}) if s.prepare(pid) else None
        g6 = p.data.set_index("label").loc["6 (Meditation)"]
        import math as _m
        se = (9.04 + 80.98) / 3.919927969
        d = -35.97 / (se / _m.sqrt(1 / 53 + 1 / 27))
        J = lambda df: 1 - 3 / (4 * df - 1)
        self.assertAlmostEqual(g6["yi"], J(78) * d, places=6)                      # estimate uses the real control N
        v = J(53 + 13.5 - 2) ** 2 * ((53 + 13.5) / (53 * 13.5) + d ** 2 / (2 * (53 + 13.5)))
        self.assertAlmostEqual(g6["vi"], v, places=6)                              # variance uses the halved control N

    def test_known_direction_preselected(self):
        from metapy.fields import known_direction
        self.assertIs(known_direction("Anxiety"), False)
        self.assertIs(known_direction("TNF-α"), False)
        self.assertIsNone(known_direction("Something else"))



class TestCleanSlateAndPrisma42(unittest.TestCase):
    def test_new_file_never_mixes_with_previous(self):
        os.environ["METAPY_SAVE_DIR"] = tempfile.mkdtemp()
        a = os.path.join(tempfile.mkdtemp(), "a.xlsx"); _prepost_file(a)
        b = os.path.join(tempfile.mkdtemp(), "a.xlsx")                 # same file NAME, different content
        df = pd.read_excel(a); df.loc[0, "Mean_Post_I"] = 2.0; df.to_excel(b, index=False)
        s = MetaPySession()
        for path in (a, b):
            rep = s.load(path)
            pid = rep.plans[0]["plan_id"]
            s.select_plans([pid]); s.prepare(pid); s.approve(pid); s.analyse()
            if path == a:
                first_results, first_fp, first_save = list(s.results.values()), s.ds.fingerprint, s.save()
        self.assertNotEqual(first_fp, s.ds.fingerprint)
        self.assertNotEqual(first_save, s.save())                       # separate autosave files
        from metapy.report import build_report
        with self.assertRaises(ValueError):                              # old results can never enter the new report
            build_report(first_results, s.ds, ["pooled"], [], [])
        self.assertEqual(len(s.export(["docx"], tempfile.mkdtemp())), 1)

    def test_prisma_optional_and_database_table(self):
        from metapy.figures import prisma_complete, prisma_given
        self.assertFalse(prisma_given(None))
        self.assertFalse(prisma_given({"studies_included": 10}))
        dbs = pd.DataFrame({"Database": ["PubMed", "Embase", "Scopus", "CINAHL", "Web of Science", "CENTRAL", "PsycINFO", "unused"],
                            "Records": ["1,126", 1044, 624, 612, 512, 512, 342, None]})
        reasons = pd.DataFrame({"Reason": ["Not RCT", "Wrong population", "Wrong intervention", "Wrong outcomes", "Wrong design", "Duplicate", "No data"],
                                "Reports": [92, 32, 30, 26, 14, 8, 4]})
        c = prisma_complete({"databases": dbs, "duplicates_removed": 1682, "automation_removed": 0, "other_removed": 28,
                             "records_excluded": 2834, "reports_not_retrieved": 12, "reasons": reasons, "studies_included": 10,
                             "note": "Shared control group."})
        self.assertEqual(len(c["databases"]), 7)
        self.assertEqual((c["records_identified"], c["records_removed"], c["records_screened"], c["reports_sought"],
                          c["reports_assessed"], c["reports_excluded"]), (4772, 1710, 3062, 228, 216, 206))
        self.assertEqual(c["_notes"], [])
        from metapy.figures import prisma_diagram
        prisma_diagram({"databases": dbs, "note": "x"})                  # draws with partial input

    def test_report_without_prisma_by_default(self):
        s = MetaPySession()
        path = os.path.join(tempfile.mkdtemp(), "p.xlsx"); _prepost_file(path)
        rep = s.load(path); pid = rep.plans[0]["plan_id"]
        s.select_plans([pid]); s.prepare(pid); s.approve(pid); s.analyse()
        r = s.report(["pooled"], ["forest"], ["methods"])
        self.assertFalse(any(sec.heading == "Study selection" for sec in r.sections))
        self.assertEqual(s.prisma_autofill()["participants"], int(pd.read_excel(path)[["N_I", "N_C"]].sum().sum()))
