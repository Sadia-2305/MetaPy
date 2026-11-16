"""Tests for the MetaPy 5.1 methods, against values published in metafor's documentation and test suite."""
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from metapy import advmodels as AM
from metapy import extended as E
from metapy import glmm as G
from metapy import metareg as MR

ROOT = Path(__file__).resolve().parents[1]
warnings.simplefilter("ignore")


def _bcg():
    d = pd.read_csv(ROOT / "validation" / "data" / "bcg.csv")
    a, n1, c, n2 = d.Events_I.values, d.N_I.values, d.Events_C.values, d.N_C.values
    d["yi"] = np.log((a / n1) / (c / n2))
    d["vi"] = 1 / a - 1 / n1 + 1 / c - 1 / n2
    return d


def _dk2007():
    n1 = np.array([156, 303, 565, 1570, 103, 4659.]); n2 = np.array([74, 303, 477, 1565, 105, 4650.])
    a = np.array([5, 5, 12, 69, 9, 313.]); c = np.array([8, 17, 9, 94, 11, 352.])
    b, d = n1 - a, n2 - c
    return np.log(a * d / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d


def test_tau2_estimators_dersimonian_kacker_2007():
    # metafor tests/testthat/test_analysis_example_dersimonian2007.r (tau, mu, se)
    y, v = _dk2007()
    expected = {"PM": (0.3681, -0.3811, 0.2060), "HE": (0.4410, -0.4035, 0.2327), "DL": (0.2323, -0.3240, 0.1540),
                "EB": (0.3681, -0.3811, 0.2060), "REML": (0.1843, -0.2980, 0.1343), "HS": (0.1330, -0.2666, 0.1125),
                "SJ": (0.4572, -0.4079, 0.2386), "HSk": (0.1644, -0.2863, 0.1259)}
    for m, (tau, mu, se) in expected.items():
        t2 = E.estimate_tau2(y, v, m)
        w = 1 / (v + t2)
        assert abs(math.sqrt(t2) - tau) < 6e-5, m
        assert abs(np.sum(w * y) / np.sum(w) - mu) < 6e-5, m
        assert abs(math.sqrt(1 / np.sum(w)) - se) < 6e-5, m
    # ML: the exact likelihood maximum is at tau^2 = 0 (metafor stops at 5e-6, printed tau = 0.0023, mu = -0.1974)
    assert E.estimate_tau2(y, v, "ML") < 1e-5


def test_tau2_estimators_viechtbauer_2005():
    # metafor-project.org/doku.php/analyses:viechtbauer2005 (tau^2 rounded to 3 decimals)
    y = np.array([-0.581, 0.530, 0.771, 1.031, 0.553, 0.295, 0.078, 0.573, -0.176, -0.232])
    v = np.array([0.023, 0.052, 0.060, 0.115, 0.095, 0.203, 0.200, 0.211, 0.051, 0.040])
    ref = dict(HS=0.228, HE=0.148, DL=0.277, ML=0.197, REML=0.223, EB=0.192, SJ=0.199)
    for m, r in ref.items():
        assert round(E.estimate_tau2(y, v, m), 3) == r, m


def test_mantel_haenszel_and_peto_bcg():
    # metafor tests: test_misc_metan_vs_rma.mh_with_dat.bcg.r and ..._rma.peto_with_dat.bcg.r
    d = _bcg()
    a, n1, c, n2 = d.Events_I.values, d.N_I.values, d.Events_C.values, d.N_C.values
    ref = {"OR": (-0.4734, -0.5538, -0.3930, -11.5444, 163.9426), "RR": (-0.4537, -0.5308, -0.3766, -11.5338, 152.5676),
           "RD": (-0.0033, -0.0039, -0.0027, -11.4708, 386.7759)}
    for m, (b, lo, hi, z, qe) in ref.items():
        r = E.mantel_haenszel(a, n1 - a, c, n2 - c, m)
        assert round(r["estimate"], 4) == b and round(r["ci_lo"], 4) == lo and round(r["ci_hi"], 4) == hi, m
        assert round(r["z"], 4) == z and round(r["QE"], 4) == qe, m
    r = E.peto(a, n1 - a, c, n2 - c)
    assert (round(r["estimate"], 4), round(r["ci_lo"], 4), round(r["ci_hi"], 4)) == (-0.4744, -0.5541, -0.3948)
    assert round(r["z"], 4) == -11.6689 and round(r["QE"], 4) == 167.7302
    assert (round(r["exp_estimate"], 4), round(r["exp_ci_lo"], 4), round(r["exp_ci_hi"], 4)) == (0.6222, 0.5746, 0.6738)


def test_metareg_bcg():
    d = _bcg()
    f = MR.fit_df(d, ["Latitude"])
    assert abs(f.tau2 - 0.076348) < 2e-6
    assert abs(MR.omnibus(f)["statistic"] - 12.5905) < 1e-3
    f2 = MR.fit_df(d, ["Latitude", "Year"], test="z")
    assert abs(MR.r2(f2) / 100 - 0.6463217) < 1e-4            # value used in the 'performance' package tests
    f3 = MR.fit_df(d, ["Latitude*Year"], test="z")
    assert f3.names[-1] == "Latitude × Year" and f3.p == 4
    pr = MR.predict(f, [{"Latitude": 0}])
    assert abs(pr["pred"][0] - f.beta[0]) < 1e-12
    rb = MR.robust(f, d["Study"])
    assert rb.shape[0] == 2 and (rb["df (Satterthwaite)"] > 0).all()


def test_selection_stepfun_matches_3psm():
    from metapy import advanced as A
    rng = np.random.default_rng(3)
    v = rng.uniform(0.01, 0.2, 40)
    y = rng.normal(0.3, np.sqrt(v + 0.05))
    s1 = E.selection_stepfun(y, v, [0.025])
    s0 = A.selection_3psm(y, v, 0.025, direction=1)
    assert abs(s1["adjusted_estimate"] - s0["adjusted_estimate"]) < 1e-4
    assert abs(s1["LRT"] - s0["LRT"]) < 1e-3
    s3 = E.selection_stepfun(y, v, [0.025, 0.05, 0.5])
    assert s3["LRT_df"] == 3 and len(s3["delta"]) == 4


def test_location_scale_and_multilevel_special_cases():
    d = _bcg()
    re_t2 = E.estimate_tau2(d.yi.values, d.vi.values, "REML")
    r = AM.location_scale(d.yi, d.vi, np.ones((13, 1)), np.ones((13, 1)), ["Intercept"], ["Intercept"])
    assert abs(math.exp(r["scale"].estimate[0]) - re_t2) < 1e-6
    d["es"] = range(13)
    m = AM.multilevel(d, ["es"])
    assert abs(m["components"].sigma2[0] - re_t2) < 1e-6


def test_glmm_quadrature_and_fit():
    from scipy import integrate
    x, n, mu, tau = np.array([3.0]), np.array([40.0]), -2.0, 0.8
    q = G._log_integral(lambda u: G._lbinom(x[:, None], n[:, None], mu + u), tau, 1)[0]
    bf = math.log(integrate.quad(lambda u: math.exp(G._lbinom(3.0, 40.0, mu + u)) * math.exp(-u * u / (2 * tau * tau))
                                 / math.sqrt(2 * math.pi * tau * tau), -15, 15, epsabs=1e-14)[0])
    assert abs(q - bf) < 1e-9
    d = _bcg()
    r = G.binomial_normal(d.Events_I, d.N_I)
    assert r["status"] == "Pooled" and r["tau2"] > 0


def test_tes_and_influence_run():
    d = _bcg()
    t = E.tes(d.yi.values, d.vi.values)
    assert t["O"] == 8 and 0 < t["p"] < 1
    inf = E.influence_full(d.yi.values, d.vi.values, d.Study)
    assert list(inf.columns[:5]) == ["study", "rstudent", "dffits", "cook_d", "cov_r"]
    assert abs(inf["hat"].sum() - 1) < 1e-9


def test_template_is_readable_and_interpretation_is_cautious():
    import tempfile
    from metapy import guide as GD
    from metapy.session import MetaPySession
    d = tempfile.mkdtemp()
    path = GD.make_template(Path(d) / "t.xlsx")
    s = MetaPySession()
    s.load(str(path))
    pids = [p["plan_id"] for p in s.scan_report.plans if p["recommended"]]
    assert len(pids) == 2                                   # instructions sheet ignored; two example outcomes found
    s.select_plans(pids)
    s.approve_many(pids)
    res = s.analyse({"model": "random", "tau_method": "REML", "ci_method": "HK", "extras": []})
    for r in res.values():
        text = " ".join(GD.interpret(r)).lower()
        assert "works" not in text and "proves" not in text
        assert any("studies" in w for w in GD.warnings_for(r))   # few studies: always warned
