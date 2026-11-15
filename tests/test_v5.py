"""Tests for MetaPy 5 methods: dependent effects, heterogeneity CIs, bias methods, Bayesian model, network meta-analysis."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import unittest


class _Approx:                       # minimal stand-in for pytest.approx, so tests run under plain unittest too
    def __init__(self, v, abs=None, rel=None):
        self.v, self.abs, self.rel = v, abs, rel

    def __eq__(self, other):
        tol = max(self.abs or 0.0, (self.rel or 0.0) * _abs(self.v))
        if self.abs is None and self.rel is None:
            tol = 1e-6 * max(1.0, _abs(self.v))
        return _abs(other - self.v) <= tol

    def __repr__(self):
        return f"{self.v} ± {self.abs or self.rel}"


_abs = abs


class pytest:                        # noqa: N801
    approx = _Approx

    class raises:                    # noqa: N801
        def __init__(self, exc, match=""):
            self.exc, self.match = exc, match

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            if t is None:
                raise AssertionError(f"{self.exc.__name__} not raised")
            if not issubclass(t, self.exc) or self.match not in str(v):
                return False
            return True

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metapy import advanced as AD, dependent as D, nma as N, stats  # noqa: E402
from test_metapy import BCG, bcg_yv  # noqa: E402


# ------------------------------------------------------------------ dependent effects
def _three_level(seed=1, n_st=25, s2b=0.08, s2w=0.03, mu=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_st):
        u = rng.normal(0, math.sqrt(s2b))
        for i in range(int(rng.integers(1, 5))):
            v = rng.uniform(0.02, 0.08)
            rows.append({"study_id": f"S{j}", "yi": mu + u + rng.normal(0, math.sqrt(s2w)) + rng.normal(0, math.sqrt(v)), "vi": v})
    return pd.DataFrame(rows)


def check_multilevel_equals_re_with_one_effect_per_study():
    y, v = bcg_yv()
    df = pd.DataFrame({"study_id": range(len(y)), "yi": y, "vi": v})
    f = D.fit_dependent(df)
    assert f.estimate == pytest.approx(-0.7145, abs=1e-4)
    assert f.sigma2_between + f.sigma2_within == pytest.approx(0.3132, abs=1e-4)


def check_multilevel_recovers_components_and_lrt():
    df = _three_level(seed=7, n_st=60)
    r = D.multilevel(df)
    f = r["fit"]
    assert f.estimate == pytest.approx(0.3, abs=0.12)
    assert 0.02 < f.sigma2_between < 0.2
    lt = r["level_tests"]
    assert lt["p (boundary mixture)"].between(0, 1).all()
    assert lt["p (boundary mixture)"].iloc[0] < 0.05          # between-study variance clearly present


def check_cr2_coverage():
    """Monte Carlo: CR2 + Satterthwaite 95% CIs should cover the true mean close to nominally (CHE model correctly specified).
    Quick version by default (20 simulated meta-analyses); set METAPY_SLOW_TESTS=1 for the full 120."""
    import os
    reps = 120 if os.environ.get("METAPY_SLOW_TESTS") == "1" else 20
    hits = 0
    for s in range(reps):
        df = _three_level(seed=100 + s, n_st=15)
        rob = D.che_rve(df, rho=0.0)["robust"].iloc[0]
        hits += rob["ci_lo"] <= 0.3 <= rob["ci_hi"]
    assert hits / reps >= (0.88 if reps >= 100 else 0.85)


def check_cr2_reduces_to_model_se_order():
    df = _three_level(seed=3)
    r = D.che_rve(df, rho=0.6)
    assert r["robust"]["se (CR2)"].iloc[0] > 0
    assert r["robust"]["df (Satterthwaite)"].iloc[0] < df["study_id"].nunique()


# ------------------------------------------------------------------ advanced
def check_qprofile_bcg():
    y, v = bcg_yv()
    c = AD.tau2_qprofile(y, v)
    assert c["tau2_lo"] == pytest.approx(0.1197, abs=2e-4)     # metafor confint(): 0.1197 to 1.1115
    assert c["tau2_hi"] == pytest.approx(1.1115, abs=2e-4)


def check_pet_peese_matches_weighted_lm():
    y, v = bcg_yv()
    r = AD.pet_peese(y, v)
    X = np.column_stack([np.ones(len(y)), np.sqrt(v)])
    w = np.sqrt(1 / v)
    b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    assert r["PET"]["estimate"] == pytest.approx(b[0], abs=1e-10)


def check_3psm_without_selection_equals_ml_re():
    y, v = bcg_yv()
    s = AD.selection_3psm(y, v)
    from scipy.optimize import minimize_scalar
    def nll(t2):
        w = 1 / (v + t2); mu = np.sum(w * y) / w.sum()
        return 0.5 * np.sum(np.log(v + t2) + w * (y - mu) ** 2)
    t2 = minimize_scalar(nll, bounds=(0, 5), method="bounded").x
    w = 1 / (v + t2)
    assert s["unadjusted_ml_estimate"] == pytest.approx(np.sum(w * y) / w.sum(), abs=1e-3)
    assert 0 <= s["p_selection"] <= 1


def check_failsafe_rosenthal_formula():
    y, v = bcg_yv()
    z = y / np.sqrt(v)
    assert AD.failsafe_n(y, v)["rosenthal_N"] == pytest.approx((z.sum() / 1.6448536) ** 2 - len(y), rel=1e-6)


def check_bayes_matches_brute_force_grid():
    y, v = bcg_yv()
    r = AD.bayes_re(y, v, tau_prior_scale=0.5, mu_prior_sd=10)
    from scipy.stats import halfnorm, norm
    mus, taus = np.linspace(-2, 0.6, 700), np.linspace(1e-4, 2.5, 700)
    M, T = np.meshgrid(mus, taus, indexing="ij")
    ll = sum(norm.logpdf(yi, M, np.sqrt(vi + T ** 2)) for yi, vi in zip(y, v))
    lp = ll + norm.logpdf(M, 0, 10) + halfnorm.logpdf(T, scale=0.5)
    p = np.exp(lp - lp.max())
    pm = p.sum(1); pm /= pm.sum()
    med = np.interp(0.5, np.cumsum(pm), mus)
    assert r["mu_median"] == pytest.approx(med, abs=0.01)
    pt = p.sum(0); pt /= pt.sum()
    assert r["tau_median"] == pytest.approx(np.interp(0.5, np.cumsum(pt), taus), abs=0.01)


def check_permutation_metareg():
    y, v = bcg_yv()
    df = pd.DataFrame({"yi": y, "vi": v, "ablat": BCG["ablat"].to_numpy(), "alloc": np.arange(len(y)) % 3})
    r = AD.metareg_permutation(df, ["ablat", "alloc"], n_perm=199)
    assert 0 < r["permutation"]["QM_p"] <= 1
    assert "p_permutation" in r["coefficients"]


def check_peters_runs():
    y, v = bcg_yv()
    n = (BCG.tpos + BCG.tneg + BCG.cpos + BCG.cneg).to_numpy()
    r = AD.peters(y, v, n)
    assert 0 <= r["p"] <= 1


# ------------------------------------------------------------------ network meta-analysis
def _arm_data(seed=3):
    rng = np.random.default_rng(seed)
    true = {"A": 0.0, "B": -0.5, "C": -0.8, "D": -0.2}
    designs = [("A", "B")] * 4 + [("A", "C")] * 3 + [("B", "C")] * 2 + [("A", "D")] * 2 + [("A", "B", "C")] * 2 + [("C", "D")]
    rows = []
    for i, ds in enumerate(designs):
        base = rng.normal(10, 1)
        for t in ds:
            n = int(rng.integers(30, 80))
            rows.append({"study": f"S{i+1}", "treatment": t, "mean": base + true[t] * 2 + rng.normal(0, 2 / math.sqrt(n)), "sd": 2.0, "n": n})
    return pd.DataFrame(rows)


def check_nma_md_equals_arm_based_wls():
    arm = _arm_data()
    fe = N.fit_nma(N.contrasts_from_arms(arm, "MD"), "A", "fixed")
    studies, trts = sorted(arm["study"].unique()), ["B", "C", "D"]
    X = np.zeros((len(arm), len(studies) + 3))
    for r, (s, t) in enumerate(zip(arm["study"], arm["treatment"])):
        X[r, studies.index(s)] = 1
        if t != "A":
            X[r, len(studies) + trts.index(t)] = 1
    W = np.diag(1 / (arm["sd"] ** 2 / arm["n"]).to_numpy())
    b = np.linalg.solve(X.T @ W @ X, X.T @ W @ arm["mean"].to_numpy())
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ W @ X)))
    np.testing.assert_allclose(fe.d, b[-3:], atol=1e-8)
    np.testing.assert_allclose(np.sqrt(np.diag(fe.cov)), se[-3:], atol=1e-8)


def check_nma_star_equals_pairwise():
    y, v = bcg_yv()
    df = pd.DataFrame({"study": [f"T{i}" for i in range(len(y))], "treat1": "Vaccine", "treat2": "Control", "TE": y, "seTE": np.sqrt(v)})
    r = N.fit_nma(N.contrasts_from_table(df), "Control", "random")
    assert r.d[0] == pytest.approx(-0.7145, abs=1e-4)
    assert r.tau2 == pytest.approx(0.3132, abs=1e-4)


def check_nma_consistent_triangle_and_outputs():
    d = pd.DataFrame({"study": ["a", "b", "c"], "treat1": ["B", "C", "C"], "treat2": ["A", "A", "B"], "TE": [-0.5, -0.8, -0.3], "seTE": [0.2] * 3})
    out = N.run_nma(N.contrasts_from_table(d), "A", "fixed")
    assert out["result"].Q["Q_inconsistency"] == pytest.approx(0, abs=1e-10)
    assert out["pscores"]["treatment"].iloc[0] == "C"
    assert out["league"].shape == (3, 3)


def check_nma_smd_multiarm_cov_positive_definite():
    con = N.contrasts_from_arms(_arm_data(), "SMD")
    for C in con.cov.values():
        assert np.all(np.linalg.eigvalsh(C) > 0)


def check_nma_disconnected_raises():
    d = pd.DataFrame({"study": ["a", "b"], "treat1": ["B", "D"], "treat2": ["A", "C"], "TE": [0.1, 0.2], "seTE": [0.1, 0.1]})
    with pytest.raises(ValueError, match="disconnected"):
        N.fit_nma(N.contrasts_from_table(d))


def check_app_network_panel_and_build():
    import os
    import tempfile
    import fake_gradio
    fake_gradio.install()
    from metapy import app as A
    from test_metapy import _prepost_file
    path = os.path.join(tempfile.mkdtemp(), "prepost_nma.xlsx")
    _prepost_file(path)
    with pd.ExcelWriter(path, mode="a", engine="openpyxl") as xw:
        _arm_data().to_excel(xw, sheet_name="Arms", index=False)
    s, *_ = A.on_scan(path, False)
    dd, msg = A.on_nma_sources(s)
    assert any(v == "sheet::Arms" for _, v in dd["choices"])
    out = A.on_nma(s, "sheet::Arms", "MD", "Random effects", "Smaller values are better", "A")
    assert "4 treatments" in out[1]
    assert len(s.nma_results) == 1
    A.build_app()
    # a new file clears the network results
    s.load(path)
    assert s.nma_results == []








def check_prisma_free_text():
    import matplotlib
    matplotlib.use("Agg")
    from metapy import app as A
    from metapy.figures import prisma_diagram, prisma_given, prisma_checks
    vals = [True, "Databases (n = 3)\n  Source one (n = 1,200)\nRegisters (n = 40)", "Duplicates (n = 300)", "(n = 940)", "(n = 900)",
            "(n = 40)", "(n = 2)", "(n = 38)", "(n = 26)\nWrong design (n = 20)\nNo data (n = 6)", "(n = 12 RCTs)", "", "", "", "", "",
            "One trial contributed two comparisons."]
    d = A._prisma_dict(vals)
    assert prisma_given(d) and prisma_checks(d) == []
    assert not prisma_given(A._prisma_dict([True] + [""] * (len(A.PRISMA_KEYS) - 1)))
    assert A._prisma_dict([False] + vals[1:]) is None
    fig = prisma_diagram(d)
    texts = [t.get_text() for t in fig.axes[0].texts]
    for expected in ("Records identified from:", "Source one (n = 1,200)", "Wrong design (n = 20)", "(n = 12 RCTs)", "Note:",
                     "One trial contributed two comparisons."):
        assert expected in texts, expected
    assert not any("PubMed" in t or "Scopus" in t for t in texts)






def check_web_mode_saves_nothing():
    import importlib
    import os
    import tempfile
    import fake_gradio
    fake_gradio.install()
    from test_metapy import _prepost_file
    save_to = tempfile.mkdtemp()
    old = {k: os.environ.get(k) for k in ("METAPY_WEB", "METAPY_SAVE_DIR")}
    os.environ["METAPY_WEB"], os.environ["METAPY_SAVE_DIR"] = "1", save_to
    try:
        from metapy import app as A
        A = importlib.reload(A)
        assert A.WEB_MODE
        path = os.path.join(tempfile.mkdtemp(), "web.xlsx")
        _prepost_file(path)
        s, *_ = A.on_scan(path, False)
        s, _, dd = A.on_select(s, [s.scan_report.plans[0]["plan_id"]])
        assert os.listdir(save_to) == []                      # nothing written to the server
        assert A.on_resume(os.path.join(save_to, "x.metapy"))[0] is None
        A.build_app()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        importlib.reload(A)


class TestV5(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("check_") and callable(_fn):
        setattr(TestV5, "test_" + _name[6:], (lambda f: lambda self: f())(_fn))


del _name, _fn
