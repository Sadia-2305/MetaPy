"""Generalized linear mixed models for binary data (MetaPy 5.1).

binomial_normal    : random-effects meta-analysis of single proportions on the logit scale (binomial-normal model;
                     Hamza, van Houwelingen & Stijnen 2008; Stijnen, Hamza & Ozdemir 2010)
logistic_fixed_study: random-effects meta-analysis of odds ratios by a mixed-effects logistic regression with fixed study
                     intercepts and a random treatment effect, group coded +1/2 and -1/2 (model "UM.FS" in metafor;
                     Stijnen et al. 2010; Jackson et al. 2018)

Both models are fitted by maximum likelihood with adaptive Gauss-Hermite quadrature over the random effect (one
dimension per study), and use the exact binomial likelihood, so no continuity correction is needed for zero cells.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, gammaln, log_expit
from scipy.stats import chi2, norm

Z975 = float(norm.ppf(0.975))
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(21)


def _lbinom(x, n, eta):
    """log binomial probability with logit eta (vectorised, stable)."""
    return gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1) + x * log_expit(eta) + (n - x) * log_expit(-eta)


def _log_integral(loglik_u, tau, k):
    """Integrate over u ~ N(0, tau^2) with adaptive Gauss-Hermite (mode and curvature found numerically per study)."""
    if tau < 1e-7:
        return loglik_u(np.zeros((k, 1)))[:, 0]
    u = np.zeros(k)
    h = 1e-4
    lp = lambda uu: loglik_u(uu[:, None])[:, 0] - uu ** 2 / (2 * tau ** 2)
    for _ in range(50):
        f0, fp, fm = lp(u), lp(u + h), lp(u - h)
        g = (fp - fm) / (2 * h)
        H = (fp - 2 * f0 + fm) / h ** 2
        H = np.where(H < -1e-12, H, -1e-12)
        step = -g / H
        step = np.clip(step, -2.0, 2.0)
        u = u + step
        if np.max(np.abs(step)) < 1e-10:
            break
    f0, fp, fm = lp(u), lp(u + h), lp(u - h)
    H = (fp - 2 * f0 + fm) / h ** 2
    sig = 1 / np.sqrt(np.clip(-H, 1e-12, None))
    nodes = u[:, None] + math.sqrt(2) * sig[:, None] * _GH_X[None, :]
    lv = loglik_u(nodes) - nodes ** 2 / (2 * tau ** 2) - 0.5 * math.log(2 * math.pi * tau ** 2) + _GH_X[None, :] ** 2
    lw = np.log(_GH_W)[None, :] + math.log(math.sqrt(2)) + np.log(sig)[:, None]
    m = np.max(lv + lw, axis=1)
    return m + np.log(np.sum(np.exp(lv + lw - m[:, None]), axis=1))


def _num_hess(f, x, h=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    E = np.eye(n) * h
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j]) - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return H


# ============================================================================ single proportions
def binomial_normal(xi, ni, level=0.95) -> dict[str, Any]:
    """Random-effects meta-analysis of proportions: logit(p_i) = mu + u_i, u_i ~ N(0, tau^2), exact binomial likelihood."""
    x, n = np.asarray(xi, float), np.asarray(ni, float)
    keep = n > 0
    x, n = x[keep], n[keep]
    k = len(x)
    if k < 2:
        return {"status": "Not pooled: fewer than 2 studies", "k": k}

    def ll(par):
        mu, tau = par[0], math.exp(par[1])
        return float(np.sum(_log_integral(lambda u: _lbinom(x[:, None], n[:, None], mu + u), tau, k)))

    def ll0(mu):
        return float(np.sum(_lbinom(x, n, mu)))

    p0 = np.clip((x + 0.5) / (n + 1), 1e-6, 1 - 1e-6)
    lg = np.log(p0 / (1 - p0))
    starts = [np.array([np.mean(lg), math.log(max(np.std(lg), 0.05))]), np.array([np.mean(lg), math.log(0.3)])]
    best = min((minimize(lambda p_: -ll(p_), s, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 4000})
                for s in starts), key=lambda r: r.fun)
    fe = minimize_scalar(lambda m_: -ll0(m_), bounds=(-30, 30), method="bounded", options={"xatol": 1e-12})
    mu, tau = best.x[0], math.exp(best.x[1])
    use_fe = -fe.fun >= -best.fun - 1e-9 and tau < 1e-3
    if use_fe:
        mu, tau = float(fe.x), 0.0
        info = -_num_hess(lambda p_: ll0(p_[0]), np.array([mu]))[0, 0]
        se = math.sqrt(1 / info)
    else:
        # Hessian on (mu, tau) scale
        f = lambda p_: ll(np.array([p_[0], math.log(max(p_[1], 1e-10))]))
        H = _num_hess(f, np.array([mu, tau]), h=1e-4)
        cov = np.linalg.inv(-H)
        se = math.sqrt(max(cov[0, 0], 0))
    zc = float(norm.ppf(0.5 + level / 2))
    lrt = max(0.0, 2 * (-best.fun - (-fe.fun)))
    lo, hi = mu - zc * se, mu + zc * se
    return {"status": "Pooled", "method": "Binomial-normal model (ML)", "k": k, "estimate": mu, "se": se, "ci_lo": lo,
            "ci_hi": hi, "z": mu / se, "p": float(2 * norm.sf(abs(mu / se))), "tau2": tau ** 2, "tau": tau,
            "LRT_tau2": lrt, "p_tau2": 0.5 * float(chi2.sf(lrt, 1)) if lrt > 0 else 1.0,
            "proportion": float(expit(mu)), "proportion_ci": [float(expit(lo)), float(expit(hi))],
            "loglik": -best.fun if not use_fe else -fe.fun}


# ============================================================================ odds ratios (UM.FS)
def logistic_fixed_study(ai, n1i, ci, n2i, level=0.95, coding=0.5) -> dict[str, Any]:
    """Mixed-effects logistic model with fixed study intercepts and a random log odds ratio (metafor 'UM.FS').

    logit P(event | study i, group g) = alpha_i + (theta + u_i) * x_g, x = +coding (group 1), -(1 - coding) (group 2);
    with coding = 1/2, x = +1/2 / -1/2 (metafor's default). Studies without events or without non-events in both
    groups carry no information and are dropped.
    """
    a, n1, c, n2 = (np.asarray(z, float) for z in (ai, n1i, ci, n2i))
    keep = ~(((a == 0) & (c == 0)) | ((a == n1) & (c == n2)))
    a, n1, c, n2 = a[keep], n1[keep], c[keep], n2[keep]
    k = len(a)
    if k < 2:
        return {"status": "Not pooled: fewer than 2 informative studies", "k": k}
    x1, x2 = coding, -(1 - coding) if coding != 0.5 else -0.5

    def study_ll(alpha, theta, tau):
        f = lambda u: (_lbinom(a[:, None], n1[:, None], alpha[:, None] + (theta + u) * x1)
                       + _lbinom(c[:, None], n2[:, None], alpha[:, None] + (theta + u) * x2))
        return _log_integral(f, tau, k)

    def prof_alpha(theta, tau, alpha0):
        """Maximise each alpha_i separately (Newton with numerical derivatives, vectorised)."""
        al = alpha0.copy()
        h = 1e-5
        for _ in range(60):
            f0, fp, fm = study_ll(al, theta, tau), study_ll(al + h, theta, tau), study_ll(al - h, theta, tau)
            g = (fp - fm) / (2 * h)
            H = (fp - 2 * f0 + fm) / h ** 2
            H = np.where(H < -1e-10, H, -1e-10)
            step = np.clip(-g / H, -3, 3)
            al = al + step
            if np.max(np.abs(step)) < 1e-9:
                break
        return al, float(np.sum(study_ll(al, theta, tau)))

    pa = np.clip((a + c + 0.5) / (n1 + n2 + 1), 1e-6, 1 - 1e-6)
    alpha_start = np.log(pa / (1 - pa))
    state = {"alpha": alpha_start.copy()}

    def negprof(par):
        theta, tau = par[0], math.exp(par[1])
        al, ll = prof_alpha(theta, tau, state["alpha"])
        state["alpha"] = al
        return -ll

    lor = np.log((a + 0.5) * (n2 - c + 0.5) / ((n1 - a + 0.5) * (c + 0.5)))
    best = None
    for lt in (math.log(0.3), math.log(0.05), math.log(1.0)):
        state["alpha"] = alpha_start.copy()
        r = minimize(negprof, np.array([float(np.mean(lor)), lt]), method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    theta, tau = best.x[0], math.exp(best.x[1])
    if tau < 1e-4:
        tau = 0.0
    state["alpha"], llmax = prof_alpha(theta, tau, state["alpha"])

    # SE of theta from the curvature of the profile log-likelihood in theta (alpha_i and tau re-maximised)
    def prof_theta(th):
        if tau == 0.0:
            return prof_alpha(th, 0.0, state["alpha"])[1]
        r = minimize_scalar(lambda lt: -prof_alpha(th, math.exp(lt), state["alpha"])[1],
                            bounds=(math.log(tau) - 3, math.log(tau) + 3), method="bounded", options={"xatol": 1e-8})
        return -r.fun
    hh = 1e-3
    c2 = (prof_theta(theta + hh) - 2 * prof_theta(theta) + prof_theta(theta - hh)) / hh ** 2
    se = math.sqrt(-1 / c2) if c2 < 0 else float("nan")
    # common-effect model (tau = 0) for the likelihood-ratio test of heterogeneity
    r0 = minimize_scalar(lambda th: -prof_alpha(th, 0.0, state["alpha"])[1], bounds=(theta - 5, theta + 5), method="bounded",
                         options={"xatol": 1e-10})
    lrt = max(0.0, 2 * (llmax + r0.fun))
    zc = float(norm.ppf(0.5 + level / 2))
    lo, hi = theta - zc * se, theta + zc * se
    return {"status": "Pooled", "method": "Mixed-effects logistic model, fixed study effects (ML)", "measure": "OR", "k": k,
            "estimate": theta, "se": se, "ci_lo": lo, "ci_hi": hi, "z": theta / se, "p": float(2 * norm.sf(abs(theta / se))),
            "tau2": tau ** 2, "tau": tau, "LRT_tau2": lrt, "p_tau2": 0.5 * float(chi2.sf(lrt, 1)) if lrt > 0 else 1.0,
            "exp_estimate": math.exp(theta), "exp_ci_lo": math.exp(lo), "exp_ci_hi": math.exp(hi), "loglik": llmax}
