# MetaPy 5.1 - comparison with R (metafor, clubSandwich, netmeta, bayesmeta)
#
# How to run (Google Colab):
#   1. Runtime > Change runtime type > Runtime type: R > Save.
#   2. Paste this whole script into one cell and run it (installing packages takes about 3-5 minutes).
#   3. Open the Files panel (folder icon on the left) and download "metafor_results.json".
#   4. Give that file to MetaPy: python validation/r_check/compare_with_r.py metafor_results.json
#
# Every analysis below uses the same data files as MetaPy (read from the public GitHub repository).

pkgs <- c("metafor", "clubSandwich", "netmeta", "bayesmeta", "jsonlite")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(metafor); library(clubSandwich); library(netmeta); library(bayesmeta); library(jsonlite)
})

base <- "https://raw.githubusercontent.com/Sadia-2305/MetaPy/main/validation/data/"
rd <- function(f) read.csv(paste0(base, f), stringsAsFactors = FALSE)

out <- list(versions = list(R = R.version.string,
                            metafor = as.character(packageVersion("metafor")),
                            clubSandwich = as.character(packageVersion("clubSandwich")),
                            netmeta = as.character(packageVersion("netmeta")),
                            bayesmeta = as.character(packageVersion("bayesmeta"))))
safe <- function(name, expr) {
  out[[name]] <<- tryCatch(expr, error = function(e) list(error = conditionMessage(e)))
  cat("done:", name, if (!is.null(out[[name]]$error)) paste("ERROR", out[[name]]$error) else "", "\n")
}

bcg <- rd("bcg.csv")
dat <- escalc(measure = "RR", ai = Events_I, n1i = N_I, ci = Events_C, n2i = N_C, data = bcg)

# ---------------------------------------------------------------- pairwise models
safe("tau2", {
  r <- list()
  for (m in c("REML", "ML", "DL", "PM", "EB", "HE", "HS", "HSk", "SJ")) {
    f <- rma(yi, vi, data = dat, method = m)
    r[[m]] <- list(tau2 = f$tau2, beta = c(f$beta), se = f$se)
  }
  r
})
safe("qprofile", {
  ci <- confint(rma(yi, vi, data = dat))
  list(tau2_lo = ci$random["tau^2", "ci.lb"], tau2_hi = ci$random["tau^2", "ci.ub"],
       I2_lo = ci$random["I^2(%)", "ci.lb"], I2_hi = ci$random["I^2(%)", "ci.ub"])
})
safe("mh", lapply(c(OR = "OR", RR = "RR", RD = "RD"), function(m) {
  f <- rma.mh(measure = m, ai = Events_I, n1i = N_I, ci = Events_C, n2i = N_C, data = bcg)
  list(beta = c(f$beta), se = f$se, QE = f$QE)
}))
safe("peto", {
  f <- rma.peto(ai = Events_I, n1i = N_I, ci = Events_C, n2i = N_C, data = bcg)
  list(beta = c(f$beta), se = f$se, QE = f$QE)
})
safe("glmm_or", {
  f <- rma.glmm(measure = "OR", ai = Events_I, n1i = N_I, ci = Events_C, n2i = N_C, data = bcg, model = "UM.FS")
  list(beta = c(f$beta), se = f$se, tau2 = f$tau2)
})
safe("glmm_plo", {
  f <- rma.glmm(measure = "PLO", xi = Events_I, ni = N_I, data = bcg)
  list(beta = c(f$beta), se = f$se, tau2 = f$tau2)
})
safe("influence", {
  f <- rma(yi, vi, data = dat)
  inf <- influence(f)
  x <- as.data.frame(inf$inf)
  list(rstudent = x$rstudent, dffits = x$dffits, cook_d = x$cook.d, cov_r = x$cov.r, tau2_del = x$tau2.del,
       QE_del = x$QE.del, hat = x$hat, dfbetas = as.data.frame(inf$dfbs)[, 1])
})
safe("tes", {
  t <- tes(dat$yi, dat$vi)
  list(O = t$O, E = t$E, pval = t$pval, theta = t$theta)
})
safe("trimfill", {
  tf <- trimfill(rma(yi, vi, data = dat))
  list(k0 = tf$k0, beta = c(tf$beta), se = tf$se)
})
safe("egger", {
  e <- regtest(rma(yi, vi, data = dat), model = "lm")
  list(t = e$zval, p = e$pval)
})
safe("begg", {
  r <- ranktest(rma(yi, vi, data = dat))
  list(tau = r$tau, p = r$pval)
})

# ---------------------------------------------------------------- meta-regression
safe("metareg_interaction", {
  f <- rma(yi, vi, mods = ~ Latitude * Year, data = dat, test = "z")
  a <- anova(f, btt = c(2, 4))
  list(beta = c(f$beta), se = f$se, tau2 = f$tau2, QM = f$QM, QM_lat = a$QM)
})
safe("predict", {
  f <- rma(yi, vi, mods = ~ Latitude, data = dat, test = "knha")
  p <- predict(f, newmods = c(10, 50))
  list(pred = p$pred, se = p$se, ci_lb = p$ci.lb, ci_ub = p$ci.ub, pi_lb = p$pi.lb, pi_ub = p$pi.ub)
})
safe("robust_cr2", {
  f <- rma(yi, vi, mods = ~ Latitude, data = dat)
  r <- robust(f, cluster = dat$Study, clubSandwich = TRUE)
  list(beta = c(r$beta), se = r$se, df = r$ddf)
})

# ---------------------------------------------------------------- selection models
safe("selection", {
  s <- rd("selection_example.csv")
  f <- rma(yi, vi, data = s, method = "ML")
  s1 <- selmodel(f, type = "stepfun", steps = c(0.025))
  s3 <- selmodel(f, type = "stepfun", steps = c(0.025, 0.05, 0.5))
  list(one = list(beta = c(s1$beta), se = s1$se, tau2 = s1$tau2, delta = s1$delta, LRT = s1$LRT),
       three = list(beta = c(s3$beta), se = s3$se, tau2 = s3$tau2, delta = s3$delta, LRT = s3$LRT))
})

# ---------------------------------------------------------------- location-scale and multilevel models
safe("location_scale", {
  l <- rd("locscale_example.csv")
  f <- rma(yi, vi, scale = ~ z, data = l)
  list(beta = c(f$beta), se = f$se, alpha = c(f$alpha), se_alpha = f$se.alpha)
})
safe("nested", {
  n <- rd("nested_example.csv")
  f <- rma.mv(yi, vi, random = ~ 1 | country / study / es, data = n)
  list(sigma2 = f$sigma2, beta = c(f$beta), se = f$se)
})
d <- rd("dependent_example.csv")
safe("threelevel", {
  f <- rma.mv(yi, vi, random = ~ 1 | study / es, data = d)
  list(sigma2 = f$sigma2, beta = c(f$beta), se = f$se)
})
safe("che_cr2", {
  V <- vcalc(vi, cluster = study, obs = es, rho = 0.6, data = d)
  f <- rma.mv(yi, V, mods = ~ dose, random = ~ 1 | study / es, data = d)
  r <- robust(f, cluster = d$study, clubSandwich = TRUE)
  list(sigma2 = f$sigma2, beta = c(r$beta), se = r$se, df = r$ddf)
})

# ---------------------------------------------------------------- network meta-analysis
safe("nma", {
  a <- rd("nma_arms.csv")
  p <- pairwise(treat = treatment, n = n, mean = mean, sd = sd, studlab = study, data = a, sm = "MD")
  nm <- netmeta(p, reference.group = "A", common = TRUE, random = TRUE)
  tr <- c("B", "C", "D")
  list(TE_common = nm$TE.common[tr, "A"], se_common = nm$seTE.common[tr, "A"],
       TE_random = nm$TE.random[tr, "A"], se_random = nm$seTE.random[tr, "A"], tau2 = nm$tau^2,
       Q = nm$Q, Q_heterogeneity = nm$Q.heterogeneity, Q_inconsistency = nm$Q.inconsistency)
})

# ---------------------------------------------------------------- Bayesian model
safe("bayes", {
  b <- bayesmeta(y = dat$yi, sigma = sqrt(dat$vi), mu.prior.mean = 0, mu.prior.sd = 10,
                 tau.prior = function(t) dhalfnormal(t, scale = 0.5))
  list(mu_median = b$summary["median", "mu"], mu_lo = b$summary["95% lower", "mu"], mu_hi = b$summary["95% upper", "mu"],
       tau_median = b$summary["median", "tau"])
})

write_json(out, "metafor_results.json", auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("\nDone. Download metafor_results.json from the Files panel (folder icon on the left).\n")
