# MetaPy derivation rules

180 rules. Generated from `metapy/derive/registry.py`.

**exact**: algebraic identity or standard formula; applied automatically and logged.  
**approximation**: published estimator; applied automatically, logged, and listed for the researcher to accept or veto.  
**assumption**: needs a researcher decision; only applied when the researcher switches it on.

## Researcher-controllable groups

| Group | What it does | Source |
|---|---|---|
| `median_to_mean` | Estimate mean/SD from median, IQR or range | Wan et al. 2014; Luo et al. 2018 |
| `or_to_smd` | Convert between log odds ratio and SMD | Chinn 2000; Cochrane Handbook 10.6 |
| `r_to_smd` | Convert between correlation r and SMD | Borenstein et al. 2009, ch. 7 |
| `spearman` | Convert Spearman rho to Pearson r | Rupinski & Dunlap 1996 |
| `pct_to_events` | Reconstruct event counts from percentages | Rounded to nearest integer |
| `p_inequality` | Use p-value thresholds such as p<0.05 as exact | Conservative; Cochrane Handbook 6.5.2.3 |
| `assumed_r` | Impute a pre-post correlation | Researcher-chosen value; Cochrane Handbook 6.5.2.8 |
| `equal_split` | Split a total N equally between two arms | Researcher assumption |
| `smd_type` | Interpret an SMD of unstated type | Researcher decision |
| `zero_cell` | Add 0.5 to all cells when a cell is zero | Cochrane Handbook 10.4.4.2 |
| `g_type` | Use a reported Hedges' g whose design (endpoint / change / PPC2) was confirmed by the researcher | Researcher decision, supported by matching rows |
| `ppc2_var_simple` | Approximate the PPC2 variance with the independent-groups SMD formula (no pre-post r needed) | Borenstein et al. 2009, eq. 4.20; equivalent to Morris 2008 with r ≈ 0.5 |

## All rules

| Rule | Type | Needs | Gives | Formula | Citation |
|---|---|---|---|---|---|
| Total N from arm sizes | exact | N (intervention), N (control) | Total N | `N = n_i + n_c` |  |
| Control N from total | exact | Total N, N (intervention) | N (control) | `n_c = N - n_i` |  |
| Intervention N from total | exact | Total N, N (control) | N (intervention) | `n_i = N - n_c` |  |
| Equal split of total N | assumption | Total N | N (intervention), N (control) | `n_i = n_c = N / 2` | Researcher assumption |
| SD from SE (post, i) | exact | SE post-test / endpoint (intervention), N (intervention) | SD post-test / endpoint (intervention) | `SD = SE x sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| SE from SD (post, i) | exact | SD post-test / endpoint (intervention), N (intervention) | SE post-test / endpoint (intervention) | `SE = SD / sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| Mean from 95% CI (post, i) | exact | Ci lo post-test / endpoint (intervention), Ci hi post-test / endpoint (intervention) | Mean post-test / endpoint (intervention) | `mean = (lower + upper) / 2` | Cochrane Handbook 6.5.2.2 |
| SD from 95% CI of mean (post, i) | exact | Ci lo post-test / endpoint (intervention), Ci hi post-test / endpoint (intervention), N (intervention) | SD post-test / endpoint (intervention) | `SD = sqrt(n) x (upper - lower) / (2 x t(0.975, n-1))` | Cochrane Handbook 6.5.2.2 |
| Mean from median and IQR (post, i) | approximation | Median post-test / endpoint (intervention), Q1 post-test / endpoint (intervention), Q3 post-test / endpoint (intervention), N (intervention) | Mean post-test / endpoint (intervention) | `mean = w(q1+q3)/2 + (1-w)median, w = 0.7 + 0.39/n` | Luo et al. 2018, Stat Methods Med Res |
| SD from IQR (post, i) | approximation | Q1 post-test / endpoint (intervention), Q3 post-test / endpoint (intervention), N (intervention) | SD post-test / endpoint (intervention) | `SD = (q3 - q1) / (2 x Phi^-1((0.75n - 0.125)/(n + 0.25)))` | Wan et al. 2014, BMC Med Res Methodol |
| Mean from median and range (post, i) | approximation | Median post-test / endpoint (intervention), Min post-test / endpoint (intervention), Max post-test / endpoint (intervention), N (intervention) | Mean post-test / endpoint (intervention) | `mean = w(min+max)/2 + (1-w)median, w = 4/(4 + n^0.75)` | Luo et al. 2018 |
| SD from range (post, i) | approximation | Min post-test / endpoint (intervention), Max post-test / endpoint (intervention), N (intervention) | SD post-test / endpoint (intervention) | `SD = (max - min) / (2 x Phi^-1((n - 0.375)/(n + 0.25)))` | Wan et al. 2014 |
| Mean from five-number summary (post, i) | approximation | Median post-test / endpoint (intervention), Q1 post-test / endpoint (intervention), Q3 post-test / endpoint (intervention), Min post-test / endpoint (intervention), Max post-test / endpoint (intervention), N (intervention) | Mean post-test / endpoint (intervention) | `mean = w1(min+max)/2 + w2(q1+q3)/2 + (1-w1-w2)median` | Luo et al. 2018 |
| SD from five-number summary (post, i) | approximation | Q1 post-test / endpoint (intervention), Q3 post-test / endpoint (intervention), Min post-test / endpoint (intervention), Max post-test / endpoint (intervention), N (intervention) | SD post-test / endpoint (intervention) | `SD = (max-min)/(4 xi) + (q3-q1)/(4 eta)` | Wan et al. 2014 |
| SD from SE (pre, i) | exact | SE baseline (intervention), N (intervention) | SD baseline (intervention) | `SD = SE x sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| SE from SD (pre, i) | exact | SD baseline (intervention), N (intervention) | SE baseline (intervention) | `SE = SD / sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| Mean from 95% CI (pre, i) | exact | Ci lo baseline (intervention), Ci hi baseline (intervention) | Mean baseline (intervention) | `mean = (lower + upper) / 2` | Cochrane Handbook 6.5.2.2 |
| SD from 95% CI of mean (pre, i) | exact | Ci lo baseline (intervention), Ci hi baseline (intervention), N (intervention) | SD baseline (intervention) | `SD = sqrt(n) x (upper - lower) / (2 x t(0.975, n-1))` | Cochrane Handbook 6.5.2.2 |
| Mean from median and IQR (pre, i) | approximation | Median baseline (intervention), Q1 baseline (intervention), Q3 baseline (intervention), N (intervention) | Mean baseline (intervention) | `mean = w(q1+q3)/2 + (1-w)median, w = 0.7 + 0.39/n` | Luo et al. 2018, Stat Methods Med Res |
| SD from IQR (pre, i) | approximation | Q1 baseline (intervention), Q3 baseline (intervention), N (intervention) | SD baseline (intervention) | `SD = (q3 - q1) / (2 x Phi^-1((0.75n - 0.125)/(n + 0.25)))` | Wan et al. 2014, BMC Med Res Methodol |
| Mean from median and range (pre, i) | approximation | Median baseline (intervention), Min baseline (intervention), Max baseline (intervention), N (intervention) | Mean baseline (intervention) | `mean = w(min+max)/2 + (1-w)median, w = 4/(4 + n^0.75)` | Luo et al. 2018 |
| SD from range (pre, i) | approximation | Min baseline (intervention), Max baseline (intervention), N (intervention) | SD baseline (intervention) | `SD = (max - min) / (2 x Phi^-1((n - 0.375)/(n + 0.25)))` | Wan et al. 2014 |
| Mean from five-number summary (pre, i) | approximation | Median baseline (intervention), Q1 baseline (intervention), Q3 baseline (intervention), Min baseline (intervention), Max baseline (intervention), N (intervention) | Mean baseline (intervention) | `mean = w1(min+max)/2 + w2(q1+q3)/2 + (1-w1-w2)median` | Luo et al. 2018 |
| SD from five-number summary (pre, i) | approximation | Q1 baseline (intervention), Q3 baseline (intervention), Min baseline (intervention), Max baseline (intervention), N (intervention) | SD baseline (intervention) | `SD = (max-min)/(4 xi) + (q3-q1)/(4 eta)` | Wan et al. 2014 |
| SD from SE (change, i) | exact | SE change from baseline (intervention), N (intervention) | SD change from baseline (intervention) | `SD = SE x sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| SE from SD (change, i) | exact | SD change from baseline (intervention), N (intervention) | SE change from baseline (intervention) | `SE = SD / sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| Mean from 95% CI (change, i) | exact | Ci lo change from baseline (intervention), Ci hi change from baseline (intervention) | Mean change from baseline (intervention) | `mean = (lower + upper) / 2` | Cochrane Handbook 6.5.2.2 |
| SD from 95% CI of mean (change, i) | exact | Ci lo change from baseline (intervention), Ci hi change from baseline (intervention), N (intervention) | SD change from baseline (intervention) | `SD = sqrt(n) x (upper - lower) / (2 x t(0.975, n-1))` | Cochrane Handbook 6.5.2.2 |
| Mean from median and IQR (change, i) | approximation | Median change from baseline (intervention), Q1 change from baseline (intervention), Q3 change from baseline (intervention), N (intervention) | Mean change from baseline (intervention) | `mean = w(q1+q3)/2 + (1-w)median, w = 0.7 + 0.39/n` | Luo et al. 2018, Stat Methods Med Res |
| SD from IQR (change, i) | approximation | Q1 change from baseline (intervention), Q3 change from baseline (intervention), N (intervention) | SD change from baseline (intervention) | `SD = (q3 - q1) / (2 x Phi^-1((0.75n - 0.125)/(n + 0.25)))` | Wan et al. 2014, BMC Med Res Methodol |
| Mean from median and range (change, i) | approximation | Median change from baseline (intervention), Min change from baseline (intervention), Max change from baseline (intervention), N (intervention) | Mean change from baseline (intervention) | `mean = w(min+max)/2 + (1-w)median, w = 4/(4 + n^0.75)` | Luo et al. 2018 |
| SD from range (change, i) | approximation | Min change from baseline (intervention), Max change from baseline (intervention), N (intervention) | SD change from baseline (intervention) | `SD = (max - min) / (2 x Phi^-1((n - 0.375)/(n + 0.25)))` | Wan et al. 2014 |
| Mean from five-number summary (change, i) | approximation | Median change from baseline (intervention), Q1 change from baseline (intervention), Q3 change from baseline (intervention), Min change from baseline (intervention), Max change from baseline (intervention), N (intervention) | Mean change from baseline (intervention) | `mean = w1(min+max)/2 + w2(q1+q3)/2 + (1-w1-w2)median` | Luo et al. 2018 |
| SD from five-number summary (change, i) | approximation | Q1 change from baseline (intervention), Q3 change from baseline (intervention), Min change from baseline (intervention), Max change from baseline (intervention), N (intervention) | SD change from baseline (intervention) | `SD = (max-min)/(4 xi) + (q3-q1)/(4 eta)` | Wan et al. 2014 |
| Mean change (i) | exact | Mean baseline (intervention), Mean post-test / endpoint (intervention) | Mean change from baseline (intervention) | `mean change = mean post - mean pre` | Cochrane Handbook 6.5.2.8 |
| Post mean from baseline and change (i) | exact | Mean baseline (intervention), Mean change from baseline (intervention) | Mean post-test / endpoint (intervention) | `mean post = mean pre + mean change` |  |
| SD of change from correlation (i) | exact | SD baseline (intervention), SD post-test / endpoint (intervention), Pre-post correlation (intervention) | SD change from baseline (intervention) | `SD change = sqrt(SDpre^2 + SDpost^2 - 2 r SDpre SDpost)` | Cochrane Handbook 6.5.2.8 |
| Pre-post correlation from SDs (i) | exact | SD baseline (intervention), SD post-test / endpoint (intervention), SD change from baseline (intervention) | Pre-post correlation (intervention) | `r = (SDpre^2 + SDpost^2 - SDchange^2) / (2 SDpre SDpost)` | Cochrane Handbook 6.5.2.8 |
| Imputed pre-post correlation (i) | assumption |  | Pre-post correlation (intervention) | `r = researcher-chosen value` | Cochrane Handbook 6.5.2.8; run sensitivity analysis |
| SD from SE (post, c) | exact | SE post-test / endpoint (control), N (control) | SD post-test / endpoint (control) | `SD = SE x sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| SE from SD (post, c) | exact | SD post-test / endpoint (control), N (control) | SE post-test / endpoint (control) | `SE = SD / sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| Mean from 95% CI (post, c) | exact | Ci lo post-test / endpoint (control), Ci hi post-test / endpoint (control) | Mean post-test / endpoint (control) | `mean = (lower + upper) / 2` | Cochrane Handbook 6.5.2.2 |
| SD from 95% CI of mean (post, c) | exact | Ci lo post-test / endpoint (control), Ci hi post-test / endpoint (control), N (control) | SD post-test / endpoint (control) | `SD = sqrt(n) x (upper - lower) / (2 x t(0.975, n-1))` | Cochrane Handbook 6.5.2.2 |
| Mean from median and IQR (post, c) | approximation | Median post-test / endpoint (control), Q1 post-test / endpoint (control), Q3 post-test / endpoint (control), N (control) | Mean post-test / endpoint (control) | `mean = w(q1+q3)/2 + (1-w)median, w = 0.7 + 0.39/n` | Luo et al. 2018, Stat Methods Med Res |
| SD from IQR (post, c) | approximation | Q1 post-test / endpoint (control), Q3 post-test / endpoint (control), N (control) | SD post-test / endpoint (control) | `SD = (q3 - q1) / (2 x Phi^-1((0.75n - 0.125)/(n + 0.25)))` | Wan et al. 2014, BMC Med Res Methodol |
| Mean from median and range (post, c) | approximation | Median post-test / endpoint (control), Min post-test / endpoint (control), Max post-test / endpoint (control), N (control) | Mean post-test / endpoint (control) | `mean = w(min+max)/2 + (1-w)median, w = 4/(4 + n^0.75)` | Luo et al. 2018 |
| SD from range (post, c) | approximation | Min post-test / endpoint (control), Max post-test / endpoint (control), N (control) | SD post-test / endpoint (control) | `SD = (max - min) / (2 x Phi^-1((n - 0.375)/(n + 0.25)))` | Wan et al. 2014 |
| Mean from five-number summary (post, c) | approximation | Median post-test / endpoint (control), Q1 post-test / endpoint (control), Q3 post-test / endpoint (control), Min post-test / endpoint (control), Max post-test / endpoint (control), N (control) | Mean post-test / endpoint (control) | `mean = w1(min+max)/2 + w2(q1+q3)/2 + (1-w1-w2)median` | Luo et al. 2018 |
| SD from five-number summary (post, c) | approximation | Q1 post-test / endpoint (control), Q3 post-test / endpoint (control), Min post-test / endpoint (control), Max post-test / endpoint (control), N (control) | SD post-test / endpoint (control) | `SD = (max-min)/(4 xi) + (q3-q1)/(4 eta)` | Wan et al. 2014 |
| SD from SE (pre, c) | exact | SE baseline (control), N (control) | SD baseline (control) | `SD = SE x sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| SE from SD (pre, c) | exact | SD baseline (control), N (control) | SE baseline (control) | `SE = SD / sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| Mean from 95% CI (pre, c) | exact | Ci lo baseline (control), Ci hi baseline (control) | Mean baseline (control) | `mean = (lower + upper) / 2` | Cochrane Handbook 6.5.2.2 |
| SD from 95% CI of mean (pre, c) | exact | Ci lo baseline (control), Ci hi baseline (control), N (control) | SD baseline (control) | `SD = sqrt(n) x (upper - lower) / (2 x t(0.975, n-1))` | Cochrane Handbook 6.5.2.2 |
| Mean from median and IQR (pre, c) | approximation | Median baseline (control), Q1 baseline (control), Q3 baseline (control), N (control) | Mean baseline (control) | `mean = w(q1+q3)/2 + (1-w)median, w = 0.7 + 0.39/n` | Luo et al. 2018, Stat Methods Med Res |
| SD from IQR (pre, c) | approximation | Q1 baseline (control), Q3 baseline (control), N (control) | SD baseline (control) | `SD = (q3 - q1) / (2 x Phi^-1((0.75n - 0.125)/(n + 0.25)))` | Wan et al. 2014, BMC Med Res Methodol |
| Mean from median and range (pre, c) | approximation | Median baseline (control), Min baseline (control), Max baseline (control), N (control) | Mean baseline (control) | `mean = w(min+max)/2 + (1-w)median, w = 4/(4 + n^0.75)` | Luo et al. 2018 |
| SD from range (pre, c) | approximation | Min baseline (control), Max baseline (control), N (control) | SD baseline (control) | `SD = (max - min) / (2 x Phi^-1((n - 0.375)/(n + 0.25)))` | Wan et al. 2014 |
| Mean from five-number summary (pre, c) | approximation | Median baseline (control), Q1 baseline (control), Q3 baseline (control), Min baseline (control), Max baseline (control), N (control) | Mean baseline (control) | `mean = w1(min+max)/2 + w2(q1+q3)/2 + (1-w1-w2)median` | Luo et al. 2018 |
| SD from five-number summary (pre, c) | approximation | Q1 baseline (control), Q3 baseline (control), Min baseline (control), Max baseline (control), N (control) | SD baseline (control) | `SD = (max-min)/(4 xi) + (q3-q1)/(4 eta)` | Wan et al. 2014 |
| SD from SE (change, c) | exact | SE change from baseline (control), N (control) | SD change from baseline (control) | `SD = SE x sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| SE from SD (change, c) | exact | SD change from baseline (control), N (control) | SE change from baseline (control) | `SE = SD / sqrt(n)` | Cochrane Handbook 6.5.2.2 |
| Mean from 95% CI (change, c) | exact | Ci lo change from baseline (control), Ci hi change from baseline (control) | Mean change from baseline (control) | `mean = (lower + upper) / 2` | Cochrane Handbook 6.5.2.2 |
| SD from 95% CI of mean (change, c) | exact | Ci lo change from baseline (control), Ci hi change from baseline (control), N (control) | SD change from baseline (control) | `SD = sqrt(n) x (upper - lower) / (2 x t(0.975, n-1))` | Cochrane Handbook 6.5.2.2 |
| Mean from median and IQR (change, c) | approximation | Median change from baseline (control), Q1 change from baseline (control), Q3 change from baseline (control), N (control) | Mean change from baseline (control) | `mean = w(q1+q3)/2 + (1-w)median, w = 0.7 + 0.39/n` | Luo et al. 2018, Stat Methods Med Res |
| SD from IQR (change, c) | approximation | Q1 change from baseline (control), Q3 change from baseline (control), N (control) | SD change from baseline (control) | `SD = (q3 - q1) / (2 x Phi^-1((0.75n - 0.125)/(n + 0.25)))` | Wan et al. 2014, BMC Med Res Methodol |
| Mean from median and range (change, c) | approximation | Median change from baseline (control), Min change from baseline (control), Max change from baseline (control), N (control) | Mean change from baseline (control) | `mean = w(min+max)/2 + (1-w)median, w = 4/(4 + n^0.75)` | Luo et al. 2018 |
| SD from range (change, c) | approximation | Min change from baseline (control), Max change from baseline (control), N (control) | SD change from baseline (control) | `SD = (max - min) / (2 x Phi^-1((n - 0.375)/(n + 0.25)))` | Wan et al. 2014 |
| Mean from five-number summary (change, c) | approximation | Median change from baseline (control), Q1 change from baseline (control), Q3 change from baseline (control), Min change from baseline (control), Max change from baseline (control), N (control) | Mean change from baseline (control) | `mean = w1(min+max)/2 + w2(q1+q3)/2 + (1-w1-w2)median` | Luo et al. 2018 |
| SD from five-number summary (change, c) | approximation | Q1 change from baseline (control), Q3 change from baseline (control), Min change from baseline (control), Max change from baseline (control), N (control) | SD change from baseline (control) | `SD = (max-min)/(4 xi) + (q3-q1)/(4 eta)` | Wan et al. 2014 |
| Mean change (c) | exact | Mean baseline (control), Mean post-test / endpoint (control) | Mean change from baseline (control) | `mean change = mean post - mean pre` | Cochrane Handbook 6.5.2.8 |
| Post mean from baseline and change (c) | exact | Mean baseline (control), Mean change from baseline (control) | Mean post-test / endpoint (control) | `mean post = mean pre + mean change` |  |
| SD of change from correlation (c) | exact | SD baseline (control), SD post-test / endpoint (control), Pre-post correlation (control) | SD change from baseline (control) | `SD change = sqrt(SDpre^2 + SDpost^2 - 2 r SDpre SDpost)` | Cochrane Handbook 6.5.2.8 |
| Pre-post correlation from SDs (c) | exact | SD baseline (control), SD post-test / endpoint (control), SD change from baseline (control) | Pre-post correlation (control) | `r = (SDpre^2 + SDpost^2 - SDchange^2) / (2 SDpre SDpost)` | Cochrane Handbook 6.5.2.8 |
| Imputed pre-post correlation (c) | assumption |  | Pre-post correlation (control) | `r = researcher-chosen value` | Cochrane Handbook 6.5.2.8; run sensitivity analysis |
| Mean difference (post) | exact | Mean post-test / endpoint (intervention), Mean post-test / endpoint (control) | Mean difference (endpoint) | `MD = mean_i - mean_c` |  |
| SE of MD from SDs (post) | exact | SD post-test / endpoint (intervention), SD post-test / endpoint (control), N (intervention), N (control) | SE of mean difference (endpoint) | `SE = sqrt(SD_i^2/n_i + SD_c^2/n_c)` | Borenstein et al. 2009, eq. 4.5 |
| Pooled SD (post) | exact | SD post-test / endpoint (intervention), SD post-test / endpoint (control), N (intervention), N (control) | Pooled SD (endpoint) | `SDp = sqrt(((n_i-1)SD_i^2 + (n_c-1)SD_c^2)/(n_i+n_c-2))` | Borenstein et al. 2009, eq. 4.19 |
| MD from its 95% CI (post) | exact | MD 95% CI lower (endpoint), MD 95% CI upper (endpoint) | Mean difference (endpoint) | `MD = (lower + upper)/2` | Cochrane Handbook 6.3.1 |
| SE of MD from its 95% CI (post) | exact | MD 95% CI lower (endpoint), MD 95% CI upper (endpoint) | SE of mean difference (endpoint) | `SE = (upper - lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.1 |
| |t| from two-sided p value (post) | exact | p value (endpoint), N (intervention), N (control) | |t| (endpoint) | `|t| = t^-1(1 - p/2, n_i + n_c - 2)` | Cochrane Handbook 6.5.2.3 |
| |t| from F with 1 df (post) | exact | F statistic (endpoint) | |t| (endpoint) | `|t| = sqrt(F)` | Borenstein et al. 2009, ch. 7 |
| |t| from t (post) | exact | t statistic (endpoint) | |t| (endpoint) | `|t| = abs(t)` |  |
| Signed t from |t| and MD sign (post) | exact | |t| (endpoint), Mean difference (endpoint) | t statistic (endpoint) | `t = sign(MD) x |t|` |  |
| Signed t from |t| and stated direction (post) | exact | |t| (endpoint), Direction (+1 intervention higher, -1 lower) | t statistic (endpoint) | `t = direction x |t|` |  |
| SE of MD from MD and t (post) | exact | Mean difference (endpoint), |t| (endpoint) | SE of mean difference (endpoint) | `SE = |MD| / |t|` | Cochrane Handbook 6.5.2.3 |
| Pooled SD from SE of MD (post) | exact | SE of mean difference (endpoint), N (intervention), N (control) | Pooled SD (endpoint) | `SDp = SE / sqrt(1/n_i + 1/n_c)` | Cochrane Handbook 6.5.2.3 |
| Cohen's d from MD and pooled SD (post) | exact | Mean difference (endpoint), Pooled SD (endpoint) | Cohen's d (endpoint) | `d = MD / SDp` | Borenstein et al. 2009, eq. 4.18 |
| Cohen's d from t (post) | exact | t statistic (endpoint), N (intervention), N (control) | Cohen's d (endpoint) | `d = t x sqrt(1/n_i + 1/n_c)` | Borenstein et al. 2009, eq. 7.1 |
| Hedges' g from d (post) | exact | Cohen's d (endpoint), N (intervention), N (control) | Hedges' g (endpoint) | `g = J x d, J = 1 - 3/(4(n_i+n_c-2) - 1)` | Hedges 1981; Borenstein et al. 2009, eq. 4.22 |
| Cohen's d from g (post) | exact | Hedges' g (endpoint), N (intervention), N (control) | Cohen's d (endpoint) | `d = g / J` | Borenstein et al. 2009, eq. 4.22 |
| Variance of d (post) | exact | Cohen's d (endpoint), N (intervention), N (control) | Variance of d (endpoint) | `V(d) = (n_i+n_c)/(n_i n_c) + d^2 / (2(n_i+n_c))` | Borenstein et al. 2009, eq. 4.20 |
| Variance of g from variance of d (post) | exact | Variance of d (endpoint), N (intervention), N (control) | Variance of g (endpoint) | `V(g) = J^2 x V(d)` | Borenstein et al. 2009, eq. 4.24 |
| Variance of d from variance of g (post) | exact | Variance of g (endpoint), N (intervention), N (control) | Variance of d (endpoint) | `V(d) = V(g) / J^2` |  |
| Variance of g from its SE (post) | exact | SE of g (endpoint) | Variance of g (endpoint) | `V = SE^2` |  |
| Variance of d from its SE (post) | exact | SE of d (endpoint) | Variance of d (endpoint) | `V = SE^2` |  |
| g from its 95% CI (post) | exact | g 95% CI lower (endpoint), g 95% CI upper (endpoint) | Hedges' g (endpoint) | `g = (lower + upper)/2` |  |
| SE of g from its 95% CI (post) | exact | g 95% CI lower (endpoint), g 95% CI upper (endpoint) | SE of g (endpoint) | `SE = (upper - lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.1 |
| d from its 95% CI (post) | exact | d 95% CI lower (endpoint), d 95% CI upper (endpoint) | Cohen's d (endpoint) | `d = (lower + upper)/2` |  |
| SE of d from its 95% CI (post) | exact | d 95% CI lower (endpoint), d 95% CI upper (endpoint) | SE of d (endpoint) | `SE = (upper - lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.1 |
| Mean difference (change) | exact | Mean change from baseline (intervention), Mean change from baseline (control) | Mean difference (change) | `MD = mean_i - mean_c` |  |
| SE of MD from SDs (change) | exact | SD change from baseline (intervention), SD change from baseline (control), N (intervention), N (control) | SE of mean difference (change) | `SE = sqrt(SD_i^2/n_i + SD_c^2/n_c)` | Borenstein et al. 2009, eq. 4.5 |
| Pooled SD (change) | exact | SD change from baseline (intervention), SD change from baseline (control), N (intervention), N (control) | Pooled SD (change) | `SDp = sqrt(((n_i-1)SD_i^2 + (n_c-1)SD_c^2)/(n_i+n_c-2))` | Borenstein et al. 2009, eq. 4.19 |
| MD from its 95% CI (change) | exact | MD 95% CI lower (change), MD 95% CI upper (change) | Mean difference (change) | `MD = (lower + upper)/2` | Cochrane Handbook 6.3.1 |
| SE of MD from its 95% CI (change) | exact | MD 95% CI lower (change), MD 95% CI upper (change) | SE of mean difference (change) | `SE = (upper - lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.1 |
| |t| from two-sided p value (change) | exact | p value (change), N (intervention), N (control) | |t| (change) | `|t| = t^-1(1 - p/2, n_i + n_c - 2)` | Cochrane Handbook 6.5.2.3 |
| |t| from F with 1 df (change) | exact | F statistic (change) | |t| (change) | `|t| = sqrt(F)` | Borenstein et al. 2009, ch. 7 |
| |t| from t (change) | exact | t statistic (change) | |t| (change) | `|t| = abs(t)` |  |
| Signed t from |t| and MD sign (change) | exact | |t| (change), Mean difference (change) | t statistic (change) | `t = sign(MD) x |t|` |  |
| Signed t from |t| and stated direction (change) | exact | |t| (change), Direction (+1 intervention higher, -1 lower) | t statistic (change) | `t = direction x |t|` |  |
| SE of MD from MD and t (change) | exact | Mean difference (change), |t| (change) | SE of mean difference (change) | `SE = |MD| / |t|` | Cochrane Handbook 6.5.2.3 |
| Pooled SD from SE of MD (change) | exact | SE of mean difference (change), N (intervention), N (control) | Pooled SD (change) | `SDp = SE / sqrt(1/n_i + 1/n_c)` | Cochrane Handbook 6.5.2.3 |
| Cohen's d from MD and pooled SD (change) | exact | Mean difference (change), Pooled SD (change) | Cohen's d (change) | `d = MD / SDp` | Borenstein et al. 2009, eq. 4.18 |
| Cohen's d from t (change) | exact | t statistic (change), N (intervention), N (control) | Cohen's d (change) | `d = t x sqrt(1/n_i + 1/n_c)` | Borenstein et al. 2009, eq. 7.1 |
| Hedges' g from d (change) | exact | Cohen's d (change), N (intervention), N (control) | Hedges' g (change) | `g = J x d, J = 1 - 3/(4(n_i+n_c-2) - 1)` | Hedges 1981; Borenstein et al. 2009, eq. 4.22 |
| Cohen's d from g (change) | exact | Hedges' g (change), N (intervention), N (control) | Cohen's d (change) | `d = g / J` | Borenstein et al. 2009, eq. 4.22 |
| Variance of d (change) | exact | Cohen's d (change), N (intervention), N (control) | Variance of d (change) | `V(d) = (n_i+n_c)/(n_i n_c) + d^2 / (2(n_i+n_c))` | Borenstein et al. 2009, eq. 4.20 |
| Variance of g from variance of d (change) | exact | Variance of d (change), N (intervention), N (control) | Variance of g (change) | `V(g) = J^2 x V(d)` | Borenstein et al. 2009, eq. 4.24 |
| Variance of d from variance of g (change) | exact | Variance of g (change), N (intervention), N (control) | Variance of d (change) | `V(d) = V(g) / J^2` |  |
| Variance of g from its SE (change) | exact | SE of g (change) | Variance of g (change) | `V = SE^2` |  |
| Variance of d from its SE (change) | exact | SE of d (change) | Variance of d (change) | `V = SE^2` |  |
| g from its 95% CI (change) | exact | g 95% CI lower (change), g 95% CI upper (change) | Hedges' g (change) | `g = (lower + upper)/2` |  |
| SE of g from its 95% CI (change) | exact | g 95% CI lower (change), g 95% CI upper (change) | SE of g (change) | `SE = (upper - lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.1 |
| d from its 95% CI (change) | exact | d 95% CI lower (change), d 95% CI upper (change) | Cohen's d (change) | `d = (lower + upper)/2` |  |
| SE of d from its 95% CI (change) | exact | d 95% CI lower (change), d 95% CI upper (change) | SE of d (change) | `SE = (upper - lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.1 |
| SMD of unstated type treated as Hedges' g | assumption | SMD (type not stated) | Hedges' g (endpoint) | `g = reported SMD` | Researcher decision |
| SMD of unstated type treated as Cohen's d | assumption | SMD (type not stated) | Cohen's d (endpoint) | `d = reported SMD` | Researcher decision |
| SE of unstated SMD treated as SE of g | assumption | SE of SMD (type not stated) | SE of g (endpoint) | `SE(g) = reported SE` |  |
| SE of unstated SMD treated as SE of d | assumption | SE of SMD (type not stated) | SE of d (endpoint) | `SE(d) = reported SE` |  |
| Morris PPC2 effect size | exact | Mean baseline (intervention), Mean post-test / endpoint (intervention), Mean baseline (control), Mean post-test / endpoint (control), SD baseline (intervention), SD baseline (control), N (intervention), N (control) | Morris PPC2 g | `g = J x [(Mpost_i - Mpre_i) - (Mpost_c - Mpre_c)] / SDpre_pooled` | Morris 2008, Organ Res Methods, eq. 8 |
| Variance of Morris PPC2 | exact | Morris PPC2 g, N (intervention), N (control), Pre-post correlation (intervention), Pre-post correlation (control) | Variance of Morris PPC2 g | `V = J^2[2(1-r)(n_i+n_c)/(n_i n_c) x (N-2)/(N-4) x (1 + d^2/(2(1-r)(n_i+n_c)/(n_i n_c))) - d^2/J^2]` | Morris 2008, eq. 25; r = N-weighted mean of group correlations |
| Events from percentage (i) | approximation | Event % (intervention), N (intervention) | Events (intervention) | `events = round(% / 100 x n)` | Reconstructed count; rounding error possible |
| Events from percentage (c) | approximation | Event % (control), N (control) | Events (control) | `events = round(% / 100 x n)` | Reconstructed count; rounding error possible |
| log OR from 2x2 table | exact | Events (intervention), N (intervention), Events (control), N (control) | log odds ratio, Variance of log OR | `lnOR = ln(ad/bc); V = 1/a + 1/b + 1/c + 1/d` | Borenstein et al. 2009, eq. 5.8-5.9 |
| log RR from 2x2 table | exact | Events (intervention), N (intervention), Events (control), N (control) | log risk ratio, Variance of log RR | `lnRR = ln[(a/n_i)/(c/n_c)]; V = 1/a - 1/n_i + 1/c - 1/n_c` | Borenstein et al. 2009, eq. 5.1-5.2 |
| Risk difference from 2x2 table | exact | Events (intervention), N (intervention), Events (control), N (control) | Risk difference, Variance of RD | `RD = a/n_i - c/n_c; V = p_i(1-p_i)/n_i + p_c(1-p_c)/n_c` | Borenstein et al. 2009, eq. 5.13-5.14 |
| log OR from reported OR | exact | Odds ratio | log odds ratio | `lnOR = ln(OR)` |  |
| log OR from its 95% CI | exact | OR 95% CI lower, OR 95% CI upper | log odds ratio | `lnOR = (ln lower + ln upper)/2` |  |
| SE of log OR from 95% CI | exact | OR 95% CI lower, OR 95% CI upper | SE of log OR | `SE = (ln upper - ln lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.2 |
| SE of log OR from p value | exact | log odds ratio, p value for OR | SE of log OR | `SE = |lnOR| / z(1 - p/2)` | Altman & Bland 2011, BMJ |
| Variance of log OR from SE | exact | SE of log OR | Variance of log OR | `V = SE^2` |  |
| SE of log OR from variance | exact | Variance of log OR | SE of log OR | `SE = sqrt(V)` |  |
| log RR from reported RR | exact | Risk ratio | log risk ratio | `lnRR = ln(RR)` |  |
| log RR from its 95% CI | exact | RR 95% CI lower, RR 95% CI upper | log risk ratio | `lnRR = (ln lower + ln upper)/2` |  |
| SE of log RR from 95% CI | exact | RR 95% CI lower, RR 95% CI upper | SE of log RR | `SE = (ln upper - ln lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.2 |
| SE of log RR from p value | exact | log risk ratio, p value for RR | SE of log RR | `SE = |lnRR| / z(1 - p/2)` | Altman & Bland 2011, BMJ |
| Variance of log RR from SE | exact | SE of log RR | Variance of log RR | `V = SE^2` |  |
| SE of log RR from variance | exact | Variance of log RR | SE of log RR | `SE = sqrt(V)` |  |
| log HR from reported HR | exact | Hazard ratio | log hazard ratio | `lnHR = ln(HR)` |  |
| log HR from its 95% CI | exact | HR 95% CI lower, HR 95% CI upper | log hazard ratio | `lnHR = (ln lower + ln upper)/2` |  |
| SE of log HR from 95% CI | exact | HR 95% CI lower, HR 95% CI upper | SE of log HR | `SE = (ln upper - ln lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.2 |
| SE of log HR from p value | exact | log hazard ratio, p value for HR | SE of log HR | `SE = |lnHR| / z(1 - p/2)` | Altman & Bland 2011, BMJ |
| Variance of log HR from SE | exact | SE of log HR | Variance of log HR | `V = SE^2` |  |
| SE of log HR from variance | exact | Variance of log HR | SE of log HR | `SE = sqrt(V)` |  |
| Cohen's d from log OR | approximation | log odds ratio, Variance of log OR | Cohen's d (endpoint), Variance of d (endpoint) | `d = lnOR x sqrt(3)/pi; V(d) = V(lnOR) x 3/pi^2` | Chinn 2000, Stat Med; Cochrane Handbook 10.6 |
| log OR from Cohen's d | approximation | Cohen's d (endpoint), Variance of d (endpoint) | log odds ratio, Variance of log OR | `lnOR = d x pi/sqrt(3); V = V(d) x pi^2/3` | Chinn 2000; Borenstein et al. 2009, eq. 7.1 |
| Pearson r from Spearman rho | approximation | Spearman rho | Correlation r | `r = 2 sin(pi rho / 6)` | Rupinski & Dunlap 1996 |
| Fisher z from r | exact | Correlation r | Fisher z | `z = atanh(r)` | Borenstein et al. 2009, eq. 6.2 |
| Variance of Fisher z | exact | Total N | Variance of Fisher z | `V(z) = 1/(N - 3)` | Borenstein et al. 2009, eq. 6.3 |
| Cohen's d from correlation r | approximation | Correlation r, Total N | Cohen's d (endpoint), Variance of d (endpoint) | `d = 2r/sqrt(1-r^2); V(d) = 4V(r)/(1-r^2)^3` | Borenstein et al. 2009, eq. 7.5-7.6 |
| Correlation r from Cohen's d | approximation | Cohen's d (endpoint), N (intervention), N (control) | Correlation r | `r = d / sqrt(d^2 + a), a = (n_i+n_c)^2/(n_i n_c)` | Borenstein et al. 2009, eq. 7.7 |
| Events from proportion | approximation | Proportion (single group), Total N | Events (single group) | `events = round(p x N)` |  |
| Logit proportion | exact | Events (single group), Total N | Logit proportion, Variance of logit proportion | `logit = ln(p/(1-p)); V = 1/e + 1/(N-e)` | Lipsey & Wilson 2001 |
| Generic effect from 95% CI | exact | Generic effect 95% CI lower, Generic effect 95% CI upper | Generic effect estimate | `estimate = (lower + upper)/2` |  |
| SE of generic effect from 95% CI | exact | Generic effect 95% CI lower, Generic effect 95% CI upper | SE of generic effect | `SE = (upper - lower) / (2 x 1.959964)` |  |
| Variance of generic effect | exact | SE of generic effect | Variance of generic effect | `V = SE^2` |  |
| Variance of MD from its SE (post) | exact | SE of mean difference (endpoint) | md_var_post | `V = SE^2` |  |
| Variance of MD from its SE (change) | exact | SE of mean difference (change) | md_var_change | `V = SE^2` |  |
| Reported g used as endpoint g | assumption | Hedges' g as reported (design not stated) | Hedges' g (endpoint) | `g_post = reported g` | Researcher-confirmed design |
| Reported SE of g used for endpoint g | assumption | SE of reported Hedges' g | SE of g (endpoint) | `se_g_post = reported SE` |  |
| Reported CI of g used for endpoint g | assumption | Reported g 95% CI lower, Reported g 95% CI upper | g 95% CI lower (endpoint), g 95% CI upper (endpoint) | `CI = reported CI` |  |
| Reported g used as change-score g | assumption | Hedges' g as reported (design not stated) | Hedges' g (change) | `g_change = reported g` | Researcher-confirmed design |
| Reported SE of g used for change-score g | assumption | SE of reported Hedges' g | SE of g (change) | `se_g_change = reported SE` |  |
| Reported CI of g used for change-score g | assumption | Reported g 95% CI lower, Reported g 95% CI upper | g 95% CI lower (change), g 95% CI upper (change) | `CI = reported CI` |  |
| Reported g used as Morris PPC2 g | assumption | Hedges' g as reported (design not stated) | Morris PPC2 g | `g_ppc2 = reported g` | Researcher-confirmed design |
| Reported SE of g used for Morris PPC2 g | assumption | SE of reported Hedges' g | SE of Morris PPC2 g | `se_g_ppc2 = reported SE` |  |
| Reported CI of g used for Morris PPC2 g | assumption | Reported g 95% CI lower, Reported g 95% CI upper | PPC2 g 95% CI lower, PPC2 g 95% CI upper | `CI = reported CI` |  |
| Variance of PPC2 g from its SE | exact | SE of Morris PPC2 g | Variance of Morris PPC2 g | `V = SE^2` |  |
| SE of PPC2 g from its 95% CI | exact | PPC2 g 95% CI lower, PPC2 g 95% CI upper | SE of Morris PPC2 g | `SE = (upper - lower) / (2 x 1.959964)` | Cochrane Handbook 6.3.1 |
| Variance of change-score g from its SE | exact | SE of g (change) | Variance of g (change) | `V = SE^2` |  |
| Approximate PPC2 variance (independent-groups formula) | approximation | Morris PPC2 g, N (intervention), N (control) | Variance of Morris PPC2 g | `V ≈ (n_i+n_c)/(n_i n_c) + g^2/(2(n_i+n_c))` | Borenstein et al. 2009, eq. 4.20; approximates Morris 2008 with r = 0.5 |
