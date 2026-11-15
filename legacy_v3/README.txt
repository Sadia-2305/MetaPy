MetaPy Unified v3.3.1 Colab bundle

This bundle combines:
- MetaPy workbook harmonisation and uncertainty reconstruction
- Researcher decision workbench v3.1.3
- Dependency-aware data preparation v3.2.1 and corrections v3.2.2
- Pairwise continuous/binary analysis engine v3.3.0

Improvements in this integration:
1. Explicit Intervention_Arm/Treatment_Arm/Experimental_Arm aliases are recognised and retained.
2. Group-specific source pre/post correlations are carried into canonical records and surfaced as the preferred primary variance inputs when present. Assumed correlation grids are labelled as separate sensitivity analyses.
3. Unit status distinguishes a single unit declared within a source worksheet from unresolved multi-unit compatibility. This is a within-workbook check, not external verification.
4. Every prepared comparison is marked Pending individual sign-off. Project-level approvals no longer imply record-level approval or final analysis readiness.

Important:
- The original uploaded workbook is not modified.
- This bundle does not perform external study verification.
- Record-level sign-off remains an explicit researcher task; do not treat pending records as approved.
- Pairwise methods only are included in the v3.3.0 analysis engine. Network, multilevel, meta-regression, diagnostic accuracy, survival, crossover and cluster analyses are not included.
- Statistical validation against independent reference implementations remains necessary before publication use.

Validation performed for this integration build:
- 22 variance-reconstruction tests passed.
- 6 reconstruction-integration tests passed.
- 15 researcher-decision-engine tests passed.
- 26 pairwise-analysis core tests passed.
- A targeted smoke test confirmed Intervention_Arm is retained and r_prepost_I/r_prepost_C are carried as separate source correlations.
Note: the legacy combined data-preparation regression runner still contains two v3.2.1 expectations that do not match the v3.2.2 engine, plus fixture-dependent tests. It is not presented as passing. These need reconciliation in a later maintenance pass.
