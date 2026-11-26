# Independent review of the software-repair pilot

The original [pilot preregistration](preregistration.md) defines the learning contrast. The [fixed-patch protocol](reviews/grader-calibration-protocol-2026-11-26.md) fixes the new grader-control matrix and its sensitivity analysis. The audit uses saved outputs and CPU test execution.

1. Reconstruct the frozen human cohort from the pinned dataset. Check the synthetic manifest, source mutation hashes and saved clean/mutant test maps.
2. Trace each collected action to its episode and saved likelihoods. Recompute the length-weighted advantages. Verify the selected and initial adapter/configuration bytes, base revision, recovery accounting and stopping deviations.
3. Reparse original test output with the locked release. Preserve every official score and unscored attempt. Identify missing tests, parser artifacts, expected-pass failures and incomplete pairs.
4. Execute the prospective gold and empty controls. Report every task. Compute the finite-cohort range and the historical sensitivity contrast on the prespecified passing-control subset.
5. Propagate conclusions to the paper, dashboard, README, reports and adapter card. Verify the rendered and public artifacts against the evidence. [Claim review](reviews/claim-review-2026-11-26.md)
