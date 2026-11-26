# CPU calibration of the human-bug grading oracle

Libor "Burny" Burian · 2026-11-26

This is a prospective protocol for new control measurements and a retrospective audit of the November pilot. Historical scores, empty submissions and expected-pass failures were known when this protocol was written. It adds no policy observations. The original [preregistration](../preregistration.md) governs the pilot.

## Question and inputs

Can the pilot grader distinguish the published human repair from an empty submission on each frozen task? Use every ID in the [human cohort](../../manifests/evaluation.json), its original base commit and [image digest](../../manifests/image_pins.json). Verify the private grading inputs against the pinned dataset. Each task contributes an empty UTF-8 patch and the dataset's exact human patch. Reference fixes are accessed exclusively by the host-side calibration grader. [Pinned dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/tree/78f471bf655a3137b2e8a75af1501690ec009ec3)

## Execution and exclusions

Use the locked SWE-bench grader, the pilot Docker wrapper, its network and resource restrictions, and a 300-second test timeout per fresh container. Process tasks in manifest order, with up to two independent workers. Alternate the order of the two conditions by task index. Each invocation gets its own official log directory. Preserve timestamps, inputs and output hashes, per-test classifications, process failures and raw logs outside the pilot. Resume completed controls idempotently. Retain an interrupted attempt, then perform its replacement in a fresh directory. An admitted grade or a recorded grader failure is final for this calibration. Report missing evidence explicitly. [Pilot grading wrapper](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/src/multi_agent_ssr/evaluation.py) · [Upstream grader](https://github.com/SWE-bench/SWE-bench/blob/ca6e4e0d252f32f8762625b73575d5dee49d0a5a/swebench/harness/grading.py)

## Analysis fixed before the controls

- The primary calibration result is the number of tasks with a resolved gold patch and an unresolved empty patch, divided by all planned tasks. Two completed booleans in any other combination fail discrimination. Missing or infrastructure-corrupted results are indeterminate. Publish the outcome of both controls for every task.
- Report expected-pass failures separately from repair-required failures. Resolve test names using the locked parser's alias rules. Distinguish explicit failures, errors, skipped tests, absent parsed IDs and benchmark metadata artifacts. A failed control establishes a limitation of that task in this execution. Attribution requires evidence beyond the aggregate score.
- Retain the original complete-pair estimate and interval. Also calculate the same descriptive paired estimate on the prespecified subset of tasks passing both controls, using only historical policy outcomes. Report the subset's numerator, denominator and incomplete coverage. Report the full-cohort attainable range under all binary completions of the missing observations. These supplementary analyses address measurement and missingness.
- Determine the primary learning-hypothesis status from the delivered intervention, checkpoint identity, paired evidence and original decision rule. A zero-step, byte-identical selected adapter yields `not_tested_no_update` irrespective of either control or the mechanical interval endpoint. Synthetic tasks and deferred training controls retain their own coverage.
