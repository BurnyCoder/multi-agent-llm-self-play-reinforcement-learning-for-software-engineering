# Preregistration: local multi-agent SSR pilot

Author: Libor "Burny" Burian. This document and the source/data/model manifests must be committed and pushed before any LLM execution. CPU unit tests, container checks, and policy-independent dataset validation are preparation, not model-performance results.

## Question and hypothesis

Can one local Qwen3.5-2B policy, trained through bug injection and repair with asynchronous subagents, improve repository repair on held-out human bugs? The primary directional hypothesis is that its final checkpoint resolves more fixed human-bug tasks than the unchanged initialization, using the same agent harness and inference budget. This pilot does not isolate the causal benefit of subagents from the benefit of RL; a trained single-agent control is deferred.

The shared model, task construction and reward design build on [Self-play SWE-RL](https://arxiv.org/html/2512.18552v3). The clipped objective follows [CWM Appendix C](https://arxiv.org/html/2510.02387v1#A3). This is an adaptation, not a numerical reproduction of their large-scale experiments. The novel implementation choice being tested is a dynamically delegated team in each role with a single tied adapter and complete action-token credit accounting.

## Frozen design

The executable settings are in `configs/pilot.toml`; exact model, repository, and evaluation revisions are in `manifests/`. Their committed contents, rather than a prose copy of their values, are authoritative. Training uses only more-itertools and boltons snapshots. Cachetools is reserved for development; humanize and all human-benchmark repositories are excluded from policy training. No human bug/fix examples, gold fixes, evaluation test patches, external teacher LLM, or remote inference service enter training.

Both primary agents can asynchronously launch general or read-only explore subagents and continue their own work. Children get fresh contexts, may receive follow-ups, and return completion notifications. There is one delegation level and a shared team budget. GPU inference requests serialize through the same model; tool execution and session progress overlap. No optimizer step occurs while any rollout worker is running. Scheduling and the complete exact inference inputs are logged, since asynchronous scheduling is an additional reproducibility limitation.

The source-only bug patch and test-only weakening patch are independently validated. Selected clean tests must all pass, include at least five tests, and have at least one induced failure plus one remaining pass after the bug. Weakening must remove at least one exposed failure. Reverting each changed source file must repair at least one induced failure. Repairs are graded in fresh sandboxes with original test/configuration files. Missing tests and parser/collection errors are never successes.

Python/pytest execution and parsing are trusted harness components, unlike SSR's agent-authored runners/parsers. The pilot uses compact bounded prompts and code edits; it does not reproduce SSR's original model, context length, asynchronous multi-GPU learner, history-guided curriculum, or full evaluation suite. The model's original weights, embeddings, output head and vision tower remain frozen; one language-model LoRA adapter is trained and shared by all roles.

## Learning objective and credit assignment

For an invalid injector artifact the return is -1. For a valid artifact with empirical repair success rate s, the return is -0.8 at s=0 or s=1 and otherwise 1-1.8s. Each solver team gets +1 if all expected oracle tests pass and -1 otherwise. One failed repair per original bug may become a validated second-order task; injector difficulty uses only first-order repair attempts. Second-order tasks contribute solver gradients only.

Within each identical starting task and team, compute the mean return weighted by the number of unique generated assistant tokens. Subtract that mean without dividing by a standard deviation. Apply the resulting team advantage to every actual root/child generation from that episode. Copied child responses in parent observations never become additional actions. Keep each call's original context/token IDs and sampling log probabilities rather than flattening the session tree.

Use token PPO ratios clipped to [0.8, 1.25], a fixed maximum collection token capacity, no KL term, and two optimizer passes. Microbatch one call at a time, accumulating gradients before each pass's optimizer step. AdamW betas are (0.9, 0.99), weight decay zero, and gradient clipping 1.0. Zero-advantage groups make no update. Every completed group, including groups with no updates, is recorded. Interrupted or infrastructure-corrupted groups are not optimized; their traces are retained and counted separately.

## Evaluation and analysis

Primary cohort: the 32 tasks with the lowest SHA-256 hashes of instance IDs among SWE-bench Verified tasks from requests, Flask, pylint and pytest, at the pinned dataset revision. The frozen ID list is committed before any model call. The official SWE-bench grading oracle is used with local container privilege/network restrictions. This deliberately small, repository-restricted cohort is not a full SWE-bench score.

Secondary cohort: 16 deterministic validated procedural mutations in humanize, with at most one accepted mutation per original function. The public manifest records IDs, function clusters, revision and patch hashes; private manifests retain patches and exact oracle outcomes. These diagnostic bugs expose ordinary existing tests; they do not use training's test-weakening construction. Selection uses syntax/executable validity only, never either model's repair performance.

Evaluate base and final checkpoints once per task under identical limits, sampling settings and delegation availability. Pair results by original task ID, separately for human and synthetic sources. Report the difference in resolved fraction, a deterministic 95% paired percentile bootstrap interval using 10,000 task resamples, and raw paired outcomes. Repository dependence and the single training seed limit generalization. If all paired differences coincide, flag the bootstrap's degeneracy and do not interpret its zero width as certainty or equivalence.

A positive lower interval bound supports the directional hypothesis on this fixed cohort. An interval spanning zero is inconclusive. An upper interval bound at or below zero contradicts the predicted improvement, subject to the small-sample/degeneracy limitations. Report all negative outcomes. Do not select checkpoints, tasks, prompts or hyperparameters using held-out repair scores. The selected checkpoint is the last completed training group, even if earlier results would look better.

Policy budget exhaustion and malformed actions are model failures when the resulting patch can be graded. Infrastructure errors and deadline omissions have null success, are reported separately, and are not silently converted into failures or removed from coverage summaries. Report planned count, attempted count, scored count, paired count and all exclusions. Do not silently replace unavailable tasks. Primary paired estimates use complete pairs; incomplete coverage is a limitation, not evidence for improvement.

Additional descriptive measures: valid-artifact fraction, invalidity reasons, repair success, pass-to-pass regressions, generated tokens, actual child calls, concurrency timelines, optimizer steps, gradients/clipping, peak memory, and elapsed time. Compare human and synthetic trends because [Anchored Self-Play](https://arxiv.org/html/2607.03523v1) demonstrates that synthetic gains need not transfer to human bugs. No synthetic-only gain establishes the primary hypothesis.

## Budget, stopping, and publication

The first model invocation starts one persisted eight-hour wall-clock budget on the single GPU. This conservative cap includes model loading/compilation, preflight, baseline evaluation, training and final evaluation. It cannot be reset by restarting the process. Baseline evaluation may use at most `(remaining_seconds - 1800) / 2.25` seconds, preserving a minimum training allocation and a final-evaluation margin; tasks omitted by this ceiling remain explicitly unscored. Before training, reserve the greater of two hours or 1.25 times measured baseline evaluation duration. Stop launching groups when the measured group duration cannot fit before the reserve. Stop likelihood computation at the reserve boundary; if an optimization pass is incomplete, restore the last completed adapter/optimizer checkpoint before final evaluation. Nonfinite gradients, GPU memory failure, incompatible likelihood accounting, or corrupted checkpoints stop training and produce a failure report; never silently switch model or precision.

Preflight's synthetic optimizer update is a software check only. Its adapter is labeled `preflight_only` and is not the research initialization. The actual pilot starts from a fresh base adapter. Publish the first run and model status before any seed/hyperparameter study. One training seed cannot establish robustness. If no informative group yields an update, explicitly report no learned checkpoint rather than describing an unchanged adapter as trained.

Larger single-agent, repair-only, procedural-mutation, no-delegation and seed studies are provided as commands/configurations, but are outside this first execution. Their results must remain marked not run. All empirical report numbers, tables and figures are generated from saved records. Preserve setup failures, amendments and negative results; amend this preregistration with a timestamp and explanation before any changed experiment, without erasing the original.
