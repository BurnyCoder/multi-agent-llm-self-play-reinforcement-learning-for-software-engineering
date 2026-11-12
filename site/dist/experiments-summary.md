# Experiments recorded in this run

Recorded failure history (retained after later successful phases):

- 2026-11-05T21:30:15.600645+00:00, **preflight**: RuntimeError: Generation/scoring log probabilities disagree: 0.152308.
- 2026-11-05T21:58:00.073528+00:00, **preflight**: RuntimeError: Generation/scoring log probabilities disagree: 0.152151.
- 2026-11-06T04:49:29.416350+00:00, **pilot**: Worker and supervisor remained alive 45.619 seconds past the persisted wall-clock deadline; the worker process group was externally terminated. The interrupted task has no admitted score. The overrun's cause is not established..

## Interrupted attempt and recovery amendment

Prior attempt runs/pilot has recorded status preflight_failed. Its original wall-clock deadline was 2026-10-30T15:31:24.450111+00:00; original deadline preserved: True. Recorded prior failure: preflight: BrokenPipeError: [Errno 32] Broken pipe. The prior attempt's recorded elapsed time was 479.063 seconds. The recovery amendment charges 600 seconds for that attempt and allocates 28200 seconds to this recovery, within 28800 seconds of total active allocation. The amendment excludes the offline interval; it does not erase the expired original attempt. Saved prior artifacts: model outputs 0, evaluations 0, checkpoints 0. Recorded reason: User requested recovery after OS restart. Original preflight failed with BrokenPipeError; original wall-clock deadline expired. Offline interval is excluded only by the dated preregistration amendment.

## Preflight likelihood diagnostics

Check at 2026-11-05T21:58:00.067+00:00: 101 sampled action tokens; maximum cached-replay log-probability difference 0; maximum trainer-versus-sampling log-probability difference 0.152151. Pre-update scoring/behavior ratio range [0.858858, 1.10626]; 0/101 tokens outside [0.8, 1.25]. Recorded trainer guard: absolute log-probability difference limit 0.05; cached-replay difference limit 0.05. Check at 2026-11-05T22:16:41.480+00:00: 101 sampled action tokens; maximum cached-replay log-probability difference 0; maximum trainer-versus-sampling log-probability difference 0.151829. Pre-update scoring/behavior ratio range [0.859135, 1.06795]; 0/101 tokens outside [0.8, 1.25]. Recorded trainer guard: preupdate_ratios_inside_clip_interval with interval [0.8, 1.25]; cached-replay difference limit 0.05. CWM retains sampled behavior probabilities in the denominator [2]. Upstream Qwen analysis attributes some BF16 cached/full-forward differences to rounding [10]. Each check is a single-prompt software diagnostic; it does not bound drift in training contexts or establish training quality.

Sources: [CWM Appendix C](https://arxiv.org/html/2510.02387v1#A3) and [Qwen numerical analysis](https://github.com/huggingface/transformers/issues/46190).

## Unscored protocol diagnostics

At 2026-11-05T22:49:05.240382+00:00, episode 97baaad7b45b4308bd6fed5b9fdcc085, task pytest-dev__pytest-5631, code e49e7442f0243efcbd00879327525c4f04639a0b: chat_eos_mismatch; status interrupted_unscored. Recorded model calls: 9; generated tokens: 2094; calls continuing after chat EOS: 9. Source-model EOS IDs: [248044]; tokenizer EOS ID: 248046. Stopped at 2026-11-05T22:47:26.520168+00:00. Recorded reason: Pinned model text config stops only on endoftext; tokenizer ends assistant turns with im_end. Model generated im_end then continued fabricating user/tool observations. No completed benchmark outcome or research update preceded correction. Incomplete streaming call, if any, remains in model.jsonl but lacks a completed action record.. Correction: Stop on the deduplicated union of model and tokenizer EOS IDs. Retain stop token and its behavior log probability. No post-hoc truncation of saved trajectories; retry same task from fresh base initialization. Both evaluation conditions and all training agents use the corrected stop contract..

These are unscored harness diagnostics, excluded from benchmark success and training. Their recorded calls and tokens remain visible as physical model usage; action-event counters retain the corresponding diagnostic events without adding these summaries again. Model-authored role or observation text is generated text; only recorded harness tool events establish actual tool execution.

| Bug source | Condition | Planned | Attempted | Resolved / scored | Rate | Unscored recorded | Unrecorded |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| human | base | 32 | 22 | 0 / 21 | 0.0% | 11 | 0 |
| human | final condition | 32 | 30 | 0 / 29 | 0.0% | 3 | 0 |
| synthetic | base | 16 | 0 | Not evaluated | Not evaluated | 16 | 0 |
| synthetic | final condition | 16 | 0 | Not evaluated | Not evaluated | 16 | 0 |

Recorded research optimizer steps: 0. The initial and selected adapter weights and configurations have identical SHA256 digests and the same pinned base identity. This is two evaluations of unchanged weights, not an estimated self-play training effect. The planned learning hypothesis was not tested by this unchanged-policy comparison. The raw trained label and mechanical preregistered interval decision are preserved for provenance.

### Human bugs

Across 21 paired tasks, the final-minus-base resolution difference was +0.0 percentage points (95% paired bootstrap interval [+0.0, +0.0]). Improved: 0; regressed: 0; unchanged: 21. The mechanical preregistered interval rule contradicts the predicted improvement. This rule alone does not establish that the planned learning intervention occurred. The bootstrap interval is degenerate; it does not establish equivalence or rule out a population effect.

### Synthetic bugs

Not evaluated: no tasks have completed outcomes for both conditions.

## Recorded training outcome

Completed collection groups: 2 (with saved training_group outcomes); valid bugs / injector episodes: 0 / 4; recorded repair attempts: 0; zero-advantage episodes: 4; optimizer steps: 0. Collection groups with sampled actions but only zero advantages: 2. Subtracting the length-weighted group mean turns equal returns into zero advantages. These groups had no nonzero-advantage action tokens and recorded no learning update. Saved invalidity reason counts across all logged artifact checks, including incomplete collections: {"At least one test target is required": 4}.

## Deferred controls

| Deferred condition | Purpose | Result |
| --- | --- | --- |
| single-agent self-play | Separate delegation from self-play training | Not run |
| repair-only training | Control for updating the injector | Not run |
| procedural-mutation training | Control the training bug distribution | Not run |
| no-delegation evaluation | Measure inference-time delegation dependence | Not run |
| seed sensitivity | Measure training-seed variation | Not run |

## Recorded diagnostics

| Recorded diagnostic | Value |
| --- | ---: |
| episodes | 57 |
| child sessions | 7 |
| episodes with children | 7 |
| delegation rate | 0.122807 |
| delegation calls | 8 |
| generated tokens from research agents (excludes preflight) | 64920 |
| generation calls by research agents (excludes preflight) | 1322 |
| artifact validation checks (accepted and rejected) | 4 |
| accepted valid artifacts | 0 |
| invalid artifacts | 4 |
| invalid artifact rate | 1 |
| optimizer steps | 0 |
| recorded training groups | 2 |
| peak allocated bytes | 5136889856 |
| evaluation seconds | 18533.5 |
| training group seconds | 995.584 |
| PASS_TO_PASS failure entries (causal attribution unverified) | 208 |
| scored rows with PASS_TO_PASS reports | 50 |

## Runtime interface and context diagnostics

This post-hoc descriptive analysis counts 1242 actual tool_finished events across recorded phases, excluding protocol-diagnostic episodes. Of these, 426 have result.error values, including 316 ValueError unsafe-path rejections. Generated text and nonzero command/test exit codes alone are not counted as tool errors. Evaluation agent_error counts are separated by source/condition after the report's task deduplication, excluding protocol diagnostics: human/base: 6 ContextLimitError records among 32 retained rows (6 with completed scoring); human/trained: 10 ContextLimitError records among 32 retained rows (9 with completed scoring); synthetic/base: 0 ContextLimitError records among 16 retained rows (0 with completed scoring); synthetic/trained: 0 ContextLimitError records among 16 retained rows (0 with completed scoring). Completed scoring can still be recorded after an agent context error.

Native read/edit path arguments require repository-relative paths [11]. The frozen prompt says Work in /repo without explicitly stating that read/edit path arguments must be relative [12]. This interface limitation and the fixed context capacity are potential confounders when interpreting repair outcomes. These counts do not establish a causal explanation for every unresolved result. Prompts, API semantics, and budgets remain fixed after held-out scoring; this analysis does not adapt them to observed benchmark outcomes.

Sources: [pinned path guard](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/src/multi_agent_ssr/sandbox.py#L29) and [pinned prompt](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/src/multi_agent_ssr/agents.py#L146).

## Official test failures and empty patches

The saved official reports contain 208 PASS_TO_PASS failure entries across 9 scored rows. Of those rows, 9 submitted an empty patch. Across all 50 scored human rows, 50 saved patches are empty. These are failures of tests designated as passing in benchmark metadata, not established regressions caused by agent edits. The grader's failure category can also include tests missing from parsed output. The cause of the failures is unverified; no clean-environment re-evaluation was performed in this audit. The official scores remain unchanged, with this limitation on their interpretation. This is a post-hoc descriptive check of saved reports and submitted patches.

Source: [SWE-bench scoring](https://www.swebench.com/SWE-bench/guides/evaluation/) and the saved per-task reports in [summary.json](summary.json).



## Network and experiment design

| Condition | Architecture | Training in this experiment | Objective | Data and domain |
| --- | --- | --- | --- | --- |
| base | Pinned base plus the initial shared language-model LoRA adapter | Frozen initialization | Frozen comparison | Held-out repository repair |
| final condition (raw: trained) | Same adapter architecture; identity: unchanged | 0 recorded optimizer steps | Configured CWM-style objective with SSR team returns | Configured self-play on training repositories; held-out repository repair |

## Terms in plain language

- **Self-play:** a training design in which a model generates challenges and can learn from attempts to solve them; a particular run may produce no learning update.
- **Shared policy:** every primary agent and subagent uses the same model parameters.
- **LoRA:** small trainable adapter matrices modify a frozen base model.
- **Oracle:** executable tests that determine whether the required behavior is satisfied.
- **Paired comparison:** compare the two policies on the same task before averaging differences.
- **Bootstrap interval:** uncertainty estimated by repeatedly resampling the observed tasks.


The base condition uses the initial adapter; the raw trained label denotes the final selected condition. Human bugs come from benchmark issues; synthetic bugs are independently generated diagnostic mutations. Resolution requires passing the executable oracle. An unscored task has no valid completed outcome. The larger controlled study and seed sensitivity remain unexecuted.
