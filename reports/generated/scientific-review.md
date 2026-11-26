# Scientific review of the software-repair pilot

2026-11-26

The primary learning hypothesis was not tested. The research run recorded zero optimizer steps and selected an adapter identical to initialization.

## Collection and training exposure

The 2 completed groups contained 4 injector episodes. 4 reached the tool budget with no declared test targets. 4 episodes had zero length-centered advantage. The original action stream matched 138 saved collection actions. The effective messages for 128 distinct training prompt token sequences contained 0 verbatim complete held-out reference or test patches. Partial overlap, memorization and pretraining exposure remain unknown.

## Human cohort and missing outcomes

The frozen selection took 32 tasks from 38 eligible rows of the 500-row pinned dataset. Both conditions were scored on 21 tasks, base only on 0, final only on 8, and neither on 3. Assigning every missing binary outcome its extremes gives a possible net change of -11 to +3 resolutions out of 32 (-34.38 to +9.38 percentage points). These endpoints describe possible completions of this finite cohort. The paired bootstrap describes only the observed pairs.

## Replay of historical scoring

The locked scorer reproduced all 50 archived human grades from stdout. 50 of the submitted patches were empty. 14 scored submissions recorded zero generated tokens and 15 recorded an agent context-limit error. A recorded failure can therefore include a task for which the agent could not generate a repair. The PASS_TO_PASS failure category contained 208 entries across 9 submissions on 5 tasks. Parsed states among those entries were {"FAILED": 208}. The FAIL_TO_PASS failure category contained 122 entries, with parsed states {"FAILED": 100, "MISSING": 22}. Expected IDs included a literal pytest progress marker on 2 tasks. 4 of the 4 affected submissions also had a separate repair-required failure. Test-level classifications and alias matches are retained in the per-test archive.

## Fixed-patch grader calibration

The grader resolved the published repair and rejected the empty patch on 25/32 tasks. The remaining classifications were 6 failed and 1 indeterminate. Gold repairs resolved on 25/32 planned tasks. Empty submissions resolved on 0/32. These CPU measurements characterize the pinned grading environment. A passing pair validates only these two inputs.

## Sensitivity to grader discrimination

Of 25 tasks passing calibration, 16 had both historical policy scores and 9 lacked at least one. The final-minus-base difference is +0.0 percentage points (paired percentile interval [+0.0, +0.0]). The restriction was specified before control execution and after the pilot outcomes were known. The full original analysis retains every admitted score.

## Synthetic diagnostic cohort

The private records and public manifest agree on 16 mutations across 16 distinct functions. The saved clean suite sizes were 723. The policy evaluation stream contains 0 scored synthetic outcomes.

## Where reference repairs failed

Among the completed gold controls, the grader rejected 6 in psf/requests. On 2 of these tasks all repair-required tests were accepted, but expected-pass failures still blocked resolution. Original stdout contained a DNS or network-unreachable message on 6/6 unresolved gold controls. This retrospective text check identifies a concrete environmental limitation. The data do not isolate the cause of every failed test.

## Protocol changes and stopping

The interrupted original pilot records match the recovery amendment. The final observed worker overrun was 45.619 seconds. The stopping diagnostic identified 9 calls continuing past chat EOS in the unscored held-out episode. That exposure preceded the stopping correction. The likelihood gate was also amended after inspection of the same preflight sample.

## All frozen human tasks

| Frozen task | Historical base | Historical final | Gold | Empty | Discrimination |
| --- | --- | --- | --- | --- | --- |
| pytest-dev__pytest-5631 | unresolved | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-7080 | unresolved | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-6386 | unresolved | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-7982 | unresolved | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-4970 | unresolved | unresolved | resolved | unresolved | pass |
| psf__requests-2931 | unresolved | unresolved | unresolved | unresolved | fail |
| pytest-dev__pytest-7432 | unresolved | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-5809 | unresolved | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-7521 | unresolved | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-6903 | unresolved | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-5262 | unresolved | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-10356 | unresolved | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-4661 | unresolved | unresolved | unscored | unresolved | indeterminate |
| pytest-dev__pytest-10051 | unresolved | unresolved | resolved | unresolved | pass |
| psf__requests-1142 | unresolved | unresolved | resolved | unresolved | pass |
| psf__requests-1921 | unresolved | unresolved | unresolved | unresolved | fail |
| pytest-dev__pytest-6202 | unresolved | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-4551 | unresolved | unresolved | resolved | unresolved | pass |
| psf__requests-1766 | unresolved | unresolved | unresolved | unresolved | fail |
| psf__requests-5414 | unresolved | unresolved | unresolved | unresolved | fail |
| pytest-dev__pytest-7205 | unresolved | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-6528 | unscored | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-5787 | unscored | unresolved | resolved | unresolved | pass |
| psf__requests-6028 | unscored | unresolved | resolved | unresolved | pass |
| psf__requests-2317 | unscored | unresolved | unresolved | unresolved | fail |
| pallets__flask-5014 | unscored | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-7490 | unscored | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-7571 | unscored | unresolved | resolved | unresolved | pass |
| pytest-dev__pytest-6197 | unscored | unresolved | resolved | unresolved | pass |
| pylint-dev__pylint-8898 | unscored | unscored | resolved | unresolved | pass |
| psf__requests-1724 | unscored | unscored | unresolved | unresolved | fail |
| pytest-dev__pytest-7236 | unscored | unscored | resolved | unresolved | pass |

## Replayed control test entries

Counts include available stdout from unscored attempts.

| Patch | Required category | Official failure entries | Parsed state counts |
| --- | --- | ---: | --- |
| empty | FAIL_TO_PASS | 80 | {"FAILED": 69, "MISSING": 11} |
| empty | PASS_TO_PASS | 166 | {"FAILED": 166} |
| gold | FAIL_TO_PASS | 25 | {"FAILED": 25} |
| gold | PASS_TO_PASS | 165 | {"FAILED": 165} |

- pylint-dev__pylint-4661 (gold), infra_error: network_unreachable. Observed log markers: temporary failure in name resolution, no module named 'appdirs'.

## Evidence and scope

The [calculated review and complete task table](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/generated/scientific-review.md) accompany this assessment. The reconstruction starts with original run streams, checkpoint bytes, private frozen task records, official test stdout and new CPU controls. It checks upstream source bytes independently. Previous reviews and generated narratives supply no measurements. Historical prompt transcripts contain model-authored assertions. A transcript establishes what was said. Tool events, process output, grader reports and saved artifacts establish which actions and outcomes were observed. [Original records](https://huggingface.co/BurnyCoder/multi-agent-swe-rl-qwen3.5-2b-pilot/tree/9ab275d4811534a6c7912d80f790810bacd9323a/run-records/pilot-recovery) · [Independent evidence](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/generated/evidence.json) · [Per-test archive](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/generated/oracle-tests.json.gz) · [Upstream byte comparisons](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/generated/source-checks.json)

| Claim examined | Finding and scope | Direct evidence |
| --- | --- | --- |
| The pilot tested whether learning improves human repository repair. | The planned intervention required a trained final policy. The completed groups contributed no nonzero-advantage action. Hashes of the initial and selected adapter weights and configuration agree, as does the pinned base identity. The primary learning hypothesis remains untested. The raw `trained` condition and the registered interval-rule label retain their historical meanings. | [Preregistration](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/0c9552c/reports/preregistration.md) · [Collection records](https://huggingface.co/BurnyCoder/multi-agent-swe-rl-qwen3.5-2b-pilot/tree/9ab275d4811534a6c7912d80f790810bacd9323a/run-records/pilot-recovery/collection_groups) · [Reconstruction](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/generated/evidence.json) |
| An observed tie establishes zero effect or equivalent performance. | A task is paired only if both conditions have an admitted boolean score. When all observed differences are zero, every task bootstrap resample is zero. This supplies no evidence of population equivalence, a precise success rate, robustness across seeds or a causal training effect. No equivalence margin or powered equivalence test was specified. | [Paired percentile procedure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) · [Frozen cohort and calculated bounds](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/generated/evidence.json) |
| Missing records can be treated as failed repairs. | Deadline and infrastructure statuses carry no binary outcome. The finite-cohort lower bound assigns every missing final outcome failure and every missing base outcome success. Reversing those assignments gives the upper bound. Each endpoint is attainable without assumptions about missingness. Time allocation and manifest order can select which cases finish. The cohort was restricted to eligible repositories and was never a random sample of all software bugs. | [Run budgets and evaluations](https://huggingface.co/BurnyCoder/multi-agent-swe-rl-qwen3.5-2b-pilot/tree/9ab275d4811534a6c7912d80f790810bacd9323a/run-records/pilot-recovery) · [Worst-case bounds](https://doi.org/10.1080/01621459.2000.10473902) · [Frozen dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified/tree/78f471bf655a3137b2e8a75af1501690ec009ec3) |
| Self-play rewards and token credit implement the cited methods. | The injector reward follows SSR's piecewise function. The local optimizer centers returns using generated-token lengths, uses stored behavior probabilities for the denominator and applies asymmetric clipping. Team-level credit across separately recorded contexts and the collection-capacity normalization are local implementation choices. Equal returns imply zero centered advantages for these groups. Convergence, stability under repeated updates and learning utility remain unmeasured. | [SSR equations](https://arxiv.org/html/2512.18552v3#S2.SS3) · [CWM §5.2](https://arxiv.org/html/2510.02387v1#S5.SS2) · [CWM Appendix C](https://arxiv.org/html/2510.02387v1#A3) · [Local objective](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/src/multi_agent_ssr/learning.py) |
| Local artifact validation reproduces SSR exactly. | SSR's stated consistency criterion requires an exposing test to pass after weakening. The local predicate requires a strict subset of the induced failing IDs and permits removal of an exposing test. The repair oracle restores the original tests and requires identical expected IDs and all passes. Trusted pytest execution, file-level reversions and the inverse weakening patch supplied to repair agents define this adaptation. No collected valid artifact supports an empirical claim for these choices. | [SSR §§2.3–2.4](https://arxiv.org/html/2512.18552v3#S2.SS3) · [Validator](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/src/multi_agent_ssr/validation.py) · [Removal example](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/tests/test_validation.py) |
| Delegation, curriculum, or a proposed control caused comparative performance. | Asynchronous child sessions and serialized model calls are implemented. The records distinguish tool execution from text that resembles a tool call. Neither a trained single-agent contrast nor independently trained seed replicates were completed. No inference-time or training-time delegation effect, curriculum effect, or advantage over another training method can be estimated. | [Agent harness](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/src/multi_agent_ssr/agents.py) · [Control designs](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/docs/controls.md) · [Event stream](https://huggingface.co/BurnyCoder/multi-agent-swe-rl-qwen3.5-2b-pilot/tree/9ab275d4811534a6c7912d80f790810bacd9323a/run-records/pilot-recovery) |
| Held-out tasks were isolated and the model generalizes. | The repository split and hash-ordered human cohort are reproducible. The private human rows equal the selected rows of the upstream Parquet file. The leakage check searches complete reference/test patch strings in recorded effective messages for completed research collection actions. It cannot rule out partial overlap, source similarities, information in pretraining or effects of developer exposure to held-out behavior. The synthetic preparation verifies test-detected mutations in distinct functions. A prepared mutation supplies no policy success measurement. | [Split and evaluation manifests](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/tree/main/manifests) · [Human prompt and privileged grading path](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/109dfa67740ff9f0260d22be9407d047d90565f7/src/multi_agent_ssr/evaluation.py) · [Frozen model card](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md) · [Calculated exposure and coverage](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/generated/evidence.json) |
| The preflight proves optimization correctness or holds out all diagnostics. | The trainer-forward tolerance was revised after inspecting the same saved preflight sample. Cached replay agreement and in-range ratios on that sample are engineering checks. They do not measure numerical drift on new contexts. A chat-EOS change used an unscored held-out episode, which was retained as a protocol diagnostic and retried. The recovery amendment also modified continuous wall-clock accounting. These are deviations from a fully untouched procedure. No evidence here identifies a BF16 mechanism or demonstrates score-driven selection. | [Likelihood amendment](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/commit/d90c86d) · [EOS amendment](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/commit/0c9552c) · [Recovery amendment](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/commit/46b51c7) · [Original diagnostic records](https://huggingface.co/BurnyCoder/multi-agent-swe-rl-qwen3.5-2b-pilot/tree/9ab275d4811534a6c7912d80f790810bacd9323a/run-records/pilot-recovery/preflight) |
| Official resolution or a PASS_TO_PASS failure establishes real-world patch correctness or an agent-caused regression. | Resolution is the locked harness's predicate on designated tests. Its failure lists can include missing IDs. Its matching tolerates some truncated parameter IDs, and a skipped expected-pass test can count as maintained. Replaying stdout checks scoring reproducibility. Gold/empty controls check discrimination for two known submissions per task under the historical restrictions. Neither procedure establishes the behavior of arbitrary fixes, all tests, or other environments. | [Locked grader](https://github.com/SWE-bench/SWE-bench/blob/ca6e4e0d252f32f8762625b73575d5dee49d0a5a/swebench/harness/grading.py) · [Python parsers](https://github.com/SWE-bench/SWE-bench/blob/ca6e4e0d252f32f8762625b73575d5dee49d0a5a/swebench/harness/log_parsers/python.py) · [Registered control criteria](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/d9bbab3/reports/reviews/grader-calibration-protocol-2026-11-26.md) |
| Prior work establishes this system's novelty, transfer or performance. | SSR supplies the research task. CWM supplies the underlying token objective. OpenCode supplies a reference for child-session tools. Anchored Self-Play's function-level bug-source experiments motivate separate human evaluation here. None measures this shared-adapter repository experiment or establishes priority for its combination. | [SSR](https://arxiv.org/html/2512.18552v3) · [CWM](https://arxiv.org/html/2510.02387v1) · [OpenCode source](https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts) · [Anchored Self-Play §4.4](https://arxiv.org/html/2607.03523v1#S4.SS4) |

## What the public files represent

The README, generated experiment summaries, paper, figures, dashboard data, and Hub card expose the same recorded denominators and evidence-gated conclusion. The review table retains every frozen human task. The machine-readable reconstruction supplies per-attempt provenance, while the compressed archive supplies expected IDs, normalized parser keys, raw parsed states and official classifications. The original scorer caches and calibration stdout are retained locally. A public reader can recompute the displayed summaries from exported evidence or reproduce fixed-patch grading from the pinned public dataset and images. [Report generator](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/src/multi_agent_ssr/reporting.py) · [Control runner](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/17a3558/src/multi_agent_ssr/grader_calibration.py)

Configuration, lock, split and image manifests describe settings and identities. Source modules and tests establish implemented behavior under their stated inputs. A test fixture, a planned run, a generated figure, or two copies of a summary do not supply independent performance measurements. Repository instructions describe intended work. The original preregistration, author-supplied protocol, third-party notices, historical verification files and previous dated reviews retain their original scope and dates. Project source notices have been compared to their pinned upstream bytes. [Source guide](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/docs/sources.md) · [Notices](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/tree/main/docs/licenses) · [NeurIPS reporting guidance](https://neurips.cc/public/guides/PaperChecklist)
