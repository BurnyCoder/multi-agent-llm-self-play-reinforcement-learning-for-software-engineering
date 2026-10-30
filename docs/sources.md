# Sources, reuse, and interpretation

This project is a local multi-agent adaptation of existing bug/repair self-play research. The novelty claim is limited to the implemented combination until controlled experiments establish more.

| Component | Source | Use and boundary |
| --- | --- | --- |
| Repository bug injection and repair | [Self-play SWE-RL, v3](https://arxiv.org/html/2512.18552v3) | Bug patch, test weakening, oracle restoration, consistency checks, curriculum rewards, and higher-order bugs. The Python/pytest runner is an adaptation, not SSR's general artifact discovery. |
| Policy objective | [CWM, Appendix C](https://arxiv.org/html/2510.02387v1#A3) | Generated-token masks, length-weighted centered returns, asymmetric clipped likelihood ratios, and fixed token-capacity scaling. Team-level credit accounting is this project's extension. |
| Concurrent subagents | [OpenCode pinned source](https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts) | Fresh/resumable child sessions, explicit delegation, and background work. Reimplemented in Python for trace accounting; model overrides are intentionally unavailable. [MIT license](https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/LICENSE). |
| Base model | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Original model revision is pinned in the run manifest. Apache-2.0 weights; no claim that pretrained knowledge is uncontaminated. |
| Local adapter training | [Unsloth Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md), [implementation](https://github.com/unslothai/unsloth/blob/main/unsloth/models/vision.py) | BF16 shared language-model LoRA; measure actual memory rather than treating documentation estimates as guarantees. Library is [Apache-2.0](https://github.com/unslothai/unsloth/blob/main/LICENSE); do not copy LGPL notebook code into this repository. |
| Human-bug evaluation | [SWE-bench harness](https://www.swebench.com/SWE-bench/guides/evaluation/), [Verified dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | Fixed held-out task selection, official scoring, and no reference fixes in agent context. Underlying repository licenses remain applicable. |
| Transfer failure motivation | [Anchored Self-Play for Code Repair](https://arxiv.org/html/2607.03523v1) | Evidence that synthetic-bug improvement can coincide with worse human-bug repair. Its reference mixing and embedding rewards are not part of this primary experiment. |
| Paired uncertainty | [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) | Task-paired percentile resampling implemented with the Python standard library. These intervals condition on one checkpoint; they do not measure training-seed variance. Degenerate intervals are flagged. |
| Paper rendering | [ReportLab User Guide](https://www.reportlab.com/docs/reportlab-userguide.pdf) | Platypus paragraphs, tables, images, and page callbacks. Report text and numerical claims are generated from saved run records. |
| Procedural-bug controls | [SWE-smith](https://github.com/SWE-bench/SWE-smith) | Reference for executable validation of synthetic bugs and controls. Any implemented restricted mutations are documented separately from the learned injector. |

## Corpus provenance

Training repositories: [more-itertools](https://github.com/more-itertools/more-itertools) ([MIT](https://github.com/more-itertools/more-itertools/blob/master/LICENSE)) and [boltons](https://github.com/mahmoud/boltons) ([BSD-3-Clause](https://github.com/mahmoud/boltons/blob/master/LICENSE)). Development: [cachetools](https://github.com/tkem/cachetools) ([MIT](https://github.com/tkem/cachetools/blob/master/LICENSE)). Synthetic held-out evaluation: [humanize](https://github.com/python-humanize/humanize) ([MIT](https://github.com/python-humanize/humanize/blob/main/LICENCE)). The environment manifest records the actual selected commits and container identities.

## Claim boundaries

An executable test is an oracle for the specified behavior, not a proof of complete program correctness. A training update is evidence of weight optimization, not of generalization. A comparison with the unchanged policy tests the combined pipeline; attributing an effect to subagents requires the planned ablations. Unrun controls and missing measurements are explicitly unscored. Both positive and negative observations belong in the generated reports.

## Human evaluation implementation boundary

[The local evaluator](../src/multi_agent_ssr/evaluation.py) calls the installed, locked official SWE-bench `make_test_spec` and `run_instance` APIs. It preserves their test oracle and parser. Its Docker client wrapper disables networking, drops capabilities, and applies memory, process, and CPU limits; these isolation settings differ from upstream defaults. The grader receives reference test patches only on the host, after the agent finishes.

The agent image is derived from the official dependency image with its original `/testbed` checkout and Git history removed. A separate source copy is checked against the pinned base commit, allowing mode-only changes made by the image builder, and receives no original Git metadata. Agent imports resolve to `/repo` or `/repo/src`. Committed official image digests remain fixed across preparation and fresh clones; rebuilt dependency-image IDs are recorded locally.

The upstream grader caches reports by run ID without comparing a new submitted patch. Each actual grading attempt therefore receives its own ID, while the evaluator reuses a task only after a completed outcome has been recorded. Infrastructure errors remain unscored. Invalid agent artifacts count as failures, and context-budget exhaustion is evaluated using the available patch when the official oracle can run. The generated reports retain coverage and distinguish the preregistered interval decision from the limitations of a degenerate bootstrap interval.
