# Experiments recorded in this run

Recorded failure history (retained after later successful phases):

- 2026-11-05T21:30:15.600645+00:00, **preflight**: RuntimeError: Generation/scoring log probabilities disagree: 0.152308.

## Interrupted attempt and recovery amendment

Prior attempt runs/pilot has recorded status preflight_failed. Its original wall-clock deadline was 2026-10-30T15:31:24.450111+00:00; original deadline preserved: True. Recorded prior failure: preflight: BrokenPipeError: [Errno 32] Broken pipe. The prior attempt's recorded elapsed time was 479.063 seconds. The recovery amendment charges 600 seconds for that attempt and allocates 28200 seconds to this recovery, within 28800 seconds of total active allocation. The amendment excludes the offline interval; it does not erase the expired original attempt. Saved prior artifacts: model outputs 0, evaluations 0, checkpoints 0. Recorded reason: User requested recovery after OS restart. Original preflight failed with BrokenPipeError; original wall-clock deadline expired. Offline interval is excluded only by the dated preregistration amendment.

| Bug source | Condition | Planned | Attempted | Resolved / scored | Rate | Unscored recorded | Unrecorded |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| human | base | 32 | 0 | Not evaluated | Not evaluated | 0 | 32 |
| human | trained | 32 | 0 | Not evaluated | Not evaluated | 0 | 32 |
| synthetic | base | 16 | 0 | Not evaluated | Not evaluated | 0 | 16 |
| synthetic | trained | 16 | 0 | Not evaluated | Not evaluated | 0 | 16 |

### Human bugs

Not evaluated: no tasks have completed outcomes for both conditions.

### Synthetic bugs

Not evaluated: no tasks have completed outcomes for both conditions.

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
| episodes | 0 |
| child sessions | 0 |
| episodes with children | 0 |
| delegation rate | Not measured |
| delegation calls | 0 |
| generated action tokens | 0 |
| generation calls | 0 |
| validated artifacts | 0 |
| invalid artifacts | 0 |
| invalid artifact rate | Not measured |
| optimizer steps | 0 |
| recorded training groups | 0 |
| peak allocated bytes | Not measured |
| evaluation seconds | 0 |
| training group seconds | 0 |
| pass to pass regressions | Not measured |
| regression scored records | 0 |

## Network and experiment design

| Condition | Architecture | Training in this experiment | Objective | Data and domain |
| --- | --- | --- | --- | --- |
| base | Qwen3.5-2B; one shared policy for all roles | None | Frozen comparison | Held-out repository repair |
| trained | Same base plus one shared language-model LoRA adapter | Recorded local RL updates | CWM-style clipped token objective with SSR team returns | Self-play on training repositories; held-out repository repair |

## Terms in plain language

- **Self-play:** the model generates challenges and learns from attempts to solve them.
- **Shared policy:** every primary agent and subagent uses the same model parameters.
- **LoRA:** small trainable adapter matrices modify a frozen base model.
- **Oracle:** executable tests that determine whether the required behavior is satisfied.
- **Paired comparison:** compare the two policies on the same task before averaging differences.
- **Bootstrap interval:** uncertainty estimated by repeatedly resampling the observed tasks.


The base condition is the unchanged model; trained is the final completed checkpoint. Human bugs come from benchmark issues; synthetic bugs are independently generated diagnostic mutations. Resolution requires passing the executable oracle. An unscored task has no valid completed outcome. The larger controlled study and seed sensitivity remain unexecuted.
