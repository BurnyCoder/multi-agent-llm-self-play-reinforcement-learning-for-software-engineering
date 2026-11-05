# Experiment: pilot-recovery

Status: **not_evaluated**. This report is generated from saved records; absent measurements remain unknown. All tables and diagnostics are derived from the [machine-readable analysis](summary.json). The [committed preregistration](preregistration.md) defines the hypotheses and decision rules.

No failure metadata is recorded. This alone does not establish successful execution.

## Interrupted attempt and recovery amendment

Prior attempt runs/pilot has recorded status preflight_failed. Its original wall-clock deadline was 2026-10-30T15:31:24.450111+00:00; original deadline preserved: True. Recorded prior failure: preflight: BrokenPipeError: [Errno 32] Broken pipe. The prior attempt's recorded elapsed time was 479.063 seconds. The recovery amendment charges 600 seconds for that attempt and allocates 28200 seconds to this recovery, within 28800 seconds of total active allocation. The amendment excludes the offline interval; it does not erase the expired original attempt. Saved prior artifacts: model outputs 0, evaluations 0, checkpoints 0. Recorded reason: User requested recovery after OS restart. Original preflight failed with BrokenPipeError; original wall-clock deadline expired. Offline interval is excluded only by the dated preregistration amendment.

## Evaluation

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

## Execution diagnostics

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

These are event-level counts, including incomplete collections; memory can include preflight. Unrecorded planned tasks remain missing coverage.

Invalidity reasons: `{}`. Untrained-group reasons: `{}`.

## Method and learnings

One shared model serves the primary agents and asynchronous children. The pilot tests the combined system; it does not identify the causal contribution of delegation. Inspect the saved run summary below for observed failures, decisions, and completed work. No unrun control has a reported score.

```json
{}
```

## Failure, preflight, and preparation records

```json
{
  "failure": {},
  "preflight": {},
  "preparation": {
    "preparation": {
      "human_environments_ready": 32,
      "human_environments_planned": 32,
      "synthetic_validated_tasks": 16,
      "synthetic_distinct_functions": 16,
      "synthetic_clean_test_counts": [
        723
      ],
      "official_grader_controls": [
        {
          "control": "gold-control",
          "instance_id": "pytest-dev__pytest-5631",
          "success": true,
          "status": "completed"
        },
        {
          "control": "empty-control",
          "instance_id": "pytest-dev__pytest-5631",
          "success": false,
          "status": "completed"
        }
      ],
      "environment_fingerprint": "e224dc39132c8c890a38537104315b4f11dbce93e38759e56e6f420404290793",
      "evidence_type": "CPU executable preparation checks; not model-performance results"
    },
    "environment_validation": {
      "recorded_at": "2026-11-05T21:03:39.058256+00:00",
      "original_data_directory": "/mnt/d/AI/multi-agent-ssr",
      "current_data_directory": "/mnt/d/AI/multi-agent-ssr",
      "reason": "D: was absent from Windows volume inventory after OS restart; reconstruct exactly pinned public inputs in the user-authorized WSL project directory.",
      "restoration_status": "external_drive_restored_and_verified",
      "startup_failures": [
        {
          "error": "ModuleNotFoundError: multi_agent_ssr",
          "cause": "Launcher resolved the venv Python symlink to the system interpreter",
          "fix": "Use the absolute venv executable path without resolving symlinks"
        },
        {
          "error": "FileNotFoundError: human_envs.json",
          "cause": "Configured D: data storage absent",
          "fix": "Reconstruct pinned inputs in WSL; retain original manifests and record local relocations"
        },
        {
          "error": "FileNotFoundError: uv",
          "cause": "Non-login launcher PATH omitted the user binary directory",
          "fix": "Include the user .local/bin directory in the preparation launcher PATH"
        }
      ],
      "logs": [
        "terminal.log",
        "preparation-terminal.log",
        "model-download.log"
      ],
      "model_selection_changed": false,
      "deadline_reset": false,
      "external_restored_at": "2026-11-05T21:14:51.629541+00:00",
      "model_file_checksums": [
        {
          "file": ".gitattributes",
          "bytes": 1570,
          "sha256": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"
        },
        {
          "file": "LICENSE",
          "bytes": 11544,
          "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
        },
        {
          "file": "README.md",
          "bytes": 62814,
          "sha256": "c0e83a849c776e6fa843d011f023132d942a0f0140d903205bf1c363adad2275"
        },
        {
          "file": "chat_template.jinja",
          "bytes": 7755,
          "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
        },
        {
          "file": "config.json",
          "bytes": 2908,
          "sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
        },
        {
          "file": "merges.txt",
          "bytes": 3353259,
          "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
        },
        {
          "file": "model.safetensors-00001-of-00001.safetensors",
          "bytes": 4548221488,
          "sha256": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
        },
        {
          "file": "model.safetensors.index.json",
          "bytes": 64460,
          "sha256": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
        },
        {
          "file": "preprocessor_config.json",
          "bytes": 390,
          "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
        },
        {
          "file": "tokenizer.json",
          "bytes": 12807982,
          "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
        },
        {
          "file": "tokenizer_config.json",
          "bytes": 16709,
          "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
        },
        {
          "file": "video_preprocessor_config.json",
          "bytes": 385,
          "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
        },
        {
          "file": "vocab.json",
          "bytes": 6722759,
          "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
        }
      ],
      "reconstructed_inputs_matched": {
        "human_environments": 32,
        "synthetic_tasks": 16
      },
      "recovery_grader_controls": "/mnt/d/AI/multi-agent-ssr/validation/human-grader-smoke-recovery-20261105/results.json",
      "duplicate_directory_removed": "/home/burny/projects/multi-agent-ssr",
      "storage_steering": "User requested no duplicate downloads on C: after reconnecting D:",
      "duplicates_removed_at": "2026-11-05T21:14:53.148265+00:00"
    }
  }
}
```

## Recorded configuration

```json
{
  "created_at": "2026-11-05T20:58:47.386714+00:00",
  "code_commit": "57f5624bd567110799ee098a7c0ea27d5abef608",
  "config": {
    "model": {
      "model_id": "Qwen/Qwen3.5-2B",
      "revision": "",
      "context_tokens": 2048,
      "action_tokens": 256,
      "lora_rank": 16,
      "lora_alpha": 16,
      "learning_rate": 5e-06,
      "seed": 0
    },
    "limits": {
      "generated_tokens": 4096,
      "tool_calls": 32,
      "child_sessions": 2,
      "child_depth": 1,
      "command_seconds": 90,
      "episode_seconds": 900
    },
    "experiment": {
      "budget_seconds": 28200,
      "injector_samples": 2,
      "repair_samples": 4,
      "optimization_passes": 2,
      "max_bug_order": 2,
      "human_tasks": 32,
      "synthetic_tasks": 16,
      "minimum_final_reserve_seconds": 7200,
      "delegation": true,
      "condition": "selfplay",
      "bootstrap_samples": 10000,
      "recovery_evidence": "manifests/recovery.json"
    }
  },
  "recovery": {
    "prior_run": "runs/pilot",
    "prior_status": "preflight_failed",
    "prior_budget": {
      "started_epoch": 1793345484.450111,
      "budget_seconds": 28800,
      "deadline_epoch": 1793374284.450111,
      "started_at": "2026-10-30T07:31:24.450125+00:00"
    },
    "prior_events": [
      {
        "timestamp": "2026-10-30T07:31:26.384084+00:00",
        "event": "phase_started",
        "phase": "preflight",
        "code_commit": "a49d13369ec17c005c844af17b88aa6dff66d86a",
        "remaining_seconds": 28798.066036701202
      },
      {
        "timestamp": "2026-10-30T07:39:23.513025+00:00",
        "event": "phase_failed",
        "phase": "preflight",
        "error": "BrokenPipeError: [Errno 32] Broken pipe"
      }
    ],
    "prior_attempt_elapsed_seconds": 479.0629141330719,
    "charged_prior_seconds": 600,
    "recovery_budget_seconds": 28200,
    "total_active_allocation_seconds": 28800,
    "original_deadline_preserved": true,
    "saved_model_generations": 0,
    "saved_checkpoints": 0,
    "saved_evaluations": 0,
    "original_record_sha256": {
      "budget.json": "c2efc46ea1aa4ee59e82a08855ddfb74b3bb4811a45b39b625b55d1bed5202e4",
      "events.jsonl": "0387ecd61f92982dc0ccdd31746f45b8f276885d72b20318c66c45bac00ecc27",
      "manifest.json": "15d1cfe5cd7cd423c536b23b5e0b60644258ed9e72b99d2c0c1a447d429a0160",
      "terminal.log": "b529901c78267e59030d5ee1fdd0687a60970ab8b3867a0f40a3cb63200af8fd"
    },
    "reason": "User requested recovery after OS restart. Original preflight failed with BrokenPipeError; original wall-clock deadline expired. Offline interval is excluded only by the dated preregistration amendment."
  },
  "model": {
    "model_id": "Qwen/Qwen3.5-2B",
    "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "context_tokens": 2048,
    "action_tokens": 256,
    "lora_rank": 16,
    "lora_alpha": 16,
    "learning_rate": 5e-06,
    "seed": 0
  },
  "hardware": "name, memory.total [MiB], compute_cap\nNVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, 12.0\n",
  "expected_human_tasks": 32,
  "expected_synthetic_tasks": 16,
  "hypothesis": "Trained team improves held-out human bug resolution over frozen initialization"
}
```

## Reproduction and limitations

Regenerate this report with `uv run selfplay report --run <run-directory>`. Task-paired intervals condition on one training checkpoint; training-seed uncertainty is unmeasured. Degenerate bootstrap intervals do not establish equivalence. Missing evaluations and infrastructure errors are excluded from scored denominators and reported separately. Public-data pretraining contamination remains unknown.

## Sources

- [Self-play SWE-RL: task construction and rewards](https://arxiv.org/html/2512.18552v3)
- [CWM: clipped token objective and trajectory masks](https://arxiv.org/html/2510.02387v1#A3)
- [OpenCode: asynchronous child sessions](https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts)
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Unsloth Qwen3.5 fine-tuning](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)
- [SWE-bench evaluation harness](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [Anchored Self-Play: synthetic-to-human bug distribution shift](https://arxiv.org/html/2607.03523v1)
- [Paired percentile bootstrap procedure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Committed preregistration and analysis plan](https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering/blob/main/reports/preregistration.md)
