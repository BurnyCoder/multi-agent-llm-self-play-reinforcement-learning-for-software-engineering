"""Regenerate research artifacts from immutable run records, never hand-entered scores.

Task-paired resampling follows the percentile construction documented at
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html.
PDF layout uses ReportLab Platypus; all dataset text is escaped before rendering.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

SOURCES = [
    ("Self-play SWE-RL: task construction and rewards", "https://arxiv.org/html/2512.18552v3"),
    ("CWM: clipped token objective and trajectory masks", "https://arxiv.org/html/2510.02387v1#A3"),
    ("OpenCode: asynchronous child sessions", "https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts"),
    ("Qwen3.5-2B model card", "https://huggingface.co/Qwen/Qwen3.5-2B"),
    ("Unsloth Qwen3.5 fine-tuning", "https://unsloth.ai/docs/models/qwen3.5/fine-tune.md"),
    ("SWE-bench evaluation harness", "https://www.swebench.com/SWE-bench/guides/evaluation/"),
    ("Anchored Self-Play: synthetic-to-human bug distribution shift", "https://arxiv.org/html/2607.03523v1"),
    ("Paired percentile bootstrap procedure", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html"),
    ("Committed preregistration and analysis plan", "preregistration.md"),
]

AI_STATEMENT = (
    "Codex wrote this codebase and paper from a specification provided by the human author. "
    "Human cleanup and rewriting are in progress."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Fail loudly on damaged data; an absent optional stream is still empty."""
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path.name}:{line_number}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"Expected object at {path.name}:{line_number}")
        records.append(value)
    return records


def _json(path: Path) -> dict[str, Any]:
    """Keep missing metadata visibly missing rather than guessing its contents."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, value: Any) -> None:
    """Reject NaNs so downstream JavaScript and scientific consumers agree."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _display_event(event: dict[str, Any]) -> dict[str, Any]:
    """Keep complete event text, but omit huge token arrays from the browser copy.

The original events.jsonl is untouched and remains authoritative for replay.
This representation is for human inspection, not likelihood reconstruction.
"""
    result = dict(event)
    for name in ("generation", "data"):
        if isinstance(result.get(name), dict):
            result[name] = _display_event(result[name])
    for name in ("prompt_ids", "completion_ids", "old_logprobs"):
        if isinstance(result.get(name), list):
            result[f"{name}_count"] = len(result.pop(name))
    return result


def _run_events(run_dir: Path) -> list[dict[str, Any]]:
    """Merge the designated agent logs; never crawl scorer or repository files."""
    paths = [run_dir / "events.jsonl", *sorted(run_dir.glob("synthetic/*/*/attempt-*/agents/events.jsonl"))]
    events = [_display_event(event) for path in paths for event in load_jsonl(path)]
    return sorted(events, key=lambda event: str(event.get("timestamp", "")))


def _run_patches(run_dir: Path) -> list[dict[str, str]]:
    """Publish actual generated changes while keeping official oracle patches private."""
    patches = []
    for pattern, kind in [("artifacts/*/bug_inject.diff", "Training bug"),
                          ("artifacts/*/test_weaken.diff", "Training test weakening"),
                          ("human/*/*/prediction.diff", "Human-bug repair"),
                          ("synthetic/*/*/attempt-*/repair.diff", "Synthetic-bug repair")]:
        for path in sorted(run_dir.glob(pattern)):
            patches.append({"path": path.relative_to(run_dir).as_posix(), "kind": kind,
                            "text": path.read_text(encoding="utf-8")})
    for path in sorted((run_dir / "episodes").glob("*.json")):
        episode = _json(path)
        patch = episode.get("result", {}).get("repair_patch")
        if isinstance(patch, str):
            patches.append({"path": path.relative_to(run_dir).as_posix() + "#result.repair_patch",
                            "kind": "Training repair", "text": patch})
    return patches


def _quantile(ordered: list[float], probability: float) -> float:
    """Linear interpolation of the sorted empirical bootstrap distribution."""
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _scored(row: dict[str, Any]) -> bool:
    """Boolean outcomes require a completed evaluation; missing is not failure."""
    return row.get("status") == "completed" and type(row.get("success")) is bool


def analyze_evaluations(rows: list[dict[str, Any]], *, bootstrap_samples: int = 10000,
                        seed: int = 0) -> dict[str, Any]:
    """Pair by source/task, permitting infra retries but refusing outcome selection.

Intervals describe task resampling conditional on this one trained checkpoint;
they do not quantify variation across training seeds or arbitrary repositories.
"""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("source", "unknown")), str(row.get("condition", "unknown")),
               str(row.get("task_id", "")))
        if not key[2]:
            raise ValueError("Evaluation record has no task_id")
        previous = indexed.get(key)
        if previous and _scored(previous) and _scored(row) and previous["success"] != row["success"]:
            raise ValueError(f"Conflicting completed outcomes for {key}; refusing to select an outcome")
        if previous is None or not _scored(previous):
            indexed[key] = row
    conditions: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, Any] = {}
    sources = sorted({"human", "synthetic"} | {key[0] for key in indexed})
    for source in sources:
        conditions[source] = {}
        names = sorted({"base", "trained"} | {key[1] for key in indexed if key[0] == source})
        for condition in names:
            selected = [row for key, row in indexed.items() if key[:2] == (source, condition)]
            completed = [row for row in selected if _scored(row)]
            successes = sum(row["success"] for row in completed)
            conditions[source][condition] = {
                "recorded_tasks": len(selected), "completed": len(completed),
                "attempted": sum(bool(row.get("attempted", _scored(row) or (row.get("generated_tokens", 0) or 0) > 0)) for row in selected),
                "successes": successes, "unscored": len(selected) - len(completed),
                "rate": successes / len(completed) if completed else None,
                "statuses": dict(Counter(str(row.get("status", "unknown")) for row in selected)),
                "generated_tokens": sum(row.get("generated_tokens", 0) or 0 for row in selected),
                "elapsed_seconds": sum(row.get("elapsed_seconds", 0) or 0 for row in selected),
                "delegations": sum(row.get("delegations", 0) or 0 for row in selected),
            }
        base = {key[2]: row for key, row in indexed.items() if key[:2] == (source, "base") and _scored(row)}
        trained = {key[2]: row for key, row in indexed.items() if key[:2] == (source, "trained") and _scored(row)}
        task_ids = sorted(base.keys() & trained.keys())
        differences = [int(trained[task]["success"]) - int(base[task]["success"]) for task in task_ids]
        comparison: dict[str, Any] = {
            "paired_tasks": len(task_ids), "task_ids": task_ids,
            "difference": None, "ci95": None, "interval_degenerate": False,
            "improved": differences.count(1), "regressed": differences.count(-1),
            "unchanged": differences.count(0), "interpretation": "not_evaluated",
            "preregistered_decision": "not_evaluated",
        }
        if differences:
            rng = random.Random(seed)
            draws = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                           for _ in range(bootstrap_samples))
            interval = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
            interpretation = "inconclusive"
            if not any(differences):
                interpretation = "no_observed_difference"
            elif interval[0] > 0:
                interpretation = "positive_task_level_evidence"
            elif interval[1] <= 0:
                interpretation = "nonpositive_task_level_evidence"
            comparison.update(difference=sum(differences) / len(differences), ci95=interval,
                              interval_degenerate=interval[0] == interval[1], interpretation=interpretation,
                              preregistered_decision=("supports_predicted_improvement" if interval[0] > 0 else
                                  "contradicts_predicted_improvement" if interval[1] <= 0 else "inconclusive"))
        comparisons[source] = comparison
    return {"conditions": conditions, "comparisons": comparisons,
            "bootstrap": {"method": "paired percentile", "samples": bootstrap_samples,
                          "seed": seed, "confidence": 0.95},
            "records": [indexed[key] for key in sorted(indexed)]}


def _diagnostics(events: list[dict[str, Any]], metrics: list[dict[str, Any]], preflight: dict[str, Any],
                 evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe recorded activity without treating absent streams as measurements."""
    sessions = {event["session_id"]: event for event in events
                if event.get("event") == "session_started" and event.get("session_id")}
    episodes = {event.get("episode_id") for event in sessions.values() if event.get("episode_id")}
    children = [event for event in sessions.values() if event.get("parent_session_id")]
    delegated = {event.get("episode_id") for event in children if event.get("episode_id")}
    generations = {event["call_id"]: event.get("generation", {}) for event in events
                   if event.get("event") == "generation" and event.get("call_id")}
    validations = [event for event in events if event.get("event") == "artifact_validated" and type(event.get("valid")) is bool]
    invalid = [event for event in validations if not event["valid"]]
    memory = [row["memory"] for row in metrics if isinstance(row.get("memory"), (int, float))]
    if isinstance(preflight.get("peak_allocated_bytes"), (int, float)):
        memory.append(preflight["peak_allocated_bytes"])
    regressions = []
    for row in evaluations:
        statuses = row.get("official_report", {}).get("tests_status", {})
        if "PASS_TO_PASS" in statuses:
            regressions.append(len(statuses["PASS_TO_PASS"].get("failure", [])))
        elif isinstance(row.get("regressions"), (int, list)):
            regressions.append(len(row["regressions"]) if isinstance(row["regressions"], list) else row["regressions"])
    return {
        "episodes": len(episodes), "child_sessions": len(children), "episodes_with_children": len(delegated),
        "delegation_rate": len(delegated) / len(episodes) if episodes else None,
        "delegation_calls": sum(event.get("event") == "tool_started" and event.get("name") == "subagent" for event in events),
        "generated_action_tokens": sum(len(g["completion_ids"]) if isinstance(g.get("completion_ids"), list)
                                       else g.get("completion_ids_count", 0) for g in generations.values()),
        "generation_calls": len(generations), "validated_artifacts": len(validations), "invalid_artifacts": len(invalid),
        "invalid_artifact_rate": len(invalid) / len(validations) if validations else None,
        "invalid_reasons": dict(Counter(str(event.get("reason", "Unspecified")) for event in invalid)),
        "untrained_group_reasons": dict(Counter(str(event.get("reason", "Unspecified")) for event in events if event.get("event") == "group_not_trained")),
        "optimizer_steps": sum(row.get("optimizer_steps", 0) for row in metrics),
        "recorded_training_groups": len(metrics), "peak_allocated_bytes": max(memory) if memory else None,
        "evaluation_seconds": sum(row.get("elapsed_seconds", 0) or 0 for row in evaluations),
        "training_group_seconds": sum(row.get("group_seconds", 0) or 0 for row in metrics),
        "pass_to_pass_regressions": sum(regressions) if regressions else None,
        "regression_scored_records": len(regressions),
    }


def _controls() -> list[dict[str, str]]:
    """Deferred study arms are design entries, never fabricated score rows."""
    return [{"condition": name, "purpose": purpose, "status": "not_run"} for name, purpose in [
        ("single-agent self-play", "Separate delegation from self-play training"),
        ("repair-only training", "Control for updating the injector"),
        ("procedural-mutation training", "Control the training bug distribution"),
        ("no-delegation evaluation", "Measure inference-time delegation dependence"),
        ("seed sensitivity", "Measure training-seed variation")]]


def _control_table(controls: list[dict[str, str]]) -> str:
    lines = ["| Deferred condition | Purpose | Result |", "| --- | --- | --- |"]
    lines.extend(f"| {row['condition']} | {row['purpose']} | Not run |" for row in controls)
    return "\n".join(lines)


def _diagnostic_table(diagnostics: dict[str, Any]) -> str:
    lines = ["| Recorded diagnostic | Value |", "| --- | ---: |"]
    for key, value in diagnostics.items():
        if not isinstance(value, dict):
            shown = "Not measured" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)
            lines.append(f"| {key.replace('_', ' ')} | {shown} |")
    return "\n".join(lines)


def _update_readme(path: Path, content: str) -> None:
    """Only replace the explicitly designated generated block, preserving human text."""
    start, end = "<!-- SSR_RESULTS:START -->", "<!-- SSR_RESULTS:END -->"
    text = path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1 or text.index(end) < text.index(start):
        raise ValueError("README requires exactly one ordered SSR_RESULTS marker pair")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before + start + "\n\n" + content + "\n\n" + end + after, encoding="utf-8")


def _pct(value: float | None) -> str:
    """Only presentation rounding belongs here; stored estimates retain precision."""
    return "Not evaluated" if value is None else f"{100 * value:.1f}%"


def _table(analysis: dict[str, Any]) -> str:
    """One table is reused by the experiment report and paper."""
    lines = ["| Bug source | Condition | Planned | Attempted | Resolved / scored | Rate | Unscored recorded | Unrecorded |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for source, conditions in analysis["conditions"].items():
        for condition, result in conditions.items():
            count = f"{result['successes']} / {result['completed']}" if result["completed"] else "Not evaluated"
            planned = result.get("planned") if result.get("planned") is not None else "Unknown"
            unrecorded = result.get("unrecorded") if result.get("unrecorded") is not None else "Unknown"
            lines.append(f"| {source} | {condition} | {planned} | {result['attempted']} | {count} | {_pct(result['rate'])} | {result['unscored']} | {unrecorded} |")
    return "\n".join(lines)


def _effect_text(comparison: dict[str, Any]) -> str:
    """A degenerate bootstrap interval must not be sold as precise equivalence."""
    if not comparison["paired_tasks"]:
        return "Not evaluated: no tasks have completed outcomes for both conditions."
    lower, upper = comparison["ci95"]
    text = (f"Across {comparison['paired_tasks']} paired tasks, the trained-minus-base resolution difference "
            f"was {100 * comparison['difference']:+.1f} percentage points "
            f"(95% paired bootstrap interval [{100 * lower:+.1f}, {100 * upper:+.1f}]). "
            f"Improved: {comparison['improved']}; regressed: {comparison['regressed']}; "
            f"unchanged: {comparison['unchanged']}.")
    decisions = {"supports_predicted_improvement": "supports the predicted improvement",
                 "contradicts_predicted_improvement": "contradicts the predicted improvement", "inconclusive": "is inconclusive"}
    text += f" Under the preregistered interval rule, this comparison {decisions[comparison['preregistered_decision']]}."
    if comparison["interval_degenerate"]:
        text += " The bootstrap interval is degenerate; it does not establish equivalence or rule out a population effect."
    return text


def _paper_sections(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Narrative claims are bounded by the observed record coverage and design."""
    primary = _effect_text(payload["analysis"]["comparisons"]["human"])
    secondary = _effect_text(payload["analysis"]["comparisons"]["synthetic"])
    config = payload["manifest"].get("config", {})
    model = payload["manifest"].get("model", config.get("model", {}))
    limits = config.get("limits", {})
    settings = config.get("experiment", {})
    def value(data, key):
        return str(data.get(key, "unrecorded"))
    state = f"Recorded run status: {payload['status']}."
    if payload["failure"]:
        state += f" A recorded {payload['failure'].get('phase', 'run')} failure was: {payload['failure'].get('error', 'unspecified')}."
    if not payload["diagnostics"]["optimizer_steps"]:
        state += " No completed policy optimizer steps are recorded; this record does not establish a learned checkpoint."
    sections = [
        ("Abstract", "We implement a local shared-policy extension of Self-play SWE-RL in which bug-injection "
         "and repair teams can delegate to asynchronous subagents. All roles use one Qwen3.5-2B policy with a "
         "shared trainable LoRA adapter. This report is generated from the saved run records. " + state + " " + primary),
        ("Context and motivation", ("The experiment asks whether reinforcement learning from executable "
         "repository feedback improves held-out human-bug repair. Prior SSR work motivates repository-grounded "
         "bug injection and repair [1]. Anchored Self-Play demonstrates that gains on synthetic bugs need not "
         "transfer to human bugs [7], so the two evaluation sources remain separate.")),
        ("Method", ("The injector edits source and weakens existing tests. A trusted Python runner validates "
         "the artifact, including restoration checks and inverse mutations. The repair team sees the weakened "
         "buggy state and the inverse weakening patch, but not clean reference code or the injection patch. "
         "Original oracle tests are restored for scoring. This Python test interface is a deliberate adaptation "
         "of SSR's more general artifact-discovery procedure [1].")),
        ("Task construction and trusted validation", ("The clean selected test suite must contain at least five passing "
         "tests. The source mutation must induce a failure and preserve a pass. The weakening patch must hide at "
         "least one induced failure. Inverse mutation checks revert each changed source file independently and "
         "require that reversal to repair at least one induced failure. Empty selections, missing expected tests, "
         "collection failures, and parser errors cannot produce success. Artifacts retain the SSR five-file "
         "interface: source mutation, test weakening, selected test files, runner, and parser; the last two are "
         "constructed by this harness rather than delegated to a model [1]. One failed repair per original bug "
         "can become an independently revalidated second-order task. Its repair attempts contribute solver "
         "gradients; the original injector return uses first-order attempts only.")),
        ("Agents and optimization", ("Primary agents delegate through a tool patterned after OpenCode [3]. "
         "Child sessions have independent contexts and can work while the primary agent continues. GPU generation "
         "requests are serialized through the shared policy. One team execution includes all descendant actions. "
         "Each actual generated token contributes once; a child's text repeated in its parent's context is an "
         "observation. The update extends CWM's length-weighted centered returns, clipped token likelihood ratios, "
         "and action masks [2]. The precise run configuration is included in the appendix.")),
        ("Exact objective and team credit", ("For a group G of episodes with the same task and team role, let "
         "L_e be the number of unique generated action tokens and R_e the episode return. The length-weighted "
         "baseline and advantage are mu_G = sum_e(L_e R_e) / sum_e(L_e), and A_e = R_e - mu_G. Injector and "
         "solver groups remain separate. Every primary/child action in episode e receives A_e; prompt tokens, "
         "tool output, and copied child output receive no action loss [2].\n\n"
         "For an action token t, r_et = exp(log pi_theta(a_et | original_context_et) - old_logprob_et). "
         "The minimized objective is L(theta) = -(1/C) sum_G sum_e sum_t min(r_et A_e, "
         "clip(r_et, 0.8, 1.25) A_e). C is the maximum planned collection token capacity, not the observed "
         "trajectory length. In primary self-play, C = injector_samples * (1 + repair_samples * max_bug_order) "
         "* team_generated_token_budget. There is no reward-standard-deviation normalization or KL term. "
         "Each generated call appears once in the collection and once per optimization pass. The original "
         "behavior log probabilities remain fixed across both passes [2].\n\n"
         "Injector returns follow SSR: -1 for an invalid artifact; -0.8 for a valid bug with empirical repair "
         "success rate s equal to zero or one; otherwise 1 - 1.8s. A repair receives +1 only when the restored "
         "oracle passes completely, and -1 otherwise [1]. Zero-advantage groups are recorded without an update. "
         "Incomplete or infrastructure-corrupted groups never enter optimization.")),
        ("Model, sampling, and optimization settings", (
         f"The recorded model is {value(model, 'model_id')} at revision {value(model, 'revision')}. "
         f"LoRA rank is {value(model, 'lora_rank')}, alpha {value(model, 'lora_alpha')}, and learning rate "
         f"{value(model, 'learning_rate')}; recorded training seed is {value(model, 'seed')}. "
         "The implementation uses BF16 adapters with dropout zero, a frozen base, frozen vision tower, embeddings "
         "and output head, and language-model attention/MLP adapters only [4, 5].\n\n"
         f"A call has at most {value(model, 'context_tokens')} input-plus-action tokens and "
         f"{value(model, 'action_tokens')} generated tokens. Each team shares a generated-token cap of "
         f"{value(limits, 'generated_tokens')}, a tool-call cap of {value(limits, 'tool_calls')}, and "
         f"{value(limits, 'child_sessions')} child sessions at depth {value(limits, 'child_depth')}. "
         "Sampling temperature is one; thinking is disabled. Root and child threads progress concurrently, "
         "but calls acquire one shared GPU policy lock. All workers finish before an update.\n\n"
         f"The recorded collection uses {value(settings, 'injector_samples')} independent injectors per clean "
         f"starting state and {value(settings, 'repair_samples')} repairs per valid bug, followed by "
         f"{value(settings, 'optimization_passes')} optimization passes. AdamW uses betas (0.9, 0.99), "
         "weight decay zero, gradient clipping at norm 1.0, gradient checkpointing, and one call per microbatch. "
         "Likelihood computation retains only action-position logits and accumulates gradients sequentially.")),
        ("Evaluation and analysis", ("Frozen and final policies are compared on matched tasks and inference "
         "budgets. Task-level paired percentile bootstrap intervals use the resampling scheme in [8]. Only "
         "completed boolean outcomes are scored. Infrastructure failures and unfinished evaluations are reported "
         "as missing coverage, not failed repairs. Conflicting duplicate completed outcomes are rejected. The "
         "final checkpoint is selected by completion order rather than held-out score.")),
        ("Evaluation sampling and stopping", (
         f"Planned human and synthetic task counts in the run manifest are "
         f"{payload['manifest'].get('expected_human_tasks', 'unrecorded')} and "
         f"{payload['manifest'].get('expected_synthetic_tasks', 'unrecorded')}. Human tasks are fixed before "
         "training by sorting SHA-256 instance-ID hashes within the eligible requests, Flask, pylint, and "
         "pytest repositories. Synthetic humanize mutations are selected by deterministic syntax and executable "
         "validity, with at most one accepted mutation per function. Neither cohort is selected using model "
         "performance. Gold human fixes and evaluator test patches are withheld from agents [6].\n\n"
         f"Each source is analyzed separately with {payload['analysis']['bootstrap']['samples']} task resamples, "
         f"bootstrap seed {payload['analysis']['bootstrap']['seed']}, and percentile endpoints at 2.5% and 97.5%. "
         "The paired difference is mean(trained_success - base_success) over completed pairs. A positive lower "
         "bound supports the prediction; an upper bound at or below zero contradicts it; an interval spanning "
         "zero is inconclusive. Degenerate intervals are explicitly qualified [8].\n\n"
         f"The recorded total single-GPU wall-clock cap is {value(settings, 'budget_seconds')} seconds. It "
         "includes model loading, compilation, preflight, baseline, training, and final evaluation, using a "
         "persisted deadline that restarts cannot reset. Baseline allocation is capped at (remaining - 1800) / "
         "2.25 seconds. Training reserves the greater of the configured minimum final-evaluation reserve and "
         "1.25 times measured baseline runtime. A new collection must fit its measured duration with margin. "
         "If an optimizer pass crosses the reserve boundary, restore the last complete checkpoint. Nonfinite "
         "gradients, memory failure, likelihood incompatibility, or corrupt state stop training without a silent "
         "model/precision substitution. Preregistration and any amendments remain authoritative [9].")),
        ("Differences from prior SSR", ("This is a Python-only, local, single-GPU adaptation, not a numerical "
         "reproduction of SSR [1]. It uses Qwen3.5-2B, one tied LoRA adapter for both teams and all descendants, "
         "asynchronous prompted delegation, trusted pytest execution/JUnit parsing, bounded contexts, and a "
         "small fixed evaluation cohort. It does not reproduce SSR's original model scale, context length, "
         "distributed asynchronous learner, history-guided curriculum, or full evaluation suite. The CWM "
         "objective is extended from trajectories to complete action trees using the exact original contexts [2].")),
        ("Results", state + "\n\n" + primary + "\n\nSynthetic diagnostic set: " + secondary),
        ("Observed execution diagnostics", "The following diagnostics are derived from saved event, evaluation, "
         "and optimizer streams, including incomplete collections. Counts describe recorded activity, not "
         "generalization. Preflight memory belongs to a software check, not a trained-model evaluation. "
         f"Recorded action tokens: {payload['diagnostics']['generated_action_tokens']}; actual subagent tool "
         f"calls: {payload['diagnostics']['delegation_calls']}; child sessions: {payload['diagnostics']['child_sessions']}; "
         f"optimizer steps: {payload['diagnostics']['optimizer_steps']}. "
         "Full invalidity reasons, exclusions, failure metadata, and per-task outcomes accompany the generated report."),
        ("Claims and implications", ("These measurements are a comparison of the recorded policies under this "
         "harness. They do not isolate the causal contribution of delegation, self-play, or parameter sharing. "
         "Those claims require the planned controlled comparisons. A functioning update or improved training "
         "reward alone does not establish better software engineering.")),
        ("Related work", ("SSR supplies the bug/repair self-play task [1], CWM supplies the optimization "
         "foundation [2], and OpenCode supplies a reference for tool-mediated child sessions [3]. The model and "
         "local LoRA configuration follow Qwen and Unsloth documentation [4, 5]. Human-bug scoring uses the "
         "SWE-bench harness [6]. This implementation does not claim the first bug/repair self-play method.")),
        ("Limitations and discussion", ("The pilot uses one training seed, a small task sample, a short context "
         "window, and limited compute. Task bootstrap intervals condition on this checkpoint and cannot quantify "
         "training-seed variation. Small samples can yield degenerate intervals. Public repositories may have "
         "appeared in model pretraining; contamination is unknown. Trusted tests reduce particular scoring "
         "failures but cannot guarantee realism or prevent every form of reward gaming. Unrun controls are not "
         "evidence. All incomplete coverage is retained in the tables.")),
        ("Conclusion", "The saved artifacts establish what ran and what was measured. " + primary),
    ]
    return sections


def render_figures(payload: dict[str, Any], destination: Path) -> dict[str, str]:
    """Use consistent full-scale success axes; draw only observed learning points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fafaf6", "axes.facecolor": "#fafaf6"})
    paths: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    boxes = [(0.2, 2.8, "Injector team\nPrimary + asynchronous children", "#e9d9c0"),
             (6.0, 2.8, "Repair team\nPrimary + asynchronous children", "#cfe2db"),
             (3.1, 0.4, "Shared Qwen3.5-2B + LoRA\nOne policy, separate contexts", "#e0e2ed")]
    for x, y, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 3.7, 1.2, boxstyle="round,pad=0.1", fc=color, ec="#3b4541"))
        ax.text(x + 1.85, y + 0.6, label, ha="center", va="center", fontsize=10)
    for start, end in [((3.95, 3.4), (5.9, 3.4)), ((4, 1.7), (2.8, 2.7)), ((6, 1.7), (7.2, 2.7))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#3b4541", "lw": 1.5})
    ax.text(4.95, 4.15, "Validated bug +\nrestored-test specification", ha="center", fontsize=9)
    ax.text(5, 0, "Parallel agent/tools; serialized GPU requests. Executable tests provide rewards.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(destination / "architecture.png", dpi=180)
    plt.close(fig)
    paths["architecture"] = "architecture.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    points = []
    for source in ("human", "synthetic"):
        for condition, color in (("base", "#b8996b"), ("trained", "#177b67")):
            result = payload["analysis"]["conditions"][source][condition]
            if result["rate"] is not None:
                points.append((f"{source}\n{condition}", 100 * result["rate"], color, result))
    if points:
        for index, (label, rate, color, result) in enumerate(points):
            ax.bar(index, rate, color=color, width=.6)
            ax.text(index, min(96, rate + 3), f"{result['successes']}/{result['completed']}", ha="center")
        ax.set_xticks(range(len(points)), [point[0] for point in points])
    else:
        ax.text(.5, .5, "No completed evaluations", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
    ax.set(ylim=(0, 105), ylabel="Resolved tasks (%)", title="Observed resolution; counts show evaluated coverage")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(destination / "scores.png", dpi=180)
    plt.close(fig)
    paths["scores"] = "scores.png"
    for name, keys, label in [("loss", ["loss"], "Policy objective loss"),
                              ("rewards", ["inject_reward", "repair_reward"], "Mean team return"),
                              ("memory", ["memory"], "Peak GPU allocated memory (GiB)"),
                              ("tokens", ["token_count"], "Unique generated collection tokens"),
                              ("validity", ["valid_bug_rate"], "Valid artifact fraction")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        any_points = False
        for key in keys:
            scale = 1 / (1024 ** 3) if key == "memory" else 1
            points = [(row.get("step", i), row[key] * scale) for i, row in enumerate(payload["metrics"])
                      if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points], ".-", label=key.replace("_", " "), lw=1.2)
                any_points = True
        if any_points:
            ax.legend(frameon=False)
        else:
            ax.text(.5, .5, "No recorded measurements", transform=ax.transAxes, ha="center", va="center")
        ax.set(xlabel="Completed collection group (record index if unavailable)", ylabel=label)
        if name == "rewards":
            ax.set_ylim(-1.05, 1.05)
        elif name == "validity":
            ax.set_ylim(0, 1.05)
        elif name in {"memory", "tokens"}:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=180)
        plt.close(fig)
        paths[name] = f"{name}.png"
    return paths


def render_paper(payload: dict[str, Any], sections: list[tuple[str, str]], destination: Path,
                 figures: dict[str, str], figure_dir: Path) -> None:
    """Build a paginated paper using escaped paragraphs and repeatable table headers."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontSize=20, leading=24, textColor=colors.HexColor("#163b33")))
    styles.add(ParagraphStyle(name="Meta", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9, leading=13, spaceAfter=10))
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 8
    styles["Heading2"].textColor = colors.HexColor("#177b67")
    story = [Paragraph("Local multi-agent self-play RL<br/>for software engineering", styles["PaperTitle"]),
             Paragraph("Libor &quot;Burny&quot; Burian", styles["Meta"]),
             Paragraph(escape(f"Run: {payload['run_id']} | Status: {payload['status']}"), styles["Meta"])]
    for title, body in sections:
        story.append(Paragraph(escape(title), styles["Heading2"]))
        for paragraph in body.split("\n\n"):
            story.append(Paragraph(escape(paragraph), styles["BodyText"]))
        if title == "Method" and figures.get("architecture"):
            story.append(Image(str(figure_dir / figures["architecture"]), width=16.3 * cm, height=8.7 * cm))
        if title == "Results":
            data = [["Source", "Policy", "Planned", "Attempted", "Resolved / scored", "Rate", "Unscored recorded", "Unrecorded"]]
            for source, conditions in payload["analysis"]["conditions"].items():
                for condition, row in conditions.items():
                    data.append([source, condition, str(row.get("planned") if row.get("planned") is not None else "Unknown"),
                                 str(row["attempted"]), f"{row['successes']} / {row['completed']}" if row["completed"] else "Not evaluated",
                                 _pct(row["rate"]), str(row["unscored"]),
                                 str(row.get("unrecorded") if row.get("unrecorded") is not None else "Unknown")])
            table = Table([[Paragraph(escape(cell), styles["BodyText"]) for cell in row] for row in data],
                          colWidths=[width * cm for width in [1.6, 1.7, 1.3, 1.7, 2.6, 2.1, 2.7, 2.3]], repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3ece7")),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                       ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#177b67"))]))
            story.extend([table, Spacer(1, .3 * cm)])
            story.append(Paragraph("Deferred controls (not executed in this pilot)", styles["Heading3"]))
            for control in payload["controls"]:
                story.append(Paragraph(escape(control["condition"] + ": not run. " + control["purpose"] + "."), styles["BodyText"]))
            for key in ("scores", "loss", "rewards"):
                if figures.get(key):
                    story.append(Image(str(figure_dir / figures[key]), width=16 * cm, height=(9 if key == "scores" else 8) * cm))
        if title == "Observed execution diagnostics":
            for key, value in payload["diagnostics"].items():
                shown = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else "Not measured" if value is None else str(value)
                story.append(Paragraph(f"<b>{escape(key.replace('_', ' '))}</b>: {escape(shown)}", styles["BodyText"]))
            for key in ("memory", "tokens", "validity"):
                if figures.get(key):
                    story.append(Image(str(figure_dir / figures[key]), width=16 * cm, height=8 * cm))
    story.append(Paragraph("Appendix: recorded configuration", styles["Heading2"]))
    for key, value in sorted(payload["manifest"].items()):
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        story.append(Paragraph(f"<b>{escape(str(key))}</b>: {escape(text)}", styles["BodyText"]))
    story.append(Paragraph("References", styles["Heading2"]))
    for index, (title, url) in enumerate(SOURCES, 1):
        story.append(Paragraph(f'[{index}] <a href="{escape(url, quote=True)}" color="#177b67">{escape(title)}</a>. {escape(url)}', styles["BodyText"]))
    story.extend([Paragraph("AI statement", styles["Heading2"]), Paragraph(escape(AI_STATEMENT), styles["BodyText"])])

    def footer(canvas, doc):
        """Keep provenance visible without adding an empirical claim."""
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#68706c"))
        canvas.drawString(1.9 * cm, 1.2 * cm, "Generated from saved run records")
        canvas.drawRightString(A4[0] - 1.9 * cm, 1.2 * cm, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=1.9 * cm, leftMargin=1.9 * cm,
                      topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                      title="Local multi-agent self-play RL for software engineering",
                      author='Libor "Burny" Burian').build(story, onFirstPage=footer, onLaterPages=footer)


def generate_report(run_dir: Path, output_dir: Path, site_dir: Path, *, readme_path: Path | None = None) -> dict[str, str]:
    """One command regenerates tables, charts, Markdown, paper, and static site data."""
    run_dir, output_dir, site_dir = Path(run_dir), Path(output_dir), Path(site_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)
    manifest = _json(run_dir / "manifest.json")
    run_summary = _json(run_dir / "summary.json")
    failure = _json(run_dir / "failure.json")
    preflight = _json(run_dir / "preflight.json") or _json(run_dir / "preflight" / "preflight.json")
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    events = _run_events(run_dir)
    evaluations = load_jsonl(run_dir / "evaluation.jsonl")
    settings = manifest.get("config", {}).get("experiment", {})
    analysis = analyze_evaluations(evaluations, bootstrap_samples=settings.get("bootstrap_samples", 10000))
    for source, conditions in analysis["conditions"].items():
        planned = manifest.get(f"expected_{source}_tasks", settings.get(f"{source}_tasks"))
        for result in conditions.values():
            result["planned"] = planned
            result["unrecorded"] = max(0, planned - result["recorded_tasks"]) if isinstance(planned, int) else None
    status = run_summary.get("status") or (f"{failure.get('phase', 'run')}_failed" if failure else
        "preflight_passed" if preflight.get("success") else "in_progress" if metrics else manifest.get("status", "not_evaluated"))
    payload = {
        "schema_version": 1, "run_id": manifest.get("run_id", run_dir.name),
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status, "failure": failure, "budget": _json(run_dir / "budget.json"),
        "preparation": {name: _json(run_dir / f"{name}.json") for name in
                        ("preparation", "preparation_validation", "environment_validation", "validation")
                        if (run_dir / f"{name}.json").exists()},
        "manifest": manifest, "run_summary": run_summary, "preflight": preflight,
        "analysis": analysis, "metrics": metrics,
        "events": events, "patches": _run_patches(run_dir), "controls": _controls(),
        "diagnostics": _diagnostics(events, metrics, preflight, analysis["records"]),
        "sources": [{"title": title, "url": url} for title, url in SOURCES],
    }
    figures = render_figures(payload, output_dir / "figures")
    payload["figures"] = {key: f"figures/{value}" for key, value in figures.items()}
    _write_json(output_dir / "summary.json", payload)
    table = _table(payload["analysis"])
    controls = _control_table(payload["controls"])
    diagnostics = _diagnostic_table(payload["diagnostics"])
    failure_text = (f"Recorded failure in **{failure.get('phase', 'run')}**: {failure.get('error', 'Unspecified')}.\n\n"
                    if failure else "No failure metadata is recorded. This alone does not establish successful execution.\n\n")
    details = "\n\n".join(f"### {source.title()} bugs\n\n{_effect_text(comparison)}"
                            for source, comparison in payload["analysis"]["comparisons"].items())
    report = (f"# Experiment: {payload['run_id']}\n\nStatus: **{payload['status']}**. "
              "This report is generated from saved records; absent measurements remain unknown. "
              "All tables and diagnostics are derived from the [machine-readable analysis](summary.json). "
              "The [committed preregistration](preregistration.md) defines the hypotheses and decision rules.\n\n"
              + failure_text + f"## Evaluation\n\n{table}\n\n{details}\n\n"
              f"## Deferred controls\n\n{controls}\n\n"
              f"## Execution diagnostics\n\n{diagnostics}\n\n"
              "These are event-level counts, including incomplete collections; memory can include preflight. "
              "Unrecorded planned tasks remain missing coverage.\n\n"
              f"Invalidity reasons: `{json.dumps(payload['diagnostics']['invalid_reasons'], ensure_ascii=False)}`. "
              f"Untrained-group reasons: `{json.dumps(payload['diagnostics']['untrained_group_reasons'], ensure_ascii=False)}`.\n\n"
              "## Method and learnings\n\n"
              "One shared model serves the primary agents and asynchronous children. The pilot tests the combined "
              "system; it does not identify the causal contribution of delegation. Inspect the saved run summary "
              "below for observed failures, decisions, and completed work. No unrun control has a reported score.\n\n"
              f"```json\n{json.dumps(run_summary, ensure_ascii=False, indent=2)}\n```\n\n"
              "## Failure, preflight, and preparation records\n\n"
              f"```json\n{json.dumps({'failure': failure, 'preflight': preflight, 'preparation': payload['preparation']}, ensure_ascii=False, indent=2)}\n```\n\n"
              "## Recorded configuration\n\n"
              f"```json\n{json.dumps(manifest, ensure_ascii=False, indent=2)}\n```\n\n"
              "## Reproduction and limitations\n\n"
              "Regenerate this report with `uv run selfplay report --run <run-directory>`. "
              "Task-paired intervals condition on one training checkpoint; training-seed uncertainty is unmeasured. "
              "Degenerate bootstrap intervals do not establish equivalence. Missing evaluations and infrastructure "
              "errors are excluded from scored denominators and reported separately. Public-data pretraining "
              "contamination remains unknown.\n\n## Sources\n\n" +
              "\n".join(f"- [{title}]({url})" for title, url in SOURCES) + "\n")
    (output_dir / "experiment.md").write_text(report, encoding="utf-8")
    (output_dir / "experiments-summary.md").write_text(
        "# Experiments recorded in this run\n\n" + failure_text + table + "\n\n" + details +
        "\n\n## Deferred controls\n\n" + controls + "\n\n## Recorded diagnostics\n\n" + diagnostics +
        "\n\n## Network and experiment design\n\n"
        "| Condition | Architecture | Training in this experiment | Objective | Data and domain |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| base | Qwen3.5-2B; one shared policy for all roles | None | Frozen comparison | Held-out repository repair |\n"
        "| trained | Same base plus one shared language-model LoRA adapter | Recorded local RL updates | CWM-style clipped token objective with SSR team returns | Self-play on training repositories; held-out repository repair |\n"
        "\n## Terms in plain language\n\n"
        "- **Self-play:** the model generates challenges and learns from attempts to solve them.\n"
        "- **Shared policy:** every primary agent and subagent uses the same model parameters.\n"
        "- **LoRA:** small trainable adapter matrices modify a frozen base model.\n"
        "- **Oracle:** executable tests that determine whether the required behavior is satisfied.\n"
        "- **Paired comparison:** compare the two policies on the same task before averaging differences.\n"
        "- **Bootstrap interval:** uncertainty estimated by repeatedly resampling the observed tasks.\n"
        "\n\nThe base condition is the unchanged model; trained is the final completed checkpoint. "
        "Human bugs come from benchmark issues; synthetic bugs are independently generated diagnostic mutations. "
        "Resolution requires passing the executable oracle. An unscored task has no valid completed outcome. "
        "The larger controlled study and seed sensitivity remain unexecuted.\n", encoding="utf-8")
    sections = _paper_sections(payload)
    paper_md = '# Local multi-agent self-play RL for software engineering\n\nLibor "Burny" Burian\n\n'
    for title, body in sections:
        paper_md += f"## {title}\n\n{body}\n\n"
        if title == "Results":
            paper_md += table + "\n\n" + controls + "\n\n"
        elif title == "Observed execution diagnostics":
            paper_md += diagnostics + "\n\n"
    paper_md += "## Appendix: configuration\n\n```json\n" + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n```\n\n"
    paper_md += "## References\n\n" + "\n".join(f"{i}. [{title}]({url})" for i, (title, url) in enumerate(SOURCES, 1))
    paper_md += "\n\n## AI statement\n\n" + AI_STATEMENT + "\n"
    (output_dir / "paper.md").write_text(paper_md, encoding="utf-8")
    render_paper(payload, sections, output_dir / "paper.pdf", figures, output_dir / "figures")
    template_dir = Path(__file__).resolve().parents[2] / "site" / "template"
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(template_dir / name, site_dir / name)
    _write_json(site_dir / "data.json", payload)
    _write_json(site_dir / "summary.json", payload)
    if figures:
        shutil.copytree(output_dir / "figures", site_dir / "figures", dirs_exist_ok=True)
    for name in ("experiment.md", "experiments-summary.md", "paper.md", "paper.pdf"):
        if (output_dir / name).exists():
            shutil.copyfile(output_dir / name, site_dir / name)
    preregistration = Path(__file__).resolve().parents[2] / "reports" / "preregistration.md"
    if preregistration.exists():
        for destination in (output_dir / "preregistration.md", site_dir / "preregistration.md"):
            if destination.resolve() != preregistration.resolve():
                shutil.copyfile(preregistration, destination)
    if readme_path is not None:
        readme_path = Path(readme_path)
        links = ""
        if output_dir.resolve().is_relative_to(readme_path.resolve().parent):
            relative = output_dir.resolve().relative_to(readme_path.resolve().parent).as_posix()
            links = (f"\n\n![Recorded evaluation outcomes]({relative}/figures/scores.png)\n\n"
                     f"[Experiment report]({relative}/experiment.md) · [PDF paper]({relative}/paper.pdf)")
        _update_readme(readme_path, f"Recorded run status: **{status}**.\n\n" + failure_text + table + "\n\n" +
                       details + "\n\n" + controls + links + "\n\nRegenerate all results with `uv run selfplay report --run <run-directory>`. ")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    return {"summary.json": str(output_dir / "summary.json"), "experiment.md": str(output_dir / "experiment.md"),
            "paper.pdf": str(output_dir / "paper.pdf"), "index.html": str(site_dir / "index.html")}
