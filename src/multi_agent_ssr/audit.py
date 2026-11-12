"""Independently recompute saved evidence without reporting code or model loading.

This second calculation checks accounting, not the scientific validity of an
oracle or completeness of logging. Raw records remain untouched. Bootstrap
percentiles follow https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    """Read only explicit JSONL streams; missing streams have no recorded rows."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def recompute(run: Path, samples: int = 10000) -> dict:
    """Return checkable observations from raw evaluations, actions and collections."""
    streams = [run / "evaluation.jsonl", run / "metrics.jsonl", run / "events.jsonl"]
    events = _rows(streams[2])
    for path in sorted(run.glob("synthetic/*/*/attempt-*/agents/events.jsonl")):
        streams.append(path)
        events.extend(_rows(path))
    rows, metrics = _rows(streams[0]), _rows(streams[1])
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["condition"], row["task_id"])].append(row)
    admitted = {}
    duplicates = []
    for key, attempts in grouped.items():
        completed = [row for row in attempts if row.get("status") == "completed" and type(row.get("success")) is bool]
        if len(completed) > 1:
            duplicates.append({"key": list(key), "records": len(completed),
                               "identical": all(row == completed[0] for row in completed)})
        admitted[key] = completed[0] if completed else attempts[-1]
    counts, pairs = {}, {}
    for source in sorted({"human", "synthetic"} | {key[0] for key in grouped}):
        scores = {}
        counts[source] = {}
        for condition in ("base", "trained"):
            selected = [row for key, row in admitted.items() if key[:2] == (source, condition)]
            scores[condition] = {row["task_id"]: int(row["success"]) for row in selected
                                 if row.get("status") == "completed" and type(row.get("success")) is bool}
            counts[source][condition] = {"recorded_tasks": len(selected), "scored": len(scores[condition]),
                                         "resolved": sum(scores[condition].values()),
                                         "unscored": len(selected) - len(scores[condition]),
                                         "statuses": dict(Counter(row.get("status", "unknown") for row in selected))}
        ids = sorted(scores["base"].keys() & scores["trained"].keys())
        deltas = [scores["trained"][key] - scores["base"][key] for key in ids]
        interval = None
        if deltas:
            rng = random.Random(0)
            draws = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(samples))
            interval = []
            for percentile in (0.025, 0.975):
                position = percentile * (samples - 1)
                index = int(position)
                interval.append(draws[index] + (draws[min(index + 1, samples - 1)] - draws[index]) * (position - index))
        pairs[source] = {"task_ids": ids, "deltas": deltas,
                         "difference": sum(deltas) / len(deltas) if deltas else None, "ci95": interval}
    calls = {}
    for row in events:
        if row.get("event") == "generation":
            key = row["call_id"]
            if key in calls and calls[key] != row["generation"]:
                raise ValueError("Conflicting generation records for one call_id")
            calls[key] = row["generation"]
    checks = [row for row in events if row.get("event") == "artifact_validated" and type(row.get("valid")) is bool]
    official = {"scored_human_records": 0, "embedded_reports": 0, "embedded_score_agreements": 0,
                "cached_reports_available": 0, "cached_report_agreements": 0,
                "saved_patches": 0, "empty_patches": 0, "pass_to_pass_failure_entries": 0,
                "rows_with_pass_to_pass_failures": 0}
    for row in admitted.values():
        if row.get("source") != "human" or row.get("status") != "completed" or type(row.get("success")) is not bool:
            continue
        official["scored_human_records"] += 1
        report = row.get("official_report")
        if report:
            official["embedded_reports"] += 1
            official["embedded_score_agreements"] += report.get("resolved") == row["success"]
            failures = report.get("tests_status", {}).get("PASS_TO_PASS", {}).get("failure", [])
            official["pass_to_pass_failure_entries"] += len(failures)
            official["rows_with_pass_to_pass_failures"] += bool(failures)
            # Inspect only a cache within this run, never follow an exported host path.
            suffix = row.get("official_log_dir", "").partition("/official/")[2]
            cache = (run / "official" / suffix / "report.json").resolve()
            if suffix and cache.is_relative_to(run.resolve()) and cache.is_file():
                cached = json.loads(cache.read_text())
                official["cached_reports_available"] += 1
                official["cached_report_agreements"] += cached.get(row["task_id"]) == report
        patch = run / "human" / row["condition"] / row["task_id"] / "prediction.diff"
        if patch.is_file():
            official["saved_patches"] += 1
            official["empty_patches"] += not patch.read_text().strip()
    collections = []
    for path in sorted(run.glob("collection_groups/*.json")):
        record = json.loads(path.read_text())
        episode_paths = [run / "episodes" / (identifier + ".json")
                         for group in record.get("episode_groups", []) for identifier in group]
        available = bool(episode_paths) and all(path.is_file() for path in episode_paths)
        episodes = [json.loads(path.read_text()) for path in episode_paths] if available else []
        collections.append({"path": path.relative_to(run).as_posix(),
                            "episode_rewards": [row["reward"] for row in episodes] if available else None,
                            "action_tokens": sum(len(action["generation"]["completion_ids"]) for row in episodes for action in row["actions"]) if available else None,
                            "recorded_sampled_tokens": record.get("update", {}).get("sampled_tokens"),
                            "optimizer_steps": record.get("update", {}).get("optimizer_steps"),
                            "optimized_field_meaning": "update routine completed; inspect optimizer_steps for actual updates"})
    values = [row.get("optimizer_steps") for row in metrics]
    complete_actions = bool(calls) and all(isinstance(row.get("completion_ids"), list) for row in calls.values())
    complete_probabilities = complete_actions and all(isinstance(row.get("old_logprobs"), list) for row in calls.values())
    inputs = {path.relative_to(run).as_posix(): {"bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in streams if path.is_file()}
    return {"method": "independent raw-record recomputation; no model execution or prior audit conclusions",
            "input_streams": inputs, "raw_evaluation_rows": len(rows), "conditions": counts, "paired": pairs,
            "duplicate_completed_outcomes": duplicates, "official": official, "collections": collections,
            "action_calls": len(calls), "action_tokens": sum(len(row["completion_ids"]) for row in calls.values()) if complete_actions else None,
            "action_token_logprob_lengths_agree": all(len(row["completion_ids"]) == len(row["old_logprobs"]) for row in calls.values()) if complete_probabilities else None,
            "delegation_calls": sum(row.get("event") == "tool_started" and row.get("name") == "subagent" for row in events),
            "child_sessions": len({row["session_id"] for row in events if row.get("event") == "session_started" and row.get("parent_session_id")}),
            "artifact_validation_checks": len(checks), "valid_artifacts": sum(row["valid"] for row in checks),
            "invalidity_reasons": dict(Counter(row.get("reason") for row in checks if not row["valid"])),
            "optimizer_steps": sum(values) if values and all(type(value) is int and value >= 0 for value in values) else None,
            "limitations": "Counts describe available records. Missing original grader caches in an export cannot be rechecked. No clean-environment regrading or contamination test was performed."}
