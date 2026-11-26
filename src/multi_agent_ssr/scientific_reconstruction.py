"""Rebuild the independent review from original files and the new fixed-patch control logs.

Run with the project venv via `python -m multi_agent_ssr.scientific_reconstruction`.
Official parser semantics: https://github.com/SWE-bench/SWE-bench/blob/490635b2e9e775dca1e1d6b40ce9dbcff91e780f/swebench/harness/grading.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from .archived_training import json_rows, reconstruct_training
from .evidence import file_sha256
from .grader_calibration import PROJECT, STUDY, pinned_inputs
from .io import write_json
from .prepare import data_directory
from .scientific_review import REVIEW, calibration_outcomes, completion_bounds


def test_entries(reference, official, parsed, resolve):
    """Keep the upstream classification, actual parsed status, and alias for every expected entry."""
    entries = []
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        grades = {name: outcome for outcome, names in official.get(category, {}).items() for name in names}
        for name in reference[category]:
            key = resolve(name, parsed)
            entries.append({"category": category, "expected_id": name, "parsed_id": key,
                            "parsed_status": parsed[key] if key is not None else "MISSING",
                            "official_class": grades.get(name, "unclassified"),
                            "progress_marker": bool(re.fullmatch(r"\[\d+%\]", name))})
    return entries


def replay(task, condition, patch, directory, recorded):
    """Reparse original stdout, verify the cache and embedded report, then preserve individual test entries."""
    from swebench.harness.grading import _resolve_case, get_eval_report, get_logs_eval
    from swebench.harness.utils import make_test_spec

    log = directory / "test_output.txt"
    cache = directory / "report.json"
    spec = make_test_spec(task)
    prediction = {"instance_id": task["instance_id"], "model_name_or_path": "local-ssr-" + condition,
                  "model_patch": patch}
    report = get_eval_report(spec, prediction, str(log), include_tests_status=True)[task["instance_id"]]
    cached = json.loads(cache.read_bytes())[task["instance_id"]] if cache.exists() else None
    if (cached is not None and report != cached) or (recorded is not None and report != recorded):
        raise ValueError(f"Official reconstruction disagrees for {task['instance_id']}/{condition}")
    parsed, readable = get_logs_eval(spec, str(log))
    entries = test_entries(task, report.get("tests_status", {}), parsed, _resolve_case)
    stdout = log.read_text(encoding="utf-8", errors="replace").lower()
    markers = [s for s in ("temporary failure in name resolution", "network is unreachable", "no module named 'appdirs'")
               if s in stdout]
    return {"parser_accepted_log": readable, "resolved": report["resolved"], "cache_present": cached is not None,
            "test_output_sha256": file_sha256(log), "cached_report_sha256": file_sha256(cache) if cached else None,
            "stdout_markers": markers, "entries": entries}


def _failure_summary(details):
    """Count entries including repeated tasks across conditions. Preserve unique task denominators."""
    summary = {}
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        failures = [(r, t) for r in details for t in r["entries"]
                    if t["category"] == category and t["official_class"] == "failure"]
        summary[category] = {"entries": len(failures), "by_parsed_status": dict(Counter(t["parsed_status"] for r, t in failures)),
                             "rows": len({(r["task_id"], r["condition"]) for r, t in failures}),
                             "task_ids": sorted({r["task_id"] for r, t in failures})}
    return summary


def paired_results(ids, rows, draws):
    """Recalculate the registered comparison from matched binary outcomes."""
    scored = {(r["task_id"], r["condition"]): int(r["success"]) for r in rows if r["status"] == "completed"}
    paired = sorted(t for t in ids if (t, "base") in scored and (t, "trained") in scored)
    deltas = [scored[t, "trained"] - scored[t, "base"] for t in paired]
    rng = random.Random(0)
    interval = None
    if deltas:
        means = [sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(draws)]
        quantiles = statistics.quantiles(means, n=40, method="inclusive")
        interval = [quantiles[0], quantiles[-1]]
    return {"cohort": len(ids), "paired": len(paired), "task_ids": paired, "deltas": deltas,
            "difference": statistics.mean(deltas) if deltas else None, "ci95": interval,
            "samples": draws, "seed": 0}


def historical_scores(run, tasks):
    """Admit one scored outcome per task/condition while retaining the complete attempt history."""
    records, accepted, details = {}, set(), []
    private = {t["instance_id"]: t for t in tasks}
    attempts = []
    for row in json_rows(run / "evaluation.jsonl"):
        if row["source"] != "human":
            continue
        tid, c = row["task_id"], row["condition"]
        key = (tid, c)
        if tid not in private or c not in {"base", "trained"}:
            raise ValueError(f"Unexpected historical task or condition: {key}")
        brief = {k: row.get(k) for k in ("task_id", "condition", "timestamp", "status", "success", "attempted")}
        attempts.append(brief)
        if key not in accepted:
            records[key] = brief
        if row["status"] != "completed":
            if row["success"] is not None:
                raise ValueError("An unscored row has a success value")
            continue
        if key in accepted or type(row["success"]) is not bool:
            raise ValueError(f"Ambiguous completed historical outcome: {key}")
        accepted.add(key)
        prediction = run / "human" / c / tid / "prediction.diff"
        folder = run / "official" / Path(*Path(row["official_log_dir"]).parts[-3:])
        check = replay(private[tid], c, prediction.read_text(), folder, row["official_report"])
        if not check["cache_present"] or check["resolved"] != row["success"]:
            raise ValueError("Historical cache or recorded score disagrees")
        details.append({"task_id": tid, "condition": c, "phase": "historical", "patch_sha256": file_sha256(prediction),
                        "patch_bytes": prediction.stat().st_size, "log_directory": folder.relative_to(run).as_posix(), **check})
    return list(records.values()), attempts, details


def marker_sensitivity(details):
    """Check all marked historical rows for independent fail-to-pass failures."""
    marked = [r for r in details if any(e["progress_marker"] for e in r["entries"])]
    return {"rows": len(marked), "separate_fail_to_pass_failure": sum(
        any(e["category"] == "FAIL_TO_PASS" and not e["progress_marker"] and e["official_class"] == "failure"
            for e in r["entries"]) for r in marked)}


def calibration_scores(root, tasks, pins):
    """Validate terminal records and their immutable log hashes, preserving incomplete attempts."""
    rows, trace, attempts = [], [], []
    for task in tasks:
        tid = task["instance_id"]
        for condition in ("empty", "gold"):
            directory = root / "attempts" / tid / condition
            results = list(directory.glob("*/result.json"))
            if len(results) != 1:
                raise ValueError(f"Expected one terminal control for {tid}/{condition}, found {len(results)}")
            for start in sorted(directory.glob("*/started.json")):
                attempts.append({**json.loads(start.read_bytes()), "terminal": (start.parent / "result.json").exists()})
            path = results[0]
            row = json.loads(path.read_bytes())
            patch = "" if condition == "empty" else task["patch"]
            if (row["task_id"] != tid or row["condition"] != condition or row["attempt_id"] != path.parent.name
                    or row["image_digest"] != pins[tid]["image_digest"] or row["base_commit"] != task["base_commit"]
                    or row["patch_sha256"] != hashlib.sha256(patch.encode()).hexdigest()):
                raise ValueError("Control identity or patch changed")
            for relative, digest in row["logs"].items():
                if file_sha256(path.parent / relative) != digest:
                    raise ValueError(f"Changed control log {relative}")
            logs = list(path.parent.glob("official/*/*/*/test_output.txt"))
            if len(logs) > 1 or (not logs and row["status"] == "completed"):
                raise ValueError("Missing or ambiguous completed control stdout")
            diagnostics = {}
            if logs:
                check = replay(task, "audit-" + condition, patch, logs[0].parent, row.get("official_report"))
                if row["status"] == "completed" and check["resolved"] != row["success"]:
                    raise ValueError("Control score differs on log replay")
                trace.append({"task_id": tid, "condition": condition, "phase": "calibration", **check})
                diagnostics = {"stdout_markers": check["stdout_markers"], "failure_entries": {
                    c: sum(e["category"] == c and e["official_class"] == "failure" for e in check["entries"])
                    for c in ("FAIL_TO_PASS", "PASS_TO_PASS")}}
            rows.append({**{k: v for k, v in row.items() if k != "official_report"},
                         **diagnostics, "repository": task["repo"], "result_sha256": file_sha256(path)})
    return rows, attempts, trace


def _synthetic(data, run):
    """Verify all public/private mutant identities and stored oracle transitions without running a model."""
    private = json.loads((data / "synthetic/tasks.json").read_bytes())
    public = json.loads((PROJECT / "manifests/synthetic.json").read_bytes())["tasks"]
    for row, frozen in zip(private, public, strict=True):
        clean, mutant = row["clean_results"], row["mutant_results"]
        if (any(row[k] != v for k, v in frozen.items()) or clean.keys() != mutant.keys()
                or set(clean.values()) != {"passed"} or set(mutant.values()) != {"passed", "failed"}
                or sorted(k for k, v in mutant.items() if v == "failed") != sorted(row["fail_to_pass"])
                or sorted(k for k, v in mutant.items() if v == "passed") != sorted(row["pass_to_pass"])
                or hashlib.sha256(row["bug_patch"].encode()).hexdigest() != row["patch_sha256"]
                or file_sha256(data / "synthetic" / row["buggy_tree"] / row["source_file"]) != row["source_sha256"]):
            raise ValueError("Synthetic identity or saved oracle evidence differs")
    ids = [r["task_id"] for r in private]
    rows = [r for r in json_rows(run / "evaluation.jsonl") if r["source"] == "synthetic"]
    if len({r["cluster_id"] for r in private}) != len(ids):
        raise ValueError("Repeated synthetic function")
    return {"task_ids": ids, "distinct_functions": len({r["cluster_id"] for r in private}),
            "clean_test_counts": sorted({len(r["clean_results"]) for r in private}),
            "coverage": completion_bounds(ids, rows), "scored": sum(r["status"] == "completed" for r in rows)}


def reconstruct(run, data, destination):
    """Produce portable evidence from originals. Previously published audits are never inputs."""
    tasks, pins, dataset = pinned_inputs(data)
    ids = [t["instance_id"] for t in tasks]
    original = json.loads((run / "manifest.json").read_bytes())
    if original["expected_human_tasks"] != len(ids):
        raise ValueError("Selected cohort size differs from the original pilot manifest")
    if any(r["source"] not in {"human", "synthetic"} for r in json_rows(run / "evaluation.jsonl")):
        raise ValueError("Historical attempt has an unknown evaluation source")
    raw = data / STUDY
    experiment = json.loads((raw / "manifest.json").read_bytes())
    protocol = PROJECT / "reports/reviews/grader-calibration-protocol-2026-11-26.md"
    inputs = experiment["inputs"]
    if (inputs["dataset"] != dataset or inputs["protocol_sha256"] != file_sha256(protocol)
            or inputs["timeout_seconds"] != 300 or inputs["workers"] != 2 or inputs["swebench"] != "5.0.2"
            or set(inputs["images"]) != {p["image_digest"] for p in pins.values()}
            or any(file_sha256(Path(__file__).with_name(k)) != v for k, v in inputs["code_sha256"].items())):
        raise ValueError("Calibration environment disagrees with its registered protocol and runner")
    history, history_attempts, original_tests = historical_scores(run, tasks)
    controls, control_attempts, new_tests = calibration_scores(raw, tasks, pins)
    calibration = calibration_outcomes(ids, controls)
    passed = [r["task_id"] for r in calibration if r["discrimination"] == "pass"]
    draws = original["config"]["experiment"]["bootstrap_samples"]
    synthetic = _synthetic(data, run)
    if original["expected_synthetic_tasks"] != len(synthetic["task_ids"]):
        raise ValueError("Selected synthetic cohort size differs from the original manifest")
    if not {t["instance_id"] for t in tasks} == {item["task_id"] for item in calibration}:
        raise ValueError("Calibration did not cover the original task identities")
    result = {"schema": 1, "date": "2026-11-26", "dataset": dataset,
              "run_sha256": {name: file_sha256(run / name) for name in
                             ("manifest.json", "evaluation.jsonl", "metrics.jsonl", "events.jsonl", "model.jsonl", "latest.json",
                              "budget.json", "failure.json", "protocol_diagnostics.json")},
              "training": reconstruct_training(run, tasks), "synthetic": synthetic,
              "historical": {"attempts": history_attempts, "retained": history, "test_summary": _failure_summary(original_tests),
                             "empty_submissions": sum(r["patch_bytes"] == 0 for r in original_tests),
                             "replayed_reports": len(original_tests), "progress_marker_tasks": sorted({r["task_id"]
                                 for r in original_tests if any(t["progress_marker"] for t in r["entries"])}),
                             "marker_sensitivity": marker_sensitivity(original_tests),
                             "bounds": completion_bounds(ids, history), "paired": paired_results(ids, history, draws)},
              "calibration": {"manifest": experiment, "manifest_sha256": file_sha256(raw / "manifest.json"), "tasks": calibration,
                              "counts": dict(Counter(r["discrimination"] for r in calibration)), "attempts": control_attempts,
                              "controls": controls, "test_summary": {c: _failure_summary([r for r in new_tests if r["condition"] == c])
                                                                    for c in ("empty", "gold")},
                              "sensitivity": paired_results(passed, history, draws)}}
    destination.mkdir(parents=True, exist_ok=True)
    # Per-test evidence is losslessly compressed and excludes upstream source patches and host credentials.
    archive = destination / "oracle-tests.json.gz"
    material = json.dumps(original_tests + new_tests, ensure_ascii=False, separators=(",", ":")).encode()
    archive.write_bytes(gzip.compress(material, mtime=0))
    result["test_archive"] = {"name": archive.name, "sha256": file_sha256(archive)}
    write_json(destination / "evidence.json", result)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    reconstruct(PROJECT / "runs/pilot-recovery", data_directory(), REVIEW)
