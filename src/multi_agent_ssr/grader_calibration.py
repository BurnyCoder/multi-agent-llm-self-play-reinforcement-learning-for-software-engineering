"""Replay fixed reference and empty patches through the historical CPU grading path.

Protocol: reports/reviews/grader-calibration-protocol-2026-11-26.md.
Upstream: https://github.com/SWE-bench/SWE-bench/blob/ca6e4e0d252f32f8762625b73575d5dee49d0a5a/swebench/harness/run_evaluation.py
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import uuid
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from .evaluation import grade_prediction
from .evidence import file_sha256
from .io import utc_now, write_json
from .prepare import command, data_directory

PROJECT = Path(__file__).resolve().parents[2]
STUDY = "grader-calibration-2026-11-26"
# Git LFS object at the frozen dataset revision, checked independently of the local cache.
# https://huggingface.co/api/datasets/SWE-bench/SWE-bench_Verified/tree/78f471bf655a3137b2e8a75af1501690ec009ec3/data
DATASET_SHA256 = "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25"


def pinned_inputs(data: Path):
    """Reconstruct the original selection from the full byte-verified dataset."""
    import pyarrow.parquet as pq

    public = json.loads((PROJECT / "manifests/evaluation.json").read_bytes())
    images = json.loads((PROJECT / "manifests/image_pins.json").read_bytes())
    snapshot = data / "huggingface/hub/datasets--SWE-bench--SWE-bench_Verified/snapshots" / public["revision"]
    parquet = snapshot / "data/test-00000-of-00001.parquet"
    if file_sha256(parquet) != DATASET_SHA256 or images["dataset_revision"] != public["revision"]:
        raise ValueError("Frozen dataset or image-manifest identity differs")
    population = pq.read_table(parquet).to_pylist()
    repos = {"psf/requests", "pallets/flask", "pylint-dev/pylint", "pytest-dev/pytest"}
    eligible = [r for r in population if r["repo"] in repos]
    chosen = sorted(eligible, key=lambda r: hashlib.sha256(r["instance_id"].encode()).digest())[:len(public["tasks"])]
    private = json.loads((data / "human_tasks_private.json").read_bytes())
    if private != chosen:
        raise ValueError("Private grading inputs differ from the selected pinned dataset rows")
    pins = {r["instance_id"]: r for r in images["tasks"]}
    if len(pins) != len(chosen) or len(chosen) != len(public["tasks"]):
        raise ValueError("Frozen task or image coverage differs")
    for row, declared in zip(chosen, public["tasks"], strict=True):
        pin = pins[row["instance_id"]]
        if any(row[k] != val for k, val in declared.items()) or pin["base_commit"] != row["base_commit"]:
            raise ValueError(f"Frozen identity differs for {row['instance_id']}")
    return chosen, pins, {"revision": public["revision"], "parquet_sha256": DATASET_SHA256,
                          "population": len(population), "eligible": len(eligible)}


def _grade_task(job):
    """Preserve each attempt and reuse only its terminal record on restart."""
    index, task, image, destination = job
    task_id = task["instance_id"]
    for condition in (("empty", "gold") if index % 2 == 0 else ("gold", "empty")):
        home = destination / "attempts" / task_id / condition
        admitted = list(home.glob("*/result.json"))
        if len(admitted) > 1:
            raise ValueError(f"Duplicate terminal controls for {task_id}/{condition}")
        if admitted:
            continue
        patch = task["patch"] if condition == "gold" else ""
        if condition == "gold" and not patch.strip():
            raise ValueError(f"Frozen human patch is empty for {task_id}")
        attempt = home / uuid.uuid4().hex
        start = {"task_id": task_id, "condition": condition, "attempt_id": attempt.name,
                 "started_at": utc_now(), "image_digest": image, "base_commit": task["base_commit"],
                 "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}
        write_json(attempt / "started.json", start)
        print(f"{start['started_at']} start {task_id} {condition}", flush=True)
        try:
            grade = grade_prediction(task, patch, "audit-" + condition, attempt, image_digest=image, timeout=300)
        except Exception as exc:
            # A grader exception is an indeterminate measurement. Preserve it and continue the fixed matrix.
            grade = {"status": "infra_error", "success": None, "reason": f"{type(exc).__name__}: {exc}"}
        relative_logs = sorted(attempt.glob("official/*/*/*"))
        grade.pop("official_log_dir", None)
        row = {**start, **grade, "finished_at": utc_now(),
               "logs": {f.relative_to(attempt).as_posix(): file_sha256(f)
                        for directory in relative_logs for f in directory.iterdir() if f.is_file()}}
        write_json(attempt / "result.json", row)
        print(f"{row['finished_at']} {task_id} {condition} {row['status']} {row['success']}", flush=True)


def run_calibration() -> None:
    """Hold one coordinator lock and give the two workers independent official grader state."""
    data = data_directory()
    if not data.is_dir():
        raise FileNotFoundError(data)
    tasks, pins, dataset = pinned_inputs(data)
    destination = data / STUDY
    destination.mkdir(exist_ok=True)
    with (destination / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digests = [pins[t["instance_id"]]["image_digest"] for t in tasks]
        local_ids = command(["docker", "image", "inspect", "--format", "{{.Id}}", *digests]).splitlines()
        inputs = {"dataset": dataset, "timeout_seconds": 300, "workers": 2, "swebench": version("swebench"),
                  "protocol_sha256": file_sha256(PROJECT / "reports/reviews/grader-calibration-protocol-2026-11-26.md"),
                  "code_sha256": {name: file_sha256(Path(__file__).with_name(name))
                                  for name in ("grader_calibration.py", "evaluation.py")},
                  "images": dict(zip(digests, local_ids, strict=True))}
        if inputs["swebench"] != "5.0.2":
            raise ValueError("Expected the pilot's locked SWE-bench version")
        # Import before recording attempts so an unavailable installation cannot manufacture infrastructure outcomes.
        from swebench.harness.run_evaluation import run_instance

        assert callable(run_instance)
        manifest = destination / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_bytes())["inputs"] != inputs:
                raise ValueError("Calibration inputs changed on resume")
        else:
            write_json(manifest, {"started_at": utc_now(), "commit": command(["git", "rev-parse", "HEAD"]),
                                  "inputs": inputs})
        jobs = [(i, t, digest, destination) for i, (t, digest) in enumerate(zip(tasks, digests, strict=True))]
        # Linux fork shares the imported read-only grader. Docker clients are constructed inside each worker.
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork")) as pool:
            list(pool.map(_grade_task, jobs, chunksize=1))
        terminal = list(destination.glob("attempts/*/*/*/result.json"))
        if len(terminal) != 2 * len(tasks):
            raise RuntimeError("The fixed control matrix is incomplete")
        print(f"{utc_now()} complete {len(terminal)} controls at {destination}", flush=True)
