"""Pinned human-bug rollouts with the official SWE-bench scoring implementation.

SWE-bench 5.0.2 provides make_test_spec and run_instance. Its test oracle remains
unchanged; a Docker client wrapper restricts the grader's privileges and network.
Source: https://www.swebench.com/SWE-bench/guides/evaluation/
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agents import run_team
from .contracts import Limits, Policy, RepoSpec
from .io import EventLog, utc_now, write_json
from .prepare import data_directory
from .sandbox import DockerSandbox
from .validation import source_diff

_GRADER_LOCK = threading.Lock()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _command(arguments: list[str], *, timeout: int = 1800, stream: bool = False) -> str:
    """Setup uses fixed argument arrays; generated code is never executed here."""
    result = subprocess.run(arguments, check=True, text=True, timeout=timeout,
                            stdout=None if stream else subprocess.PIPE, stderr=None if stream else subprocess.PIPE)
    return (result.stdout or "").strip()


def source_roots(source: Path) -> list[str]:
    """Restrict submitted patches to the selected Python packages, never tests."""
    candidates = ["src", "requests", "flask", "pylint", "_pytest", "pytest.py", "py"]
    roots = [name for name in candidates if (source / name).exists()]
    if not roots:
        raise ValueError(f"No supported Python package layout in {source}")
    return roots


def _clean_agent_source(source: Path) -> None:
    """Remove control directories and links without following targets outside the copy."""
    for directory, dirs, files in os.walk(source, followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                path.unlink()
                dirs.remove(name)
            elif name in {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                resolved = path.resolve()
                if not resolved.is_relative_to(source.resolve()):
                    raise ValueError("Refusing cleanup outside the extracted source copy")
                shutil.rmtree(path)
                dirs.remove(name)
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or name == ".env" or name.startswith(".env.") and name != ".env.example":
                path.unlink()


def _verify_base_content(container: str, commit: str) -> None:
    """Official image commits may normalize modes but must retain all base content.

    core.fileMode=false only ignores worktree/index mode differences. The image's
    synthetic index can itself differ from the base tree, so examine its patch.
    Additions, deletions, renames, binary changes, and textual hunks all fail.
    """
    diff = _command(["docker", "exec", container, "git", "-C", "/testbed", "diff",
                     "--no-ext-diff", "--no-textconv", "--no-renames", commit, "--"])
    if any(not line.startswith(("diff --git ", "old mode ", "new mode ")) for line in diff.splitlines()):
        raise ValueError("Official image source content differs from the pinned base commit")


def _prepare_one(task: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Pin the official image and derive an agent image with no original checkout."""
    instance_id = task["instance_id"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", instance_id) or not re.fullmatch(r"[0-9a-f]{40}", task["base_commit"]):
        raise ValueError("Invalid task identity or commit in pinned manifest")
    image = task["image"]
    requested = task.get("image_digest", image)
    _command(["docker", "pull", requested], stream=True)
    inspection = json.loads(_command(["docker", "image", "inspect", requested]))[0]
    digests = inspection.get("RepoDigests", [])
    if not digests:
        raise ValueError(f"Official image has no registry digest: {image}")
    if "@sha256:" in requested:
        if not any(digest.rsplit("@", 1)[-1] == requested.rsplit("@", 1)[-1] for digest in digests):
            raise ValueError("Pulled official image does not match the committed digest")
        pinned = requested
    else:
        pinned = digests[0]
    directory = destination / "human_environments" / instance_id
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source"
    temporary = directory / f"source-{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    container = _command(["docker", "create", "--network", "none", "--entrypoint", "sleep", pinned, "infinity"])
    try:
        _command(["docker", "start", container])
        # Images may replace Git metadata with a synthetic commit, but their
        # tracked working tree must equal the declared problem's base commit.
        _verify_base_content(container, task["base_commit"])
        _command(["docker", "exec", container, "test", "-x", "/opt/miniconda3/envs/testbed/bin/python"])
        _command(["docker", "cp", f"{container}:/testbed/.", str(temporary)], timeout=600)
    finally:
        _command(["docker", "rm", "--force", container], timeout=60)
    _clean_agent_source(temporary)
    if source.exists():
        if not source.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Refusing to replace a source outside the task data directory")
        shutil.rmtree(source)
    temporary.replace(source)
    roots = source_roots(source)
    # The dependency image itself must not expose /testbed or its Git objects.
    # /usr/local/bin aliases also survive a login shell resetting PATH.
    dockerfile = (
        f"FROM {pinned}\n"
        "USER root\n"
        "RUN rm -rf /testbed /root/.cache/pip /root/.bash_history && mkdir -p /testbed "
        "&& ln -sf /opt/miniconda3/envs/testbed/bin/python /usr/local/bin/python "
        "&& ln -sf /opt/miniconda3/envs/testbed/bin/python /usr/local/bin/python3\n"
        "ENV PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "ENV PYTHONPATH=/repo:/repo/src PYTHONDONTWRITEBYTECODE=1 HOME=/tmp\n"
        "WORKDIR /repo\n"
    )
    build_dir = directory / "build"
    build_dir.mkdir(exist_ok=True)
    (build_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    tag = "multi-agent-ssr-eval:" + hashlib.sha256((instance_id + pinned).encode()).hexdigest()[:16]
    _command(["docker", "build", "-t", tag, str(build_dir)], timeout=1200, stream=True)
    agent_id = _command(["docker", "image", "inspect", tag, "--format", "{{.Id}}"])
    _command(["docker", "run", "--rm", "--network", "none", "--user", "65534:65534",
              "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "sh",
              agent_id, "-lc", "test ! -e /testbed/.git && test -z \"$(ls -A /testbed)\" && python --version"])
    return {"instance_id": instance_id, "repo": task["repo"], "base_commit": task["base_commit"],
            "source_path": str(source), "image": image, "image_digest": pinned,
            "agent_image": agent_id, "agent_image_tag": tag, "source_roots": roots,
            "test_roots": [name for name in ["tests", "testing", "test"] if (source / name).exists()],
            "status": "ready", "prepared_at": utc_now()}


def prepare_human_environments(data_dir: Path) -> dict[str, Any]:
    """Prepare at most two images concurrently and retain every fixed task/error."""
    data_dir = Path(data_dir).resolve()
    manifest = json.loads((_PROJECT_ROOT / "manifests" / "evaluation.json").read_text())
    pins_path = _PROJECT_ROOT / "manifests" / "image_pins.json"
    pin_document = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    if pin_document and pin_document.get("dataset_revision") != manifest["revision"]:
        raise ValueError("Official image pins belong to a different dataset revision")
    pins = {row["instance_id"]: row for row in pin_document.get("tasks", [])}
    previous_path = data_dir / "human_envs.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {"tasks": []}
    ready = {row["instance_id"]: row for row in previous["tasks"] if row.get("status") == "ready"}
    records: dict[str, dict[str, Any]] = {}
    jobs = []
    for public_task in manifest["tasks"]:
        task = dict(public_task)
        pin = pins.get(task["instance_id"], {})
        if pin and pin.get("base_commit") != task["base_commit"]:
            raise ValueError("Official image pin belongs to a different base commit")
        if pin.get("image_digest"):
            task["image_digest"] = pin["image_digest"]
        old = ready.get(task["instance_id"])
        if (old and old.get("base_commit") == task["base_commit"] and Path(old["source_path"]).is_dir()
                and (not task.get("image_digest") or old.get("image_digest") == task["image_digest"])):
            try:
                _command(["docker", "image", "inspect", old["agent_image"]], timeout=30)
                _command(["docker", "image", "inspect", old["image_digest"]], timeout=30)
                records[task["instance_id"]] = old
                continue
            except (subprocess.SubprocessError, OSError):
                pass
        jobs.append(task)

    def persist() -> dict[str, Any]:
        """The public manifest contains image pins, never host paths or gold tests."""
        result = {"dataset_revision": manifest["revision"],
                  "tasks": [records[task["instance_id"]] for task in manifest["tasks"] if task["instance_id"] in records]}
        write_json(previous_path, result)
        public_keys = ["instance_id", "base_commit", "image", "image_digest", "agent_image", "status", "reason"]
        # Original oracle pins are immutable. Rebuilt dependency-only images can
        # receive a new local ID; that ID belongs in the local environment record.
        public_rows = []
        for task in manifest["tasks"]:
            pin = pins.get(task["instance_id"], {})
            row = records.get(task["instance_id"], {})
            if pin.get("image_digest"):
                public_rows.append(pin)
            elif row:
                public_rows.append({key: row[key] for key in public_keys if key in row})
            elif pin:
                public_rows.append(pin)
        write_json(pins_path, {"dataset_revision": manifest["revision"], "tasks": public_rows})
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_prepare_one, task, data_dir): task for task in jobs}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                result = {**task, "status": "infra_error", "reason": f"{type(exc).__name__}: {exc}"}
            records[task["instance_id"]] = result
            print(json.dumps({"event": "human_environment_prepared", **{key: value for key, value in result.items() if key != "source_path"}}), flush=True)
            persist()
    return persist()


class _HardenedContainers:
    """Enforce local isolation even if upstream defaults request extra privilege."""

    def __init__(self, containers):
        self._containers = containers

    def create(self, *args, **kwargs):
        kwargs.pop("cap_add", None)
        kwargs.update(network_mode="none", cap_drop=["ALL"], security_opt=["no-new-privileges:true"],
                      privileged=False, mem_limit="2g", memswap_limit="2g", nano_cpus=1_000_000_000,
                      pids_limit=256)
        return self._containers.create(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._containers, name)


class HardenedDockerClient:
    """Delegate official scoring operations while constraining container creation."""

    def __init__(self, client):
        self._client = client
        self.containers = _HardenedContainers(client.containers)

    def __getattr__(self, name):
        return getattr(self._client, name)


def agent_prompt(task: dict[str, Any]) -> str:
    """Whitelist the issue description; never serialize the private task record."""
    return (f"Repair this Python repository issue ({task['instance_id']}). Inspect source in /repo. "
            "Delegate independent investigation when useful; continue your own work while children run. "
            "Modify source code only, not tests or test configuration. Use python -m pytest for local tests. "
            "Finish with a concise description of the repair.\n\nIssue:\n" + task["problem_statement"])


def grade_prediction(task: dict[str, Any], patch: str, condition: str, run_dir: Path,
                     *, image_digest: str, timeout: int) -> dict[str, Any]:
    """Run the official oracle in a fresh hardened container, retaining its report."""
    import docker
    from swebench.harness.utils import make_test_spec

    official = importlib.import_module("swebench.harness.run_evaluation")
    try:
        client = docker.from_env(timeout=timeout + 120)
    except docker.errors.DockerException as exc:
        return {"success": None, "status": "infra_error", "reason": f"Docker grader unavailable: {exc}"}
    spec = make_test_spec({**task, "image": image_digest})
    prediction = {"instance_id": task["instance_id"], "model_name_or_path": f"local-ssr-{condition}", "model_patch": patch}
    # Upstream reuses report.json by run ID without checking the submitted patch.
    # A fresh attempt ID prevents interrupted/retried rollouts receiving stale
    # scores. Completed task idempotency is enforced by evaluate_human instead.
    run_id = hashlib.sha256(str(Path(run_dir).resolve()).encode()).hexdigest()[:16] + "-" + uuid.uuid4().hex[:8]
    # Upstream stores artifacts under a module-level root. Serialize this scoped
    # override; no process-wide chdir can race with agent or logging threads.
    with _GRADER_LOCK:
        old_root = official.RUN_EVALUATION_LOG_DIR
        official.RUN_EVALUATION_LOG_DIR = Path(run_dir).resolve() / "official"
        try:
            outcome = official.run_instance(spec, prediction, HardenedDockerClient(client), run_id,
                                            timeout=timeout, rewrite_reports=False, skip_patch=not patch.strip())
        except docker.errors.DockerException as exc:
            return {"success": None, "status": "infra_error", "reason": f"Docker grader failed: {exc}"}
        finally:
            official.RUN_EVALUATION_LOG_DIR = old_root
            client.close()
    report_path = Path(run_dir).resolve() / "official" / run_id / prediction["model_name_or_path"] / task["instance_id"]
    if outcome is None:
        return {"success": None, "status": "infra_error", "reason": "Official harness produced no score; inspect grader logs",
                "official_log_dir": str(report_path)}
    _, report = outcome
    result = report.get(task["instance_id"], {})
    if result.get("infra_failure"):
        return {"success": None, "status": "infra_error", "reason": result.get("infra_failure_reason", "Official infrastructure failure"),
                "official_log_dir": str(report_path)}
    if type(result.get("resolved")) is not bool:
        return {"success": None, "status": "infra_error", "reason": "Official report has no boolean resolved result",
                "official_log_dir": str(report_path)}
    return {"success": result["resolved"], "status": "completed", "reason": "Official SWE-bench test evaluation",
            "official_log_dir": str(report_path), "official_report": result}


def _existing_results(run_dir: Path, condition: str) -> dict[str, dict[str, Any]]:
    """Reuse completed outcomes during resume, never choose the more favorable one."""
    path = run_dir / "evaluation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    results = {}
    for row in rows:
        if row.get("condition") == condition and row.get("source") == "human" and row.get("status") == "completed":
            previous = results.get(row["task_id"])
            if previous and previous.get("success") != row.get("success"):
                raise ValueError("Conflicting completed human evaluation results")
            results[row["task_id"]] = row
    return results


def _pinned_private_tasks(destination: Path) -> list[dict[str, Any]]:
    """Check private grading records against the committed public task manifest."""
    tasks = json.loads((destination / "human_tasks_private.json").read_text())
    public = json.loads((_PROJECT_ROOT / "manifests" / "evaluation.json").read_text())["tasks"]
    def identity(row):
        return row["instance_id"], row["repo"], row["base_commit"]

    if [identity(row) for row in tasks] != [identity(row) for row in public]:
        raise ValueError("Private evaluation tasks do not match the pinned public cohort")
    return tasks


def evaluate_human(policy: Policy, condition: str, run_dir: Path, limits: Limits,
                   deadline: float, delegation: bool = True) -> list[dict[str, Any]]:
    """Evaluate the fixed cohort in order; deadline omissions remain explicitly unscored."""
    destination = data_directory()
    tasks = _pinned_private_tasks(destination)
    envs = {row["instance_id"]: row for row in json.loads((destination / "human_envs.json").read_text())["tasks"]}
    run_dir = Path(run_dir)
    log = EventLog(run_dir / "evaluation.jsonl")
    existing = _existing_results(run_dir, condition)
    rows = []
    for task in tasks:
        task_id = task["instance_id"]
        if task_id in existing:
            rows.append(existing[task_id])
            continue
        started = time.monotonic()
        row: dict[str, Any] = {"task_id": task_id, "condition": condition, "source": "human", "success": None,
                               "status": "pending", "generated_tokens": 0, "delegations": 0,
                               "policy_version": getattr(policy, "version", None)}
        env = envs.get(task_id, {})
        if deadline - time.time() < 120:
            row["reason"] = "Insufficient remaining pilot time to start rollout and grading"
        elif env.get("status") != "ready":
            row.update(status="infra_error", reason=env.get("reason", "Pinned environment unavailable"))
        elif env.get("base_commit") != task["base_commit"]:
            row.update(status="infra_error", reason="Prepared environment commit differs from pinned task")
        else:
            task_dir = run_dir / "human" / condition / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            available = max(1, int(deadline - time.time() - 90))
            local_limits = replace(limits, episode_seconds=min(limits.episode_seconds, available))
            spec = RepoSpec(task_id, "https://github.com/" + task["repo"], task["base_commit"], env["agent_image"],
                            env["source_path"], "test", env["source_roots"], env["test_roots"], image_digest=env["agent_image"])
            stage = "environment"
            try:
                with DockerSandbox(spec.image, Path(spec.source_path), task_dir / "sandbox") as sandbox:
                    stage = "rollout"
                    episode = run_team(policy, sandbox, "solver", task_id, agent_prompt(task), run_dir,
                                       limits=local_limits, delegation=delegation)
                    write_json(task_dir / "episode.json", episode.to_dict())
                    row["generated_tokens"] = episode.token_count
                    row["delegations"] = len({action.session_id for action in episode.actions if action.parent_session_id})
                    context_budget = "ContextLimitError" in (episode.error or "")
                    policy_budget = episode.result.get("budget_exhausted", False) or context_budget
                    if episode.error:
                        row["agent_error"] = episode.error
                    if episode.error and not context_budget:
                        raise RuntimeError(episode.error)
                    if not episode.completed and not policy_budget:
                        raise RuntimeError(episode.error or "Agent episode did not complete")
                    stage = "snapshot"
                    snapshot = task_dir / "prediction"
                    sandbox.snapshot(snapshot)
                stage = "prediction"
                patch = source_diff(Path(spec.source_path), snapshot, spec)
                (task_dir / "prediction.diff").write_text(patch, encoding="utf-8")
                if deadline <= time.time():
                    row.update(status="timeout", reason="Pilot deadline reached before official grading")
                else:
                    stage = "grading"
                    row.update(grade_prediction(task, patch, condition, run_dir, image_digest=env["image_digest"],
                                                timeout=max(1, min(300, int(deadline - time.time())))))
            except ValueError as exc:
                if stage in {"snapshot", "prediction"}:
                    row.update(status="completed", success=False, reason=f"Invalid agent artifact: {exc}")
                else:
                    row.update(status="infra_error", reason=f"{type(exc).__name__}: {exc}")
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                row.update(status="infra_error", reason=f"{type(exc).__name__}: {exc}")
        row["elapsed_seconds"] = time.monotonic() - started
        row = log.emit("evaluation", **row)
        rows.append(row)
    return rows
