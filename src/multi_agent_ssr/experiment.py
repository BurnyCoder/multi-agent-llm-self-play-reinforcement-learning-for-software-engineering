"""Phase orchestration: preregistration, preflight, pilot, and evaluation.

Collection and optimization are separated by immutable policy versions. Partial
groups never enter training. See https://arxiv.org/html/2512.18552v3#S2 .
"""
from __future__ import annotations

import gc
import json
import subprocess
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

from .contracts import Limits, ModelConfig
from .io import EventLog, utc_now, write_json


def load_config(path: Path) -> dict:
    """Controls override explicit sections of one shared pilot preset."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    parent = data.pop("extends", None)
    result = load_config(path.parent / parent) if parent else {}
    for key, value in data.items():
        result[key] = {**result.get(key, {}), **value} if isinstance(value, dict) else value
    return result


class Budget:
    """One persisted deadline includes preflight, inference and optimizer time."""
    def __init__(self, run_dir: Path, seconds: int, *, now: float | None = None):
        self.path = run_dir / "budget.json"
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if self.record["budget_seconds"] != seconds:
                raise ValueError("An existing run's budget cannot be reset")
        else:
            started = time.time() if now is None else now
            self.record = {"started_epoch": started, "budget_seconds": seconds,
                           "deadline_epoch": started + seconds, "started_at": utc_now()}
            write_json(self.path, self.record)
        self.deadline = self.record["deadline_epoch"]

    def remaining(self, *, now: float | None = None) -> float:
        """An exhausted budget never authorizes a new episode."""
        return max(0.0, self.deadline - (time.time() if now is None else now))


def require_preregistered() -> str:
    """Require pushed code and frozen manifests before executing the LLM."""
    from .prepare import command
    for path in ["reports/preregistration.md", "manifests/model.json", "manifests/repositories.json",
                 "manifests/evaluation.json", "manifests/synthetic.json"]:
        command(["git", "ls-files", "--error-unmatch", path])
    if command(["git", "status", "--porcelain"]):
        raise RuntimeError("Commit and push code/config changes before executing the model")
    commit = command(["git", "rev-parse", "HEAD"])
    if commit != command(["git", "rev-parse", "@{upstream}"]):
        raise RuntimeError("Push the preregistered commit before executing the model")
    return commit


def _evaluate(policy, condition: str, run_dir: Path, limits: Limits,
              deadline: float, delegation: bool) -> list[dict]:
    """Keep human and synthetic bug distributions separate."""
    from .evaluation import evaluate_human
    from .synthetic import evaluate_synthetic
    rows = evaluate_human(policy, condition, run_dir, limits, deadline, delegation=delegation)
    return rows + evaluate_synthetic(policy, condition, run_dir, limits, deadline, delegation=delegation)


def execute(phase: str, config_path: Path, run_dir: Path) -> None:
    """Execute a phase and preserve diagnostic records even when it fails."""
    config = load_config(config_path)
    commit = require_preregistered()
    pin = json.loads(Path("manifests/model.json").read_text())
    model_config = ModelConfig(**{**config["model"], "revision": pin["revision"]})
    limits = Limits(**config["limits"])
    settings = config["experiment"]
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = Budget(run_dir, settings["budget_seconds"])
    log = EventLog(run_dir / "events.jsonl")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config"] != config or manifest["model"] != asdict(model_config):
            raise ValueError("Resume requires the original run configuration and model revision")
    else:
        write_json(manifest_path, {"created_at": utc_now(), "code_commit": commit, "config": config,
            "model": asdict(model_config), "hardware": subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"],
                capture_output=True, text=True).stdout,
            "expected_human_tasks": settings["human_tasks"], "expected_synthetic_tasks": settings["synthetic_tasks"],
            "hypothesis": "Trained team improves held-out human bug resolution over frozen initialization"})
    if budget.remaining() <= 0:
        log.emit("budget_exhausted", phase=phase)
        return
    log.emit("phase_started", phase=phase, code_commit=commit, remaining_seconds=budget.remaining())
    try:
        if phase == "preflight":
            from .learning import run_preflight
            write_json(run_dir / "preflight.json", run_preflight(model_config, run_dir / "preflight"))
            return
        checkpoint_run = Path(settings.get("checkpoint_run", run_dir))
        preflight_path = checkpoint_run / "preflight.json"
        if not preflight_path.exists() or not json.loads(preflight_path.read_text()).get("success"):
            raise RuntimeError("Run selfplay preflight successfully before pilot/evaluation")
        from .learning import build_optimizer, update_policy
        from .model import LocalPolicy
        from .prepare import load_repositories
        from .rollouts import collect_group
        policy = LocalPolicy(model_config, run_dir / "model.jsonl")
        if phase == "evaluate":
            if settings["condition"] == "evaluation_only":
                policy.load_checkpoint(checkpoint_run / "checkpoints" / "initial", restore_rng=False)
                _evaluate(policy, "base", run_dir, limits, budget.deadline, settings["delegation"])
            latest = json.loads((checkpoint_run / "latest.json").read_text())
            policy.load_checkpoint(checkpoint_run / latest["path"], restore_rng=False)
            _evaluate(policy, "trained", run_dir, limits, budget.deadline, settings["delegation"])
            return
        if settings["condition"] == "evaluation_only":
            raise ValueError("The evaluation-only preset requires the evaluate command")
        baseline_path = run_dir / "baseline.json"
        if not baseline_path.exists():
            started = time.monotonic()
            policy.save_checkpoint(run_dir / "checkpoints" / "initial", metadata={"purpose": "frozen_base"})
            # Bound baseline coverage before it consumes the final comparison.
            # Keep a minimum training allocation and a 25% final-evaluation margin.
            baseline_seconds = max(0, (budget.remaining() - 1800) / 2.25)
            rows = _evaluate(policy, "base", run_dir, limits,
                             time.time() + baseline_seconds, settings["delegation"])
            policy.save_checkpoint(run_dir / "checkpoints" / "post-baseline",
                                   metadata={"purpose": "unchanged_base_with_post_baseline_rng"})
            write_json(baseline_path, {"elapsed_seconds": time.monotonic() - started, "rows": len(rows)})
        optimizer = build_optimizer(policy)
        latest_path = run_dir / "latest.json"
        group_index = 0
        if latest_path.exists():
            latest = json.loads(latest_path.read_text())
            policy.load_checkpoint(run_dir / latest["path"], optimizer)
            group_index = latest["completed_groups"]
        elif (run_dir / "checkpoints" / "post-baseline").exists():
            policy.load_checkpoint(run_dir / "checkpoints" / "post-baseline")
        reserve = max(settings["minimum_final_reserve_seconds"],
                      1.25 * json.loads(baseline_path.read_text())["elapsed_seconds"])
        training_deadline = budget.deadline - reserve
        repositories = [s for s in load_repositories() if s.split == "train"]
        metrics_log = EventLog(run_dir / "metrics.jsonl")
        durations = []
        while time.time() < training_deadline:
            if training_deadline - time.time() < max(300, max(durations, default=0) * 1.25):
                break
            started = time.monotonic()
            spec = repositories[group_index % len(repositories)]
            try:
                if settings["condition"] == "selfplay":
                    groups, metrics = collect_group(policy, spec, run_dir, limits,
                        group_index=group_index, injector_samples=settings["injector_samples"],
                        repair_samples=settings["repair_samples"], max_bug_order=settings["max_bug_order"],
                        deadline=training_deadline, delegation=settings["delegation"])
                else:
                    from .controls import collect_control_group
                    groups, metrics = collect_control_group(policy, spec, run_dir, limits,
                        group_index=group_index, settings=settings, deadline=training_deadline)
            except TimeoutError as exc:
                log.emit("group_not_trained", group_index=group_index, reason=str(exc))
                break
            if time.time() >= training_deadline:
                log.emit("group_not_trained", group_index=group_index, reason="Evaluation reserve reached")
                break
            capacity = settings["injector_samples"] * (1 + settings["repair_samples"] * settings["max_bug_order"]) * limits.generated_tokens
            if settings["condition"] != "selfplay":
                capacity = settings["repair_samples"] * limits.generated_tokens
            collection_path = run_dir / "collection_groups" / f"group-{group_index + 1:05d}.json"
            collection = {"completed": True, "optimized": False, "behavior_policy_version": policy.version,
                "episode_groups": [[episode.episode_id for episode in group] for group in groups],
                "capacity": capacity, "metrics": metrics}
            write_json(collection_path, collection)
            try:
                learned = update_policy(policy, groups, optimizer, capacity,
                    epochs=settings["optimization_passes"], deadline=training_deadline)
            except TimeoutError as exc:
                # A partly applied pass is not an eligible research checkpoint.
                # Reload both adapter and optimizer before final evaluation.
                if latest_path.exists():
                    previous = json.loads(latest_path.read_text())
                    policy.load_checkpoint(run_dir / previous["path"], optimizer)
                else:
                    policy.load_checkpoint(run_dir / "checkpoints" / "post-baseline")
                    optimizer = build_optimizer(policy)
                log.emit("group_not_trained", group_index=group_index, reason=str(exc))
                break
            group_index += 1
            directory = Path("checkpoints") / f"group-{group_index:05d}"
            policy.save_checkpoint(run_dir / directory, optimizer,
                metadata={"completed_groups": group_index, "collection": collection})
            write_json(latest_path, {"path": str(directory), "completed_groups": group_index,
                                     "policy_version": policy.version})
            write_json(collection_path, {**collection, "optimized": True, "update": learned})
            duration = time.monotonic() - started
            durations.append(duration)
            metrics_log.emit("training_group", step=group_index, **metrics, **learned,
                loss=sum(p["loss"] for p in learned["passes"]) if learned["passes"] else None,
                memory=policy.torch.cuda.max_memory_allocated(), group_seconds=duration)
        if not latest_path.exists():
            policy.save_checkpoint(run_dir / "checkpoints" / "no-updates", optimizer,
                                   metadata={"completed_groups": 0, "purpose": "no_completed_training_updates"})
            write_json(latest_path, {"path": "checkpoints/no-updates", "completed_groups": 0,
                                     "policy_version": policy.version})
        _evaluate(policy, "trained", run_dir, limits, budget.deadline, settings["delegation"])
        write_json(run_dir / "summary.json", {"status": "completed", "completed_groups": group_index,
            "optimizer_steps": policy.version, "remaining_seconds": budget.remaining(),
            "selected_checkpoint": json.loads(latest_path.read_text()),
            "selection_rule": "last completed training group, independent of evaluation score"})
        log.emit("pilot_complete", completed_groups=group_index, optimizer_steps=policy.version)
        del optimizer, policy
        gc.collect()
    except BaseException as exc:
        log.emit("phase_failed", phase=phase, error=f"{type(exc).__name__}: {exc}")
        write_json(run_dir / "failure.json", {"phase": phase, "error": f"{type(exc).__name__}: {exc}",
                                              "timestamp": utc_now(), "remaining_seconds": budget.remaining()})
        raise
