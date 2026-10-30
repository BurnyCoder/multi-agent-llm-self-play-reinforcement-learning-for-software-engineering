"""Collect complete shared-policy games before any optimizer update.

The environment and test-patch specification follow SSR; root/child concurrency
and whole-team credit assignment are this project's explicit extensions.
https://arxiv.org/html/2512.18552v3#S2
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace
from pathlib import Path

from .agents import run_team
from .contracts import BugArtifact, Episode, Limits, RepoSpec, ValidationResult
from .io import EventLog, write_json
from .learning import injection_reward
from .sandbox import DockerSandbox
from .validation import (
    _diff,
    _files,
    build_artifact,
    score_repair,
    source_diff,
    validate_bug,
    write_artifact,
)


def repair_prompt(spec: RepoSpec, artifact: BugArtifact, validation: ValidationResult) -> str:
    """Expose the reversed test change, never the clean implementation/bug diff."""
    weak = Path(validation.buggy_tree)
    # Build the inverse by swapping the two verified test-file snapshots. No
    # source text enters the task prompt or the repair agent's filesystem.
    before = _files(weak)
    after = _files(Path(spec.source_path))
    inverse = _diff(before, after, artifact.test_files)
    return ("Repair the repository so it satisfies the following test-suite improvement. "
            "Inspect the code, delegate independent investigation, change source code, and test the repair. "
            "Do not change tests, test configuration, or dependencies.\n" + inverse)


def save_episode(run_dir: Path, episode: Episode) -> None:
    """Keep training tokens separate from public summary data."""
    write_json(run_dir / "episodes" / f"{episode.episode_id}.json", episode.to_dict())


def solve_artifact(policy, spec: RepoSpec, artifact: BugArtifact,
                   validation: ValidationResult, run_dir: Path, limits: Limits,
                   samples: int, deadline: float, delegation: bool = True,
                   prompt_override: str | None = None) -> tuple[list[Episode], list[Path]]:
    """Each solver gets a fresh buggy snapshot and no other solver's context."""
    episodes, snapshots = [], []
    prompt = prompt_override or repair_prompt(spec, artifact, validation)
    for index in range(samples):
        if time.time() >= deadline:
            raise TimeoutError("Collection deadline reached before completing a repair group")
        work = run_dir / "work" / artifact.task_id / f"solver-{index}"
        with DockerSandbox(spec.image_digest or spec.image, Path(validation.buggy_tree), work) as sandbox:
            episode = run_team(policy, sandbox, "solver", artifact.task_id, prompt, run_dir,
                               limits=replace(limits, episode_seconds=min(limits.episode_seconds,
                                   max(1, int(deadline - time.time())))), delegation=delegation)
            snapshot = work / "snapshot"
            sandbox.snapshot(snapshot)
        if episode.error and not episode.error.startswith("ContextLimitError:"):
            save_episode(run_dir, episode)
            raise RuntimeError(f"Incomplete solver episode: {episode.error}")
        try:
            patch = source_diff(Path(validation.buggy_tree), snapshot, spec)
            score = score_repair(spec, artifact, patch, validation, work / "score")
            if score.get("infrastructure_error"):
                raise RuntimeError(score["reason"])
        except ValueError as exc:
            patch, score = "", {"success": False, "reason": str(exc)}
        episode.reward = 1.0 if score["success"] else -1.0
        episode.result.update(score=score, repair_patch=patch)
        save_episode(run_dir, episode)
        episodes.append(episode)
        snapshots.append(snapshot)
    return episodes, snapshots


def collect_group(policy, spec: RepoSpec, run_dir: Path, limits: Limits,
                  *, group_index: int, injector_samples: int = 2, repair_samples: int = 4,
                  max_bug_order: int = 2, deadline: float, delegation: bool = True,
                  condition: str = "selfplay") -> tuple[list[list[Episode]], dict]:
    """A group is atomic: interrupted collection is logged but never optimized."""
    log = EventLog(run_dir / "events.jsonl")
    if condition != "selfplay":
        raise ValueError("Non-self-play conditions must use the independent control collector")
    injector_group, solver_groups = [], []
    valid_count = repaired = attempts = 0
    for index in range(injector_samples):
        if time.time() >= deadline:
            raise TimeoutError("Collection deadline reached before completing an injector group")
        task_id = f"{spec.name}-{group_index:05d}-{index}"
        work = run_dir / "work" / task_id / "injector"
        prompt = ("Create a reproducible semantic bug in this real Python repository. "
                  f"Source roots: {spec.source_roots}; test roots: {spec.test_roots}. "
                  "Inspect existing source and tests. Delegate independent test investigation or source work. "
                  "Modify source to introduce one bug while keeping the package importable. "
                  "Also weaken existing tests so at least one newly failing test becomes passing or is removed. "
                  "Select at least five initially passing tests with at least one remaining pass after the bug. "
                  "Do not change conftest.py, project configuration, runners, or dependencies. "
                  "Keep edits small. Finish with test_targets listing the existing test files or pytest node IDs to run.")
        with DockerSandbox(spec.image_digest or spec.image, Path(spec.source_path), work) as sandbox:
            injector = run_team(policy, sandbox, "injector", f"{spec.name}:{spec.commit}",
                prompt, run_dir, limits=replace(limits, episode_seconds=min(limits.episode_seconds,
                    max(1, int(deadline - time.time())))), delegation=delegation)
            changed = work / "snapshot"
            sandbox.snapshot(changed)
        if injector.error and not injector.error.startswith("ContextLimitError:"):
            save_episode(run_dir, injector)
            raise RuntimeError(f"Incomplete injector episode: {injector.error}")
        try:
            targets = injector.result.get("test_targets", [])
            artifact = build_artifact(Path(spec.source_path), changed, spec, targets, task_id)
            validation = validate_bug(spec, artifact, work / "validation")
            write_artifact(artifact, run_dir / "artifacts" / task_id)
        except ValueError as exc:
            artifact, validation = None, ValidationResult(False, str(exc))
        log.emit("artifact_validated", task_id=task_id, **asdict(validation))
        if validation.valid:
            valid_count += 1
            write_json(run_dir / "artifact_records" / f"{task_id}.json",
                       {"repository_commit": spec.commit, "artifact": asdict(artifact)})
            solvers, snapshots = solve_artifact(policy, spec, artifact, validation, run_dir, limits,
                                                repair_samples, deadline, delegation)
            solver_groups.append(solvers)
            successes = sum(episode.reward == 1 for episode in solvers)
            repaired += successes
            attempts += len(solvers)
            injector.reward = injection_reward(True, successes / len(solvers))
            if max_bug_order >= 2:
                failed = next((i for i, episode in enumerate(solvers) if episode.reward == -1), None)
                if failed is not None:
                    try:
                        # Reuse exactly the same oracle/test gap, replacing only
                        # buggy source with a failed solver's attempted repair.
                        second = replace(artifact, task_id=task_id + "-order2", order=2,
                                         parent_task_id=task_id,
                                         bug_patch=source_diff(Path(spec.source_path), snapshots[failed], spec))
                        checked = validate_bug(spec, second, work / "order2-validation")
                        log.emit("higher_order_validated", task_id=second.task_id, **asdict(checked))
                        if checked.valid:
                            higher, _ = solve_artifact(policy, spec, second, checked, run_dir, limits,
                                                       repair_samples, deadline, delegation)
                            solver_groups.append(higher)
                            write_artifact(second, run_dir / "artifacts" / second.task_id)
                    except ValueError as exc:
                        log.emit("higher_order_rejected", task_id=task_id, reason=str(exc))
        else:
            injector.reward = injection_reward(False, None)
        injector.result.update(validation=asdict(validation))
        save_episode(run_dir, injector)
        injector_group.append(injector)
    groups = [injector_group] + solver_groups
    metrics = {"valid_bug_rate": valid_count / injector_samples,
               "valid_bugs": valid_count, "injector_episodes": injector_samples,
               "inject_reward": sum(x.reward for x in injector_group) / injector_samples,
               "repair_success_rate": repaired / attempts if attempts else None,
               "repair_attempts": attempts,
               "repair_reward": (2 * repaired / attempts - 1) if attempts else None,
               "token_count": sum(e.token_count for group in groups for e in group)}
    return groups, metrics
