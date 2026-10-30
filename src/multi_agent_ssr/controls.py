"""Runnable optional controls, kept outside the first pilot's execution path.

The repair-only control fixes the pilot's bug corpus. The procedural control
uses AST mutations and original visible tests on training repositories only.
They answer different questions and are not pooled with the primary experiment.
"""
import hashlib
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .contracts import BugArtifact, RepoSpec, ValidationResult
from .io import write_json
from .validation import _copy, run_tests, source_diff, validate_bug


def require_training_repository(spec: RepoSpec) -> None:
    """Enforce the repository split at the control's data boundary."""
    if spec.split != "train":
        raise ValueError("Control training accepts training repositories only")


def frozen_artifacts(corpus_run: Path, spec: RepoSpec) -> list[BugArtifact]:
    """Read only successfully validated, pinned artifacts from a completed run."""
    require_training_repository(spec)
    artifacts = []
    for path in sorted((corpus_run / "artifact_records").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["artifact"]["repo_name"] != spec.name:
            continue
        if row["repository_commit"] != spec.commit:
            raise ValueError("Frozen corpus repository revision differs from training pin")
        artifacts.append(BugArtifact(**row["artifact"]))
    if not artifacts:
        raise ValueError(f"No frozen valid artifacts for {spec.name}; prepare the corpus first")
    return artifacts


def procedural_task(spec: RepoSpec, run_dir: Path, group_index: int, deadline: float):
    """Deterministically select a test-detected AST mutation, with no LLM use."""
    from .synthetic import mutation_candidates
    require_training_repository(spec)
    clean = Path(spec.source_path)
    # These targets exercise the selected source modules and are fixed controls.
    target_map = {"more-itertools": ("more_itertools/more.py", "tests/test_more.py"),
                  "boltons": ("boltons/iterutils.py", "tests/test_iterutils.py")}
    if spec.name not in target_map:
        raise ValueError("Procedural control has no preregistered target for this training repository")
    source_file, test_file = target_map[spec.name]
    root = run_dir / "procedural_corpus" / spec.name
    root.mkdir(parents=True, exist_ok=True)
    baseline_file = root / "baseline.json"
    if not baseline_file.exists():
        write_json(baseline_file, run_tests(spec, clean, [test_file], root))
    baseline = json.loads(baseline_file.read_text())
    if len(baseline) < 5 or any(value != "passed" for value in baseline.values()):
        raise ValueError("Procedural baseline requires five clean passing tests and no skips/errors")
    candidates = list(mutation_candidates((clean / source_file).read_text(), source_file))
    # Rotation changes the starting function across completed groups without
    # adapting task selection to repair success or held-out outcomes.
    offset = group_index % len(candidates)
    for candidate in candidates[offset:] + candidates[:offset]:
        if time.time() >= deadline:
            raise TimeoutError("Procedural control collection reached its deadline")
        with tempfile.TemporaryDirectory(prefix="ssr-control-") as temporary:
            tree = Path(temporary) / "buggy"
            _copy(clean, tree)
            (tree / source_file).write_text(candidate["source"], encoding="utf-8")
            try:
                results = run_tests(spec, tree, [test_file], root)
            except ValueError:
                continue  # Undetected/invalid mutants carry no policy reward.
            failed = [key for key, value in results.items() if value == "failed"]
            passed = [key for key, value in results.items() if value == "passed"]
            if results.keys() != baseline.keys() or not failed or not passed:
                continue
            patch = source_diff(clean, tree, spec)
            task_id = f"procedural-{spec.name}-{group_index:05d}-" + hashlib.sha256(patch.encode()).hexdigest()[:10]
            destination = root / task_id
            _copy(tree, destination)
            artifact = BugArtifact(task_id, spec.name, patch, "", [test_file], [test_file])
            checked = ValidationResult(True, "Independently detected single-file AST mutation",
                passed, failed, baseline, results, results, str(destination))
            write_json(root / f"{task_id}.json", {"artifact": asdict(artifact), "validation": asdict(checked)})
            prompt = ("Repair the behavior regression in this repository using its original visible tests. "
                      f"Selected tests: {test_file}. Failing examples: {failed[:3]}. "
                      "Inspect and edit source code only, test your repair, and finish.")
            return artifact, checked, prompt
    raise ValueError("No valid procedural mutation found within the fixed candidate set")


def collect_control_group(policy, spec, run_dir, limits, *, group_index, settings, deadline):
    """Collect solver-only return groups without invoking an adaptive injector."""
    from .rollouts import solve_artifact
    require_training_repository(spec)
    if settings["condition"] == "repair_only":
        corpus_run = Path(settings["corpus_run"])
        artifacts = frozen_artifacts(corpus_run, spec)
        artifact = artifacts[group_index % len(artifacts)]
        validation = validate_bug(spec, artifact, run_dir / "control_validation")
        if not validation.valid:
            raise ValueError("Frozen artifact no longer passes validation: " + validation.reason)
        prompt = None
    elif settings["condition"] == "procedural":
        artifact, validation, prompt = procedural_task(spec, run_dir, group_index, deadline)
    else:
        raise ValueError("Unknown training control")
    episodes, _ = solve_artifact(policy, spec, artifact, validation, run_dir, limits,
        settings["repair_samples"], deadline, settings["delegation"], prompt_override=prompt)
    successes = sum(episode.reward == 1 for episode in episodes)
    return [episodes], {"condition": settings["condition"], "valid_bugs": 1,
        "injector_episodes": 0, "repair_attempts": len(episodes),
        "repair_success_rate": successes / len(episodes),
        "repair_reward": 2 * successes / len(episodes) - 1,
        "token_count": sum(episode.token_count for episode in episodes)}
