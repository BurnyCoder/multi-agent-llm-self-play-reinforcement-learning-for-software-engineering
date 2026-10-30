"""Frozen, policy-independent mutation diagnostics on held-out humanize code.

These tasks retain visible existing tests and do not use SSR test weakening.
Small operator mutations follow standard mutation-testing practice (for example
https://github.com/boxed/mutmut); generation here uses only the Python AST API:
https://docs.python.org/3/library/ast.html . No external mutation code is copied.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agents import run_team
from .contracts import Episode, Limits, RepoSpec
from .io import EventLog, utc_now, write_json
from .prepare import (
    _environment_fingerprint,
    _read_manifest,
    data_directory,
    load_repositories,
)
from .sandbox import DockerSandbox
from .validation import _apply, _copy, run_tests, source_diff

TEST_TARGETS = ["tests/test_number.py", "tests/test_filesize.py",
                "tests/test_time.py", "tests/test_lists.py"]
SOURCE_FILES = ["src/humanize/filesize.py", "src/humanize/lists.py",
                "src/humanize/number.py", "src/humanize/time.py"]
_COMPARE_FLIPS = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE,
                  ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt,
                  ast.Is: ast.IsNot, ast.IsNot: ast.Is,
                  ast.In: ast.NotIn, ast.NotIn: ast.In}


def _functions(node: ast.AST, prefix: str = "") -> Iterator[tuple[str, ast.AST]]:
    """Qualify nested definitions so each original function forms one cluster."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = f"{prefix}.{child.name}" if prefix else child.name
            if not isinstance(child, ast.ClassDef):
                yield name, child
            yield from _functions(child, name)
        else:
            yield from _functions(child, prefix)


def _body_nodes(function: ast.AST) -> Iterator[ast.AST]:
    """Exclude defaults, annotations, and nested functions from a function's mutations."""
    def visit(node: ast.AST) -> Iterator[ast.AST]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from visit(child)

    for statement in function.body:
        yield from visit(statement)


def mutation_candidates(source: str, relative_path: str) -> Iterator[dict[str, Any]]:
    """Yield deterministic single-expression edits without reformatting whole files.

    AST offsets are UTF-8 byte offsets, not Python string indices. Replacing the
    precise byte span preserves comments, docstrings and unrelated formatting.
    """
    tree = ast.parse(source, filename=relative_path)
    encoded = source.encode("utf-8")
    starts = [0]
    for line in encoded.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    functions = sorted(_functions(tree), key=lambda pair: (pair[1].lineno, pair[0]))
    for name, function in functions:
        seen: set[str] = set()
        nodes = sorted((node for node in _body_nodes(function) if hasattr(node, "lineno")),
                       key=lambda node: (node.lineno, node.col_offset, type(node).__name__))
        for node in nodes:
            variants: list[tuple[str, ast.AST]] = []
            if isinstance(node, ast.Compare):
                for index, operator in enumerate(node.ops):
                    replacement = _COMPARE_FLIPS.get(type(operator))
                    if replacement:
                        mutated = copy.deepcopy(node)
                        mutated.ops[index] = replacement()
                        variants.append((f"compare:{type(operator).__name__}->{replacement.__name__}", mutated))
            elif isinstance(node, ast.BoolOp):
                mutated = copy.deepcopy(node)
                mutated.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
                variants.append(("boolean_operator", mutated))
            elif isinstance(node, ast.Constant):
                if type(node.value) is bool:
                    variants.append(("boolean_constant", ast.Constant(not node.value)))
                elif type(node.value) is int or (type(node.value) is float and math.isfinite(node.value)):
                    variants.append(("numeric_constant_plus_one", ast.Constant(node.value + 1)))
            for kind, mutated in variants:
                begin = starts[node.lineno - 1] + node.col_offset
                end = starts[node.end_lineno - 1] + node.end_col_offset
                expression = ast.unparse(mutated)
                if not isinstance(mutated, ast.Constant):
                    expression = "(" + expression + ")"
                changed = (encoded[:begin] + expression.encode("utf-8") + encoded[end:]).decode("utf-8")
                digest = hashlib.sha256(changed.encode("utf-8")).hexdigest()
                if digest in seen or changed == source:
                    continue
                ast.parse(changed, filename=relative_path)
                seen.add(digest)
                description = (ast.get_docstring(function) or "Restore the behavior exercised by the failing tests.")
                yield {"source_file": relative_path, "function": name,
                       "function_line": function.lineno, "mutation_line": node.lineno,
                       "mutation": kind, "source": changed, "source_sha256": digest,
                       "behavior": description.splitlines()[0][:300]}


def _spec() -> RepoSpec:
    """Require the held-out repository pin instead of selecting a training source."""
    matches = [spec for spec in load_repositories() if spec.name == "humanize" and spec.split == "test"]
    if len(matches) != 1:
        raise ValueError("Exactly one held-out humanize repository must be prepared")
    return matches[0]


def _load_records(root: Path, spec: RepoSpec) -> list[dict[str, Any]]:
    """Validate cohort provenance and mutation hashes before any evaluation."""
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    environment = _environment_fingerprint()
    legacy = "environment_fingerprint" not in metadata
    if metadata["repository_commit"] != spec.commit:
        raise ValueError("Synthetic cohort repository pin differs from current preparation")
    if legacy:
        if metadata["image_digest"] != _read_manifest("environment")["original_image_id"]:
            raise ValueError("Legacy synthetic cohort has no verified environment provenance")
    elif metadata["environment_fingerprint"] != environment:
        raise ValueError("Synthetic cohort environment recipe differs from current preparation")
    records = json.loads((root / "tasks.json").read_text(encoding="utf-8"))
    if len({record["cluster_id"] for record in records}) != len(records):
        raise ValueError("Synthetic cohort repeats an original function")
    for record in records:
        digest = hashlib.sha256(record["bug_patch"].encode("utf-8")).hexdigest()
        if digest != record["patch_sha256"] or record["repository_commit"] != spec.commit:
            raise ValueError("Stored synthetic mutation failed its integrity check")
        buggy = (root / record["buggy_tree"]).resolve()
        if not buggy.is_relative_to(root.resolve()) or not buggy.is_dir():
            raise ValueError("Synthetic mutant tree is missing or outside its private root")
        current = (buggy / record["source_file"]).read_bytes()
        if hashlib.sha256(current).hexdigest() != record["source_sha256"]:
            raise ValueError("An immutable synthetic mutant tree was changed")
    if legacy:
        # Migrate only the original verified image; equivalent future rebuilds
        # may have another OCI ID, but must retain the committed complete recipe.
        metadata["environment_fingerprint"] = environment
        write_json(root / "preparation.json", metadata)
    return records


def _publish_manifest(records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Publish cohort identities without patches, reference code or test maps."""
    fields = ("task_id", "repository", "repository_commit", "cluster_id",
              "source_file", "function", "patch_sha256")
    public_path = Path("manifests/synthetic.json")
    manifest = {
        "created_at": metadata["created_at"], "requested_count": metadata["count"],
        "accepted_count": len(records),
        "diagnostic": "visible_existing_tests_without_ssr_weakening",
        "selection": "Sorted file/function/AST position; first test-detected mutant per function; no learned policy",
        "tasks": [{key: record[key] for key in fields} for record in records],
    }
    if public_path.exists():
        existing = json.loads(public_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in manifest.items() if key != "created_at"):
            raise ValueError("Prepared synthetic cohort differs from the committed manifest")
        return  # A new local preparation never rewrites the original timestamp.
    write_json(public_path, manifest)


def prepare_synthetic(data_dir: Path, count: int = 16) -> list[dict[str, Any]]:
    """Freeze source-only bugs using baseline and mutant execution before model use."""
    if count < 1:
        raise ValueError("Synthetic task count must be positive")
    spec = _spec()
    original = Path(spec.source_path)
    if not (original / "src/humanize/_version.py").is_file():
        raise ValueError("Build the pinned humanize wheel first to generate _version.py")
    root = Path(data_dir) / "synthetic"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "preparation.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["count"] != count:
            raise ValueError("Cannot change the requested count of an existing frozen cohort")
        records = _load_records(root, spec)
        if len(records) == count:
            _publish_manifest(records, metadata)
            return records
    else:
        _copy(original, root / "clean")
        metadata = {"created_at": utc_now(), "count": count, "repository_commit": spec.commit,
                    "image_digest": spec.image_digest, "environment_fingerprint": _environment_fingerprint(),
                    "test_targets": TEST_TARGETS,
                    "max_candidates_per_function": 64}
        write_json(metadata_path, metadata)
        records = []
        write_json(root / "tasks.json", records)
    clean = root / "clean"
    log = EventLog(root / "preparation.jsonl")
    baseline_path = root / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        baseline = run_tests(spec, clean, TEST_TARGETS, root / "work")
        write_json(baseline_path, baseline)
    if len(baseline) < 5 or any(status != "passed" for status in baseline.values()):
        raise ValueError("Synthetic baseline requires at least five tests, all passing")
    log.emit("synthetic_baseline", repository=spec.name, commit=spec.commit,
             passed_tests=len(baseline), resumed_tasks=len(records))
    # DrvFS has expensive per-file operations. Scratch copies may be ephemeral;
    # accepted trees and all scientific records remain in the supplied data root.
    with tempfile.TemporaryDirectory(prefix="ssr-synthetic-") as scratch:
        scratch_clean = Path(scratch) / "clean"
        _copy(clean, scratch_clean)
        return _prepare_candidates(spec, root, scratch_clean, baseline, metadata, records, log)


def _prepare_candidates(spec: RepoSpec, root: Path, clean: Path, baseline: dict[str, str],
                        metadata: dict[str, Any], records: list[dict[str, Any]],
                        log: EventLog) -> list[dict[str, Any]]:
    """Select the same deterministic candidates using fast, disposable working copies."""
    count = metadata["count"]
    accepted = {record["cluster_id"] for record in records}
    attempts: Counter[str] = Counter()
    (root / "work").mkdir(exist_ok=True)
    for relative in SOURCE_FILES:
        path = clean / relative
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        for candidate in mutation_candidates(source, relative):
            cluster = f"{spec.name}@{spec.commit}:{relative}:{candidate['function']}"
            if cluster in accepted or attempts[cluster] >= metadata["max_candidates_per_function"]:
                continue
            attempts[cluster] += 1
            with tempfile.TemporaryDirectory(prefix="candidate-", dir=clean.parent) as temporary:
                tree = Path(temporary) / "tree"
                _copy(clean, tree)
                (tree / relative).write_text(candidate["source"], encoding="utf-8", newline="")
                patch = source_diff(clean, tree, spec)
                patch_digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
                try:
                    results = run_tests(spec, tree, TEST_TARGETS, root / "work")
                    if results.keys() != baseline.keys():
                        raise ValueError("Mutation changed the set of expected test IDs")
                    failing = sorted(key for key, value in results.items() if value == "failed")
                    passing = sorted(key for key, value in results.items() if value == "passed")
                    if not failing or not passing:
                        log.emit("synthetic_rejected", cluster_id=cluster, patch_sha256=patch_digest,
                                 reason="Mutation must preserve a pass and produce a failure")
                        continue
                except (ValueError, RuntimeError, OSError) as error:
                    log.emit("synthetic_rejected", cluster_id=cluster, patch_sha256=patch_digest,
                             reason=f"Validation error: {error}")
                    continue
                task_id = "synthetic-humanize-" + hashlib.sha256(
                    f"{cluster}\0{patch_digest}".encode()).hexdigest()[:16]
                buggy_relative = f"tasks/{task_id}/buggy"
                _copy(tree, root / buggy_relative)
                record = {key: value for key, value in candidate.items() if key != "source"}
                record.update(task_id=task_id, repository=spec.name, repository_commit=spec.commit,
                              cluster_id=cluster, bug_patch=patch, patch_sha256=patch_digest,
                              buggy_tree=buggy_relative, test_targets=TEST_TARGETS,
                              clean_results=baseline, mutant_results=results,
                              fail_to_pass=failing, pass_to_pass=passing)
                records.append(record)
                accepted.add(cluster)
                write_json(root / "tasks.json", records)
                log.emit("synthetic_accepted", task_id=task_id, cluster_id=cluster,
                         patch_sha256=patch_digest, failing_tests=len(failing), passing_tests=len(passing))
            if len(records) == count:
                _publish_manifest(records, metadata)
                return records
    raise ValueError(f"Only {len(records)} distinct valid function clusters found; required {count}")


def _prompt(record: dict[str, Any]) -> str:
    """Describe observable behavior, never the mutation or reference implementation."""
    failures = "\n".join(record["fail_to_pass"][:3])
    return (
        "Repair the behavior regression in this repository. Existing tests are visible. "
        "Modify source code only; do not alter tests or test configuration. "
        f"Documented behavior: {record['behavior']}\n"
        f"Examples of currently failing tests:\n{failures}\n"
        "Relevant test targets: " + " ".join(record["test_targets"]) + ". "
        "Inspect the code and tests, repair the behavior, run relevant tests, and use finish."
    )


def _delegations(episode: Episode, log_dir: Path) -> int:
    """Count child sessions started, including children that exhausted their budget."""
    events = log_dir / "events.jsonl"
    if events.exists():
        rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line]
        return len({row["session_id"] for row in rows if row.get("event") == "session_started"
                    and row.get("parent_session_id") is not None})
    return len({action.session_id for action in episode.actions if action.parent_session_id is not None})


def evaluate_synthetic(policy: Any, condition: str, run_dir: Path, limits: Limits,
                       deadline: float, delegation: bool = True) -> list[dict[str, Any]]:
    """Evaluate fixed tasks in fresh contexts; score only fresh source-patched trees."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", condition) or not math.isfinite(deadline):
        raise ValueError("Condition must be a simple identifier and deadline finite")
    root = data_directory() / "synthetic"
    spec = _spec()
    records = _load_records(root, spec)
    metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    if len(records) != metadata["count"]:
        raise ValueError("Cannot evaluate a partially prepared synthetic cohort")
    run_dir = Path(run_dir)
    log = EventLog(run_dir / "evaluation.jsonl")
    fingerprint = policy.adapter_fingerprint() if hasattr(policy, "adapter_fingerprint") else None
    previous = {}
    if log.path.exists():
        for line in log.path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (row.get("source") == "synthetic" and row.get("condition") == condition
                    and row.get("status") == "completed"):
                if row.get("policy_fingerprint") != fingerprint:
                    raise ValueError("A completed condition belongs to a different policy checkpoint")
                previous[row["task_id"]] = row
    rows = []
    for record in records:
        if record["task_id"] in previous:
            rows.append(previous[record["task_id"]])
            continue
        started = time.monotonic()
        row = {"condition": condition, "task_id": record["task_id"], "source": "synthetic",
               "cluster_id": record["cluster_id"], "status": "infrastructure_error", "success": None,
               "attempted": False, "generated_tokens": 0, "elapsed_seconds": 0.0, "delegations": 0, "reason": "",
               "policy_version": getattr(policy, "version", 0), "policy_fingerprint": fingerprint}
        remaining = deadline - time.time()
        # Reserve the trusted runner's 90-second command ceiling before rollout.
        if remaining <= 95:
            row.update(status="not_run_budget", reason="Insufficient global time for rollout and trusted scoring")
            rows.append(log.emit("evaluation", **row))
            continue
        destination = run_dir / "synthetic" / condition / record["task_id"]
        destination.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="attempt-", dir=destination))
        episode = None
        stage = "environment"
        try:
            episode_limits = replace(limits, episode_seconds=min(limits.episode_seconds, int(remaining - 95)))
            with DockerSandbox(spec.image_digest or spec.image, root / record["buggy_tree"], work) as sandbox:
                stage = "rollout"
                row["attempted"] = True
                episode = run_team(policy, sandbox, "solver", record["task_id"], _prompt(record),
                                   work / "agents", limits=episode_limits, delegation=delegation)
                row.update(generated_tokens=episode.token_count, delegations=_delegations(episode, work / "agents"))
                write_json(work / "episode.json", episode.to_dict())
                context_budget = (episode.error or "").startswith("ContextLimitError:")
                policy_budget = bool(episode.result.get("budget_exhausted", False) or context_budget)
                row["policy_budget_exhausted"] = policy_budget
                if episode.error:
                    row["agent_error"] = episode.error
                # A GPU failure remains infrastructure even if another session
                # simultaneously exhausts its budget. Only known context limits
                # permit an incomplete episode's existing patch to be graded.
                if episode.error and not context_budget:
                    raise RuntimeError(episode.error)
                if not episode.completed and not policy_budget:
                    raise RuntimeError("Agent episode did not complete")
                stage = "snapshot"
                sandbox.snapshot(work / "changed")
            repair = source_diff(root / record["buggy_tree"], work / "changed", spec)
            (work / "repair.diff").write_text(repair, encoding="utf-8")
            if deadline <= time.time():
                row.update(status="not_run_budget", reason="Global deadline reached before trusted grading")
            else:
                stage = "score"
                scoring_tree = work / "scoring"
                _copy(root / "clean", scoring_tree)
                _apply(scoring_tree, record["bug_patch"])
                _apply(scoring_tree, repair)
                results = run_tests(spec, scoring_tree, record["test_targets"], work)
                write_json(work / "test_results.json", results)
                if results.keys() != record["clean_results"].keys():
                    raise ValueError("Repair changed or removed expected test IDs")
                success = all(value == "passed" for value in results.values())
                row.update(status="completed", success=success,
                           reason="All original tests pass" if success else "Original tests still fail")
        except (ValueError, SyntaxError) as error:
            # Code-induced collection failures, invalid diffs and test tampering
            # are failed repairs; environment/setup failures remain unscored.
            row.update(status="completed" if stage in {"snapshot", "score"} else "infrastructure_error",
                       success=False if stage in {"snapshot", "score"} else None,
                       reason=f"{type(error).__name__}: {error}")
        except Exception as error:
            row.update(status="infrastructure_error", success=None,
                       reason=f"{type(error).__name__}: {error}")
        row["elapsed_seconds"] = time.monotonic() - started
        rows.append(log.emit("evaluation", **row))
    return rows
