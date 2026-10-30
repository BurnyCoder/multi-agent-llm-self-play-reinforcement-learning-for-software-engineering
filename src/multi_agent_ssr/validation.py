"""Executable SSR artifact validation and repair scoring on immutable originals.

The five-file artifact and inverse-mutation checks follow SSR §2:
https://arxiv.org/html/2512.18552v3. This Python-only adaptation uses trusted
pytest/JUnit parsing rather than model-authored parsers; see
https://docs.pytest.org/en/stable/how-to/output.html#creating-junitxml-format-files.
Diff inspection/application follows https://git-scm.com/docs/git-apply.
Passing these checks establishes test correctness, not general semantic correctness.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import os
import re
import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .contracts import BugArtifact, RepoSpec, ValidationResult
from .sandbox import (
    _EXCLUDED,
    DockerSandbox,
    SandboxInfrastructureError,
    _extract_archive,
    _relative_path,
    _source_archive,
)

_PROTECTED = {"pyproject.toml", "pytest.ini", "setup.cfg", "setup.py", "tox.ini",
              "conftest.py", "Dockerfile", "docker-compose.yml", ".coveragerc"}


_PYTEST_RUNNER = r'''
import contextlib, os, pathlib, sys, tempfile
os.environ.pop('PYTEST_ADDOPTS', None)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(pathlib.Path.cwd()))
if pathlib.Path('src').is_dir():
    sys.path.insert(0, str(pathlib.Path('src').resolve()))
import pytest
class NodeIds:
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        report.user_properties.append(('ssr_nodeid', report.nodeid))
with tempfile.TemporaryDirectory(prefix='ssr-junit-') as temporary:
    report = pathlib.Path(temporary) / 'results.xml'
    args = ['-q', '--color=no', '--tb=short', '--rootdir=.', '--confcutdir=.', '-p', 'no:cacheprovider',
            '-o', 'junit_family=xunit2', '--junitxml=' + str(report), *sys.argv[1:]]
    with contextlib.redirect_stdout(sys.stderr):
        status = pytest.main(args, plugins=[NodeIds()])
    if report.is_file():
        sys.stdout.write(report.read_text(encoding='utf-8'))
    sys.exit(status)
'''


def parse_junit(xml: str) -> dict[str, str]:
    """Retain exact pytest node IDs; absent IDs and duplicate cases fail closed."""
    root = ET.fromstring(xml)
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError("Unexpected JUnit root")
    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        node_ids = {prop.attrib.get("value", "") for prop in case.findall("./properties/property")
                    if prop.attrib.get("name") == "ssr_nodeid"}
        if len(node_ids) != 1 or not next(iter(node_ids), ""):
            raise ValueError("JUnit testcase is missing an unambiguous trusted node ID")
        node_id = next(iter(node_ids))
        if node_id in results:
            raise ValueError(f"Duplicate JUnit test ID: {node_id}")
        results[node_id] = next((status for tag, status in
                                [("error", "error"), ("failure", "failed"), ("skipped", "skipped")]
                                if case.find(tag) is not None), "passed")
    if not results:
        raise ValueError("Empty JUnit report contains no tests")
    return results


def _under(path: str, roots: list[str]) -> bool:
    candidate = Path(path)
    return any(candidate == Path(root) or candidate.is_relative_to(Path(root)) for root in roots)


def _category(path: str, spec: RepoSpec) -> str:
    path = _relative_path(path)
    if Path(path).name in _PROTECTED:
        return "protected"
    if _under(path, spec.test_roots):
        return "test"
    if _under(path, spec.source_roots):
        return "source"
    return "outside"


def _files(tree: Path) -> dict[str, bytes]:
    """Read the source manifest without following links or including runtime caches."""
    result = {}
    for directory, dirs, files in os.walk(tree, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED)
        for name in [*dirs, *sorted(files)]:
            if name in _EXCLUDED:
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Source symlinks are not allowed: {path}")
            if path.is_file():
                result[path.relative_to(tree).as_posix()] = path.read_bytes()
    return result


def _diff(before: dict[str, bytes], after: dict[str, bytes], paths: list[str]) -> str:
    patches = []
    for path in sorted(paths):
        old = before.get(path, b"").decode("utf-8").splitlines(keepends=True)
        new = after.get(path, b"").decode("utf-8").splitlines(keepends=True)
        lines = difflib.unified_diff(old, new, fromfile=f"a/{path}" if path in before else "/dev/null",
                                    tofile=f"b/{path}" if path in after else "/dev/null")
        for line in lines:
            patches.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(patches)


def source_diff(clean: Path, changed: Path, spec: RepoSpec) -> str:
    """Return only source changes, excluding tests and all harness/configuration files."""
    before, after = _files(Path(clean)), _files(Path(changed))
    paths = [path for path in before.keys() | after.keys()
             if before.get(path) != after.get(path) and _category(path, spec) == "source"]
    return _diff(before, after, paths)


def _targets(clean: Path, spec: RepoSpec, targets: list[str]) -> tuple[list[str], list[str]]:
    if not targets:
        raise ValueError("At least one test target is required")
    files = set()
    normalized = []
    for target in targets:
        path, *selectors = target.split("::")
        path = _relative_path(path)
        full = clean / path
        if not full.exists() or full.is_symlink() or not _under(path, spec.test_roots):
            raise ValueError(f"Test target must exist inside an allowed test root: {target}")
        if path.startswith("-") or any(character in target for character in "\n\r\x00"):
            raise ValueError("Invalid pytest target")
        if full.is_dir():
            if selectors:
                raise ValueError("Pytest node selectors require a file")
            files.update(item.relative_to(clean).as_posix() for item in full.rglob("*.py") if item.is_file())
        else:
            files.add(path)
        normalized.append("::".join([path, *selectors]))
    return sorted(set(normalized)), sorted(files)


def build_artifact(clean_tree: Path, changed_tree: Path, spec: RepoSpec,
                   test_targets: list[str], task_id: str) -> BugArtifact:
    """Separate source corruption from weakening of existing repository tests."""
    clean = Path(clean_tree)
    targets, test_files = _targets(clean, spec, test_targets)
    before, after = _files(clean), _files(Path(changed_tree))
    changed = [path for path in before.keys() | after.keys() if before.get(path) != after.get(path)]
    groups = {"source": [], "test": []}
    for path in changed:
        category = _category(path, spec)
        if category not in groups:
            raise ValueError(f"Edit outside allowed source/tests or in protected configuration: {path}")
        if category == "test" and path not in before:
            raise ValueError("SSR weakening may only change existing test files")
        groups[category].append(path)
    if not groups["source"] or not groups["test"]:
        raise ValueError("An SSR artifact requires both a source bug and weakened tests")
    return BugArtifact(task_id, spec.name, _diff(before, after, groups["source"]),
                       _diff(before, after, groups["test"]),
                       sorted(set(test_files) | set(groups["test"])), targets)


def _patch_paths(patch: str) -> list[str]:
    """Ask Git to parse diff syntax; reject non-text and link/rename operations."""
    if not patch.strip():
        return []
    if re.search(r"^(?:rename |copy |GIT binary patch|Binary files |.*mode 120000)", patch, re.MULTILINE):
        raise ValueError("Only regular, text-file edits are allowed")
    result = subprocess.run(["git", "apply", "--numstat", "-z", "--"], input=patch.encode(),
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Invalid unified diff: " + result.stderr.decode(errors="replace"))
    paths = []
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3 or fields[0] == b"-" or fields[1] == b"-":
            raise ValueError("Unsupported or binary patch")
        paths.append(_relative_path(fields[2].decode("utf-8")))
    if not paths:
        raise ValueError("Patch changes no files")
    return sorted(set(paths))


def _check_patch(patch: str, spec: RepoSpec, category: str, *, required: bool = True) -> list[str]:
    paths = _patch_paths(patch)
    if required and not paths:
        raise ValueError(f"Missing {category} patch")
    for path in paths:
        if _category(path, spec) != category:
            raise ValueError(f"Patch must contain {category}-only edits: {path}")
    return paths


def _apply(tree: Path, patch: str) -> None:
    if not patch.strip():
        return
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tree.parent))
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    result = subprocess.run(["git", "apply", "--whitespace=nowarn", "--"],
                            input=patch.encode(), capture_output=True,
                            cwd=tree, env=env, check=False)
    if result.returncode:
        raise ValueError("Patch did not apply: " + result.stderr.decode(errors="replace"))


def _copy(source: Path, destination: Path) -> None:
    _extract_archive(_source_archive(source), destination)


def _check_syntax(tree: Path, paths: list[str]) -> None:
    for path in paths:
        file = tree / path
        if file.is_file() and file.suffix == ".py":
            ast.parse(file.read_text(encoding="utf-8"), filename=path)


def _run_tests(spec: RepoSpec, tree: Path, targets: list[str], work_root: Path) -> dict:
    """Run generated programs only inside Docker and parse a trusted XML report."""
    command = shlex.join(["python", "-c", _PYTEST_RUNNER, *targets])
    with DockerSandbox(spec.image_digest or spec.image, tree, work_root) as sandbox:
        process = sandbox.exec(command, timeout=90)
    # pytest's collection/no-test errors are invalid artifacts. Failure to start
    # the trusted interpreter/timeout launcher is an environment failure instead.
    # https://docs.pytest.org/en/stable/reference/exit-codes.html
    if process["returncode"] in {125, 126, 127}:
        raise SandboxInfrastructureError(f"Test runner unavailable: {process['stderr']}")
    if process["returncode"] not in {0, 1}:
        raise ValueError(f"Test execution error (exit {process['returncode']}): {process['stderr']}")
    results = parse_junit(process["stdout"])
    if any(value in {"error", "skipped"} for value in results.values()):
        raise ValueError("Selected tests contain execution errors or skips")
    if process["returncode"] == 0 and any(value != "passed" for value in results.values()):
        raise ValueError("Test exit code disagrees with the JUnit report")
    if process["returncode"] == 1 and all(value == "passed" for value in results.values()):
        raise ValueError("Failing test process has no recorded failing test")
    return results


def run_tests(spec: RepoSpec, tree: Path, targets: list[str], work_root: Path) -> dict[str, str]:
    """Public trusted runner; return exact IDs/statuses or raise with diagnostics."""
    targets, _ = _targets(Path(tree), spec, targets)
    return _run_tests(spec, Path(tree), targets, Path(work_root))


def validate_bug(spec: RepoSpec, artifact: BugArtifact, work_root: Path) -> ValidationResult:
    """Require reproducible failure, a test gap, and causal contribution of each file."""
    result = ValidationResult(False, "Artifact has not been checked")
    try:
        if artifact.repo_name != spec.name:
            raise ValueError("Artifact belongs to a different repository")
        clean = Path(spec.source_path).resolve(strict=True)
        targets, selected_files = _targets(clean, spec, artifact.test_targets)
        source_paths = _check_patch(artifact.bug_patch, spec, "source")
        test_paths = _check_patch(artifact.weaken_patch, spec, "test")
        declared = {_relative_path(path) for path in artifact.test_files}
        if not set(test_paths).issubset(declared) or not set(selected_files).issubset(declared):
            raise ValueError("Oracle test files do not cover selected and weakened tests")
        if any(not (clean / path).is_file() or not _under(path, spec.test_roots) for path in declared):
            raise ValueError("Oracle test files must exist in the original test tree")
        Path(work_root).mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="validate-", dir=work_root))
        pristine, buggy, weak = (directory / name for name in ("clean", "bug", "weak"))
        _copy(clean, pristine)
        result.clean_results = _run_tests(spec, pristine, targets, directory)
        if len(result.clean_results) < 5 or any(value != "passed" for value in result.clean_results.values()):
            raise ValueError("At least five selected original tests must all pass")
        _copy(clean, buggy)
        _apply(buggy, artifact.bug_patch)
        _check_syntax(buggy, source_paths)
        result.bug_results = _run_tests(spec, buggy, targets, directory)
        if result.bug_results.keys() != result.clean_results.keys():
            raise ValueError("Bug changes or removes expected oracle test IDs")
        result.fail_to_pass = sorted(key for key, value in result.bug_results.items() if value == "failed")
        result.pass_to_pass = sorted(key for key, value in result.bug_results.items() if value == "passed")
        if not result.fail_to_pass or not result.pass_to_pass:
            raise ValueError("Bug must fail at least one test while preserving at least one pass")
        _copy(buggy, weak)
        _apply(weak, artifact.weaken_patch)
        # Directory targets permit deleted tests. A deleted explicit file is
        # omitted only for the weakening check, never for final oracle scoring.
        weak_targets = [target for target in targets if (weak / target.split("::")[0]).exists()]
        if not weak_targets:
            raise ValueError("Weakening removes every selected test target")
        result.weak_results = _run_tests(spec, weak, weak_targets, directory)
        weak_failures = {key for key, value in result.weak_results.items() if value == "failed"}
        if not result.weak_results.keys() <= result.clean_results.keys():
            raise ValueError("Weakening introduces new test IDs")
        if not weak_failures < set(result.fail_to_pass):
            raise ValueError("Weakening does not hide an original failing test")
        for index, path in enumerate(source_paths):
            inverse = directory / f"inverse-{index}"
            _copy(buggy, inverse)
            original, target = clean / path, inverse / path
            if original.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
            elif target.exists():
                target.unlink()
            checked = _run_tests(spec, inverse, targets, directory)
            if checked.keys() != result.clean_results.keys() or not any(
                checked.get(key) == "passed" for key in result.fail_to_pass
            ):
                raise ValueError(f"Inverse mutation check failed for source file: {path}")
        result.valid, result.reason, result.buggy_tree = True, "All consistency checks passed", str(weak)
    except (ValueError, SyntaxError, ET.ParseError) as error:
        result.reason = str(error)
    return result


def score_repair(spec: RepoSpec, artifact: BugArtifact, repair_patch: str,
                 expected: ValidationResult, work_root: Path) -> dict:
    """Score a source-only patch in a fresh copy with all original tests/configs."""
    results = {}
    try:
        if not expected.valid:
            raise ValueError("Cannot score a repair for an invalid artifact")
        if artifact.repo_name != spec.name:
            raise ValueError("Artifact belongs to a different repository")
        paths = _check_patch(repair_patch, spec, "source", required=False)
        _check_patch(artifact.bug_patch, spec, "source")
        clean = Path(spec.source_path).resolve(strict=True)
        targets, _ = _targets(clean, spec, artifact.test_targets)
        Path(work_root).mkdir(parents=True, exist_ok=True)
        tree = Path(tempfile.mkdtemp(prefix="score-", dir=work_root))
        _copy(clean, tree)
        _apply(tree, artifact.bug_patch)
        # Omitting the test weakening is equivalent to restoring every original
        # test/config after a source-only repair, including unselected test files.
        _apply(tree, repair_patch)
        _check_syntax(tree, paths)
        results = _run_tests(spec, tree, targets, Path(work_root))
        if results.keys() != expected.clean_results.keys():
            raise ValueError("Repair has missing or unexpected oracle test IDs")
        if any(value != "passed" for value in results.values()):
            return {"success": False, "reason": "Oracle tests still fail", "results": results}
        return {"success": True, "reason": "All original oracle tests pass", "results": results}
    except (ValueError, SyntaxError, ET.ParseError) as error:
        return {"success": False, "reason": str(error), "results": results}


def write_artifact(artifact: BugArtifact, destination: Path) -> None:
    """Export five auditable files; the script emits XML for the trusted parser."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {"bug_inject.diff": artifact.bug_patch, "test_weaken.diff": artifact.weaken_patch,
             "test_files.txt": "\n".join(artifact.test_files) + "\n",
             "test_script.sh": "#!/bin/sh\nexec " + shlex.join(
                 ["python", "-c", _PYTEST_RUNNER, *artifact.test_targets]) + "\n",
             "test_parser.py": "import json, sys\nimport xml.etree.ElementTree as ET\n\n"
                 + inspect.getsource(parse_junit)
                 + "\nprint(json.dumps(parse_junit(sys.stdin.read()), sort_keys=True))\n"}
    for name, text in files.items():
        (destination / name).write_text(text, encoding="utf-8", newline="\n")
    (destination / "test_script.sh").chmod(0o755)
