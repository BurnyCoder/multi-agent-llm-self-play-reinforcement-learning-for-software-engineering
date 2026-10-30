"""Freeze independent function clusters and score repairs in pristine test trees."""

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_agent_ssr import synthetic
from multi_agent_ssr.contracts import ActionTrace, Episode, Generation, Limits, RepoSpec
from multi_agent_ssr.validation import _copy

SOURCE = '''# Keep this comment and the Unicode character: é
def first(value):
    """Return whether a value is positive."""
    return value > 0

def second(value):
    """Return whether a value is below ten."""
    return value < 10
'''


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """The trusted runner is mocked; generated programs never execute on the host."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "repository"
    (source / "src/humanize").mkdir(parents=True)
    (source / "src/humanize/number.py").write_text(SOURCE, encoding="utf-8")
    (source / "src/humanize/_version.py").write_text("__version__ = 'fixture'\n")
    (source / "tests").mkdir()
    for target in synthetic.TEST_TARGETS:
        (source / target).write_text("# immutable oracle\n")
    spec = RepoSpec("humanize", "https://example.invalid/humanize", "a" * 40,
                    "fixture", str(source), "test", ["src/humanize"], ["tests"], "MIT", "fixture-digest")
    monkeypatch.setattr(synthetic, "load_repositories", lambda: [spec])
    data = tmp_path / "private-data"
    monkeypatch.setattr(synthetic, "data_directory", lambda: data)
    calls = []

    def run_tests(specification, tree, targets, work_root):
        calls.append(Path(tree))
        assert targets == synthetic.TEST_TARGETS
        assert all((Path(tree) / target).read_text() == "# immutable oracle\n" for target in targets)
        changed = (Path(tree) / "src/humanize/number.py").read_text(encoding="utf-8") != SOURCE
        return {f"tests/test_number.py::test_{index}": "failed" if changed and index == 0 else "passed"
                for index in range(6)}

    monkeypatch.setattr(synthetic, "run_tests", run_tests)
    records = synthetic.prepare_synthetic(data, count=2)
    return data, records, spec, calls


def test_mutations_preserve_surrounding_source_and_are_deterministic():
    first = list(synthetic.mutation_candidates(SOURCE, "src/humanize/number.py"))
    second = list(synthetic.mutation_candidates(SOURCE, "src/humanize/number.py"))
    assert first == second
    assert {item["function"] for item in first} == {"first", "second"}
    for item in first:
        ast.parse(item["source"])
        assert item["source"].startswith("# Keep this comment and the Unicode character: é\n")
        assert '"""Return whether a value is positive."""' in item["source"]


def test_ast_offsets_are_bytes_when_unicode_precedes_a_mutation():
    source = 'def first(value):\n    return ("é" == value) and True\n'
    candidates = list(synthetic.mutation_candidates(source, "src/humanize/number.py"))
    assert any(item["mutation"] == "boolean_constant" for item in candidates)
    for candidate in candidates:
        ast.parse(candidate["source"])
        assert "é" in candidate["source"]


def test_preparation_freezes_one_case_per_function_and_hides_answers(prepared):
    data, records, _, calls = prepared
    assert len(records) == len({row["cluster_id"] for row in records}) == 2
    assert all(row["fail_to_pass"] and row["pass_to_pass"] for row in records)
    public = json.loads(Path("manifests/synthetic.json").read_text())
    assert public["diagnostic"] == "visible_existing_tests_without_ssr_weakening"
    assert all("bug_patch" not in row and "clean_results" not in row for row in public["tasks"])
    assert all((data / "synthetic" / row["buggy_tree"]).is_dir() for row in records)
    before = len(calls)
    assert synthetic.prepare_synthetic(data, count=2) == records
    assert len(calls) == before  # Repeated preparation cannot select a different cohort.


def test_preparation_rejects_a_changed_requested_cohort(prepared):
    data, _, _, _ = prepared
    with pytest.raises(ValueError, match="count"):
        synthetic.prepare_synthetic(data, count=1)


def test_cached_mutant_integrity_is_checked_before_model_evaluation(prepared):
    data, records, _, _ = prepared
    mutant = data / "synthetic" / records[0]["buggy_tree"] / records[0]["source_file"]
    mutant.write_text("# corrupt cached mutant\n")
    with pytest.raises(ValueError, match="immutable"):
        synthetic.prepare_synthetic(data, count=2)


class FixtureSandbox:
    """Fresh local file copies only; no shell or generated code execution."""

    def __init__(self, image, source, work_root):
        self.tree = Path(work_root) / "container-copy"
        _copy(source, self.tree)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def snapshot(self, destination):
        _copy(self.tree, destination)


def test_evaluation_restores_tests_and_does_not_leak_reference_patch(prepared, monkeypatch, tmp_path):
    _, records, _, _ = prepared
    monkeypatch.setattr(synthetic, "DockerSandbox", FixtureSandbox)
    seen_prompts = []

    def repair(policy, sandbox, team, task_id, prompt, log_dir, **kwargs):
        seen_prompts.append(prompt)
        assert team == "solver"
        assert "--- a/" not in prompt and "return value" not in prompt
        (sandbox.tree / "src/humanize/number.py").write_text(SOURCE, encoding="utf-8")
        (sandbox.tree / "tests/test_number.py").write_text("# attempted oracle change\n")
        action = ActionTrace(task_id, "child", "root", "general",
                             Generation("repair", [1], [2], [-0.5]))
        return Episode(task_id, task_id, team, [action], completed=True)

    monkeypatch.setattr(synthetic, "run_team", repair)
    policy = SimpleNamespace(version=0, adapter_fingerprint=lambda: "fixed-policy")
    rows = synthetic.evaluate_synthetic(policy, "base", tmp_path / "run", Limits(), time.time() + 10000)
    assert len(rows) == len(records) == len(seen_prompts)
    assert all(row["status"] == "completed" and row["success"] is True for row in rows)
    assert all(row["delegations"] == 1 and row["generated_tokens"] == 1 for row in rows)
    # Same-condition resume reuses results, never a previous agent workspace.
    again = synthetic.evaluate_synthetic(policy, "base", tmp_path / "run", Limits(), time.time() + 10000)
    assert len(seen_prompts) == len(records)
    assert [row["task_id"] for row in again] == [row["task_id"] for row in rows]


def test_global_budget_is_unscored_but_policy_budget_is_failure(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(synthetic, "DockerSandbox", FixtureSandbox)
    policy = SimpleNamespace(version=0, adapter_fingerprint=lambda: "fixed-policy")
    expired = synthetic.evaluate_synthetic(policy, "expired", tmp_path / "run", Limits(), time.time() - 1)
    assert all(row["success"] is None and row["status"] == "not_run_budget" for row in expired)

    def exhausted(policy, sandbox, team, task_id, prompt, log_dir, **kwargs):
        return Episode(task_id, task_id, team, result={"budget_exhausted": True}, completed=True)

    monkeypatch.setattr(synthetic, "run_team", exhausted)
    budgeted = synthetic.evaluate_synthetic(policy, "budgeted", tmp_path / "run", Limits(), time.time() + 10000)
    assert all(row["success"] is False and row["status"] == "completed" for row in budgeted)


def test_docker_infrastructure_failure_is_not_a_model_failure(prepared, monkeypatch, tmp_path):
    class UnavailableSandbox:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Docker unavailable")

    monkeypatch.setattr(synthetic, "DockerSandbox", UnavailableSandbox)
    rows = synthetic.evaluate_synthetic(SimpleNamespace(version=0), "base", tmp_path / "run",
                                        Limits(), time.time() + 10000)
    assert all(row["success"] is None and row["status"] == "infrastructure_error" for row in rows)


@pytest.mark.parametrize("error,repair,expected", [
    ("ContextLimitError: latest interaction cannot fit", True, True),
    ("ContextLimitError: latest interaction cannot fit", False, False),
    ("OutOfMemoryError: CUDA allocation failed", True, None),
])
def test_context_budget_grades_available_patch_but_gpu_errors_are_unscored(
        prepared, monkeypatch, tmp_path, error, repair, expected):
    """An interaction limit is policy behavior; a failed device is infrastructure."""
    _, records, _, test_calls = prepared
    monkeypatch.setattr(synthetic, "DockerSandbox", FixtureSandbox)

    def interrupted(policy, sandbox, team, task_id, prompt, log_dir, **kwargs):
        if repair:
            (sandbox.tree / "src/humanize/number.py").write_text(SOURCE, encoding="utf-8")
        return Episode(task_id, task_id, team, result={"budget_exhausted": True},
                       error=error, completed=False)

    monkeypatch.setattr(synthetic, "run_team", interrupted)
    before = len(test_calls)
    rows = synthetic.evaluate_synthetic(SimpleNamespace(version=0), "base", tmp_path / "run",
                                        Limits(), time.time() + 10000)
    assert all(row["success"] is expected for row in rows)
    if expected is None:
        assert all(row["status"] == "infrastructure_error" for row in rows)
        assert len(test_calls) == before
    else:
        assert all(row["status"] == "completed" and row["policy_budget_exhausted"] for row in rows)
        assert len(test_calls) == before + len(records)
