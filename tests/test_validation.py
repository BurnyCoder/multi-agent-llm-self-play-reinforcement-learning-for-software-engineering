"""Exercise scoring with actual pytest runs on a small, trusted fixture package."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from multi_agent_ssr import validation
from multi_agent_ssr.contracts import RepoSpec, ValidationResult


class LocalFixtureSandbox:
    """Test double only: the fixture is our code, never a generated program."""

    def __init__(self, image, source, work_root, readonly=False):
        self.source = Path(source)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def exec(self, command, timeout=90):
        args = shlex.split(command)
        assert args[0] == "python", "Fixture evaluator only executes trusted Python"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        result = subprocess.run([sys.executable, *args[1:]], cwd=self.source,
                                capture_output=True, text=True, timeout=timeout,
                                env=env, encoding="utf-8", check=False)
        return {"stdout": result.stdout, "stderr": result.stderr,
                "returncode": result.returncode}


@pytest.fixture
def repository(tmp_path, monkeypatch):
    clean = tmp_path / "clean"
    shutil.copytree(Path(__file__).parent / "fixtures" / "tiny_repo", clean)
    changed = tmp_path / "changed"
    shutil.copytree(clean, changed)
    spec = RepoSpec("tiny", "https://example.invalid/tiny", "fixture", "fixture",
                    str(clean), "train", ["sample"], ["tests"])
    monkeypatch.setattr(validation, "DockerSandbox", LocalFixtureSandbox)
    return clean, changed, spec, tmp_path / "work"


def make_bug(repository):
    clean, changed, spec, _ = repository
    implementation = changed / "sample" / "__init__.py"
    implementation.write_text(implementation.read_text().replace("value + 1", "value - 1"))
    tests = changed / "tests" / "test_sample.py"
    tests.write_text(tests.read_text().replace("assert increment(0) == 1",
                                             "assert increment(0) == -1"))
    return validation.build_artifact(clean, changed, spec, ["tests/test_sample.py"], "bug-1")


def test_validated_bug_and_actual_repair(repository):
    clean, changed, spec, work = repository
    artifact = make_bug(repository)
    expected = validation.validate_bug(spec, artifact, work)
    assert expected.valid, expected.reason
    assert len(expected.clean_results) == 6
    assert len(expected.fail_to_pass) == 2
    assert len(expected.pass_to_pass) == 4
    assert Path(expected.buggy_tree).is_dir()
    patch = validation.source_diff(changed, clean, spec)
    scored = validation.score_repair(spec, artifact, patch, expected, work)
    assert scored["success"], scored
    failed = validation.score_repair(spec, artifact, "", expected, work)
    assert not failed["success"]


def test_weakening_may_remove_an_exposing_test(repository):
    clean, changed, spec, work = repository
    artifact = make_bug(repository)
    tests = changed / "tests" / "test_sample.py"
    text = tests.read_text()
    start = text.index("def test_increment_zero")
    end = text.index("def test_increment_negative")
    tests.write_text(text[:start] + text[end:])
    artifact = validation.build_artifact(clean, changed, spec, ["tests/test_sample.py"], "removed")
    result = validation.validate_bug(spec, artifact, work)
    assert result.valid, result.reason
    assert len(result.weak_results) == 5


def test_cannot_score_by_changing_tests_or_configuration(repository):
    clean, changed, spec, work = repository
    artifact = make_bug(repository)
    expected = validation.validate_bug(spec, artifact, work)
    result = validation.score_repair(spec, artifact, artifact.weaken_patch, expected, work)
    assert not result["success"]
    assert "source" in result["reason"].lower()
    (changed / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts='--ignore=tests'\n")
    with pytest.raises(ValueError, match="outside|protected|allowed"):
        validation.build_artifact(clean, changed, spec, ["tests/test_sample.py"], "config")


def test_missing_test_ids_and_empty_reports_fail_closed():
    with pytest.raises(ValueError, match="empty|test"):
        validation.parse_junit("<testsuites><testsuite tests='0'/></testsuites>")
    with pytest.raises(ValueError, match="node|id"):
        validation.parse_junit("<testsuites><testsuite><testcase name='x'/></testsuite></testsuites>")


def test_errors_and_skips_are_not_passes():
    xml = """<testsuites><testsuite>
    <testcase name="a"><properties><property name="ssr_nodeid" value="tests/a.py::a"/></properties><error/></testcase>
    <testcase name="b"><properties><property name="ssr_nodeid" value="tests/a.py::b"/></properties><skipped/></testcase>
    </testsuite></testsuites>"""
    assert validation.parse_junit(xml) == {"tests/a.py::a": "error", "tests/a.py::b": "skipped"}


def test_original_syntax_errors_and_missing_targets_are_rejected(repository):
    clean, changed, spec, work = repository
    artifact = make_bug(repository)
    (clean / "sample" / "__init__.py").write_text("def broken(\n")
    result = validation.validate_bug(spec, artifact, work)
    assert not result.valid
    with pytest.raises(ValueError):
        validation.build_artifact(clean, changed, spec, ["../outside.py"], "escape")


def test_orphan_source_edits_fail_inverse_mutation(repository):
    clean, changed, spec, work = repository
    (clean / "sample" / "unused.py").write_text("value = 1\n")
    (changed / "sample" / "unused.py").write_text("value = 2\n")
    artifact = make_bug(repository)
    result = validation.validate_bug(spec, artifact, work)
    assert not result.valid
    assert "inverse" in result.reason.lower()


def test_export_writes_five_canonical_artifacts(repository, tmp_path):
    artifact = make_bug(repository)
    destination = tmp_path / "export"
    validation.write_artifact(artifact, destination)
    assert {path.name for path in destination.iterdir()} == {
        "bug_inject.diff", "test_weaken.diff", "test_files.txt",
        "test_script.sh", "test_parser.py",
    }
    compile((destination / "test_parser.py").read_text(), "test_parser.py", "exec")


def test_patch_cannot_escape_source_tree(repository):
    _, _, spec, work = repository
    artifact = make_bug(repository)
    artifact.bug_patch = "--- a/../../outside.py\n+++ b/../../outside.py\n@@ -1 +1 @@\n-x\n+y\n"
    result = validation.validate_bug(spec, artifact, work)
    assert not result.valid


@pytest.mark.parametrize("error", [RuntimeError("Docker daemon unavailable"), OSError("Host disk unavailable")])
def test_infrastructure_failures_propagate_instead_of_becoming_rewards(repository, monkeypatch, error):
    _, _, spec, work = repository
    artifact = make_bug(repository)
    def unavailable(*args):
        raise error
    monkeypatch.setattr(validation, "_run_tests", unavailable)
    with pytest.raises(type(error), match=str(error)):
        validation.validate_bug(spec, artifact, work)
    with pytest.raises(type(error), match=str(error)):
        validation.score_repair(spec, artifact, "", ValidationResult(True, "valid"), work)


def test_collection_failures_still_reject_policy_artifacts(repository, monkeypatch):
    _, _, spec, work = repository
    artifact = make_bug(repository)
    def collection_error(*args):
        raise ValueError("Test collection failed")
    monkeypatch.setattr(validation, "_run_tests", collection_error)
    assert not validation.validate_bug(spec, artifact, work).valid
    assert not validation.score_repair(spec, artifact, "", ValidationResult(True, "valid"), work)["success"]


@pytest.mark.parametrize("returncode", [2, 5, 124, 127])
def test_missing_tests_and_timeouts_differ_from_unavailable_runner(repository, monkeypatch, returncode):
    clean, _, spec, work = repository
    monkeypatch.setattr(LocalFixtureSandbox, "exec", lambda *args, **kwargs: {
        "stdout": "", "stderr": "fixture diagnostic", "returncode": returncode})
    error = validation.SandboxInfrastructureError if returncode == 127 else ValueError
    with pytest.raises(error, match="fixture diagnostic"):
        validation.run_tests(spec, clean, ["tests/test_sample.py"], work)
