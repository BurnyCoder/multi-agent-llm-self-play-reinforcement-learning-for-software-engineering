"""Human evaluation must preserve task pins, oracle separation, and honest coverage."""

import json
import time
from pathlib import Path

import pytest

from multi_agent_ssr.contracts import Episode, Limits
from multi_agent_ssr.evaluation import (
    HardenedDockerClient,
    _verify_base_content,
    agent_prompt,
    evaluate_human,
    source_roots,
)


def test_prompt_contains_issue_but_never_private_oracles():
    task = {"instance_id": "r-1", "problem_statement": "Observed wrong output.",
            "patch": "SECRET_GOLD", "test_patch": "SECRET_TESTS", "eval_script": "SECRET_SCRIPT"}
    prompt = agent_prompt(task)
    assert "Observed wrong output." in prompt
    assert "SECRET" not in prompt


def test_source_roots_use_existing_python_package_layouts(tmp_path):
    (tmp_path / "src" / "_pytest").mkdir(parents=True)
    (tmp_path / "testing").mkdir()
    assert source_roots(tmp_path) == ["src"]


def test_official_container_creation_is_hardened():
    class Containers:
        def create(self, **kwargs):
            return kwargs

    class Client:
        containers = Containers()

    args = HardenedDockerClient(Client()).containers.create(
        image="test", cap_add=["SYS_ADMIN"], network_mode="bridge", privileged=True)
    assert args["network_mode"] == "none"
    assert args["cap_drop"] == ["ALL"]
    assert args["security_opt"] == ["no-new-privileges:true"]
    assert not args.get("privileged")
    assert "cap_add" not in args


def test_base_verification_ignores_mode_only_changes_but_rejects_content_or_empty_additions(monkeypatch):
    mode = "diff --git a/a.py b/a.py\nold mode 100644\nnew mode 100755\n"
    monkeypatch.setattr("multi_agent_ssr.evaluation._command", lambda *a, **k: mode)
    _verify_base_content("fixture-container", "a" * 40)
    for diff in [mode + "@@ -1 +1 @@\n-old\n+new\n", "diff --git a/empty b/empty\nnew file mode 100644\n"]:
        monkeypatch.setattr("multi_agent_ssr.evaluation._command", lambda *a, fixture=diff, **k: fixture)
        with pytest.raises(ValueError, match="content differs"):
            _verify_base_content("fixture-container", "a" * 40)


def setup_environment(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "requests").mkdir()
    task = {"instance_id": "requests-1", "repo": "psf/requests", "base_commit": "abc",
            "problem_statement": "Fix a bug", "patch": "gold", "test_patch": "tests"}
    (tmp_path / "human_tasks_private.json").write_text(json.dumps([task]))
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "evaluation.json").write_text(json.dumps({"tasks": [task]}))
    monkeypatch.setattr("multi_agent_ssr.evaluation._PROJECT_ROOT", tmp_path)
    (tmp_path / "human_envs.json").write_text(json.dumps({"tasks": [{
        "instance_id": "requests-1", "status": "ready", "source_path": str(source),
        "agent_image": "agent@sha256:123", "image_digest": "official@sha256:123",
        "source_roots": ["requests"], "test_roots": ["tests"], "base_commit": "abc"}]}))
    monkeypatch.setenv("SSR_DATA_DIR", str(tmp_path))
    return tmp_path / "run"


def test_deadline_records_missing_coverage_without_running_model(tmp_path, monkeypatch):
    run = setup_environment(tmp_path, monkeypatch)
    rows = evaluate_human(object(), "base", run, Limits(), time.time() - 1)
    assert rows[0]["status"] == "pending"
    assert rows[0]["success"] is None


def test_completed_task_is_idempotent_on_resume(tmp_path, monkeypatch):
    run = setup_environment(tmp_path, monkeypatch)
    run.mkdir()
    row = {"task_id": "requests-1", "condition": "base", "source": "human",
           "success": False, "status": "completed"}
    (run / "evaluation.jsonl").write_text(json.dumps(row) + "\n")
    rows = evaluate_human(object(), "base", run, Limits(), time.time() + 60)
    assert rows == [row]


def test_grading_result_preserves_false_and_reports_infrastructure_errors(tmp_path, monkeypatch):
    run = setup_environment(tmp_path, monkeypatch)

    class Sandbox:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def snapshot(self, destination: Path):
            destination.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("multi_agent_ssr.evaluation.DockerSandbox", Sandbox)
    monkeypatch.setattr("multi_agent_ssr.evaluation.run_team", lambda *a, **k: Episode("ep", "requests-1", "solver", completed=True))
    monkeypatch.setattr("multi_agent_ssr.evaluation.source_diff", lambda *a: "patch")
    monkeypatch.setattr("multi_agent_ssr.evaluation.grade_prediction", lambda *a, **k: {"success": False, "status": "completed", "reason": "tests fail"})
    rows = evaluate_human(object(), "base", run, Limits(), time.time() + 1000)
    assert rows[0]["success"] is False
    assert rows[0]["status"] == "completed"
    monkeypatch.setattr("multi_agent_ssr.evaluation.grade_prediction", lambda *a, **k: {"success": None, "status": "infra_error", "reason": "container error"})
    rows = evaluate_human(object(), "trained", run, Limits(), time.time() + 1000)
    assert rows[0]["success"] is None
    assert rows[0]["status"] == "infra_error"


def test_official_grading_attempts_cannot_reuse_a_stale_cached_patch(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from multi_agent_ssr.evaluation import grade_prediction

    attempts = []
    client = SimpleNamespace(containers=object(), close=lambda: None)
    monkeypatch.setitem(sys.modules, "docker", SimpleNamespace(from_env=lambda **kw: client))
    monkeypatch.setitem(sys.modules, "swebench.harness.utils", SimpleNamespace(make_test_spec=lambda row: row))

    def score(spec, prediction, client, run_id, **kwargs):
        attempts.append((run_id, prediction["model_patch"]))
        return "fixture", {"fixture": {"resolved": False}}

    official = SimpleNamespace(RUN_EVALUATION_LOG_DIR=tmp_path / "original", run_instance=score)
    monkeypatch.setattr("multi_agent_ssr.evaluation.importlib.import_module", lambda name: official)
    for patch in ["first", "second"]:
        result = grade_prediction({"instance_id": "fixture"}, patch, "base", tmp_path,
                                  image_digest="fixture@sha256:pin", timeout=1)
        assert result["status"] == "completed"
    assert attempts[0][0] != attempts[1][0]
    assert official.RUN_EVALUATION_LOG_DIR == tmp_path / "original"


def test_environment_preparation_uses_and_preserves_committed_image_pins(tmp_path, monkeypatch):
    from multi_agent_ssr.evaluation import prepare_human_environments
    from multi_agent_ssr.io import write_json

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    task = {"instance_id": "fixture-1", "repo": "fixture/repo", "base_commit": "a" * 40, "image": "fixture:latest"}
    pin = {**task, "image_digest": "fixture@sha256:immutable", "agent_image": "original-agent", "status": "ready"}
    tasks = [task, {**task, "instance_id": "fixture-2"}]
    pins = [pin, {**pin, "instance_id": "fixture-2"}]
    (manifest_dir / "evaluation.json").write_text(json.dumps({"revision": "dataset-pin", "tasks": tasks}))
    (manifest_dir / "image_pins.json").write_text(json.dumps({"dataset_revision": "dataset-pin", "tasks": pins}))
    monkeypatch.setattr("multi_agent_ssr.evaluation._PROJECT_ROOT", tmp_path)
    received = []
    public_write_sizes = []

    def save(path, data):
        if path.name == "image_pins.json":
            public_write_sizes.append(len(data["tasks"]))
        write_json(path, data)

    def prepare(task, destination):
        received.append(task)
        return {**task, "agent_image": "rebuilt-agent", "status": "ready", "source_path": str(destination / "source")}

    monkeypatch.setattr("multi_agent_ssr.evaluation._prepare_one", prepare)
    monkeypatch.setattr("multi_agent_ssr.evaluation.write_json", save)
    local = prepare_human_environments(tmp_path / "data")
    assert received[0]["image_digest"] == pin["image_digest"]
    assert local["tasks"][0]["agent_image"] == "rebuilt-agent"
    assert json.loads((manifest_dir / "image_pins.json").read_text())["tasks"][0] == pin
    assert min(public_write_sizes) == len(pins)


@pytest.mark.parametrize("case,expected", [("malformed_patch", "completed"),
                                           ("runtime_failure", "infra_error"),
                                           ("runtime_failure_with_budget", "infra_error"),
                                           ("context_budget", "completed")])
def test_agent_failures_are_distinguished_from_infrastructure(tmp_path, monkeypatch, case, expected):
    from contextlib import nullcontext
    from types import SimpleNamespace

    run = setup_environment(tmp_path, monkeypatch)
    sandbox = SimpleNamespace(snapshot=lambda path: path.mkdir())
    monkeypatch.setattr("multi_agent_ssr.evaluation.DockerSandbox", lambda *a, **k: nullcontext(sandbox))
    error = {"runtime_failure": "RuntimeError: CUDA out of memory",
             "runtime_failure_with_budget": "RuntimeError: CUDA out of memory",
             "context_budget": "ContextLimitError: budget"}.get(case)
    episode = Episode("ep", "requests-1", "solver", completed=error is None, error=error,
                      result={"budget_exhausted": case == "runtime_failure_with_budget"})
    monkeypatch.setattr("multi_agent_ssr.evaluation.run_team", lambda *a, **k: episode)

    def patch(*args):
        if case == "malformed_patch":
            raise ValueError("Agent created an unsupported source artifact")
        return ""

    monkeypatch.setattr("multi_agent_ssr.evaluation.source_diff", patch)
    monkeypatch.setattr("multi_agent_ssr.evaluation.grade_prediction", lambda *a, **k: {"success": False, "status": "completed"})
    row = evaluate_human(object(), "base", run, Limits(), time.time() + 1000)[0]
    assert row["status"] == expected
    assert row["success"] is (False if expected == "completed" else None)
