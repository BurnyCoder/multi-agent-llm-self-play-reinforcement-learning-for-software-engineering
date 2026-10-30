"""Exercise the user pipeline with executable fixture bugs and a tiny CPU policy.

These deterministic software checks are never published as model evaluations.
Real generated repositories always use Docker; only this trusted fixture uses
the local test double, keeping CI independent of GPU and container availability.
"""
import json
import shutil
import time
from pathlib import Path

import pytest
from test_learning import TinyPolicy
from test_validation import LocalFixtureSandbox

from multi_agent_ssr.contracts import Generation, Limits, RepoSpec
from multi_agent_ssr.io import write_json


class WorkingFixtureSandbox(LocalFixtureSandbox):
    def __enter__(self):
        return self

    def read(self, path):
        return (self.source / path).read_text()

    def write(self, path, content):
        (self.source / path).write_text(content)

    def snapshot(self, destination):
        shutil.copytree(self.source, destination)


class GamePolicy(TinyPolicy):
    """Script tool decisions while differentiating the actual recorded actions."""
    def __init__(self, torch):
        super().__init__(torch)
        self.injectors = self.solvers = 0

    def generate(self, messages, *, max_new_tokens, tools=None):
        with self.lock:
            system = messages[0]["content"]
            calls = sum(item["role"] == "assistant" for item in messages)
            finish = {"name": "finish", "arguments": {"summary": "fixture complete", "test_targets": ["tests/test_sample.py"]}}
            token = 0
            if "TEAM: injector" in system:
                if calls == 0:
                    self.injectors += 1
                actions = [
                    {"name": "read", "arguments": {"path": "sample/__init__.py"}},
                    {"name": "edit", "arguments": {"path": "sample/__init__.py", "old": "value + 1", "new": "value - 1"}},
                    {"name": "edit", "arguments": {"path": "tests/test_sample.py", "old": "assert increment(0) == 1", "new": "assert increment(0) == -1"}},
                    finish,
                ]
                action = actions[min(calls, 3)] if self.injectors == 1 else finish
                token = int(self.injectors == 1)
            elif "SESSION ROLE: primary" in system:
                if calls == 0:
                    self.solvers += 1
                token = int(self.solvers % 2 == 1)
                action = ([{"name": "subagent", "arguments": {"agent": "general",
                    "description": "Fixture repair", "prompt": "fix" if token else "inspect"}},
                    {"name": "read", "arguments": {"path": "tests/test_sample.py"}}, finish][min(calls, 2)])
            else:
                token = int(messages[1]["content"] == "fix")
                action = ({"name": "edit", "arguments": {"path": "sample/__init__.py",
                    "old": "value - 1", "new": "value + 1"}} if token and calls == 0 else finish)
            return Generation(json.dumps(action), [1], [token],
                [float(self.weights.log_softmax(0)[token].detach())], self.version)


def test_complete_game_update_evaluation_and_reporting(tmp_path, monkeypatch):
    from multi_agent_ssr import reporting, rollouts, validation
    from multi_agent_ssr.learning import update_policy
    torch = pytest.importorskip("torch")
    fixture = Path(__file__).parent / "fixtures/tiny_repo"
    clean = tmp_path / "clean"
    shutil.copytree(fixture, clean)
    spec = RepoSpec("fixture", "https://example.invalid/fixture", "fixture", "fixture",
                    str(clean), "train", ["sample"], ["tests"], "MIT")

    class FreshSandbox(WorkingFixtureSandbox):
        def __init__(self, image, source, work_root, readonly=False):
            self.source = Path(work_root) / "agent"
            self.source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, self.source)

    monkeypatch.setattr(rollouts, "DockerSandbox", FreshSandbox)
    monkeypatch.setattr(validation, "DockerSandbox", LocalFixtureSandbox)
    monkeypatch.setattr(reporting, "render_figures", lambda *args: {})
    monkeypatch.setattr(reporting, "render_paper", lambda *args: None)
    policy = GamePolicy(torch)
    run = tmp_path / "run"
    groups, metrics = rollouts.collect_group(policy, spec, run, Limits(), group_index=0,
        injector_samples=2, repair_samples=4, max_bug_order=1, deadline=time.time() + 300)
    assert metrics["valid_bugs"] == 1
    assert metrics["repair_success_rate"] == 0.5
    assert all(any(action.parent_session_id for action in episode.actions) for episode in groups[1])
    assert "value + 1" in (clean / "sample/__init__.py").read_text()
    before = policy.adapter_fingerprint()
    learned = update_policy(policy, groups, torch.optim.SGD(policy.trainable_parameters(), lr=0.1), capacity=40960)
    assert learned["optimizer_steps"] == 2
    assert policy.adapter_fingerprint() != before
    assert len(list((run / "artifact_records").glob("*.json"))) == 1
    write_json(run / "manifest.json", {"run_id": "software-test-fixture"})
    (run / "evaluation.jsonl").write_text("\n".join(json.dumps({
        "task_id": episode.episode_id, "condition": "trained", "source": "synthetic",
        "success": episode.reward == 1, "status": "completed"}) for episode in groups[1]))
    reporting.generate_report(run, tmp_path / "reports", tmp_path / "site")
    analysis = json.loads((tmp_path / "reports/summary.json").read_text())["analysis"]
    assert analysis["conditions"]["synthetic"]["trained"]["completed"] == 4
    assert (tmp_path / "site/data.json").is_file()
