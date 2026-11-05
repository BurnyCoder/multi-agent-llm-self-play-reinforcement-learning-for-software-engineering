"""A detached terminal must not interrupt durable research records."""

import io
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from multi_agent_ssr.io import EventLog
from multi_agent_ssr.model import LocalPolicy


class DisconnectedTerminal(io.StringIO):
    """Reproduce a terminal consumer exiting during either write or flush."""

    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    def write(self, text):
        if self.failure == "write":
            raise BrokenPipeError("Terminal reader disconnected")
        return super().write(text)

    def flush(self):
        if self.failure == "flush":
            raise BrokenPipeError("Terminal reader disconnected")
        return super().flush()


@pytest.mark.parametrize("failure", ["write", "flush"])
@pytest.mark.parametrize("kind", ["episode", "model"])
def test_disconnected_terminal_preserves_complete_events(tmp_path, monkeypatch, failure, kind):
    path = tmp_path / "events.jsonl"
    monkeypatch.setattr(sys, "stdout", DisconnectedTerminal(failure))
    if kind == "episode":
        emit = EventLog(path).emit
    else:
        # Exercise real model logging without importing or executing an LLM.
        policy = LocalPolicy.__new__(LocalPolicy)
        policy.log_path, policy._log_lock = path, threading.RLock()
        emit = policy.log_event
    full_text = "αβγ\n" * 10_000
    emit("prompt", text=full_text)
    emit("output_chunk", text=full_text, stream_end=True)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["prompt", "output_chunk"]
    assert all(record["text"] == full_text and record["timestamp"] for record in records)


def test_disk_logging_errors_remain_fatal(tmp_path):
    path = tmp_path / "events.jsonl"
    path.mkdir()
    with pytest.raises(IsADirectoryError):
        EventLog(path).emit("cannot_save")


def test_phase_failure_metadata_survives_disconnected_terminal(tmp_path, monkeypatch):
    from multi_agent_ssr import experiment, prepare

    monkeypatch.chdir(tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/model.json").write_text('{"revision": "frozen-test-revision"}')
    config = {"model": {}, "limits": {}, "experiment": {
        "budget_seconds": 60, "human_tasks": 32, "synthetic_tasks": 16}}
    monkeypatch.setattr(experiment, "load_config", lambda _: config)
    monkeypatch.setattr(experiment, "require_preregistered", lambda: "committed-test-code")
    monkeypatch.setattr(experiment.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="test GPU"))
    monkeypatch.setattr(prepare, "preparation_evidence", lambda: {})
    monkeypatch.setattr(sys, "stdout", DisconnectedTerminal("flush"))

    def failed_preflight(*args):
        raise RuntimeError("Deliberate preflight failure")

    monkeypatch.setitem(sys.modules, "multi_agent_ssr.learning", SimpleNamespace(run_preflight=failed_preflight))
    with pytest.raises(RuntimeError, match="Deliberate preflight failure"):
        experiment.execute("preflight", tmp_path / "config.toml", tmp_path / "run")
    failure = json.loads((tmp_path / "run/failure.json").read_text())
    assert failure["error"] == "RuntimeError: Deliberate preflight failure"
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["phase_started", "phase_failed"]
