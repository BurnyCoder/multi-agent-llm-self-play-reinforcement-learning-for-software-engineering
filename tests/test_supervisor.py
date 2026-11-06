"""A blocked model worker must not reset or overrun the persisted GPU deadline."""
import json
import subprocess

import pytest


def test_hard_deadline_kills_only_own_worker_and_cleans_its_label(tmp_path, monkeypatch):
    from multi_agent_ssr import supervisor
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nbudget_seconds = 60\n")
    signals, cleaned, waits = [], [], []
    clock = [1000.0]
    monkeypatch.setattr("multi_agent_ssr.experiment.time.time", lambda: clock[0])

    class Worker:
        pid = 12345
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            waits.append(timeout)
            if self.calls == 1:
                # A forward wall-clock adjustment can expire the absolute budget
                # before an originally computed monotonic wait would finish.
                clock[0] = 1061.0
                raise subprocess.TimeoutExpired("test worker", timeout)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Worker())
    monkeypatch.setattr(supervisor.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(supervisor, "cleanup_containers", cleaned.append)
    assert supervisor.bounded_phase("pilot", config, tmp_path / "run") == 124
    assert signals == [(12345, supervisor.signal.SIGKILL)]
    assert waits == [1.0, 15]
    assert len(cleaned) == 1 and cleaned[0].startswith("ssr-")
    failure = json.loads((tmp_path / "run/failure.json").read_text())
    assert failure["remaining_seconds"] == 0


def test_normal_wait_slice_timeout_does_not_kill_worker_early(tmp_path, monkeypatch):
    from multi_agent_ssr import supervisor
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nbudget_seconds = 60\n")
    clock, waits, signals, cleaned = [1000.0], [], [], []
    monkeypatch.setattr("multi_agent_ssr.experiment.time.time", lambda: clock[0])

    class Worker:
        pid = 12345

        def wait(self, timeout):
            waits.append(timeout)
            if len(waits) == 1:
                clock[0] += 1
                raise subprocess.TimeoutExpired("test worker", timeout)
            return 7

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Worker())
    monkeypatch.setattr(supervisor.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(supervisor, "cleanup_containers", cleaned.append)
    run = tmp_path / "run"
    assert supervisor.bounded_phase("pilot", config, run) == 7
    assert waits == [1.0, 1.0]
    assert not signals and not (run / "failure.json").exists()
    assert len(cleaned) == 1


@pytest.mark.parametrize("spawn_elapsed, expected_timeout", [(0.0, 1.0), (59.75, 0.25)])
def test_ordinary_worker_exit_uses_bounded_remaining_wait_and_cleans_label(tmp_path, monkeypatch,
                                                                       spawn_elapsed, expected_timeout):
    from multi_agent_ssr import supervisor
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nbudget_seconds = 60\n")
    clock, waits, signals, cleaned, launched = [1000.0], [], [], [], []
    monkeypatch.setattr("multi_agent_ssr.experiment.time.time", lambda: clock[0])

    class Worker:
        pid = 12345

        def wait(self, timeout):
            waits.append(timeout)
            return 0

    def launch(*args, **kwargs):
        launched.append(kwargs)
        clock[0] += spawn_elapsed
        return Worker()

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(supervisor, "cleanup_containers", cleaned.append)
    run = tmp_path / "run"
    assert supervisor.bounded_phase("pilot", config, run) == 0
    assert waits == [expected_timeout]
    assert not signals and not (run / "failure.json").exists()
    assert launched[0]["start_new_session"] is True
    assert cleaned == [launched[0]["env"]["SSR_RUN_ID"]]


def test_cleanup_filters_on_exact_owned_run_label(monkeypatch):
    from multi_agent_ssr import supervisor
    calls = []

    def docker(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "owned-container\n", "")

    monkeypatch.setattr(subprocess, "run", docker)
    supervisor.cleanup_containers("ssr-test")
    assert calls[0][-1] == "label=org.multi-agent-ssr.run=ssr-test"
    assert calls[1] == ["docker", "rm", "--force", "owned-container"]
