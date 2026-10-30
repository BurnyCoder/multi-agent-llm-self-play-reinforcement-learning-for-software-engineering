"""A blocked model worker must not reset or overrun the persisted GPU deadline."""
import json
import subprocess


def test_hard_deadline_kills_only_own_worker_and_cleans_its_label(tmp_path, monkeypatch):
    from multi_agent_ssr import supervisor
    config = tmp_path / "config.toml"
    config.write_text("[experiment]\nbudget_seconds = 60\n")
    signals, cleaned = [], []

    class Worker:
        pid = 12345
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("test worker", timeout)
            return -9

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Worker())
    monkeypatch.setattr(supervisor.os, "killpg", lambda *args: signals.append(args))
    monkeypatch.setattr(supervisor, "cleanup_containers", cleaned.append)
    assert supervisor.bounded_phase("pilot", config, tmp_path / "run") == 124
    assert signals == [(12345, supervisor.signal.SIGKILL)]
    assert len(cleaned) == 1 and cleaned[0].startswith("ssr-")
    failure = json.loads((tmp_path / "run/failure.json").read_text())
    assert failure["remaining_seconds"] == 0


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
