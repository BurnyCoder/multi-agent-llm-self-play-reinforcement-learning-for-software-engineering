"""Check isolation flags, safe copying and parallel Docker execution contracts."""

import io
import subprocess
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from multi_agent_ssr.sandbox import (
    DockerSandbox,
    SandboxInfrastructureError,
    _extract_archive,
    _source_archive,
)


def test_source_archive_never_contains_git_or_local_credentials(tmp_path):
    (tmp_path / "code.py").write_text("value = 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "original").write_text("correct solution")
    (tmp_path / ".env").write_text("TOKEN=secret")
    data = _source_archive(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        names = archive.getnames()
    assert "code.py" in names
    assert all(".git" not in name and ".env" not in name for name in names)


def test_snapshot_rejects_traversal_and_links(tmp_path):
    for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            item = tarfile.TarInfo(name)
            item.type = kind
            item.linkname = "../../outside"
            archive.addfile(item)
        with pytest.raises(ValueError):
            _extract_archive(stream.getvalue(), tmp_path / "snapshot")
    assert not (tmp_path / "escape").exists()


def test_container_is_unprivileged_and_has_no_network_or_host_mounts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print(1)\n")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, b"container-id\n", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with DockerSandbox("python:3.12-slim", source, tmp_path / "work"):
        pass
    run = next(args for args in calls if "run" in args)
    assert run[run.index("--network") + 1] == "none"
    assert run[run.index("--user") + 1] == "65534:65534"
    assert "--read-only" in run and "--cap-drop" in run
    assert "--mount" not in run and "-v" not in run and "--volume" not in run
    assert any("rm" in args and "container-id" in args for args in calls)


def test_exec_does_not_serialize_independent_agent_commands(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    barrier = threading.Barrier(2)

    def fake_run(args, **kwargs):
        if "timeout" in args:
            barrier.wait(timeout=1)
        return subprocess.CompletedProcess(args, 0, b"container-id\n", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with (DockerSandbox("fixture", source, tmp_path / "work") as sandbox,
          ThreadPoolExecutor(max_workers=2) as executor):
        futures = [executor.submit(sandbox.exec, "printf hello") for _ in range(2)]
        for future in futures:
            assert future.result(timeout=2)["returncode"] == 0


def test_readonly_and_path_checks(tmp_path):
    sandbox = DockerSandbox("fixture", tmp_path, tmp_path / "work", readonly=True)
    with pytest.raises(PermissionError):
        sandbox.write("a.py", "bad")
    for path in ["../escape", "/etc/passwd", "a/../../escape", "C:/outside"]:
        with pytest.raises(ValueError):
            sandbox.read(path)


def test_daemon_failure_is_distinct_from_a_failed_command(tmp_path, monkeypatch):
    sandbox = DockerSandbox("fixture", tmp_path, tmp_path / "work")
    sandbox.container_id = "fixture-container"
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(
        args, 1, b"", b"Error response from daemon: container is not running"))
    with pytest.raises(SandboxInfrastructureError):
        sandbox.exec("true")
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(
        args, 1, b"", b"assertion failed"))
    assert sandbox.exec("false")["returncode"] == 1
