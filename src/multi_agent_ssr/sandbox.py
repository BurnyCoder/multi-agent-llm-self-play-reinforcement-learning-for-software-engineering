"""Disposable Docker workspaces; model-generated commands never run on the host.

Docker resource and privilege controls follow https://docs.docker.com/engine/containers/run/.
Source transfer deliberately uses tar streams instead of host mounts. This also
removes original Git history, as required by https://arxiv.org/html/2512.18552v3.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Self

_EXCLUDED = {".git", ".env", ".venv", "venv", "__pycache__", ".pytest_cache",
             ".mypy_cache", ".ruff_cache", ".ssh", ".aws", ".azure"}
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


class SandboxInfrastructureError(RuntimeError):
    """The container service failed; this must never become a policy reward."""


def _relative_path(path: str) -> str:
    """Allow repository-relative paths only, including on Windows hosts."""
    value = PurePosixPath(path.replace("\\", "/"))
    if not path or value.is_absolute() or ".." in value.parts or ":" in path or "\x00" in path:
        raise ValueError(f"Unsafe repository path: {path!r}")
    if any(part in _EXCLUDED for part in value.parts):
        raise ValueError(f"Protected repository path: {path!r}")
    return value.as_posix()


def _source_archive(source: Path) -> bytes:
    """Copy ordinary repository files without following links or copying secrets."""
    source = Path(source).resolve(strict=True)
    stream = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for directory, dirs, files in os.walk(source, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED)
            for name in [*dirs, *sorted(files)]:
                if name in _EXCLUDED:
                    continue
                path = Path(directory) / name
                if path.is_symlink():
                    raise ValueError(f"Source links are not allowed: {path}")
                if not path.is_file():
                    continue
                total += path.stat().st_size
                if total > _MAX_ARCHIVE_BYTES:
                    raise ValueError("Repository exceeds the sandbox copy limit")
                info = archive.gettarinfo(str(path), arcname=path.relative_to(source).as_posix())
                info.uid = info.gid = 65534
                info.uname = info.gname = ""
                info.mode = 0o755 if info.mode & 0o111 else 0o644
                with path.open("rb") as file:
                    archive.addfile(info, file)
    return stream.getvalue()


def _extract_archive(data: bytes, destination: Path) -> None:
    """Extract only regular files/directories inside an explicitly supplied root."""
    if len(data) > _MAX_ARCHIVE_BYTES:
        raise ValueError("Sandbox snapshot exceeds the copy limit")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
        total = 0
        for member in members:
            name = member.name.removeprefix("./")
            if name in {"", "."} and member.isdir():
                continue
            relative = _relative_path(name)
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Sandbox snapshot contains a link or special file: {name}")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Sandbox snapshot escapes destination: {name}")
            total += member.size
            if total > _MAX_ARCHIVE_BYTES:
                raise ValueError("Sandbox snapshot exceeds the expanded copy limit")
        for member in members:
            name = member.name.removeprefix("./")
            if name in {"", "."}:
                continue
            target = destination / _relative_path(name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(member.mode & 0o777)


_FILE_IO = """
import os, pathlib, sys, tempfile
root = pathlib.Path('/repo').resolve()
path = root.joinpath(sys.argv[2]).resolve()
if not path.is_relative_to(root):
    raise ValueError('Path escapes workspace')
if sys.argv[1] == 'read':
    sys.stdout.buffer.write(path.read_bytes())
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.ssr-write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as file:
            file.write(sys.stdin.buffer.read())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
"""


class DockerSandbox:
    """One isolated filesystem shared by concurrent root/child tool calls.

    Python file writes are atomic and locked with snapshots. Independent exec
    calls intentionally do not take that lock, so delegated work overlaps.
    Role-specific command permissions belong to the harness above this layer.
    """

    workspace = "/repo"

    def __init__(self, image: str, source: Path, work_root: Path, readonly: bool = False):
        self.image = image
        self.source = Path(source)
        self.work_root = Path(work_root)
        self.readonly = readonly
        self.container_id: str | None = None
        self._copy_lock = threading.RLock()
        self._docker = ["docker"] if shutil.which("docker") or os.name != "nt" else ["wsl", "-e", "docker"]

    def _run(self, arguments: list[str], *, data: bytes | None = None,
             timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run([*self._docker, *arguments], input=data,
                                    capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SandboxInfrastructureError(f"Docker invocation failed: {error}") from error
        error = result.stderr.decode("utf-8", errors="replace").strip()
        daemon_error = error.startswith(("Error response from daemon:", "Cannot connect to the Docker daemon",
                                         "error during connect:", "OCI runtime exec failed:", "docker: Error"))
        if result.returncode and (check or daemon_error):
            raise SandboxInfrastructureError(error or f"Docker exited {result.returncode}")
        return result

    def __enter__(self) -> Self:
        """Start from a dependency-only image and copy no original Git metadata."""
        if self.container_id is not None:
            raise RuntimeError("Sandbox is already running")
        archive = _source_archive(self.source)
        self.work_root.mkdir(parents=True, exist_ok=True)
        result = self._run([
            "run", "--detach", "--rm", "--name", f"ssr-{uuid.uuid4().hex}",
            "--network", "none", "--user", "65534:65534", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "128", "--memory", "768m", "--memory-swap", "768m",
            "--cpus", "1", "--ulimit", "nofile=256:256",
            "--tmpfs", "/repo:rw,nosuid,nodev,size=256m,mode=1777",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m,mode=1777",
            "--workdir", "/repo", "--env", "HOME=/tmp",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONUNBUFFERED=1",
            "--env", "PYTHONPATH=/repo:/repo/src",
            "--entrypoint", "sleep", self.image, "infinity",
        ])
        self.container_id = result.stdout.decode().strip()
        try:
            copy_user = ["--user", "0:0"] if self.readonly else []
            self._run(["exec", "--interactive", *copy_user, self.container_id, "tar", "--no-same-owner",
                       "-xf", "-", "-C", "/repo"], data=archive)
            if self.readonly:
                # Read-only files were created as root during initialization;
                # no agent command runs as root or can restore write permission.
                self._run(["exec", "--user", "0:0", self.container_id,
                           "chmod", "-R", "a-w", "/repo"])
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args) -> None:
        """Removing the container also terminates any still-running child process."""
        container_id, self.container_id = self.container_id, None
        if container_id:
            self._run(["rm", "--force", container_id], timeout=30, check=False)

    def _id(self) -> str:
        if self.container_id is None:
            raise RuntimeError("Sandbox is not running")
        return self.container_id

    def exec(self, command: str, timeout: int = 90) -> dict:
        """Run a bounded shell command; GNU timeout kills its process group."""
        if not isinstance(timeout, int) or not 1 <= timeout <= 900:
            raise ValueError("Command timeout must be between 1 and 900 seconds")
        result = self._run(["exec", self._id(), "timeout", "--signal=TERM", "--kill-after=2",
                            str(timeout), "sh", "-lc", command],
                           timeout=timeout + 15, check=False)
        return {"stdout": result.stdout.decode("utf-8", errors="replace"),
                "stderr": result.stderr.decode("utf-8", errors="replace"),
                "returncode": result.returncode}

    def read(self, path: str) -> str:
        """Read a relative file after resolving links inside the container."""
        relative = _relative_path(path)
        result = self._run(["exec", self._id(), "python", "-c", _FILE_IO, "read", relative], check=False)
        if result.returncode:
            raise ValueError(result.stderr.decode("utf-8", errors="replace"))
        return result.stdout.decode("utf-8")

    def write(self, path: str, content: str) -> None:
        """Atomically replace a relative file, preventing partly written reads."""
        if self.readonly:
            raise PermissionError("This sandbox is read-only")
        relative = _relative_path(path)
        with self._copy_lock:
            result = self._run(["exec", "--interactive", self._id(), "python", "-c", _FILE_IO,
                                "write", relative], data=content.encode("utf-8"), check=False)
            if result.returncode:
                raise ValueError(result.stderr.decode("utf-8", errors="replace"))

    def snapshot(self, destination: Path) -> None:
        """Save source state safely; callers must join shell editors beforehand."""
        with self._copy_lock:
            excludes = [f"--exclude={name}" for name in sorted(_EXCLUDED)]
            result = self._run(["exec", self._id(), "tar", *excludes, "-cf", "-",
                               "-C", "/repo", "."])
            _extract_archive(result.stdout, Path(destination))
