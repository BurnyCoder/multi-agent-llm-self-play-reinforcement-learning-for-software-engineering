"""Enforce the persisted GPU budget outside the model's Python process.

Cooperative deadlines cannot interrupt a stuck CUDA call. A separate parent
process kills its own worker group at the deadline and cleans only containers
carrying this run's unique label. No unrelated processes or containers are used.
https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
"""
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from .experiment import Budget, load_config
from .io import utc_now, write_json


def cleanup_containers(run_id: str) -> None:
    """Remove only disposable containers explicitly labeled by this run."""
    found = subprocess.run(["docker", "ps", "-aq", "--filter",
                            f"label=org.multi-agent-ssr.run={run_id}"],
                           capture_output=True, text=True, timeout=15, check=False)
    identifiers = found.stdout.split()
    if found.returncode == 0 and identifiers:
        subprocess.run(["docker", "rm", "--force", *identifiers],
                       capture_output=True, timeout=30, check=False)


def bounded_phase(phase: str, config_path: Path, run_dir: Path) -> int:
    """Keep one worker per run, sharing the same deadline across CLI invocations."""
    if os.name != "posix":
        raise RuntimeError("Run GPU phases inside WSL2/Linux")
    import fcntl

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = load_config(config_path)["experiment"]
        budget = Budget(run_dir, settings["budget_seconds"])
        if budget.remaining() <= 0:
            return 124
        label = "ssr-" + uuid.uuid4().hex
        environment = {**os.environ, "SSR_RUN_ID": label, "PYTHONUNBUFFERED": "1"}
        worker = subprocess.Popen([sys.executable, "-m", "multi_agent_ssr.cli", phase,
            "--config", str(config_path), "--run", str(run_dir), "--worker"],
            env=environment, start_new_session=True)
        try:
            return worker.wait(timeout=budget.remaining())
        except subprocess.TimeoutExpired:
            # SIGKILL ends the CUDA worker even when Python cannot handle signals.
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=15)
            write_json(run_dir / "failure.json", {"phase": phase,
                "error": "Hard persisted budget deadline reached; incomplete work was not admitted",
                "timestamp": utc_now(), "remaining_seconds": 0})
            return 124
        except BaseException:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=15)
            raise
        finally:
            cleanup_containers(label)
