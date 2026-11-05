"""Append-only experiment records and atomic JSON metadata, shared by phases."""

from __future__ import annotations

import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """UTC timestamps remain comparable across WSL, Windows and publication."""
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: Any) -> None:
    """Replace metadata atomically so interrupted runs never leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def mirror_terminal(line: str) -> None:
    """Mirror a saved record without making terminal availability a run dependency.

    Python raises BrokenPipeError when a pipe reader disconnects; the durable
    file has already received this record. Keep disk errors fatal and do not
    change SIGPIPE globally, which would affect unrelated library operations.
    https://docs.python.org/3.12/library/signal.html#note-on-sigpipe
    """
    try:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.buffer.write((line + "\n").encode("utf-8"))
            sys.stdout.buffer.flush()
    except BrokenPipeError:
        # A live consumer can reconnect later; subsequent records retry mirroring.
        pass


class EventLog:
    """Serialize concurrent root/child events without truncating their content."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def emit(self, kind: str, **values: Any) -> dict[str, Any]:
        """Durably append an event before showing the identical record live."""
        event = {"timestamp": utc_now(), "event": kind, **values}
        line = json.dumps(event, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
            mirror_terminal(line)
        return event
