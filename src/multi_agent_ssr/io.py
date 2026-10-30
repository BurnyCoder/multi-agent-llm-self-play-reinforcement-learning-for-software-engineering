"""Append-only experiment records and atomic JSON metadata, shared by phases."""

from __future__ import annotations

import json
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
            print(line, flush=True)
        return event
