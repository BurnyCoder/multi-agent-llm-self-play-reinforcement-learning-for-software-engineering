"""CPU-only checkpoint evidence; absence of optimization logs is not identity.

Hash saved bytes without loading tensors or executing checkpoint pickle payloads.
https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Stream the file so even a full checkpoint need not fit in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checkpoint(run: Path, relative: str) -> dict[str, Any]:
    """Record identity only for an existing checkpoint within the requested run."""
    directory = (run / relative).resolve()
    if not directory.is_relative_to(run.resolve()):
        raise ValueError("Checkpoint evidence path escapes the run directory")
    metadata, weights = directory / "checkpoint.json", directory / "adapter_model.safetensors"
    adapter_config = directory / "adapter_config.json"
    if not metadata.is_file() or not weights.is_file():
        return {}
    record = json.loads(metadata.read_text())
    if record.get("metadata", {}).get("purpose") == "preflight_only":
        return {}
    config = record.get("model_config", {})
    return {"path": relative, "sha256": file_sha256(weights),
            "adapter_config_sha256": file_sha256(adapter_config) if adapter_config.is_file() else None,
            "checkpoint_metadata_sha256": file_sha256(metadata),
            "base_model": config.get("model_id"), "base_revision": config.get("revision"),
            "policy_version": record.get("policy_version"),
            "adapter_fingerprint": record.get("adapter_fingerprint")}


def checkpoint_evidence(run: Path) -> dict[str, Any]:
    """Compare actual bytes, or retain exported hash observations with their basis.

    An export stores this derived record with its byte-hashed bundle inventory.
    The status is recomputed from the observations, never trusted on input.
    Equal serialized adapters, adapter settings and pinned bases support an
    unchanged-weight comparison; unequal bytes alone do not prove learning.
    """
    run = Path(run)
    latest_path = run / "latest.json"
    latest = json.loads(latest_path.read_text()) if latest_path.exists() else {}
    initial = _checkpoint(run, "checkpoints/initial")
    selected = _checkpoint(run, latest["path"]) if latest.get("path") else {}
    result = {"initial": initial, "selected": selected, "status": "unknown",
              "basis": "SHA256 of saved adapter weights/configuration and pinned base identities"}
    exported = run / "checkpoint-evidence.json"
    if not initial and not selected and exported.is_file():
        result = json.loads(exported.read_text())
        initial, selected = result.get("initial", {}), result.get("selected", {})
    result["status"] = "unknown"
    if all(re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) for row in (initial, selected)):
        same_base = (all(initial.get(key) and initial[key] == selected.get(key)
                         for key in ("base_model", "base_revision")))
        same_adapter_config = (bool(initial.get("adapter_config_sha256"))
                               and initial["adapter_config_sha256"] == selected.get("adapter_config_sha256"))
        if same_base and same_adapter_config:
            result["status"] = "unchanged" if initial["sha256"] == selected["sha256"] else "changed"
    return result


def optimizer_steps(metrics: list[dict[str, Any]]) -> int | None:
    """Missing or incomplete step measurements remain unknown, including empty logs."""
    values = [row.get("optimizer_steps") for row in metrics]
    return sum(values) if values and all(type(value) is int and value >= 0 for value in values) else None


def intervention_text(evidence: dict[str, Any], steps: int | None) -> str:
    """Separate the statistical rule from whether its planned intervention occurred."""
    recorded = f"Recorded research optimizer steps: {steps}. " if steps is not None else "Research optimizer-step evidence is missing or incomplete. "
    status = evidence.get("status", "unknown")
    if status == "unchanged":
        return (recorded + "The initial and selected adapter weights and configurations have identical SHA256 digests and the same pinned base identity. "
                "This is two evaluations of unchanged weights, not an estimated self-play training effect. "
                "The planned learning hypothesis was not tested by this unchanged-policy comparison. "
                "The raw trained label and mechanical preregistered interval decision are preserved for provenance.")
    if status == "changed":
        return (recorded + "The saved adapter bytes differ. " +
                ("This conflicts with the zero-step record; the cause is unverified. " if steps == 0 else "") +
                "Byte differences alone do not establish successful learning or improved repair. "
                "The raw trained label denotes the final condition.")
    return (recorded + "Initial-to-final checkpoint identity is unverified; missing records do not establish unchanged weights. "
            "The raw trained label denotes the final condition, not evidence of completed learning.")
