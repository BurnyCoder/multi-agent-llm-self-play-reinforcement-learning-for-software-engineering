"""A software preflight must never be mislabeled as the research checkpoint."""
import json

import pytest


def test_export_refuses_preflight_adapter(tmp_path):
    from multi_agent_ssr.publication import export_checkpoint
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "latest.json").write_text(json.dumps({"path": "checkpoint", "policy_version": 1}))
    (checkpoint / "checkpoint.json").write_text(json.dumps({"metadata": {"purpose": "preflight_only"}}))
    with pytest.raises(ValueError, match="preflight-only"):
        export_checkpoint(tmp_path, tmp_path, tmp_path / "export")
