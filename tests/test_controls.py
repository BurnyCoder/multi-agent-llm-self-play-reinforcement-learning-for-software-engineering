"""Control conditions must never draw mutation training tasks from held-out data."""
import json

import pytest

from multi_agent_ssr.contracts import RepoSpec


def spec(split="train"):
    return RepoSpec("example", "https://example.test/repo", "a" * 40, "image", ".",
                    split, ["package"], ["tests"], "MIT", "sha256:example")


def test_control_rejects_held_out_repository_before_touching_data():
    from multi_agent_ssr.controls import require_training_repository
    for split in ["dev", "test"]:
        with pytest.raises(ValueError, match="training"):
            require_training_repository(spec(split))


def test_frozen_corpus_requires_exact_training_revision(tmp_path):
    from multi_agent_ssr.controls import frozen_artifacts
    root = tmp_path / "artifact_records"
    root.mkdir()
    record = {"repository_commit": "b" * 40, "artifact": {"repo_name": "example"}}
    (root / "task.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="revision"):
        frozen_artifacts(tmp_path, spec())


def test_control_never_falls_back_to_adaptive_injection(tmp_path):
    from multi_agent_ssr.controls import frozen_artifacts
    with pytest.raises(ValueError, match="No frozen"):
        frozen_artifacts(tmp_path, spec())
