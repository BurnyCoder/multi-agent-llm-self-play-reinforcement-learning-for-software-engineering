"""A software preflight must never be mislabeled as the research checkpoint."""
import json
from dataclasses import asdict
from hashlib import sha256

import pytest

from multi_agent_ssr.contracts import ModelConfig
from multi_agent_ssr.io import write_json
from multi_agent_ssr.publication import export_checkpoint
from multi_agent_ssr.reporting import generate_report


def _fixture(tmp_path, version=0):
    """Build software-only evidence; no fixture value is a research result."""
    run, report = tmp_path / "run", tmp_path / "report"
    checkpoint = run / "checkpoints" / "selected"
    checkpoint.mkdir(parents=True)
    report.mkdir()
    write_json(run / "latest.json", {"path": "checkpoints/selected", "policy_version": version,
                                    "completed_groups": int(version > 0)})
    write_json(checkpoint / "checkpoint.json", {
        "format_version": 1, "model_config": asdict(ModelConfig(revision="fixture-base-revision")),
        "policy_version": version, "adapter_fingerprint": "fixture-fingerprint",
        "metadata": {"completed_groups": int(version > 0)}})
    (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture adapter, not model weights")
    write_json(run / "manifest.json", {"code_commit": "fixture-original",
        "phase_code_commits": {"pilot": "fixture-pilot"}, "expected_human_tasks": 1,
        "config": {"experiment": {"bootstrap_samples": 20}}})
    write_json(report / "summary.json", {"status": "completed", "analysis": {"fixture": True}})
    return run, report


def test_export_refuses_preflight_adapter(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (tmp_path / "latest.json").write_text(json.dumps({"path": "checkpoint", "policy_version": 1}))
    (checkpoint / "checkpoint.json").write_text(json.dumps({"metadata": {"purpose": "preflight_only"}}))
    with pytest.raises(ValueError, match="preflight-only"):
        export_checkpoint(tmp_path, tmp_path, tmp_path / "export")


@pytest.mark.parametrize("version", [0, 2])
def test_card_distinguishes_unchanged_selection_and_reloads_exact_checkpoint(tmp_path, monkeypatch, version):
    run, report = _fixture(tmp_path, version)
    bundle = export_checkpoint(run, report, tmp_path / "export")
    card = (bundle / "README.md").read_text()
    assert "fixture-base-revision" in card and "fixture-fingerprint" in card
    assert "fixture-pilot" in card and "completed" in card
    assert "tokenizer's chat EOS" in card
    if version:
        assert "final completed collection group" in card
        assert "not a trained checkpoint" not in card
    else:
        assert "not a trained checkpoint" in card
        assert "The selected adapter has no optimizer updates" in card
        assert "`trained` identifies the final-evaluation condition" in card

    loaded = []
    monkeypatch.setattr("multi_agent_ssr.model.LocalPolicy", lambda config, **kwargs: loaded.append((config, kwargs)))
    snippet = card.split("```python\n", 1)[1].split("```", 1)[0]
    snippet = snippet.replace('Path("/path/to/downloaded-bundle")', repr(bundle))
    # Resolve the concrete Path repr without importing a GPU backend or sampling.
    exec(snippet, {type(bundle).__name__: type(bundle)})
    assert loaded[0][0].revision == "fixture-base-revision"
    assert loaded[0][1]["checkpoint"] == bundle


def test_unchanged_adapter_can_be_selected_after_completed_zero_advantage_groups(tmp_path):
    run, report = _fixture(tmp_path)
    latest = json.loads((run / "latest.json").read_text())
    latest["completed_groups"] = 3
    write_json(run / "latest.json", latest)
    checkpoint = run / latest["path"] / "checkpoint.json"
    record = json.loads(checkpoint.read_text())
    record["metadata"]["completed_groups"] = 3
    write_json(checkpoint, record)
    card = (export_checkpoint(run, report, tmp_path / "export") / "README.md").read_text()
    assert "The selected adapter has no optimizer updates" in card
    assert "Recorded completed collection groups: `3`" in card
    assert "No completed training group was selected" not in card
    assert "`trained` identifies the final-evaluation condition" in card


def test_exported_records_regenerate_report_and_retain_exact_action_tokens(tmp_path, monkeypatch):
    run, report = _fixture(tmp_path)
    notices = tmp_path / "notices"
    notice = notices / "files" / "fixture-hash" / "NOTICE"
    notice.parent.mkdir(parents=True)
    notice.write_bytes(b"Exact upstream notice\r\n")
    (notices / "README.md").write_text("Fixture notice guide")
    write_json(notices / "provenance.json", {"files": [{"path": "files/fixture-hash/NOTICE"}]})
    (notices / "unrelated.json").write_text("PRIVATE_FIXTURE_MUST_NOT_EXPORT")
    monkeypatch.setattr("multi_agent_ssr.publication._NOTICE_DIR", notices, raising=False)
    generation = {"prompt_ids": [101, 102], "completion_ids": [7, 8], "old_logprobs": [-0.125, -1.5],
                  "text": "fixture action", "policy_version": 0}
    events = [{"event": "phase_started", "phase": "pilot", "timestamp": "2020-01-01"},
              {"event": "generation", "call_id": "fixture-call", "generation": generation}]
    (run / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    (run / "model.jsonl").write_text(json.dumps({"event": "model_loaded", "parameter_dtype_counts": {"fixture": 3}}) + "\n")
    (run / "evaluation.jsonl").write_text(json.dumps({"task_id": "fixture-task", "source": "human",
        "condition": "base", "status": "completed", "success": False}) + "\n")
    write_json(run / "preparation.json", {"evidence_type": "software fixture"})
    write_json(run / "protocol_diagnostics.json", [])
    write_json(run / "kernel-review.json", {"fixture": "preserved"})
    write_json(run / "preflight" / "likelihood.json", {})
    write_json(run / "episodes" / "fixture.json", {"actions": [{"generation": generation}],
                                                  "result": {"repair_patch": "training fixture patch"}})
    write_json(run / "collection_groups" / "group-00001.json", {"completed": True, "optimized": False})
    nested = run / "synthetic" / "base" / "fixture" / "attempt-1"
    (nested / "agents").mkdir(parents=True)
    (nested / "agents" / "events.jsonl").write_text(json.dumps({"event": "session_started", "session_id": "fixture-child"}) + "\n")
    (nested / "repair.diff").write_text("synthetic fixture patch")
    human = run / "human" / "base" / "fixture"
    human.mkdir(parents=True)
    (human / "prediction.diff").write_text("human fixture patch")
    write_json(human / "episode.json", {"actions": [{"generation": generation}]})
    for relative in ("official/grader/report.json", "human/base/fixture/prediction/private.json",
                     "preflight/checkpoint/checkpoint.json", "unrelated.json", ".env"):
        private = run / relative
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_text("PRIVATE_FIXTURE_MUST_NOT_EXPORT")
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, report, tmp_path / "site")
    original = json.loads((report / "summary.json").read_text())
    bundle = export_checkpoint(run, report, tmp_path / "export")
    copied = bundle / "run-records" / run.name
    assert (copied / "events.jsonl").read_bytes() == (run / "events.jsonl").read_bytes()
    assert json.loads((copied / "episodes" / "fixture.json").read_text())["actions"][0]["generation"] == generation
    assert (copied / "collection_groups" / "group-00001.json").is_file()
    assert (copied / "human" / "base" / "fixture" / "episode.json").is_file()
    assert not any(b"PRIVATE_FIXTURE_MUST_NOT_EXPORT" in path.read_bytes() for path in copied.rglob("*") if path.is_file())
    generate_report(copied, tmp_path / "regenerated", tmp_path / "regenerated-site")
    regenerated = json.loads((tmp_path / "regenerated" / "summary.json").read_text())
    for data in (original, regenerated):
        data.pop("generated_at")
    assert regenerated == original
    inventory = json.loads((bundle / "bundle-inventory.json").read_text())
    for entry in inventory["files"]:
        path = bundle / entry["path"]
        assert sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert path.stat().st_size == entry["bytes"]
    assert f"run-records/{run.name}/events.jsonl" in {entry["path"] for entry in inventory["files"]}
    for destination in (report / "licenses", tmp_path / "site" / "licenses", bundle / "docs" / "licenses"):
        assert (destination / "files" / "fixture-hash" / "NOTICE").read_bytes() == notice.read_bytes()
        assert (destination / "provenance.json").read_bytes() == (notices / "provenance.json").read_bytes()
        assert not (destination / "unrelated.json").exists()
    assert "docs/licenses/files/fixture-hash/NOTICE" in {entry["path"] for entry in inventory["files"]}
    assert "selfplay report --run" in (bundle / "README.md").read_text()


def test_allowlisted_record_symlink_cannot_export_private_file(tmp_path):
    run, report = _fixture(tmp_path)
    private = run / "official" / "private.json"
    private.parent.mkdir()
    private.write_text("PRIVATE_FIXTURE")
    (run / "preparation.json").symlink_to(private)
    with pytest.raises(ValueError, match="symlink"):
        export_checkpoint(run, report, tmp_path / "export")
