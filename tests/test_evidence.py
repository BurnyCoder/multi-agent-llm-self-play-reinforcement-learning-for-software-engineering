"""Reporting must distinguish measured absence, missing evidence, and identity."""

import json

import pytest

from multi_agent_ssr.io import write_json
from multi_agent_ssr.reporting import _diagnostics, generate_report


def checkpoints(run, *, final_bytes=b"initial", revision="pinned-base"):
    """Use inert bytes to exercise identity checks without loading any model."""
    for name, weights in (("initial", b"initial"), ("final", final_bytes)):
        directory = run / "checkpoints" / name
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "checkpoint.json", {
            "model_config": {"model_id": "fixture/model", "revision": revision},
            "policy_version": 0,
            "metadata": {"purpose": "frozen_base" if name == "initial" else "research"},
        })
        (directory / "adapter_model.safetensors").write_bytes(weights)
        write_json(directory / "adapter_config.json", {"r": 8, "lora_alpha": 16})
    write_json(run / "latest.json", {"path": "checkpoints/final", "policy_version": 0})


def report_fixture(tmp_path, monkeypatch, *, steps=None, identity=None):
    """Create a complete paired observation with independently optional training evidence."""
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "manifest.json", {"config": {"experiment": {"bootstrap_samples": 20}}})
    rows = [{"task_id": "fixture", "source": "human", "condition": condition,
             "status": "completed", "success": False} for condition in ("base", "trained")]
    (run / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if steps is not None:
        (run / "metrics.jsonl").write_text(json.dumps({"event": "training_group", "optimizer_steps": steps}) + "\n")
    if identity is not None:
        checkpoints(run, final_bytes=b"initial" if identity else b"different")
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    readme = tmp_path / "README.md"
    readme.write_text("<!-- SSR_RESULTS:START -->\n<!-- SSR_RESULTS:END -->\n")
    generate_report(run, tmp_path / "report", tmp_path / "site", readme_path=readme)
    return json.loads((tmp_path / "report/summary.json").read_text())


def test_checks_are_not_valid_artifacts():
    events = [{"event": "artifact_validated", "valid": result} for result in (False, True, False)]
    result = _diagnostics(events, [], {}, [])
    assert result["artifact_validation_checks"] == result["validated_artifacts"] == 3
    assert result["valid_artifacts"] == 1
    assert result["invalid_artifacts"] == 2
    assert result["optimizer_steps"] is None


@pytest.mark.parametrize("steps", [None, 0])
def test_absent_checkpoint_evidence_does_not_assert_unchanged_weights(tmp_path, monkeypatch, steps):
    payload = report_fixture(tmp_path, monkeypatch, steps=steps)
    assert payload["diagnostics"]["optimizer_steps"] == steps
    assert payload["checkpoint_evidence"]["status"] == "unknown"
    text = payload["training_intervention_disclosure"]
    assert "identity is unverified" in text
    assert "two evaluations of unchanged weights" not in text
    for path in (tmp_path / "report/experiments-summary.md", tmp_path / "report/paper.md"):
        assert "Recorded local RL updates" not in path.read_text()


def test_verified_unchanged_weights_leave_learning_hypothesis_untested(tmp_path, monkeypatch):
    payload = report_fixture(tmp_path, monkeypatch, steps=0, identity=True)
    assert payload["checkpoint_evidence"]["status"] == "unchanged"
    assert "learning hypothesis was not tested" in payload["training_intervention_disclosure"]
    assert payload["analysis"]["comparisons"]["human"]["preregistered_decision"] == "contradicts_predicted_improvement"
    assert "Recorded local RL updates" not in (tmp_path / "report/experiments-summary.md").read_text()
    assert "final condition" in (tmp_path / "README.md").read_text()


def test_zero_steps_cannot_override_different_checkpoint_bytes(tmp_path, monkeypatch):
    payload = report_fixture(tmp_path, monkeypatch, steps=0, identity=False)
    assert payload["checkpoint_evidence"]["status"] == "changed"
    assert "unchanged weights" not in payload["training_intervention_disclosure"]
    assert "conflict" in payload["training_intervention_disclosure"].lower()


def test_equal_bytes_with_different_base_revision_do_not_prove_same_policy(tmp_path, monkeypatch):
    from multi_agent_ssr.evidence import checkpoint_evidence

    checkpoints(tmp_path)
    path = tmp_path / "checkpoints/final/checkpoint.json"
    value = json.loads(path.read_text())
    value["model_config"]["revision"] = "different-base"
    write_json(path, value)
    assert checkpoint_evidence(tmp_path)["status"] == "unknown"


def test_exported_identity_evidence_is_recomputed_and_survives_bundle_regeneration(tmp_path):
    from multi_agent_ssr.evidence import checkpoint_evidence

    checkpoints(tmp_path)
    evidence = checkpoint_evidence(tmp_path)
    exported = tmp_path / "exported-records"
    exported.mkdir()
    write_json(exported / "checkpoint-evidence.json", evidence)
    assert checkpoint_evidence(exported) == evidence
    evidence["selected"]["sha256"] = "0" * 64
    evidence["status"] = "unchanged"
    write_json(exported / "checkpoint-evidence.json", evidence)
    assert checkpoint_evidence(exported)["status"] == "changed"


def test_empty_patch_test_failures_are_not_attributed_to_agent_edits():
    from multi_agent_ssr.reporting import _oracle_diagnostics, _oracle_text

    row = {"source": "human", "condition": "base", "task_id": "fixture",
           "status": "completed", "success": False, "official_report": {
               "tests_status": {"PASS_TO_PASS": {"failure": ["test_a", "test_b"]}}}}
    patches = [{"path": "human/base/fixture/prediction.diff", "text": ""}]
    result = _oracle_diagnostics([row], patches)
    assert result["pass_to_pass_test_failures"] == 2
    assert result["empty_patch_rows_with_failures"] == 1
    assert "not established regressions caused by agent edits" in _oracle_text(result)


def test_adapter_configuration_changes_prevent_unchanged_policy_claim(tmp_path):
    from multi_agent_ssr.evidence import checkpoint_evidence

    checkpoints(tmp_path)
    write_json(tmp_path / "checkpoints/final/adapter_config.json", {"r": 8, "lora_alpha": 32})
    assert checkpoint_evidence(tmp_path)["status"] == "unknown"


def test_distinct_completed_attempts_cannot_be_silently_selected_even_if_scores_match():
    from multi_agent_ssr.reporting import analyze_evaluations

    row = {"source": "human", "condition": "base", "task_id": "fixture",
           "status": "completed", "success": False, "generated_tokens": 10}
    with pytest.raises(ValueError, match="completed outcomes"):
        analyze_evaluations([row, {**row, "generated_tokens": 20}])
