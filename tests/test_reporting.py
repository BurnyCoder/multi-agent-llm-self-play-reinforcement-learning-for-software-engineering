"""Analysis tests protect against invented coverage and selected duplicate outcomes."""

import json

import pytest

from multi_agent_ssr.reporting import analyze_evaluations, generate_report, load_jsonl


def record(task, condition, success, status="completed", source="human"):
    """Construct synthetic test records; these are never research measurements."""
    return {"task_id": task, "condition": condition, "source": source,
            "success": success, "status": status}


def test_missing_and_infra_results_do_not_become_failures():
    rows = [record("a", "base", False), record("a", "trained", True),
            record("b", "base", True), record("b", "trained", None, "infra_error"),
            record("c", "trained", None, "pending")]
    result = analyze_evaluations(rows, bootstrap_samples=100)
    assert result["conditions"]["human"]["trained"]["completed"] == 1
    assert result["conditions"]["human"]["trained"]["unscored"] == 2
    assert result["comparisons"]["human"]["paired_tasks"] == 1
    assert result["comparisons"]["human"]["difference"] == 1


def test_pairing_is_by_task_and_separates_bug_sources():
    rows = [record("a", "base", False), record("b", "base", True),
            record("b", "trained", False), record("a", "trained", True),
            record("a", "base", True, source="synthetic"),
            record("a", "trained", True, source="synthetic")]
    result = analyze_evaluations(rows, bootstrap_samples=100)
    assert result["comparisons"]["human"]["difference"] == 0
    assert result["comparisons"]["human"]["improved"] == 1
    assert result["comparisons"]["human"]["regressed"] == 1
    assert result["comparisons"]["synthetic"]["paired_tasks"] == 1


def test_bootstrap_reproducible_and_all_ties_not_proof_of_no_effect():
    rows = [record(str(i), c, False) for i in range(4) for c in ("base", "trained")]
    first = analyze_evaluations(rows, bootstrap_samples=200)
    assert first == analyze_evaluations(list(reversed(rows)), bootstrap_samples=200)
    paired = first["comparisons"]["human"]
    assert paired["ci95"] == [0, 0]
    assert paired["interval_degenerate"]
    assert paired["interpretation"] == "no_observed_difference"
    assert paired["preregistered_decision"] == "contradicts_predicted_improvement"


def test_retry_and_duplicates_cannot_select_a_better_score():
    rows = [record("a", "base", None, "infra_error"), record("a", "base", False)]
    assert analyze_evaluations(rows)["conditions"]["human"]["base"]["completed"] == 1
    with pytest.raises(ValueError, match="Conflicting completed"):
        analyze_evaluations(rows + [record("a", "base", True)])


def test_malformed_result_is_not_silently_ignored(tmp_path):
    path = tmp_path / "evaluation.jsonl"
    path.write_text('{"task_id":"a"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation.jsonl:2"):
        load_jsonl(path)


def test_empty_report_preserves_unknowns_and_cites_sources(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"run_id": "test-fixture", "seed": 0}))
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    outputs = generate_report(run, tmp_path / "report", tmp_path / "site")
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["analysis"]["comparisons"]["human"]["difference"] is None
    assert "Not evaluated" in (tmp_path / "report" / "experiment.md").read_text()
    assert "https://arxiv.org/html/2512.18552v3" in (tmp_path / "report" / "paper.md").read_text()
    assert "Codex wrote" not in (tmp_path / "report" / "experiment.md").read_text()
    assert "Codex wrote" in (tmp_path / "report" / "paper.md").read_text()
    assert "index.html" in outputs
    assert (tmp_path / "site" / "data.json").exists()


def test_report_collects_saved_patches_and_nested_traces_without_grader_secrets(tmp_path, monkeypatch):
    run = tmp_path / "run"
    public = run / "human" / "base" / "fixture-1"
    public.mkdir(parents=True)
    (public / "prediction.diff").write_text("fixture prediction")
    private = run / "official" / "grader"
    private.mkdir(parents=True)
    (private / "prediction.diff").write_text("SECRET_ORACLE")
    nested = run / "synthetic" / "base" / "fixture-2" / "attempt-1" / "agents"
    nested.mkdir(parents=True)
    event = {"event": "session_started", "session_id": "child", "episode_id": "fixture", "timestamp": "2020-01-01"}
    (nested / "events.jsonl").write_text(json.dumps(event) + "\n")
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    assert [patch["text"] for patch in data["patches"]] == ["fixture prediction"]
    assert data["events"][0]["session_id"] == "child"
    assert "SECRET_ORACLE" not in json.dumps(data)
