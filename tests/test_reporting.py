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


@pytest.mark.parametrize("optimizer_steps, final_scored", [(0, True), (2, True), (0, False)])
def test_zero_update_final_condition_is_not_presented_as_a_training_effect(tmp_path, monkeypatch,
                                                                          optimizer_steps, final_scored):
    """The interval rule is unchanged even when the learning intervention is absent."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"config": {"experiment": {"bootstrap_samples": 20}}}))
    (run / "metrics.jsonl").write_text(json.dumps({"optimizer_steps": optimizer_steps}) + "\n")
    rows = [record("fixture", "base", False)]
    if final_scored:
        rows.append(record("fixture", "trained", True))
    (run / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    readme = tmp_path / "README.md"
    readme.write_text("Fixture README\n<!-- SSR_RESULTS:START -->\n<!-- SSR_RESULTS:END -->\n")
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site", readme_path=readme)
    payload = json.loads((tmp_path / "site" / "data.json").read_text())
    assert payload["analysis"]["comparisons"] == analyze_evaluations(rows, bootstrap_samples=20)["comparisons"]
    disclosure = payload["training_intervention_disclosure"]
    if optimizer_steps == 0 and final_scored:
        assert "two evaluations of unchanged weights" in disclosure
        assert "not an estimated self-play training effect" in disclosure
        assert payload["analysis"]["comparisons"]["human"]["preregistered_decision"] == "supports_predicted_improvement"
        for path in [readme, *(tmp_path / "report" / name for name in
                              ("experiment.md", "experiments-summary.md", "paper.md"))]:
            assert disclosure in path.read_text()
    else:
        assert disclosure == ""


@pytest.mark.parametrize("case", ["equal", "varied", "mixed"])
def test_training_outcome_explains_zero_advantages_without_projecting_future_groups(tmp_path, monkeypatch, case):
    """Completed zero-signal groups remain distinct from later effective updates."""
    run = tmp_path / "run"
    run.mkdir()
    equal = {"event": "training_group", "step": 1, "valid_bugs": 0, "injector_episodes": 2,
             "repair_attempts": 0, "episodes": 2, "zero_advantage_episodes": 2, "optimizer_steps": 0,
             "no_action_groups": 0, "sampled_tokens": 8, "trained_tokens_per_pass": 0}
    varied = {**equal, "step": 2, "valid_bugs": 1, "repair_attempts": 4, "episodes": 6,
              "zero_advantage_episodes": 0, "optimizer_steps": 2, "trained_tokens_per_pass": 8}
    metrics = [equal] if case == "equal" else [varied] if case == "varied" else [equal, varied]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    event = {"event": "artifact_validated", "valid": False, "reason": "fixture: no test targets"}
    (run / "events.jsonl").write_text((json.dumps(event) + "\n") * 3)
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    monkeypatch.setattr("multi_agent_ssr.reporting.copy_source_notices", lambda *args: [])
    readme = tmp_path / "README.md"
    readme.write_text("Fixture README\n<!-- SSR_RESULTS:START -->\n<!-- SSR_RESULTS:END -->\n")
    generate_report(run, tmp_path / "report", tmp_path / "site", readme_path=readme)
    payload = json.loads((tmp_path / "site" / "data.json").read_text())
    text = payload["training_outcome_disclosure"]
    assert f"Completed collection groups: {len(metrics)}" in text
    assert '"fixture: no test targets": 3' in text
    assert "including incomplete collections" in text
    if case == "equal":
        assert "valid bugs / injector episodes: 0 / 2" in text
        assert "repair attempts: 0" in text and "optimizer steps: 0" in text
    else:
        assert "optimizer steps: 2" in text and "repair attempts: 4" in text
    if case != "varied":
        assert "Collection groups with sampled actions but only zero advantages: 1" in text
        assert "equal returns into zero advantages" in text
    else:
        assert "only zero advantages" not in text
    for name in ("experiment.md", "experiments-summary.md", "paper.md"):
        assert text in (tmp_path / "report" / name).read_text()
    assert text in readme.read_text()


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


def test_failure_metadata_coverage_and_configured_bootstrap_are_reported(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"expected_human_tasks": 3,
        "config": {"experiment": {"bootstrap_samples": 123}}}))
    (run / "failure.json").write_text(json.dumps({"phase": "preflight", "error": "fixture failure"}))
    (run / "preparation.json").write_text(json.dumps({"status": "fixture_prepared"}))
    (run / "evaluation.jsonl").write_text(json.dumps(record("a", "base", False)) + "\n")
    readme = tmp_path / "README.md"
    readme.write_text("Preserve this intro.\n<!-- SSR_RESULTS:START -->\nold\n<!-- SSR_RESULTS:END -->\nPreserve this end.\n")
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site", readme_path=readme)
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    assert data["status"] == "preflight_failed"
    assert data["failure"]["error"] == "fixture failure"
    assert data["preparation"]["preparation"]["status"] == "fixture_prepared"
    assert data["analysis"]["bootstrap"]["samples"] == 123
    assert data["analysis"]["conditions"]["human"]["base"]["planned"] == 3
    assert data["analysis"]["conditions"]["human"]["base"]["unrecorded"] == 2
    assert all(control["status"] == "not_run" for control in data["controls"])
    assert "fixture failure" in (tmp_path / "report" / "experiment.md").read_text()
    assert "Preserve this intro." in readme.read_text() and "Preserve this end." in readme.read_text()
    assert "single-agent" in readme.read_text()


def test_diagnostics_count_real_calls_not_copied_observations(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    events = [{"event": "session_started", "episode_id": "e", "session_id": "r", "parent_session_id": None},
              {"event": "session_started", "episode_id": "e", "session_id": "c", "parent_session_id": "r"},
              {"event": "tool_started", "name": "subagent", "session_id": "r"},
              {"event": "generation", "call_id": "unique", "generation": {"completion_ids": [1, 2, 3]}},
              {"event": "artifact_validated", "valid": False, "reason": "fixture invalid"}]
    (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (run / "metrics.jsonl").write_text(json.dumps({"step": 1, "memory": 100, "optimizer_steps": 2}) + "\n")
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site")
    diagnostics = json.loads((tmp_path / "site" / "data.json").read_text())["diagnostics"]
    assert diagnostics["generated_action_tokens"] == 3
    assert diagnostics["delegation_calls"] == 1
    assert diagnostics["child_sessions"] == 1
    assert diagnostics["invalid_artifact_rate"] == 1
    assert diagnostics["invalid_reasons"] == {"fixture invalid": 1}
    assert diagnostics["peak_allocated_bytes"] == 100


def test_likelihood_history_retains_checks_and_derives_drift_without_counting_preflight_actions(tmp_path, monkeypatch):
    """A later guard amendment must not erase earlier diagnostics or imply training."""
    run = tmp_path / "run"
    preflight = run / "preflight"
    preflight.mkdir(parents=True)
    history = [
        {"event": "likelihood_checked", "timestamp": "2020-01-01T00:00:00Z",
         "sampled_logprobs": [-1.0, -2.0], "scoring_logprobs": [-1.2, -1.8],
         "replayed_logprobs": [-1.0, -2.0], "scoring_max_abs_difference": 0.2,
         "replay_max_abs_difference": 0.0, "difference_limit": 0.05, "finite": True},
        {"event": "likelihood_checked", "timestamp": "2020-01-02T00:00:00Z",
         "sampled_logprobs": [-1.0, -2.0], "scoring_logprobs": [-0.9, -1.7],
         "replayed_logprobs": [-1.0, -2.01], "scoring_max_abs_difference": 0.3,
         "replay_max_abs_difference": 0.01, "difference_limit": None, "replay_limit": 0.05,
         "scoring_guard": {"kind": "preupdate_ratios_inside_clip_interval",
                           "ratio_lower": 0.8, "ratio_upper": 1.25}, "finite": True},
    ]
    other = {"event": "generation_complete", "completion_ids": [1, 2]}
    (preflight / "model.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [history[0], other, history[1]]))
    latest = {key: value for key, value in history[-1].items() if key not in {"event", "timestamp"}}
    (preflight / "likelihood.json").write_text(json.dumps(latest))
    paper_sections = []
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda payload, sections, *args: paper_sections.extend(sections))
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    assert data["likelihood_history"] == history
    assert data["likelihood"] == latest
    assert not data["events"]
    assert data["diagnostics"]["generated_action_tokens"] == 0
    assert data["diagnostics"]["generation_calls"] == 0
    disclosure = data["likelihood_disclosure"]
    assert "2020-01-01T00:00:00Z" in disclosure and "2020-01-02T00:00:00Z" in disclosure
    assert "[0.818731, 1.2214]" in disclosure and "[1.10517, 1.34986]" in disclosure
    assert "0/2" in disclosure and "1/2" in disclosure
    assert "absolute log-probability difference limit 0.05" in disclosure
    assert "preupdate_ratios_inside_clip_interval" in disclosure
    assert "does not bound drift in training contexts" in disclosure
    assert ("Preflight likelihood diagnostics", disclosure) in paper_sections
    for name in ("experiment.md", "experiments-summary.md", "paper.md"):
        text = (tmp_path / "report" / name).read_text()
        assert disclosure in text
        assert "https://github.com/huggingface/transformers/issues/46190" in text
        assert "https://arxiv.org/html/2510.02387v1#A3" in text
    paper = (tmp_path / "report" / "paper.md").read_text()
    assert "https://github.com/huggingface/transformers/issues/46190" in paper
    assert "BF16 mixed-precision LoRA training" in paper and "BF16 adapters" not in paper


def test_protocol_diagnostics_preserve_usage_and_mark_unscored_traces(tmp_path, monkeypatch):
    """A harness failure remains visible without turning generated role text into tools."""
    run = tmp_path / "run"
    run.mkdir()
    records = [{"timestamp": "2020-01-01", "kind": "chat_eos_mismatch", "episode_id": "diagnostic",
                "task_id": "fixture-task", "code_commit": "fixture-commit", "source_model_eos_ids": [9, 10],
                "tokenizer_eos_id": 11, "observed_calls": 2, "generated_tokens": 5,
                "calls_continuing_after_chat_eos": 1, "status": "interrupted_unscored",
                "reason": "fixture role continuation", "stop_time": "2020-01-01T00:01:00Z",
                "correction": "fixture explicit chat EOS"},
               {"timestamp": "2020-01-02", "kind": "chat_eos_mismatch", "episode_id": "later-diagnostic",
                "task_id": "later-task", "observed_calls": 1, "generated_tokens": 4,
                "calls_continuing_after_chat_eos": 1, "status": "interrupted_unscored",
                "reason": "later fixture diagnostic"}]
    (run / "protocol_diagnostics.json").write_text(json.dumps(records))
    events = [{"event": "session_started", "episode_id": "diagnostic", "session_id": "session"},
              {"event": "generation", "episode_id": "diagnostic", "session_id": "session", "call_id": "a",
               "generation": {"completion_ids": [1, 2, 3], "text": "fabricated subagent tool observation"}},
              {"event": "generation", "episode_id": "diagnostic", "session_id": "session", "call_id": "b",
               "generation": {"completion_ids": [4, 5]}},
              {"event": "tool_started", "session_id": "session", "name": "read"},
              {"event": "generation", "episode_id": "later-diagnostic", "call_id": "c",
               "generation": {"completion_ids": [6, 7, 8, 9]}},
              {"event": "generation", "episode_id": "ordinary", "call_id": "d",
               "generation": {"completion_ids": [10]}}]
    raw_events = "".join(json.dumps(event) + "\n" for event in events)
    (run / "events.jsonl").write_text(raw_events)
    (run / "evaluation.jsonl").write_text(json.dumps(record("fixture-task", "base", None, "interrupted_unscored")) + "\n")
    sections = []
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda payload, current, *args: sections.extend(current))
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    assert data["protocol_diagnostics"] == records
    assert data["diagnostics"]["generation_calls"] == 4
    assert data["diagnostics"]["generated_action_tokens"] == 10
    assert data["diagnostics"]["delegation_calls"] == data["diagnostics"]["optimizer_steps"] == 0
    assert data["analysis"]["conditions"]["human"]["base"]["completed"] == 0
    assert data["events"][3]["protocol_diagnostic"]["status"] == "interrupted_unscored"
    assert "protocol_diagnostic" not in data["events"][-1]
    assert (run / "events.jsonl").read_text() == raw_events
    disclosure = data["protocol_disclosure"]
    assert "Recorded model calls: 2; generated tokens: 5" in disclosure
    assert "Recorded model calls: 1; generated tokens: 4" in disclosure
    assert "fixture explicit chat EOS" in disclosure and "later fixture diagnostic" in disclosure
    assert "excluded from benchmark success and training" in disclosure
    assert "only recorded harness tool events establish actual tool execution" in disclosure
    assert ("Unscored protocol diagnostics", disclosure) in sections
    for name in ("experiment.md", "experiments-summary.md", "paper.md"):
        assert disclosure in (tmp_path / "report" / name).read_text()


def test_runtime_errors_are_descriptive_scoped_and_exclude_protocol_diagnostics(tmp_path, monkeypatch):
    """Count emitted errors, not model text, test failures, or excluded protocol episodes."""
    run = tmp_path / "run"
    run.mkdir()
    unsafe = {"error": "ValueError: Unsafe repository path: '/repo/example.py'"}
    events = [{"event": "session_started", "session_id": "diagnostic", "episode_id": "diagnostic"},
              {"event": "generation", "call_id": "d", "episode_id": "diagnostic", "generation": {"completion_ids": [1, 2, 3, 4]}},
              {"event": "tool_finished", "session_id": "diagnostic", "result": unsafe},
              {"event": "session_started", "session_id": "ordinary", "episode_id": "ordinary"},
              {"event": "generation", "call_id": "r", "episode_id": "ordinary",
               "generation": {"completion_ids": [5, 6], "text": "ContextLimitError: Unsafe repository path"}},
              {"event": "tool_finished", "session_id": "ordinary", "result": unsafe},
              {"event": "tool_finished", "session_id": "ordinary", "result": {"error": "ValueError: unknown tool"}},
              {"event": "tool_finished", "session_id": "ordinary", "result": {"stdout": "Unsafe repository path", "exit_code": 1}}]
    (run / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    (run / "protocol_diagnostics.json").write_text(json.dumps([
        {"episode_id": "diagnostic", "task_id": "diagnostic-task", "status": "interrupted_unscored"}]))
    context = {"agent_error": "ContextLimitError: required prompt exceeds capacity"}
    evaluations = [{**record("a", "base", None, "infra_error"), **context},
                   {**record("a", "base", False), **context}, record("b", "base", False),
                   {**record("c", "trained", True), **context},
                   {**record("d", "base", False, source="synthetic"), **context},
                   {**record("diagnostic-task", "base", None, "interrupted_unscored"), **context}]
    (run / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in evaluations))
    sections = []
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda payload, current, *args: sections.extend(current))
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    diagnostics = data["runtime_diagnostics"]
    assert diagnostics["tool_result_events"] == 3
    assert diagnostics["tool_error_events"] == 2
    assert diagnostics["unsafe_repository_path_errors"] == 1
    assert diagnostics["evaluation_context_limits"] == {
        "human/base": {"records": 2, "context_limit_errors": 1, "completed_context_limit_errors": 1},
        "human/trained": {"records": 1, "context_limit_errors": 1, "completed_context_limit_errors": 1},
        "synthetic/base": {"records": 1, "context_limit_errors": 1, "completed_context_limit_errors": 1}}
    assert data["diagnostics"]["generated_action_tokens"] == 6
    assert data["diagnostics"]["generation_calls"] == 2
    disclosure = data["runtime_disclosure"]
    assert "human/base" in disclosure and "synthetic/base" in disclosure
    assert "post-hoc descriptive" in disclosure
    assert "do not establish a causal explanation" in disclosure
    assert "remain fixed after held-out scoring" in disclosure
    assert ("Runtime interface and context diagnostics", disclosure) in sections
    for name in ("experiment.md", "experiments-summary.md", "paper.md"):
        assert disclosure in (tmp_path / "report" / name).read_text()


def test_recovery_disclosure_preserves_failed_attempt_after_success(tmp_path, monkeypatch):
    """A successful recovery cannot erase a failed attempt or its compute charge."""
    run = tmp_path / "recovery"
    run.mkdir()
    recovery = {
        "prior_run": "runs/interrupted-fixture", "prior_status": "preflight_failed",
        "prior_budget": {"deadline_epoch": 100},
        "prior_events": [{"event": "phase_failed", "phase": "preflight", "error": "fixture broken pipe"}],
        "prior_attempt_elapsed_seconds": 12.5, "charged_prior_seconds": 17,
        "recovery_budget_seconds": 83, "total_active_allocation_seconds": 100,
        "original_deadline_preserved": True,
        "saved_model_generations": 0, "saved_evaluations": 0, "saved_checkpoints": 0,
        "reason": "An OS restart interrupted this fixture before model output.",
    }
    (run / "manifest.json").write_text(json.dumps({"recovery": recovery}))
    (run / "summary.json").write_text(json.dumps({"status": "completed"}))
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    assert data["status"] == "completed"
    assert data["recovery"] == recovery
    for name in ("experiment.md", "experiments-summary.md", "paper.md"):
        text = (tmp_path / "report" / name).read_text()
        assert "runs/interrupted-fixture" in text and "fixture broken pipe" in text
        assert "12.5 seconds" in text and "charges 17 seconds" in text
        assert "83 seconds" in text and "100 seconds" in text
        assert "offline interval" in text and "amendment" in text
        assert "model outputs 0, evaluations 0, checkpoints 0" in text


@pytest.mark.parametrize("later_events,new_failure,expected", [
    ([("phase_started", "preflight")], False, "preflight_running"),
    ([("phase_completed", "preflight"), ("phase_started", "pilot")], False, "pilot_running"),
    ([("phase_completed", "preflight"), ("phase_started", "pilot"),
      ("phase_interrupted", "pilot")], False, "pilot_interrupted"),
    ([("phase_completed", "preflight"), ("phase_started", "pilot")], True, "pilot_failed"),
])
def test_running_and_interrupted_phases_supersede_older_terminal_state(tmp_path, monkeypatch, later_events, new_failure, expected):
    """Active recovery is not an old success/failure, while newer failures still win."""
    run = tmp_path / "run"
    run.mkdir()
    original = {"timestamp": "2020-01-01T00:00:01Z", "phase": "preflight", "error": "original failure"}
    events = [{**original, "event": "phase_failed"}]
    events.extend({"timestamp": f"2020-01-01T00:00:{index:02d}Z", "event": event, "phase": phase}
                  for index, (event, phase) in enumerate(later_events, 2))
    (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    failure = ({"timestamp": "2020-01-01T00:00:09Z", "phase": "pilot", "error": "new supervisor failure"}
               if new_failure else original)
    (run / "failure.json").write_text(json.dumps(failure))
    (run / "summary.json").write_text(json.dumps({"status": "completed"}))
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    assert data["status"] == expected
    assert [row["error"] for row in data["failure_history"]] == ["original failure"] + (
        ["new supervisor failure"] if new_failure else [])
    assert expected in (tmp_path / "report" / "paper.md").read_text()


@pytest.mark.parametrize("latest_source", ["preflight", "research"])
def test_parameter_storage_dtypes_use_latest_model_record_without_counting_actions(tmp_path, monkeypatch, latest_source):
    """Storage evidence may come from either model log, independently of action accounting."""
    run = tmp_path / "run"
    (run / "preflight").mkdir(parents=True)
    expected = {"frozen:torch.bfloat16": 100, "frozen:torch.float32": 2, "trainable:torch.float32": 7}
    for source, path in (("preflight", run / "preflight" / "model.jsonl"), ("research", run / "model.jsonl")):
        event = {"event": "model_loaded", "timestamp": "2020-01-02" if source == latest_source else "2020-01-01",
                 "parameter_dtype_counts": expected if source == latest_source else {"trainable:torch.float32": 3}}
        other = {"event": "generation_complete", "completion_ids": [1, 2, 3]}
        path.write_text(json.dumps(event) + "\n" + json.dumps(other) + "\n")
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site" / "data.json").read_text())
    assert data["parameter_dtype_counts"] == expected
    assert not data["events"]
    assert data["diagnostics"]["generation_calls"] == 0
    assert data["diagnostics"]["generated_action_tokens"] == 0
    paper = (tmp_path / "report" / "paper.md").read_text()
    assert "Recorded parameter storage counts" in paper
    assert all(f"{name} = {count}" in paper for name, count in expected.items())


@pytest.mark.parametrize("later_failure", [None, "pilot", "supervisor"])
def test_preflight_recovery_status_uses_latest_phase_and_retains_failures(tmp_path, monkeypatch, later_failure):
    """Exercise failed then successful preflight orchestration without an LLM."""
    import sys
    from types import SimpleNamespace

    from multi_agent_ssr import experiment, prepare
    from multi_agent_ssr.io import EventLog, utc_now, write_json

    monkeypatch.chdir(tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/model.json").write_text('{"revision": "fixture-pin"}')
    config = {"model": {}, "limits": {}, "experiment": {
        "budget_seconds": 60, "human_tasks": 32, "synthetic_tasks": 16}}
    monkeypatch.setattr(experiment, "load_config", lambda _: config)
    monkeypatch.setattr(experiment, "require_preregistered", lambda: "fixture-commit")
    monkeypatch.setattr(experiment.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="fixture GPU"))
    monkeypatch.setattr(prepare, "preparation_evidence", lambda: {})
    calls = 0

    def fixture_preflight(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fixture first preflight failure")
        return {"success": True, "purpose": "preflight_only"}

    monkeypatch.setitem(sys.modules, "multi_agent_ssr.learning", SimpleNamespace(run_preflight=fixture_preflight))
    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="first preflight failure"):
        experiment.execute("preflight", tmp_path / "config.toml", run)
    original_failure = (run / "failure.json").read_bytes()
    experiment.execute("preflight", tmp_path / "config.toml", run)
    assert (run / "failure.json").read_bytes() == original_failure
    assert load_jsonl(run / "events.jsonl")[-1]["event"] == "phase_completed"
    likelihood = {"sampled_logprobs": [-0.5], "max_logprob_error": 0.1}
    kernel_review = {"recommendation": "fixture only; no installation"}
    write_json(run / "preflight" / "likelihood.json", likelihood)
    write_json(run / "kernel-review.json", kernel_review)

    if later_failure:
        error = f"fixture {later_failure} failure"
        if later_failure == "pilot":
            EventLog(run / "events.jsonl").emit("phase_failed", phase="pilot", error=error)
        write_json(run / "failure.json", {"timestamp": utc_now(), "phase": "pilot", "error": error})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_figures", lambda *args: {})
    monkeypatch.setattr("multi_agent_ssr.reporting.render_paper", lambda *args: None)
    generate_report(run, tmp_path / "report", tmp_path / "site")
    data = json.loads((tmp_path / "site/data.json").read_text())
    assert data["status"] == ("pilot_failed" if later_failure else "preflight_passed")
    assert data["likelihood"] == likelihood
    assert data["preparation"]["kernel_review"] == kernel_review
    errors = [row["error"] for row in data["failure_history"]]
    assert errors == ["RuntimeError: fixture first preflight failure"] + ([error] if later_failure else [])
    for name in ("experiment.md", "experiments-summary.md", "paper.md"):
        text = (tmp_path / "report" / name).read_text()
        assert "fixture first preflight failure" in text
        if later_failure:
            assert error in text
