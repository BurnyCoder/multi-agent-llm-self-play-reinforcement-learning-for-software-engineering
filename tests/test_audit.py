"""Cross-check independent accounting with distinct scored and missing outcomes."""

import json

from multi_agent_ssr.audit import recompute


def test_raw_recomputation_keeps_missing_and_duplicate_evidence(tmp_path):
    rows = [{"source": "human", "condition": condition, "task_id": "x",
             "status": "completed", "success": False} for condition in ("base", "trained")]
    rows += [{"source": "synthetic", "condition": "trained", "task_id": "y",
              "status": "not_run_budget", "success": None}, rows[0]]
    (tmp_path / "evaluation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = recompute(tmp_path, samples=20)
    assert result["paired"]["human"]["ci95"] == [0.0, 0.0]
    assert result["paired"]["synthetic"]["ci95"] is None
    assert result["conditions"]["synthetic"]["trained"]["unscored"] == 1
    assert result["duplicate_completed_outcomes"][0]["identical"] is True
    assert result["optimizer_steps"] is None
    assert result["official"]["cached_reports_available"] == 0
