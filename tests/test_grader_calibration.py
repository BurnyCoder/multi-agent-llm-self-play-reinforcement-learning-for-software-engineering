"""Verify durable fixed controls without Docker or model execution."""

import json

from multi_agent_ssr import grader_calibration as calibration


def test_interrupted_attempt_is_retained_and_terminal_controls_are_reused(tmp_path, monkeypatch):
    task = {"instance_id": "sample", "base_commit": "base", "patch": "human fix"}
    interrupted = tmp_path / "attempts/sample/empty/interrupted/started.json"
    interrupted.parent.mkdir(parents=True)
    interrupted.write_text("{}")
    calls = []

    def grade(task, patch, condition, attempt, *, image_digest, timeout):
        calls.append((patch, image_digest, timeout))
        if patch:
            raise RuntimeError("fixture grader unavailable")
        return {"success": False, "status": "completed", "official_report": {"resolved": False}}

    monkeypatch.setattr(calibration, "grade_prediction", grade)
    for _ in range(2):
        calibration._grade_task((0, task, "pinned", tmp_path))
    assert calls == [("", "pinned", 300), ("human fix", "pinned", 300)]
    assert interrupted.exists()
    rows = {r["condition"]: r for p in tmp_path.glob("attempts/*/*/*/result.json")
            for r in [json.loads(p.read_bytes())]}
    assert rows["empty"]["success"] is False
    assert rows["gold"]["status"] == "infra_error" and rows["gold"]["success"] is None
