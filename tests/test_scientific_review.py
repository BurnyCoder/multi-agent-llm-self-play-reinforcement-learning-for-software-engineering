"""Model-free counterexamples for cohort membership, missingness and control grading."""

import itertools
import json
from collections import Counter

import pytest

from multi_agent_ssr.scientific_review import (
    calibration_outcomes,
    completion_bounds,
    hypothesis_status,
)


def observation(task, condition, success):
    return {"task_id": task, "source": "human", "condition": condition, "success": success,
            "status": "completed" if success is not None else "infra_error"}


def test_missing_outcomes_have_sharp_finite_cohort_bounds():
    rows = [observation("a", "base", False), observation("a", "trained", False),
            observation("b", "trained", False), observation("c", "base", None)]
    assert completion_bounds(["a", "b", "c"], rows) == {
        "planned": 3, "paired": 1, "base_only": 0, "final_only": 1, "neither": 1,
        "minimum_net_resolutions": -2, "maximum_net_resolutions": 1,
        "difference_range": [-2/3, 1/3]}
    with pytest.raises(ValueError, match="frozen"):
        completion_bounds(["a"], rows)


@pytest.mark.parametrize(("steps", "identity", "pairs", "expected"), [
    (0, "unchanged", 2, "not_tested_no_update"), (0, "changed", 2, "intervention_unverified"),
    (None, "unknown", 2, "intervention_unverified"), (2, "changed", 0, "not_evaluated"),
    (2, "changed", 3, "contradicts_predicted_improvement"),
    (2, "unchanged", 2, "not_tested_unchanged_checkpoint")])
def test_interval_label_requires_delivered_learning_intervention(steps, identity, pairs, expected):
    comparison = {"paired_tasks": pairs, "preregistered_decision": "contradicts_predicted_improvement"}
    assert hypothesis_status(identity, steps, comparison) == expected


def test_positive_negative_and_unavailable_controls_are_distinct():
    rows = [observation("a", "gold", True), observation("a", "empty", False),
            observation("b", "gold", False), observation("b", "empty", False),
            observation("c", "gold", None), observation("d", "empty", True),
            observation("d", "gold", True)]
    tasks = calibration_outcomes(list("abcd"), rows)
    assert [t["discrimination"] for t in tasks] == ["pass", "fail", "indeterminate", "fail"]
    with pytest.raises(ValueError, match="Duplicate"):
        calibration_outcomes(list("abcd"), rows + rows[:1])
    with pytest.raises(ValueError, match="frozen"):
        calibration_outcomes(["a"], rows)


def test_official_entry_and_parser_status_are_separate():
    from multi_agent_ssr.scientific_reconstruction import test_entries as entries

    reference = {"FAIL_TO_PASS": ["x[", "missing"], "PASS_TO_PASS": ["[100%]", "skipped"]}
    official = {"FAIL_TO_PASS": {"failure": ["x[", "missing"], "success": []},
                "PASS_TO_PASS": {"failure": [], "success": ["[100%]", "skipped"]}}
    parsed = {"x[1]": "FAILED", "[100%]": "PASSED", "skipped": "SKIPPED"}
    def resolve(key, observed):
        return next((s for s in observed if s.startswith(key)), None)
    result = entries(reference, official, parsed, resolve)
    assert [row["parsed_status"] for row in result] == ["FAILED", "MISSING", "PASSED", "SKIPPED"]
    assert result[0]["parsed_id"] == "x[1]"
    assert result[2]["progress_marker"] and result[3]["official_class"] == "success"


def test_logical_bounds_match_all_binary_completions():
    ids = ["a", "b"]
    cells = [(t, c) for t in ids for c in ("base", "trained")]
    for states in itertools.product((True, False, None), repeat=len(cells)):
        rows = [observation(t, c, v) for (t, c), v in zip(cells, states, strict=True)]
        measured = completion_bounds(ids, rows)["difference_range"]
        missing = [k for k, v in enumerate(states) if v is None]
        outcomes = []
        for choices in itertools.product((0, 1), repeat=len(missing)):
            filled = list(states)
            for k, v in zip(missing, choices, strict=True):
                filled[k] = v
            outcomes.append((filled[1] - filled[0] + filled[3] - filled[2])/2)
        assert measured == [min(outcomes), max(outcomes)]


def test_archived_review_only_attaches_to_its_original_inputs(tmp_path):
    from multi_agent_ssr.evidence import file_sha256
    from multi_agent_ssr.scientific_review import linked_review

    evidence, run = tmp_path / "forensics", tmp_path / "run"
    evidence.mkdir()
    run.mkdir()
    manifest, events = run / "manifest.json", run / "events.jsonl"
    manifest.write_text("{}")
    events.write_text('{"event":"example"}')
    archive = evidence / "oracle-tests.json.gz"
    archive.write_bytes(b"fixture")
    record = {"run_sha256": {p.name: file_sha256(p) for p in (manifest, events)},
              "test_archive": {"sha256": file_sha256(archive)}}
    (evidence / "evidence.json").write_text(json.dumps(record))
    assert linked_review(run, evidence) == record
    manifest.write_text('{"run":"another"}')
    assert linked_review(run, evidence) is None
    manifest.write_text("{}")
    events.write_text("")
    with pytest.raises(ValueError, match="events.jsonl"):
        linked_review(run, evidence)


def test_portable_control_summary_matches_each_saved_submission():
    import gzip

    from multi_agent_ssr.scientific_review import REVIEW, public_findings

    audit = json.loads((REVIEW / "evidence.json").read_bytes())
    cal = audit["calibration"]
    tasks = [r["task_id"] for r in cal["tasks"]]
    assert calibration_outcomes(tasks, cal["controls"]) == cal["tasks"]
    assert cal["counts"] == Counter(r["discrimination"] for r in cal["tasks"])
    assert completion_bounds(tasks, audit["historical"]["retained"]) == audit["historical"]["bounds"]
    traces = json.loads(gzip.decompress((REVIEW / audit["test_archive"]["name"]).read_bytes()))
    assert sum(t["phase"] == "historical" for t in traces) == audit["historical"]["replayed_reports"]
    assert public_findings(audit)["tasks"] == cal["tasks"]
