"""Statistical interpretation of a delivered intervention and its frozen binary cohort.

The pilot rule is in reports/preregistration.md. The binary endpoint bounds use the
worst-case completion principle in https://doi.org/10.1080/01621459.2000.10473902.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evidence import file_sha256

PROJECT = Path(__file__).resolve().parents[2]
REVIEW = PROJECT / "results/scientific-audit-2026-11-26"


def hypothesis_status(identity, steps, comparison):
    """Distinguish a measured training contrast from a numerical endpoint on two runs."""
    if identity == "unchanged" and steps == 0:
        return "not_tested_no_update"
    if identity == "unchanged" and steps is not None and steps > 0:
        return "not_tested_unchanged_checkpoint"
    if identity != "changed" or steps is None or steps <= 0:
        return "intervention_unverified"
    return comparison["preregistered_decision"] if comparison["paired_tasks"] else "not_evaluated"


def _index(ids, rows, conditions):
    """Require one retained observation per frozen task and condition."""
    cohort = set(ids)
    if len(cohort) != len(ids):
        raise ValueError("Duplicate frozen task")
    values = {}
    for row in rows:
        task, condition = row["task_id"], row["condition"]
        if task not in cohort or condition not in conditions:
            raise ValueError(f"Observation outside frozen cohort or conditions: {task}/{condition}")
        if (task, condition) in values:
            raise ValueError(f"Duplicate observation: {task}/{condition}")
        score = row.get("success")
        if row.get("status") == "completed":
            if type(score) is not bool:
                raise ValueError("A completed observation requires a boolean score")
        elif score is not None:
            raise ValueError("An unscored observation cannot carry a score")
        values[task, condition] = score
    return values


def completion_bounds(ids, rows):
    """Fill each missing binary cell at the extrema of final minus base."""
    values = _index(ids, rows, {"base", "trained"})
    counts = dict.fromkeys(("paired", "base_only", "final_only", "neither"), 0)
    low = high = 0
    for task in ids:
        base, final = values.get((task, "base")), values.get((task, "trained"))
        cell = {(True, True): "paired", (True, False): "base_only",
                (False, True): "final_only", (False, False): "neither"}[base is not None, final is not None]
        counts[cell] += 1
        low += (final if final is not None else 0) - (base if base is not None else 1)
        high += (final if final is not None else 1) - (base if base is not None else 0)
    return {"planned": len(ids), **counts, "minimum_net_resolutions": low, "maximum_net_resolutions": high,
            "difference_range": [low / len(ids), high / len(ids)] if ids else None}


def calibration_outcomes(ids, rows):
    """Grade the predeclared discrimination criterion independently of policy outcomes."""
    values = _index(ids, rows, {"gold", "empty"})
    tasks = []
    for task in ids:
        gold, empty = values.get((task, "gold")), values.get((task, "empty"))
        result = "indeterminate" if gold is None or empty is None else ("pass" if gold and not empty else "fail")
        tasks.append({"task_id": task, "gold": gold, "empty": empty, "discrimination": result})
    return tasks


def linked_review(run, folder=REVIEW):
    """Attach the portable forensic results only to the exact historical run they describe."""
    evidence_file = folder / "evidence.json"
    manifest = run / "manifest.json"
    if not evidence_file.is_file() or not manifest.is_file():
        return None
    audit = json.loads(evidence_file.read_bytes())
    inputs = audit["run_sha256"]
    if file_sha256(manifest) != inputs["manifest.json"]:
        return None
    for name in inputs:
        if not (run / name).is_file() or file_sha256(run / name) != inputs[name]:
            raise ValueError(f"Archived scientific review input changed: {name}")
    archive = folder / "oracle-tests.json.gz"
    if file_sha256(archive) != audit["test_archive"]["sha256"]:
        raise ValueError("Archived per-test evidence changed")
    return audit


def describe_hypothesis(status):
    """Translate the evidence-gated decision into the report's primary conclusion."""
    return {
        "not_tested_no_update": "The primary learning hypothesis was not tested. The research run recorded zero optimizer steps and selected an adapter identical to initialization.",
        "not_tested_unchanged_checkpoint": "The primary learning hypothesis was not tested. The selected adapter had the same saved weights and configuration as initialization.",
        "intervention_unverified": "The evidence is insufficient to verify delivery of the learning intervention.",
        "not_evaluated": "The learning intervention has no completed paired evaluation.",
        "supports_predicted_improvement": "The paired interval meets the registered criterion for improvement for these checkpoints.",
        "contradicts_predicted_improvement": "The paired interval meets the registered criterion for a nonpositive contrast for these checkpoints.",
        "inconclusive": "The paired interval is inconclusive about improvement for these checkpoints.",
    }[status]


def public_findings(audit):
    """Compact results for readers. Full attempts and expected test entries travel as downloads."""
    hist, cal = audit["historical"], audit["calibration"]
    training = audit["training"]
    episodes = [e for g in training["groups"] for e in g["episodes"]]
    exhausted = sum(e["budget_exhausted"] and not e["test_targets"] for e in episodes)
    bounds = hist["bounds"]
    lo, hi = bounds["difference_range"]
    passes = cal["counts"].get("pass", 0)
    failed = cal["counts"].get("fail", 0)
    unknown = cal["counts"].get("indeterminate", 0)
    subset = cal["sensitivity"]
    p2p = hist["test_summary"]["PASS_TO_PASS"]
    f2p = hist["test_summary"]["FAIL_TO_PASS"]
    marker = hist["marker_sensitivity"]
    count = len(cal["tasks"])
    def effect(pair):
        if not pair["paired"]:
            return "The subset has no completed historical pairs."
        a, b = pair["ci95"]
        return (f"The final-minus-base difference is {100 * pair['difference']:+.1f} percentage points "
                f"(paired percentile interval [{100*a:+.1f}, {100*b:+.1f}]).")
    notes = [
        ("Collection and training exposure",
         f"The {len(training['groups'])} completed groups contained {len(episodes)} injector episodes. "
         f"{exhausted} reached the tool budget with no declared test targets. "
         f"{sum(e['advantage'] == 0 for e in episodes)} episodes had zero length-centered advantage. "
         f"The original action stream matched {training['training_action_count']} saved collection actions. "
         f"The effective messages for {training['unique_training_prompt_token_sequences']} distinct training prompt token sequences "
         f"contained {len(training['verbatim_reference_patch_matches'])} verbatim complete held-out reference or test patches. "
         "Partial overlap, memorization and pretraining exposure remain unknown."),
        ("Human cohort and missing outcomes",
         f"The frozen selection took {count} tasks from {audit['dataset']['eligible']} eligible rows of the "
         f"{audit['dataset']['population']}-row pinned dataset. Both conditions were scored on {bounds['paired']} tasks, "
         f"base only on {bounds['base_only']}, final only on {bounds['final_only']}, and neither on {bounds['neither']}. "
         f"Assigning every missing binary outcome its extremes gives a possible net change of "
         f"{bounds['minimum_net_resolutions']:+d} to {bounds['maximum_net_resolutions']:+d} resolutions out of {count} "
         f"({100*lo:+.2f} to {100*hi:+.2f} percentage points). These endpoints describe possible completions of "
         "this finite cohort. The paired bootstrap describes only the observed pairs."),
        ("Replay of historical scoring",
         f"The locked scorer reproduced all {hist['replayed_reports']} archived human grades from stdout. "
         f"{hist['empty_submissions']} of the submitted patches were empty. The PASS_TO_PASS failure category contained "
         f"{p2p['entries']} entries across {p2p['rows']} submissions on {len(p2p['task_ids'])} tasks. "
         f"Parsed states among those entries were {json.dumps(p2p['by_parsed_status'], sort_keys=True)}. "
         f"The FAIL_TO_PASS failure category contained {f2p['entries']} entries, with parsed states "
         f"{json.dumps(f2p['by_parsed_status'], sort_keys=True)}. Expected IDs included a literal pytest progress marker "
         f"on {len(hist['progress_marker_tasks'])} tasks. {marker['separate_fail_to_pass_failure']} of the "
         f"{marker['rows']} affected submissions also had a separate repair-required failure. "
         "Test-level classifications and alias matches are retained in the per-test archive."),
        ("Fixed-patch grader calibration",
         f"The grader resolved the published repair and rejected the empty patch on {passes}/{count} tasks. "
         f"{failed} tasks failed this criterion and {unknown} were indeterminate. "
         f"Gold repairs resolved on {sum(t['gold'] is True for t in cal['tasks'])}/{count} planned tasks. "
         f"Empty submissions resolved on {sum(t['empty'] is True for t in cal['tasks'])}/{count}. "
         "These CPU measurements characterize the pinned grading environment. A passing pair validates only these two inputs."),
        ("Sensitivity to grader discrimination",
         f"Of {passes} tasks passing calibration, {subset['paired']} had both historical policy scores "
         f"and {passes - subset['paired']} lacked at least one. " + effect(subset) +
         " The restriction was specified before control execution and after the pilot outcomes were known. "
         "The full original analysis retains every admitted score."),
    ]
    notes.append(("Synthetic diagnostic cohort",
                  f"The private records and public manifest agree on {len(audit['synthetic']['task_ids'])} mutations "
                  f"across {audit['synthetic']['distinct_functions']} distinct functions. The saved clean suite sizes were "
                  f"{', '.join(map(str, audit['synthetic']['clean_test_counts']))}. "
                  f"The policy evaluation stream contains {audit['synthetic']['scored']} scored synthetic outcomes."))
    notes.append(("Protocol changes and stopping",
                  f"The interrupted original pilot records match the recovery amendment. The final observed worker "
                  f"overrun was {training['deadline_overrun_seconds']:.3f} seconds. The stopping diagnostic identified "
                  f"{sum(x['continued_after_chat_eos'] for x in training['stopping_diagnostics'])} calls continuing "
                  "past chat EOS in the unscored held-out episode. That exposure preceded the stopping correction. "
                  "The likelihood gate was also amended after inspection of the same preflight sample."))
    return {"sections": notes, "tasks": cal["tasks"], "bounds": bounds, "sensitivity": subset}


def review_report(audit, findings, headline, claim_ledger):
    """Render fixed-cohort counts, the per-task controls and the source-by-claim review."""
    hist = {(r["task_id"], r["condition"]): r.get("success") for r in audit["historical"]["retained"]}
    def label(value):
        return {True: "resolved", False: "unresolved", None: "unscored"}[value]
    table = ["| Frozen task | Historical base | Historical final | Gold | Empty | Discrimination |",
             "| --- | --- | --- | --- | --- | --- |"]
    for task in findings["tasks"]:
        tid = task["task_id"]
        cells = [tid, label(hist.get((tid, "base"))), label(hist.get((tid, "trained"))),
                 label(task["gold"]), label(task["empty"]), task["discrimination"]]
        table.append("| " + " | ".join(cells) + " |")
    failures = ["| Patch | Required category | Official failure entries | Parsed state counts |",
                "| --- | --- | ---: | --- |"]
    for kind, cats in audit["calibration"]["test_summary"].items():
        for cat, summary in cats.items():
            failures.append(f"| {kind} | {cat} | {summary['entries']} | {json.dumps(summary['by_parsed_status'], sort_keys=True)} |")
    infra = [f"- {r['task_id']} ({r['condition']}), {r['status']}: {r.get('reason') or 'no reason recorded'}"
             for r in audit["calibration"]["controls"] if r["status"] != "completed"]
    issue = ("\n\n" + "\n".join(infra)) if infra else "\n\nEvery control had an admitted grade."
    return ("# Scientific review of the software-repair pilot\n\n" + audit["date"] + "\n\n" + headline + "\n\n" +
            "\n\n".join(f"## {title}\n\n{body}" for title, body in findings["sections"]) +
            "\n\n## All frozen human tasks\n\n" + "\n".join(table) +
            "\n\n## Replayed control test entries\n\nCounts include available stdout from unscored attempts.\n\n" +
            "\n".join(failures) + issue + "\n\n" + claim_ledger)
