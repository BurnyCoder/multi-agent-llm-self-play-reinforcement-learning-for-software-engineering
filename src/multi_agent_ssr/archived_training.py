"""Reconstruct the pilot's intervention from saved actions, optimizer rows and checkpoint bytes.

Length-weighted centering is specified at https://arxiv.org/html/2510.02387v1#A3.
Reading JSON and hashing saved adapters does not initialize a policy.
"""

import hashlib
import json
import math
from collections import Counter

from .evidence import file_sha256


def json_rows(path):
    """Stream JSONL so token chunks do not require a second in-memory transcript."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def reconstruct_training(run, tasks):
    """Follow collection IDs to episodes and compare original action and model evidence."""
    events = list(json_rows(run / "events.jsonl"))
    episode_for = {r["session_id"]: r["episode_id"] for r in events if r["event"] == "session_started"}
    actions = {r["call_id"]: r for r in events if r["event"] == "generation"}
    if len(actions) != sum(r["event"] == "generation" for r in events):
        raise ValueError("Duplicate action identity")
    tool_calls = Counter(episode_for[r["session_id"]] for r in events if r["event"] == "tool_started")
    metrics = list(json_rows(run / "metrics.jsonl"))
    groups, prompts, used_actions = [], set(), set()
    files = sorted((run / "collection_groups").glob("group-*.json"))
    for path, metric in zip(files, metrics, strict=True):
        record = json.loads(path.read_bytes())
        entries = []
        for episode_group in record["episode_groups"]:
            group = []
            for eid in episode_group:
                episode = json.loads((run / "episodes" / (eid + ".json")).read_bytes())
                count = 0
                for act in episode["actions"]:
                    cid, gen = act["call_id"], act["generation"]
                    if cid in used_actions or actions[cid]["episode_id"] != eid or actions[cid]["generation"] != gen:
                        raise ValueError(f"Duplicated or misattributed recorded action {cid}")
                    if (len(gen["old_logprobs"]) != len(gen["completion_ids"])
                            or gen["policy_version"] != record["behavior_policy_version"]
                            or not all(math.isfinite(x) for x in gen["old_logprobs"])):
                        raise ValueError("Invalid behavior-policy record")
                    used_actions.add(cid)
                    prompts.add(tuple(gen["prompt_ids"]))
                    count += len(gen["completion_ids"])
                result = episode["result"]
                group.append({"episode_id": eid, "team": episode["team"], "task": episode["task_id"],
                              "return": episode["reward"], "tokens": count, "tool_calls": tool_calls[eid],
                              "budget_exhausted": result.get("budget_exhausted", False),
                              "test_targets": result.get("test_targets", []),
                              "validation_reason": result.get("validation", {}).get("reason")})
            length = sum(e["tokens"] for e in group)
            center = math.fsum(e["tokens"] * e["return"] for e in group) / length if length else None
            for e in group:
                e["advantage"] = e["return"] - center if center is not None else 0
            entries.extend(group)
        if (sum(e["tokens"] for e in entries) != metric["sampled_tokens"]
                or sum(e["advantage"] == 0 for e in entries) != metric["zero_advantage_episodes"]
                or len(metric["passes"]) != metric["optimizer_steps"]):
            raise ValueError("Episode reconstruction differs from saved training metrics")
        groups.append({"group": metric["step"], "episodes": entries, "optimizer_steps": len(metric["passes"]),
                       "trained_tokens_per_pass": metric["trained_tokens_per_pass"], "sha256": file_sha256(path)})
    gold = {t[k] for t in tasks for k in ("patch", "test_patch") if t[k]}
    observed, matched_refs = set(), set()
    for r in json_rows(run / "model.jsonl"):
        if r["event"] == "prompt_effective" and (key := tuple(r["prompt_ids"])) in prompts:
            observed.add(key)
            text = "\n".join(str(m.get("content", "")) for m in r["messages"])
            matched_refs.update(hashlib.sha256(p.encode()).hexdigest() for p in gold if p in text)
    if observed != prompts:
        raise ValueError("Missing effective training prompt in model stream")
    checkpoint = json.loads((run / "latest.json").read_bytes())["path"]
    hashes = {}
    for label, directory in (("initial", run / "checkpoints/initial"), ("selected", run / checkpoint)):
        meta = json.loads((directory / "checkpoint.json").read_bytes())
        hashes[label] = {"weights": file_sha256(directory / "adapter_model.safetensors"),
                         "config": file_sha256(directory / "adapter_config.json"),
                         "base": meta["model_config"]["model_id"], "revision": meta["model_config"]["revision"],
                         "policy_version": meta["policy_version"]}
    likelihood = []
    for path in sorted((run / "preflight").glob("likelihood*.json")):
        r = json.loads(path.read_bytes())
        values = list(zip(r["sampled_logprobs"], r["scoring_logprobs"], r["replayed_logprobs"], strict=True))
        logs = [new - old for old, new, replay in values]
        likelihood.append({"file": path.name, "sha256": file_sha256(path), "tokens": len(values),
                           "trainer_max_abs_difference": max(map(abs, logs)),
                           "replay_max_abs_difference": max(abs(replay-old) for old, new, replay in values),
                           "ratios_outside_clip": sum(not 0.8 <= math.exp(x) <= 1.25 for x in logs)})
    run_manifest = json.loads((run / "manifest.json").read_bytes())
    recovery = run_manifest["recovery"]
    prior = {name: file_sha256(run.parent / "pilot" / name) for name in recovery["original_record_sha256"]}
    if prior != recovery["original_record_sha256"]:
        raise ValueError("Original interrupted pilot bytes differ from the amendment")
    deadline = json.loads((run / "failure.json").read_bytes())["deadline_enforcement"]
    budget = json.loads((run / "budget.json").read_bytes())
    elapsed = deadline["observed_epoch"] - budget["deadline_epoch"]
    if deadline["deadline_epoch"] != budget["deadline_epoch"] or not math.isclose(elapsed, deadline["overrun_seconds"]):
        raise ValueError("Deadline arithmetic disagrees with the historical observation")
    stopping = []
    for row in json.loads((run / "protocol_diagnostics.json").read_bytes()):
        before_fix = [r["generation"]["completion_ids"] for r in actions.values() if r["episode_id"] == row["episode_id"]]
        eos = row["tokenizer_eos_id"]
        continued = sum(eos in ids[:-1] for ids in before_fix)
        if (len(before_fix) != row["observed_calls"] or continued != row["calls_continuing_after_chat_eos"]
                or sum(map(len, before_fix)) != row["generated_tokens"]):
            raise ValueError("Held-out stopping diagnostic differs from original token sequences")
        stopping.append({"task": row["task_id"], "episode_id": row["episode_id"], "calls": len(before_fix),
                         "continued_after_chat_eos": continued})
    return {"groups": groups, "checkpoint": hashes, "likelihood_checks": likelihood,
            "recovery_original_sha256": prior, "deadline_overrun_seconds": elapsed, "stopping_diagnostics": stopping,
            "unique_training_prompt_token_sequences": len(prompts), "verbatim_reference_patch_matches": sorted(matched_refs),
            "training_action_count": len(used_actions), "research_optimizer_steps": sum(g["optimizer_steps"] for g in groups),
            "phase_starts": [{k: r[k] for k in ("phase", "timestamp", "code_commit")}
                             for r in events if r["event"] == "phase_started"]}
