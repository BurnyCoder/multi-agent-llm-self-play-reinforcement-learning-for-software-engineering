"""Build a reviewable Hub upload bundle from the selected completed checkpoint.

Uploading is a separate explicit CLI operation; this module never reads tokens.
Hub model-card metadata: https://huggingface.co/docs/hub/model-cards .
"""
import json
import shutil
from pathlib import Path

from .io import write_json


def export_checkpoint(run_dir: Path, report_dir: Path, destination: Path) -> Path:
    """Export recorded results faithfully, refusing any preflight-only adapter."""
    latest = json.loads((run_dir / "latest.json").read_text())
    checkpoint = (run_dir / latest["path"]).resolve()
    if not checkpoint.is_relative_to(run_dir.resolve()):
        raise ValueError("Selected checkpoint escapes the run directory")
    record = json.loads((checkpoint / "checkpoint.json").read_text())
    if record.get("metadata", {}).get("purpose") == "preflight_only":
        raise ValueError("A preflight-only adapter cannot be published as a research result")
    if record["policy_version"] != latest["policy_version"]:
        raise ValueError("Selected checkpoint policy version differs from completed-group metadata")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report = json.loads((report_dir / "summary.json").read_text())
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise ValueError("Selected adapter weights are missing")
    destination.mkdir(parents=True, exist_ok=True)
    # Checkpoints are harness-owned; exclude arbitrary extensions and symlinks.
    allowed = {".json", ".safetensors", ".pt", ".jinja", ".txt"}
    for path in checkpoint.iterdir():
        if path.is_file() and not path.is_symlink() and path.suffix in allowed:
            shutil.copyfile(path, destination / path.name)
    write_json(destination / "experiment.json", manifest)
    write_json(destination / "evaluation-summary.json", report["analysis"])
    write_json(destination / "selection.json", latest)
    for name in ["experiment.md", "paper.pdf"]:
        if (report_dir / name).exists():
            shutil.copyfile(report_dir / name, destination / name)
    base = record["model_config"]
    updated = record["policy_version"] > 0
    status = ("A locally updated shared LoRA research adapter." if updated else
              "An unchanged research adapter: the pilot completed no optimizer updates. This is not a trained checkpoint.")
    card = ("---\nlicense: apache-2.0\nlibrary_name: peft\nbase_model: " + base["model_id"] +
        "\nbase_model_relation: adapter\ntags:\n- reinforcement-learning\n- code\n- lora\n- multi-agent\n---\n\n"
        "# Local multi-agent software self-play pilot\n\n" + status + "\n\n"
        "All primaries and asynchronous subagents use exactly one shared adapter. The frozen base weights, "
        "vision tower, embeddings and output head are not trained.\n\n"
        f"Exact base revision: `{base['revision']}`. Selected policy version: `{record['policy_version']}`. "
        "Selection uses the final completed collection group, independently of evaluation score.\n\n"
        "## Reproduction\n\n"
        "Use the locked [research harness](https://github.com/BurnyCoder/local-multi-agent-swe-rl). "
        "The adapter, processor, checkpoint metadata, optimizer/RNG state and experiment configuration "
        "are included. Load the exact base revision and this adapter; do not initialize from a preflight checkpoint.\n\n"
        "## Recorded evaluation\n\n```json\n" + json.dumps(report["analysis"], indent=2) + "\n```\n\n"
        "Human and synthetic bugs are separate evaluations. Null results are unscored, not failures or successes. "
        "The small cohort and single training seed do not establish robustness; pretraining contamination is unknown. "
        "See experiment.md and paper.pdf for coverage, paired uncertainty, limitations and negative results.\n\n"
        "## Method and licenses\n\n"
        "This adapts [Self-play SWE-RL](https://arxiv.org/html/2512.18552v3), "
        "[CWM](https://arxiv.org/html/2510.02387v1#A3), and "
        "[OpenCode delegation](https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts). "
        "The base is Apache-2.0 licensed. Training repositories retain their MIT/BSD notices; "
        "the harness is MIT licensed. No held-out reference fix or evaluation test patch enters training.\n")
    (destination / "README.md").write_text(card, encoding="utf-8")
    return destination
