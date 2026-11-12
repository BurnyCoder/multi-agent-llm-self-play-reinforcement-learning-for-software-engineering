"""Build a reviewable Hub upload bundle from the selected completed checkpoint.

Uploading is a separate explicit CLI operation; this module never reads tokens.
Hub model-card metadata: https://huggingface.co/docs/hub/model-cards .
"""
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from .evidence import checkpoint_evidence, intervention_text
from .io import write_json

_HARNESS = "https://github.com/BurnyCoder/multi-agent-llm-self-play-reinforcement-learning-for-software-engineering"
_NOTICE_DIR = Path(__file__).resolve().parents[2] / "docs" / "licenses"
_SOURCE_GUIDE = _NOTICE_DIR.parent / "sources.md"
_PEFT_REVISION_SOURCE = "https://github.com/huggingface/peft/blob/v0.21.2/src/peft/auto.py#L104"
# Match the designated reporting inputs and action records, never a run-wide
# recursive glob. Oracle caches, repository snapshots and host config are absent.
_RECORD_PATTERNS = (
    "manifest.json", "budget.json", "summary.json", "latest.json", "failure.json", "baseline.json",
    "events.jsonl", "model.jsonl", "evaluation.jsonl", "metrics.jsonl", "preflight.json",
    "preparation.json", "preparation_validation.json", "environment_validation.json", "validation.json",
    "protocol_diagnostics.json", "kernel-review.json",
    "preflight/preflight.json", "preflight/likelihood.json", "preflight/likelihood-*.json",
    "preflight/model.jsonl", "preflight/events.jsonl",
    "collection_groups/*.json", "episodes/*.json", "artifact_records/*.json",
    "artifacts/*/bug_inject.diff", "artifacts/*/test_weaken.diff", "artifacts/*/test_files.txt",
    "artifacts/*/test_script.sh", "artifacts/*/test_parser.py",
    "human/*/*/episode.json", "human/*/*/prediction.diff",
    "synthetic/*/*/attempt-*/episode.json", "synthetic/*/*/attempt-*/repair.diff",
    "synthetic/*/*/attempt-*/test_results.json", "synthetic/*/*/attempt-*/agents/events.jsonl",
)


def copy_source_notices(destination: Path) -> list[Path]:
    """Copy only the canonical, deduplicated notice documents, without rewriting bytes."""
    files = []
    for pattern in ("README.md", "provenance.json", "files/*/*"):
        for source in sorted(_NOTICE_DIR.glob(pattern)):
            if not source.is_file() or source.is_symlink():
                continue
            target = destination / source.relative_to(_NOTICE_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files.append(target)
    return files


def _record_paths(run_dir: Path) -> list[Path]:
    """Keep original bytes and reject links, including linked parent directories."""
    paths = sorted({path for pattern in _RECORD_PATTERNS for path in run_dir.glob(pattern)})
    for path in paths:
        relative = path.relative_to(run_dir)
        if any(run_dir.joinpath(*relative.parts[:index]).is_symlink()
               for index in range(1, len(relative.parts) + 1)):
            raise ValueError(f"Evidence record is a symlink: {relative}")
    return [path for path in paths if path.is_file()]


def _source_commit() -> str:
    """Identify the reporting/export source separately from the pilot's code."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unrecorded"


def _copy_source_guide(destination: Path, source_commit: str) -> list[Path]:
    """Refresh authored guidance while preserving all upstream notice entries."""
    guide = destination / "docs/sources.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_SOURCE_GUIDE, guide)
    provenance = destination / "license-provenance.json"
    if provenance.exists():
        record = json.loads(provenance.read_text())
        entries = record.setdefault("files", [])
        entry = next((row for row in entries if row["path"] == "docs/sources.md"), None)
        if entry is None:
            entry = {"path": "docs/sources.md"}
            entries.append(entry)
        entry.update(source=f"{_HARNESS}/blob/{source_commit}/docs/sources.md",
                     local_source="${PROJECT}/docs/sources.md", revision=source_commit,
                     license_or_notice="Source attribution and upstream license references",
                     bytes=guide.stat().st_size, sha256=hashlib.sha256(guide.read_bytes()).hexdigest())
        if "benchmark_source_notices" in record:
            notice = destination / "licenses/provenance.json"
            record["benchmark_source_notices"].update(
                path="licenses/provenance.json", sha256=hashlib.sha256(notice.read_bytes()).hexdigest())
        record["report_source_commit"] = source_commit
        write_json(provenance, record)
    return [guide]


def _inventory(destination: Path, files: list[Path], source_commit: str) -> None:
    """Hash bounded exported files without loading large weights into memory.

    https://docs.python.org/3.12/library/hashlib.html#hashlib.file_digest
    """
    entries = []
    for path in sorted(set(files)):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append({"path": path.relative_to(destination).as_posix(),
                        "bytes": path.stat().st_size, "sha256": digest})
    write_json(destination / "bundle-inventory.json", {
        "format_version": 1, "export_source_commit": source_commit, "files": entries,
        "scope": "Listed exported files; this inventory excludes its own bytes."})


def _pin_exported_adapter_revision(checkpoint: Path, destination: Path, revision: str) -> list[Path]:
    """PEFT's Auto loader forwards the adapter config revision to the base model.

    The official lookup is cited in _PEFT_REVISION_SOURCE; do not change the
    saved research checkpoint to repair publication metadata.
    """
    original = (checkpoint / "adapter_config.json").read_bytes()
    config = json.loads(original)
    previous = config.get("revision")
    original_copy = destination / "source-metadata" / "adapter_config.json"
    original_copy.parent.mkdir(parents=True, exist_ok=True)
    original_copy.write_bytes(original)
    config["revision"] = revision
    exported = destination / "adapter_config.json"
    write_json(exported, config)
    provenance = destination / "export-transformations.json"
    write_json(provenance, {"format_version": 1, "transformations": [{
        "path": "adapter_config.json", "original_copy": "source-metadata/adapter_config.json",
        "field": "revision", "previous_value": previous, "exported_value": revision,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "exported_sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
        "reason": "Pin standalone PEFT base loading to checkpoint.model_config.revision.",
        "serialization": "Harness JSON writer; all other configuration values are preserved.",
        "reference": _PEFT_REVISION_SOURCE,
    }]})
    return [original_copy, provenance]


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
    evidence = checkpoint_evidence(run_dir)
    if report.get("checkpoint_evidence") and report["checkpoint_evidence"] != evidence:
        raise ValueError("Report checkpoint evidence differs from the selected run; regenerate the report")
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise ValueError("Selected adapter weights are missing")
    records = _record_paths(run_dir)
    source_commit = _source_commit()
    files = copy_source_notices(destination / "licenses")
    destination.mkdir(parents=True, exist_ok=True)
    files.extend(_copy_source_guide(destination, source_commit))
    # Checkpoints are harness-owned; exclude arbitrary extensions and symlinks.
    allowed = {".json", ".safetensors", ".pt", ".jinja", ".txt"}
    for path in checkpoint.iterdir():
        if path.is_file() and not path.is_symlink() and path.suffix in allowed:
            shutil.copyfile(path, destination / path.name)
            files.append(destination / path.name)
    files.extend(_pin_exported_adapter_revision(checkpoint, destination, record["model_config"]["revision"]))
    write_json(destination / "experiment.json", manifest)
    write_json(destination / "evaluation-summary.json", report["analysis"])
    write_json(destination / "selection.json", latest)
    # Preserve the generated report's local analysis/preregistration links.
    write_json(destination / "summary.json", report)
    files.extend(destination / name for name in
                 ("experiment.json", "evaluation-summary.json", "selection.json", "summary.json"))
    # The original basename preserves reporting's path-derived run_id fallback.
    records_relative = Path("run-records") / run_dir.name
    for path in records:
        target = destination / records_relative / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files.append(target)
    # Derived hash evidence travels separately; original run files remain intact.
    evidence_path = destination / records_relative / "checkpoint-evidence.json"
    write_json(evidence_path, evidence)
    files.append(evidence_path)
    for name in ["experiment.md", "experiments-summary.md", "paper.md", "paper.pdf", "preregistration.md",
                 "evidence-audit.json", "independent-audit-2026-11-12.md"]:
        if (report_dir / name).exists():
            shutil.copyfile(report_dir / name, destination / name)
            files.append(destination / name)
    base = record["model_config"]
    aggregate_analysis = {key: value for key, value in report["analysis"].items() if key != "records"}
    status = ("An unchanged research adapter, verified against the initial saved adapter. This is not a trained checkpoint."
              if evidence["status"] == "unchanged" else
              "A research adapter whose saved bytes differ from initialization; this alone does not establish improvement."
              if evidence["status"] == "changed" else
              "A selected research adapter; initial-to-final identity is unverified.")
    selection = ("Selection uses the final completed collection group, independently of evaluation score. "
                 "In the saved analysis, `trained` identifies the final-evaluation condition, not evidence of an optimizer update. "
                 + intervention_text(evidence, report.get("diagnostics", {}).get("optimizer_steps")))
    pilot_commit = manifest.get("phase_code_commits", {}).get("pilot", manifest.get("code_commit", "unrecorded"))
    # License files may already be staged from pinned upstream snapshots; keep
    # their original names and provenance, and never infer files from log paths.
    for pattern in ("LICENSE", "license-provenance.json", "docs/sources.md", "licenses/harness/LICENSE",
                    "licenses/repositories/*/LICENSE", "licenses/repositories/*/LICENCE"):
        files.extend(path for path in destination.glob(pattern) if path.is_file() and not path.is_symlink())
    card = ("---\nlicense: apache-2.0\nlibrary_name: peft\nbase_model: " + base["model_id"] +
        "\nbase_model_relation: adapter\ntags:\n- reinforcement-learning\n- code\n- lora\n- multi-agent\n---\n\n"
        "# Local multi-agent software self-play pilot\n\n" + status + "\n\n"
        "All primaries and asynchronous subagents use exactly one shared adapter. The frozen base weights, "
        "vision tower, embeddings and output head are not trained.\n\n"
        f"Exact base revision: `{base['revision']}`. Selected policy version: `{record['policy_version']}`. "
        f"Adapter fingerprint: `{record['adapter_fingerprint']}`. " + selection + "\n\n"
        f"Recorded completed collection groups: `{latest.get('completed_groups', 'unrecorded')}`. "
        f"Saved report status: `{report.get('status', 'unrecorded')}`. "
        f"Pilot source commit: `{pilot_commit}`. Report/export source commit: `{source_commit}`.\n\n"
        "## Reproduction\n\n"
        f"Use the locked [research harness]({_HARNESS}/tree/{source_commit}) and its documented GPU environment for model loading. "
        "The adapter, processor, checkpoint metadata, optimizer/RNG state and experiment configuration "
        "are included. The harness loader reads the exact base revision and verifies the adapter fingerprint:\n\n"
        "```python\nimport json\nfrom pathlib import Path\n"
        "from multi_agent_ssr.contracts import ModelConfig\nfrom multi_agent_ssr.model import LocalPolicy\n\n"
        'bundle = Path("/path/to/downloaded-bundle")\n'
        'record = json.loads((bundle / "checkpoint.json").read_text())\n'
        'policy = LocalPolicy(ModelConfig(**record["model_config"]),\n'
        '                     log_path=Path("reload-model.jsonl"), checkpoint=bundle)\n```\n\n'
        f"The export pins `adapter_config.json`'s `revision` for [PEFT's base-model lookup]({_PEFT_REVISION_SOURCE}). "
        "`source-metadata/adapter_config.json` preserves the original saved bytes, and `export-transformations.json` "
        "records the metadata change and before/after hashes. The source checkpoint and adapter weights are unchanged.\n\n"
        f"The harness's [generation configuration]({_HARNESS}/blob/{source_commit}/src/multi_agent_ssr/model.py) "
        "unions the base model's EOS IDs with the tokenizer's chat EOS and deduplicates them; prompt encoding disables thinking. "
        "Reusing only the base generation defaults can continue past "
        "a chat turn; retain the harness configuration when sampling.\n\n"
        "The allowlisted raw records retain exact prompt tokens, generated action tokens, behavior log probabilities, "
        "policy versions and parent identities. Stored paths are unchanged provenance metadata, not instructions "
        "to fetch excluded files. This bundle excludes repository snapshots, official scorer caches and reference fixes. "
        "It supports report regeneration and inspection of recorded actions; rerunning sampling is not guaranteed "
        "to reproduce actions bit for bit. `bundle-inventory.json` contains byte counts and SHA256 hashes.\n\n"
        "From a checkout at the report/export source commit, regenerate all tables, figures, paper and dashboard "
        "without a model call (the report command also refreshes that checkout's generated README results block):\n\n"
        '```bash\nBUNDLE=/path/to/downloaded-bundle\n'
        f'uv run --no-sync selfplay report --run "$BUNDLE"/{shlex.quote(records_relative.as_posix())} '
        '--output "$BUNDLE/regenerated-reports" --site "$BUNDLE/regenerated-site"\n```\n\n'
        "## Recorded evaluation\n\n"
        + report.get("oracle_disclosure", "") + "\n\n"
        "Aggregate results are shown below. [Complete evaluation records](evaluation-summary.json) "
        "contain the full saved analysis, including task-level outcomes and official reports.\n\n```json\n"
        + json.dumps(aggregate_analysis, indent=2) + "\n```\n\n"
        "Human and synthetic bugs are separate evaluations. Null results are unscored, not failures or successes. "
        "The small cohort and single training seed do not establish robustness; pretraining contamination is unknown. "
        "See experiment.md and paper.pdf for coverage, paired uncertainty, limitations and negative results.\n\n"
        "The [independent correction ledger](independent-audit-2026-11-12.md) and "
        "[raw-record cross-check](evidence-audit.json) distinguish observed execution from design. "
        "Original scorer-cache agreement in that audit refers to the local source run; the caches are excluded from this export.\n\n"
        "## Method and licenses\n\n"
        "This adapts [Self-play SWE-RL](https://arxiv.org/html/2512.18552v3), "
        "[CWM](https://arxiv.org/html/2510.02387v1#A3), and "
        "[OpenCode delegation](https://github.com/anomalyco/opencode/blob/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts). "
        f"The [pinned base license](https://huggingface.co/{base['model_id']}/blob/{base['revision']}/LICENSE) is Apache-2.0. "
        "Training repositories retain their MIT/BSD notices; "
        f"the [harness license]({_HARNESS}/blob/{source_commit}/LICENSE) is MIT. "
        "Benchmark repository source excerpts in raw model/tool records have separate "
        "[upstream notices and pinned source associations](licenses/README.md). "
        "These include notices from requests, Flask, pylint and pytest; the cached dataset card's missing "
        "license declaration is documented in that provenance. "
        "When present, `license-provenance.json` identifies the exact bundled license/notice files and their source revisions. "
        "No held-out reference fix or evaluation test patch enters training.\n")
    (destination / "README.md").write_text(card, encoding="utf-8")
    files.append(destination / "README.md")
    _inventory(destination, files, source_commit)
    return destination
