"""Preparation must replay committed scientific inputs on another machine."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_agent_ssr import prepare
from multi_agent_ssr.contracts import RepoSpec


@pytest.fixture
def manifests(tmp_path, monkeypatch):
    directory = tmp_path / "manifests"
    directory.mkdir()
    monkeypatch.setattr(prepare, "_MANIFESTS", directory, raising=False)
    monkeypatch.setattr(prepare, "data_directory", lambda: tmp_path / "data")
    return directory


def _git(directory, *arguments):
    return subprocess.run(["git", "-C", str(directory), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_fresh_clone_uses_committed_revision_and_preserves_notices(manifests, tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init")
    (origin / "module.py").write_text("value = 1\n")
    (origin / "LICENSE").write_text("Upstream copyright and license notice\n")
    _git(origin, "add", ".")
    _git(origin, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "pinned")
    pinned = _git(origin, "rev-parse", "HEAD")
    (origin / "module.py").write_text("value = 2\n")
    _git(origin, "add", ".")
    _git(origin, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "later")
    later = _git(origin, "rev-parse", "HEAD")
    record = RepoSpec("tiny", str(origin), pinned, "image", "/old/machine/tiny",
                      "train", ["module.py"], ["tests"], "MIT", "old-image")
    public = manifests / "repositories.json"
    public.write_text(json.dumps([asdict(record)]))
    original_bytes = public.read_bytes()
    monkeypatch.setattr(prepare, "_prepare_dependency_image", lambda destination, build: "local-equivalent-image", raising=False)
    monkeypatch.setattr(prepare, "_environment_fingerprint", lambda: "fixed-environment", raising=False)
    destination = tmp_path / "data"
    records = prepare.prepare_repositories(destination)
    clone = Path(records[0].source_path)
    assert _git(clone, "rev-parse", "HEAD") == pinned
    assert (clone / "module.py").read_text() == "value = 1\n"
    assert (clone / "LICENSE").read_bytes() == (origin / "LICENSE").read_bytes()
    assert public.read_bytes() == original_bytes
    assert prepare.load_repositories()[0].image_digest == "local-equivalent-image"
    _git(clone, "checkout", "--detach", later)
    with pytest.raises(ValueError, match="pin|commit|revision"):
        prepare.prepare_repositories(destination)
    assert public.read_bytes() == original_bytes


def test_model_pin_is_reused_without_resolving_latest(manifests, monkeypatch):
    record = {"model_id": "Qwen/Qwen3.5-2B", "revision": "a" * 40, "pinned_at": "original"}
    path = manifests / "model.json"
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    def unexpected():
        raise AssertionError("Existing pins must not ask the Hub for latest")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=unexpected))
    assert prepare.pin_model() == record
    assert path.read_bytes() == before


def test_dataset_pin_and_cohort_are_reused_exactly(manifests, tmp_path, monkeypatch):
    rows = [{"instance_id": "chosen", "repo": "psf/requests", "base_commit": "b" * 40,
             "image": "image:tag", "patch": "PRIVATE GOLD", "test_patch": "PRIVATE TEST"},
            {"instance_id": "newer", "repo": "psf/requests", "base_commit": "c" * 40,
             "image": "other:tag", "patch": "OTHER GOLD"}]
    public_task = {key: rows[0][key] for key in ("instance_id", "repo", "base_commit", "image")}
    record = {"dataset": "SWE-bench/SWE-bench_Verified", "revision": "d" * 40,
              "created_at": "original", "tasks": [public_task]}
    path = manifests / "evaluation.json"
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    calls = []
    def load_dataset(dataset_id, **kwargs):
        calls.append((dataset_id, kwargs))
        return rows
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: (_ for _ in ()).throw(AssertionError("Do not resolve latest"))))
    destination = tmp_path / "data"
    assert prepare.prepare_evaluation(destination, count=1) == record
    assert calls == [(record["dataset"], {"revision": "d" * 40, "split": "test"})]
    assert json.loads((destination / "human_tasks_private.json").read_text()) == rows[:1]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="count|cohort"):
        prepare.prepare_evaluation(destination, count=2)


def test_dependency_recipe_has_no_floating_base_or_package_versions(manifests):
    record = {"image": "fixture", "base_image": "python:3.12-slim@sha256:" + "a" * 64,
              "python_version": "3.12.16", "platform": "linux/amd64",
              "packages": {"pip": "25.0.1", "pytest": "9.2.0"}, "original_image_id": "sha256:" + "b" * 64}
    (manifests / "environment.json").write_text(json.dumps(record))
    dockerfile, requirements = prepare._dependency_recipe()
    assert "FROM python:3.12-slim@sha256:" + "a" * 64 in dockerfile
    assert requirements == "pip==25.0.1\npytest==9.2.0\n"
    assert "--no-deps" in dockerfile


def test_prepare_all_runs_both_frozen_eval_preparations(manifests, tmp_path, monkeypatch):
    from multi_agent_ssr import evaluation, synthetic

    calls = []
    monkeypatch.setattr(prepare, "prepare_repositories", lambda destination: calls.append("repositories") or [])
    monkeypatch.setattr(prepare, "prepare_evaluation", lambda destination: calls.append("evaluation") or {})
    monkeypatch.setattr(prepare, "pin_model", lambda: calls.append("model") or {})
    monkeypatch.setattr(synthetic, "prepare_synthetic", lambda destination: calls.append("synthetic") or [])
    monkeypatch.setattr(evaluation, "prepare_human_environments", lambda destination: calls.append("human_environments") or {})
    result = prepare.prepare_all()
    assert calls == ["repositories", "evaluation", "model", "synthetic", "human_environments"]
    assert {"synthetic", "human_environments"} <= result.keys()


def test_existing_original_image_is_verified_and_reused_without_build(manifests, tmp_path, monkeypatch):
    record = {"original_image_id": "original-image", "packages": {"pytest": "9.2.0"}}
    (manifests / "environment.json").write_text(json.dumps(record))
    calls = []
    monkeypatch.setattr(prepare, "_image_id", lambda image: calls.append(("inspect", image)) or image)
    monkeypatch.setattr(prepare, "_verify_dependency_image", lambda image: calls.append(("verify", image)))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not rebuild existing image"))
    assert prepare._prepare_dependency_image(tmp_path, build=True) == "original-image"
    assert calls == [("inspect", "original-image"), ("verify", "original-image")]
    local = json.loads((tmp_path / "prepared_environment.json").read_text())
    assert local["image_digest"] == "original-image"
    assert local["environment_fingerprint"] == prepare._environment_fingerprint()


def test_changed_dependency_inventory_is_rejected(manifests, monkeypatch):
    record = {"python_version": "3.12.16", "platform": "linux/amd64", "packages": {"pytest": "9.2.0"}}
    (manifests / "environment.json").write_text(json.dumps(record))
    actual = {"python": "3.12.16", "machine": "x86_64", "packages": {"pytest": "different-version"}}
    monkeypatch.setattr(prepare, "command", lambda args: json.dumps(actual))
    with pytest.raises(ValueError, match="differs"):
        prepare._verify_dependency_image("changed-image")


def test_generated_version_cannot_drift_with_future_upstream_tags(manifests, tmp_path, monkeypatch):
    source = tmp_path / "repositories/humanize"
    version = source / "src/humanize/_version.py"
    version.parent.mkdir(parents=True)
    original = b"version = '4.16.1.dev42'\n"
    version.write_bytes(original)
    (manifests / "environment.json").write_text(json.dumps({
        "humanize_version": "4.16.1.dev42",
        "humanize_version_file_sha256": hashlib.sha256(original).hexdigest()}))
    spec = RepoSpec("humanize", "url", "pinned", "image", "", "test", ["src/humanize"], ["tests"])
    monkeypatch.setattr(prepare, "command", lambda args, cwd: "pinned" if "rev-parse" in args else "")
    assert prepare._prepare_source(spec, tmp_path).source_path == str(source)
    version.write_text("version = 'future-version'\n")
    with pytest.raises(ValueError, match="version|metadata"):
        prepare._prepare_source(spec, tmp_path)
