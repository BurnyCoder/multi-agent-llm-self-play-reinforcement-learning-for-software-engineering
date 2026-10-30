"""Replay committed scientific inputs without loading a policy or advancing pins.

Digest-pinned images follow https://docs.docker.com/build/building/best-practices/;
Hub revisions follow https://huggingface.co/docs/datasets/loading. Public manifests
describe the experiment; machine-specific paths and rebuilt image IDs stay local.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from .contracts import RepoSpec
from .io import utc_now, write_json

_MANIFESTS = Path(__file__).resolve().parents[2] / "manifests"
REPOSITORIES = [
    ("more-itertools", "more-itertools/more-itertools", "train", ["more_itertools"], ["tests"], "MIT"),
    ("boltons", "mahmoud/boltons", "train", ["boltons"], ["tests"], "BSD-3-Clause"),
    ("cachetools", "tkem/cachetools", "dev", ["src/cachetools"], ["tests"], "MIT"),
    ("humanize", "python-humanize/humanize", "test", ["src/humanize"], ["tests"], "MIT"),
]


def data_directory() -> Path:
    """An explicit task directory avoids unrelated projects and global model state."""
    return Path(os.environ.get("SSR_DATA_DIR", "data")).resolve()


def command(args: list[str], cwd: Path | None = None) -> str:
    """Use argument lists, not interpolated shell commands, for setup operations."""
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _read_manifest(name: str):
    return json.loads((_MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _environment_fingerprint() -> str:
    return _fingerprint(_read_manifest("environment"))


def _requirements(packages: dict[str, str]) -> str:
    """Require exact versions, including transitive dependencies, in the recipe."""
    for name, version in packages.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
            raise ValueError(f"Dependency is not exactly pinned: {name}={version}")
    return "".join(f"{name}=={version}\n" for name, version in sorted(packages.items()))


def _dependency_recipe() -> tuple[str, str]:
    """Build dependencies only; no clean implementation enters an image layer."""
    record = _read_manifest("environment")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", record["base_image"]):
        raise ValueError("Dependency image base must be pinned by SHA-256 digest")
    dockerfile = (
        f"FROM {record['base_image']}\n"
        "COPY requirements.txt /tmp/requirements.txt\n"
        "RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.txt && python -m pip check\n"
        "ENV PYTHONPATH=/repo:/repo/src PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1\n"
        f"LABEL org.multi-agent-ssr.environment={_environment_fingerprint()}\n"
        "WORKDIR /repo\n"
    )
    return dockerfile, _requirements(record["packages"])


def _image_id(image: str) -> str | None:
    """A missing image permits a rebuild; an unavailable daemon does not."""
    result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                            capture_output=True, text=True, check=False)
    if result.returncode and "No such image" in result.stderr:
        return None
    result.check_returncode()
    return result.stdout.strip()


def _verify_dependency_image(image: str) -> None:
    """Verify the complete installed package inventory against the public recipe."""
    record = _read_manifest("environment")
    script = (
        "import importlib.metadata as m,json,platform; "
        "print(json.dumps({'python':platform.python_version(),'machine':platform.machine(),"
        "'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}}))"
    )
    actual = json.loads(command(["docker", "run", "--rm", "--network", "none", "--read-only",
                                 "--cap-drop", "ALL", "--user", "65534:65534",
                                 "--entrypoint", "python", image, "-c", script]))
    expected = {name.lower().replace("_", "-"): version for name, version in record["packages"].items()}
    if (actual["python"] != record["python_version"] or actual["packages"] != expected
            or (record["platform"] == "linux/amd64" and actual["machine"] != "x86_64")):
        raise ValueError(f"Dependency image differs from committed environment: {image}")


def _prepare_dependency_image(destination: Path, build: bool) -> str:
    """Keep the original image when available; rebuild the pinned recipe elsewhere."""
    record = _read_manifest("environment")
    fingerprint = _environment_fingerprint()
    local_path = destination / "prepared_environment.json"
    candidates = [record["original_image_id"]]
    if local_path.exists():
        local = json.loads(local_path.read_text())
        if local.get("environment_fingerprint") != fingerprint:
            raise ValueError("Local dependency environment differs from committed pins")
        candidates.append(local["image_digest"])
    digest = next((found for candidate in candidates if (found := _image_id(candidate))), None)
    if digest is None:
        if not build:
            raise FileNotFoundError("Pinned dependency image is absent; rerun preparation with build enabled")
        build_dir = destination / "environment"
        build_dir.mkdir(parents=True, exist_ok=True)
        dockerfile, requirements = _dependency_recipe()
        (build_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        (build_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
        subprocess.run(["docker", "build", "--platform", record["platform"],
                        "-t", record["image"], str(build_dir)], check=True)
        digest = _image_id(record["image"])
        if digest is None:
            raise RuntimeError("Docker build produced no dependency image")
    _verify_dependency_image(digest)
    write_json(local_path, {"environment_fingerprint": fingerprint, "image_digest": digest})
    return digest


def _prepare_source(record: RepoSpec, destination: Path) -> RepoSpec:
    """Checkout the frozen commit and preserve upstream license/notice files."""
    source = destination / "repositories" / record.name
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        command(["git", "clone", "--no-checkout", record.url, str(source)])
        # A full clone retains the tags needed by upstream's VCS version hook.
        command(["git", "checkout", "--detach", record.commit], source)
    actual = command(["git", "rev-parse", "HEAD"], source)
    if actual != record.commit:
        raise ValueError(f"Repository {record.name} is at {actual}, not committed pin {record.commit}")
    if command(["git", "status", "--porcelain"], source):
        raise ValueError(f"Prepared source has unexpected local changes: {source}")
    if record.name == "humanize" and not (source / "src/humanize/_version.py").is_file():
        # Let upstream hatch-vcs generate metadata; constrain its isolated build.
        # https://docs.astral.sh/uv/reference/cli/#uv-build--build-constraints
        constraints = destination / "environment" / "build-constraints.txt"
        constraints.parent.mkdir(parents=True, exist_ok=True)
        constraints.write_text(_requirements(_read_manifest("environment")["build_packages"]), encoding="utf-8")
        subprocess.run(["uv", "build", "--build-constraints", str(constraints), "--wheel",
                        "--out-dir", str(destination / "wheels"), str(source)], check=True)
    return replace(record, source_path=str(source))


def prepare_repositories(destination: Path, *, build: bool = True) -> list[RepoSpec]:
    """Reproduce committed source commits and keep local paths out of public pins."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    public_path = _MANIFESTS / "repositories.json"
    if not public_path.exists():
        # This path is only for establishing a new experiment, before preregistration.
        records = []
        for name, repository, split, roots, tests, license_name in REPOSITORIES:
            url = "https://github.com/" + repository
            commit = command(["git", "ls-remote", url, "HEAD"]).split()[0]
            records.append(RepoSpec(name, url, commit, _read_manifest("environment")["image"],
                                    "", split, roots, tests, license_name))
        write_json(public_path, [asdict(record) for record in records])
    public = _read_manifest("repositories")
    digest = _prepare_dependency_image(destination, build)
    records = [replace(_prepare_source(RepoSpec(**row), destination), image_digest=digest) for row in public]
    write_json(destination / "prepared_repositories.json", {
        "pins_fingerprint": _fingerprint(public), "environment_fingerprint": _environment_fingerprint(),
        "repositories": [asdict(record) for record in records]})
    return records


def prepare_evaluation(destination: Path, count: int = 32) -> dict:
    """Reload the exact committed cohort; reference patches stay outside Git."""
    from datasets import load_dataset

    path = _MANIFESTS / "evaluation.json"
    manifest = _read_manifest("evaluation") if path.exists() else None
    if manifest and len(manifest["tasks"]) != count:
        raise ValueError("Requested count differs from the committed evaluation cohort")
    if manifest:
        dataset_id, revision = manifest["dataset"], manifest["revision"]
    else:
        from huggingface_hub import HfApi

        dataset_id = "SWE-bench/SWE-bench_Verified"
        revision = HfApi().dataset_info(dataset_id).sha
    rows = list(load_dataset(dataset_id, revision=revision, split="test"))
    if manifest:
        lookup = {row["instance_id"]: row for row in rows}
        selected = []
        for task in manifest["tasks"]:
            row = lookup.get(task["instance_id"])
            if row is None or any(row.get(key) != value for key, value in task.items()):
                raise ValueError(f"Pinned dataset does not match task {task['instance_id']}")
            selected.append(row)
    else:
        eligible = [row for row in rows if row["repo"] in {
            "psf/requests", "pallets/flask", "pylint-dev/pylint", "pytest-dev/pytest"}]
        selected = sorted(eligible, key=lambda row: hashlib.sha256(row["instance_id"].encode()).hexdigest())[:count]
        if len(selected) != count:
            raise ValueError(f"Expected {count} eligible tasks, found {len(selected)}")
        manifest = {"dataset": dataset_id, "revision": revision,
                    "selection": "sha256(instance_id), ascending; fixed repository eligibility",
                    "created_at": utc_now(), "tasks": [{key: row[key] for key in
                    ("instance_id", "repo", "base_commit", "image")} for row in selected]}
        write_json(path, manifest)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "human_tasks_private.json", selected)
    return manifest


def pin_model() -> dict:
    """Reuse an existing revision; resolve the Hub only for a new experiment."""
    path = _MANIFESTS / "model.json"
    if path.exists():
        return _read_manifest("model")
    from huggingface_hub import HfApi

    info = HfApi().model_info("Qwen/Qwen3.5-2B")
    record = {"model_id": info.id, "revision": info.sha, "license": "Apache-2.0", "pinned_at": utc_now()}
    write_json(path, record)
    return record


def load_repositories() -> list[RepoSpec]:
    """Use validated local relocations without rewriting the public source pins."""
    public = _read_manifest("repositories")
    path = data_directory() / "prepared_repositories.json"
    if not path.exists():
        return [RepoSpec(**row) for row in public]
    local = json.loads(path.read_text(encoding="utf-8"))
    if (local.get("pins_fingerprint") != _fingerprint(public)
            or local.get("environment_fingerprint") != _environment_fingerprint()):
        raise ValueError("Local prepared repositories do not match committed pins; rerun preparation")
    records = local["repositories"]
    if len(records) != len(public) or any(
        any(row[key] != pinned[key] for key in pinned if key not in {"source_path", "image_digest"})
        for row, pinned in zip(records, public, strict=True)
    ):
        raise ValueError("Local repository metadata differs from committed pins")
    return [RepoSpec(**row) for row in records]


def prepare_all() -> dict:
    """Prepare both frozen evaluations before any LLM or preregistration results."""
    from .evaluation import prepare_human_environments
    from .synthetic import prepare_synthetic

    destination = data_directory()
    result = {"repositories": [asdict(x) for x in prepare_repositories(destination)],
              "evaluation": prepare_evaluation(destination), "model": pin_model()}
    result["synthetic"] = prepare_synthetic(destination)
    result["human_environments"] = prepare_human_environments(destination)
    return result
