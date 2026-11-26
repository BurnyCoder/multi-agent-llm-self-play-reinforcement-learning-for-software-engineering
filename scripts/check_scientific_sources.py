"""Verify cited implementation and notice bytes without importing a model library.

Run inside the locked project environment. GitHub raw blobs and Hugging Face resolve
URLs refer to pinned revisions. The dataset object ID comes from the Hub tree API.
"""

import hashlib
import io
import json
import os
import subprocess
import time
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distribution
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from multi_agent_ssr.evidence import file_sha256
from multi_agent_ssr.grader_calibration import DATASET_SHA256
from multi_agent_ssr.io import write_json
from multi_agent_ssr.scientific_review import PROJECT, REVIEW

SWE = "490635b2e9e775dca1e1d6b40ce9dbcff91e780f"  # Upstream release commit for 5.0.2.
REVIEW_INPUT = "a62534d0f283134c709c73fd8f5df6f9b5dbc811"


def original_materials():
    """Map every file at audit intake to the evidence appropriate to its role."""
    inventory = []
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", REVIEW_INPUT], cwd=PROJECT)
    for entry in tree.decode().split("\0"):
        if not entry:
            continue
        meta, name = entry.split("\t", 1)
        if name.startswith(("docs/licenses/", "reports/generated/licenses/", "site/dist/licenses/")):
            role = "upstream notices, provenance and copies"
        elif name.startswith(("reports/generated/", "site/dist/")):
            role = "derived presentation of archived records"
        elif name.startswith("reports/") and name != "reports/preregistration.md":
            role = "dated historical verification record"
        elif name == "reports/preregistration.md" or name.startswith(("configs/", "manifests/")):
            role = "original design or input identities"
        elif name.startswith(("src/", "scripts/")):
            role = "implemented behavior or historical verification software"
        elif name.startswith("tests/"):
            role = "software fixture or assertion"
        elif name.startswith(("docs/", "site/template/")) or name == "README.md":
            role = "authored scientific and user-facing interpretation"
        else:
            role = "project instructions, license, dependency or operational configuration"
        inventory.append({"path": name, "git_blob": meta.split()[-1], "role": role})
    return inventory


def fetch(url):
    """Retry transient upstream errors and let a persistent error stop verification."""
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "scientific-source-verification"}), timeout=30) as resp:
                return resp.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def compare(item):
    """Independently hash the local file and its upstream bytes."""
    label, path, url = item
    local = file_sha256(path)
    remote = hashlib.sha256(fetch(url)).hexdigest()
    return {"file": label, "source": url, "local_sha256": local, "upstream_sha256": remote, "match": local == remote}


def main():
    load_dotenv(PROJECT / ".env")
    data = Path(os.environ["SSR_DATA_DIR"])
    locked = tomllib.loads((PROJECT / "uv.lock").read_text())["package"]
    versions = {name: {"installed": distribution(name).version,
                       "locked": sorted({r["version"] for r in locked if r["name"] == name})}
                for name in ("swebench", "transformers", "peft", "unsloth", "unsloth-zoo", "torch", "huggingface-hub")}
    if any(v["installed"].split("+")[0] not in {x.split("+")[0] for x in v["locked"]} for v in versions.values()):
        raise ValueError("An audited package version differs from the lock")
    wheel = next(p for p in locked if p["name"] == "swebench")["wheels"][0]
    archive = fetch(wheel["url"])
    if "sha256:" + hashlib.sha256(archive).hexdigest() != wheel["hash"]:
        raise ValueError("SWE-bench wheel differs from the project lock")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        wheel_sources = {name: hashlib.sha256(bundle.read(name)).hexdigest() for name in bundle.namelist()
                         if name.startswith("swebench/") and name.endswith(".py")}
    if any(file_sha256(distribution("swebench").locate_file(k)) != v for k, v in wheel_sources.items()):
        raise ValueError("An installed SWE-bench source file differs from the locked wheel")
    jobs = []
    for package, base, paths in (
        ("swebench", f"https://raw.githubusercontent.com/SWE-bench/SWE-bench/{SWE}/",
         ["swebench/harness/" + f for f in ("grading.py", "run_evaluation.py", "log_parsers/python.py",
                                             "docker_utils.py", "utils.py")]),
        ("transformers", "https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/",
         ["transformers/models/qwen3_5/modeling_qwen3_5.py", "transformers/generation/utils.py"]),
        ("peft", "https://raw.githubusercontent.com/huggingface/peft/v0.21.2/src/", ["peft/auto.py"])):
        jobs.extend((p, distribution(package).locate_file(p), base + p) for p in paths)
    model = json.loads((PROJECT / "manifests/model.json").read_bytes())
    mid, rev = model["model_id"], model["revision"]
    snapshot = data / "huggingface/hub" / ("models--" + mid.replace("/", "--")) / "snapshots" / rev
    for name in ("README.md", "LICENSE", "config.json", "tokenizer_config.json", "chat_template.jinja"):
        jobs.append((mid + "/" + name, snapshot / name, f"https://huggingface.co/{mid}/resolve/{rev}/{name}"))
    notices = PROJECT / "docs/licenses"
    origins = json.loads((notices / "provenance.json").read_bytes())
    for item in origins["files"]:
        for origin in item["origins"]:
            url = origin["source"]
            if url.startswith("https://github.com/"):
                url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/", 1)
            else:
                url = url.replace("/blob/", "/resolve/")
            jobs.append((item["path"], notices / item["path"], url))
    with ThreadPoolExecutor(max_workers=5) as pool:
        checks = list(pool.map(compare, dict.fromkeys(jobs)))
    dataset = origins["dataset"]
    url = f"https://huggingface.co/api/datasets/{dataset['id']}/tree/{dataset['revision']}/data"
    blobs = json.loads(fetch(url))
    expected = next(r["lfs"]["oid"] for r in blobs if r["path"] == "data/test-00000-of-00001.parquet")
    if expected != DATASET_SHA256:
        raise ValueError("Upstream pinned Parquet object differs from the calibration input")
    write_json(REVIEW / "source-checks.json", {"versions": versions, "byte_checks": checks,
               "swebench": {"release_commit": SWE, "wheel": wheel, "installed_source_sha256": wheel_sources},
               "review_input_commit": REVIEW_INPUT, "materials": original_materials(),
               "dataset": {"tree": url, "parquet_sha256": expected}})
    if any(not r["match"] for r in checks):
        raise ValueError("Byte differences require inspection. See source-checks.json.")
    print(f"Verified {len(checks)} local/upstream pairs and the frozen dataset object.")


if __name__ == "__main__":
    main()
