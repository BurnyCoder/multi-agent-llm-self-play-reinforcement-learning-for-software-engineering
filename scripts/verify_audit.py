"""Reproduce dated source, file-coverage and immutable-record checks without a model.

Run from the checkout in its WSL environment. Network reads retrieve original
sources; no package installation, model execution or raw-record rewrite occurs.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import tomllib
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(item):
    """An inaccessible source is reported explicitly, not silently verified."""
    name, url, expected = item
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        result = {"url": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if expected:
            result.update(expected_sha256=expected, matches=result["sha256"] == expected)
        if url.endswith("/json"):
            info = json.loads(content)["info"]
            result["declared_license"] = info.get("license_expression") or info.get("license")
        return name, result
    except Exception as error:
        return name, {"url": url, "verification": "unknown", "error": str(error)}


def sources():
    """Compare installed API sources and exact upstream notices, not prior audits."""
    locked = {row["name"]: row["version"] for row in tomllib.loads(Path("uv.lock").read_text())["package"] if "version" in row}
    packages = ("torch", "transformers", "peft", "unsloth", "unsloth-zoo", "swebench",
                "huggingface-hub", "numpy", "reportlab", "pypdf", "docker", "datasets")
    versions = {name: {"installed": metadata.version(name), "locked": locked[name],
                       "matches": metadata.version(name) == locked[name]} for name in packages}
    specs = [("grading", "swebench", "swebench/harness/grading.py",
              "https://raw.githubusercontent.com/SWE-bench/SWE-bench/ca6e4e0d252f32f8762625b73575d5dee49d0a5a/swebench/harness/grading.py"),
             ("qwen_api", "transformers", "transformers/models/qwen3_5/modeling_qwen3_5.py",
              "https://raw.githubusercontent.com/huggingface/transformers/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py"),
             ("peft_loading", "peft", "peft/auto.py",
              "https://raw.githubusercontent.com/huggingface/peft/v0.21.2/src/peft/auto.py")]
    jobs = [(name, url, digest(metadata.distribution(package).locate_file(path))) for name, package, path, url in specs]
    jobs += [("qwen_license", "https://huggingface.co/Qwen/Qwen3.5-2B/resolve/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE", None),
             ("unsloth_license", "https://pypi.org/pypi/unsloth/2026.10.10/json", None),
             ("zoo_license", "https://pypi.org/pypi/unsloth-zoo/2026.10.10/json", None),
             ("ssr_paper", "https://arxiv.org/html/2512.18552v3", None),
             ("cwm_paper", "https://arxiv.org/html/2510.02387v1", None),
             ("asp_paper", "https://arxiv.org/html/2607.03523v1", None),
             ("opencode_delegation", "https://raw.githubusercontent.com/anomalyco/opencode/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/packages/core/src/tool/plugin/subagent.ts", None),
             ("opencode_license", "https://raw.githubusercontent.com/anomalyco/opencode/e761e265d2d5e6ff4a1c0756ab8d8eeee0db501e/LICENSE", None)]
    for row in json.loads(Path("manifests/repositories.json").read_text()):
        filename = "LICENCE" if row["name"] == "humanize" else "LICENSE"
        url = row["url"].replace("https://github.com/", "https://raw.githubusercontent.com/")
        jobs.append((row["name"] + "_license", f"{url}/{row['commit']}/{filename}", None))
    notices = json.loads(Path("docs/licenses/provenance.json").read_text())
    local_notices = []
    for row in notices["files"]:
        path = Path("docs/licenses") / row["path"]
        local_notices.append({"path": str(path), "matches": digest(path) == row["sha256"]})
        for origin in row["origins"]:
            url = origin["source"]
            if "github.com/" in url:
                url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
            elif "huggingface.co/" in url:
                url = url.replace("/blob/", "/resolve/")
            jobs.append((origin["source"], url, row["sha256"]))
    # Deduplicate repeated source associations without dropping any unique origin.
    jobs = list({name: (name, url, expected) for name, url, expected in jobs}.values())
    with ThreadPoolExecutor(max_workers=8) as executor:
        retrieved = dict(executor.map(fetch, jobs))
    return {"method": "installed metadata versus uv.lock; upstream-byte comparison; original notices",
            "installed_versions": versions, "sources": retrieved, "local_notice_hashes": local_notices}


def coverage():
    """Map all tracked/untracked project surfaces to ledger claim families."""
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()
    rows = []
    for path in sorted(set(paths)):
        if "/licenses/" in path or path == "LICENSE":
            kind, claims = "upstream attribution or preserved notice", ["C27", "C28"]
        elif path.startswith("tests/"):
            kind, claims = "software test or fixture, not research measurement", ["C38"]
        elif path.startswith("configs/") or path.startswith("manifests/") or path == ".env.example":
            kind, claims = "frozen design and provenance", ["C21", "C22", "C23", "C24", "C26", "C30"]
        elif path.endswith("preregistration.md") or path == "reports/delivery-verification.json":
            kind, claims = "historical protocol or prior statement; not independent audit evidence", ["C37"]
        elif path.startswith("site/"):
            kind, claims = "UI/template or generated copy", ["C01", "C03", "C04", "C07", "C08", "C16", "C33", "C34", "C35"]
        elif path.startswith("src/"):
            kind, claims = "implementation contract, comments and canonical generators", [f"C{n:02d}" for n in range(1, 39)]
        elif path.startswith("reports/generated/"):
            kind, claims = "generated evidence and narrative", [f"C{n:02d}" for n in range(1, 39)]
        elif path.startswith(".github/") or path in {"pyproject.toml", "uv.lock", ".gitignore"}:
            kind, claims = "dependency or deployment configuration", ["C30", "C38"]
        else:
            kind, claims = "authored documentation or audit provenance", [f"C{n:02d}" for n in range(1, 39)]
        rows.append({"path": path, "classification": kind, "ledger_entries": claims})
    return {"scope": "file-level coverage map; claim-level verification and limitations are in the ledger", "files": rows}


def verification(bundle, regenerated, junit):
    """Check publication round trips and parse test results, never transcribe counts."""
    report = json.loads(Path("reports/generated/summary.json").read_text())
    audit = json.loads(Path("reports/generated/evidence-audit.json").read_text())
    restored = json.loads((regenerated / "summary.json").read_text())
    checks = {"exported_summary_matches": json.loads((bundle / "summary.json").read_text()) == report}
    checks["regenerated_summary_matches_except_timestamp"] = (
        {k: v for k, v in restored.items() if k != "generated_at"}
        == {k: v for k, v in report.items() if k != "generated_at"})
    for source, conditions in report["analysis"]["conditions"].items():
        for condition, row in conditions.items():
            independently = audit["conditions"][source][condition]
            for key, independent_key in (("completed", "scored"), ("successes", "resolved"),
                                          ("unscored", "unscored"), ("recorded_tasks", "recorded_tasks")):
                checks[f"raw_{source}_{condition}_{key}"] = row[key] == independently[independent_key]
    for key, independent_key in (("generated_action_tokens", "action_tokens"), ("generation_calls", "action_calls"),
                                  ("child_sessions", "child_sessions"), ("delegation_calls", "delegation_calls"),
                                  ("optimizer_steps", "optimizer_steps"), ("valid_artifacts", "valid_artifacts"),
                                  ("artifact_validation_checks", "artifact_validation_checks")):
        checks[f"raw_{key}"] = report["diagnostics"][key] == audit[independent_key]
    inventory = json.loads((bundle / "bundle-inventory.json").read_text())
    checks["bundle_file_hashes"] = all(digest(bundle / row["path"]) == row["sha256"]
                                     and (bundle / row["path"]).stat().st_size == row["bytes"] for row in inventory["files"])
    for row in json.loads((bundle / "license-provenance.json").read_text()).get("files", []):
        checks[f"license_provenance_{row['path']}"] = digest(bundle / row["path"]) == row["sha256"]
    for name in ("summary.json", "paper.pdf", "paper.md", "experiment.md", "experiments-summary.md", "evidence-audit.json"):
        checks[f"site_copy_{name}"] = digest(Path("reports/generated") / name) == digest(Path("site/dist") / name)
    checks["site_data"] = digest(Path("site/dist/data.json")) == digest(Path("reports/generated/summary.json"))
    tests = []
    for path in junit:
        suites = ET.parse(path).getroot().findall("testsuite")
        tests.append({"path": str(path), "sha256": digest(path),
                      **{key: sum(int(suite.attrib[key]) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}})
    checks["software_tests_pass"] = all(row["tests"] and not row["failures"] and not row["errors"] for row in tests)
    return {"scope": "CPU software checks and saved-evidence publication; no new model run",
            "tests": tests, "checks": checks, "all_checks_pass": all(checks.values()),
            "bundle_files": len(inventory["files"]), "export_source_commit": inventory["export_source_commit"],
            "manual_visual_review": "All rendered PDF pages and empirical/design figures inspected; dashboard filters, child traces, empty patches and absent-evidence fixture exercised in Chrome. No console errors observed.",
            "limitations": "Exported grader caches are absent by design. Original cache comparisons are preserved observations. Tests and visual inspection do not certify absence of all defects."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-11-12")
    parser.add_argument("--sources", action="store_true")
    parser.add_argument("--originals", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--regenerated", type=Path)
    parser.add_argument("--junit", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.sources:
        save(Path(f"reports/source-verification-{args.date}.json"), sources())
    if args.bundle:
        if not args.regenerated or not args.junit:
            parser.error("--bundle requires --regenerated and --junit")
        result = verification(args.bundle, args.regenerated, args.junit)
        save(Path(f"reports/audit-verification-{args.date}.json"), result)
        if not result["all_checks_pass"]:
            raise SystemExit("Publication cross-check failed; inspect the audit verification report")
    save(Path(f"reports/claim-coverage-{args.date}.json"), coverage())
    if args.originals:
        original = json.loads(args.originals.read_text())
        checked = {name: Path(name).is_file() and digest(Path(name)) == row["sha256"] for name, row in original["files"].items()}
        save(Path(f"reports/immutable-evidence-{args.date}.json"),
             {"baseline": original, "files_checked": len(checked), "all_unchanged": all(checked.values()),
              "mismatches": [name for name, passed in checked.items() if not passed]})
        if not all(checked.values()):
            raise SystemExit("Original evidence changed; inspect the immutable-evidence report")


if __name__ == "__main__":
    main()
