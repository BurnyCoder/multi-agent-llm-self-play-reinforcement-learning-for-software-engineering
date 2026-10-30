"""Readable phase wrapper for the complete local research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """Import expensive GPU libraries only inside explicitly requested phases."""
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description="Shared-policy multi-agent SSR research harness")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare", help="Pin source/model/evaluation revisions and dependency images")
    subcommands.add_parser("doctor", help="Inspect local prerequisites without model execution")
    for name in ("preflight", "pilot", "evaluate"):
        sub = subcommands.add_parser(name)
        sub.add_argument("--run", type=Path, default=Path("runs/pilot"))
        sub.add_argument("--config", type=Path, default=Path("configs/pilot.toml"))
        sub.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    report = subcommands.add_parser("report")
    report.add_argument("--run", type=Path, default=Path("runs/pilot"))
    report.add_argument("--output", type=Path, default=Path("reports/generated"))
    report.add_argument("--site", type=Path, default=Path("site/dist"))
    export = subcommands.add_parser("export", help="Build a Hub bundle from the final completed checkpoint")
    export.add_argument("--run", type=Path, default=Path("runs/pilot"))
    export.add_argument("--report", type=Path, default=Path("reports/generated"))
    export.add_argument("--output", type=Path, default=Path("runs/pilot/publication"))
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare_all
        print(json.dumps(prepare_all(), indent=2))
    elif args.command == "doctor":
        import subprocess
        for command in (["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], ["docker", "version", "--format", "{{.Server.Version}}"]):
            subprocess.run(command, check=True)
    elif args.command == "report":
        from .reporting import generate_report
        print(json.dumps(generate_report(args.run, args.output, args.site,
                                         readme_path=Path("README.md")), indent=2, default=str))
    elif args.command == "export":
        from .publication import export_checkpoint
        print(export_checkpoint(args.run, args.report, args.output))
    else:
        if args.worker:
            from .experiment import execute
            execute(args.command, args.config, args.run)
        else:
            from .supervisor import bounded_phase
            raise SystemExit(bounded_phase(args.command, args.config, args.run))


if __name__ == "__main__":
    main()
