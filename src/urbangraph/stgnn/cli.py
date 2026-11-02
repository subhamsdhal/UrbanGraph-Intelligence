"""Graph research CLI; all outputs are JSON and failures have nonzero status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.datasets.serialization import read_json
from urbangraph.stgnn.artifacts import verify_run
from urbangraph.stgnn.reporting import compare_runs
from urbangraph.stgnn.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-models")
    train = commands.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dry-run", action="store_true")
    for name in ("evaluate", "verify-run"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", required=True, type=Path)
        command.add_argument("--config", required=True, type=Path)
    compare = commands.add_parser("compare")
    compare.add_argument("--run-dir", action="append", required=True, type=Path)
    compare.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list-models":
            result: dict[str, object] = {
                "models": ["stgcn", "dcrnn", "graph-wavenet", "urbangraphformer"],
                "status": "adapted research architectures",
            }
        elif args.command == "train":
            result = run_experiment(read_json(args.config), dry_run=args.dry_run)
        elif args.command == "compare":
            config = read_json(args.config)
            context_keys = {
                "graph_dir",
                "raw_snapshot_dir",
                "canonical_dir",
                "locations",
                "location_reference",
                "distances",
                "distance_reference",
            }
            if not context_keys.issubset(config) or any(
                not isinstance(config[key], str) for key in context_keys
            ):
                raise DataError("Complete STGNN source context required")
            result = compare_runs(
                args.run_dir,
                graph_context={key: str(config[key]) for key in context_keys},
            )
        else:
            config = read_json(args.config)
            context_keys = {
                "graph_dir",
                "raw_snapshot_dir",
                "canonical_dir",
                "locations",
                "location_reference",
                "distances",
                "distance_reference",
            }
            if not context_keys.issubset(config) or any(
                not isinstance(config[key], str) for key in context_keys
            ):
                raise DataError("Complete STGNN source context required")
            result = verify_run(
                args.run_dir,
                graph_context={key: str(config[key]) for key in context_keys},
            )
            if args.command == "evaluate":
                result["metrics"] = read_json(args.run_dir / "metrics.json")
                result["note"] = (
                    "Integrity-checked saved metrics; runner authorship and independent "
                    "scientific validation not established; no retraining"
                )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (DataError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "status": "INCOMPLETE"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
