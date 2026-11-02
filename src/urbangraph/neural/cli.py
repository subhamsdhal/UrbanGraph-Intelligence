"""Local bounded neural experiments; explicit configs and machine-readable output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.data.saved_use import authorized_saved_lineage
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.artifacts import load_plan
from urbangraph.logging import configure_logging
from urbangraph.neural.artifacts import verify_run
from urbangraph.neural.contracts import config_from_dict
from urbangraph.neural.reporting import compare_runs
from urbangraph.neural.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UrbanGraph non-graph temporal neural research")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-models")
    for name in ("inspect-config", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name == "train":
            sub.add_argument("--dry-run", action="store_true")
    for name in ("verify-run", "evaluate"):
        sub = commands.add_parser(name, help="Verify weights and inspect saved held-out metrics")
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--raw-snapshot-dir", type=Path, required=True)
        sub.add_argument("--canonical-dir", type=Path, required=True)
    sub = commands.add_parser("compare")
    sub.add_argument("--run-dir", type=Path, action="append", required=True)
    sub.add_argument("--raw-snapshot-dir", type=Path, required=True)
    sub.add_argument("--canonical-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_logging()
    try:
        result: object
        if args.command == "list-models":
            result = {"models": ["gru", "lstm", "transformer"], "spatial_mixing": False}
        elif args.command in {"verify-run", "evaluate"}:
            if args.command == "evaluate":
                authorized_saved_lineage(
                    args.run_dir,
                    raw_snapshot_dir=args.raw_snapshot_dir,
                    canonical_dir=args.canonical_dir,
                    purpose="scientific_evaluation",
                    operation="model_evaluate",
                )
            result = verify_run(
                args.run_dir,
                raw_snapshot_dir=args.raw_snapshot_dir,
                canonical_dir=args.canonical_dir,
            )
            if args.command == "evaluate":
                result = {
                    "verification": result,
                    "saved_evaluation": read_json(args.run_dir / "metrics.json"),
                }
        elif args.command == "compare":
            result = compare_runs(
                args.run_dir,
                raw_snapshot_dir=args.raw_snapshot_dir,
                canonical_dir=args.canonical_dir,
            )
        else:
            suite = read_json(args.config)
            if set(suite) != {
                "canonical_dir",
                "raw_snapshot_dir",
                "plan_dir",
                "output_root",
                "model",
            }:
                raise DataError("Invalid neural experiment configuration")
            if any(
                not isinstance(suite[k], str)
                for k in ("canonical_dir", "raw_snapshot_dir", "plan_dir", "output_root")
            ):
                raise DataError("Experiment paths must be strings")
            config = config_from_dict(suite["model"])
            if args.command == "inspect-config":
                result = suite | {"model": config.to_dict()}
            elif args.dry_run:
                dataset, split, _ = load_plan(
                    str(suite["plan_dir"]),
                    canonical_dir=str(suite["canonical_dir"]),
                    raw_snapshot_dir=str(suite["raw_snapshot_dir"]),
                )
                result = {
                    "valid": True,
                    "training_executed": False,
                    "config": config.to_dict(),
                    "folds": len(split.folds),
                    "sensors": len(dataset.sensor_ids),
                }
            else:
                path = run_experiment(
                    str(suite["canonical_dir"]),
                    str(suite["plan_dir"]),
                    config,
                    raw_snapshot_dir=str(suite["raw_snapshot_dir"]),
                    output_root=str(suite["output_root"]),
                )
                result = {
                    "run_dir": str(path),
                    "verification": verify_run(
                        path,
                        raw_snapshot_dir=str(suite["raw_snapshot_dir"]),
                        canonical_dir=str(suite["canonical_dir"]),
                    ),
                }
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (DataError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
