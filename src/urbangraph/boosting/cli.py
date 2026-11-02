"""Local CPU boosting commands; no network or service requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urbangraph.baselines.registry import list_models
from urbangraph.boosting.artifacts import verify_run
from urbangraph.boosting.contracts import config_from_dict
from urbangraph.boosting.reporting import compare_runs
from urbangraph.boosting.runner import run_experiment
from urbangraph.boosting.sampling import select_rows
from urbangraph.data import DataError
from urbangraph.data.saved_use import authorized_saved_lineage
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation.artifacts import load_plan
from urbangraph.evaluation.windows import numerical_values
from urbangraph.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UrbanGraph bounded CPU boosting")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list-models")
    inspect = commands.add_parser("inspect-config")
    inspect.add_argument("--config", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--dry-run", action="store_true")
    for command in ("verify-run", "evaluate"):
        sub = commands.add_parser(
            command, help="Verify saved native models and report held-out metrics"
        )
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--raw-snapshot-dir", type=Path, required=True)
        sub.add_argument("--canonical-dir", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--run-dir", type=Path, action="append", required=True)
    compare.add_argument("--raw-snapshot-dir", type=Path, required=True)
    compare.add_argument("--canonical-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_logging()
    try:
        result: object
        if args.command == "list-models":
            result = list_models(include_boosting=True)
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
                raise DataError("Invalid experiment configuration")
            for name in ("canonical_dir", "raw_snapshot_dir", "plan_dir", "output_root"):
                if not isinstance(suite[name], str):
                    raise DataError("Experiment paths must be strings")
            config = config_from_dict(object_dict(suite["model"]))
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
                    "config": config.to_dict(),
                    "sensors": len(dataset.sensor_ids),
                    "folds": len(split.folds),
                    "training_executed": False,
                    "sampling_preview": {
                        f.fold_id: {
                            partition: select_rows(
                                dataset,
                                split,
                                i,
                                partition,
                                maximum,
                                config.seed + (partition == "validation"),
                                numerical_values(dataset.frame),
                            ).report
                            for partition, maximum in (
                                ("train", config.max_train_rows),
                                ("validation", config.max_validation_rows),
                            )
                        }
                        for i, f in enumerate(split.folds)
                    },
                }
            else:
                result = {
                    "runs": [
                        str(p)
                        for p in run_experiment(
                            str(suite["canonical_dir"]),
                            str(suite["plan_dir"]),
                            config,
                            raw_snapshot_dir=str(suite["raw_snapshot_dir"]),
                            output_root=str(suite["output_root"]),
                        )
                    ]
                }
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (DataError, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
