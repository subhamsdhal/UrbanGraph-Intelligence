"""Offline baseline registry, experiment execution and verified comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from urbangraph.baselines.artifacts import verify_run
from urbangraph.baselines.contracts import MODEL_IDS, ModelConfig, model_config
from urbangraph.baselines.registry import list_models
from urbangraph.baselines.reporting import compare_runs
from urbangraph.baselines.runner import run_suite
from urbangraph.data import DataError
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.artifacts import load_plan
from urbangraph.evaluation.dataset import identity
from urbangraph.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    registry = commands.add_parser("list-models", help="Show versions and model contracts")
    registry.add_argument("--json", action="store_true")
    run = commands.add_parser("run", help="Fit/select/evaluate on an explicit verified plan")
    run.add_argument("--canonical-dir", required=True, type=Path)
    run.add_argument("--raw-snapshot-dir", required=True, type=Path)
    run.add_argument("--plan-dir", required=True, type=Path)
    run.add_argument("--model", choices=(*MODEL_IDS, "all"), default="all")
    run.add_argument("--config", type=Path, help="Single model configuration JSON")
    run.add_argument("--output-root", type=Path, default=Path("data/baselines"))
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--json", action="store_true")
    compare = commands.add_parser("compare", help="Rank runs only on a common population")
    compare.add_argument("--run-dir", type=Path, action="append", required=True)
    compare.add_argument("--raw-snapshot-dir", type=Path, required=True)
    compare.add_argument("--canonical-dir", type=Path, required=True)
    compare.add_argument("--json", action="store_true")
    verify = commands.add_parser("verify", help="Verify an immutable experiment bundle")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--raw-snapshot-dir", type=Path, required=True)
    verify.add_argument("--canonical-dir", type=Path, required=True)
    verify.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        result: object
        if args.command == "list-models":
            result = list_models()
        elif args.command == "compare":
            result = compare_runs(
                args.run_dir,
                raw_snapshot_dir=args.raw_snapshot_dir,
                canonical_dir=args.canonical_dir,
            )
        elif args.command == "verify":
            result = verify_run(
                args.run_dir,
                raw_snapshot_dir=args.raw_snapshot_dir,
                canonical_dir=args.canonical_dir,
            )
        else:
            configs = (
                (model_config(read_json(args.config)),)
                if args.config
                else tuple(
                    ModelConfig(model=name)
                    for name in (MODEL_IDS if args.model == "all" else (args.model,))
                )
            )
            if args.config and args.model != "all" and configs[0].model != args.model:
                raise DataError("Model argument conflicts with configuration")
            if args.dry_run:
                dataset, split, plan = load_plan(
                    args.plan_dir,
                    canonical_dir=args.canonical_dir,
                    raw_snapshot_dir=args.raw_snapshot_dir,
                )
                result = {
                    "dataset": dataset.lineage,
                    "plan_id": identity(plan),
                    "split": split.to_dict(),
                    "models": [c.to_dict() for c in configs],
                }
            else:
                paths = run_suite(
                    args.canonical_dir,
                    args.plan_dir,
                    configs,
                    raw_snapshot_dir=args.raw_snapshot_dir,
                    output_root=args.output_root,
                    batch_size=args.batch_size,
                )
                result = {
                    "runs": [str(p) for p in paths],
                    "comparison": compare_runs(
                        paths,
                        raw_snapshot_dir=args.raw_snapshot_dir,
                        canonical_dir=args.canonical_dir,
                    ),
                }
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (DataError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "error": str(exc)
                    if isinstance(exc, DataError)
                    else f"Operation failed ({type(exc).__name__})"
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
