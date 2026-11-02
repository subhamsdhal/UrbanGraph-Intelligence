"""Offline evaluation planning CLI; existing raw/dataset CLIs are unchanged."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.contracts import (
    POSITIONAL_PROTOCOL,
    EvaluationProtocol,
    PlanConfig,
    config_from_dict,
)
from urbangraph.logging import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Versioned temporal evaluation plans for verified canonical datasets"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Build and verify a compact evaluation plan")
    choice = plan.add_mutually_exclusive_group(required=True)
    choice.add_argument("--protocol", choices=("benchmark", "benchmark-positional", "rolling"))
    choice.add_argument(
        "--config",
        type=Path,
        help="Validated JSON configuration, including explicit chronological boundaries",
    )
    plan.add_argument("--output-root", type=Path, default=Path("data/evaluation"))
    for name in ("inspect", "verify", "windows-preview"):
        command = commands.add_parser(name)
        command.add_argument("--plan-dir", type=Path, required=True)
        if name == "windows-preview":
            command.add_argument("--fold", type=int, default=0)
            command.add_argument(
                "--partition",
                choices=("train", "validation", "calibration", "test"),
                default="test",
            )
            command.add_argument("--sample", type=int, default=0)
    for command in (plan, *[commands.choices[n] for n in ("inspect", "verify", "windows-preview")]):
        command.add_argument("--canonical-dir", type=Path, required=True)
        command.add_argument("--raw-snapshot-dir", type=Path, required=True)
        command.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logger = configure_logging(logger_name="urbangraph.evaluation")
    output: dict[str, object]
    code = 0
    try:
        from urbangraph.evaluation.artifacts import load_plan, publish_plan, verify_plan
        from urbangraph.evaluation.windows import WindowCollection

        if args.command == "plan":
            if args.config:
                config = config_from_dict(read_json(args.config))
            else:
                config = (
                    PlanConfig(
                        protocol=EvaluationProtocol(
                            POSITIONAL_PROTOCOL
                            if args.protocol == "benchmark-positional"
                            else "dcrnn-sample-first-v1",
                            "benchmark",
                        )
                    )
                    if args.protocol in {"benchmark", "benchmark-positional"}
                    else PlanConfig()
                )
            result = publish_plan(
                args.canonical_dir,
                config,
                output_root=args.output_root,
                raw_snapshot_dir=args.raw_snapshot_dir,
            )
            output = {
                "plan_id": result.plan_id,
                "directory": str(result.directory),
                "verified": True,
                "reused": result.reused,
                "split": result.split.to_dict(),
                "sample_counts": [
                    {p.name: p.sample_count for p in f.partitions} for f in result.split.folds
                ],
            }
        elif args.command == "verify":
            checked = verify_plan(
                args.plan_dir,
                canonical_dir=args.canonical_dir,
                raw_snapshot_dir=args.raw_snapshot_dir,
            )
            output = {
                "verified": checked.valid,
                "issues": list(checked.issues),
                "plan_id": checked.plan_id,
            }
            code = 0 if checked.valid else 1
        else:
            dataset, split, plan = load_plan(
                args.plan_dir,
                canonical_dir=args.canonical_dir,
                raw_snapshot_dir=args.raw_snapshot_dir,
            )
            if args.command == "inspect":
                output = {
                    "verified": True,
                    "plan": plan,
                    "sample_counts": [
                        {p.name: p.sample_count for p in f.partitions} for f in split.folds
                    ],
                }
            else:
                windows = WindowCollection(dataset, split, fold=args.fold, partition=args.partition)
                output = {"sample_count": len(windows), "preview": windows[args.sample].preview()}
    except ImportError:
        output = {"error": 'Install optional dependencies: pip install -e ".[dev,data]"'}
        code = 1
    except (DataError, OSError, IndexError) as exc:
        output = {
            "error": str(exc)
            if isinstance(exc, DataError | IndexError)
            else f"Filesystem failure ({type(exc).__name__})"
        }
        code = 1
    if not args.json:
        print(f"{args.command.upper()}: {'PASS' if code == 0 else 'FAIL'}")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    if code:
        logger.error("Evaluation command failed; see result")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
