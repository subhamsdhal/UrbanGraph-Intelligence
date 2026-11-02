"""Traffic dataset CLI; optional dependencies load only after argument parsing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.datasets.contracts import DatasetConfig, StructuralError
from urbangraph.logging import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse, audit and materialize trusted traffic HDF snapshots"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "profile", "materialize"):
        command = commands.add_parser(name, help=f"{name.capitalize()} a verified raw snapshot")
        command.add_argument("--snapshot-dir", type=Path, required=True)
        command.add_argument("--dataset", choices=("metr-la", "pems-bay"))
        command.add_argument("--hdf-key")
        command.add_argument("--payload-file", help="Explicit inventory-relative HDF file override")
        command.add_argument("--nominal-interval-seconds", type=int, default=300)
        command.add_argument("--low-variance-std", type=float, default=0.01)
        command.add_argument("--extreme-missing-fraction", type=float, default=0.5)
        command.add_argument("--max-gap-examples", type=int, default=20)
        command.add_argument("--reject-check", action="append", default=[])
        command.add_argument("--units")
        command.add_argument("--units-reference")
        command.add_argument("--json", action="store_true", help="Machine-readable stdout")
        if name == "materialize":
            command.add_argument("--output-root", type=Path, default=Path("data/interim"))
    verify = commands.add_parser(
        "verify", help="Verify canonical artifacts and optional external raw lineage"
    )
    verify.add_argument("--artifact-dir", type=Path, required=True)
    verify.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="Raw snapshot selector; current canonical_read rights are required",
    )
    verify.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logger = configure_logging(logger_name="urbangraph.datasets")
    output: dict[str, object]
    try:
        from urbangraph.datasets.hdf5_adapter import parse_snapshot
        from urbangraph.datasets.materialization import materialize_snapshot, verify_materialization
        from urbangraph.datasets.profiling import profile_dataset

        if args.command == "verify":
            verification = verify_materialization(
                args.artifact_dir, raw_snapshot_dir=args.snapshot_dir
            )
            output = {
                "verified": verification.valid,
                "issues": list(verification.issues),
                "transformation_id": verification.transformation_id,
            }
            code = 0 if verification.valid else 1
        else:
            config = DatasetConfig(
                hdf_key=args.hdf_key,
                payload_file=args.payload_file,
                nominal_interval_seconds=args.nominal_interval_seconds,
                low_variance_std=args.low_variance_std,
                extreme_missing_fraction=args.extreme_missing_fraction,
                max_gap_examples=args.max_gap_examples,
                reject_checks=tuple(args.reject_check),
                units=args.units,
                units_reference=args.units_reference,
            )
            if args.command == "materialize":
                result = materialize_snapshot(
                    args.snapshot_dir,
                    output_root=args.output_root,
                    dataset_id=args.dataset,
                    config=config,
                )
                output = {
                    "verified": True,
                    "reused": result.reused,
                    "transformation_id": result.transformation_id,
                    "directory": str(result.directory),
                }
                code = 0
            else:
                dataset = parse_snapshot(args.snapshot_dir, dataset_id=args.dataset, config=config)
                report = profile_dataset(dataset)
                output = (
                    report.to_dict()
                    if args.command == "profile"
                    else {
                        "structurally_valid": True,
                        "accepted": report.accepted,
                        "dataset_id": dataset.dataset_id,
                        "raw_snapshot_id": dataset.raw_snapshot_id,
                        "timestamp_count": len(dataset.observations),
                        "sensor_count": len(dataset.sensors),
                        "temporal": dataset.temporal,
                        "findings": [f.to_dict() for f in report.findings],
                    }
                )
                code = 0 if report.accepted else 1
    except StructuralError as exc:
        output = {
            "structurally_valid": False,
            "error": str(exc),
            "findings": [f.to_dict() for f in exc.findings],
        }
        code = 1
    except ImportError:
        output = {
            "error": 'Data dependencies unavailable; install with pip install -e ".[dev,data]"'
        }
        code = 1
    except DataError as exc:
        output = {"error": str(exc)}
        code = 1
    except OSError as exc:
        output = {"error": f"Filesystem operation failed ({type(exc).__name__})"}
        code = 1
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(f"{args.command.upper()}: {'PASS' if code == 0 else 'FAIL'}")
        for key, value in output.items():
            print(f"{key}: {json.dumps(value, ensure_ascii=True, allow_nan=False)}")
    if code:
        logger.error("Dataset command failed; see structured result")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
