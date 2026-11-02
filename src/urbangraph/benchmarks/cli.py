"""Offline opt-in benchmark workflow. No downloader or untrusted pickle loader."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from urbangraph.benchmarks.sources import Acquisition, source_spec
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import BENCHMARK_PROTOCOLS
from urbangraph.logging import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit local benchmark evidence workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect-sources", help="Show pinned expectations; no network access")
    importer = commands.add_parser("import-local", help="Import one authorized local HDF file")
    importer.add_argument("--file", type=Path, required=True)
    importer.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    auditor = commands.add_parser("audit", help="Parse trusted HDF and verify canonical artifact")
    auditor.add_argument("--snapshot-dir", type=Path, required=True)
    auditor.add_argument("--output-root", type=Path, default=Path("data/interim"))
    auditor.add_argument("--hdf-key")
    auditor.add_argument(
        "--trusted-hdf",
        action="store_true",
        required=True,
        help="Attest input trust: pandas HDF can contain pickled metadata",
    )
    evaluation = commands.add_parser("prepare-evaluation")
    evaluation.add_argument("--canonical-dir", type=Path, required=True)
    evaluation.add_argument("--snapshot-dir", type=Path, required=True)
    evaluation.add_argument("--output-root", type=Path, default=Path("data/evaluation"))
    ready = commands.add_parser("readiness", help="Reverify evidence; absent data remains BLOCKED")
    ready.add_argument("--snapshot-dir", type=Path)
    ready.add_argument("--canonical-dir", type=Path)
    for command in (evaluation, ready):
        command.add_argument(
            "--benchmark-protocol", choices=BENCHMARK_PROTOCOLS, default="dcrnn-sample-first-v1"
        )
    for name in ("benchmark", "chronological", "rolling"):
        ready.add_argument(f"--{name}-plan", type=Path)
    ready.add_argument("--metadata-file", type=Path)
    ready.add_argument(
        "--metadata-layout",
        default="locations-header",
        choices=("locations-header", "bay-headerless", "ids"),
    )
    for command in (importer, ready):
        command.add_argument("--source-reference", required=command is importer)
        command.add_argument("--method", default="local authorized copy")
        command.add_argument(
            "--evidence-kind", choices=("real", "synthetic"), required=command is importer
        )
        command.add_argument("--usage-notes", default="Usage and redistribution unresolved")
        command.add_argument("--usage-reviewed", action="store_true")
        command.add_argument("--acquired-on")
    for name, command in commands.choices.items():
        command.add_argument("--json", action="store_true")
        command.add_argument(
            "--report-root", type=Path, help="Optional append-only JSON/Markdown report export"
        )
        if name != "inspect-sources":
            command.add_argument("--dataset", choices=("metr-la", "pems-bay"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logger = configure_logging(logger_name="urbangraph.benchmarks")
    code = 0
    output: dict[str, object]
    try:
        from urbangraph.benchmarks.audit import readiness
        from urbangraph.benchmarks.pipeline import audit_raw, import_local, prepare_evaluation
        from urbangraph.benchmarks.reports import publish_report
        from urbangraph.datasets.contracts import DatasetConfig

        acquisition = None
        if args.command in {"import-local", "readiness"} and args.source_reference:
            acquisition = Acquisition(
                args.source_reference,
                args.method,
                args.evidence_kind,
                args.usage_notes,
                args.usage_reviewed,
                args.acquired_on,
            )
        if args.command == "inspect-sources":
            output = {name: source_spec(name) for name in ("metr-la", "pems-bay")}
        elif args.command == "import-local":
            if acquisition is None:
                raise DataError("Explicit acquisition evidence required")
            raw = import_local(args.file, args.dataset, acquisition, raw_root=args.raw_root)
            output = {
                "dataset_id": args.dataset,
                "snapshot_id": raw.manifest.snapshot_id,
                "snapshot_dir": str(raw.directory),
                "reused": raw.reused,
                "acquisition": acquisition.to_dict(),
                "verified": True,
            }
        elif args.command == "audit":
            output = audit_raw(
                args.snapshot_dir,
                args.dataset,
                output_root=args.output_root,
                config=DatasetConfig(hdf_key=args.hdf_key),
            )
        elif args.command == "prepare-evaluation":
            output = {
                "dataset_id": args.dataset,
                "plans": {
                    k: str(v)
                    for k, v in prepare_evaluation(
                        args.canonical_dir,
                        args.dataset,
                        raw_snapshot_dir=args.snapshot_dir,
                        output_root=args.output_root,
                        benchmark_protocol=args.benchmark_protocol,
                    ).items()
                },
            }
        else:
            plans = {
                name: getattr(args, name + "_plan")
                for name in ("benchmark", "chronological", "rolling")
                if getattr(args, name + "_plan")
            }
            output = readiness(
                args.dataset,
                acquisition=acquisition,
                raw_dir=args.snapshot_dir,
                canonical_dir=args.canonical_dir,
                plans=plans,
                metadata_file=args.metadata_file,
                metadata_layout=args.metadata_layout,
                benchmark_protocol=args.benchmark_protocol,
            )
            code = 1 if output["status"] in {"BLOCKED", "FAIL"} else 0
        if args.report_root:
            path = publish_report(output, args.report_root)
            output = output | {"report_dir": str(path)}
    except ImportError:
        output, code = {"error": 'Install optional dependencies: pip install -e ".[dev,data]"'}, 1
    except (DataError, OSError) as exc:
        output, code = (
            {
                "error": str(exc)
                if isinstance(exc, DataError)
                else f"Filesystem failure ({type(exc).__name__})"
            },
            1,
        )
    if not args.json:
        print(f"{args.command}: {output.get('status', 'FAIL' if code else 'PASS')}")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    if code:
        logger.error("Benchmark evidence incomplete or failed; see result")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
