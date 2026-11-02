"""Offline raw-data CLI. JSON results on stdout; structured errors on stderr."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from urbangraph.data import (
    DataError,
    DatasetManifest,
    DatasetSource,
    ingest_dataset,
    verify_snapshot,
)
from urbangraph.logging import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest and verify immutable local raw datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser(
        "ingest", help="Copy opaque files into a verified content snapshot"
    )
    ingest.add_argument("--dataset", required=True, help="Dataset slug, e.g. metr-la or pems-bay")
    ingest.add_argument("--source-dir", required=True, type=Path)
    ingest.add_argument("--raw-root", default=Path("data/raw"), type=Path)
    ingest.add_argument(
        "--source-reference", help="User-supplied provenance; never include credentials"
    )
    ingest.add_argument("--source-description")
    ingest.add_argument("--attribution")
    ingest.add_argument("--licensing-notes")
    for command in ("verify", "inspect"):
        child = commands.add_parser(
            command, help=f"Fully verify a snapshot and {command} its metadata"
        )
        child.add_argument("--snapshot-dir", required=True, type=Path)
    return parser


def _summary(manifest: DatasetManifest | None, directory: Path) -> dict[str, object]:
    return {
        "dataset_id": manifest.dataset_id if manifest else None,
        "snapshot_id": manifest.snapshot_id if manifest else None,
        "file_count": len(manifest.files) if manifest else None,
        "total_bytes": manifest.total_bytes if manifest else None,
        "manifest_path": str(directory / "manifest.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logger = configure_logging(logger_name="urbangraph.data")
    try:
        if args.command == "ingest":
            snapshot = ingest_dataset(
                args.source_dir,
                args.dataset,
                raw_root=args.raw_root,
                source=DatasetSource(
                    reference=args.source_reference,
                    description=args.source_description,
                    attribution=args.attribution,
                    licensing_notes=args.licensing_notes,
                ),
            )
            output = _summary(snapshot.manifest, snapshot.directory)
            output.update(verified=True, reused=snapshot.reused)
            exit_code = 0
        else:
            result = verify_snapshot(args.snapshot_dir)
            output = _summary(result.manifest, result.snapshot_dir)
            output.update(
                verified=result.valid,
                checked_files=result.checked_files,
                issues=[asdict(issue) for issue in result.issues],
            )
            if args.command == "inspect" and result.valid and result.manifest is not None:
                output.update(
                    schema_version=result.manifest.schema_version,
                    manifest_version=result.manifest.manifest_version,
                    created_at=result.manifest.created_at,
                    files=[asdict(record) for record in result.manifest.files],
                )
            exit_code = 0 if result.valid else 1
        print(json.dumps(output, indent=2, sort_keys=True))
        return exit_code
    except DataError as exc:
        logger.error("%s", exc)
        return 1
    except OSError as exc:
        logger.error("Filesystem operation failed: %s", type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
