"""Offline graph build, inspect and integrity verification commands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from urbangraph.data import DataError
from urbangraph.datasets.serialization import read_json
from urbangraph.graphs.artifacts import build_graph, load_graph, verify_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--config", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--graph-dir", required=True)
    verify.add_argument("--config", required=True, type=Path)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--graph-dir", required=True)
    inspect.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            graph = build_graph(read_json(args.config))
            result = {
                "graph_id": graph.graph_id,
                "directory": str(graph.directory),
                "diagnostics": graph.descriptor["diagnostics"],
            }
        elif args.command == "verify":
            context = read_json(args.config)
            required = {
                "raw_snapshot_dir",
                "canonical_dir",
                "locations",
                "location_reference",
            }
            if (
                not isinstance(context, dict)
                or not required.issubset(context)
                or any(not isinstance(context[key], str) for key in required)
            ):
                raise DataError("Complete governed graph context required")
            optional = {"distances", "distance_reference"}
            if frozenset(context) not in {frozenset(required), frozenset(required | optional)}:
                raise DataError("Complete governed road graph context required")
            if any(key in context and not isinstance(context[key], str) for key in optional):
                raise DataError("Complete governed road graph context required")
            typed_context = cast(dict[str, str], context)
            verification = verify_graph(
                args.graph_dir,
                raw_snapshot_dir=typed_context["raw_snapshot_dir"],
                canonical_dir=typed_context["canonical_dir"],
                locations=typed_context["locations"],
                location_reference=typed_context["location_reference"],
                distances=typed_context.get("distances"),
                distance_reference=typed_context.get("distance_reference"),
            )
            print(json.dumps(asdict(verification), sort_keys=True))
            return 0 if verification.valid else 1
        else:
            context = read_json(args.config)
            required = {
                "raw_snapshot_dir",
                "canonical_dir",
                "locations",
                "distances",
                "location_reference",
                "distance_reference",
            }
            if not required.issubset(context) or any(
                not isinstance(context[key], str) for key in required
            ):
                raise DataError("Complete governed graph context required")
            result = load_graph(
                args.graph_dir,
                raw_snapshot_dir=str(context["raw_snapshot_dir"]),
                canonical_dir=str(context["canonical_dir"]),
                locations=str(context["locations"]),
                distances=str(context["distances"]),
                location_reference=str(context["location_reference"]),
                distance_reference=str(context["distance_reference"]),
            ).descriptor
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
