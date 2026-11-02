"""Explicit offline EG-003 experiment commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from energraph.artifacts import PendingPublication
from energraph.graph_research import run_graph_experiment, verify_graph_run
from urbangraph.data import DataError
from urbangraph.datasets.serialization import read_json
from urbangraph.logging import configure_logging
from urbangraph.stgnn.contracts import training_config

COMMANDS = ("graph-models", "stgnn-train", "stgnn-evaluate", "stgnn-compare", "stgnn-verify")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled energy temporal/graph research")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("graph-models")
    train = commands.add_parser("stgnn-train")
    for name in ("canonical", "plan", "graph", "config", "snapshot"):
        train.add_argument("--" + name, type=Path, required=True)
    for name in ("geography", "geography-reference", "geometry-reference"):
        train.add_argument("--" + name, type=Path if name == "geography" else str)
    train.add_argument("--root", type=Path, default=Path("data/energy/eg003"))
    train.add_argument("--condition", required=True)
    train.add_argument("--fold", type=int, default=0)
    train.add_argument("--widths", type=int, nargs="+", default=[8, 16])
    train.add_argument("--blocks", type=int, default=4)
    for name in ("stgnn-evaluate", "stgnn-verify", "stgnn-compare"):
        command = commands.add_parser(name)
        for context in ("canonical", "plan", "graph", "snapshot"):
            command.add_argument("--" + context, type=Path, required=True)
        for context in ("geography", "geography-reference", "geometry-reference"):
            command.add_argument("--" + context, type=Path if context == "geography" else str)
        if name == "stgnn-compare":
            command.add_argument("--directories", type=Path, nargs="+", required=True)
        else:
            command.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_logging()
    try:
        if args.command == "graph-models":
            result: object = {
                "implemented": [
                    "parameter-matched node-independent temporal",
                    "STGCN-style direct decoder",
                ],
                "conditions": [
                    "temporal",
                    "identity",
                    "geographic",
                    "correlation",
                    "random",
                    "correlation-unweighted",
                ],
                "secondary_diffusion": "deferred; existing traffic implementation preserved",
                "physical_topology": False,
            }
        elif args.command == "stgnn-train":
            path = run_graph_experiment(
                args.canonical,
                args.plan,
                args.graph,
                args.root,
                args.condition,
                args.fold,
                training_config(read_json(args.config)),
                widths=tuple(args.widths),
                blocks=args.blocks,
                raw_snapshot_dir=args.snapshot,
                geography=args.geography,
                geography_reference=args.geography_reference,
                geometry_reference=args.geometry_reference,
            )
            if isinstance(path, PendingPublication):
                raise DataError("Graph run is awaiting separate broker approval")
            result = verify_graph_run(
                path,
                canonical=args.canonical,
                plan=args.plan,
                graph=args.graph,
                raw_snapshot_dir=args.snapshot,
            )
        elif args.command == "stgnn-compare":
            records = [
                verify_graph_run(
                    p,
                    canonical=args.canonical,
                    plan=args.plan,
                    graph=args.graph,
                    raw_snapshot_dir=args.snapshot,
                )
                for p in args.directories
            ]
            descriptors = [read_json(p / "payload/descriptor.json") for p in args.directories]
            if len({str(d["population_id"]) for d in descriptors}) != 1:
                raise DataError("Comparison requires identical target populations")
            result = records
        else:
            result = verify_graph_run(
                args.directory,
                canonical=args.canonical,
                plan=args.plan,
                graph=args.graph,
                raw_snapshot_dir=args.snapshot,
            )
        print(json.dumps(result))
        return 0
    except (DataError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
