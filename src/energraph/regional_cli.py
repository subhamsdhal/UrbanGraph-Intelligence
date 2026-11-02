"""Offline regional commands delegated by the existing EnerGraph CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from energraph.artifacts import PendingPublication, require_committed_snapshot
from energraph.energy_graphs import EnergyGraphSpec, build_energy_graph, verify_energy_graph
from energraph.regional_data import materialize_regions, open_regions, regional_plan
from energraph.regional_research import run_regional, verify_regional_run
from energraph.regions import country_registry
from urbangraph.boosting.contracts import config_from_dict
from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.logging import configure_logging

COMMANDS = (
    "inspect-regions",
    "materialize-multiregion",
    "prepare-multiregion-evaluation",
    "graph-sources",
    "build-energy-graph",
    "verify-energy-graph",
    "run-multiregion",
    "verify-multiregion-run",
    "graph-readiness",
)


def readiness(
    canonical: Path,
    plan: Path,
    raw_snapshot_dir: Path,
    graphs: list[Path],
    graph_geographies: list[Path | None],
    runs: list[Path],
    run_graphs: list[Path],
    *,
    geography_reference: str | None = None,
    geometry_reference: str | None = None,
) -> dict[str, object]:
    if len(graphs) != len(graph_geographies) or len(runs) != len(run_graphs):
        raise DataError("Readiness source contexts must align with artifact lists")
    data = open_regions(canonical, raw_snapshot_dir=raw_snapshot_dir)
    kinds = []
    for path, geography in zip(graphs, graph_geographies, strict=True):
        _, desc = verify_energy_graph(
            path,
            canonical=canonical,
            plan=plan,
            raw_snapshot_dir=raw_snapshot_dir,
            geography=geography,
            geography_reference=geography_reference,
            geometry_reference=geometry_reference,
            expected_order=data.sensor_ids,
        )
        if object_dict(desc["canonical"]).get("canonical_id") != canonical.name:
            raise DataError("Missing graph lineage")
        kinds.append(str(object_dict(desc["specification"])["method"]))
    verified = [
        verify_regional_run(
            p,
            canonical=canonical,
            plan=plan,
            graph=graph,
            raw_snapshot_dir=raw_snapshot_dir,
        )
        for p, graph in zip(runs, run_graphs, strict=True)
    ]
    for run in runs:
        specification = object_dict(read_json(run / "payload/descriptor.json")["specification"])
        if object_dict(specification["canonical"]).get("canonical_id") != canonical.name:
            raise DataError("Readiness run and canonical identity mismatch")
    ready = (
        len(data.sensor_ids) >= 3
        and {"geographic-proxy", "statistical", "identity"} <= set(kinds)
        and bool(verified)
    )
    return {
        "schema_version": "energy-graph-readiness-1.0",
        "canonical_id": canonical.name,
        "regions": list(data.sensor_ids),
        "graph_count": len(graphs),
        "verified_runs": verified,
        "proxy_research_gate": "GO_BOUNDED_RESEARCH" if ready else "NO_GO",
        "physical_topology_gate": "BLOCKED: historical electrical area/EIC/edge archive unverified",
        "operational_gate": "NO_GO: publication vintages and rights unresolved",
        "limitations": [
            "Few country nodes; strong tabular baselines may suffice",
            "Geography and correlation are not physical/causal edges",
            "2020 geography is an explicitly retrospective proxy",
            "No energy GNN trained; no performance gain inferred from readiness",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EnerGraph regional data and graph evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-regions")
    inspect.add_argument("--canonical", type=Path)
    inspect.add_argument("--snapshot", type=Path)
    sources = sub.add_parser("graph-sources")
    sources.add_argument("--config", type=Path, default=Path("configs/energy/eg002-evidence.json"))
    materialize = sub.add_parser("materialize-multiregion")
    materialize.add_argument("--snapshot", type=Path, required=True)
    materialize.add_argument("--regions", nargs="+", required=True)
    materialize.add_argument("--policy", choices=["masked", "strict"], default="masked")
    materialize.add_argument("--root", type=Path, default=Path("data/energy/eg002/canonical"))
    plan = sub.add_parser("prepare-multiregion-evaluation")
    plan.add_argument("--canonical", type=Path, required=True)
    plan.add_argument("--snapshot", type=Path, required=True)
    plan.add_argument("--protocol", choices=["chronological", "rolling"], required=True)
    plan.add_argument("--latencies", type=int, nargs="+", required=True)
    plan.add_argument("--root", type=Path, default=Path("data/energy/eg002/plans"))
    graph = sub.add_parser("build-energy-graph")
    graph.add_argument("--canonical", type=Path, required=True)
    graph.add_argument("--plan", type=Path, required=True)
    graph.add_argument(
        "--method",
        choices=["identity", "empty", "geographic-proxy", "statistical", "random-control"],
        required=True,
    )
    graph.add_argument("--fold", type=int, default=0)
    graph.add_argument("--geography", type=Path)
    graph.add_argument("--geography-reference")
    graph.add_argument("--geometry-reference")
    graph.add_argument("--snapshot", type=Path, required=True)
    graph.add_argument("--root", type=Path, default=Path("data/energy/eg002/graphs"))
    for name in ("verify-energy-graph", "verify-multiregion-run"):
        c = sub.add_parser(name)
        c.add_argument("--directory", type=Path, required=True)
        for context in ("canonical", "plan", "snapshot"):
            c.add_argument("--" + context, type=Path, required=True)
        if name == "verify-multiregion-run":
            c.add_argument("--graph", type=Path, required=True)
        else:
            c.add_argument("--geography", type=Path)
            c.add_argument("--geography-reference")
            c.add_argument("--geometry-reference")
    run = sub.add_parser("run-multiregion")
    for name in ("canonical", "plan", "graph", "config", "snapshot"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--geography", type=Path)
    run.add_argument("--geography-reference")
    run.add_argument("--geometry-reference")
    run.add_argument("--root", type=Path, default=Path("data/energy/eg002/runs"))
    ready = sub.add_parser("graph-readiness")
    ready.add_argument("--canonical", type=Path, required=True)
    ready.add_argument("--plan", type=Path, required=True)
    ready.add_argument("--snapshot", type=Path, required=True)
    ready.add_argument("--graphs", type=Path, nargs="+", required=True)
    ready.add_argument("--graph-geographies", nargs="+", required=True)
    ready.add_argument("--runs", type=Path, nargs="+", required=True)
    ready.add_argument("--run-graphs", type=Path, nargs="+", required=True)
    ready.add_argument("--geography-reference")
    ready.add_argument("--geometry-reference")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        output: dict[str, object]
        if args.command == "inspect-regions":
            output = (
                country_registry().to_dict()
                if args.canonical is None
                else open_regions(args.canonical, raw_snapshot_dir=args.snapshot).lineage
            )
        elif args.command == "graph-sources":
            output = read_json(args.config)
        elif args.command == "materialize-multiregion":
            result = materialize_regions(
                args.snapshot, args.root, tuple(args.regions), policy=args.policy
            )
            result = require_committed_snapshot(result)
            output = {"directory": str(result.directory), "reused": result.reused}
        elif args.command == "prepare-multiregion-evaluation":
            result = regional_plan(
                args.canonical,
                args.root,
                args.protocol,
                tuple(args.latencies),
                raw_snapshot_dir=args.snapshot,
            )
            result = require_committed_snapshot(result)
            output = {"directory": str(result.directory), "reused": result.reused}
        elif args.command == "build-energy-graph":
            result = build_energy_graph(
                args.canonical,
                args.plan,
                args.root,
                EnergyGraphSpec(args.method),
                fold=args.fold,
                geography=args.geography,
                geography_reference=args.geography_reference,
                geometry_reference=args.geometry_reference,
                raw_snapshot_dir=args.snapshot,
            )
            result = require_committed_snapshot(result)
            output = {"directory": str(result.directory), "reused": result.reused}
        elif args.command == "verify-energy-graph":
            _, output = verify_energy_graph(
                args.directory,
                canonical=args.canonical,
                plan=args.plan,
                raw_snapshot_dir=args.snapshot,
                geography=args.geography,
                geography_reference=args.geography_reference,
                geometry_reference=args.geometry_reference,
            )
            output = {"valid": True} | output
        elif args.command == "run-multiregion":
            config = read_json(args.config)
            path = run_regional(
                args.canonical,
                args.plan,
                args.graph,
                args.root,
                config_from_dict(config["boost_config"]),
                raw_snapshot_dir=args.snapshot,
                geography=args.geography,
                geography_reference=args.geography_reference,
                geometry_reference=args.geometry_reference,
            )
            if isinstance(path, PendingPublication):
                raise DataError("Regional run is awaiting separate broker approval")
            output = verify_regional_run(
                path,
                canonical=args.canonical,
                plan=args.plan,
                graph=args.graph,
                raw_snapshot_dir=args.snapshot,
            ) | {"directory": str(path)}
        elif args.command == "verify-multiregion-run":
            output = verify_regional_run(
                args.directory,
                canonical=args.canonical,
                plan=args.plan,
                graph=args.graph,
                raw_snapshot_dir=args.snapshot,
            )
        else:
            output = readiness(
                args.canonical,
                args.plan,
                args.snapshot,
                args.graphs,
                [None if item == "-" else Path(item) for item in args.graph_geographies],
                args.runs,
                args.run_graphs,
                geography_reference=args.geography_reference,
                geometry_reference=args.geometry_reference,
            )
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (DataError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "message": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
