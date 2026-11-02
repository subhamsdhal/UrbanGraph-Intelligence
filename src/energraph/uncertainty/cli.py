"""Offline uncertainty commands using immutable local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from energraph.artifacts import (
    BrokerPublicationContext,
    resolve_broker_reference,
)
from energraph.broker_adapter import BrokerAdapter
from energraph.uncertainty.conformal import CalibrationPolicy
from energraph.uncertainty.population import reconstruct
from energraph.uncertainty.research import evaluate_population, verify_intervals
from urbangraph.data import DataError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("uncertainty-audit")
    for name in ["canonical", "plan", "point-run", "graph", "snapshot", "root"]:
        audit.add_argument("--" + name, type=Path, required=True)
    audit.add_argument("--fold", type=int, default=0)
    cal = sub.add_parser("uncertainty-calibrate")
    cal.add_argument("--population", type=Path, required=True)
    cal.add_argument("--population-parents", type=Path, nargs="+", required=True)
    cal.add_argument("--root", type=Path, required=True)
    cal.add_argument("--model", choices=["lightgbm", "persistence"], default="lightgbm")
    cal.add_argument("--coverage", type=float, default=0.9)
    cal.add_argument(
        "--grouping",
        choices=["global", "region", "horizon", "region-horizon"],
        default="region-horizon",
    )
    cal.add_argument("--normalized", action="store_true")
    cal.add_argument("--spacing", type=int, default=1)
    for name in ["uncertainty-verify", "uncertainty-evaluate", "uncertainty-report"]:
        command = sub.add_parser(name)
        directory = command.add_mutually_exclusive_group(required=True)
        directory.add_argument("--directory", type=Path)
        directory.add_argument("--artifact-id")
        population = command.add_mutually_exclusive_group(required=True)
        population.add_argument("--population", type=Path)
        population.add_argument("--population-artifact-id")
        parents = command.add_mutually_exclusive_group(required=True)
        parents.add_argument("--population-parents", type=Path, nargs="+")
        parents.add_argument("--population-parent-artifact-ids", nargs="+")
        command.add_argument("--broker-url")
        command.add_argument("--worker-token-file", type=Path)
        command.add_argument("--reader-token-file", type=Path)
        command.add_argument("--materialization-root", type=Path)
    compare = sub.add_parser("uncertainty-compare")
    compare.add_argument("--directories", type=Path, nargs="+", required=True)
    compare.add_argument("--population", type=Path, required=True)
    compare.add_argument("--population-parents", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        result: object
        if args.command == "uncertainty-audit":
            result = {
                "population": str(
                    reconstruct(
                        args.canonical,
                        args.plan,
                        args.point_run,
                        args.root,
                        args.fold,
                        graph=args.graph,
                        raw_snapshot_dir=args.snapshot,
                    )
                )
            }
        elif args.command == "uncertainty-calibrate":
            result = {
                "interval_artifact": str(
                    evaluate_population(
                        args.population,
                        args.root,
                        args.model,
                        CalibrationPolicy(
                            args.coverage, args.grouping, 30, args.spacing, args.normalized
                        ),
                        population_parents=tuple(args.population_parents),
                    )
                )
            }
        elif args.command == "uncertainty-compare":
            result = [
                verify_intervals(
                    d, args.population, population_parents=tuple(args.population_parents)
                )
                for d in args.directories
            ]
        else:
            if args.artifact_id is None:
                if (
                    args.population_parents is None
                    or args.population_parent_artifact_ids is not None
                ):
                    raise DataError("Local verification requires explicit population parent paths")
                result = verify_intervals(
                    args.directory,
                    args.population,
                    population_parents=tuple(args.population_parents),
                )
            else:
                required = (
                    args.population_artifact_id,
                    args.population_parent_artifact_ids,
                    args.broker_url,
                    args.worker_token_file,
                    args.reader_token_file,
                    args.materialization_root,
                )
                if any(value is None for value in required) or args.population_parents is not None:
                    raise DataError(
                        "Broker verification requires complete artifact and client identity"
                    )
                broker = BrokerAdapter(
                    args.broker_url,
                    worker_token_file=args.worker_token_file,
                    reader_token_file=args.reader_token_file,
                )
                context = BrokerPublicationContext(broker, args.materialization_root)
                artifact = resolve_broker_reference(
                    broker,
                    args.artifact_id,
                    purpose="scientific_evaluation",
                    operation="energy_interval_verify",
                )
                population_ref = resolve_broker_reference(
                    broker,
                    args.population_artifact_id,
                    purpose="scientific_evaluation",
                    operation="energy_interval_verify",
                )
                parent_refs = tuple(
                    resolve_broker_reference(
                        broker,
                        artifact_id,
                        purpose="scientific_evaluation",
                        operation="energy_interval_verify",
                    )
                    for artifact_id in args.population_parent_artifact_ids
                )
                result = verify_intervals(
                    artifact,
                    population_ref,
                    broker_context=context,
                    population_parent_references=parent_refs,
                )
        print(json.dumps(result, allow_nan=False))
        return 0
    except (DataError, OSError, ValueError, KeyError, AssertionError) as error:
        print(json.dumps({"error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
