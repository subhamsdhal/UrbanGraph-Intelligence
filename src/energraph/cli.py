"""Offline EnerGraph research CLI. No implicit network access or credentials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from energraph.artifacts import (
    BrokerPublicationContext,
    PendingPublication,
    require_committed_snapshot,
    resolve_broker_reference,
)
from energraph.broker_adapter import BrokerAdapter
from energraph.broker_tree import inspect_tree
from energraph.canonical import materialize, open_energy
from energraph.evaluation import build_plan
from energraph.experiments import run_experiment, verify_run
from urbangraph.data import (
    DataError,
    DatasetSnapshot,
    DatasetSource,
    ingest_dataset,
    verify_snapshot,
)
from urbangraph.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    # Energy commands do not load optional regional model libraries.
    regional_commands = (
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

    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0].startswith("uncertainty-"):
        from energraph.uncertainty.cli import main as uncertainty_main

        return uncertainty_main(arguments)
    graph_commands = (
        "graph-models",
        "stgnn-train",
        "stgnn-evaluate",
        "stgnn-compare",
        "stgnn-verify",
    )
    if arguments and arguments[0] in graph_commands:
        from energraph.graph_cli import main as graph_main

        return graph_main(arguments)
    if arguments and arguments[0] in regional_commands:
        from energraph.regional_cli import main as regional_main

        return regional_main(arguments)
    parser = argparse.ArgumentParser(description="EnerGraph hourly energy research")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Ingest an existing local OPSD directory")
    ingest.add_argument("--source-dir", type=Path, required=True)
    ingest.add_argument("--root", type=Path, default=Path("data/energy/raw"))
    ingest.add_argument("--reference", required=True)
    canonical = sub.add_parser("canonicalize")
    canonical.add_argument("--snapshot", type=Path, required=True)
    canonical.add_argument("--root", type=Path, default=Path("data/energy/canonical"))
    plan = sub.add_parser("plan")
    plan.add_argument("--canonical", type=Path, required=True)
    plan.add_argument("--snapshot", type=Path, required=True)
    plan.add_argument("--protocol", choices=["chronological", "rolling"], required=True)
    plan.add_argument("--root", type=Path, default=Path("data/energy/plans"))
    experiment = sub.add_parser("run")
    experiment.add_argument("--canonical", type=Path, required=True)
    experiment.add_argument("--plan", type=Path, required=True)
    experiment.add_argument("--snapshot", type=Path, required=True)
    experiment.add_argument("--root", type=Path, default=Path("data/energy/runs"))
    for command in (experiment,):
        _broker_cli_arguments(command, include_run=False)
    for name in ("verify", "verify-run", "inspect"):
        command = sub.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        if name == "verify-run":
            command.add_argument("--canonical", type=Path, required=True)
            command.add_argument("--plan", type=Path, required=True)
            _broker_cli_arguments(command, include_run=True)
        if name in {"inspect", "verify-run"}:
            command.add_argument("--snapshot", type=Path, required=True)
    multi = sub.add_parser("materialize-multi")
    multi.add_argument("--snapshot", type=Path, required=True)
    multi.add_argument("--filename", required=True)
    multi.add_argument("--variables", nargs="+", required=True)
    multi.add_argument("--root", type=Path, default=Path("data/energy/canonical"))
    target = sub.add_parser("target-plan")
    target.add_argument("--canonical", type=Path, required=True)
    target.add_argument("--target", required=True)
    target.add_argument("--protocol", choices=["chronological", "rolling"], required=True)
    target.add_argument("--latency", type=int, default=24)
    target.add_argument("--weather", type=Path)
    target.add_argument("--snapshot", type=Path, required=True)
    target.add_argument("--weather-snapshot", type=Path)
    target.add_argument("--root", type=Path, default=Path("data/energy/plans"))
    boost = sub.add_parser("train-energy")
    boost.add_argument("--canonical", type=Path, required=True)
    boost.add_argument("--plan", type=Path, required=True)
    boost.add_argument("--config", type=Path, required=True)
    boost.add_argument("--weather", type=Path)
    boost.add_argument("--snapshot", type=Path, required=True)
    boost.add_argument("--weather-snapshot", type=Path)
    boost.add_argument("--root", type=Path, default=Path("data/energy/runs"))
    for name in ("inspect-multi", "verify-energy-run"):
        command = sub.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        if name == "inspect-multi":
            command.add_argument("--snapshot", type=Path, required=True)
        else:
            for context in ("canonical", "plan", "snapshot"):
                command.add_argument("--" + context, type=Path, required=True)
            command.add_argument("--weather", type=Path)
            command.add_argument("--weather-snapshot", type=Path)
    for regional_command in regional_commands:
        sub.add_parser(regional_command, help="Regional research command; use command --help")
    for graph_command in graph_commands:
        sub.add_parser(graph_command, help="Energy graph research; use command --help")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        if args.command in {
            "materialize-multi",
            "inspect-multi",
            "target-plan",
            "train-energy",
            "verify-energy-run",
        }:
            from energraph.energy_boosting import feature_from_dict, run_energy, verify_energy_run
            from energraph.multivariable import materialize_multi, open_multi
            from energraph.target_plans import create_target_plan
            from urbangraph.boosting.contracts import config_from_dict
            from urbangraph.datasets.serialization import read_json

            if args.command == "materialize-multi":
                result = materialize_multi(
                    args.snapshot, args.root, args.filename, tuple(args.variables)
                )
                result = require_committed_snapshot(result)
                output: dict[str, object] = {
                    "directory": str(result.directory),
                    "reused": result.reused,
                }
            elif args.command == "inspect-multi":
                _, output = open_multi(args.directory, raw_snapshot_dir=args.snapshot)
            elif args.command == "target-plan":
                result = create_target_plan(
                    args.canonical,
                    args.root,
                    args.target,
                    args.protocol,
                    latency=args.latency,
                    weather=args.weather,
                    raw_snapshot_dir=args.snapshot,
                    weather_raw_snapshot_dir=args.weather_snapshot,
                )
                result = require_committed_snapshot(result)
                output = {"directory": str(result.directory), "reused": result.reused}
            elif args.command == "train-energy":
                config = read_json(args.config)
                directory = run_energy(
                    args.canonical,
                    args.plan,
                    args.root,
                    feature_from_dict(config["features"]),
                    config_from_dict(config["boost_config"]),
                    weather=args.weather,
                    raw_snapshot_dir=args.snapshot,
                    weather_raw_snapshot_dir=args.weather_snapshot,
                )
                if isinstance(directory, PendingPublication):
                    raise DataError("Energy run is awaiting separate broker approval")
                output = verify_energy_run(
                    directory,
                    canonical=args.canonical,
                    plan=args.plan,
                    raw_snapshot_dir=args.snapshot,
                    weather=args.weather,
                    weather_raw_snapshot_dir=args.weather_snapshot,
                ) | {"directory": str(directory)}
            else:
                output = verify_energy_run(
                    args.directory,
                    canonical=args.canonical,
                    plan=args.plan,
                    raw_snapshot_dir=args.snapshot,
                    weather=args.weather,
                    weather_raw_snapshot_dir=args.weather_snapshot,
                )
        elif args.command == "ingest":
            result = ingest_dataset(
                args.source_dir,
                "opsd-time-series-2020-10-06",
                raw_root=args.root,
                source=DatasetSource(
                    reference=args.reference,
                    licensing_notes="Dataset redistribution permission unresolved",
                ),
            )
            output = {
                "directory": str(result.directory),
                "reused": result.reused,
            }
        elif args.command == "canonicalize":
            result = require_committed_snapshot(materialize(args.snapshot, args.root))
            output = {"directory": str(result.directory), "reused": result.reused}
        elif args.command == "plan":
            result = require_committed_snapshot(
                build_plan(args.canonical, args.root, args.protocol, raw_snapshot_dir=args.snapshot)
            )
            output = {"directory": str(result.directory), "reused": result.reused}
        elif args.command in {"run", "verify-run"}:
            broker_context = _broker_cli_context(args)
            if broker_context is not None:
                use_purpose = (
                    "model_training" if args.command == "run" else "saved_run_verification"
                )
                use_operation = (
                    "energy_run_create" if args.command == "run" else "energy_run_verify"
                )
                canonical_input = _broker_cli_snapshot(
                    args.canonical,
                    args.canonical_artifact_id,
                    "canonical",
                    broker_context,
                    purpose=use_purpose,
                    operation=use_operation,
                )
                plan_input = _broker_cli_snapshot(
                    args.plan,
                    args.plan_artifact_id,
                    "plan",
                    broker_context,
                    purpose=use_purpose,
                    operation=use_operation,
                )
            else:
                canonical_input, plan_input = args.canonical, args.plan
            if args.command == "run":
                experiment_result = run_experiment(
                    canonical_input,
                    plan_input,
                    args.root,
                    raw_snapshot_dir=args.snapshot,
                    broker_context=broker_context,
                )
                if isinstance(experiment_result, PendingPublication):
                    # A CLI worker cannot approve or commit on behalf of the policy actor.
                    # The proposal ID is not a trusted/consumable run reference.
                    output = {
                        "status": "AWAITING_APPROVAL",
                        "publication_id": experiment_result.publication_id,
                        "content_sha256": experiment_result.content_sha256,
                        "artifact_type": experiment_result.artifact_type,
                    }
                else:
                    output = verify_run(
                        experiment_result,
                        canonical=canonical_input,
                        plan=plan_input,
                        raw_snapshot_dir=args.snapshot,
                        broker_context=broker_context,
                    ) | {"directory": str(experiment_result)}
            else:
                run_input = (
                    _broker_cli_snapshot(
                        args.directory,
                        args.run_artifact_id,
                        "run",
                        broker_context,
                        purpose="saved_run_verification",
                        operation="energy_run_verify",
                    )
                    if broker_context is not None
                    else args.directory
                )
                output = verify_run(
                    run_input,
                    canonical=canonical_input,
                    plan=plan_input,
                    raw_snapshot_dir=args.snapshot,
                    broker_context=broker_context,
                )
        elif args.command == "inspect":
            dataset = open_energy(args.directory, raw_snapshot_dir=args.snapshot)
            output = dataset.lineage | {"rows": len(dataset.frame)}
        else:
            raise DataError(
                "Generic context-free verification is unavailable; use a governed reader"
            )
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (DataError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "message": str(exc)}))
        return 1


def _broker_cli_arguments(parser: argparse.ArgumentParser, *, include_run: bool) -> None:
    """Reference IDs are not credentials; role-token *file paths* are explicit inputs."""
    parser.add_argument("--broker-url")
    parser.add_argument("--broker-worker-token-file", type=Path)
    parser.add_argument("--broker-reader-token-file", type=Path)
    parser.add_argument("--broker-materialization-root", type=Path)
    parser.add_argument("--broker-staging-root", type=Path)
    parser.add_argument("--canonical-artifact-id")
    parser.add_argument("--plan-artifact-id")
    if include_run:
        parser.add_argument("--run-artifact-id")


def _broker_cli_context(args: argparse.Namespace) -> BrokerPublicationContext | None:
    names = (
        "broker_url",
        "broker_worker_token_file",
        "broker_reader_token_file",
        "broker_materialization_root",
        "canonical_artifact_id",
        "plan_artifact_id",
    )
    values = tuple(getattr(args, name) for name in names)
    if not any(value is not None for value in values) and (
        getattr(args, "run_artifact_id", None) is None and args.broker_staging_root is None
    ):
        return None
    if any(value is None for value in values):
        raise DataError("Complete broker connection and parent artifact IDs are required")
    if args.command == "verify-run" and args.run_artifact_id is None:
        raise DataError("Broker run artifact ID is required for verification")
    if args.command == "run" and args.broker_staging_root is None:
        raise DataError("Broker staging root is required for publication")
    materialization_root = args.broker_materialization_root
    if not materialization_root.is_dir():
        raise DataError("Broker materialization root must already exist")
    return BrokerPublicationContext(
        BrokerAdapter(
            args.broker_url,
            worker_token_file=args.broker_worker_token_file,
            reader_token_file=args.broker_reader_token_file,
            staging_root=args.broker_staging_root,
        ),
        materialization_root,
    )


def _broker_cli_snapshot(
    path: Path,
    artifact_id: str,
    kind: str,
    context: BrokerPublicationContext,
    *,
    purpose: str,
    operation: str,
) -> DatasetSnapshot:
    """Authorize identity before inspecting a local cached manifest or payload."""
    reference = resolve_broker_reference(
        context.broker, artifact_id, purpose=purpose, operation=operation
    )
    if reference.artifact_type != kind:
        raise DataError("Broker artifact type differs from the requested CLI input")
    if inspect_tree(path / "payload").digest != reference.content_sha256:
        raise DataError("Local CLI input cache differs from the committed broker content")
    checked = verify_snapshot(path)
    if not checked.valid or checked.manifest is None:
        raise DataError("Invalid locally cached broker artifact")
    return DatasetSnapshot(path, checked.manifest, broker_reference=reference)


if __name__ == "__main__":
    raise SystemExit(main())
