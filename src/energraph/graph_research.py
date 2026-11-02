"""Sequential regional graph experiments using the existing trainer and immutable store."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from energraph.artifacts import (
    BrokerPublicationContext,
    PendingPublication,
    _publish_governed,
    broker_parent_references,
    governed_payload,
    require_committed_snapshot,
    resolve_broker_reference,
)
from energraph.broker_tree import inspect_tree
from energraph.energy_boosting import error_summary
from energraph.experiments import code_fingerprint
from energraph.graph_authority import authorize_graph_inputs, snapshot_path
from energraph.graph_binding import bound_graph
from energraph.graph_models import EnergyForecaster, EnergyModelConfig
from energraph.graph_windows import OBJECTIVE, RegionalWindows, normalized_validation
from energraph.lineage import FileParent, authorize_sources, resolve_parents
from energraph.regional_data import load_regional_plan
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.preprocessing import FittedStandardizer, fit_standardizer
from urbangraph.neural.checkpointing import load_checkpoint
from urbangraph.neural.dataloaders import make_loader
from urbangraph.neural.trainer import fit_model
from urbangraph.reproducibility import collect_environment_info
from urbangraph.stgnn.contracts import TrainingConfig


def metric_report(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    mask: NDArray[np.bool_],
    regions: tuple[str, ...],
    scales: tuple[float, ...],
) -> dict[str, object]:
    if (
        actual.shape != predicted.shape
        or mask.shape != actual.shape
        or actual.ndim != 3
        or actual.shape[1:] != (24, len(regions))
        or len(scales) != len(regions)
        or mask.dtype != np.bool_
        or any(not np.isfinite(s) or s <= 0 for s in scales)
    ):
        raise DataError("Invalid regional metric axes")
    if not np.isfinite(predicted).all() or np.any(mask & ~np.isfinite(actual)):
        raise DataError("Invalid target or prediction; no silent mask reduction")
    if not mask.any(axis=(0, 1)).all() or not mask.any(axis=(0, 2)).all():
        raise DataError("Every reported region and horizon needs valid targets")
    regional = {
        r: error_summary(actual[:, :, i][mask[:, :, i]], predicted[:, :, i][mask[:, :, i]])
        for i, r in enumerate(regions)
    }
    normalized = [
        float(np.abs(actual[:, :, i] - predicted[:, :, i])[mask[:, :, i]].mean() / scales[i])
        for i in range(len(regions))
    ]
    return {
        "micro": error_summary(actual[mask], predicted[mask]),
        "regions": regional,
        "horizons": {
            str(h + 1): error_summary(actual[:, h][mask[:, h]], predicted[:, h][mask[:, h]])
            for h in range(24)
        },
        "macro_mae_MW": float(np.mean([float(str(v["mae_MW"])) for v in regional.values()])),
        "macro_rmse_MW": float(np.mean([float(str(v["rmse_MW"])) for v in regional.values()])),
        "macro_normalized_mae": float(np.mean(normalized)),
        "normalized_by_region": dict(zip(regions, normalized, strict=True)),
    }


def run_graph_experiment(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    graph: Path | DatasetSnapshot,
    root: Path,
    condition: str,
    fold: int,
    config: TrainingConfig,
    *,
    widths: tuple[int, ...] = (8, 16),
    blocks: int = 4,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    saved_run: DatasetSnapshot | None = None,
    geography: Path | None = None,
    geography_reference: str | None = None,
    geometry_reference: str | None = None,
) -> Path | DatasetSnapshot | PendingPublication:
    if raw_snapshot_dir is None:
        raise DataError("Current raw source context required for neural training")
    if broker_context is not None and geography is not None:
        raise DataError("Broker-backed geography needs committed file evidence")
    trusted = authorize_graph_inputs(
        canonical,
        plan,
        graph=graph,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context,
        purpose="model_training",
        operation="energy_neural_train",
    )
    parents: tuple[Path | FileParent, ...] = tuple(
        snapshot_path(value) for value in (canonical, plan, graph)
    )
    sources = (
        resolve_parents((raw_snapshot_dir,)).sources
        if trusted is not None
        else resolve_parents(parents).sources
    )
    authorize_sources(
        sources,
        purpose="model_training",
        operation="energy_neural_train",
        destination=root / "runs",
    )
    if saved_run is not None:
        if trusted is None or broker_context is None or saved_run.broker_reference is None:
            raise DataError("A saved broker run requires broker context and committed identity")
        run_ref = saved_run.broker_reference
        if run_ref.artifact_type != "neural-regional":
            raise DataError("Saved broker artifact is not a regional neural run")
        # Check current saved-run rights and its committed parent identity before
        # loading any protected training payload. A revoked run grant must fail
        # without reading canonical, plan, graph, or run descriptors.
        authorized_run = resolve_broker_reference(
            broker_context.broker,
            run_ref.artifact_id,
            purpose="saved_run_verification",
            operation="energy_neural_verify",
        )
        if authorized_run != run_ref:
            raise DataError("Saved neural run broker identity changed")
        run_parents = broker_parent_references(
            broker_context.broker,
            run_ref,
            purpose="saved_run_verification",
            operation="energy_neural_verify",
        )
        if set(run_parents) != set(trusted):
            raise DataError("Neural run broker parents differ from training inputs")
    if (
        not widths
        or len(set(widths)) != len(widths)
        or config.calendar
        or config.scaling != "per_sensor"
    ):
        raise DataError("Explicit unique search widths and per-region scaling required")
    data, split, latencies = load_regional_plan(
        canonical, plan, raw_snapshot_dir=raw_snapshot_dir, broker_context=broker_context
    )
    adjacency, graph_desc = bound_graph(
        canonical,
        plan,
        graph,
        fold,
        condition,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context,
        geography=geography,
        geography_reference=geography_reference,
        geometry_reference=geometry_reference,
    )
    architectures = [EnergyModelConfig(condition, w, blocks) for w in widths]
    scaler = fit_standardizer(data, split, fold=fold)
    train = RegionalWindows(
        data,
        split,
        fold,
        "train",
        scaler,
        latencies,
        maximum=config.max_train_windows,
        seed=config.seed,
    )
    validation = RegionalWindows(
        data,
        split,
        fold,
        "validation",
        scaler,
        latencies,
        maximum=config.max_validation_windows,
        seed=config.seed,
    )
    test = RegionalWindows(data, split, fold, "test", scaler, latencies)
    specification = {
        "schema_version": "energy-graph-experiment-1.0",
        "canonical_id": snapshot_path(canonical).name,
        "canonical_lineage": data.lineage,
        "plan_id": snapshot_path(plan).name,
        "split_id": identity(split.to_dict()),
        "graph_snapshot": snapshot_path(graph).name,
        "graph": graph_desc,
        "fold": fold,
        "condition": condition,
        "architectures": [a.to_dict() for a in architectures],
        "training": config.to_dict(),
        "objective": OBJECTIVE,
        "region_order": list(data.sensor_ids),
        "scaler": scaler.to_dict(),
        "training_population": train.report,
        "validation_population": validation.report,
        "test_population": test.report,
        "code_sha256": code_fingerprint(),
        "torch": str(torch.__version__),
    }
    experiment_id = identity(specification)
    authorize_sources(
        sources,
        purpose="saved_run_verification",
        operation="energy_neural_verify",
        destination=root / "runs",
    )
    if trusted is not None:
        # Broker-backed saved search requires an explicit committed artifact ID.
        # Never inspect local descriptor paths to discover protected runs.
        if saved_run is not None:
            saved_result = verify_graph_run(
                saved_run,
                canonical=canonical,
                plan=plan,
                graph=graph,
                raw_snapshot_dir=raw_snapshot_dir,
                broker_context=broker_context,
            )
            if saved_result["experiment_id"] != experiment_id:
                raise DataError("Saved neural run is for another experiment")
            return saved_run
    else:
        if saved_run is not None:
            raise DataError("A saved broker run requires broker context")
        for existing in (root / "runs" / "energraph-neural-regional").glob(
            "*/payload/descriptor.json"
        ):
            if read_json(existing).get("experiment_id") == experiment_id:
                verify_graph_run(
                    existing.parent.parent,
                    canonical=canonical,
                    plan=plan,
                    graph=graph,
                    raw_snapshot_dir=raw_snapshot_dir,
                )
                return existing.parent.parent
    started = time.perf_counter()
    deadline = started + config.run_seconds
    reports = []
    models = []
    checkpoints = []
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, architecture in enumerate(architectures):
        checkpoint_root = root / "checkpoints" / experiment_id / f"trial-{index}"
        epochs = sorted(checkpoint_root.glob("epoch-*"))
        result = fit_model(
            train,
            validation,
            config,
            checkpoint_root,
            {
                "experiment_id": experiment_id,
                "architecture": architecture.to_dict(),
                "objective": OBJECTIVE,
            },
            model_factory=partial(EnergyForecaster, architecture, adjacency, data.sensor_ids),
            validation_function=normalized_validation,
            validation_identity=OBJECTIVE,
            resume=epochs[-1] if epochs else None,
            deadline=deadline,
        )
        reports.append(result.report)
        models.append(result.model)
        checkpoints.append(result.checkpoint)
    selected = int(np.argmin([float(str(r["best_validation_mae"])) for r in reports]))
    model = models[selected]
    model.eval()
    predicted = []
    masks = []
    origin_chunks = []
    inference_started = time.perf_counter()
    with torch.no_grad():
        for x, _, mask, origin in make_loader(test, config):
            normalized = model(x.to(config.device)).cpu().numpy()[..., 0].astype(np.float64)
            predicted.append(scaler.inverse_transform(normalized, sensor_ids=data.sensor_ids))
            masks.append(mask.numpy()[..., 0])
            origin_chunks.append(origin.numpy())
    inference_seconds = time.perf_counter() - inference_started
    prediction = np.concatenate(predicted)
    valid = np.concatenate(masks)
    origins = np.concatenate(origin_chunks)
    truth = test.raw_targets(origins)
    measured = metric_report(truth, prediction, valid, data.sensor_ids, scaler.scales)
    population_id = identity(
        {
            "plan": snapshot_path(plan).name,
            "fold": fold,
            "origins": origins.tolist(),
            "regions": list(data.sensor_ids),
            "mask": valid.tolist(),
        }
    )
    with tempfile.TemporaryDirectory(prefix="eg3-") as tmp:
        stage = Path(tmp)
        for i, checkpoint in enumerate(checkpoints):
            shutil.copytree(checkpoint, stage / f"trial-{i}")
        np.savez(
            stage / "forecasts.npz", actual=truth, predicted=prediction, mask=valid, origins=origins
        )
        np.savez(stage / "graph.npz", adjacency=adjacency)
        probe = torch.stack([test[i].x for i in range(min(3, len(test)))])
        with torch.no_grad():
            probe_y = model(probe.to(config.device)).cpu().numpy()
        np.savez(stage / "probe.npz", x=probe.numpy(), predicted=probe_y)
        write_json(
            stage / "descriptor.json",
            {
                "specification": specification,
                "experiment_id": experiment_id,
                "selected_trial": selected,
                "trials": reports,
                "metrics": measured,
                "population_id": population_id,
            },
        )
        write_json(
            stage / "runtime.json",
            {
                "environment": collect_environment_info().to_dict(),
                "seconds": time.perf_counter() - started,
                "inference_seconds": inference_seconds,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if config.device == "cuda"
                else None,
            },
        )
        publication = _publish_governed(
            stage,
            root / "runs",
            "neural-regional",
            parents=parents,
            broker_context=broker_context if trusted is not None else None,
            broker_parents=trusted,
        )
    if isinstance(publication, PendingPublication):
        return publication
    artifact = require_committed_snapshot(publication)
    verify_graph_run(
        artifact.directory,
        canonical=canonical,
        plan=plan,
        graph=graph,
        raw_snapshot_dir=raw_snapshot_dir,
    )
    return artifact.directory


def verify_graph_run(
    directory: Path | DatasetSnapshot,
    *,
    canonical: Path | DatasetSnapshot | None = None,
    plan: Path | DatasetSnapshot | None = None,
    graph: Path | DatasetSnapshot | None = None,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> dict[str, object]:
    if canonical is None or plan is None or graph is None or raw_snapshot_dir is None:
        raise DataError("Current neural run source context required")
    run_path = snapshot_path(directory)
    trusted = authorize_graph_inputs(
        canonical,
        plan,
        graph=graph,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context,
        purpose="saved_run_verification",
        operation="energy_neural_verify",
    )
    sources = (
        resolve_parents((raw_snapshot_dir,)).sources
        if trusted
        else resolve_parents(
            tuple(snapshot_path(value) for value in (canonical, plan, graph))
        ).sources
    )
    authorize_sources(
        sources,
        purpose="saved_run_verification",
        operation="energy_neural_verify",
        destination=run_path,
    )
    if trusted is not None:
        if (
            broker_context is None
            or not isinstance(directory, DatasetSnapshot)
            or directory.broker_reference is None
            or directory.broker_reference.artifact_type != "neural-regional"
        ):
            raise DataError("Committed neural run broker identity required")
        run_ref = directory.broker_reference
        broker = broker_context.broker
        if (
            resolve_broker_reference(
                broker,
                run_ref.artifact_id,
                purpose="saved_run_verification",
                operation="energy_neural_verify",
            )
            != run_ref
        ):
            raise DataError("Neural run broker identity changed")
        direct = broker_parent_references(
            broker, run_ref, purpose="saved_run_verification", operation="energy_neural_verify"
        )
        if set(direct) != set(trusted):
            raise DataError("Neural run broker parents differ from training inputs")
        record = broker.authorize_directory(
            run_ref.artifact_id, purpose="saved_run_verification", operation="energy_neural_verify"
        )
        data = broker.read_directory_file(
            record,
            "descriptor.json",
            purpose="saved_run_verification",
            operation="energy_neural_verify",
        )
        try:
            desc = json.loads(data)
        except (ValueError, UnicodeError) as exc:
            raise DataError("Invalid broker-delivered neural descriptor") from exc
        if not isinstance(desc, dict):
            raise DataError("Invalid broker-delivered neural descriptor")
        # Detect tampering of the optional local run cache only after broker rights.
        if inspect_tree(run_path / "payload").digest != run_ref.content_sha256:
            raise DataError("Local neural run cache differs from committed broker artifact")
    else:
        if broker_context is not None or (
            isinstance(directory, DatasetSnapshot) and directory.broker_reference is not None
        ):
            raise DataError("Complete broker neural run context required")
        desc = read_json(run_path / "payload" / "descriptor.json")
    spec = object_dict(desc["specification"])
    if (
        spec.get("schema_version") != "energy-graph-experiment-1.0"
        or identity(spec) != desc["experiment_id"]
        or spec["objective"] != OBJECTIVE
    ):
        raise DataError("Invalid regional neural specification")
    regions = tuple(str(x) for x in cast(list[object], spec["region_order"]))
    architectures = cast(list[object], spec["architectures"])
    reports = cast(list[object], desc["trials"])
    expected_files = {
        "descriptor.json",
        "runtime.json",
        "forecasts.npz",
        "graph.npz",
        "probe.npz",
    } | {
        f"trial-{index}/{name}"
        for index in range(len(architectures))
        for name in ("checkpoint.json", "state.pt")
    }
    if trusted is not None:
        assert broker_context is not None and isinstance(directory, DatasetSnapshot)
        assert directory.broker_reference is not None
        base = governed_payload(
            directory.broker_reference,
            expected_files,
            kind="neural-regional",
            broker=broker_context.broker,
            broker_parents=trusted,
            materialization_root=broker_context.materialization_root,
        )
    else:
        base = governed_payload(
            run_path,
            expected_files,
            parents=tuple(snapshot_path(value) for value in (canonical, plan, graph)),
            kind="neural-regional",
        )
    chosen = desc["selected_trial"]
    if (
        type(chosen) is not int
        or len(reports) != len(architectures)
        or chosen
        != int(np.argmin([float(str(object_dict(r)["best_validation_mae"])) for r in reports]))
    ):
        raise DataError("Invalid validation-only candidate selection")
    scaler = FittedStandardizer.from_dict(spec["scaler"])
    with np.load(base / "graph.npz", allow_pickle=False) as saved:
        adjacency = saved["adjacency"]
    for i, architecture in enumerate(architectures):
        state, metadata = load_checkpoint(base / f"trial-{i}")
        lineage = object_dict(metadata["lineage"])
        if (
            lineage["experiment_id"] != desc["experiment_id"]
            or metadata.get("validation_identity") != OBJECTIVE
            or identity(metadata["scaler"]) != identity(spec["scaler"])
            or metadata["sensor_ids"] != list(regions)
            or identity(metadata["config"]) != identity(spec["training"])
            or identity(metadata["training_population"]) != identity(spec["training_population"])
            or identity(metadata["validation_population"])
            != identity(spec["validation_population"])
        ):
            raise DataError("Checkpoint provenance mismatch")
        report = object_dict(reports[i])
        history = cast(list[dict[str, object]], state["history"])
        scores = [
            float(str(object_dict(object_dict(h["validation"])["mae"])["value"])) for h in history
        ]
        best = int(np.argmin(scores))
        if (
            state["best_epoch"] != history[best]["epoch"]
            or state["best_mae"] != scores[best]
            or report["best_validation_mae"] != scores[best]
            or report["best_epoch"] != state["best_epoch"]
            or report["checkpoint_id"] != metadata["checkpoint_id"]
        ):
            raise DataError("Checkpoint validation selection mismatch")
        conf = EnergyModelConfig(**object_dict(architecture))  # type: ignore[arg-type]
        if lineage["architecture"] != conf.to_dict():
            raise DataError("Checkpoint architecture mismatch")
        model = EnergyForecaster(conf, adjacency, regions)
        expected_graph = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if key in {"support", "edges", "weights"}
        }
        model.load_state_dict(cast(dict[str, Tensor], state["best_model"]))
        if any(
            not torch.equal(model.state_dict()[key], value) for key, value in expected_graph.items()
        ):
            raise DataError("Checkpoint graph mismatch")
        model.eval()
        if i == chosen:
            with np.load(base / "probe.npz", allow_pickle=False) as probe, torch.no_grad():
                np.testing.assert_allclose(
                    model(torch.from_numpy(probe["x"])).numpy(),
                    probe["predicted"],
                    rtol=1e-4,
                    atol=1e-5,
                )
    with np.load(base / "forecasts.npz", allow_pickle=False) as saved:
        actual = saved["actual"]
        predicted = saved["predicted"]
        mask = saved["mask"]
        origins = saved["origins"]
        metrics = metric_report(actual, predicted, mask, regions, scaler.scales)
        population = identity(
            {
                "plan": spec["plan_id"],
                "fold": spec["fold"],
                "origins": origins.tolist(),
                "regions": list(regions),
                "mask": mask.tolist(),
            }
        )
    if identity(metrics) != identity(desc["metrics"]) or population != desc["population_id"]:
        raise DataError("Forecast metrics/population mismatch")
    return {
        "valid": True,
        "snapshot_id": run_path.name,
        "experiment_id": desc["experiment_id"],
        "selected_trial": chosen,
        "condition": spec["condition"],
        "fold": spec["fold"],
        "metrics": metrics,
    }
