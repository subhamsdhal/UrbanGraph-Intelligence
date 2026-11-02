"""Four energy baselines with validation-only selection and immutable scored evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    PendingPublication,
    _publish_governed,
    broker_parent_references,
    governed_payload,
    require_committed_snapshot,
    resolve_broker_reference,
)
from energraph.broker_tree import inspect_tree
from energraph.evaluation import load_plan
from energraph.lineage import authorize_sources, resolve_parents
from urbangraph.baselines.contracts import FloatArray, IntArray, ModelConfig
from urbangraph.baselines.history import make_batch, training_data
from urbangraph.baselines.models import Baseline, FittedState, fit_model
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.metrics import evaluate_predictions
from urbangraph.reproducibility import collect_environment_info

MODELS = ("persistence", "daily-seasonal", "weekly-seasonal", "autoregressive")


def code_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ("urbangraph", "energraph")
        for p in (root / name).rglob("*.py")
    }
    return identity(hashes)


def origins(split: SplitManifest, fold: int, partition: str) -> IntArray:
    return np.concatenate(
        [
            np.arange(s.start, s.stop, dtype=np.int64)
            for s in split.folds[fold].partition(partition).samples
        ]
    )


def predict(
    model: Baseline, data: EvaluationDataset, split: SplitManifest, fold: int, partition: str
) -> FloatArray:
    population = origins(split, fold, partition)
    values = data.frame.to_numpy(dtype=np.float64)
    chunks = []
    for start in range(0, len(population), 256):
        batch = make_batch(data, split, fold, partition, population[start : start + 256], values)
        chunks.append(model.predict(batch).values)
    return np.concatenate(chunks)


def truth(data: EvaluationDataset, split: SplitManifest, fold: int, partition: str) -> FloatArray:
    pos = origins(split, fold, partition)[:, None] + np.arange(1, 25)
    return np.asarray(data.frame.to_numpy(dtype=np.float64)[pos, ..., None], dtype=np.float64)


def scores(
    actual: FloatArray, predictions: FloatArray, data: EvaluationDataset, split: SplitManifest
) -> dict[str, object]:
    result = evaluate_predictions(
        actual,
        predictions,
        sensor_ids=data.sensor_ids,
        protocol=split.config.protocol.name,
        units="MW",
    )
    denominator = float(np.abs(actual).sum())
    result["wape_percent"] = (
        float(np.abs(actual - predictions).sum() / denominator * 100) if denominator else None
    )
    result["prediction_availability"] = float(np.isfinite(predictions).mean())
    return result


def _run_parent_context(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    raw_snapshot_dir: Path,
    broker_context: BrokerPublicationContext | None,
    *,
    purpose: str,
    operation: str,
) -> tuple[
    Path, Path, tuple[BrokerArtifactRef, BrokerArtifactRef] | None, tuple[dict[str, object], ...]
]:
    """Resolve the two derived parents through broker authority before local inspection."""
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    plan_path = plan.directory if isinstance(plan, DatasetSnapshot) else plan
    if (
        broker_context is not None
        or isinstance(canonical, DatasetSnapshot)
        or isinstance(plan, DatasetSnapshot)
    ):
        if (
            broker_context is None
            or not isinstance(canonical, DatasetSnapshot)
            or not isinstance(plan, DatasetSnapshot)
            or canonical.broker_reference is None
            or plan.broker_reference is None
        ):
            raise DataError("Committed canonical and plan broker identities are required")
        refs = (canonical.broker_reference, plan.broker_reference)
        if refs[0].artifact_type != "canonical" or refs[1].artifact_type != "plan":
            raise DataError("Incorrect broker parent artifact types")
        broker = broker_context.broker
        for reference in refs:
            if (
                resolve_broker_reference(
                    broker, reference.artifact_id, purpose=purpose, operation=operation
                )
                != reference
            ):
                raise DataError("Broker parent identity changed")
        canonical_parents = broker_parent_references(
            broker, refs[0], purpose=purpose, operation=operation
        )
        if len(canonical_parents) != 1 or canonical_parents[0].artifact_type != "raw":
            raise DataError("Canonical broker raw parent is invalid")
        if broker_parent_references(broker, refs[1], purpose=purpose, operation=operation) != (
            refs[0],
        ):
            raise DataError("Plan broker parent differs from canonical")
        if inspect_tree(raw_snapshot_dir).digest != canonical_parents[0].content_sha256:
            raise DataError("Raw source differs from broker ancestor")
        # Only inspect locally cached derived payload after the broker grants access.
        for path, reference in zip((canonical_path, plan_path), refs, strict=True):
            if inspect_tree(path / "payload").digest != reference.content_sha256:
                raise DataError("Local derived cache differs from committed broker artifact")
        graph = resolve_parents((raw_snapshot_dir,))
        return canonical_path, plan_path, refs, graph.sources
    # Local-only paths remain deliberately default-denied for derived parents.
    graph = resolve_parents((canonical_path, plan_path))
    if graph.sources != resolve_parents((raw_snapshot_dir,)).sources:
        raise DataError("Energy run source closure differs from raw context")
    return canonical_path, plan_path, None, graph.sources


def run_experiment(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    root: Path,
    *,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> Path | PendingPublication:
    if raw_snapshot_dir is None:
        raise DataError("Current raw source context required for energy training")
    canonical_path, plan_path, refs, sources = _run_parent_context(
        canonical,
        plan,
        raw_snapshot_dir,
        broker_context,
        purpose="model_training",
        operation="energy_run_create",
    )
    authorize_sources(
        sources, purpose="model_training", operation="energy_run_create", destination=root
    )
    data, split = load_plan(
        canonical, plan, raw_snapshot_dir=raw_snapshot_dir, broker_context=broker_context
    )
    specification = {
        "schema_version": "energy-experiment-1.0",
        "canonical": data.lineage,
        "plan_id": plan_path.name,
        "split_id": identity(split.to_dict()),
        "code_sha256": code_fingerprint(),
        "models": list(MODELS),
        "ar_candidates": [{"lags": 24, "ridge": 10000.0}, {"lags": 168, "ridge": 10000.0}],
        "selection": "minimum validation MAE; first candidate wins ties",
        "mask": "finite zero-valid; MAPE excludes zero denominators",
        "interval_seconds": 3600,
        "latency_hours": 24,
        "seed": None,
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "pyarrow")},
    }
    experiment_id = identity(specification)
    destination = root
    # The verified source closure is trusted request context. Check current policy
    # for the requested destination before inspecting protected saved descriptors;
    # verify_run repeats authorization for the exact saved directory.
    authorize_sources(
        sources,
        purpose="saved_run_verification",
        operation="energy_run_verify",
        destination=destination,
    )
    if refs is not None and any((destination / "energraph-run").glob("*/manifest.json")):
        raise DataError(
            "Matching completed-run reuse is unavailable without independently attested "
            "runner authorship; verify a committed broker artifact instead"
        )
    existing = (
        [
            p
            for p in (destination / "energraph-run").glob("*/manifest.json")
            if read_json(p.parent / "payload" / "descriptor.json").get("experiment_id")
            == experiment_id
        ]
        if refs is None
        else []
    )
    if existing:
        if len(existing) != 1:
            raise DataError("Conflicting existing energy experiments")
        verify_run(
            existing[0].parent,
            canonical=canonical,
            plan=plan,
            raw_snapshot_dir=raw_snapshot_dir,
        )
        raise DataError(
            "Matching saved experiment passed integrity inspection, but completed-run reuse "
            "is unavailable without independently attested runner authorship; use verify-run "
            "for integrity-only inspection or select a fresh output root"
        )
    folds = []
    durations = []
    with tempfile.TemporaryDirectory(prefix="energraph-experiment-") as tmp:
        stage = Path(tmp)
        for f, fold in enumerate(split.folds):
            trained = training_data(data, split, f)
            val_truth = truth(data, split, f, "validation")
            candidates = []
            trials = []
            for lag in (24, 168):
                config = ModelConfig(
                    "autoregressive", history_fallback=168, lags=lag, ridge=10000.0
                )
                began = time.perf_counter()
                model = fit_model(config, trained, 24, interval_seconds=3600)
                duration = time.perf_counter() - began
                val = predict(model, data, split, f, "validation")
                score = float(np.abs(val - val_truth).mean())
                if (
                    not np.isfinite(score)
                    or model.state is None
                    or not (model.state.support > 0).all()
                ):
                    raise DataError("Autoregression trial unsupported or nonfinite")
                trials.append(
                    {
                        "config": config.to_dict(),
                        "validation_mae": score,
                        "state_id": model.state.state_id,
                    }
                )
                durations.append(
                    {
                        "fold": fold.fold_id,
                        "lags": lag,
                        "fit_seconds": duration,
                        "training_windows": model.state.support.tolist(),
                    }
                )
                candidates.append(model)
            chosen = int(np.argmin([float(str(t["validation_mae"])) for t in trials]))
            selected = candidates[chosen]
            if selected.state is None:
                raise DataError("Fitted model missing state")
            state_name = f"{fold.fold_id}-state.json"
            write_json(stage / state_name, selected.state.to_dict())
            restored = Baseline(
                selected.config, FittedState.from_dict(read_json(stage / state_name)), 3600
            )
            actual = truth(data, split, f, "test")
            population = origins(split, f, "test")
            positions = population[:, None] + np.arange(1, 25)
            table = pd.DataFrame(
                {
                    "origin_row": np.repeat(population, 24),
                    "target_row": positions.ravel(),
                    "target_utc": data.timestamps[positions.ravel()],
                    "actual_MW": actual.ravel(),
                }
            )
            reports = {}
            for name in MODELS:
                model = (
                    restored
                    if name == "autoregressive"
                    else Baseline(
                        ModelConfig(name, history_fallback=168, fallback="unavailable"),
                        interval_seconds=3600,
                    )
                )
                started = time.perf_counter()
                predictions = predict(model, data, split, f, "test")
                durations.append(
                    {
                        "fold": fold.fold_id,
                        "model": name,
                        "test_prediction_seconds": time.perf_counter() - started,
                    }
                )
                if not np.isfinite(predictions).all():
                    raise DataError("Incomplete prediction population; no silent mask reduction")
                if name == "autoregressive":
                    np.testing.assert_array_equal(
                        predictions, predict(selected, data, split, f, "test")
                    )
                table[name] = predictions.ravel()
                reports[name] = scores(actual, predictions, data, split)
            table.to_parquet(stage / f"{fold.fold_id}-predictions.parquet", index=False)
            folds.append(
                {
                    "fold": fold.fold_id,
                    "forecast_origins": len(population),
                    "origin_sha256": hashlib.sha256(population.astype("<i8").tobytes()).hexdigest(),
                    "target_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
                    "models": reports,
                    "trials": trials,
                    "selected_trial": chosen,
                    "fitted_state": state_name,
                    "training_lineage": trained.lineage,
                }
            )
        write_json(
            stage / "descriptor.json",
            {
                "specification": specification,
                "experiment_id": experiment_id,
                "folds": folds,
                "split": split.to_dict(),
            },
        )
        write_json(
            stage / "runtime.json",
            {"timings": durations, "environment": collect_environment_info().to_dict()},
        )
        published = _publish_governed(
            stage,
            destination,
            "run",
            parents=(canonical_path, plan_path),
            broker_context=broker_context if refs is not None else None,
            broker_parents=refs,
        )
    if isinstance(published, PendingPublication):
        return published  # A separate policy actor must approve before commit.
    result = require_committed_snapshot(published)
    verify_run(
        result.directory,
        canonical=canonical,
        plan=plan,
        raw_snapshot_dir=raw_snapshot_dir,
    )
    return result.directory


def verify_run(
    directory: Path | DatasetSnapshot,
    *,
    canonical: Path | DatasetSnapshot | None = None,
    plan: Path | DatasetSnapshot | None = None,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> dict[str, object]:
    if canonical is None or plan is None or raw_snapshot_dir is None:
        raise DataError("Current complete source context required for energy run verification")
    run_path = directory.directory if isinstance(directory, DatasetSnapshot) else directory
    canonical_path, plan_path, refs, sources = _run_parent_context(
        canonical,
        plan,
        raw_snapshot_dir,
        broker_context,
        purpose="saved_run_verification",
        operation="energy_run_verify",
    )
    authorize_sources(
        sources,
        purpose="saved_run_verification",
        operation="energy_run_verify",
        destination=run_path,
    )
    if refs is not None:
        if (
            broker_context is None
            or not isinstance(directory, DatasetSnapshot)
            or directory.broker_reference is None
            or directory.broker_reference.artifact_type != "run"
        ):
            raise DataError("Committed broker run identity is required for verification")
        run_ref = directory.broker_reference
        broker = broker_context.broker
        if (
            resolve_broker_reference(
                broker,
                run_ref.artifact_id,
                purpose="saved_run_verification",
                operation="energy_run_verify",
            )
            != run_ref
        ):
            raise DataError("Run broker identity changed")
        run_parents = broker_parent_references(
            broker, run_ref, purpose="saved_run_verification", operation="energy_run_verify"
        )
        if tuple(sorted(run_parents, key=lambda ref: ref.artifact_id)) != tuple(
            sorted(refs, key=lambda ref: ref.artifact_id)
        ):
            raise DataError("Run broker parents differ from supplied plan and canonical")
        record = broker.authorize_directory(
            run_ref.artifact_id,
            purpose="saved_run_verification",
            operation="energy_run_verify",
        )
        # Fetch the descriptor from the broker under current rights, before
        # opening any untrusted local run descriptor or calculating expected files.
        descriptor_bytes = broker.read_directory_file(
            record,
            "descriptor.json",
            purpose="saved_run_verification",
            operation="energy_run_verify",
        )
        try:
            descriptor = json.loads(descriptor_bytes)
        except (ValueError, UnicodeError) as exc:
            raise DataError("Invalid broker-delivered run descriptor") from exc
        if not isinstance(descriptor, dict):
            raise DataError("Invalid broker-delivered run descriptor")
    else:
        descriptor = read_json(run_path / "payload" / "descriptor.json")
    spec = object_dict(descriptor["specification"])
    if (
        spec.get("schema_version") != "energy-experiment-1.0"
        or identity(spec) != descriptor["experiment_id"]
    ):
        raise DataError("Energy experiment identity mismatch")
    folds = descriptor["folds"]
    if not isinstance(folds, list) or not folds:
        raise DataError("Missing energy folds")
    expected = {"descriptor.json", "runtime.json"}
    for item in folds:
        fold = object_dict(item)
        name = str(fold["fold"])
        if not name.startswith("fold-") or not name[5:].isdigit():
            raise DataError("Invalid fold name")
        expected.update({f"{name}-state.json", f"{name}-predictions.parquet"})
    if refs is not None:
        assert (
            broker_context is not None
            and isinstance(directory, DatasetSnapshot)
            and directory.broker_reference is not None
        )
        base = governed_payload(
            directory.broker_reference,
            expected,
            kind="run",
            broker=broker_context.broker,
            broker_parents=refs,
            materialization_root=broker_context.materialization_root,
        )
        if inspect_tree(run_path / "payload").digest != directory.broker_reference.content_sha256:
            raise DataError("Local run cache differs from authorized broker content")
    else:
        base = governed_payload(run_path, expected, parents=(canonical_path, plan_path), kind="run")
    for item in folds:
        fold = object_dict(item)
        if fold["fitted_state"] != f"{fold['fold']}-state.json":
            raise DataError("Unsafe fitted state reference")
        state = FittedState.from_dict(read_json(base / str(fold["fitted_state"])))
        trials = fold.get("trials")
        chosen = fold.get("selected_trial")
        if not isinstance(trials, list) or type(chosen) is not int or not 0 <= chosen < len(trials):
            raise DataError("Invalid selected validation trial")
        validation_scores = [float(str(object_dict(t)["validation_mae"])) for t in trials]
        if not np.isfinite(validation_scores).all() or chosen != int(np.argmin(validation_scores)):
            raise DataError("Selection disagrees with validation scores")
        if state.state_id != object_dict(trials[chosen])["state_id"]:
            raise DataError("Selected state lineage mismatch")
        variable = object_dict(object_dict(spec["canonical"])["variable"])
        if state.sensor_ids != (str(variable["column"]),):
            raise DataError("Energy fitted sensor identity mismatch")
        table = pd.read_parquet(base / f"{fold['fold']}-predictions.parquet")
        if (
            len(table) != int(str(fold["forecast_origins"])) * 24
            or not np.isfinite(table["actual_MW"]).all()
        ):
            raise DataError("Energy score population invalid")
        population = np.asarray(table["origin_row"], dtype=np.int64).reshape(-1, 24)
        targets = np.asarray(table["target_row"], dtype=np.int64).reshape(-1, 24)
        if not np.all(population == population[:, :1]) or not np.array_equal(
            targets, population[:, :1] + np.arange(1, 25)
        ):
            raise DataError("Forecast offsets mismatch")
        if (
            hashlib.sha256(population[:, 0].astype("<i8").tobytes()).hexdigest()
            != fold["origin_sha256"]
        ):
            raise DataError("Origin population hash mismatch")
        if hashlib.sha256(targets.astype("<i8").tobytes()).hexdigest() != fold["target_sha256"]:
            raise DataError("Target population hash mismatch")
        a = np.asarray(table["actual_MW"], dtype=np.float64).reshape(-1, 24, 1, 1)
        for model in MODELS:
            p = np.asarray(table[model], dtype=np.float64).reshape(a.shape)
            stored = object_dict(object_dict(fold["models"])[model])
            recomputed = evaluate_predictions(
                a,
                p,
                sensor_ids=state.sensor_ids,
                protocol=str(
                    object_dict(
                        object_dict(object_dict(descriptor["split"])["config"])["protocol"]
                    )["name"]
                ),
                units="MW",
            )
            denominator = float(np.abs(a).sum())
            wape = float(np.abs(a - p).sum() / denominator * 100) if denominator else None
            if stored.get("wape_percent") != wape or stored.get("prediction_availability") != 1.0:
                raise DataError("Energy WAPE or availability mismatch")
            if (
                recomputed["overall"] != stored["overall"]
                or recomputed["by_horizon"] != stored["by_horizon"]
            ):
                raise DataError("Saved metrics disagree with prediction evidence")
    return {
        "valid": True,
        "assurance": "integrity_and_internal_metric_consistency_only",
        "integrity": "pass",
        "internal_metric_consistency": "pass",
        "source_truth_agreement": "not_established",
        "authorship_assurance": "unverified",
        "scientific_validation": "not_established",
        "production_approval": "not_approved",
        "snapshot_id": run_path.name,
        "experiment_id": descriptor["experiment_id"],
        "fold_count": len(folds),
    }
