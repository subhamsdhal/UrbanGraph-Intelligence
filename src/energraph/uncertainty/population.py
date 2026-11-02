"""Reconstruct disjoint predictions from verified fitted EG-002 local models."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from energraph.artifacts import PendingPublication, _publish_governed, governed_payload
from energraph.experiments import code_fingerprint, origins
from energraph.lineage import authorize_sources, resolve_parents
from energraph.regional_data import load_regional_plan
from energraph.regional_research import RegionalFeatures, regional_features, verify_regional_run
from urbangraph.boosting.contracts import config_from_dict
from urbangraph.boosting.models import load_model
from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.preprocessing import FittedStandardizer


def reconstruct(
    canonical: Path,
    plan: Path,
    point_run: Path,
    root: Path,
    fold: int,
    *,
    graph: Path | None = None,
    raw_snapshot_dir: Path | None = None,
) -> Path:
    if graph is None or raw_snapshot_dir is None:
        raise DataError("Complete current population source context required")
    verify_regional_run(
        point_run,
        canonical=canonical,
        plan=plan,
        graph=graph,
        raw_snapshot_dir=raw_snapshot_dir,
    )
    parents = (canonical, plan, point_run)
    closure = resolve_parents(parents)
    if any(
        source not in closure.sources for source in resolve_parents((raw_snapshot_dir,)).sources
    ):
        raise DataError("Population raw source closure mismatch")
    authorize_sources(
        closure.sources,
        purpose="scientific_evaluation",
        operation="energy_population_create",
        destination=root,
    )
    data, split, latency = load_regional_plan(canonical, plan, raw_snapshot_dir=raw_snapshot_dir)
    if not 0 <= fold < len(split.folds):
        raise DataError("Unknown fold")
    desc = read_json(point_run / "payload/descriptor.json")
    spec = object_dict(desc["specification"])
    if spec["plan_id"] != plan.name or spec["canonical"] != data.lineage:
        raise DataError("Point-model plan or canonical mismatch")
    fold_name = split.folds[fold].fold_id
    records = [
        object_dict(r)
        for r in cast(list[object], desc["records"])
        if object_dict(r)["fold"] == fold_name and object_dict(r)["mode"] == "M0"
    ]
    if {str(r["target"]) for r in records} != set(data.sensor_ids) or len(records) != len(
        data.sensor_ids
    ):
        raise DataError("Incomplete point-model region axis")
    specification = {
        "schema_version": "energy-uncertainty-population-1.0",
        "canonical": data.lineage,
        "plan_id": plan.name,
        "point_run": point_run.name,
        "fold": fold,
        "region_ids": list(data.sensor_ids),
        "latencies": latency,
        "code_sha256": code_fingerprint(),
        "protocol": split.config.protocol.name,
        "selection": spec["selection"],
        "units": "MW",
        "horizons": list(range(1, 25)),
        "simple_reference": "last observation available at origin minus region delay",
    }
    experiment_id = identity(specification)
    authorize_sources(
        closure.sources,
        purpose="scientific_evaluation",
        operation="energy_population_read",
        destination=root,
    )
    for path in (root / "energraph-uncertainty-population").glob("*/payload/descriptor.json"):
        if read_json(path).get("experiment_id") == experiment_id:
            governed_payload(
                path.parent.parent,
                {"predictions.npz", "descriptor.json", "runtime.json"},
                parents=parents,
                kind="uncertainty-population",
            )
            return path.parent.parent
    scaler = FittedStandardizer.from_dict(
        read_json(point_run / "payload" / f"{fold_name}-scaler.json")
    )
    scaled = pd.DataFrame(
        scaler.transform(data.frame.to_numpy(dtype=np.float64), sensor_ids=data.sensor_ids),
        index=data.frame.index,
        columns=data.frame.columns,
    )
    arrays: dict[
        str,
        np.ndarray[tuple[int, ...], np.dtype[np.float64]]
        | np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    ] = {}
    audit: dict[str, object] = {}
    began = time.perf_counter()
    for part in ["validation", "calibration", "test"]:
        o = origins(split, fold, part)
        targets = o[:, None] + np.arange(1, 25)
        actual = data.frame.to_numpy(dtype=np.float64)[targets]
        predictions = np.empty_like(actual)
        simple = np.empty_like(actual)
        for j, region in enumerate(data.sensor_ids):
            r = next(r for r in records if r["target"] == region)
            trials = cast(list[object], r["trials"])
            trial = object_dict(trials[int(str(r["selected_trial"]))])
            config = config_from_dict(trial["config"])
            f = RegionalFeatures(region, (region,), (latency[region],), "M0")
            names = tuple(str(n) for n in cast(list[object], r["names"]))
            groups = tuple(str(n) for n in cast(list[object], r["groups"]))
            model = load_model(
                point_run / "payload" / str(trial["model_file"]),
                config,
                f,
                names,
                groups,
                (region,),
                24,
                int(str(object_dict(trial["metadata"])["best_iteration"])),
            )
            for start in range(0, len(o), 128):
                batch = o[start : start + 128]
                x, n, _ = regional_features(
                    scaled,
                    np.repeat(batch, 24),
                    np.tile(np.arange(1, 25, dtype=np.int64), len(batch)),
                    f,
                )
                predictions[start : start + len(batch), :, j] = (
                    model.predict(x, n, (region,)).reshape(-1, 24) * scaler.scales[j]
                    + scaler.means[j]
                )
            simple[:, :, j] = data.frame[region].to_numpy(dtype=np.float64)[
                o - latency[region], None
            ]
            if part == "test":
                old = pd.read_parquet(point_run / "payload" / f"{r['prefix']}-predictions.parquet")
                np.testing.assert_array_equal(old.origin_row.to_numpy(), np.repeat(o, 24))
                np.testing.assert_allclose(
                    old.lightgbm.to_numpy(), predictions[:, :, j].ravel(), rtol=1e-12, atol=1e-9
                )
                np.testing.assert_array_equal(old.actual_MW.to_numpy(), actual[:, :, j].ravel())
        arrays[part + "_origins"] = o
        arrays[part + "_truth"] = actual
        arrays[part + "_lightgbm"] = predictions
        arrays[part + "_persistence"] = simple
        arrays[part + "_targets_ns"] = np.asarray(
            data.timestamps[targets.ravel()].as_unit("ns").astype("int64")
        ).reshape(targets.shape)
        audit[part] = {
            "origin_count": len(o),
            "first_origin": int(o[0]),
            "last_origin": int(o[-1]),
            "first_target_utc": str(data.timestamps[targets[0, 0]]),
            "last_target_utc": str(data.timestamps[targets[-1, -1]]),
            "population_id": identity(
                {"origins": o.tolist(), "regions": data.sensor_ids, "plan": plan.name, "fold": fold}
            ),
        }
    cal = arrays["calibration_origins"]
    test = arrays["test_origins"]
    if int(cal[-1]) + 24 + max(latency.values()) > int(test[0]):
        raise DataError("Calibration labels unavailable at first test origin")
    with tempfile.TemporaryDirectory(prefix="uncertainty-population-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(stage / "predictions.npz", allow_pickle=False, **arrays)
        write_json(
            stage / "descriptor.json",
            {
                "specification": specification,
                "experiment_id": experiment_id,
                "partitions": audit,
                "scaler": scaler.to_dict(),
                "selected_models": records,
                "retrospective_selection": (
                    "LightGBM family chosen using earlier published test evidence; "
                    "no untouched prospective claim"
                ),
            },
        )
        write_json(
            stage / "runtime.json",
            {"inference_seconds": time.perf_counter() - began, "refitted_models": 0},
        )
        result = _publish_governed(stage, root, "uncertainty-population", parents=parents)
        if isinstance(result, PendingPublication):
            raise DataError("Uncertainty population is awaiting separate broker approval")
        return result.directory
