"""Recomputed temporal evidence and fail-closed study readiness reports."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from urbangraph.benchmarks.sources import VERSION, Acquisition, sensor_expectation, source_spec
from urbangraph.benchmarks.spatial import inspect_sensor_metadata
from urbangraph.data import DataError, verify_snapshot
from urbangraph.data.filesystem import stream_file
from urbangraph.datasets.materialization import verify_materialization
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.artifacts import code_provenance, load_plan
from urbangraph.evaluation.contracts import BENCHMARK_PROTOCOLS, POSITIONAL_PROTOCOL
from urbangraph.evaluation.dataset import identity, open_dataset
from urbangraph.evaluation.preprocessing import fit_standardizer
from urbangraph.evaluation.temporal import temporal_diagnostics
from urbangraph.evaluation.windows import WindowCollection
from urbangraph.reproducibility import collect_environment_info


def leakage_audit(
    plan_dir: str | Path, canonical_dir: str | Path, *, raw_snapshot_dir: str | Path | None = None
) -> dict[str, object]:
    """Verify all compact origin runs algebraically and preview every run endpoint.

    Cadence is checked over the union of each run's full input-to-target span.
    This covers every sample without materializing overlapping tensors.
    """
    dataset, split, _ = load_plan(
        plan_dir, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    config = split.config
    benchmark = config.protocol.name in BENCHMARK_PROTOCOLS
    positional = config.protocol.name == POSITIONAL_PROTOCOL
    ns = np.asarray(dataset.timestamps.astype("int64"), dtype=np.int64)
    step = config.window.interval_seconds * 1_000_000_000
    checked = 0
    fits: list[dict[str, object]] = []
    for fold_number, fold in enumerate(split.folds):
        prior_stop = -1
        for part in fold.partitions:
            if not benchmark and part.start < prior_stop:
                raise DataError("Strict target partitions overlap")
            prior_stop = part.stop
            windows = WindowCollection(dataset, split, fold=fold_number, partition=part.name)
            offset = 0
            for run in part.samples:
                left = run.start - config.features.observed_delay_steps - config.window.history + 1
                right = run.stop - 1 + config.window.horizon
                if (
                    left < 0
                    or right >= len(ns)
                    or (not positional and np.any(np.diff(ns[left : right + 1]) != step))
                ):
                    raise DataError("Invalid cadence or sample span")
                if not benchmark and (run.start + 1 < part.start or right >= part.stop):
                    raise DataError("Target escaped strict partition")
                length = run.stop - run.start
                for position in {offset, offset + length - 1}:
                    sample = windows[position]
                    if (
                        sample.input_positions[-1]
                        > sample.origin.position - config.features.observed_delay_steps
                        or sample.target_positions[0] <= sample.origin.position
                        or sample.sensor_ids != dataset.sensor_ids
                    ):
                        raise DataError("Origin, latency or sensor-order leakage")
                offset += length
                checked += length
        if config.scaling != "none":
            fitted = fit_standardizer(dataset, split, fold=fold_number)
            train = fold.partition("train")
            if not benchmark and fitted.lineage["fit_rows"] != [train.start, train.stop]:
                raise DataError("Fitted scope differs from training partition")
            fits.append(fitted.lineage)
    try:
        config.features.validate_request("speed", "historical_observed", future=True)
    except DataError:
        pass
    else:
        raise DataError("Future unknown observation accepted")
    result: dict[str, object] = {
        "status": "PASS",
        "protocol": config.protocol.name,
        "samples_checked": checked,
        "folds": len(split.folds),
        "method": "All origin runs checked; endpoints materialized; fitted state recomputed",
        "sample_counts": [{p.name: p.sample_count for p in f.partitions} for f in split.folds],
        "training_fit_lineage": fits,
        "benchmark_caveat": "Sample-first target overlap is intentional; not strict isolation"
        if benchmark
        else None,
        "resolved_configuration": config.to_dict(),
    }
    if positional:
        result["temporal_diagnostics"] = temporal_diagnostics(dataset.timestamps, split)
        result["cadence_claim"] = "No elapsed-time parity claim; row-position benchmark only"
    return result


def readiness(
    dataset_id: str,
    *,
    acquisition: Acquisition | None = None,
    raw_dir: str | Path | None = None,
    canonical_dir: str | Path | None = None,
    plans: dict[str, Path] | None = None,
    metadata_file: str | Path | None = None,
    metadata_layout: str = "locations-header",
    benchmark_protocol: str = "dcrnn-sample-first-v1",
) -> dict[str, object]:
    """Recheck artifacts now; supplied labels or stale report booleans cannot pass gates."""
    spec = source_spec(dataset_id)
    if benchmark_protocol not in BENCHMARK_PROTOCOLS:
        raise DataError("Unknown benchmark protocol")
    gates: dict[str, str] = {
        name: "BLOCKED"
        for name in (
            "source",
            "usage_review",
            "real_file",
            "raw",
            "canonical",
            "sensors",
            "temporal",
            "quality",
            "benchmark",
            "chronological",
            "rolling",
            "leakage",
            "unaltered",
        )
    }
    report: dict[str, object] = {
        "schema_version": VERSION,
        "dataset_id": dataset_id,
        "source_expectations": spec,
        "gates": gates,
        "status": "BLOCKED",
        "evidence_kind": acquisition.evidence_kind if acquisition else "unavailable",
        "acquisition": acquisition.to_dict() if acquisition else None,
        "environment": collect_environment_info().to_dict(),
        "code": code_provenance(),
        "spatial": {"status": "UNKNOWN", "reason": "Canonical axis unavailable"},
    }
    if acquisition:
        gates["source"] = "PASS"
        gates["usage_review"] = "PASS" if acquisition.usage_reviewed else "WARNING"
    if raw_dir is not None:
        raw = verify_snapshot(raw_dir)
        if not raw.valid or raw.manifest is None or raw.manifest.dataset_id != dataset_id:
            gates["raw"] = "FAIL"
        else:
            gates["raw"] = "PASS"
            report["raw_snapshot_id"] = raw.manifest.snapshot_id
            report["raw_manifest_sha256"] = stream_file(Path(raw_dir) / "manifest.json")[1]
            report["raw_source"] = asdict(raw.manifest.source)
            report["source_files"] = [asdict(r) for r in raw.manifest.files]
            # Reuse never rewrites old provenance; conflicting new claims are explicit failures.
            if acquisition and raw.manifest.source.reference != acquisition.reference:
                gates["source"] = "FAIL"
            if acquisition and raw.manifest.source.description != (
                "Local import; evidence kind: " + acquisition.evidence_kind
            ):
                gates["source"] = "FAIL"
    if canonical_dir is not None and raw_dir is not None and gates["raw"] == "PASS":
        canonical = verify_materialization(canonical_dir, raw_snapshot_dir=raw_dir)
        gates["canonical"] = "PASS" if canonical.valid else "FAIL"
        if canonical.valid:
            dataset = open_dataset(canonical_dir, raw_snapshot_dir=raw_dir)
            if dataset.lineage["dataset_id"] != dataset_id:
                gates["canonical"] = "FAIL"
            else:
                quality = read_json(Path(canonical_dir) / "quality.json")
                schema = read_json(Path(canonical_dir) / "schema.json")
                report.update(
                    canonical_id=canonical.transformation_id,
                    canonical_lineage=dataset.lineage,
                    quality=quality,
                    quality_id=identity(quality),
                    schema=schema,
                )
                report["sensor_expectation"] = sensor_expectation(
                    dataset_id, len(dataset.sensor_ids)
                )
                report["spatial"] = inspect_sensor_metadata(
                    metadata_file, dataset.sensor_ids, layout=metadata_layout
                )
                gates.update(sensors="PASS", quality="PASS", unaltered="PASS")
                if len(dataset.sensor_ids) != spec["expected_sensor_count"]:
                    gates["sensors"] = "WARNING"
                if (
                    acquisition
                    and acquisition.evidence_kind == "real"
                    and gates["source"] == "PASS"
                ):
                    gates["real_file"] = "PASS"
                gates["temporal"] = (
                    "PASS"
                    if np.all(
                        np.diff(np.asarray(dataset.timestamps.astype("int64"), dtype=np.int64))
                        == 300_000_000_000
                    )
                    else "WARNING"
                )
    audits: dict[str, object] = {}
    plan_ids: dict[str, str] = {}
    expected = {
        "benchmark": benchmark_protocol,
        "chronological": "strict-chronological-v1",
        "rolling": "strict-rolling-origin-v1",
    }
    if gates["canonical"] == "PASS" and canonical_dir is not None:
        for name, directory in (plans or {}).items():
            if name not in expected:
                raise DataError("Unknown study plan label")
            try:
                audit = leakage_audit(directory, canonical_dir, raw_snapshot_dir=raw_dir)
                if audit["protocol"] != expected[name]:
                    raise DataError("Study plan label differs from verified protocol")
                audits[name] = audit
                plan_ids[name] = directory.name
                gates[name] = "PASS"
            except (DataError, OSError):
                gates[name] = "FAIL"
        if all(gates[name] == "PASS" for name in expected):
            gates["leakage"] = "PASS"
    report["evaluation_plan_ids"] = plan_ids
    report["leakage_audits"] = audits
    report["status"] = next(
        (s for s in ("FAIL", "BLOCKED", "WARNING") if s in gates.values()), "PASS"
    )
    report["research_ready"] = report["status"] == "PASS"
    report["certification_scope"] = (
        "Verified local evidence plus caller attestations; no authenticity or legal guarantee"
    )
    return report
