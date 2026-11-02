from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from energraph.uncertainty.conformal import CalibrationPolicy, finite_quantile, fit_calibrator
from energraph.uncertainty.metrics import interval_metrics
from urbangraph.data import DataError


@pytest.mark.parametrize("coverage,rank", [(0.9, 91), (0.95, 96)])
def test_exact_rank(coverage: float, rank: int) -> None:
    q, k = finite_quantile(np.arange(100, dtype=np.float64), coverage)
    assert k == rank and q == rank - 1


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, float("nan")])
def test_invalid_coverage(coverage: float) -> None:
    with pytest.raises(DataError):
        CalibrationPolicy(coverage=coverage)


@pytest.mark.parametrize("grouping", ["global", "region", "horizon", "region-horizon"])
def test_grouping_and_temporal_isolation(grouping: str) -> None:
    truth = np.ones((100, 24, 2), dtype=np.float64)
    truth[:, :, 1] *= 4
    prediction = np.zeros_like(truth)
    mask = np.ones_like(truth, dtype=np.bool_)
    origins = np.arange(100, dtype=np.int64)
    cal = fit_calibrator(
        truth,
        prediction,
        mask,
        origins,
        policy=CalibrationPolicy(grouping=grouping),
        region_ids=("A", "B"),
        population_id="cal",
        model_id="point",
        observation_delay=24,
    )
    assert cal.thresholds.shape == (24, 2)
    if "region" in grouping:
        np.testing.assert_array_equal(cal.thresholds[0], [1, 4])
    future = np.arange(200, 300, dtype=np.int64)
    lo, hi = cal.intervals(prediction, future, region_ids=("A", "B"), model_id="point")
    assert interval_metrics(truth, prediction, lo, hi, mask, 0.9)["coverage"] == 1
    with pytest.raises(DataError):
        cal.intervals(prediction, origins, region_ids=("A", "B"), model_id="point")
    with pytest.raises(DataError):
        cal.intervals(prediction, future, region_ids=("B", "A"), model_id="point")
    with pytest.raises(DataError):
        cal.intervals(prediction, future, region_ids=("A", "B"), model_id="other")


def test_empty_ties_and_finite_support() -> None:
    assert finite_quantile(np.zeros(100), 0.95) == (0, 96)
    for scores in [np.array([]), np.array([1.0]), np.array([np.inf]), np.array([-1.0])]:
        with pytest.raises(DataError):
            finite_quantile(scores, 0.95)


def test_metric_arithmetic_and_trajectory() -> None:
    y = np.array([[[0.0], [3.0]]])
    p = np.ones_like(y)
    result = interval_metrics(y, p, p - 1, p + 1, np.ones_like(y, dtype=np.bool_), 0.9)
    assert result["coverage"] == 0.5
    assert result["mean_width_MW"] == 2
    assert result["interval_score_MW"] == pytest.approx(12)
    assert result["simultaneous_coverage"] == 0


@pytest.mark.parametrize("normalized", [False, True])
def test_masks_fallback_and_normalization(normalized: bool) -> None:
    shape = (100, 24, 2)
    truth = np.full(shape, 6.0)
    prediction = np.zeros(shape)
    mask = np.ones(shape, dtype=np.bool_)
    mask[:, 0, 0] = False
    scales = np.full((24, 2), 2.0) if normalized else None
    cal = fit_calibrator(
        truth,
        prediction,
        mask,
        np.arange(100, dtype=np.int64),
        policy=CalibrationPolicy(normalized=normalized),
        region_ids=("A", "B"),
        population_id="cal",
        model_id="model",
        observation_delay=24,
        scales=scales,
    )
    assert cal.fallback[0, 0] and cal.counts[0, 0] == 0
    np.testing.assert_array_equal(cal.thresholds * cal.scales, np.full((24, 2), 6))
    # Masked labels cannot alter fitted thresholds.
    truth[:, 0, 0] = np.nan
    other = fit_calibrator(
        truth,
        prediction,
        mask,
        np.arange(100, dtype=np.int64),
        policy=CalibrationPolicy(normalized=normalized),
        region_ids=("A", "B"),
        population_id="cal",
        model_id="model",
        observation_delay=24,
        scales=scales,
    )
    np.testing.assert_array_equal(cal.thresholds, other.thresholds)


def test_future_labels_not_part_of_interval_generation() -> None:
    a = np.zeros((100, 24, 1))
    cal = fit_calibrator(
        a,
        a,
        np.ones_like(a, dtype=np.bool_),
        np.arange(100, dtype=np.int64),
        policy=CalibrationPolicy(),
        region_ids=("A",),
        population_id="cal",
        model_id="p",
        observation_delay=24,
    )
    future = np.arange(200, 300, dtype=np.int64)
    before = cal.intervals(a, future, region_ids=("A",), model_id="p")
    unrelated_future_truth = np.full_like(a, 1e10)
    assert unrelated_future_truth.max() == 1e10
    after = cal.intervals(a, future, region_ids=("A",), model_id="p")
    np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_scales_reject(bad: float) -> None:
    a = np.zeros((100, 24, 1))
    with pytest.raises(DataError):
        fit_calibrator(
            a,
            a,
            np.ones_like(a, dtype=np.bool_),
            np.arange(100, dtype=np.int64),
            policy=CalibrationPolicy(normalized=True),
            region_ids=("A",),
            population_id="p",
            model_id="m",
            observation_delay=24,
            scales=np.full((24, 1), bad),
        )


def test_spacing_and_dependence() -> None:
    from energraph.uncertainty.metrics import dependence

    a = np.ones((100, 24, 1))
    origins = np.arange(100, dtype=np.int64)
    c = fit_calibrator(
        a,
        np.zeros_like(a),
        np.ones_like(a, dtype=np.bool_),
        origins,
        policy=CalibrationPolicy(origin_spacing=24),
        region_ids=("A",),
        population_id="p",
        model_id="m",
        observation_delay=24,
    )
    assert c.counts[0, 0] == 5 and c.fallback.all()
    d = dependence(a, origins)
    assert d["unique_target_rows"] == 123 and d["overlapping_target_occurrences"] == 2277


def test_cli_failure_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    from energraph.uncertainty.cli import main

    assert (
        main(
            [
                "uncertainty-verify",
                "--directory",
                "does-not-exist",
                "--population",
                "also-missing",
                "--population-parents",
                "missing-parent",
            ]
        )
        == 1
    )
    assert "error" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_synthetic_immutable_workflow(
    tmp_path: Path,
    invented_research_rights: Any,
    monkeypatch: pytest.MonkeyPatch,
    broker_context_factory: Any,
) -> None:
    import json

    from energraph.artifacts import (
        PendingPublication,
        _publish_governed,
        finalize_governed_publication,
    )
    from energraph.uncertainty.research import evaluate_population, verify_intervals
    from urbangraph.data import DatasetSnapshot, ingest_dataset
    from urbangraph.datasets.serialization import write_json

    stage = tmp_path / "source"
    stage.mkdir()
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for part, start in [("validation", 0), ("calibration", 200), ("test", 400)]:
        origin = np.arange(start, start + 100, dtype=np.int64)
        truth = np.ones((100, 24, 2))
        arrays[part + "_truth"] = truth
        arrays[part + "_lightgbm"] = np.zeros_like(truth)
        arrays[part + "_persistence"] = np.zeros_like(truth)
        arrays[part + "_origins"] = origin
        arrays[part + "_targets_ns"] = (origin[:, None] + np.arange(1, 25)) * 3600 * 10**9
    np.savez_compressed(stage / "predictions.npz", allow_pickle=False, **arrays)
    write_json(
        stage / "descriptor.json",
        {
            "specification": {"region_ids": ["A", "B"], "latencies": {"A": 24, "B": 24}},
            "partitions": {"calibration": {"population_id": "synthetic-only"}},
        },
    )

    (stage / "runtime.json").write_text("{}", encoding="utf-8")

    def make_source(
        label: str, values: dict[str, np.ndarray[Any, Any]]
    ) -> tuple[Path, DatasetSnapshot]:
        payload = tmp_path / ("stage-" + label)
        payload.mkdir()
        np.savez_compressed(payload / "predictions.npz", allow_pickle=False, **values)
        write_json(
            payload / "descriptor.json",
            {
                "specification": {"region_ids": ["A", "B"], "latencies": {"A": 24, "B": 24}},
                "partitions": {"calibration": {"population_id": "synthetic-only"}},
            },
        )
        write_json(payload / "runtime.json", {})
        incoming = tmp_path / ("invented-" + label)
        incoming.mkdir()
        for name in ("predictions.npz", "descriptor.json"):
            (incoming / name).write_bytes((payload / name).read_bytes())
        dataset_id = "invented-uncertainty-" + label
        invented_research_rights.source(incoming, dataset_id, tmp_path / "raw")
        raw = ingest_dataset(incoming, dataset_id, raw_root=tmp_path / "raw")
        return payload, raw

    original_values = {key: value.copy() for key, value in arrays.items()}
    original_stage, source_snapshot = make_source("original", original_values)
    changed_values = {key: value.copy() for key, value in arrays.items()}
    changed_values["test_truth"] += 10000.0
    changed_stage, changed_source = make_source("changed", changed_values)
    population_root = tmp_path / "populations"
    run_root = tmp_path / "runs"
    operations = (
        ("scientific_evaluation", "energy_population_create"),
        ("scientific_evaluation", "energy_population_read"),
        ("scientific_evaluation", "energy_interval_create"),
        ("scientific_evaluation", "energy_interval_verify"),
        ("local_research", "derived_commit"),
    )
    broker_context, source_reference, policy_approver, additional = broker_context_factory(
        source_snapshot.directory,
        invented_research_rights,
        operations,
        additional_raw_sources=((changed_source.directory, operations),),
    )
    changed_reference = additional[changed_source.directory.parent.name]
    for raw in (source_snapshot, changed_source):
        source_id = raw.directory.parent.name
        for root, grants in (
            (
                population_root,
                (
                    ("scientific_evaluation", "energy_population_create"),
                    ("local_research", "derived_commit"),
                ),
            ),
            (
                run_root,
                (
                    ("scientific_evaluation", "energy_interval_create"),
                    ("scientific_evaluation", "energy_interval_verify"),
                    ("local_research", "derived_commit"),
                ),
            ),
        ):
            for purpose, operation in grants:
                invented_research_rights._grant(
                    source_id,
                    raw.directory,
                    raw.manifest.snapshot_id,
                    operation,
                    root,
                    purpose,
                )

    def publish_population(payload: Path, raw: Any, raw_reference: Any) -> Any:
        pending = _publish_governed(
            payload,
            population_root,
            "uncertainty-population",
            parents=(raw.directory,),
            broker_context=broker_context,
            broker_parents=(raw_reference,),
        )
        assert isinstance(pending, PendingPublication)
        policy_approver(pending)
        committed = finalize_governed_publication(broker_context, pending)
        assert committed.broker_reference is not None
        return committed

    population = publish_population(original_stage, source_snapshot, source_reference)
    policy = CalibrationPolicy()
    pending_run = evaluate_population(
        population,
        run_root,
        "lightgbm",
        policy,
        population_parents=(source_snapshot.directory,),
        broker_context=broker_context,
        population_parent_references=(source_reference,),
    )
    assert isinstance(pending_run, PendingPublication)
    policy_approver(pending_run)
    run = finalize_governed_publication(broker_context, pending_run)
    assert run.broker_reference is not None
    assert (
        verify_intervals(
            run,
            population,
            broker_context=broker_context,
            population_parent_references=(source_reference,),
        )["valid"]
        is True
    )
    assert (
        evaluate_population(
            population,
            run_root,
            "lightgbm",
            policy,
            population_parents=(source_snapshot.directory,),
            broker_context=broker_context,
            population_parent_references=(source_reference,),
        )
        == run
    )
    interval_grants = [
        grant
        for grant in invented_research_rights.grants
        if grant["operation"] == "energy_interval_verify"
    ]
    assert interval_grants
    for grant in interval_grants:
        grant["status"] = "revoked"
    invented_research_rights.path.write_text(
        json.dumps({"version": 1, "grants": invented_research_rights.grants})
    )
    research_module = __import__("energraph.uncertainty.research", fromlist=["read_json"])
    original_read_json = research_module.read_json

    def forbid_interval_descriptor(path: Path) -> object:
        if "energraph-interval" in str(path):
            raise AssertionError("saved interval descriptor read before verify permission")
        return original_read_json(path)

    with monkeypatch.context() as guarded:
        guarded.setattr("energraph.uncertainty.research.read_json", forbid_interval_descriptor)
        with pytest.raises(DataError):
            evaluate_population(
                population,
                run_root,
                "lightgbm",
                policy,
                population_parents=(source_snapshot.directory,),
                broker_context=broker_context,
                population_parent_references=(source_reference,),
            )
    for grant in interval_grants:
        grant["status"] = "approved"
    invented_research_rights.path.write_text(
        json.dumps({"version": 1, "grants": invented_research_rights.grants})
    )
    # Future-label changes must not alter any fitted threshold or scale.
    from urbangraph.datasets.serialization import read_json

    changed_population = publish_population(changed_stage, changed_source, changed_reference)
    changed_pending = evaluate_population(
        changed_population,
        run_root,
        "lightgbm",
        policy,
        population_parents=(changed_source.directory,),
        broker_context=broker_context,
        population_parent_references=(changed_reference,),
    )
    assert isinstance(changed_pending, PendingPublication)
    policy_approver(changed_pending)
    changed_run = finalize_governed_publication(broker_context, changed_pending)
    original_cal = read_json(run.directory / "payload/descriptor.json")["calibrator"]
    changed_cal = read_json(changed_run.directory / "payload/descriptor.json")["calibrator"]
    assert isinstance(original_cal, dict) and isinstance(changed_cal, dict)
    for field in ("thresholds", "scales", "counts", "ranks"):
        assert original_cal[field] == changed_cal[field]
    from energraph.uncertainty.cli import main

    assert (
        main(
            [
                "uncertainty-verify",
                "--artifact-id",
                run.broker_reference.artifact_id,
                "--population-artifact-id",
                population.broker_reference.artifact_id,
                "--population-parent-artifact-ids",
                source_reference.artifact_id,
                "--broker-url",
                broker_context.broker.base_url,
                "--worker-token-file",
                str(broker_context.broker.worker_token_file),
                "--reader-token-file",
                str(broker_context.broker.reader_token_file),
                "--materialization-root",
                str(broker_context.materialization_root),
            ]
        )
        == 0
    )
    with (run.directory / "payload/intervals.npz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(DataError):
        verify_intervals(
            run,
            population,
            broker_context=broker_context,
            population_parent_references=(source_reference,),
        )


@pytest.mark.parametrize(
    "fault", ["overlap", "reordered", "timestamp", "axis", "nonfinite", "missing"]
)
def test_population_adversaries(fault: str) -> None:
    from energraph.uncertainty.validation import validate_partitions

    arrays: dict[
        str,
        np.ndarray[tuple[int, ...], np.dtype[np.float64]]
        | np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    ] = {}
    for part, start in [("validation", 0), ("calibration", 200), ("test", 400)]:
        o = np.arange(start, start + 100, dtype=np.int64)
        arrays[part + "_origins"] = o
        arrays[part + "_truth"] = np.ones((100, 24, 2))
        arrays[part + "_lightgbm"] = np.ones((100, 24, 2))
        arrays[part + "_persistence"] = np.ones((100, 24, 2))
        arrays[part + "_targets_ns"] = (o[:, None] + np.arange(1, 25)) * 3600 * 10**9
    validate_partitions(arrays, ("A", "B"), 24)
    if fault == "overlap":
        arrays["test_origins"] -= 200
    elif fault == "reordered":
        arrays["test_origins"] = arrays["test_origins"][::-1]
    elif fault == "timestamp":
        arrays["test_targets_ns"][0, 0] += 1
    elif fault == "axis":
        arrays["test_truth"] = np.ones((100, 24, 3))
    elif fault == "nonfinite":
        arrays["test_lightgbm"][0, 0, 0] = np.inf
    else:
        del arrays["test_persistence"]
    with pytest.raises(DataError):
        validate_partitions(arrays, ("A", "B"), 24)
