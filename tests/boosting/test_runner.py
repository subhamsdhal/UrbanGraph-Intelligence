import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from urbangraph.boosting.artifacts import verify_run
from urbangraph.boosting.cli import main
from urbangraph.boosting.contracts import BoostConfig, FeatureSpec
from urbangraph.boosting.runner import run_experiment as run_experiment_public
from urbangraph.boosting.sampling import sampled_matrix, select_rows
from urbangraph.data import DataError, ingest_dataset
from urbangraph.datasets.governed_canonical import create_canonical
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation.artifacts import load_plan, publish_plan
from urbangraph.evaluation.contracts import PlanConfig
from urbangraph.evaluation.windows import numerical_values


def raw_for(canonical: Path) -> Path:
    return next((canonical.parents[3] / "raw" / "metr-la").iterdir())


def run_experiment(
    canonical: Path, plan: Path, config: BoostConfig, **kwargs: object
) -> list[Path]:
    return run_experiment_public(
        canonical, plan, config, raw_snapshot_dir=raw_for(canonical), **kwargs
    )


@pytest.fixture
def inputs(tmp_path: Path, invented_research_rights: Any) -> tuple[Path, Path]:
    source = tmp_path / "incoming"
    source.mkdir()
    rows = np.arange(300, dtype=np.float64)
    frame = pd.DataFrame(
        {"Z": rows % 50, "001": 10 + rows / 10},
        index=pd.date_range("2024-01-01", periods=300, freq="5min"),
    )
    frame.iloc[100, 0] = np.nan
    frame.to_hdf(source / "metr-la.h5", key="traffic")
    invented_research_rights.source(source, "metr-la", tmp_path / "raw")
    raw = ingest_dataset(source, "metr-la", raw_root=tmp_path / "raw")
    invented_research_rights.snapshot(raw, tmp_path / "interim")
    invented_research_rights.model_use(raw, tmp_path)
    canonical = create_canonical(raw.directory, output_root=tmp_path / "interim")
    plan = publish_plan(
        canonical.directory,
        PlanConfig(),
        output_root=tmp_path / "evaluation",
        raw_snapshot_dir=raw.directory,
    )
    return canonical.directory, plan.directory


def test_sampling_deterministic_scoped(inputs: tuple[Path, Path]) -> None:
    canonical, plan = inputs
    dataset, split, _ = load_plan(
        plan, canonical_dir=canonical, raw_snapshot_dir=raw_for(canonical)
    )
    values = numerical_values(dataset.frame)
    a = select_rows(dataset, split, 0, "train", 240, 42, values)
    b = select_rows(dataset, split, 0, "train", 240, 42, values)
    assert a.report == b.report
    assert len(a.origins) <= 240
    assert len(set(zip(a.origins, a.sensors, a.horizons, strict=True))) == len(a.origins)
    assert np.all(a.origins + a.horizons < split.folds[0].partition("train").stop)
    first, _, schema = sampled_matrix(dataset, split, 0, "train", values, a, FeatureSpec(), 13)
    second, _, _ = sampled_matrix(dataset, split, 0, "train", values, a, FeatureSpec(), 101)
    np.testing.assert_array_equal(first, second)
    assert schema.names[-1] == "sensor_id"
    with pytest.raises(DataError):
        select_rows(dataset, split, 0, "test", 240, 42, values)
    with pytest.raises(DataError):
        select_rows(dataset, split, 0, "train", 1, 42, values)


@pytest.mark.parametrize("operation", ["model_train", "model_evaluate"])
def test_public_boosting_denies_missing_model_purpose(
    inputs: tuple[Path, Path], tmp_path: Path, operation: str
) -> None:
    canonical, plan = inputs
    policy_path = tmp_path / "invented-research-rights.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["grants"] = [grant for grant in policy["grants"] if grant["operation"] != operation]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    output = tmp_path / "denied-boosting"
    with pytest.raises(DataError, match="No matching rights grant"):
        run_experiment_public(
            canonical,
            plan,
            BoostConfig(family="lightgbm", trees=3, threads=1),
            raw_snapshot_dir=raw_for(canonical),
            output_root=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("family", ["lightgbm", "xgboost"])
def test_end_to_end(inputs: tuple[Path, Path], tmp_path: Path, family: str) -> None:
    canonical, plan = inputs
    config = BoostConfig(
        family=family,
        trees=3,
        threads=1,
        min_child_samples=2,
        min_child_weight=1,
        max_train_rows=240,
        max_validation_rows=120,
        variants=("F0", "F4"),
        alternative_depth=5,
        evaluation_batch=16,
    )
    paths = run_experiment(canonical, plan, config, output_root=tmp_path / "runs")
    assert len(paths) == 2
    for path in paths:
        assert verify_run(path, raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)[
            "valid"
        ]
        with pytest.raises(DataError, match="Complete current source context"):
            verify_run(path)
        groups = object_dict(read_json(path / "metrics.json")["groups"])
        assert len([key for key in groups if key.startswith("common/horizon/")]) == 12
        assert (
            main(
                [
                    "verify-run",
                    "--run-dir",
                    str(path),
                    "--raw-snapshot-dir",
                    str(raw_for(canonical)),
                    "--canonical-dir",
                    str(canonical),
                ]
            )
            == 0
        )
    policy_path = tmp_path / "invented-research-rights.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    grant = next(item for item in policy["grants"] if item["operation"] == "model_verify")
    grant["status"] = "revoked"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    assert (
        main(
            [
                "verify-run",
                "--run-dir",
                str(paths[0]),
                "--raw-snapshot-dir",
                str(raw_for(canonical)),
                "--canonical-dir",
                str(canonical),
            ]
        )
        == 1
    )
    grant["status"] = "approved"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    before = {p: {f.name: f.read_bytes() for f in p.iterdir()} for p in paths}
    again = run_experiment(canonical, plan, config, output_root=tmp_path / "runs")
    assert again == paths
    assert before == {p: {f.name: f.read_bytes() for f in p.iterdir()} for p in paths}
    (paths[0] / "metrics.json").write_text("{}")
    with pytest.raises(DataError, match="checksum"):
        verify_run(paths[0], raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)
    assert (
        main(
            [
                "verify-run",
                "--run-dir",
                str(paths[0]),
                "--raw-snapshot-dir",
                str(raw_for(canonical)),
                "--canonical-dir",
                str(canonical),
            ]
        )
        == 1
    )


def test_publication_failure_cleans_stage(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from urbangraph.boosting import artifacts

    def fail(stage: Path, destination: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(artifacts, "_publish", fail)
    config = BoostConfig(
        trees=1,
        threads=1,
        max_train_rows=240,
        max_validation_rows=120,
        variants=("F0",),
        evaluation_batch=16,
    )
    output = tmp_path / "failed"
    with pytest.raises(OSError, match="simulated"):
        run_experiment(*inputs, config, output_root=output)
    assert not list(output.glob(".staging-*"))
    assert not list(output.glob("*/manifest.json"))


def test_selection_ignores_calibration_and_test(inputs: tuple[Path, Path]) -> None:
    import time

    from urbangraph.boosting.tuning import select_models

    canonical, plan = inputs
    dataset, split, _ = load_plan(
        plan, canonical_dir=canonical, raw_snapshot_dir=raw_for(canonical)
    )
    config = BoostConfig(
        trees=2,
        threads=1,
        max_train_rows=240,
        max_validation_rows=120,
        variants=("F0", "F1", "F2", "F3", "F4"),
        alternative_depth=5,
    )
    first = select_models(dataset, split, 0, config, time.perf_counter() + 60)
    dataset.frame.iloc[split.folds[0].partition("calibration").start :] = -99999
    second = select_models(dataset, split, 0, config, time.perf_counter() + 60)
    first.report.pop("training_seconds")
    second.report.pop("training_seconds")
    assert first.report == second.report
    assert first.models["selected"].importance() == second.models["selected"].importance()


def test_baseline_comparison_population(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    from urbangraph.baselines.contracts import ModelConfig
    from urbangraph.baselines.runner import run_suite
    from urbangraph.boosting.reporting import compare_runs

    config = BoostConfig(
        trees=1,
        threads=1,
        max_train_rows=240,
        max_validation_rows=120,
        variants=("F0",),
        evaluation_batch=16,
    )
    boosted = run_experiment(*inputs, config, output_root=tmp_path / "boosted")
    baseline = run_suite(
        *inputs,
        (ModelConfig(),),
        raw_snapshot_dir=raw_for(inputs[0]),
        output_root=tmp_path / "baseline",
        batch_size=16,
    )
    comparison = compare_runs(
        boosted + baseline, raw_snapshot_dir=raw_for(inputs[0]), canonical_dir=inputs[0]
    )
    runs = comparison["runs"]
    assert isinstance(runs, list) and len(runs) == 3
    different = run_suite(
        *inputs,
        (ModelConfig(),),
        raw_snapshot_dir=raw_for(inputs[0]),
        output_root=tmp_path / "other",
        batch_size=17,
    )
    with pytest.raises(DataError, match="Incompatible"):
        compare_runs(
            boosted + different, raw_snapshot_dir=raw_for(inputs[0]), canonical_dir=inputs[0]
        )


def test_registry_extension_and_cli_failure(tmp_path: Path) -> None:
    from urbangraph.baselines.registry import list_models

    assert len(list_models()) == 6
    assert len(list_models(include_boosting=True)) == 8
    assert main(["list-models"]) == 0
    assert main(["inspect-config", "--config", str(tmp_path / "missing.json")]) == 1


@pytest.mark.parametrize("family", ["lightgbm", "xgboost"])
def test_nonfinite_labels_rejected(family: str) -> None:
    import time

    from urbangraph.boosting.models import train_model

    x = np.ones((3, 1), dtype=np.float32)
    with pytest.raises(DataError, match="Finite"):
        train_model(
            BoostConfig(family=family),
            FeatureSpec(),
            ("lag_1",),
            ("lags",),
            ("S",),
            12,
            x,
            np.array([1, np.nan, 3]),
            x,
            np.ones(3),
            time.perf_counter() + 30,
        )
