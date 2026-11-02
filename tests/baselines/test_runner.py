import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.baselines.artifacts import verify_run
from urbangraph.baselines.cli import main
from urbangraph.baselines.contracts import MODEL_IDS, ModelConfig, Prediction
from urbangraph.baselines.reporting import compare_runs
from urbangraph.baselines.runner import common_population
from urbangraph.baselines.runner import run_suite as run_suite_public
from urbangraph.baselines.scoring import MetricAccumulator
from urbangraph.data import DataError, ingest_dataset
from urbangraph.datasets.governed_canonical import create_canonical
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation.artifacts import publish_plan
from urbangraph.evaluation.contracts import (
    EvaluationProtocol,
    PartitionBoundary,
    PlanConfig,
    TargetValidityPolicy,
)
from urbangraph.evaluation.metrics import evaluate_predictions


def raw_for(canonical: Path) -> Path:
    return next((canonical.parents[3] / "raw" / "metr-la").iterdir())


def run_suite(
    canonical: Path, plan: Path, configs: tuple[ModelConfig, ...], **kwargs: object
) -> list[Path]:
    return run_suite_public(canonical, plan, configs, raw_snapshot_dir=raw_for(canonical), **kwargs)


@pytest.fixture
def inputs(tmp_path: Path, invented_research_rights: object) -> tuple[Path, Path]:
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


def test_metrics_streaming_matches_existing_library() -> None:
    truth = np.arange(96, dtype=np.float64).reshape(4, 12, 2, 1)
    prediction = truth + 2
    mask = np.ones(truth.shape, dtype=bool)
    mask[0, 0, 0, 0] = False
    score = MetricAccumulator()
    for i in range(4):
        score.add(truth[i : i + 1], prediction[i : i + 1], mask[i : i + 1])
    reference = object_dict(
        evaluate_predictions(
            truth,
            prediction,
            sensor_ids=("Z", "001"),
            protocol="strict-rolling-origin-v1",
            mask=mask,
        )["overall"]
    )
    for name in ("mae", "rmse", "mape"):
        actual = object_dict(score.result()[name])
        expected = object_dict(reference[name])
        assert actual["value"] == pytest.approx(expected["value"])
        assert actual["valid_count"] == expected["valid_count"]


@pytest.mark.parametrize("invalid", ["absent", "revoked", "expired", "wrong-purpose", "forged"])
def test_public_runner_requires_current_exact_source_authority(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    canonical, plan = inputs
    output = tmp_path / "denied-runs"
    policy_path = tmp_path / "invented-research-rights.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    grants = policy["grants"]
    grant = next(item for item in grants if item["operation"] == "canonical_read")
    if invalid == "absent":
        monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY")
    elif invalid == "revoked":
        grant["status"] = "revoked"
    elif invalid == "expired":
        grant["expires_utc"] = "2020-01-01T00:00:00+00:00"
    elif invalid == "wrong-purpose":
        grant["purpose"] = "model_training"
    else:
        grant["source_sha256"] = "0" * 64
    if invalid != "absent":
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(DataError):
        run_suite_public(
            canonical,
            plan,
            (ModelConfig(),),
            raw_snapshot_dir=raw_for(canonical),
            output_root=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("operation", "invalid"),
    [
        ("model_train", "absent"),
        ("model_evaluate", "absent"),
        ("model_train", "revoked"),
        ("model_evaluate", "expired"),
        ("model_train", "wrong-purpose"),
        ("model_evaluate", "wrong-destination"),
        ("model_train", "wrong-digest"),
        ("model_evaluate", "wrong-digest"),
    ],
)
def test_public_runner_requires_distinct_model_purposes(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    operation: str,
    invalid: str,
) -> None:
    canonical, plan = inputs
    output = tmp_path / "purpose-denied-runs"
    policy_path = tmp_path / "invented-research-rights.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    grants = policy["grants"]
    grant = next(item for item in grants if item["operation"] == operation)
    if invalid == "absent":
        grants.remove(grant)
    elif invalid == "revoked":
        grant["status"] = "revoked"
    elif invalid == "expired":
        grant["expires_utc"] = "2020-01-01T00:00:00+00:00"
    elif invalid == "wrong-purpose":
        grant["purpose"] = "local_research"
    elif invalid == "wrong-destination":
        grant["destination_root"] = str(tmp_path / "elsewhere")
    else:
        grant["source_sha256"] = "0" * 64
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(DataError):
        run_suite_public(
            canonical,
            plan,
            (ModelConfig(),),
            raw_snapshot_dir=raw_for(canonical),
            output_root=output,
        )
    assert not output.exists()


def test_missing_predictions_and_nonfinite_truth_common_population() -> None:
    truth = np.array([1, np.nan, 0, 4], dtype=np.float64).reshape(1, 2, 2, 1)
    a = np.array([1, 2, 0, np.nan], dtype=np.float64).reshape(truth.shape)
    b = np.ones(truth.shape, dtype=np.float64)
    predictions = [Prediction(x, np.isfinite(x), np.zeros(x.shape, dtype=bool)) for x in (a, b)]
    common = common_population(np.isfinite(truth), predictions)
    assert common.sum() == 2
    score = MetricAccumulator()
    score.add(truth, a, common)
    assert object_dict(score.result()["mae"])["value"] == 0
    assert object_dict(score.result()["mape"])["valid_count"] == 1


def test_end_to_end_reuse_integrity_cli(
    inputs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    canonical, plan = inputs
    configs = tuple(ModelConfig(model=m) for m in MODEL_IDS)
    paths = run_suite(canonical, plan, configs, output_root=tmp_path / "runs", batch_size=8)
    assert len(paths) == 6
    first = {p: {f.name: f.read_bytes() for f in p.iterdir()} for p in paths}
    again = run_suite(canonical, plan, configs, output_root=tmp_path / "runs", batch_size=8)
    assert paths == again
    assert first == {p: {f.name: f.read_bytes() for f in p.iterdir()} for p in paths}
    for path in paths:
        assert verify_run(path, raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)[
            "valid"
        ]
        groups = object_dict(read_json(path / "metrics.json")["groups"])
        assert len([k for k in groups if k.startswith("common/horizon/")]) == 12
        assert len([k for k in groups if k.startswith("common/sensor/")]) == 2
        assert len([k for k in groups if k.startswith("common/fold/")]) == 2
    ranking = compare_runs(paths, raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)[
        "ranking"
    ]
    assert isinstance(ranking, list) and len(ranking) == 6
    assert main(["list-models", "--json"]) == 0
    assert (
        main(
            [
                "verify",
                "--run-dir",
                str(paths[0]),
                "--raw-snapshot-dir",
                str(raw_for(canonical)),
                "--canonical-dir",
                str(canonical),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "run",
                "--canonical-dir",
                str(canonical),
                "--raw-snapshot-dir",
                str(raw_for(canonical)),
                "--plan-dir",
                str(plan),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    (paths[0] / "metrics.json").write_text("{}")
    with pytest.raises(DataError, match="checksum"):
        verify_run(paths[0], raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)
    assert (
        main(
            [
                "verify",
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
    with pytest.raises(DataError, match="checksum"):
        run_suite(canonical, plan, configs, output_root=tmp_path / "runs", batch_size=8)


def test_different_populations_cannot_rank(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    canonical, plan = inputs
    first = run_suite(canonical, plan, (ModelConfig(),), output_root=tmp_path / "runs")
    second = run_suite(
        canonical, plan, (ModelConfig(model="drift"),), output_root=tmp_path / "runs"
    )
    with pytest.raises(DataError, match="incompatible"):
        compare_runs(first + second, raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)


def test_zero_mask_and_cli_failure(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    canonical, _ = inputs
    plan = publish_plan(
        canonical,
        PlanConfig(validity=TargetValidityPolicy(True)),
        output_root=tmp_path / "evaluation",
        raw_snapshot_dir=raw_for(canonical),
    )
    paths = run_suite(canonical, plan.directory, (ModelConfig(),), output_root=tmp_path / "runs")
    availability = read_json(paths[0] / "availability.json")
    assert int(str(availability["valid_targets"])) < int(str(availability["eligible_targets"]))
    assert (
        main(
            [
                "run",
                "--canonical-dir",
                str(tmp_path / "missing"),
                "--raw-snapshot-dir",
                str(raw_for(canonical)),
                "--plan-dir",
                str(plan.directory),
            ]
        )
        == 1
    )


def test_unknown_files_and_missing_manifest(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    canonical, _ = inputs
    paths = run_suite(*inputs, (ModelConfig(),), output_root=tmp_path / "runs")
    (paths[0] / "unexpected.json").write_text("{}")
    with pytest.raises(DataError, match="inventory"):
        verify_run(paths[0], raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)


def test_failed_publication_cleans_only_owned_staging(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urbangraph.baselines.artifacts as artifacts

    root = tmp_path / "runs"
    existing = run_suite(*inputs, (ModelConfig(),), output_root=root)
    original = {f.name: f.read_bytes() for f in existing[0].iterdir()}

    def fail(stage: Path, destination: Path) -> None:
        raise OSError("simulated finalization failure")

    monkeypatch.setattr(artifacts, "_publish", fail)
    with pytest.raises(OSError, match="simulated"):
        run_suite(*inputs, (ModelConfig(model="drift"),), output_root=root)
    assert {f.name: f.read_bytes() for f in existing[0].iterdir()} == original
    assert list(root.iterdir()) == existing


def test_runner_rejects_output_inside_canonical(inputs: tuple[Path, Path]) -> None:
    with pytest.raises(DataError, match="overlap"):
        run_suite(*inputs, (ModelConfig(),), output_root=inputs[0] / "runs")


def test_chronological(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    canonical, _ = inputs
    config = PlanConfig(
        protocol=EvaluationProtocol("strict-chronological-v1", "isolated"),
        boundaries=(
            PartitionBoundary("train", 0, 150),
            PartitionBoundary("validation", 150, 200),
            PartitionBoundary("calibration", 200, 240),
            PartitionBoundary("test", 240, 300),
        ),
    )
    plan = publish_plan(
        canonical, config, output_root=tmp_path / "chron", raw_snapshot_dir=raw_for(canonical)
    )
    runs = run_suite(canonical, plan.directory, (ModelConfig(),), output_root=tmp_path / "runs")
    descriptor = object_dict(
        verify_run(runs[0], raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)[
            "descriptor"
        ]
    )
    assert descriptor["protocol"] == "strict-chronological-v1"
    assert read_json(runs[0] / "availability.json")["eligible_targets"] == 37 * 12 * 2


def test_saved_run_requires_current_verify_and_comparison_rights(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    canonical, plan = inputs
    run = run_suite(canonical, plan, (ModelConfig(),), output_root=tmp_path / "saved")[0]
    with pytest.raises(DataError, match="Complete current source context"):
        verify_run(run)
    with pytest.raises(DataError, match="Complete current source context"):
        compare_runs([run])
    policy_path = tmp_path / "invented-research-rights.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    verification = next(grant for grant in policy["grants"] if grant["operation"] == "model_verify")
    verification["status"] = "revoked"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        verify_run(run, raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)
    assert (
        main(
            [
                "verify",
                "--run-dir",
                str(run),
                "--raw-snapshot-dir",
                str(raw_for(canonical)),
                "--canonical-dir",
                str(canonical),
            ]
        )
        == 1
    )
    verification["status"] = "approved"
    comparison = next(grant for grant in policy["grants"] if grant["operation"] == "model_compare")
    comparison["source_sha256"] = "0" * 64
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(DataError, match="differs from current"):
        compare_runs([run], raw_snapshot_dir=raw_for(canonical), canonical_dir=canonical)
