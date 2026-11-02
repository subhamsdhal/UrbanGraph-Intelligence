"""Direct saved-run publishers must not inherit a runner's earlier grants."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from urbangraph.baselines.artifacts import publish_run as publish_baseline
from urbangraph.baselines.artifacts import verify_run as verify_baseline
from urbangraph.baselines.contracts import ModelConfig
from urbangraph.baselines.runner import run_suite
from urbangraph.boosting.artifacts import publish_run as publish_boosting
from urbangraph.boosting.artifacts import restore as restore_boosting
from urbangraph.boosting.artifacts import verify_run as verify_boosting
from urbangraph.boosting.contracts import BoostConfig
from urbangraph.boosting.runner import run_experiment as run_boosting
from urbangraph.data import DataError, ingest_dataset
from urbangraph.datasets.governed_canonical import create_canonical
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.artifacts import publish_plan
from urbangraph.evaluation.contracts import PlanConfig
from urbangraph.neural.artifacts import publish_run as publish_neural
from urbangraph.neural.artifacts import restore as restore_neural
from urbangraph.neural.artifacts import verify_run as verify_neural
from urbangraph.neural.contracts import NeuralConfig
from urbangraph.neural.runner import run_experiment as run_neural


@pytest.fixture
def invented_inputs(tmp_path: Path, invented_research_rights: Any) -> tuple[Path, Path, Path]:
    source = tmp_path / "invented-source"
    source.mkdir()
    values = np.arange(300, dtype=np.float64)
    frame = pd.DataFrame(
        {"Z": values % 50, "001": 10 + values / 10},
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
    return raw.directory, canonical.directory, plan.directory


@pytest.mark.parametrize("kind", ["baseline", "boosting", "neural"])
def test_direct_publish_requires_current_authority(
    kind: str,
    invented_inputs: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, canonical, plan = invented_inputs
    if kind == "baseline":
        run = run_suite(
            canonical,
            plan,
            (ModelConfig(),),
            raw_snapshot_dir=raw,
            output_root=tmp_path / "baseline-runs",
        )[0]
        payloads = {
            name: read_json(run / f"{name}.json")
            for name in (
                "model-config",
                "fitted-state",
                "metrics",
                "availability",
                "lineage",
                "selection",
                "runtime",
            )
        }

        def publish(destination: Path, **context: object) -> Path:
            return publish_baseline(payloads, descriptor, destination, **context)

        verify = verify_baseline
    elif kind == "boosting":
        runs = run_boosting(
            canonical,
            plan,
            BoostConfig(
                family="lightgbm",
                trees=3,
                threads=1,
                min_child_samples=2,
                min_child_weight=1,
                max_train_rows=240,
                max_validation_rows=120,
                variants=("F0", "F4"),
                alternative_depth=5,
                evaluation_batch=16,
            ),
            raw_snapshot_dir=raw,
            output_root=tmp_path / "boost-runs",
        )
        run = runs[0]
        models = {
            fold: restore_boosting(run, metadata)
            for fold, metadata in read_json(run / "models.json").items()
        }
        payloads = {
            name: read_json(run / f"{name}.json")
            for name in ("metrics", "selection", "runtime", "importance")
        }

        def publish(destination: Path, **context: object) -> Path:
            return publish_boosting(models, payloads, descriptor, destination, **context)

        verify = verify_boosting
    else:
        run = run_neural(
            canonical,
            plan,
            NeuralConfig(
                epochs=1,
                hidden=8,
                threads=1,
                max_train_windows=16,
                max_validation_windows=8,
                trial_seconds=120,
            ),
            raw_snapshot_dir=raw,
            output_root=tmp_path / "neural-runs",
        )
        models = {
            fold: restore_neural(run, metadata)
            for fold, metadata in read_json(run / "models.json").items()
        }
        scalers = read_json(run / "scalers.json")
        payloads = {
            name: read_json(run / f"{name}.json") for name in ("metrics", "selection", "runtime")
        }

        def publish(destination: Path, **context: object) -> Path:
            return publish_neural(models, scalers, payloads, descriptor, destination, **context)

        verify = verify_neural
    descriptor = dict(read_json(run / "run.json"))
    descriptor.pop("content_sha256")
    context = {"raw_snapshot_dir": raw, "canonical_dir": canonical, "plan_dir": plan}
    destination = tmp_path / (kind + "-direct")
    policy_path = tmp_path / "invented-research-rights.json"
    original = policy_path.read_text(encoding="utf-8")
    with pytest.raises(DataError, match="Complete current source and plan context"):
        publish(destination)
    assert not destination.exists()
    for defect in (
        "missing",
        "revoked",
        "expired",
        "wrong-purpose",
        "wrong-destination",
        "wrong-digest",
    ):
        policy = json.loads(original)
        grants = [g for g in policy["grants"] if g["operation"] == "model_publish"]
        assert len(grants) == 1
        grant = grants[0]
        if defect == "missing":
            policy["grants"].remove(grant)
        elif defect == "revoked":
            grant["status"] = "revoked"
        elif defect == "expired":
            grant["expires_utc"] = "2020-01-01T00:00:00+00:00"
        elif defect == "wrong-purpose":
            grant["purpose"] = "benchmark_comparison"
        elif defect == "wrong-destination":
            grant["destination_root"] = str(tmp_path / "other")
        else:
            grant["source_sha256"] = "0" * 64
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        with pytest.raises(DataError):
            publish(destination, **context)
        assert not destination.exists(), defect
    policy_path.write_text(original, encoding="utf-8")
    forged = dict(descriptor)
    forged["canonical"] = {"forged": True}
    with pytest.raises(DataError, match="lineage or plan"):
        if kind == "baseline":
            publish_baseline(payloads, forged, destination, **context)
        elif kind == "boosting":
            publish_boosting(models, payloads, forged, destination, **context)
        else:
            publish_neural(models, scalers, payloads, forged, destination, **context)
    assert not destination.exists()
    forged_plan = dict(descriptor)
    forged_plan.pop("plan_id")
    with pytest.raises(DataError, match="lineage or plan"):
        if kind == "baseline":
            publish_baseline(payloads, forged_plan, destination, **context)
        elif kind == "boosting":
            publish_boosting(models, payloads, forged_plan, destination, **context)
        else:
            publish_neural(models, scalers, payloads, forged_plan, destination, **context)
    with pytest.raises(DataError):
        publish(destination, **(context | {"raw_snapshot_dir": raw.parent / ("0" * 64)}))
    assert not destination.exists()
    for claim in (
        {"trusted": True},
        {"validated": True},
        {"production_ready": True},
        {"saved_run_trust": {"authorship_assurance": "verified"}},
    ):
        forged = descriptor | claim
        with pytest.raises(DataError, match="trust or endorsement"):
            if kind == "baseline":
                publish_baseline(payloads, forged, destination, **context)
            elif kind == "boosting":
                publish_boosting(models, payloads, forged, destination, **context)
            else:
                publish_neural(models, scalers, payloads, forged, destination, **context)
        assert not destination.exists()
    result = publish(destination, **context)
    checked = verify(result, raw_snapshot_dir=raw, canonical_dir=canonical)
    assert checked["valid"] and checked["integrity"] == "pass"
    assert checked["classification"] == "caller_supplied_unverified"
    assert checked["authorship_assurance"] == "unverified"
    assert checked["scientific_validation"] == "not_established"
    assert checked["production_approval"] == "not_established"
    assert read_json(result / "run.json")["saved_run_trust"]["submission"] == "caller_supplied"
    caller_payloads = payloads | {"metrics": {"caller_authored": "not a runner score"}}
    caller_descriptor = descriptor | {"runner_name": "approved-runner"}
    caller_root = tmp_path / (kind + "-caller-authored")
    if kind == "baseline":
        caller_result = publish_baseline(caller_payloads, caller_descriptor, caller_root, **context)
    elif kind == "boosting":
        caller_result = publish_boosting(
            models, caller_payloads, caller_descriptor, caller_root, **context
        )
    else:
        caller_result = publish_neural(
            models, scalers, caller_payloads, caller_descriptor, caller_root, **context
        )
    caller_checked = verify(caller_result, raw_snapshot_dir=raw, canonical_dir=canonical)
    assert caller_checked["integrity"] == "pass"
    assert caller_checked["classification"] == "caller_supplied_unverified"
    assert caller_checked["authorship_assurance"] == "unverified"
    assert caller_checked["scientific_validation"] == "not_established"
    assert caller_checked["production_approval"] == "not_established"
    assert read_json(caller_result / "metrics.json") == caller_payloads["metrics"]
    if kind == "neural":
        matching_root = tmp_path / "matching-neural-experiment"
        matching = publish_neural(
            models,
            scalers,
            caller_payloads,
            caller_descriptor,
            matching_root / "runs",
            **context,
        )
        assert (
            verify_neural(matching, raw_snapshot_dir=raw, canonical_dir=canonical)["classification"]
            == "caller_supplied_unverified"
        )
        with pytest.raises(DataError, match="completed-run reuse denied"):
            run_neural(
                canonical,
                plan,
                NeuralConfig(
                    epochs=1,
                    hidden=8,
                    threads=1,
                    max_train_windows=16,
                    max_validation_windows=8,
                    trial_seconds=120,
                ),
                raw_snapshot_dir=raw,
                output_root=matching_root,
            )
        assert read_json(matching / "metrics.json") == caller_payloads["metrics"]
    assert read_json(result / "run.json")["canonical"] == descriptor["canonical"]
    (result / "metrics.json").write_bytes(b"tampered")
    with pytest.raises(DataError):
        verify(result, raw_snapshot_dir=raw, canonical_dir=canonical)
