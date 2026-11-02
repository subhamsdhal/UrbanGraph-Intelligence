import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError, ingest_dataset
from urbangraph.data.filesystem import stream_file
from urbangraph.datasets.governed_canonical import create_canonical
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.artifacts import publish_plan
from urbangraph.evaluation.contracts import PlanConfig
from urbangraph.evaluation.dataset import identity
from urbangraph.neural.artifacts import verify_run
from urbangraph.neural.cli import main
from urbangraph.neural.contracts import NeuralConfig
from urbangraph.neural.runner import run_experiment


def test_neural_end_to_end_and_idempotency(tmp_path: Path, invented_research_rights: Any) -> None:
    source = tmp_path / "incoming"
    source.mkdir()
    rows = np.arange(300, dtype=np.float64)
    frame = pd.DataFrame(
        {"z": rows % 30, "a": 10 + rows / 10},
        index=pd.date_range("2024-01-01", periods=300, freq="5min"),
    )
    frame.iloc[20, 0] = np.nan
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
    config = NeuralConfig(
        epochs=1,
        hidden=8,
        threads=1,
        max_train_windows=16,
        max_validation_windows=8,
        trial_seconds=120,
    )
    run = run_experiment(
        canonical.directory,
        plan.directory,
        config,
        raw_snapshot_dir=raw.directory,
        output_root=tmp_path / "neural",
    )
    assert verify_run(run, raw_snapshot_dir=raw.directory, canonical_dir=canonical.directory)[
        "valid"
    ]
    with pytest.raises(DataError, match="Complete current source context"):
        verify_run(run)
    groups = read_json(run / "metrics.json")["groups"]
    assert isinstance(groups, dict)
    assert len([key for key in groups if key.startswith("common/horizon/")]) == 12
    assert (
        main(
            [
                "verify-run",
                "--run-dir",
                str(run),
                "--raw-snapshot-dir",
                str(raw.directory),
                "--canonical-dir",
                str(canonical.directory),
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
                str(run),
                "--raw-snapshot-dir",
                str(raw.directory),
                "--canonical-dir",
                str(canonical.directory),
            ]
        )
        == 1
    )
    grant["status"] = "approved"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    with pytest.raises(DataError, match="completed-run reuse denied"):
        run_experiment(
            canonical.directory,
            plan.directory,
            config,
            raw_snapshot_dir=raw.directory,
            output_root=tmp_path / "neural",
        )
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    # A disposable legacy bundle can pass byte verification but cannot attest
    # the completed execution asserted by this runner API.
    legacy_descriptor = dict(read_json(run / "run.json"))
    legacy_descriptor.pop("saved_run_trust")
    (run / "run.json").write_text(
        json.dumps(legacy_descriptor, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    legacy_id = identity(legacy_descriptor)
    legacy_manifest = dict(read_json(run / "manifest.json"))
    legacy_manifest["run_id"] = legacy_id
    legacy_manifest["files"]["run.json"] = stream_file(run / "run.json")[1]
    (run / "manifest.json").write_text(
        json.dumps(legacy_manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    legacy_run = run.with_name(legacy_id)
    run.rename(legacy_run)
    run = legacy_run
    assert (
        verify_run(run, raw_snapshot_dir=raw.directory, canonical_dir=canonical.directory)[
            "classification"
        ]
        == "legacy_unknown"
    )
    with pytest.raises(DataError, match="completed-run reuse denied"):
        run_experiment(
            canonical.directory,
            plan.directory,
            config,
            raw_snapshot_dir=raw.directory,
            output_root=tmp_path / "neural",
        )
    (run / "model-0.pt").write_bytes(b"corrupted")
    with pytest.raises(DataError, match="checksum"):
        verify_run(run, raw_snapshot_dir=raw.directory, canonical_dir=canonical.directory)
    assert (
        main(
            [
                "verify-run",
                "--run-dir",
                str(run),
                "--raw-snapshot-dir",
                str(raw.directory),
                "--canonical-dir",
                str(canonical.directory),
            ]
        )
        == 1
    )


@pytest.mark.parametrize("operation", ["model_train", "model_evaluate"])
def test_public_neural_denies_missing_model_purpose(
    tmp_path: Path, invented_research_rights: Any, operation: str
) -> None:
    source = tmp_path / "incoming"
    source.mkdir()
    frame = pd.DataFrame(
        {"z": np.arange(300, dtype=np.float64), "a": np.arange(300, dtype=np.float64)},
        index=pd.date_range("2024-01-01", periods=300, freq="5min"),
    )
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
    policy_path = invented_research_rights.path
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["grants"] = [grant for grant in policy["grants"] if grant["operation"] != operation]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    output = tmp_path / "denied-neural"
    with pytest.raises(DataError, match="No matching rights grant"):
        run_experiment(
            canonical.directory,
            plan.directory,
            NeuralConfig(epochs=1, hidden=8, threads=1),
            raw_snapshot_dir=raw.directory,
            output_root=output,
        )
    assert not output.exists()
