import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DatasetSource, RawFileRecord, content_identity, ingest_dataset
from urbangraph.data.filesystem import discover, stream_file
from urbangraph.datasets.materialization import materialize_snapshot
from urbangraph.evaluation.contracts import ForecastWindowSpecification, PlanConfig
from urbangraph.evaluation.dataset import EvaluationDataset, open_dataset


@pytest.fixture
def config() -> PlanConfig:
    return PlanConfig(
        window=ForecastWindowSpecification(3, 2),
        minimum_train=20,
        validation_length=10,
        calibration_length=10,
        test_length=10,
        stride=10,
        folds=2,
        scaling="per_sensor",
    )


@pytest.fixture
def dataset(tmp_path: Path) -> EvaluationDataset:
    # Pure algorithm fixture, not a claim of disk verification. Integration uses canonical_dir.
    values = np.arange(100, dtype=np.float64)
    frame = pd.DataFrame(
        {"sensor-Z": values, "001": 2 * values + 1},
        index=pd.date_range("2024-01-01", periods=100, freq="5min"),
    )
    return EvaluationDataset(tmp_path, frame, {"fixture": "pure-unit-test"})


@pytest.fixture
def canonical_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "evaluation-input"
    root.mkdir()
    source = root / "incoming"
    source.mkdir()
    values = np.arange(300, dtype=np.float64)
    frame = pd.DataFrame(
        {"sensor-Z": values, "001": values * 2 + 1},
        index=pd.date_range("2024-01-01", periods=300, freq="5min"),
    )
    frame.iloc[20, 0] = np.nan
    frame.to_hdf(source / "metr-la.h5", key="traffic")
    now = datetime.now(UTC)
    grants: list[dict[str, object]] = []
    policy = tmp_path / "invented-evaluation-rights.json"

    def grant(source_path: Path, digest: str, operation: str, destination: Path) -> None:
        grants.append(
            {
                "source_id": "metr-la",
                "source_sha256": digest,
                "source_path": str(source_path),
                "purpose": "local_research",
                "operation": operation,
                "destination_root": str(destination),
                "evidence_reference": "invented-evaluation-fixture-only",
                "status": "approved",
                "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                "retention_days": 1,
            }
        )
        policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")

    records = tuple(
        RawFileRecord(name, *stream_file(source / name)) for name in discover(source).files
    )
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    grant(source, content_identity(records), "raw_ingest", root / "raw")
    raw = ingest_dataset(
        source, "metr-la", raw_root=root / "raw", source=DatasetSource(reference="invented")
    )
    grant(raw.directory, raw.manifest.snapshot_id, "snapshot_parse", raw.directory.parent)
    grant(raw.directory, raw.manifest.snapshot_id, "canonical_create", root / "interim")
    grant(raw.directory, raw.manifest.snapshot_id, "canonical_read", root)
    return materialize_snapshot(raw.directory, output_root=root / "interim").directory


@pytest.fixture
def raw_snapshot_dir(canonical_dir: Path) -> Path:
    return next((canonical_dir.parents[3] / "raw" / "metr-la").iterdir())


@pytest.fixture
def verified_dataset(canonical_dir: Path, raw_snapshot_dir: Path) -> EvaluationDataset:
    return open_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
