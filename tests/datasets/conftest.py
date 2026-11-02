import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import (
    DatasetSnapshot,
    DatasetSource,
    RawFileRecord,
    content_identity,
    ingest_dataset,
)
from urbangraph.data.filesystem import discover, stream_file

from . import SnapshotFactory


@pytest.fixture
def traffic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [[10.0, 0.0, 1.0], [np.nan, 20.0, 1.0], [30.0, -2.0, 1.0], [np.inf, 40.0, 1.0]],
        index=pd.DatetimeIndex(
            ["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:15", "2024-01-01 00:20"]
        ),
        columns=["sensor-Z", "001", "A"],
    )


@pytest.fixture
def snapshot_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SnapshotFactory:
    count = 0
    grants: list[dict[str, object]] = []
    policy_path = tmp_path / "invented-dataset-rights.json"

    def grant(source_id: str, source: Path, digest: str, operation: str, destination: Path) -> None:
        if any(
            item["source_id"] == source_id
            and item["source_path"] == str(source)
            and item["operation"] == operation
            for item in grants
        ):
            return
        now = datetime.now(UTC)
        grants.append(
            {
                "source_id": source_id,
                "source_sha256": digest,
                "source_path": str(source),
                "purpose": "local_research",
                "operation": operation,
                "destination_root": str(destination),
                "evidence_reference": "invented-dataset-test-fixture-only",
                "status": "approved",
                "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                "retention_days": 1,
            }
        )
        policy_path.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
        monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy_path))

    def create(
        frame: pd.DataFrame,
        dataset_id: str = "metr-la",
        *,
        key: str = "traffic",
        storage: Literal["fixed", "table"] = "fixed",
        other: pd.DataFrame | None = None,
    ) -> DatasetSnapshot:
        nonlocal count
        count += 1
        source = tmp_path / f"source-{count}"
        source.mkdir()
        path = source / f"{dataset_id}.h5"
        frame.to_hdf(path, key=key, mode="w", format=storage)
        if other is not None:
            other.to_hdf(path, key="other", mode="a", format=storage)
        records = tuple(
            RawFileRecord(relative, *stream_file(source / relative))
            for relative in discover(source).files
        )
        grant(dataset_id, source, content_identity(records), "raw_ingest", tmp_path / "raw")
        snapshot = ingest_dataset(
            source,
            dataset_id,
            raw_root=tmp_path / "raw",
            source=DatasetSource(reference="synthetic test fixture"),
        )
        grant(
            snapshot.directory.parent.name,
            snapshot.directory,
            snapshot.manifest.snapshot_id,
            "snapshot_parse",
            snapshot.directory.parent,
        )
        grant(
            snapshot.directory.parent.name,
            snapshot.directory,
            snapshot.manifest.snapshot_id,
            "canonical_create",
            tmp_path,
        )
        grant(
            snapshot.directory.parent.name,
            snapshot.directory,
            snapshot.manifest.snapshot_id,
            "canonical_read",
            tmp_path,
        )
        return snapshot

    return create
