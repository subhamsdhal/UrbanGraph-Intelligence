import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from urbangraph.data import RawFileRecord, content_identity, ingestion
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.reproducibility import EnvironmentInfo

_INVENTED_DATASET_IDS = frozenset({"metr-la", "pems-bay", "future-city"})


@pytest.fixture(autouse=True)
def fixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit tests do not depend on the host Git checkout or run Git subprocesses.
    monkeypatch.setattr(
        ingestion,
        "collect_environment_info",
        lambda: EnvironmentInfo("3.11.0", "test-platform", "0.1.0", False, "unavailable", None),
    )


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "metr-la.h5").write_bytes(b"synthetic\x00\xfftraffic\n")
    (source / "nested" / "notes.txt").write_bytes(b"fixture only\n")
    return source


@pytest.fixture
def invented_source_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Path, str, Path], None]:
    """Grant one explicit invented fixture in this test only; no wildcard policy."""

    def grant(source: Path, dataset_id: str, destination: Path) -> None:
        root = safe_absolute(tmp_path)
        source = safe_absolute(source)
        destination = safe_absolute(destination)
        if source == root or not source.is_relative_to(root):
            raise ValueError("Invented source grant must remain under this test's tmp_path")
        if dataset_id not in _INVENTED_DATASET_IDS:
            raise ValueError("Invented source grant does not support this dataset identity")
        if not destination.is_relative_to(root):
            raise ValueError("Invented source grant destination must remain under tmp_path")
        if not source.is_dir():
            raise ValueError("Invented source grant requires a fixture directory")
        records = tuple(
            RawFileRecord(relative, *stream_file(source / relative))
            for relative in discover(source).files
        )
        now = datetime.now(UTC)
        policy = {
            "version": 1,
            "grants": [
                {
                    "source_id": dataset_id,
                    "source_sha256": content_identity(records),
                    "source_path": str(source),
                    "purpose": "local_research",
                    "operation": "raw_ingest",
                    "destination_root": str(destination),
                    "evidence_reference": "invented-test-fixture-only",
                    "status": "approved",
                    "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                    "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                    "retention_days": 1,
                }
            ],
        }
        path = tmp_path / "invented-rights-policy.json"
        path.write_text(json.dumps(policy), encoding="utf-8")
        monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(path))

    return grant
