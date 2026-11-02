"""Narrow invented-source grants for offline research regression fixtures."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from urbangraph.data import DatasetSnapshot, RawFileRecord, content_identity, verify_snapshot
from urbangraph.data.filesystem import discover, stream_file


class InventedResearchRights:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from urbangraph.data.filesystem import safe_absolute

        self.root = safe_absolute(root)
        self.path = self.root / "invented-research-rights.json"
        self.grants: list[dict[str, object]] = []
        monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(self.path))

    def _confined(self, path: Path, label: str, *, allow_root: bool = False) -> Path:
        from urbangraph.data.filesystem import safe_absolute

        resolved = safe_absolute(path)
        if (resolved == self.root and not allow_root) or not resolved.is_relative_to(self.root):
            raise ValueError(f"Invented fixture {label} must remain under this test's tmp_path")
        return resolved

    def _grant(
        self,
        source_id: str,
        source: Path,
        digest: str,
        operation: str,
        destination: Path,
        purpose: str = "local_research",
    ) -> None:
        source = self._confined(source, "source")
        destination = self._confined(destination, "destination", allow_root=True)
        if source.is_dir():
            tree = discover(source)
            records = tuple(RawFileRecord(name, *stream_file(source / name)) for name in tree.files)
            if (source / "manifest.json").is_file():
                checked = verify_snapshot(source)
                if not checked.valid or checked.manifest is None:
                    raise ValueError("Invented fixture snapshot does not verify")
                actual_digest = checked.manifest.snapshot_id
            else:
                actual_digest = content_identity(records)
        elif source.is_file():
            actual_digest = stream_file(source)[1]
        else:
            raise ValueError("Invented fixture source must exist")
        if actual_digest != digest:
            raise ValueError("Invented fixture grant digest does not match source content")
        now = datetime.now(UTC)
        self.grants.append(
            {
                "source_id": source_id,
                "source_sha256": digest,
                "source_path": str(source),
                "purpose": purpose,
                "operation": operation,
                "destination_root": str(destination),
                "evidence_reference": "invented-offline-research-fixture-only",
                "status": "approved",
                "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                "retention_days": 1,
            }
        )
        self.path.write_text(json.dumps({"version": 1, "grants": self.grants}), encoding="utf-8")

    def broker_tree(
        self,
        source_id: str,
        source: Path,
        digest: str,
        operations: tuple[tuple[str, str], ...],
        destination: Path,
    ) -> None:
        """Grant exact operations for a byte-verified invented broker tree."""
        from energraph.broker_tree import inspect_tree

        source = self._confined(source, "broker tree")
        destination = self._confined(destination, "broker destination", allow_root=True)
        if inspect_tree(source).digest != digest:
            raise ValueError("Invented broker-tree grant digest does not match content")
        for purpose, operation in operations:
            now = datetime.now(UTC)
            self.grants.append(
                {
                    "source_id": source_id,
                    "source_sha256": digest,
                    "source_path": str(source),
                    "purpose": purpose,
                    "operation": operation,
                    "destination_root": str(destination),
                    "evidence_reference": "invented-offline-broker-tree-fixture-only",
                    "status": "approved",
                    "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                    "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                    "retention_days": 1,
                }
            )
        self.path.write_text(json.dumps({"version": 1, "grants": self.grants}), encoding="utf-8")

    def source(self, incoming: Path, dataset_id: str, raw_root: Path) -> None:
        incoming = self._confined(incoming, "source")
        raw_root = self._confined(raw_root, "destination", allow_root=True)
        if incoming.parent != self.root:
            raise ValueError("Invented source must be an immediate test fixture directory")
        if not (
            dataset_id in {"metr-la", "pems-bay", "future-city"}
            or dataset_id.startswith(("invented-", "synthetic-"))
            or dataset_id.endswith("-fixture")
        ):
            raise ValueError("Invented fixture dataset identity is not supported")
        records = tuple(
            RawFileRecord(name, *stream_file(incoming / name)) for name in discover(incoming).files
        )
        self._grant(dataset_id, incoming, content_identity(records), "raw_ingest", raw_root)

    def snapshot(self, raw: DatasetSnapshot, canonical_root: Path) -> None:
        source_id = raw.directory.parent.name
        self._grant(
            source_id,
            raw.directory,
            raw.manifest.snapshot_id,
            "snapshot_parse",
            raw.directory.parent,
        )
        for operation in ("canonical_create", "canonical_read"):
            self._grant(
                source_id,
                raw.directory,
                raw.manifest.snapshot_id,
                operation,
                canonical_root,
            )

    def model_use(self, raw: DatasetSnapshot, output_root: Path) -> None:
        for purpose, operation in (
            ("model_training", "model_train"),
            ("scientific_evaluation", "model_evaluate"),
            ("model_training", "model_publish"),
            ("saved_run_verification", "model_verify"),
            ("benchmark_comparison", "model_compare"),
        ):
            self._grant(
                raw.directory.parent.name,
                raw.directory,
                raw.manifest.snapshot_id,
                operation,
                output_root,
                purpose,
            )


@pytest.fixture
def invented_research_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> InventedResearchRights:
    return InventedResearchRights(tmp_path, monkeypatch)
