"""Unit tests for R02 Governance Lifecycle and R25 Disaster Recovery Restoration."""

from __future__ import annotations

import os
import time
from pathlib import Path
import pytest

from urbangraph.data import DataError
from urbangraph.governance.lifecycle import (
    DataLifecycleManager,
    DatasetRetentionPolicy,
    DisasterRecoveryManager,
)


def test_retention_policy_and_dataset_revocation() -> None:
    policy = DatasetRetentionPolicy(
        dataset_id="traffic-pems-bay-subset",
        license_type="ACADEMIC_RESEARCH",
        retention_days=90,
        legal_hold=False,
    )
    assert policy.dataset_id == "traffic-pems-bay-subset"

    manager = DataLifecycleManager()
    manager.register_policy(policy)

    assert manager.is_dataset_revoked("traffic-pems-bay-subset") is False

    tombstone = manager.revoke_dataset(
        dataset_id="traffic-pems-bay-subset",
        reason="Research term expired; dataset access revoked per rights policy.",
        revoked_by="data-governance-officer",
    )

    assert tombstone.dataset_id == "traffic-pems-bay-subset"
    assert "Research term expired" in tombstone.reason
    assert manager.is_dataset_revoked("traffic-pems-bay-subset") is True


def test_staging_purge_with_legal_hold(tmp_path: Path) -> None:
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()

    # Create 2 staging files
    file_old = staging_dir / "ds_alpha_temp1.parquet"
    file_hold = staging_dir / "ds_beta_temp2.parquet"
    file_old.write_bytes(b"old_data")
    file_hold.write_bytes(b"hold_data")

    # Set file mtime to 10 days ago
    ten_days_ago = time.time() - 864000
    os.utime(file_old, (ten_days_ago, ten_days_ago))
    os.utime(file_hold, (ten_days_ago, ten_days_ago))

    policy_beta = DatasetRetentionPolicy(
        dataset_id="ds_beta",
        license_type="ACADEMIC_RESEARCH",
        retention_days=30,
        legal_hold=True,  # Active legal hold
    )
    manager = DataLifecycleManager()
    manager.register_policy(policy_beta)

    purged = manager.scan_and_purge_staging(staging_dir, max_age_seconds=86400)
    assert "ds_alpha_temp1.parquet" in purged
    assert not file_old.exists()

    # Protected by legal hold
    assert file_hold.exists()


def test_disaster_recovery_backup_and_restore(tmp_path: Path) -> None:
    src_dir = tmp_path / "source_registry"
    src_dir.mkdir()
    (src_dir / "registry.sqlite").write_bytes(b"SQLITE_MOCK_DATABASE_BINARY_PAYLOAD")
    (src_dir / "models" / "urbangraphformer").mkdir(parents=True)
    (src_dir / "models" / "urbangraphformer" / "model.pt").write_bytes(b"PYTORCH_WEIGHTS_123456")

    backup_tar = tmp_path / "backups" / "backup-2026-10-31.tar.gz"
    restore_dir = tmp_path / "restored_system"

    # 1. Create backup archive with manifest
    archive_sha = DisasterRecoveryManager.create_backup_archive(src_dir, backup_tar)
    assert backup_tar.is_file()
    assert len(archive_sha) == 64

    # 2. Restore and verify bit-for-bit integrity
    result = DisasterRecoveryManager.restore_and_verify_archive(backup_tar, restore_dir)
    assert result["status"] == "RESTORE_VERIFIED_SUCCESSFUL"
    assert result["restored_file_count"] == 2
    assert result["manifest_verified"] is True
    assert result["external_hsm_custody_gated"] is True

    assert (restore_dir / "registry.sqlite").read_bytes() == b"SQLITE_MOCK_DATABASE_BINARY_PAYLOAD"
    assert (restore_dir / "models" / "urbangraphformer" / "model.pt").read_bytes() == b"PYTORCH_WEIGHTS_123456"


def test_disaster_recovery_corruption_and_tampering(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "data.bin").write_bytes(b"ORIGINAL_VALID_PAYLOAD")

    backup_tar = tmp_path / "backup.tar.gz"
    DisasterRecoveryManager.create_backup_archive(src_dir, backup_tar)

    # Corrupt the archive by modifying archive payload bytes
    tar_bytes = bytearray(backup_tar.read_bytes())
    # Flip bytes
    for i in range(len(tar_bytes) - 50, len(tar_bytes)):
        tar_bytes[i] ^= 0xFF
    corrupt_tar = tmp_path / "corrupt.tar.gz"
    corrupt_tar.write_bytes(bytes(tar_bytes))

    with pytest.raises((DataError, Exception)):
        DisasterRecoveryManager.restore_and_verify_archive(corrupt_tar, tmp_path / "restored")


def test_disaster_recovery_traversal_rejection(tmp_path: Path) -> None:
    import tarfile
    import io
    import json

    malicious_tar = tmp_path / "malicious.tar.gz"
    manifest = {"../../evil.txt": "dummyhash"}
    manifest_bytes = json.dumps(manifest).encode("utf-8")

    with tarfile.open(malicious_tar, "w:gz") as tar:
        ti = tarfile.TarInfo(name="backup_manifest.json")
        ti.size = len(manifest_bytes)
        tar.addfile(ti, fileobj=io.BytesIO(manifest_bytes))

        ti2 = tarfile.TarInfo(name="data/../../evil.txt")
        ti2.size = 4
        tar.addfile(ti2, fileobj=io.BytesIO(b"evil"))

    with pytest.raises(DataError, match="Malicious archive member path"):
        DisasterRecoveryManager.restore_and_verify_archive(malicious_tar, tmp_path / "restored")

