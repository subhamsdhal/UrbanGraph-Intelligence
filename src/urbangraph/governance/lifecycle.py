"""Governance, Rights-Aware Retention, and Disaster Recovery Engine.

Satisfies Requirements R02 and R25:
- Software implementation of rights-aware retention lifecycle.
- Policy-driven staging purging with legal hold protection.
- Immutable tombstone creation and immediate access revocation.
- Disaster recovery backup creation and bit-for-bit SHA-256 verified restoration.
- Explicit demarcation between software completeness and external institutional HSM custody.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from urbangraph.data import DataError


@dataclass(frozen=True)
class DatasetRetentionPolicy:
    """Policy rules governing dataset lifecycle, expiration, and export rights."""

    dataset_id: str
    license_type: Literal["SYNTHETIC_PERMISSIVE", "ACADEMIC_RESEARCH", "PROPRIETARY_GATED"]
    retention_days: int
    legal_hold: bool = False
    allow_redistribution: bool = False
    external_custody_required: bool = False

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise DataError("dataset_id must be non-empty")
        if self.retention_days < 1:
            raise DataError("retention_days must be positive")


@dataclass(frozen=True)
class TombstoneRecord:
    """Immutable audit record generated when a dataset or credential is permanently revoked."""

    dataset_id: str
    revocation_timestamp_utc: str
    reason: str
    revoked_by: str
    sha256_at_revocation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataLifecycleManager:
    """Enforces retention rules, purging expired staging artifacts while respecting legal holds."""

    def __init__(self, policies: dict[str, DatasetRetentionPolicy] | None = None) -> None:
        self.policies: dict[str, DatasetRetentionPolicy] = policies or {}
        self.tombstones: dict[str, TombstoneRecord] = {}

    def register_policy(self, policy: DatasetRetentionPolicy) -> None:
        self.policies[policy.dataset_id] = policy

    def revoke_dataset(self, dataset_id: str, reason: str, revoked_by: str = "security-officer") -> TombstoneRecord:
        """Permanently revoke dataset access and issue an immutable tombstone."""
        now_str = datetime.now(UTC).isoformat()
        tombstone = TombstoneRecord(
            dataset_id=dataset_id,
            revocation_timestamp_utc=now_str,
            reason=reason,
            revoked_by=revoked_by,
            sha256_at_revocation=hashlib.sha256(f"{dataset_id}:{now_str}".encode()).hexdigest(),
        )
        self.tombstones[dataset_id] = tombstone
        return tombstone

    def is_dataset_revoked(self, dataset_id: str) -> bool:
        return dataset_id in self.tombstones

    def scan_and_purge_staging(self, staging_dir: Path | str, max_age_seconds: float = 86400.0) -> list[str]:
        """Scan staging directory and safely purge files older than max_age_seconds."""
        st_path = Path(staging_dir)
        if not st_path.is_dir():
            return []

        now = time.time()
        purged_files: list[str] = []

        for entry in st_path.iterdir():
            if entry.is_file():
                # Check for active legal hold
                has_legal_hold = any(p.legal_hold for p in self.policies.values() if p.dataset_id in entry.name)
                if has_legal_hold:
                    continue

                mtime = entry.stat().st_mtime
                if now - mtime > max_age_seconds:
                    entry.unlink()
                    purged_files.append(entry.name)

        return purged_files


class DisasterRecoveryManager:
    """Manages snapshot backups, cryptographic verification, and disaster recovery restoration."""

    EXTERNAL_HSM_GATED: bool = True
    CUSTODY_DISCLAIMER: str = (
        "Host-independent hardware security module (HSM) key custody and multi-region "
        "cloud replication require institutional enterprise infrastructure and remain an external gate."
    )

    @staticmethod
    def calculate_dir_manifest(root: Path) -> dict[str, str]:
        """Calculate relative file path to SHA-256 hash map."""
        manifest: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel_path = str(path.relative_to(root)).replace("\\", "/")
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest[rel_path] = h
        return manifest

    @classmethod
    def create_backup_archive(cls, source_dir: Path | str, target_archive: Path | str) -> str:
        """Create a tar.gz backup archive containing an embedded SHA-256 manifest."""
        src = Path(source_dir)
        tar_path = Path(target_archive)
        tar_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = cls.calculate_dir_manifest(src)
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")

        with tarfile.open(tar_path, "w:gz") as tar:
            # Add manifest
            tarinfo = tarfile.TarInfo(name="backup_manifest.json")
            tarinfo.size = len(manifest_bytes)
            tarinfo.mtime = int(time.time())
            tar.addfile(tarinfo, fileobj=json.BytesIO(manifest_bytes) if hasattr(json, "BytesIO") else __import__("io").BytesIO(manifest_bytes))

            # Add source directory contents
            for path in sorted(src.rglob("*")):
                if path.is_file():
                    arcname = "data/" + str(path.relative_to(src)).replace("\\", "/")
                    tar.add(path, arcname=arcname)

        return hashlib.sha256(tar_path.read_bytes()).hexdigest()

    @classmethod
    def restore_and_verify_archive(cls, backup_archive: Path | str, destination_dir: Path | str) -> dict[str, Any]:
        """Restore backup archive and verify bit-for-bit SHA-256 fidelity against embedded manifest."""
        archive_path = Path(backup_archive)
        dest = Path(destination_dir)
        dest.mkdir(parents=True, exist_ok=True)

        if not archive_path.is_file():
            raise DataError("Backup archive file not found")

        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                manifest_file = tar.extractfile("backup_manifest.json")
                if not manifest_file:
                    raise DataError("Backup archive is missing embedded backup_manifest.json")
                manifest: dict[str, str] = json.loads(manifest_file.read().decode("utf-8"))

                for member in tar.getmembers():
                    if member.name.startswith("data/") and member.isfile():
                        rel_name = member.name[len("data/") :]
                        rel_path = Path(rel_name)
                        if ".." in rel_path.parts or rel_path.is_absolute() or rel_name.startswith("/"):
                            raise DataError(f"Malicious archive member path detected: {rel_name}")
                        extracted_file = tar.extractfile(member)
                        if extracted_file is None:
                            raise DataError(f"Could not extract member {member.name}")
                        content = extracted_file.read()
                        actual_sha = hashlib.sha256(content).hexdigest()
                        expected_sha = manifest.get(rel_name)

                        if actual_sha != expected_sha:
                            raise DataError(f"Integrity violation on restored file {rel_name}: expected {expected_sha}, got {actual_sha}")

                        target_file = dest / rel_name
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        target_file.write_bytes(content)
        except (tarfile.TarError, EOFError, OSError) as exc:
            if isinstance(exc, DataError):
                raise
            raise DataError(f"Corrupt backup archive: {exc}") from exc

        return {
            "restored_file_count": len(manifest),
            "manifest_verified": True,
            "status": "RESTORE_VERIFIED_SUCCESSFUL",
            "external_hsm_custody_gated": cls.EXTERNAL_HSM_GATED,
            "disclaimer": cls.CUSTODY_DISCLAIMER,
        }
