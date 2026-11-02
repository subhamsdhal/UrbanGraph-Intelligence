"""Zero-Downtime Token and Secret Rotation Engine for Governed Broker Services.

Satisfies Requirement R23 & Distributed Multi-Replica Security:
- Cryptographically secure token generation (32-byte CSPRNG hex).
- Dual-token grace period: allows in-flight worker requests to complete while rotating.
- Constant-time comparison preventing timing attacks (`hmac.compare_digest`).
- Role-based isolation (admin, worker, reader).
- Secret-redacted audit history.
- Centralized token store support (SQLite WAL-mode ACID repository) for distributed multi-replica synchronization.
- Distributed safety enforcement: blocks unsafe process-local rotation in multi-replica deployments when no centralized store is configured.
"""

from __future__ import annotations

import hmac
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from urbangraph.data import DataError

RoleType = Literal["admin", "worker", "reader"]
ROLES = frozenset({"admin", "worker", "reader"})


@dataclass(frozen=True)
class TokenRecord:
    """Represents an active or grace-period authorization credential."""

    role: RoleType
    token: str
    created_at_utc: str
    expires_at_epoch: float
    is_primary: bool
    is_revoked: bool = False

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "token_fingerprint": self.token[:6] + "..." + self.token[-4:],
            "created_at_utc": self.created_at_utc,
            "expires_at_epoch": self.expires_at_epoch,
            "is_primary": self.is_primary,
            "is_revoked": self.is_revoked,
        }


class TokenRotationEngine:
    """Coordinates zero-downtime secret rotation and validation across broker roles.
    
    Supports both in-memory standalone operation and centralized SQLite persistent
    store for multi-replica Kubernetes / distributed cluster deployments.
    """

    def __init__(
        self,
        rotation_ttl_seconds: float = 3600.0,
        grace_period_seconds: float = 900.0,
        store_path: str | Path | None = None,
    ) -> None:
        if rotation_ttl_seconds <= 0 or grace_period_seconds <= 0:
            raise DataError("TTL and grace period must be strictly positive numbers")
        self._lock = threading.RLock()
        self.rotation_ttl = rotation_ttl_seconds
        self.grace_period = grace_period_seconds

        # Resolve centralized store path from argument or environment
        env_store = os.environ.get("URBANGRAPH_TOKEN_STORE_PATH")
        target_store = store_path or env_store
        self.store_path: Path | None = Path(target_store).resolve() if target_store else None

        # Distributed mode flag
        self.is_multi_replica = os.environ.get("URBANGRAPH_MULTI_REPLICA", "").lower() in ("1", "true", "yes")

        # In-memory storage for standalone mode
        self._active_tokens: dict[RoleType, list[TokenRecord]] = {}
        self._rotation_audit_log: list[dict[str, Any]] = []

        if self.store_path:
            self._init_sqlite_store()
        else:
            # Initialize credentials in memory for all supported roles
            for role in ("admin", "worker", "reader"):
                self._rotate_internal_memory(role)  # type: ignore

    def _get_db_conn(self) -> sqlite3.Connection:
        if not self.store_path:
            raise DataError("Centralized store path not configured")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.store_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _init_sqlite_store(self) -> None:
        with self._lock, self._get_db_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    expires_at_epoch REAL NOT NULL,
                    is_primary INTEGER NOT NULL,
                    is_revoked INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    token_fingerprint TEXT NOT NULL,
                    has_grace_period INTEGER NOT NULL
                )
            """)
            conn.commit()

            # Ensure all roles have an active primary token
            now = time.time()
            for role in ("admin", "worker", "reader"):
                cursor = conn.execute(
                    "SELECT token FROM tokens WHERE role = ? AND is_primary = 1 AND is_revoked = 0 AND expires_at_epoch > ?",
                    (role, now),
                )
                if not cursor.fetchone():
                    self._rotate_internal_db(conn, role)  # type: ignore
            conn.commit()

    def _rotate_internal_memory(self, role: RoleType) -> str:
        now = time.time()
        now_utc = datetime.now(UTC).isoformat()
        new_token_str = secrets.token_hex(32)

        new_record = TokenRecord(
            role=role,
            token=new_token_str,
            created_at_utc=now_utc,
            expires_at_epoch=now + self.rotation_ttl + self.grace_period,
            is_primary=True,
            is_revoked=False,
        )

        current_records = self._active_tokens.get(role, [])
        updated_records: list[TokenRecord] = [new_record]

        # Demote previous primary token to grace period
        for rec in current_records:
            if rec.is_primary and (rec.expires_at_epoch > now) and not rec.is_revoked:
                demoted = TokenRecord(
                    role=rec.role,
                    token=rec.token,
                    created_at_utc=rec.created_at_utc,
                    expires_at_epoch=now + self.grace_period,
                    is_primary=False,
                    is_revoked=False,
                )
                updated_records.append(demoted)
                break

        self._active_tokens[role] = updated_records

        self._rotation_audit_log.append({
            "action": "TOKEN_ROTATED",
            "role": role,
            "timestamp_utc": now_utc,
            "new_token_fingerprint": new_record.redacted_dict()["token_fingerprint"],
            "has_grace_period": len(updated_records) > 1,
        })

        return new_token_str

    def _rotate_internal_db(self, conn: sqlite3.Connection, role: RoleType) -> str:
        now = time.time()
        now_utc = datetime.now(UTC).isoformat()
        new_token_str = secrets.token_hex(32)
        new_expiry = now + self.rotation_ttl + self.grace_period

        # Demote any current active primary token to grace period
        cursor = conn.execute(
            "SELECT token, created_at_utc FROM tokens WHERE role = ? AND is_primary = 1 AND is_revoked = 0 AND expires_at_epoch > ?",
            (role, now),
        )
        old_primary = cursor.fetchone()
        has_grace = False
        if old_primary:
            has_grace = True
            conn.execute(
                "UPDATE tokens SET is_primary = 0, expires_at_epoch = ? WHERE token = ?",
                (now + self.grace_period, old_primary["token"]),
            )

        # Insert new primary token
        conn.execute(
            "INSERT INTO tokens (token, role, created_at_utc, expires_at_epoch, is_primary, is_revoked) VALUES (?, ?, ?, ?, 1, 0)",
            (new_token_str, role, now_utc, new_expiry),
        )

        # Audit log
        fingerprint = new_token_str[:6] + "..." + new_token_str[-4:]
        conn.execute(
            "INSERT INTO token_audit_log (action, role, timestamp_utc, token_fingerprint, has_grace_period) VALUES (?, ?, ?, ?, ?)",
            ("TOKEN_ROTATED", role, now_utc, fingerprint, 1 if has_grace else 0),
        )
        return new_token_str

    def rotate_token(self, role: RoleType) -> str:
        """Rotate secret for specified role and return new primary token."""
        if role not in ROLES:
            raise DataError(f"Unsupported role: {role}")

        # Enforce distributed safety check
        if self.is_multi_replica and not self.store_path:
            raise DataError(
                "Dynamic token rotation is securely restricted in distributed multi-replica mode "
                "without an authorized centralized token store. Configure URBANGRAPH_TOKEN_STORE_PATH "
                "or distribute updated static tokens via Kubernetes Secrets."
            )

        with self._lock:
            if self.store_path:
                with self._get_db_conn() as conn:
                    new_token = self._rotate_internal_db(conn, role)
                    conn.commit()
                    return new_token
            return self._rotate_internal_memory(role)

    def revoke_token(self, role: RoleType, candidate_token: str | bytes) -> bool:
        """Revoke a token immediately."""
        if role not in ROLES or not candidate_token:
            return False
        token_str = (
            candidate_token.decode("utf-8", errors="replace").strip()
            if isinstance(candidate_token, bytes)
            else str(candidate_token).strip()
        )
        now_utc = datetime.now(UTC).isoformat()
        fingerprint = token_str[:6] + "..." + token_str[-4:]

        with self._lock:
            if self.store_path:
                with self._get_db_conn() as conn:
                    cursor = conn.execute("SELECT token FROM tokens WHERE token = ? AND role = ?", (token_str, role))
                    if not cursor.fetchone():
                        return False
                    conn.execute("UPDATE tokens SET is_revoked = 1 WHERE token = ? AND role = ?", (token_str, role))
                    conn.execute(
                        "INSERT INTO token_audit_log (action, role, timestamp_utc, token_fingerprint, has_grace_period) VALUES (?, ?, ?, ?, 0)",
                        ("TOKEN_REVOKED", role, now_utc, fingerprint),
                    )
                    conn.commit()
                    return True
            else:
                found = False
                for rec in self._active_tokens.get(role, []):
                    if hmac.compare_digest(rec.token, token_str):
                        found = True
                        break
                if found:
                    self._active_tokens[role] = [
                        TokenRecord(
                            role=r.role,
                            token=r.token,
                            created_at_utc=r.created_at_utc,
                            expires_at_epoch=r.expires_at_epoch,
                            is_primary=r.is_primary,
                            is_revoked=True if hmac.compare_digest(r.token, token_str) else r.is_revoked,
                        )
                        for r in self._active_tokens.get(role, [])
                    ]
                    self._rotation_audit_log.append({
                        "action": "TOKEN_REVOKED",
                        "role": role,
                        "timestamp_utc": now_utc,
                        "token_fingerprint": fingerprint,
                        "has_grace_period": False,
                    })
                    return True
                return False

    def validate_token(self, role: RoleType, candidate_token: str | bytes) -> bool:
        """Validate candidate token with constant-time comparison against primary and grace tokens."""
        if role not in ROLES:
            return False
        if not candidate_token:
            return False

        if isinstance(candidate_token, bytes):
            candidate_str = candidate_token.decode("utf-8", errors="replace").strip()
        else:
            candidate_str = str(candidate_token).strip()

        now = time.time()
        with self._lock:
            if self.store_path:
                with self._get_db_conn() as conn:
                    cursor = conn.execute(
                        "SELECT token, expires_at_epoch, is_revoked FROM tokens WHERE role = ?",
                        (role,),
                    )
                    rows = cursor.fetchall()
                    for row in rows:
                        if row["is_revoked"] == 0 and row["expires_at_epoch"] > now:
                            if hmac.compare_digest(row["token"], candidate_str):
                                return True
                return False
            else:
                records = self._active_tokens.get(role, [])
                for rec in records:
                    if not rec.is_revoked and rec.expires_at_epoch > now:
                        if hmac.compare_digest(rec.token, candidate_str):
                            return True
                return False

    def get_primary_token(self, role: RoleType) -> str:
        """Retrieve current primary active token for role."""
        if role not in ROLES:
            raise DataError(f"Unsupported role: {role}")
        now = time.time()
        with self._lock:
            if self.store_path:
                with self._get_db_conn() as conn:
                    cursor = conn.execute(
                        "SELECT token FROM tokens WHERE role = ? AND is_primary = 1 AND is_revoked = 0 AND expires_at_epoch > ? ORDER BY expires_at_epoch DESC",
                        (role, now),
                    )
                    row = cursor.fetchone()
                    if row:
                        return row["token"]
                raise DataError(f"No active primary token found for {role}")
            else:
                for rec in self._active_tokens.get(role, []):
                    if rec.is_primary and not rec.is_revoked and rec.expires_at_epoch > now:
                        return rec.token
                raise DataError(f"No active primary token found for {role}")

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return redacted secret rotation audit log."""
        with self._lock:
            if self.store_path:
                with self._get_db_conn() as conn:
                    cursor = conn.execute(
                        "SELECT action, role, timestamp_utc, token_fingerprint, has_grace_period FROM token_audit_log ORDER BY id ASC"
                    )
                    return [
                        {
                            "action": row["action"],
                            "role": row["role"],
                            "timestamp_utc": row["timestamp_utc"],
                            "token_fingerprint": row["token_fingerprint"],
                            "has_grace_period": bool(row["has_grace_period"]),
                        }
                        for row in cursor.fetchall()
                    ]
            return list(self._rotation_audit_log)

    def get_active_inventory(self) -> dict[str, list[dict[str, Any]]]:
        """Return inventory of active tokens with secrets redacted."""
        now = time.time()
        with self._lock:
            if self.store_path:
                with self._get_db_conn() as conn:
                    inv: dict[str, list[dict[str, Any]]] = {}
                    for role in ("admin", "worker", "reader"):
                        cursor = conn.execute(
                            "SELECT token, role, created_at_utc, expires_at_epoch, is_primary, is_revoked FROM tokens WHERE role = ? AND is_revoked = 0 AND expires_at_epoch > ?",
                            (role, now),
                        )
                        inv[role] = [
                            TokenRecord(
                                role=row["role"],
                                token=row["token"],
                                created_at_utc=row["created_at_utc"],
                                expires_at_epoch=row["expires_at_epoch"],
                                is_primary=bool(row["is_primary"]),
                                is_revoked=bool(row["is_revoked"]),
                            ).redacted_dict()
                            for row in cursor.fetchall()
                        ]
                    return inv
            return {
                role: [r.redacted_dict() for r in records if not r.is_revoked and r.expires_at_epoch > now]
                for role, records in self._active_tokens.items()
            }
