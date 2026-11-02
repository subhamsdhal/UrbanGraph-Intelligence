"""Docker-scoped, broker-owned registry for invented file and tree artifacts.

The registry is authoritative only when a separate broker process owns its
database and artifact store. This module does not establish Windows host custody
or connect existing EnerGraph publishers and readers to the broker.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path

from energraph.broker_contract import parse_committed_record
from energraph.broker_tree import TreeIdentity, copy_verified_tree, inspect_tree
from urbangraph.data import DataError
from urbangraph.data.contracts import validate_relative_path
from urbangraph.data.rights import (
    AuthorizationRequest,
    require_authorization_decisions,
    require_raw_authorization,
)

_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")


def _name(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise DataError("Invalid broker identifier")
    return value


def _expected_digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DataError("Invalid expected broker digest")
    return value


def _stage_bytes(root: Path, stage_id: str) -> bytes:
    path = root / _name(stage_id)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
                raise DataError("Invalid broker staging object")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                content = handle.read(16 * 1024 * 1024 + 1)
            if len(content) != info.st_size:
                raise DataError("Broker stage changed while reading")
            return content
        finally:
            os.close(fd)
    except OSError as exc:
        raise DataError("Broker stage unavailable") from exc


class BrokerRegistry:
    """Single-writer registry; HTTP authentication is enforced by the service."""

    def __init__(self, db_path: Path, staging: Path, artifacts: Path):
        self.db_path = db_path
        self.staging = staging
        self.artifacts = artifacts
        for directory in (db_path.parent, staging, artifacts):
            directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS revision (
                    id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO revision VALUES (1, 0);
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, artifact_id TEXT UNIQUE NOT NULL,
                    digest TEXT NOT NULL, source_path TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, digest TEXT NOT NULL, kind TEXT NOT NULL,
                    parents TEXT NOT NULL, sources TEXT NOT NULL,
                    stage_id TEXT NOT NULL, transformation TEXT NOT NULL,
                    state TEXT NOT NULL, revision INTEGER NOT NULL,
                    issuer TEXT NOT NULL, audit_id TEXT NOT NULL,
                    purpose TEXT NOT NULL DEFAULT '', operation TEXT NOT NULL DEFAULT '',
                    storage TEXT NOT NULL DEFAULT 'file', inventory TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS audit (
                    revision INTEGER PRIMARY KEY, event TEXT NOT NULL,
                    identity TEXT NOT NULL, utc TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(artifacts)")}
            for column in ("purpose", "operation", "storage", "inventory"):
                if column not in columns:
                    default = "'file'" if column == "storage" else "''"
                    db.execute(
                        f"ALTER TABLE artifacts ADD COLUMN {column} TEXT NOT NULL DEFAULT {default}"
                    )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _revision(db: sqlite3.Connection, event: str, identity: str) -> int:
        db.execute("UPDATE revision SET value=value+1 WHERE id=1")
        value = int(db.execute("SELECT value FROM revision WHERE id=1").fetchone()[0])
        db.execute(
            "INSERT INTO audit VALUES (?,?,?,?)",
            (value, event, identity, datetime.now(UTC).isoformat()),
        )
        return value

    def _source_rights(
        self, db: sqlite3.Connection, source_ids: tuple[str, ...], purpose: str, operation: str
    ) -> None:
        requests: list[AuthorizationRequest] = []
        rows: list[sqlite3.Row] = []
        for source_id in source_ids:
            row = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
            if row is None or row["revoked"]:
                raise DataError("Current broker source rights unavailable")
            rows.append(row)
            requests.append(
                AuthorizationRequest(
                    source_id, row["source_path"], purpose, self.artifacts, operation
                )
            )
        decisions = require_authorization_decisions(tuple(requests))
        for row, decision in zip(rows, decisions, strict=True):
            if row["digest"] != decision.source_sha256:
                raise DataError("Broker source differs from current rights policy")

    def _store_bytes(self, artifact_id: str, content: bytes, digest: str) -> None:
        destination = self.artifacts / artifact_id
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise DataError("Broker artifact collision")
            return
        temporary = self.artifacts / (".owned-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise DataError("Broker staged content changed")
            if os.name == "nt":
                # Windows rename refuses an existing destination. Hard links can
                # be denied by the restricted host context even within one tree.
                os.rename(temporary, destination)
            else:
                # POSIX rename would replace, so use no-overwrite hard linking.
                os.link(temporary, destination)
        except FileExistsError as exc:
            raise DataError("Broker destination already exists") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def _store_tree(self, artifact_id: str, stage_id: str, expected: TreeIdentity) -> None:
        destination = self.artifacts / artifact_id
        if os.path.lexists(destination):
            if inspect_tree(destination) != expected:
                raise DataError("Broker directory destination conflicts with approved snapshot")
            # A prior crash may have left the complete tree before the registry
            # transition. This operation never replaces its bytes.
            return
        temporary = self.artifacts / (".owned-" + uuid.uuid4().hex)
        try:
            copy_verified_tree(self.staging / stage_id, temporary, expected)
            if os.path.lexists(destination):
                raise DataError("Broker directory destination already exists")
            os.rename(temporary, destination)
        finally:
            if os.path.lexists(temporary):
                shutil.rmtree(temporary)

    def _verify_stored(self, row: sqlite3.Row) -> None:
        path = self.artifacts / row["id"]
        if row["storage"] == "directory":
            actual = inspect_tree(path)
            if actual.digest != row["digest"] or actual.inventory_json() != row["inventory"]:
                raise DataError("Broker directory snapshot content changed")
        elif row["storage"] == "file":
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["digest"]:
                raise DataError("Broker artifact content changed")
        else:
            raise DataError("Unsupported broker artifact storage")

    def admit_source(
        self,
        *,
        stage_id: str,
        source_id: str,
        expected_sha256: str,
        issuer: str,
        directory: bool = False,
    ) -> str:
        """Policy-approver operation; staged bytes are verified by the broker."""
        _name(source_id)
        _name(issuer)
        source_path = self.staging / _name(stage_id)
        granted = require_raw_authorization(
            source_id, source_path, "local_research", self.artifacts, operation="broker_admit"
        )
        if directory:
            tree = inspect_tree(source_path)
            digest = tree.digest
        else:
            content = _stage_bytes(self.staging, stage_id)
            digest = hashlib.sha256(content).hexdigest()
        if digest != _expected_digest(expected_sha256) or digest != granted:
            raise DataError("Raw admission differs from approver-pinned content")
        artifact_id = "raw-" + uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM sources WHERE id=?", (source_id,)).fetchone():
                raise DataError("Source already admitted")
            if directory:
                self._store_tree(artifact_id, stage_id, tree)
            else:
                self._store_bytes(artifact_id, content, digest)
            db.execute(
                "INSERT INTO sources VALUES (?,?,?,?,0)",
                (source_id, artifact_id, digest, str(source_path)),
            )
            revision = self._revision(db, "admit_source", source_id)
            db.execute(
                "INSERT INTO artifacts "
                "(id,digest,kind,parents,sources,stage_id,transformation,state,revision,"
                "issuer,audit_id,storage,inventory) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    digest,
                    "raw",
                    "[]",
                    json.dumps([[source_id, digest]]),
                    stage_id,
                    "raw-admission",
                    "committed",
                    revision,
                    issuer,
                    uuid.uuid4().hex,
                    "directory" if directory else "file",
                    tree.inventory_json() if directory else "",
                ),
            )
            db.commit()
        return artifact_id

    def prepare(
        self,
        *,
        stage_id: str,
        parent_ids: tuple[str, ...],
        transformation: str,
        purpose: str,
        operation: str,
        directory: bool = False,
    ) -> str:
        """Runner request; the broker derives closure from committed parent rows."""
        _name(transformation)
        _name(purpose)
        _name(operation)
        if not parent_ids or len(parent_ids) > 64 or len(set(parent_ids)) != len(parent_ids):
            raise DataError("Invalid broker parent set")
        if directory:
            tree = inspect_tree(self.staging / _name(stage_id))
            digest = tree.digest
            storage = "directory"
            inventory = tree.inventory_json()
        else:
            content = _stage_bytes(self.staging, stage_id)
            digest = hashlib.sha256(content).hexdigest()
            storage = "file"
            inventory = ""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            parents: list[list[str]] = []
            parent_rows: list[sqlite3.Row] = []
            closure: dict[str, str] = {}
            for parent_id in sorted(parent_ids):
                row = db.execute(
                    "SELECT * FROM artifacts WHERE id=?", (_name(parent_id),)
                ).fetchone()
                if row is None or row["state"] != "committed":
                    raise DataError("Broker parent is unknown or uncommitted")
                parents.append([parent_id, row["digest"]])
                parent_rows.append(row)
                for source_id, source_digest in json.loads(row["sources"]):
                    if source_id in closure and closure[source_id] != source_digest:
                        raise DataError("Conflicting broker source ancestry")
                    closure[source_id] = source_digest
            self._source_rights(db, tuple(sorted(closure)), purpose, operation)
            # Current rights precede reads of protected parent content.
            for parent_row in parent_rows:
                try:
                    self._verify_stored(parent_row)
                except (OSError, DataError) as exc:
                    raise DataError("Broker parent content changed") from exc
            artifact_id = "derived-" + uuid.uuid4().hex
            revision = self._revision(db, "prepare", artifact_id)
            db.execute(
                "INSERT INTO artifacts "
                "(id,digest,kind,parents,sources,stage_id,transformation,state,revision,issuer,"
                "audit_id,purpose,operation,storage,inventory) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    artifact_id,
                    digest,
                    "derived",
                    json.dumps(parents),
                    json.dumps(sorted(closure.items())),
                    stage_id,
                    transformation,
                    "prepared",
                    revision,
                    "",
                    uuid.uuid4().hex,
                    purpose,
                    operation,
                    storage,
                    inventory,
                ),
            )
            db.commit()
        return artifact_id

    def prepared_identity(self, artifact_id: str) -> dict[str, object]:
        """Return only the broker-pinned identity of a still-pending publication."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (_name(artifact_id),)).fetchone()
            if row is None or row["state"] != "prepared" or row["kind"] != "derived":
                raise DataError("Broker publication is not pending approval")
            self._verify_stored_stage(row)
            return {
                "artifact_id": row["id"],
                "content_sha256": row["digest"],
                "parents": json.loads(row["parents"]),
                "purpose": row["purpose"],
                "operation": row["operation"],
                "revision": row["revision"],
                "state": row["state"],
            }

    def _verify_stored_stage(self, row: sqlite3.Row) -> None:
        """Recompute the exact staged bytes while the record is pending."""
        if row["storage"] == "directory":
            actual = inspect_tree(self.staging / row["stage_id"])
            if actual.digest != row["digest"] or actual.inventory_json() != row["inventory"]:
                raise DataError("Prepared publication content changed")
        elif row["storage"] == "file":
            content = _stage_bytes(self.staging, row["stage_id"])
            if hashlib.sha256(content).hexdigest() != row["digest"]:
                raise DataError("Prepared publication content changed")
        else:
            raise DataError("Unsupported prepared publication storage")

    def approve(self, artifact_id: str, *, issuer: str, expected_sha256: str) -> None:
        """Policy-approver operation; the runner cannot call this HTTP route."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (_name(artifact_id),)).fetchone()
            if row is None or row["state"] != "prepared" or row["kind"] != "derived":
                raise DataError("Broker publication is not prepared")
            if row["digest"] != _expected_digest(expected_sha256):
                raise DataError("Publication differs from approver-pinned content")
            sources = json.loads(row["sources"])
            self._source_rights(
                db,
                tuple(source_id for source_id, _ in sources),
                row["purpose"],
                row["operation"],
            )
            self._verify_stored_stage(row)
            db.execute(
                "UPDATE artifacts SET state='approved', issuer=? WHERE id=?",
                (_name(issuer), artifact_id),
            )
            self._revision(db, "approve", artifact_id)
            db.commit()

    def commit(self, artifact_id: str, *, purpose: str, operation: str) -> dict[str, object]:
        """Commit only approved, unchanged content with current ancestor rights."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (_name(artifact_id),)).fetchone()
            if row is None or row["state"] != "approved":
                raise DataError("Broker publication not approved")
            if row["purpose"] != purpose or row["operation"] != operation:
                raise DataError("Broker publication context differs from preparation")
            sources = json.loads(row["sources"])
            self._source_rights(
                db, tuple(source_id for source_id, _ in sources), purpose, operation
            )
            if row["storage"] == "directory":
                actual_tree = inspect_tree(self.staging / row["stage_id"])
                if (
                    actual_tree.digest != row["digest"]
                    or actual_tree.inventory_json() != row["inventory"]
                ):
                    raise DataError("Broker publication directory changed")
                self._store_tree(artifact_id, row["stage_id"], actual_tree)
            elif row["storage"] == "file":
                content = _stage_bytes(self.staging, row["stage_id"])
                if hashlib.sha256(content).hexdigest() != row["digest"]:
                    raise DataError("Broker publication content changed")
                self._store_bytes(artifact_id, content, row["digest"])
            else:
                raise DataError("Unsupported broker artifact storage")
            revision = self._revision(db, "commit", artifact_id)
            db.execute(
                "UPDATE artifacts SET state='committed', revision=? WHERE id=?",
                (revision, artifact_id),
            )
            db.commit()
        return self.authorize_read(artifact_id, purpose=purpose, operation=operation)

    def authorize_read(
        self, artifact_id: str, *, purpose: str, operation: str
    ) -> dict[str, object]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (_name(artifact_id),)).fetchone()
            if row is None or row["state"] != "committed":
                raise DataError("Broker artifact unavailable")
            sources = json.loads(row["sources"])
            self._source_rights(
                db, tuple(source_id for source_id, _ in sources), purpose, operation
            )
            for source_id, claimed_digest in sources:
                actual = db.execute(
                    "SELECT digest, artifact_id FROM sources WHERE id=?", (source_id,)
                ).fetchone()
                if actual is None or actual["digest"] != claimed_digest:
                    raise DataError("Broker source closure changed")
                source_artifact = db.execute(
                    "SELECT * FROM artifacts WHERE id=?", (actual["artifact_id"],)
                ).fetchone()
                if source_artifact is None or source_artifact["digest"] != claimed_digest:
                    raise DataError("Broker source artifact changed")
                try:
                    self._verify_stored(source_artifact)
                except (OSError, DataError) as exc:
                    raise DataError("Broker source content changed") from exc
            for parent_id, parent_digest in json.loads(row["parents"]):
                actual_parent = db.execute(
                    "SELECT * FROM artifacts WHERE id=?", (parent_id,)
                ).fetchone()
                if (
                    actual_parent is None
                    or actual_parent["state"] != "committed"
                    or actual_parent["digest"] != parent_digest
                ):
                    raise DataError("Broker direct parent changed")
                try:
                    self._verify_stored(actual_parent)
                except (OSError, DataError) as exc:
                    raise DataError("Broker direct parent content changed") from exc
            self._verify_stored(row)
            result: dict[str, object] = {
                "version": 2 if row["storage"] == "directory" else 1,
                "state": "committed",
                "artifact_id": row["id"],
                "content_sha256": row["digest"],
                "kind": row["kind"],
                "parents": [
                    {"artifact_id": identity, "content_sha256": digest}
                    for identity, digest in json.loads(row["parents"])
                ],
                "sources": [
                    {"source_id": identity, "content_sha256": digest}
                    for identity, digest in sources
                ],
                "revision": row["revision"],
                "transformation": row["transformation"],
                "issuer": row["issuer"],
                "audit_id": row["audit_id"],
            }
            if row["storage"] == "directory":
                result["snapshot"] = json.loads(row["inventory"])
            parse_committed_record(result)
            return result

    def read_directory_file(
        self, artifact_id: str, relative_path: str, *, purpose: str, operation: str
    ) -> bytes:
        """Release one verified private file only after current closure authorization."""
        record = parse_committed_record(
            self.authorize_read(artifact_id, purpose=purpose, operation=operation)
        )
        if record.snapshot is None:
            raise DataError("Broker artifact is not a directory")
        validate_relative_path(relative_path)
        files = record.snapshot["files"]
        if not isinstance(files, list):
            raise DataError("Broker directory inventory unavailable")
        match = [item for item in files if item[0] == relative_path]
        if len(match) != 1:
            raise DataError("Requested broker file is outside authorized inventory")
        expected_size, expected_digest = match[0][1], match[0][2]
        path = self.artifacts / _name(artifact_id) / relative_path
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise DataError("Broker file is not an ordinary private file")
                if before.st_size != expected_size:
                    raise DataError("Broker file size changed")
                with os.fdopen(fd, "rb", closefd=False) as handle:
                    content = handle.read(expected_size + 1)
                after = os.fstat(fd)
                if (
                    len(content) != expected_size
                    or hashlib.sha256(content).hexdigest() != expected_digest
                    or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                ):
                    raise DataError("Broker file changed during authorized delivery")
                current = parse_committed_record(
                    self.authorize_read(artifact_id, purpose=purpose, operation=operation)
                )
                if current != record:
                    raise DataError("Broker authorization changed during delivery")
                return content
            finally:
                os.close(fd)
        except OSError as exc:
            raise DataError("Broker file unavailable") from exc

    def revoke(self, source_id: str) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("UPDATE sources SET revoked=1 WHERE id=?", (_name(source_id),)).rowcount
                != 1
            ):
                raise DataError("Unknown broker source")
            self._revision(db, "revoke", source_id)
            db.commit()
