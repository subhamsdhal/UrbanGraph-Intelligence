"""Explicit, fail-closed EnerGraph client for the Docker-scoped broker.

An artifact ID is issued by the broker, never inferred from a local path or
manifest. Approval remains a separate administrator action between prepare
and commit. Materialized directories are trusted temporary copies, not an
independent custody boundary against a hostile process sharing their owner.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from energraph.broker_contract import BrokerRecord, parse_committed_record
from energraph.broker_tree import copy_verified_tree, inspect_tree
from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute

_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")


def _identity(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise DataError("Invalid broker artifact identity")
    return value


def _token(path: Path) -> str:
    path = safe_absolute(path)
    if not path.is_file():
        raise DataError("Broker credential file unavailable")
    raw = path.read_bytes().strip()
    if not 32 <= len(raw) <= 256 or any(ch < 33 or ch > 126 for ch in raw):
        raise DataError("Broker credential invalid")
    return raw.decode("ascii")


@dataclass(frozen=True)
class PreparedDirectory:
    artifact_id: str
    content_sha256: str
    parents: tuple[tuple[str, str], ...]
    purpose: str
    operation: str
    revision: int


class BrokerAdapter:
    """Worker/reader transport; intentionally has no approval or admission method."""

    def __init__(
        self,
        base_url: str,
        *,
        worker_token_file: Path,
        reader_token_file: Path,
        staging_root: Path | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"broker", "localhost", "127.0.0.1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise DataError("Broker endpoint must be the approved local service")
        self.base_url = base_url.rstrip("/")
        self.worker_token_file = safe_absolute(worker_token_file)
        self.reader_token_file = safe_absolute(reader_token_file)
        self.staging_root = safe_absolute(staging_root) if staging_root is not None else None

    def stage_directory(self, source: Path) -> tuple[str, str]:
        """Copy a bounded tree to the broker's explicitly shared staging volume.

        The staging volume is an untrusted handoff area, not broker-private
        storage. The broker independently inventories and copies it at prepare
        and commit; deployments must mount only that volume into the runner.
        """
        if self.staging_root is None or not self.staging_root.is_dir():
            raise DataError("Broker shared staging volume is unavailable")
        root_info = os.lstat(self.staging_root)
        if not stat.S_ISDIR(root_info.st_mode):
            raise DataError("Broker shared staging root is not a plain directory")
        source = safe_absolute(source)
        expected = inspect_tree(source)
        stage_id = "energraph-" + uuid.uuid4().hex
        destination = self.staging_root / stage_id
        copy_verified_tree(source, destination, expected)
        if inspect_tree(destination) != expected:
            shutil.rmtree(destination)
            raise DataError("Broker staging copy failed verification")
        return stage_id, expected.digest

    def _request(self, route: str, body: dict[str, object], *, reader: bool) -> bytes:
        token = _token(self.reader_token_file if reader else self.worker_token_file)
        request = Request(
            self.base_url + route,
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise DataError("Broker request did not succeed")
                result: bytes = response.read(64 * 1024 * 1024 + 65536)
        except HTTPError as exc:
            if exc.code in {401, 403, 404}:
                raise DataError("Broker authorization or artifact unavailable") from exc
            raise DataError("Broker request failed") from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise DataError("Broker service unavailable") from exc
        if len(result) > 64 * 1024 * 1024 + 65535:
            raise DataError("Broker response exceeds bound")
        return result

    def _json(self, route: str, body: dict[str, object], *, reader: bool) -> dict[str, object]:
        try:
            value = json.loads(self._request(route, body, reader=reader))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise DataError("Invalid broker response") from exc
        if not isinstance(value, dict):
            raise DataError("Invalid broker response")
        return value

    def prepare_directory(
        self,
        *,
        stage_id: str,
        parent_ids: tuple[str, ...],
        transformation: str,
        purpose: str,
        operation: str,
    ) -> PreparedDirectory:
        if not parent_ids or len(set(parent_ids)) != len(parent_ids):
            raise DataError("Broker publication requires distinct parent IDs")
        body: dict[str, object] = {
            "stage_id": _identity(stage_id),
            "parent_ids": [_identity(item) for item in parent_ids],
            "transformation": _identity(transformation),
            "purpose": _identity(purpose),
            "operation": _identity(operation),
            "directory": True,
        }
        result = self._json("/v1/prepare", body, reader=False)
        expected_fields = {
            "artifact_id",
            "content_sha256",
            "parents",
            "purpose",
            "operation",
            "revision",
            "state",
        }
        if set(result) != expected_fields or result.get("state") != "prepared":
            raise DataError("Invalid broker preparation response")
        artifact_id = result["artifact_id"]
        digest = result["content_sha256"]
        parents = result["parents"]
        revision = result["revision"]
        if (
            not isinstance(artifact_id, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(revision) is not int
            or not isinstance(parents, list)
        ):
            raise DataError("Invalid broker preparation response")
        normalized: list[tuple[str, str]] = []
        for item in parents:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                or not re.fullmatch(r"[0-9a-f]{64}", item[1])
            ):
                raise DataError("Invalid broker preparation parent identity")
            normalized.append((_identity(item[0]), item[1]))
        if (
            tuple(sorted(item[0] for item in normalized)) != tuple(sorted(parent_ids))
            or result.get("purpose") != purpose
            or result.get("operation") != operation
        ):
            raise DataError("Broker preparation context or parents differ")
        return PreparedDirectory(
            _identity(artifact_id), digest, tuple(normalized), purpose, operation, revision
        )

    def commit_directory(self, prepared: PreparedDirectory) -> BrokerRecord:
        result = self._json(
            "/v1/commit",
            {
                "artifact_id": prepared.artifact_id,
                "purpose": prepared.purpose,
                "operation": prepared.operation,
            },
            reader=False,
        )
        record = parse_committed_record(result)
        if record.artifact_id != prepared.artifact_id or record.snapshot is None:
            raise DataError("Broker committed a different artifact")
        return record

    def authorize_directory(
        self, artifact_id: str, *, purpose: str, operation: str
    ) -> BrokerRecord:
        result = self._json(
            "/v1/authorize-read",
            {"artifact_id": _identity(artifact_id), "purpose": purpose, "operation": operation},
            reader=True,
        )
        record = parse_committed_record(result)
        if record.artifact_id != artifact_id or record.snapshot is None:
            raise DataError("Broker directory identity mismatch")
        return record

    def read_directory_file(
        self, record: BrokerRecord, relative_path: str, *, purpose: str, operation: str
    ) -> bytes:
        if record.snapshot is None:
            raise DataError("Broker directory inventory unavailable")
        files = cast(list[list[object]], record.snapshot["files"])
        entries = [item for item in files if item[0] == relative_path]
        if len(entries) != 1:
            raise DataError("Requested file is outside broker inventory")
        size, digest = entries[0][1], entries[0][2]
        content = self._request(
            "/v1/read-directory-file",
            {
                "artifact_id": record.artifact_id,
                "relative_path": relative_path,
                "purpose": purpose,
                "operation": operation,
            },
            reader=True,
        )
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise DataError("Broker-delivered file differs from authorized inventory")
        return content

    def materialize_directory(
        self, artifact_id: str, root: Path, *, purpose: str, operation: str
    ) -> Path:
        """Copy authorized broker bytes into a new trusted, caller-owned temp tree."""
        record = self.authorize_directory(artifact_id, purpose=purpose, operation=operation)
        root = safe_absolute(root)
        if not root.is_dir() or (os.name == "posix" and root.stat().st_mode & 0o022):
            raise DataError("Broker materialization root must be trusted and private")
        target = root / ("broker-read-" + uuid.uuid4().hex)
        target.mkdir(mode=0o700)
        try:
            snapshot = record.snapshot
            if snapshot is None:
                raise DataError("Broker directory inventory unavailable")
            directories = cast(list[str], snapshot["directories"])
            files = cast(list[tuple[str, int, str]], snapshot["files"])
            for relative in directories:
                (target / relative).mkdir(mode=0o700, parents=True)
            for relative, _, _ in files:
                destination = target / relative
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                content = self.read_directory_file(
                    record, relative, purpose=purpose, operation=operation
                )
                with destination.open("xb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
            if inspect_tree(target).digest != record.content_sha256:
                raise DataError("Materialized broker directory failed integrity verification")
            current = self.authorize_directory(artifact_id, purpose=purpose, operation=operation)
            if current != record:
                raise DataError("Broker authorization changed during materialization")
            return target
        except BaseException:
            shutil.rmtree(target)
            raise
