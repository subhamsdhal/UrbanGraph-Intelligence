"""Version 1 canonical identity for bounded broker-owned directory snapshots.

The tree is untrusted until the broker independently enumerates and hashes it.
Docker-host administrators and concurrent hostile writers remain outside this
filesystem boundary; ordinary mutation is detected by repeated verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.data.contracts import validate_path_inventory, validate_relative_path
from urbangraph.data.filesystem import copy_file, reject_link, safe_absolute, stream_file

MAX_FILES = 4096
MAX_DIRECTORIES = 4096
MAX_DEPTH = 16
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TREE_BYTES = 256 * 1024 * 1024
MAX_RELATIVE_BYTES = 1024
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class TreeIdentity:
    digest: str
    directories: tuple[str, ...]
    files: tuple[tuple[str, int, str], ...]

    def inventory_json(self) -> str:
        return json.dumps(
            {"schema": "broker-tree-v1", "directories": self.directories, "files": self.files},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


def _portable_path(relative: str) -> None:
    try:
        validate_relative_path(relative)
    except DataError as exc:
        raise DataError("Broker tree has a nonportable path") from exc
    if len(relative.encode("utf-8")) > MAX_RELATIVE_BYTES:
        raise DataError("Broker tree path exceeds limit")
    if unicodedata.normalize("NFC", relative) != relative:
        raise DataError("Broker tree requires NFC path names")
    for component in relative.split("/"):
        if (
            component.rstrip(" .") != component
            or ":" in component
            or component.split(".", 1)[0].casefold() in _WINDOWS_RESERVED
        ):
            raise DataError("Broker tree has a nonportable path")


def inspect_tree(root: Path) -> TreeIdentity:
    """Enumerate the complete bounded tree without following links or reparse points."""
    root = safe_absolute(root)
    if not stat.S_ISDIR(reject_link(root).st_mode):
        raise DataError("Broker tree root must be a directory")
    pending = [root]
    directories: list[str] = []
    files: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    total = 0
    while pending:
        current = pending.pop()
        reject_link(current)
        with os.scandir(current) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            path = current / entry.name
            relative = path.relative_to(root).as_posix()
            if relative.count("/") + 1 > MAX_DEPTH:
                raise DataError("Broker tree nesting exceeds limit")
            _portable_path(relative)
            folded = relative.casefold()
            if folded in seen:
                raise DataError("Duplicate case-normalized broker tree path")
            seen.add(folded)
            info = reject_link(path)
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
                if len(directories) > MAX_DIRECTORIES:
                    raise DataError("Broker directory count exceeds limit")
                pending.append(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                if info.st_size > MAX_FILE_BYTES or total + info.st_size > MAX_TREE_BYTES:
                    raise DataError("Broker tree byte limit exceeded")
                size, digest = stream_file(path)
                total += size
                files.append((relative, size, digest))
                if len(files) > MAX_FILES:
                    raise DataError("Broker file count exceeds limit")
            else:
                raise DataError("Broker tree contains an unsupported filesystem object")
    if not files:
        raise DataError("Broker tree requires at least one regular file")
    validate_path_inventory(tuple(sorted(item[0] for item in files)))
    canonical = json.dumps(
        {"schema": "broker-tree-v1", "directories": sorted(directories), "files": sorted(files)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    digest = hashlib.sha256(b"urbangraph-broker-tree-v1\x00" + canonical).hexdigest()
    return TreeIdentity(digest, tuple(sorted(directories)), tuple(sorted(files)))


def copy_verified_tree(source: Path, destination: Path, expected: TreeIdentity) -> None:
    """Copy into a broker-owned, previously absent destination; refuse partial success."""
    if destination.exists() or destination.is_symlink():
        raise DataError("Broker destination already exists")
    if inspect_tree(source) != expected:
        raise DataError("Broker staged directory changed before copy")
    destination.mkdir(mode=0o700)
    try:
        for relative in expected.directories:
            (destination / relative).mkdir(mode=0o700, parents=True)
        for relative, size, digest in expected.files:
            target = destination / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if copy_file(source / relative, target) != (size, digest):
                raise DataError("Broker staged file changed while copying")
        if inspect_tree(source) != expected or inspect_tree(destination) != expected:
            raise DataError("Broker directory changed during ingestion")
    except BaseException:
        shutil.rmtree(destination)
        raise
