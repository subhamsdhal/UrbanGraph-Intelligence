"""Conservative local-filesystem traversal and bounded-memory byte copying."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from urbangraph.data.contracts import DataError, validate_path_inventory, validate_relative_path

CHUNK_SIZE = 1024 * 1024
# Windows FILE_ATTRIBUTE_REPARSE_POINT; stat omits the name on non-Windows hosts.
REPARSE_POINT_ATTRIBUTE = 0x400


def reject_link(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & REPARSE_POINT_ATTRIBUTE
    ):
        raise DataError("Symlinks and Windows reparse points are unsupported")
    return info


def safe_absolute(path: str | Path) -> Path:
    """Check existing ancestors before resolution; never traverse link ancestors."""
    original = Path(path)
    if ".." in original.parts:
        raise DataError("Parent traversal is not permitted in filesystem arguments")
    absolute = original.absolute()
    for candidate in (*reversed(absolute.parents), absolute):
        try:
            reject_link(candidate)
        except FileNotFoundError:
            continue
    # Resolve benign aliases only after rejecting links (e.g. Windows short paths).
    return absolute.resolve()


@dataclass(frozen=True)
class FileTree:
    files: tuple[str, ...]
    directories: tuple[str, ...]


def discover(directory: Path) -> FileTree:
    root = safe_absolute(directory)
    if not stat.S_ISDIR(reject_link(root).st_mode):
        raise DataError("Expected a directory")
    files: list[str] = []
    directories: list[str] = []
    pending = [root]
    names: dict[str, str] = {}
    while pending:
        current = pending.pop()
        reject_link(current)
        with os.scandir(current) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = current / entry.name
            relative = path.relative_to(root).as_posix()
            validate_relative_path(relative)
            key = relative.casefold()
            if key in names:
                raise DataError("Duplicate normalized path in source tree")
            names[key] = relative
            info = reject_link(path)
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
                pending.append(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                files.append(relative)
            else:
                raise DataError("Only regular files with a single hard link are supported")
    result = FileTree(tuple(sorted(files)), tuple(sorted(directories)))
    validate_path_inventory(result.files)
    return result


def _fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def stream_file(path: Path, destination: BinaryIO | None = None) -> tuple[int, str]:
    """Hash (and optionally copy) bytes; detect ordinary concurrent source changes.

    The directory tree must be under trusted local control throughout this call.
    Portable path checks cannot protect against an adversarial concurrent rename.
    """
    safe_absolute(path)
    before = reject_link(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise DataError("Expected a regular file with a single hard link")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        opened = os.fstat(source.fileno())
        # Windows Python versions can report different ctime semantics through
        # lstat and fstat. Compare common fields across APIs, then all fields
        # within each API before/after the read.
        if _fingerprint(before)[:4] != _fingerprint(opened)[:4]:
            raise DataError("File changed before reading")
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
            if destination is not None:
                destination.write(chunk)
        if _fingerprint(opened) != _fingerprint(os.fstat(source.fileno())):
            raise DataError("File changed while reading")
    if _fingerprint(before) != _fingerprint(reject_link(path)) or size != before.st_size:
        raise DataError("File changed during ingestion or verification")
    return size, digest.hexdigest()


def copy_file(source: Path, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_absolute(destination)
    with destination.open("xb") as output:
        result = stream_file(source, output)
        output.flush()
        os.fsync(output.fileno())
    return result
