"""Portable, synthetic-only evaluation evidence with strict local verification."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Any

from energraph.scientific_validation import evaluate_demo

SCHEMA = "eg007b-synthetic-evidence-1.0"
FILES = frozenset({"report.json", "manifest.json"})
MAX_FILE_BYTES = 10_000_000


class EvidenceError(ValueError):
    """The evidence package or destination violates the contract."""


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_directory(path: Path, *, must_exist: bool) -> None:
    if not path.is_absolute():
        raise EvidenceError("An absolute directory path is required")
    for entry in (path, *path.parents):
        try:
            info = entry.lstat()
        except FileNotFoundError as exc:
            raise EvidenceError("A directory path component does not exist") from exc
        is_reparse = os.name == "nt" and bool(
            info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
        if entry.is_symlink() or is_reparse:
            raise EvidenceError("Reparse points and symbolic links are forbidden")
        if not entry.is_dir():
            raise EvidenceError("A path component is not a directory")
    if must_exist and not path.is_dir():
        raise EvidenceError("Package directory does not exist")


def _outside_checkout(path: Path) -> None:
    checkout = Path(__file__).resolve().parents[2]
    resolved = path.resolve(strict=False)
    if resolved == checkout or checkout in resolved.parents:
        raise EvidenceError("Evidence destination must be outside the source checkout")


def _directory_identity(path: Path) -> tuple[int, int]:
    _safe_directory(path, must_exist=True)
    info = path.lstat()
    return info.st_dev, info.st_ino


def _creation_destination(output: Path) -> Path:
    _safe_directory(output.parent, must_exist=True)
    if output.name in {"", ".", ".."}:
        raise EvidenceError("A new directory name is required")
    canonical = output.parent.resolve(strict=True) / output.name
    _safe_directory(canonical.parent, must_exist=True)
    _outside_checkout(canonical)
    return canonical


def _read_exact(path: Path) -> bytes:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or (os.name == "nt" and bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT))
    ):
        raise EvidenceError("Package entries must be ordinary files")
    if info.st_size > MAX_FILE_BYTES:
        raise EvidenceError("Package entry exceeds size limit")
    data = path.read_bytes()
    if len(data) != info.st_size:
        raise EvidenceError("Package entry changed during verification")
    return data


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"Nonfinite JSON constant: {value}")


def _strict_json(data: bytes) -> Any:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant
        )
        _canonical(value)  # Also rejects overflowed numeric values.
    except (UnicodeError, ValueError, TypeError, OverflowError) as exc:
        raise EvidenceError("Invalid or nonfinite JSON") from exc
    return value


def _publish(stage: Path, output: Path) -> None:
    if os.name == "nt":
        os.rename(stage, output)  # Windows refuses an existing directory.
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise EvidenceError("Atomic no-replace directory publish is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(output))
    else:
        raise EvidenceError("Atomic no-replace directory publish is unavailable")


def create_package(output: Path) -> dict[str, Any]:
    """Publish an immutable new package from the fixed, verified synthetic fixture."""
    output = _creation_destination(output)
    if output.exists() or output.is_symlink():
        raise EvidenceError("Destination already exists")
    parent_identity = _directory_identity(output.parent)
    report = _canonical(evaluate_demo())
    manifest = _canonical(
        {
            "schema_version": SCHEMA,
            "content_id": _digest(report),
            "files": [
                {
                    "relative_path": "report.json",
                    "size_bytes": len(report),
                    "sha256": _digest(report),
                }
            ],
        }
    )
    if (
        _creation_destination(output) != output
        or _directory_identity(output.parent) != parent_identity
    ):
        raise EvidenceError("Destination parent changed during evaluation")
    stage = output.parent / f".eg007b-stage-{uuid.uuid4().hex}"
    stage.mkdir(exist_ok=False)
    stage_identity = _directory_identity(stage)
    try:
        for name, data in (("report.json", report), ("manifest.json", manifest)):
            with (stage / name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        if _directory_identity(output.parent) != parent_identity:
            raise EvidenceError("Destination parent changed during publication")
        if _directory_identity(stage) != stage_identity:
            raise EvidenceError("Staging directory changed during publication")
        _outside_checkout(output)
        _publish(stage, output)
    finally:
        # Remove only the two files exclusively created by this invocation.
        if (
            stage.exists()
            and _directory_identity(output.parent) == parent_identity
            and _directory_identity(stage) == stage_identity
            and {item.name for item in stage.iterdir()} <= FILES
        ):
            for name in FILES:
                if (stage / name).exists():
                    (stage / name).unlink()
            stage.rmdir()
    return verify_package(output)


def verify_package(package: Path) -> dict[str, Any]:
    """Read an untrusted package without using it as an evaluation input."""
    _safe_directory(package, must_exist=True)
    if {item.name for item in package.iterdir()} != FILES:
        raise EvidenceError("Package inventory differs from the two-file contract")
    report_bytes = _read_exact(package / "report.json")
    manifest_bytes = _read_exact(package / "manifest.json")
    report = _strict_json(report_bytes)
    manifest = _strict_json(manifest_bytes)
    if not isinstance(report, dict) or not isinstance(manifest, dict):
        raise EvidenceError("Report and manifest must be JSON objects")
    if report_bytes != _canonical(report) or manifest_bytes != _canonical(manifest):
        raise EvidenceError("Package JSON is not canonical")
    expected_manifest = {
        "schema_version": SCHEMA,
        "content_id": _digest(report_bytes),
        "files": [
            {
                "relative_path": "report.json",
                "size_bytes": len(report_bytes),
                "sha256": _digest(report_bytes),
            }
        ],
    }
    if manifest != expected_manifest:
        raise EvidenceError("Manifest schema or digest mismatch")
    if report_bytes != _canonical(evaluate_demo()):
        raise EvidenceError("Report differs from fresh trusted synthetic evaluation")
    return {
        "status": "PASS",
        "schema_version": SCHEMA,
        "content_id": _digest(report_bytes),
        "file_count": 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (
            create_package(args.output_dir)
            if args.command == "create"
            else verify_package(args.package_dir)
        )
    except (EvidenceError, OSError, ValueError) as exc:
        print(f"HOLD: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
