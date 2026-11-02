"""Compact, append-only evaluation plans with source and canonical lineage."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from urbangraph.data import DataError, RawFileRecord
from urbangraph.data.contracts import _environment
from urbangraph.data.filesystem import copy_file, discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.datasets.materialization import dependency_versions
from urbangraph.datasets.serialization import canonical_json, object_dict, read_json, write_json
from urbangraph.evaluation.compatibility import compatible_provenance
from urbangraph.evaluation.contracts import (
    EVALUATION_VERSION,
    POSITIONAL_PROTOCOL,
    REFERENCE_COMMIT,
    REFERENCE_URL,
    PlanConfig,
    SplitManifest,
    config_from_dict,
)
from urbangraph.evaluation.dataset import EvaluationDataset, identity, open_dataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.preprocessing import fit_standardizer
from urbangraph.evaluation.temporal import temporal_diagnostics
from urbangraph.reproducibility import collect_environment_info

_FILES = ("canonical-manifest.json", "fitted-transforms.json", "plan.json")


@dataclass(frozen=True)
class EvaluationManifest:
    plan_id: str
    directory: Path
    split: SplitManifest
    reused: bool


@dataclass(frozen=True)
class PlanVerification:
    valid: bool
    issues: tuple[str, ...]
    plan_id: str | None = None


def code_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    # File hashes also identify editable, uncommitted source; no invented Git revision.
    sources = {
        p.relative_to(root).as_posix(): stream_file(p)[1] for p in sorted(root.rglob("*.py"))
    }
    return {"source_sha256": sources, "dependencies": dependency_versions()}


def build_plan(
    dataset: EvaluationDataset, config: PlanConfig
) -> tuple[SplitManifest, dict[str, object], dict[str, object]]:
    split = plan_splits(dataset.timestamps, config)
    plan: dict[str, object] = {
        "schema_version": EVALUATION_VERSION,
        "lineage": dataset.lineage,
        "split": split.to_dict(),
        "code": code_provenance(),
        "reference": {
            "url": REFERENCE_URL,
            "commit": REFERENCE_COMMIT,
            "reviewed_on": "2026-10-10",
            "commit_status": "verified through upstream GitHub API and pinned source",
        },
        "metrics": {
            "value_space": "original",
            "aggregation": "micro over forecast cells",
            "mape": "percent; zero denominators excluded",
            "empty": "error or labeled null group",
        },
    }
    if config.protocol.name == POSITIONAL_PROTOCOL:
        plan["temporal_diagnostics"] = temporal_diagnostics(dataset.timestamps, split)
    fitted: dict[str, object] = {
        "schema_version": EVALUATION_VERSION,
        "folds": []
        if config.scaling == "none"
        else [fit_standardizer(dataset, split, fold=i).to_dict() for i in range(len(split.folds))],
    }
    return split, plan, fitted


def _read_verified(
    directory: Path, dataset: EvaluationDataset, *, staged: bool = False
) -> tuple[SplitManifest, dict[str, object]]:
    directory = safe_absolute(directory)
    tree = discover(directory)
    if tree.directories or set(tree.files) != {"manifest.json", *_FILES}:
        raise DataError("Evaluation inventory incomplete or contains unexpected entries")
    manifest = read_json(directory / "manifest.json")
    if (
        set(manifest) != {"schema_version", "plan_id", "created_at", "environment", "artifacts"}
        or manifest["schema_version"] != EVALUATION_VERSION
    ):
        raise DataError("Unsupported evaluation manifest schema")
    created = manifest["created_at"]
    try:
        if not isinstance(created, str):
            raise ValueError("timestamp type")
        offset = datetime.fromisoformat(created).utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("timestamp timezone")
    except ValueError as exc:
        raise DataError("Invalid evaluation creation timestamp") from exc
    _environment(manifest["environment"])
    records = manifest["artifacts"]
    if not isinstance(records, list) or len(records) != len(_FILES):
        raise DataError("Invalid evaluation artifact inventory")
    for name, value in zip(_FILES, records, strict=True):
        record = object_dict(value)
        if set(record) != {"relative_path", "size_bytes", "sha256"}:
            raise DataError("Invalid evaluation artifact fields")
        size, digest = record["size_bytes"], record["sha256"]
        if record["relative_path"] != name or type(size) is not int or not isinstance(digest, str):
            raise DataError("Invalid evaluation artifact types/order")
        RawFileRecord(name, size, digest)
        if stream_file(directory / name) != (size, digest):
            raise DataError(f"Evaluation checksum mismatch: {name}")
    plan = read_json(directory / "plan.json")
    plan_id = identity(plan)
    if manifest["plan_id"] != plan_id:
        raise DataError("Evaluation plan identity mismatch")
    if not staged and (
        directory.name != plan_id
        or directory.parent.name != dataset.lineage["canonical_id"]
        or directory.parent.parent.name != dataset.lineage["dataset_id"]
    ):
        raise DataError("Evaluation directory identity mismatch")
    if (
        stream_file(directory / "canonical-manifest.json")[1]
        != dataset.lineage["canonical_manifest_sha256"]
    ):
        raise DataError("Evaluation canonical lineage mismatch")
    split_obj = object_dict(plan.get("split"))
    config = config_from_dict(split_obj.get("config"))
    split, expected, fitted = build_plan(dataset, config)
    if not compatible_provenance(plan.get("code"), expected["code"], config.protocol.name):
        raise DataError("Unrecognized evaluation source/dependency provenance")
    # Compare all semantic fields against current reconstruction; preserve the exact
    # recognized historical source identity rather than rewriting its stored plan.
    expected["code"] = plan["code"]
    if canonical_json(plan) != canonical_json(expected):
        raise DataError(
            "Evaluation plan differs from canonical input, configuration "
            "or executing code/dependencies"
        )
    if canonical_json(read_json(directory / "fitted-transforms.json")) != canonical_json(fitted):
        raise DataError("Fitted parameters do not match training-only data")
    if discover(directory) != tree or read_json(directory / "manifest.json") != manifest:
        raise DataError("Evaluation artifacts changed during verification")
    return split, plan


def load_plan(
    directory: str | Path, *, canonical_dir: str | Path, raw_snapshot_dir: str | Path | None = None
) -> tuple[EvaluationDataset, SplitManifest, dict[str, object]]:
    dataset = open_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
    split, plan = _read_verified(Path(directory), dataset)
    return dataset, split, plan


def verify_plan(
    directory: str | Path, *, canonical_dir: str | Path, raw_snapshot_dir: str | Path | None = None
) -> PlanVerification:
    try:
        _, _, plan = load_plan(
            directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
        )
        return PlanVerification(True, (), identity(plan))
    except (DataError, OSError, ValueError, TypeError, KeyError) as exc:
        return PlanVerification(
            False,
            (
                str(exc)
                if isinstance(exc, DataError)
                else f"Cannot verify evaluation ({type(exc).__name__})",
            ),
        )


def publish_plan(
    canonical_dir: str | Path,
    config: PlanConfig,
    *,
    output_root: str | Path = "data/evaluation",
    raw_snapshot_dir: str | Path | None = None,
) -> EvaluationManifest:
    source, root = safe_absolute(canonical_dir), safe_absolute(output_root)
    dataset = open_dataset(source, raw_snapshot_dir=raw_snapshot_dir)
    for ancestor in (root, *root.parents):
        if (ancestor / "manifest.json").exists():
            raise DataError("Evaluation output cannot be nested inside an immutable artifact")
    canonical_root = source.parents[2]
    if root.is_relative_to(canonical_root) or canonical_root.is_relative_to(root):
        raise DataError("Evaluation output and managed canonical root must be disjoint")
    # Also exclude a sibling managed raw root in the conventional data layout.
    raw_root = canonical_root.parent / "raw"
    if root.is_relative_to(raw_root) or raw_root.is_relative_to(root):
        raise DataError("Evaluation output must not overlap managed raw storage")
    split, plan, fitted = build_plan(dataset, config)
    plan_id = identity(plan)
    namespace = safe_absolute(
        root / str(dataset.lineage["dataset_id"]) / str(dataset.lineage["canonical_id"])
    )
    namespace.mkdir(parents=True, exist_ok=True)
    destination = namespace / plan_id
    with _writer_lock(namespace):
        if os.path.lexists(destination):
            _read_verified(destination, dataset)
            return EvaluationManifest(plan_id, destination, split, True)
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=namespace))
        try:
            copy_file(source / "manifest.json", stage / "canonical-manifest.json")
            write_json(stage / "plan.json", plan)
            write_json(stage / "fitted-transforms.json", fitted)
            records = [asdict(RawFileRecord(name, *stream_file(stage / name))) for name in _FILES]
            write_json(
                stage / "manifest.json",
                {
                    "schema_version": EVALUATION_VERSION,
                    "plan_id": plan_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "environment": collect_environment_info().to_dict(),
                    "artifacts": records,
                },
            )
            _read_verified(stage, dataset, staged=True)
            if open_dataset(source, raw_snapshot_dir=raw_snapshot_dir).lineage != dataset.lineage:
                raise DataError("Canonical source changed during plan publication")
            _publish(stage, destination)
            return EvaluationManifest(plan_id, destination, split, False)
        finally:
            try:
                _cleanup_stage(stage, namespace)
            except OSError as exc:
                raise DataError(
                    "Evaluation staging cleanup failed; manual recovery required"
                ) from exc
