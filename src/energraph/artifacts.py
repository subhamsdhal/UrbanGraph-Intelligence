"""Governed local derived bundles; external raw ingestion remains separate."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from energraph.broker_adapter import BrokerAdapter, PreparedDirectory
from energraph.broker_contract import BrokerRecord
from energraph.broker_tree import inspect_tree
from energraph.lineage import (
    FileParent,
    ParentGraph,
    _derived,
    _raw,
    _resolve_claims,
    authorize_sources,
    resolve_derived,
    resolve_parents,
)
from urbangraph.data import (
    DataError,
    DatasetSnapshot,
    DatasetSource,
    RawFileRecord,
    content_identity,
    verify_snapshot,
)
from urbangraph.data.contracts import BrokerArtifactReference
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _ingest_verified_source
from urbangraph.data.rights import (
    AuthorizationDecision,
    require_raw_authorization,
    require_snapshot_authorization,
)
from urbangraph.data.verification import read_manifest
from urbangraph.datasets.serialization import canonical_json, read_json, write_json

_GOVERNED = {
    "canonical": (
        "energy_canonicalize",
        "energy_canonical_read",
        "canonical.py",
        "canonicalize",
        "local_research",
    ),
    "multivariable": (
        "energy_multivariable",
        "energy_multivariable_read",
        "multivariable.py",
        "canonicalize",
        "local_research",
    ),
    "multivariable-projection": (
        "energy_multivariable_project",
        "energy_multivariable_projection_read",
        "multivariable.py",
        "canonicalize",
        "local_research",
    ),
    "plan": (
        "energy_plan_create",
        "energy_plan_read",
        "evaluation.py",
        "scientific_evaluation",
        "model_training",
    ),
    "run": (
        "energy_run_create",
        "energy_run_verify",
        "experiments.py",
        "model_training",
        "saved_run_verification",
    ),
    "target-plan": (
        "energy_target_plan_create",
        "energy_target_plan_read",
        "target_plans.py",
        "scientific_evaluation",
        "model_training",
    ),
    "energy-boost": (
        "energy_boost_train",
        "energy_boost_verify",
        "energy_boosting.py",
        "model_training",
        "saved_run_verification",
    ),
    "regional-canonical": (
        "energy_regional_canonicalize",
        "energy_regional_canonical_read",
        "regional_data.py",
        "canonicalize",
        "local_research",
    ),
    "regional-plan": (
        "energy_regional_plan_create",
        "energy_regional_plan_read",
        "regional_data.py",
        "scientific_evaluation",
        "model_training",
    ),
    "energy-graph": (
        "energy_graph_create",
        "energy_graph_read",
        "energy_graphs.py",
        "graph_analysis",
        "graph_analysis",
    ),
    "regional-run": (
        "energy_regional_train",
        "energy_regional_verify",
        "regional_research.py",
        "model_training",
        "saved_run_verification",
    ),
    "neural-regional": (
        "energy_neural_train",
        "energy_neural_verify",
        "graph_research.py",
        "model_training",
        "saved_run_verification",
    ),
    "uncertainty-population": (
        "energy_population_create",
        "energy_population_read",
        "uncertainty/population.py",
        "scientific_evaluation",
        "scientific_evaluation",
    ),
    "interval": (
        "energy_interval_create",
        "energy_interval_verify",
        "uncertainty/research.py",
        "scientific_evaluation",
        "scientific_evaluation",
    ),
}


BrokerArtifactRef = BrokerArtifactReference


@dataclass(frozen=True)
class PendingPublication:
    """Broker-pinned proposal; deliberately not usable as an artifact reference."""

    prepared: PreparedDirectory
    artifact_type: str
    trusted_parents: tuple[BrokerArtifactRef, ...]
    local_output_root: Path | None = None
    state: str = "AWAITING_APPROVAL"

    @property
    def publication_id(self) -> str:
        return self.prepared.artifact_id

    @property
    def content_sha256(self) -> str:
        return self.prepared.content_sha256


@dataclass(frozen=True)
class CommittedPublication:
    reference: BrokerArtifactRef
    record: BrokerRecord
    state: str = "COMMITTED"


class CommittedPublicationMaterializationError(DataError):
    """The broker commit succeeded, but its optional local presentation failed."""

    def __init__(self, committed: CommittedPublication) -> None:
        super().__init__("Broker artifact committed; local materialization is incomplete")
        self.committed = committed


@dataclass(frozen=True)
class BrokerPublicationContext:
    """Worker/reader context; it contains no policy-administrator capability."""

    broker: BrokerAdapter
    materialization_root: Path


def prepare_transactional_publication(
    broker: BrokerAdapter,
    *,
    kind: str,
    stage: Path,
    parents: tuple[BrokerArtifactRef, ...],
    local_output_root: Path | None = None,
) -> PendingPublication:
    """Stage actual bytes, then request broker prepare under current parent rights."""
    stage_id, staged_digest = broker.stage_directory(stage)
    prepared = prepare_broker_publication(broker, kind=kind, stage_id=stage_id, parents=parents)
    if prepared.content_sha256 != staged_digest:
        raise DataError("Broker prepared a different staged content identity")
    if prepared.parents != tuple(sorted((p.artifact_id, p.content_sha256) for p in parents)):
        raise DataError("Broker prepared a different parent closure")
    return PendingPublication(prepared, kind, parents, local_output_root)


def finalize_transactional_publication(
    broker: BrokerAdapter, pending: PendingPublication
) -> CommittedPublication:
    """Commit only a proposal that a separate broker policy actor already approved."""
    if pending.state != "AWAITING_APPROVAL" or pending.artifact_type not in _GOVERNED:
        raise DataError("Publication is not awaiting an eligible approval")
    reference = commit_broker_publication(broker, pending.prepared, kind=pending.artifact_type)
    if reference.content_sha256 != pending.content_sha256:
        raise DataError("Committed artifact differs from approved content")
    record = broker.authorize_directory(
        reference.artifact_id,
        purpose=_GOVERNED[pending.artifact_type][3],
        operation=_GOVERNED[pending.artifact_type][0],
    )
    expected = tuple(sorted((p.artifact_id, p.content_sha256) for p in pending.trusted_parents))
    if record.parents != expected or _broker_ref(record, pending.artifact_type) != reference:
        raise DataError("Committed broker lineage differs from the prepared proposal")
    return CommittedPublication(reference, record)


def finalize_governed_publication(
    context: BrokerPublicationContext, pending: PendingPublication
) -> DatasetSnapshot:
    """Commit the separately approved proposal, then optionally materialize it locally.

    Broker commit is the publication point. Local DatasetSnapshot creation happens
    strictly after that commit and is only a presentation/read cache. If it fails,
    the committed broker artifact remains authoritative and can be retrieved by ID.
    """
    if pending.local_output_root is None:
        raise DataError("Pending publication lacks its post-commit local output root")
    committed = finalize_transactional_publication(context.broker, pending)
    if committed.reference.content_sha256 != pending.content_sha256:
        raise DataError("Local candidate differs from the broker-committed content")
    try:
        materialized = context.broker.materialize_directory(
            committed.reference.artifact_id,
            context.materialization_root,
            purpose=_GOVERNED[pending.artifact_type][4],
            operation=_GOVERNED[pending.artifact_type][1],
        )
        tree = inspect_tree(materialized)
        records = tuple(
            RawFileRecord(relative, size, digest) for relative, size, digest in tree.files
        )
        for record in records:
            if stream_file(materialized / record.relative_path) != (
                record.size_bytes,
                record.sha256,
            ):
                raise DataError("Committed materialization changed after broker verification")
        candidate = _ingest_verified_source(
            materialized,
            "energraph-" + pending.artifact_type,
            raw_root=pending.local_output_root,
            source=DatasetSource(
                reference="Committed broker artifact " + committed.reference.artifact_id
            ),
            expected_digest=content_identity(records),
        )
    except Exception as exc:
        raise CommittedPublicationMaterializationError(committed) from exc
    else:
        if materialized.parent != safe_absolute(
            context.materialization_root
        ) or not materialized.name.startswith("broker-read-"):
            raise CommittedPublicationMaterializationError(committed)
        try:
            shutil.rmtree(materialized)
        except OSError as exc:
            raise CommittedPublicationMaterializationError(committed) from exc
    return replace(candidate, broker_reference=committed.reference)


def _broker_ref(record: BrokerRecord, kind: str) -> BrokerArtifactRef:
    if kind not in _GOVERNED or record.kind != "derived":
        raise DataError("Broker artifact type is not an EnerGraph derivation")
    if record.transformation != _GOVERNED[kind][0] or record.snapshot is None:
        raise DataError("Broker transformation or directory inventory differs")
    return BrokerArtifactRef(record.artifact_id, record.content_sha256, record.revision, kind)


def resolve_broker_reference(
    broker: BrokerAdapter, artifact_id: str, *, purpose: str, operation: str
) -> BrokerArtifactRef:
    """Resolve a caller's ID through current broker rights before trusting its metadata."""
    record = broker.authorize_directory(artifact_id, purpose=purpose, operation=operation)
    if record.kind == "raw":
        return BrokerArtifactRef(record.artifact_id, record.content_sha256, record.revision, "raw")
    kinds = [kind for kind, item in _GOVERNED.items() if item[0] == record.transformation]
    if len(kinds) != 1:
        raise DataError("Unknown broker EnerGraph transformation")
    return _broker_ref(record, kinds[0])


def broker_parent_references(
    broker: BrokerAdapter,
    reference: BrokerArtifactRef,
    *,
    purpose: str,
    operation: str,
) -> tuple[BrokerArtifactRef, ...]:
    """Resolve direct parents only from the authorized broker record."""
    record = broker.authorize_directory(reference.artifact_id, purpose=purpose, operation=operation)
    if reference.artifact_type == "raw":
        if record.kind != "raw" or record.content_sha256 != reference.content_sha256:
            raise DataError("Broker raw identity changed")
        return ()
    if _broker_ref(record, reference.artifact_type) != reference:
        raise DataError("Broker derived identity changed")
    parents = tuple(
        resolve_broker_reference(broker, artifact_id, purpose=purpose, operation=operation)
        for artifact_id, digest in record.parents
        if digest
    )
    if len(parents) != len(record.parents) or tuple(
        sorted((parent.artifact_id, parent.content_sha256) for parent in parents)
    ) != tuple(sorted(record.parents)):
        raise DataError("Broker direct parent identities changed")
    return parents


def prepare_broker_publication(
    broker: BrokerAdapter,
    *,
    kind: str,
    stage_id: str,
    parents: tuple[BrokerArtifactRef, ...],
) -> PreparedDirectory:
    """Prepare an existing broker staging tree; approval is a separate role."""
    if kind not in _GOVERNED or not parents:
        raise DataError("Unknown EnerGraph kind or missing broker parents")
    operation, _, _, purpose, _ = _GOVERNED[kind]
    if len({item.artifact_id for item in parents}) != len(parents):
        raise DataError("Duplicate broker parent identity")
    for parent in parents:
        current = broker.authorize_directory(
            parent.artifact_id, purpose=purpose, operation=operation
        )
        if (
            current.content_sha256 != parent.content_sha256
            or current.revision != parent.revision
            or current.kind != ("raw" if parent.artifact_type == "raw" else "derived")
        ):
            raise DataError("Broker parent identity changed")
        if parent.artifact_type != "raw" and _broker_ref(current, parent.artifact_type) != parent:
            raise DataError("Broker derived parent identity changed")
    return broker.prepare_directory(
        stage_id=stage_id,
        parent_ids=tuple(item.artifact_id for item in parents),
        transformation=operation,
        purpose=purpose,
        operation=operation,
    )


def commit_broker_publication(
    broker: BrokerAdapter, prepared: PreparedDirectory, *, kind: str
) -> BrokerArtifactRef:
    """Commit only after a separately authenticated approver pinned the bytes."""
    if kind not in _GOVERNED:
        raise DataError("Unknown EnerGraph kind")
    operation, _, _, purpose, _ = _GOVERNED[kind]
    if prepared.purpose != purpose or prepared.operation != operation:
        raise DataError("Prepared publication context differs from EnerGraph kind")
    return _broker_ref(broker.commit_directory(prepared), kind)


def publish(stage: Path, root: Path, kind: str) -> DatasetSnapshot:
    """Legacy stage-only entry point cannot establish upstream provenance."""
    raise DataError("Governed derived publication requires verified upstream source context")


def require_committed_snapshot(
    result: DatasetSnapshot | PendingPublication,
) -> DatasetSnapshot:
    """Prevent completed-only legacy callers from treating a proposal as committed."""
    if not isinstance(result, DatasetSnapshot):
        raise DataError("Publication is pending separate approval and finalization")
    return result


def _authorize_parent_selectors(
    selected: tuple[Path | FileParent, ...],
    *,
    purpose: str,
    operation: str,
    destination: Path,
) -> None:
    """Authorize raw parent selectors before reading any parent metadata or bytes.

    Derived snapshot parents are deliberately unsupported until an independently
    protected closure authority exists. The path component check is only an early
    rejection; it is not treated as proof of source identity or permission.
    """
    if not selected:
        raise DataError("Empty upstream parent set")
    for parent in selected:
        path = safe_absolute(parent.path if isinstance(parent, FileParent) else parent)
        if any(component.startswith("energraph-") for component in path.parts):
            raise DataError("Derived parent closure authority unavailable")
        if isinstance(parent, FileParent):
            expected = require_raw_authorization(
                parent.source_id, path, purpose, destination, operation=operation
            )
            # The exact file digest is compared only after the matching current
            # source/path/purpose/operation grant authorizes opening it.
            if stream_file(path)[1] != expected:
                raise DataError("Raw file differs from current source authorization")
        else:
            expected = require_snapshot_authorization(
                path, purpose, destination, operation=operation
            )
            # Inspect only the bounded manifest after authorization. Refuse the
            # derived namespace before the verifier can enumerate its payload.
            manifest = read_manifest(path / "manifest.json")
            if manifest.dataset_id.startswith("energraph-"):
                raise DataError("Derived parent closure authority unavailable")
            checked = verify_snapshot(path)
            if (
                not checked.valid
                or checked.manifest is None
                or checked.manifest.snapshot_id != expected
            ):
                raise DataError("Raw parent differs from current source authorization")


def _publish_governed(
    stage: Path,
    root: Path,
    kind: str,
    *,
    raw_snapshot_dir: Path | None = None,
    parents: tuple[Path | FileParent, ...] | None = None,
    broker_context: BrokerPublicationContext | None = None,
    broker_parents: tuple[BrokerArtifactRef, ...] | None = None,
) -> DatasetSnapshot | PendingPublication:
    """Publish a trusted local transformation after distinct create/commit decisions.

    The caller's raw path is only a selector. Its contents and both decisions are
    independently checked. Trusted application code must create the stage from that
    source; this function does not authenticate arbitrary same-user Python callers.
    """
    if kind not in _GOVERNED:
        raise DataError("Unsupported governed EnerGraph transformation")
    transform_operation, _, module, purpose, _ = _GOVERNED[kind]
    root = safe_absolute(root)
    if (raw_snapshot_dir is None) == (parents is None):
        raise DataError("Exactly one explicit parent context is required")
    if (broker_context is None) != (broker_parents is None):
        raise DataError("Broker publication requires explicit context and parent identities")
    if broker_context is not None and (broker_parents is None or not broker_parents):
        raise DataError("Broker publication requires committed parent identities")
    selected = parents if parents is not None else (raw_snapshot_dir,)
    if selected is None or any(parent is None for parent in selected):
        raise DataError("Incomplete derived parent set")
    selected = tuple(
        FileParent(safe_absolute(parent.path), parent.source_id)
        if isinstance(parent, FileParent)
        else safe_absolute(parent)
        for parent in selected
        if parent is not None
    )
    if any((p / "manifest.json").exists() for p in (root, *root.parents)):
        raise DataError("Cannot publish inside an existing immutable snapshot")

    def resolve_broker_parent_graph() -> ParentGraph:
        assert broker_context is not None and broker_parents is not None
        if len(broker_parents) != len(selected) or any(
            isinstance(parent, FileParent) for parent in selected
        ):
            raise DataError("Broker publication requires one directory reference per parent")
        claims: list[dict[str, object]] = []
        broker_sources: dict[str, str] = {}
        for reference, parent in zip(broker_parents, selected, strict=True):
            record = broker_context.broker.authorize_directory(
                reference.artifact_id,
                purpose=purpose,
                operation=transform_operation,
            )
            if (
                record.content_sha256 != reference.content_sha256
                or record.revision != reference.revision
                or record.kind != ("raw" if reference.artifact_type == "raw" else "derived")
            ):
                raise DataError("Broker parent reference no longer identifies a committed artifact")
            if reference.artifact_type == "raw":
                if record.transformation != "raw-admission":
                    raise DataError("Broker raw parent has an unexpected transformation")
            elif _broker_ref(record, reference.artifact_type) != reference:
                raise DataError("Broker derived parent reference changed")
            for source_id, digest in record.sources:
                if source_id in broker_sources and broker_sources[source_id] != digest:
                    raise DataError("Conflicting broker source ancestry")
                broker_sources[source_id] = digest
            parent_path = cast(Path, parent)
            observed = inspect_tree(
                parent_path if reference.artifact_type == "raw" else parent_path / "payload"
            )
            if observed.digest != reference.content_sha256:
                raise DataError("Local parent presentation differs from broker identity")
            if reference.artifact_type == "raw":
                claim, _ = _raw(parent_path)
            else:
                claim, _ = _derived(parent_path, frozenset())
            claims.append(claim)

        def claim_sort_key(item: object) -> tuple[str, str]:
            claim = cast(dict[str, object], item)
            return str(claim["type"]), str(claim["path"])

        graph = _resolve_claims(sorted(claims, key=claim_sort_key), frozenset())
        local_sources = {
            str(item["source_id"]): str(item["content_sha256"]) for item in graph.sources
        }
        if set(local_sources) != set(broker_sources):
            raise DataError("Local descriptive lineage differs from broker source closure")
        return graph

    decisions: tuple[AuthorizationDecision, ...]
    if broker_context is not None:
        # The broker check above is the protected-read gate. Local ancestry is
        # parsed only afterward to retain the existing descriptive lineage
        # contract for trusted post-commit DatasetSnapshot presentations.
        graph = resolve_broker_parent_graph()
        create = authorize_sources(
            graph.sources,
            purpose=purpose,
            operation=transform_operation,
            destination=root,
        )
        commit = authorize_sources(
            graph.sources,
            purpose="local_research",
            operation="derived_commit",
            destination=root,
        )
        decisions = create + commit
    else:
        _authorize_parent_selectors(
            selected, purpose=purpose, operation=transform_operation, destination=root
        )
        _authorize_parent_selectors(
            selected, purpose="local_research", operation="derived_commit", destination=root
        )
        graph = resolve_parents(selected)
        create = authorize_sources(
            graph.sources, purpose=purpose, operation=transform_operation, destination=root
        )
        commit = authorize_sources(
            graph.sources, purpose="local_research", operation="derived_commit", destination=root
        )
        decisions = create + commit
        direct_parents = list(graph.direct)
        source_closure = list(graph.sources)
    if broker_context is not None:
        direct_parents = list(graph.direct)
        source_closure = list(graph.sources)
    stage = safe_absolute(stage)
    tree = discover(stage)
    if not tree.files or "governance.json" in tree.files:
        raise DataError("Invalid or caller-supplied derived governance inventory")
    descriptor = read_json(stage / "descriptor.json")
    if not isinstance(descriptor, dict):
        raise DataError("Invalid derived descriptor")
    if kind == "neural-regional":
        trials = descriptor.get("trials")
        if not isinstance(trials, list) or not trials:
            raise DataError("Missing neural trial population")
        expected_dirs = {f"trial-{i}" for i in range(len(trials))}
        expected_files = {
            "descriptor.json",
            "runtime.json",
            "forecasts.npz",
            "graph.npz",
            "probe.npz",
        } | {
            f"trial-{i}/{name}"
            for i in range(len(trials))
            for name in ("checkpoint.json", "state.pt")
        }
        if set(tree.directories) != expected_dirs or set(tree.files) != expected_files:
            raise DataError("Unexpected neural checkpoint tree")
    elif tree.directories:
        raise DataError("Unexpected derived payload directories")
    transform_source = Path(__file__).parent / module
    output_files = tuple(RawFileRecord(name, *stream_file(stage / name)) for name in tree.files)
    record = {
        "schema": "energraph-governed-derived",
        "version": 2,
        "artifact_type": kind,
        "direct_parents": direct_parents,
        "source_closure": source_closure,
        "transformation": {
            "name": transform_operation,
            "version": 1,
            "source_path": str(transform_source),
            "source_sha256": stream_file(transform_source)[1],
            "configuration_sha256": hashlib.sha256(canonical_json(descriptor)).hexdigest(),
        },
        "purpose": purpose,
        "commit_operation": "derived_commit",
        "destination_root": str(root),
        "output_payload_sha256": content_identity(output_files),
        "publication_state": "prepared-for-atomic-rename",
        "policy_evidence": [d.evidence_reference for d in decisions],
        "rights_expires_utc": (
            min(d.expires_utc for d in decisions).isoformat() if decisions else None
        ),
        "retention_days": min(d.retention_days for d in decisions) if decisions else None,
    }
    write_json(stage / "governance.json", record)
    # The internal copying primitive performs controlled staging under the final
    # root, stability checks, checksum verification and a no-overwrite commit.
    files = tuple(RawFileRecord(name, *stream_file(stage / name)) for name in discover(stage).files)

    def recheck_local(*, broker_backed: bool) -> None:
        if not broker_backed:
            _authorize_parent_selectors(
                selected, purpose=purpose, operation=transform_operation, destination=root
            )
            _authorize_parent_selectors(
                selected, purpose="local_research", operation="derived_commit", destination=root
            )
        current_graph = (
            resolve_broker_parent_graph() if broker_backed else resolve_parents(selected)
        )
        current = authorize_sources(
            current_graph.sources,
            purpose=purpose,
            operation=transform_operation,
            destination=root,
        ) + authorize_sources(
            current_graph.sources,
            purpose="local_research",
            operation="derived_commit",
            destination=root,
        )
        if current_graph != graph or current != decisions:
            raise DataError("Derived rights decision changed before publication")

    if broker_context is not None and broker_parents is not None:
        # Local data staging is untrusted and disposable. The broker pins these
        # exact bytes and parent identities before any trusted local output exists.
        recheck_local(broker_backed=True)
        return prepare_transactional_publication(
            broker_context.broker,
            kind=kind,
            stage=stage,
            parents=broker_parents,
            local_output_root=root,
        )

    def recheck() -> None:
        recheck_local(broker_backed=False)

    result = _ingest_verified_source(
        stage,
        "energraph-" + kind,
        raw_root=root,
        source=DatasetSource(reference="EnerGraph governed local derived artifact"),
        expected_digest=content_identity(files),
        before_publish=recheck,
    )
    return result


def governed_payload(
    directory: Path | BrokerArtifactRef,
    expected_files: set[str],
    *,
    raw_snapshot_dir: Path | None = None,
    parents: tuple[Path | FileParent, ...] | None = None,
    kind: str,
    broker: BrokerAdapter | None = None,
    broker_parents: tuple[BrokerArtifactRef, ...] | None = None,
    materialization_root: Path | None = None,
) -> Path:
    """Authorize current source use before releasing protected derived payload."""
    if kind not in _GOVERNED:
        raise DataError("Unsupported governed EnerGraph artifact")
    _, read_operation, _, purpose, read_purpose = _GOVERNED[kind]
    if isinstance(directory, BrokerArtifactRef):
        if (
            broker is None
            or broker_parents is None
            or materialization_root is None
            or raw_snapshot_dir is not None
            or parents is not None
            or directory.artifact_type != kind
        ):
            raise DataError("Complete broker context required before protected read")
        broker_record = broker.authorize_directory(
            directory.artifact_id, purpose=read_purpose, operation=read_operation
        )
        if _broker_ref(broker_record, kind) != directory:
            raise DataError("Broker artifact identity changed")
        expected_parents = tuple(
            sorted((item.artifact_id, item.content_sha256) for item in broker_parents)
        )
        if broker_record.parents != expected_parents:
            raise DataError("Broker parent set differs from supplied references")
        snapshot = broker_record.snapshot
        if snapshot is None:
            raise DataError("Broker artifact inventory unavailable")
        inventory_files = cast(list[list[object]], snapshot["files"])
        if {item[0] for item in inventory_files} != expected_files | {"governance.json"}:
            raise DataError("Broker artifact inventory differs from scientific contract")
        return broker.materialize_directory(
            directory.artifact_id,
            materialization_root,
            purpose=read_purpose,
            operation=read_operation,
        )
    if broker is not None or broker_parents is not None or materialization_root is not None:
        raise DataError("Broker authority requires a broker-issued artifact reference")
    if (raw_snapshot_dir is None) == (parents is None):
        raise DataError("Exactly one explicit parent context is required")
    selected = parents if parents is not None else (raw_snapshot_dir,)
    if selected is None or any(parent is None for parent in selected):
        raise DataError("Incomplete derived parent set")
    directory = safe_absolute(directory)
    normalized = tuple(
        FileParent(safe_absolute(parent.path), parent.source_id)
        if isinstance(parent, FileParent)
        else safe_absolute(parent)
        for parent in selected
        if parent is not None
    )
    _authorize_parent_selectors(
        normalized, purpose=read_purpose, operation=read_operation, destination=directory
    )
    expected = resolve_parents(normalized)
    authorize_sources(
        expected.sources,
        purpose=read_purpose,
        operation=read_operation,
        destination=directory,
    )
    graph = resolve_derived(directory)
    if graph.sources != expected.sources:
        raise DataError("Derived source closure differs from supplied parent set")
    base = _payload_unchecked(directory, expected_files | {"governance.json"})
    record = read_json(base / "governance.json")
    if (
        record["artifact_type"] != kind
        or record["purpose"] != purpose
        or record["direct_parents"] != list(expected.direct)
    ):
        raise DataError("Derived governance lineage or transformation mismatch")
    return base


def payload(directory: Path, expected: set[str]) -> Path:
    """Legacy context-free payload access is closed for protected derived data."""
    raise DataError("Current governed source context required for derived payload")


def _payload_unchecked(directory: Path, expected: set[str]) -> Path:
    result = verify_snapshot(directory)
    if not result.valid or result.manifest is None:
        raise DataError("Invalid energy snapshot integrity")
    if {f.relative_path for f in result.manifest.files} != expected:
        raise DataError("Unexpected energy bundle inventory")
    return directory / "payload"


def verify_bundle(directory: Path) -> dict[str, object]:
    """Legacy context-free verification cannot release protected metadata."""
    raise DataError("Current governed source context required for energy bundle verification")
