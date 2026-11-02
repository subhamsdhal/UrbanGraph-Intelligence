"""Broker identity/rights gate shared by regional graph producers and consumers."""

from __future__ import annotations

from pathlib import Path

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    broker_parent_references,
    resolve_broker_reference,
)
from energraph.broker_tree import inspect_tree
from energraph.lineage import resolve_parents
from urbangraph.data import DataError, DatasetSnapshot, verify_snapshot


def snapshot_path(value: Path | DatasetSnapshot) -> Path:
    return value.directory if isinstance(value, DatasetSnapshot) else value


def authorize_graph_inputs(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    *,
    raw_snapshot_dir: Path,
    broker_context: BrokerPublicationContext | None,
    purpose: str,
    operation: str,
    graph: Path | DatasetSnapshot | None = None,
    geographic_parents: tuple[tuple[Path | DatasetSnapshot, BrokerArtifactRef], ...] = (),
) -> tuple[BrokerArtifactRef, ...] | None:
    """Authorize *before* inspecting local derived cache metadata or payloads.

    A path remains a scientific input only; its directory naming is not identity.
    """
    base_inputs: tuple[Path | DatasetSnapshot, ...] = (canonical, plan)
    base_refs = tuple(
        value.broker_reference if isinstance(value, DatasetSnapshot) else None
        for value in base_inputs
    )
    geo_inputs = tuple(item[0] for item in geographic_parents)
    geo_refs = tuple(item[1] for item in geographic_parents)
    graph_ref = graph.broker_reference if isinstance(graph, DatasetSnapshot) else None
    inputs = base_inputs + geo_inputs + (() if graph is None else (graph,))
    refs = base_refs + geo_refs + (() if graph is None else (graph_ref,))
    brokered = broker_context is not None or any(ref is not None for ref in refs)
    if not brokered:
        closure = resolve_parents(tuple(snapshot_path(value) for value in inputs))
        raw_sources = resolve_parents((raw_snapshot_dir,)).sources
        if any(source not in closure.sources for source in raw_sources):
            raise DataError("Regional graph raw source closure mismatch")
        return None
    if broker_context is None or any(ref is None for ref in refs):
        raise DataError("Complete committed graph parent references and broker context required")
    trusted = tuple(ref for ref in refs if ref is not None)
    expected_types = (
        ("regional-canonical", "regional-plan")
        + ("raw",) * len(geographic_parents)
        + (() if graph is None else ("energy-graph",))
    )
    if tuple(ref.artifact_type for ref in trusted) != expected_types:
        raise DataError("Incorrect regional graph parent artifact types")
    broker = broker_context.broker
    records = []
    for _value, reference in zip(inputs, trusted, strict=True):
        resolved = resolve_broker_reference(
            broker, reference.artifact_id, purpose=purpose, operation=operation
        )
        if resolved != reference:
            raise DataError("Regional graph broker identity changed")
        record = broker.authorize_directory(
            reference.artifact_id, purpose=purpose, operation=operation
        )
        if (record.content_sha256, record.revision) != (
            reference.content_sha256,
            reference.revision,
        ):
            raise DataError("Regional graph broker revision or digest changed")
        records.append(record)
    # All current broker decisions complete before inspecting any caller-owned cache.
    for index, (value, reference) in enumerate(zip(inputs, trusted, strict=True)):
        path = snapshot_path(value)
        record = records[index]
        if reference.artifact_type == "raw":
            checked = verify_snapshot(path)
            if (
                not checked.valid
                or checked.manifest is None
                or len(record.sources) != 1
                or checked.manifest.dataset_id != record.sources[0][0]
                or record.sources[0][1] != reference.content_sha256
                or inspect_tree(path).digest != reference.content_sha256
            ):
                raise DataError("Geographic raw cache differs from committed broker identity")
        elif inspect_tree(path / "payload").digest != reference.content_sha256:
            raise DataError("Regional graph cache differs from committed broker artifact")
    raw_parents = broker_parent_references(broker, trusted[0], purpose=purpose, operation=operation)
    if len(raw_parents) != 1 or raw_parents[0].artifact_type != "raw":
        raise DataError("Regional canonical requires one committed raw ancestor")
    if broker_parent_references(broker, trusted[1], purpose=purpose, operation=operation) != (
        trusted[0],
    ):
        raise DataError("Regional plan broker parent differs from canonical")
    for reference in trusted[2 : 2 + len(geographic_parents)]:
        if broker_parent_references(broker, reference, purpose=purpose, operation=operation):
            raise DataError("Geographic raw admission must not have derived parents")
    if graph is not None:
        graph_parents = broker_parent_references(
            broker, trusted[-1], purpose=purpose, operation=operation
        )
        if set(graph_parents) != set(trusted[:-1]):
            raise DataError("Energy graph broker parents differ from its authorized inputs")
    if inspect_tree(raw_snapshot_dir).digest != raw_parents[0].content_sha256:
        raise DataError("Regional raw cache differs from broker-authorized ancestor")
    # Preserve the local descriptive lineage invariant after the broker's
    # authorization and the committed cache hash checks.
    local = resolve_parents((raw_snapshot_dir,)).sources
    if len(local) != 1 or local[0].get("type") != "raw":
        raise DataError("Local regional raw lineage is invalid")
    return trusted
