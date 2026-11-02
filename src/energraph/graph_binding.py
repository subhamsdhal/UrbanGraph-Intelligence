"""Bind immutable EG-002 graphs to exact regional data and fold semantics."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from energraph.artifacts import BrokerPublicationContext
from energraph.energy_graphs import EnergyGraphSpec, statistical_edges, verify_energy_graph
from energraph.graph_authority import snapshot_path
from energraph.regional_data import load_regional_plan
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.datasets.serialization import object_dict
from urbangraph.evaluation.dataset import identity


def bound_graph(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    graph: Path | DatasetSnapshot,
    fold: int,
    condition: str,
    *,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    geography: Path | None = None,
    geography_reference: str | None = None,
    geometry_reference: str | None = None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    data, split, _ = load_regional_plan(
        canonical, plan, raw_snapshot_dir=raw_snapshot_dir, broker_context=broker_context
    )
    if type(fold) is not int or not 0 <= fold < len(split.folds):
        raise DataError("Invalid regional graph fold")
    adjacency, desc = verify_energy_graph(
        graph,
        canonical=canonical,
        plan=plan,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context,
        geography=geography,
        geography_reference=geography_reference,
        geometry_reference=geometry_reference,
        expected_order=data.sensor_ids,
        expected_plan=snapshot_path(plan).name,
    )
    spec = object_dict(desc["specification"])
    expected = {
        "temporal": "identity",
        "identity": "identity",
        "geographic": "geographic-proxy",
        "correlation": "statistical",
        "correlation-unweighted": "statistical",
        "random": "random-control",
    }
    if condition not in expected or spec["method"] != expected[condition]:
        raise DataError("Graph semantic type does not match condition")
    if (
        desc["canonical"] != data.lineage
        or desc["split_id"] != identity(split.to_dict())
        or desc["evaluation_first"] != str(data.timestamps[0])
        or desc["evaluation_last"] != str(data.timestamps[-1])
    ):
        raise DataError("Graph canonical/time applicability mismatch")
    if spec["method"] == "statistical":
        train = split.folds[fold].partition("train")
        source = object_dict(desc["source"])
        if desc["fold_id"] != split.folds[fold].fold_id or source["training_rows"] != [
            train.start,
            train.stop,
        ]:
            raise DataError("Statistical graph belongs to another training fold")
        parsed = EnergyGraphSpec(**spec)  # type: ignore[arg-type]
        fitted, evidence = statistical_edges(data.frame.iloc[train.start : train.stop], parsed)
        if (
            not np.array_equal(fitted, adjacency)
            or evidence["training_logical_sha256"] != source["training_logical_sha256"]
        ):
            raise DataError("Statistical graph training-only evidence mismatch")
    # Static geographic/identity controls are reusable across folds of this plan;
    # their original construction fold is retained, never relabeled.
    return adjacency, desc
