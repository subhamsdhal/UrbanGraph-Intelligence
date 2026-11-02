"""Timestamp-label diagnostics without localization, repair or physical outage claims."""

from __future__ import annotations

import numpy as np
import pandas as pd

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import POSITIONAL_PROTOCOL, SplitManifest
from urbangraph.evaluation.planning import plan_splits


def temporal_diagnostics(index: pd.DatetimeIndex, split: SplitManifest) -> dict[str, object]:
    """Count each affected sample once, including with multiple discontinuities.

    Per-edge examples are capped; aggregate counts always cover every interval.
    Differences on naive timestamps are label differences, not established elapsed time.
    """
    if plan_splits(index, split.config) != split:
        raise DataError("Temporal diagnostics require a split matching source timestamps")
    config = split.config
    ns = np.asarray(index.as_unit("ns").astype("int64"), dtype=np.int64)
    delta = np.diff(ns)
    step = config.window.interval_seconds * 10**9
    edges = np.flatnonzero(delta != step)
    prefix = np.concatenate(([0], np.cumsum(delta != step)))
    folds: list[dict[str, object]] = []
    for fold in split.folds:
        counts: dict[str, object] = {}
        for part in fold.partitions:
            affected = 0
            for run in part.samples:
                origins = np.arange(run.start, run.stop, dtype=np.int64)
                left = origins - config.features.observed_delay_steps - config.window.history + 1
                right = origins + config.window.horizon
                affected += int(np.count_nonzero(prefix[right] != prefix[left]))
            counts[part.name] = {
                "samples": part.sample_count,
                "crossing_discontinuity": affected,
                "cadence_admissible": part.sample_count - affected,
                "rejected_gap_samples": part.rejected_gap_samples,
            }
        folds.append({"fold_id": fold.fold_id, "partitions": counts})
    examples: list[dict[str, object]] = []
    for position in edges[:100]:
        edge = int(position)
        affected_ranges: list[dict[str, object]] = []
        # A sample spans edge iff left <= edge and right >= edge + 1.
        first = edge + 1 - config.window.horizon
        stop = edge + config.features.observed_delay_steps + config.window.history
        for fold in split.folds:
            for part in fold.partitions:
                for run in part.samples:
                    a, b = max(first, run.start), min(stop, run.stop)
                    if a < b:
                        affected_ranges.append(
                            {
                                "fold_id": fold.fold_id,
                                "partition": part.name,
                                "origin_start": a,
                                "origin_stop_exclusive": b,
                                "count": b - a,
                            }
                        )
        examples.append(
            {
                "after_row": edge,
                "before": index[edge].isoformat(),
                "after": index[edge + 1].isoformat(),
                "label_interval_seconds": float(delta[edge] / 1e9),
                "previous_interval_seconds": float(delta[edge - 1] / 1e9) if edge else None,
                "following_interval_seconds": float(delta[edge + 1] / 1e9)
                if edge + 1 < len(delta)
                else None,
                "skipped_naive_grid_labels": int(delta[edge] // step - 1)
                if delta[edge] > step and delta[edge] % step == 0
                else None,
                "affected_origin_ranges": affected_ranges,
            }
        )
    return {
        "diagnostic_version": "timestamp-labels-v1",
        "time_basis": "naive source labels; physical elapsed time unresolved"
        if index.tz is None
        else "source timezone-aware instants",
        "timezone": str(index.tz) if index.tz else None,
        "policy": "retain positional windows; disclose label discontinuities"
        if config.protocol.name == POSITIONAL_PROTOCOL
        else "cadence-checked windows",
        "nominal_interval_seconds": config.window.interval_seconds,
        "discontinuity_count": len(edges),
        "examples": examples,
        "examples_omitted": max(0, len(edges) - len(examples)),
        "folds": folds,
        "physical_missing_measurements": "not established by timestamp labels",
    }
