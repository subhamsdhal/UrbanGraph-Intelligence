"""Describe integrity-checked candidates on a common forecast-cell population."""

from pathlib import Path

from urbangraph.baselines.artifacts import verify_run
from urbangraph.data import DataError
from urbangraph.data.saved_use import authorized_saved_lineage
from urbangraph.datasets.serialization import object_dict, read_json


def compare_runs(
    directories: list[Path],
    *,
    raw_snapshot_dir: str | Path | None = None,
    canonical_dir: str | Path | None = None,
) -> dict[str, object]:
    if not directories:
        raise DataError("At least one result is required")
    rows: list[dict[str, object]] = []
    comparison: tuple[object, ...] | None = None
    for directory in directories:
        authorized_saved_lineage(
            directory,
            raw_snapshot_dir=raw_snapshot_dir,
            canonical_dir=canonical_dir,
            purpose="benchmark_comparison",
            operation="model_compare",
        )
        verified = verify_run(
            directory, raw_snapshot_dir=raw_snapshot_dir, canonical_dir=canonical_dir
        )
        descriptor = object_dict(verified["descriptor"])
        key = tuple(
            descriptor[name]
            for name in (
                "canonical",
                "plan_id",
                "protocol",
                "population_sha256",
                "metric_policy",
                "comparison_models",
                "code",
            )
        )
        if comparison is not None and comparison != key:
            raise DataError("Cannot rank incompatible protocols or comparison populations")
        comparison = key
        groups = object_dict(read_json(directory / "metrics.json")["groups"])
        overall = object_dict(groups["common/overall"])
        rows.append(
            {
                "model": descriptor["model"],
                "run_id": verified["run_id"],
                "integrity": verified["integrity"],
                "authorship_assurance": verified["authorship_assurance"],
                "scientific_validation": verified["scientific_validation"],
                "production_approval": verified["production_approval"],
                "protocol": descriptor["protocol"],
                "metrics": overall,
                "availability": read_json(directory / "availability.json"),
                "horizons": {k: v for k, v in groups.items() if k.startswith("common/horizon/")},
            }
        )

    def rank(row: dict[str, object]) -> float:
        value = object_dict(object_dict(row["metrics"])["mae"])["value"]
        return float(value) if isinstance(value, int | float) else float("inf")

    if len({str(r["model"]) for r in rows}) != len(rows):
        raise DataError("Duplicate model in comparison")
    rows.sort(key=rank)
    return {
        "schema_version": "1.0",
        "primary": "common-population MAE",
        "ranking": rows,
        "scientific_endorsement": "not_established; caller-supplied scores may be included",
    }
