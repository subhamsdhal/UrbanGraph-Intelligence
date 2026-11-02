"""Comparable-population bridge to immutable statistical and boosting results."""

from pathlib import Path

from urbangraph.baselines.artifacts import verify_run as verify_baseline
from urbangraph.boosting.artifacts import verify_run as verify_boosting
from urbangraph.data import DataError
from urbangraph.data.saved_use import authorized_saved_lineage
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.neural.artifacts import verify_run


def compare_runs(
    paths: list[Path],
    *,
    raw_snapshot_dir: str | Path | None = None,
    canonical_dir: str | Path | None = None,
) -> dict[str, object]:
    if not paths:
        raise DataError("At least one run required")
    reference: dict[str, object] | None = None
    rows: list[dict[str, object]] = []
    for path in paths:
        authorized_saved_lineage(
            path,
            raw_snapshot_dir=raw_snapshot_dir,
            canonical_dir=canonical_dir,
            purpose="benchmark_comparison",
            operation="model_compare",
        )
        schema = read_json(path / "manifest.json").get("schema_version")
        if schema == "neural-1.0":
            checked = verify_run(
                path, raw_snapshot_dir=raw_snapshot_dir, canonical_dir=canonical_dir
            )
        elif schema == "boosting-1.0":
            checked = verify_boosting(
                path, raw_snapshot_dir=raw_snapshot_dir, canonical_dir=canonical_dir
            )
        else:
            checked = verify_baseline(
                path, raw_snapshot_dir=raw_snapshot_dir, canonical_dir=canonical_dir
            )
        run = object_dict(checked["descriptor"])
        policy = object_dict(run["metric_policy"])
        key = {
            name: run[name] for name in ("canonical", "plan_id", "population_sha256", "batch_size")
        }
        key["mask_zero"] = policy["mask_zero"]
        key["units"] = policy["units"]
        if reference is not None and reference != key:
            raise DataError("Incompatible benchmark populations or metric policies")
        reference = key
        score = object_dict(
            object_dict(read_json(path / "metrics.json")["groups"])["common/overall"]
        )
        rows.append(
            {
                "run_id": path.name,
                "integrity": checked["integrity"],
                "authorship_assurance": checked["authorship_assurance"],
                "scientific_validation": checked["scientific_validation"],
                "production_approval": checked["production_approval"],
                "family": run.get("family", run.get("model")),
                "role": run.get("role", "baseline"),
                "metrics": score,
            }
        )
    return {
        "comparison_contract": reference,
        "runs": rows,
        "scientific_endorsement": "not_established; caller-supplied scores may be included",
        "note": (
            "Identical verified target population; "
            "different recorded implementation versions allowed"
        ),
    }
