"""Compare governed STGNN runs on identical forecast populations."""

from pathlib import Path

from urbangraph.data import DataError
from urbangraph.data.rights import require_verified_snapshot_authorization
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.stgnn.artifacts import verify_run


def compare_runs(
    paths: list[Path], *, graph_context: dict[str, str] | None = None
) -> dict[str, object]:
    if not paths:
        raise DataError("At least one run required")
    reference: object = None
    rows: list[object] = []
    for path in paths:
        if graph_context is None or "raw_snapshot_dir" not in graph_context:
            raise DataError("Complete STGNN source context required")
        require_verified_snapshot_authorization(
            graph_context["raw_snapshot_dir"],
            "benchmark_comparison",
            path,
            operation="model_compare",
        )
        # Source reauthorization must precede even the saved manifest read.
        checked = verify_run(path, graph_context=graph_context)
        run = object_dict(checked["descriptor"])
        policy = object_dict(run["metric_policy"])
        key = {
            name: run[name] for name in ("canonical", "plan_id", "population_sha256", "batch_size")
        }
        key["mask_zero"], key["units"] = policy["mask_zero"], policy["units"]
        score = object_dict(read_json(path / "metrics.json")["groups"])["common/overall"]
        rows.append(
            {
                "run_id": path.name,
                "integrity": checked["integrity"],
                "authorship_assurance": checked["authorship_assurance"],
                "scientific_validation": checked["scientific_validation"],
                "production_approval": checked["production_approval"],
                "family": run["family"],
                "role": run["role"],
                "metrics": score,
            }
        )
        if reference is not None and reference != key:
            raise DataError("Incompatible benchmark populations or metric policies")
        reference = key
    return {
        "comparison_contract": reference,
        "runs": rows,
        "scientific_endorsement": "not_established; caller-supplied scores may be included",
    }
