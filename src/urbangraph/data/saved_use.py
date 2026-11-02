"""Current source and canonical lineage gate for protected saved research runs."""

from __future__ import annotations

from pathlib import Path

from urbangraph.data import DataError
from urbangraph.data.rights import require_verified_snapshot_authorization
from urbangraph.evaluation.artifacts import load_plan
from urbangraph.evaluation.dataset import identity, open_governed_dataset


def authorized_saved_lineage(
    run_dir: str | Path,
    *,
    raw_snapshot_dir: str | Path | None,
    canonical_dir: str | Path | None,
    purpose: str,
    operation: str,
) -> dict[str, object]:
    """Authorize actual upstream bytes before any protected saved-run content."""
    if raw_snapshot_dir is None or canonical_dir is None:
        raise DataError("Complete current source context required for saved research use")
    require_verified_snapshot_authorization(raw_snapshot_dir, purpose, run_dir, operation=operation)
    # This independently checks the governed canonical artifact and its current
    # canonical-read decision. The returned lineage is not accepted from a run.
    return open_governed_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir).lineage


def authorized_saved_publication(
    output_root: str | Path,
    descriptor: dict[str, object],
    *,
    raw_snapshot_dir: str | Path | None,
    canonical_dir: str | Path | None,
    plan_dir: str | Path | None,
) -> None:
    """Verify current source, plan and distinct local publication authority before writes.

    A grant authorizes local publication, not redistribution or the truth of
    caller-provided scores. Trusted runner code must produce those scores.
    """
    if raw_snapshot_dir is None or canonical_dir is None or plan_dir is None:
        raise DataError("Complete current source and plan context required for saved publication")
    for purpose, operation in (
        ("model_training", "model_train"),
        ("scientific_evaluation", "model_evaluate"),
        ("model_training", "model_publish"),
    ):
        require_verified_snapshot_authorization(
            raw_snapshot_dir, purpose, output_root, operation=operation
        )
    dataset, _split, plan = load_plan(
        plan_dir, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    if descriptor.get("canonical") != dataset.lineage or descriptor.get("plan_id") != identity(
        plan
    ):
        raise DataError("Saved publication lineage or plan differs from verified source")
