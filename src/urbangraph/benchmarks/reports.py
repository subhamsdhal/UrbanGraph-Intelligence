"""Append-only JSON and Markdown evidence bundles; fresh readiness must be recomputed."""

import tempfile
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.datasets.serialization import read_json, write_json
from urbangraph.evaluation.dataset import identity


def markdown(report: dict[str, object]) -> str:
    import json

    label = str(report.get("evidence_kind", "unspecified evidence"))
    lines = [
        "# UrbanGraph benchmark evidence",
        "",
        f"Evidence: **{label}**",
        "",
        f"Dataset: {report.get('dataset_id', 'unknown')}",
        "",
        f"Status: {report.get('status', 'audit only; not certification')}",
        "",
        "This report does not establish authenticity or redistribution rights.",
        "",
    ]
    gates = report.get("gates")
    if isinstance(gates, dict):
        lines.extend(["| Gate | Status |", "| --- | --- |"])
        lines.extend(f"| {key} | {value} |" for key, value in gates.items())
    lines.extend(
        [
            "",
            "## Complete structured evidence",
            "",
            "```json",
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def publish_report(report: dict[str, object], root: str | Path) -> Path:
    """Content-address full report; do not overwrite or trust corrupted prior exports."""
    directory = safe_absolute(root)
    # Reports must not be injected into managed raw/canonical/evaluation bundles.
    if any((ancestor / "manifest.json").exists() for ancestor in (directory, *directory.parents)):
        raise DataError("Report root must be outside existing artifact bundles")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / identity(report)
    text = markdown(report)
    with _writer_lock(directory):
        if destination.exists():
            safe_absolute(destination)
            if (
                set(p.name for p in destination.iterdir()) != {"report.json", "report.md"}
                or read_json(destination / "report.json") != report
                or safe_absolute(destination / "report.md").read_text(encoding="utf-8") != text
            ):
                raise DataError("Existing report is corrupted; no replacement performed")
            return destination
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=directory))
        try:
            write_json(stage / "report.json", report)
            with (stage / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            _publish(stage, destination)
        finally:
            _cleanup_stage(stage, directory)
    return destination
