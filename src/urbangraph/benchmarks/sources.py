"""Versioned source expectations, independently supplied acquisition evidence."""

from dataclasses import asdict, dataclass
from datetime import date

from urbangraph.data import DataError
from urbangraph.datasets.registry import get_dataset_spec
from urbangraph.evaluation.contracts import REFERENCE_COMMIT

VERSION = "1.0"
UPSTREAM = f"https://github.com/liyaguang/DCRNN/blob/{REFERENCE_COMMIT}"
DOWNLOAD = "https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"


@dataclass(frozen=True)
class Acquisition:
    """Caller attestations, not authenticity or legal conclusions."""

    reference: str
    method: str
    evidence_kind: str
    usage_notes: str
    usage_reviewed: bool = False
    acquired_on: str | None = None
    schema_version: str = VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VERSION or self.evidence_kind not in {"real", "synthetic"}:
            raise DataError("Unsupported acquisition schema or evidence kind")
        if type(self.usage_reviewed) is not bool:
            raise DataError("Usage review must be boolean")
        for value in (self.reference, self.method, self.usage_notes):
            if not isinstance(value, str) or not value.strip():
                raise DataError("Acquisition reference, method and usage notes are required")
        if self.acquired_on is not None:
            try:
                if date.fromisoformat(self.acquired_on).isoformat() != self.acquired_on:
                    raise ValueError
            except (ValueError, TypeError) as exc:
                raise DataError("Acquisition date must be YYYY-MM-DD or null") from exc

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def source_spec(dataset_id: str) -> dict[str, object]:
    spec = get_dataset_spec(dataset_id)
    return {
        "version": VERSION,
        "dataset_id": dataset_id,
        "filename": spec.filename,
        "upstream": UPSTREAM + "/README.md",
        "download_reference": DOWNLOAD,
        "expected_sensor_count": 207 if dataset_id == "metr-la" else 325,
        "expectation_reference": UPSTREAM
        + "/data/model/"
        + ("dcrnn_la.yaml" if dataset_id == "metr-la" else "dcrnn_bay.yaml"),
        "usage_status": "UNRESOLVED: source-code MIT license is not dataset permission",
    }


def sensor_expectation(dataset_id: str, observed: int | None) -> dict[str, object]:
    spec = source_spec(dataset_id)
    return {
        "status": "UNKNOWN"
        if observed is None
        else ("PASS" if observed == spec["expected_sensor_count"] else "WARNING"),
        "observed": observed,
        "expected": spec["expected_sensor_count"],
        "reference": spec["expectation_reference"],
        "version": VERSION,
    }
