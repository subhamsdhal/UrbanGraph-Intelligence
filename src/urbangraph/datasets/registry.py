"""Small explicit format registry; no invented benchmark dimensions or geography."""

from dataclasses import dataclass

from urbangraph.data import METR_LA, PEMS_BAY, DataError, DatasetIdentity


@dataclass(frozen=True)
class DatasetSpec:
    identity: DatasetIdentity
    filename: str
    adapter: str = "pandas-hdf"


_REGISTRY = {
    spec.identity.dataset_id: spec
    for spec in (DatasetSpec(METR_LA, "metr-la.h5"), DatasetSpec(PEMS_BAY, "pems-bay.h5"))
}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return _REGISTRY[dataset_id]
    except KeyError as exc:
        raise DataError("No traffic parser registered for this dataset identity") from exc
