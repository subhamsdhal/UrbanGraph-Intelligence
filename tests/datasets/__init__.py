from typing import Literal, Protocol

import pandas as pd

from urbangraph.data import DatasetSnapshot


class SnapshotFactory(Protocol):
    def __call__(
        self,
        frame: pd.DataFrame,
        dataset_id: str = "metr-la",
        *,
        key: str = "traffic",
        storage: Literal["fixed", "table"] = "fixed",
        other: pd.DataFrame | None = None,
    ) -> DatasetSnapshot: ...
