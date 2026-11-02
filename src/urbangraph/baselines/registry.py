"""Versioned model metadata; no optional service or training dependencies."""

from urbangraph.baselines.contracts import MODEL_IDS, MODEL_VERSION


def list_models(*, include_boosting: bool = False) -> list[dict[str, object]]:
    models: list[dict[str, object]] = [
        {
            "model": name,
            "version": MODEL_VERSION,
            "fit_required": name in {"historical-average", "autoregressive"},
            "prediction_shape": "[batch,horizon,sensors,1]",
            "historical_requirement": {"daily-seasonal": 288, "weekly-seasonal": 2016}.get(
                name, "configured history/lag count"
            ),
            "temporal_policies": ["positional", "timestamp-label"]
            if "seasonal" in name
            else ["verified plan"],
            "fitted_state_identity": "SHA-256 of parameters, support and training lineage",
            "reproducibility": "NumPy CPU; code/dependencies recorded; no random initialization",
        }
        for name in MODEL_IDS
    ]

    if include_boosting:
        models.extend(
            {
                "model": name,
                "version": "boosting-1.0",
                "fit_required": True,
                "optional_extra": "boosting",
                "prediction_shape": "[batch,horizon,sensors,1]",
            }
            for name in ("lightgbm", "xgboost")
        )
    return models
