"""Native CPU boosters, bounded fitting and stable feature/category schemas."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray

from urbangraph.baselines.contracts import FloatArray
from urbangraph.boosting.contracts import BoostConfig
from urbangraph.data import DataError


class FeatureMetadata(Protocol):
    """Serializable metadata; feature construction remains domain-specific."""

    def to_dict(self) -> dict[str, object]: ...


class BudgetExceeded(DataError):
    """A bounded experiment is incomplete, not an implicitly accepted shorter trial."""


class XGBBudget(xgb.callback.TrainingCallback):
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        if time.perf_counter() > self.deadline:
            raise BudgetExceeded("XGBoost trial wall-time budget exceeded")
        return False


@dataclass
class BoostModel:
    native: lgb.Booster | xgb.Booster
    config: BoostConfig
    features: FeatureMetadata
    names: tuple[str, ...]
    groups: tuple[str, ...]
    sensor_ids: tuple[str, ...]
    horizon: int
    best_iteration: int

    def predict(
        self, values: NDArray[np.float32], names: tuple[str, ...], sensor_ids: tuple[str, ...]
    ) -> FloatArray:
        if (
            values.ndim != 2
            or names != self.names
            or sensor_ids != self.sensor_ids
            or values.shape[1] != len(names)
        ):
            raise DataError("Model feature or sensor schema mismatch")
        if isinstance(self.native, lgb.Booster):
            output = self.native.predict(
                values, num_iteration=self.best_iteration, num_threads=self.config.threads
            )
        else:
            output = self.native.inplace_predict(values, iteration_range=(0, self.best_iteration))
        result = np.asarray(output, dtype=np.float64)
        if result.shape != (len(values),) or not np.isfinite(result).all():
            raise DataError("Boosting produced unavailable predictions; refusing selective scoring")
        return result

    def save(self, path: Path) -> None:
        if path.exists():
            raise DataError("Native model path already exists")
        if isinstance(self.native, lgb.Booster):
            self.native.save_model(str(path), num_iteration=self.best_iteration)
        else:
            self.native.save_model(path)

    def importance(self) -> dict[str, object]:
        if isinstance(self.native, lgb.Booster):
            values = np.asarray(self.native.feature_importance(importance_type="gain"), dtype=float)
        else:
            gains = self.native.get_score(importance_type="total_gain")
            values = np.array([float(np.asarray(gains.get(name, 0)).sum()) for name in self.names])
        grouped: dict[str, float] = {}
        for group, value in zip(self.groups, values, strict=True):
            grouped[group] = grouped.get(group, 0.0) + float(value)
        return {
            "method": "total split gain; descriptive, not causal",
            "features": dict(zip(self.names, values.tolist(), strict=True)),
            "groups": grouped,
        }


def load_model(
    path: Path,
    config: BoostConfig,
    features: FeatureMetadata,
    names: tuple[str, ...],
    groups: tuple[str, ...],
    sensors: tuple[str, ...],
    horizon: int,
    best_iteration: int,
) -> BoostModel:
    native: lgb.Booster | xgb.Booster
    if config.family == "lightgbm":
        native = lgb.Booster(model_file=str(path))
        if tuple(native.feature_name()) != names:
            raise DataError("Native LightGBM feature schema mismatch")
    else:
        native = xgb.Booster(params={"nthread": config.threads})
        native.load_model(path)
        if tuple(native.feature_names or []) != names:
            raise DataError("Native XGBoost feature schema mismatch")
    rounds = (
        native.current_iteration()
        if isinstance(native, lgb.Booster)
        else native.num_boosted_rounds()
    )
    if not 1 <= best_iteration <= rounds or len(names) != len(groups) or not sensors:
        raise DataError("Invalid native model iteration or schema")
    return BoostModel(native, config, features, names, groups, sensors, horizon, best_iteration)


def train_model(
    config: BoostConfig,
    features: FeatureMetadata,
    names: tuple[str, ...],
    groups: tuple[str, ...],
    sensors: tuple[str, ...],
    horizon: int,
    train_x: NDArray[np.float32],
    train_y: FloatArray,
    validation_x: NDArray[np.float32],
    validation_y: FloatArray,
    deadline: float,
) -> tuple[BoostModel, dict[str, object]]:
    if (
        len(train_x) != len(train_y)
        or len(validation_x) != len(validation_y)
        or not len(train_y)
        or not len(validation_y)
        or not np.isfinite(train_y).all()
        or not np.isfinite(validation_y).all()
    ):
        raise DataError("Finite, nonempty, aligned training and validation labels required")
    start = time.perf_counter()
    limit = min(deadline, start + config.trial_seconds)
    categorical = [i for i, name in enumerate(names) if name == "sensor_id"]
    native: lgb.Booster | xgb.Booster
    curves: dict[str, dict[str, list[float]]] = {}
    if config.family == "lightgbm":
        params = {
            "objective": "regression_l1",
            "metric": "l1",
            "device_type": "cpu",
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": config.threads,
            "seed": config.seed,
            "data_random_seed": config.seed,
            "verbosity": -1,
            "max_depth": config.max_depth,
            "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_child_samples,
            "learning_rate": config.learning_rate,
            "feature_fraction": config.colsample,
            "bagging_fraction": config.subsample,
            "bagging_freq": 1 if config.subsample < 1 else 0,
            "lambda_l1": config.reg_alpha,
            "lambda_l2": config.reg_lambda,
        }
        train = lgb.Dataset(
            train_x,
            label=train_y,
            feature_name=list(names),
            categorical_feature=categorical,
            free_raw_data=True,
        )
        validation = lgb.Dataset(
            validation_x,
            label=validation_y,
            reference=train,
            feature_name=list(names),
            categorical_feature=categorical,
        )

        def budget(env: lgb.callback.CallbackEnv) -> None:
            if time.perf_counter() > limit:
                raise BudgetExceeded("LightGBM trial wall-time budget exceeded")

        native = lgb.train(
            params,
            train,
            num_boost_round=config.trees,
            valid_sets=[validation],
            valid_names=["validation"],
            callbacks=[
                budget,
                lgb.record_evaluation(curves),
                lgb.early_stopping(config.early_stopping, verbose=False),
            ],
        )
        best = native.best_iteration or config.trees
    else:
        types = ["c" if name == "sensor_id" else "q" for name in names]
        xtrain = xgb.DMatrix(
            train_x,
            label=train_y,
            feature_names=list(names),
            feature_types=types,
            enable_categorical=True,
            nthread=config.threads,
        )
        xvalidation = xgb.DMatrix(
            validation_x,
            label=validation_y,
            feature_names=list(names),
            feature_types=types,
            enable_categorical=True,
            nthread=config.threads,
        )
        params = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "device": "cpu",
            "seed": config.seed,
            "nthread": config.threads,
            "max_depth": config.max_depth,
            "eta": config.learning_rate,
            "min_child_weight": config.min_child_weight,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample,
            "lambda": config.reg_lambda,
            "alpha": config.reg_alpha,
        }
        native = xgb.train(
            params,
            xtrain,
            num_boost_round=config.trees,
            evals=[(xvalidation, "validation")],
            evals_result=curves,
            verbose_eval=False,
            callbacks=[
                XGBBudget(limit),
                xgb.callback.EarlyStopping(rounds=config.early_stopping, save_best=True),
            ],
        )
        best = native.best_iteration + 1
    elapsed = time.perf_counter() - start
    if time.perf_counter() > limit:
        raise BudgetExceeded("Native training exceeded its wall-time budget")
    return BoostModel(native, config, features, names, groups, sensors, horizon, best), {
        "best_iteration": best,
        "learning_curves": curves,
        "training_seconds": elapsed,
        "objective": "absolute error",
        "parameters": params,
    }
