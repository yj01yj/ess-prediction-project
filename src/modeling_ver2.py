#
# 제작 목적: 모델링
# 제작 날짜: 2026-03-31
# 제작자: AI 3기 3반 박진
#


# 라이브러리
from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


# 하이퍼 파라미터
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_DATA_PATH = Path("outputs/feature_engineering/model_input.csv")
MODEL_SCORE_PATH = Path("outputs/modeling/model_scores.csv")
MODEL_PREDICTION_PATH = Path("outputs/modeling/model_predictions.csv")

TARGET_COLUMN = "target"
FEATURE_COLUMNS: list[str] | None = None

LINEAR_REGRESSION_PARAMS: dict[str, Any] = {}
RIDGE_REGRESSION_PARAMS: dict[str, Any] = {"alpha": 1.0}
ELASTIC_NET_PARAMS: dict[str, Any] = {
    "alpha": 0.1,
    "l1_ratio": 0.5,
    "max_iter": 10_000,
    "random_state": RANDOM_STATE,
}
RANDOM_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
XGBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}
LIGHTGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


# 데이터 호출
def load_modeling_data(path: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"모델링 입력 데이터가 없습니다: {path}. "
            "feature_engineering.py에서 생성한 파일 경로로 수정해 주세요."
        )

    return pd.read_csv(path)


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise KeyError(
            f"타깃 컬럼 '{target_column}'이 데이터에 없습니다. 실제 컬럼명으로 수정해 주세요."
        )

    selected_features = feature_columns or [
        column for column in df.columns if column != target_column
    ]
    missing_columns = [column for column in selected_features if column not in df.columns]
    if missing_columns:
        raise KeyError(f"입력 피처 컬럼이 데이터에 없습니다: {missing_columns}")

    x = df[selected_features].copy()
    y = df[target_column].copy()
    return x, y


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )


# 모델링
# 1. 선형 모델
def build_model_registry() -> dict[str, Any]:
    models: dict[str, Any] = {}

    # 1-1. 선형 회귀
    models["LinearRegression"] = LinearRegression(**LINEAR_REGRESSION_PARAMS)

    # 1-2. 릿지 회귀
    models["Ridge"] = Ridge(**RIDGE_REGRESSION_PARAMS)

    # 1-3. 엘라스틱 넷
    models["ElasticNet"] = ElasticNet(**ELASTIC_NET_PARAMS)

    # 2. 트리 모델
    # 2-1. 랜덤 포레스트
    models["RandomForest"] = RandomForestRegressor(**RANDOM_FOREST_PARAMS)

    # 2-2. XGBoost
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(**XGBOOST_PARAMS)
    else:
        warnings.warn(
            "xgboost가 설치되어 있지 않아 XGBoost 모델은 제외됩니다.",
            stacklevel=2,
        )

    # 2-3. LightGBM
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(**LIGHTGBM_PARAMS)
    else:
        warnings.warn(
            "lightgbm이 설치되어 있지 않아 LightGBM 모델은 제외됩니다.",
            stacklevel=2,
        )

    return models


# 모델 평가
# RMSE, R2, MAPE로 평가
def calculate_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    non_zero_mask = y_true_array != 0
    if not np.any(non_zero_mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (y_true_array[non_zero_mask] - y_pred_array[non_zero_mask])
                / y_true_array[non_zero_mask]
            )
        )
        * 100
    )


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)

    return {
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, model in build_model_registry().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = evaluate_regression(y_test, predictions)

        score_rows.append({"model_name": model_name, **metrics})
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model_name": model_name,
                    "actual": y_test.to_numpy(),
                    "predicted": predictions,
                    "residual": y_test.to_numpy() - predictions,
                }
            )
        )

    score_df = pd.DataFrame(score_rows).sort_values("rmse").reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    return score_df, prediction_df


def save_modeling_outputs(
    score_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    score_path: Path = MODEL_SCORE_PATH,
    prediction_path: Path = MODEL_PREDICTION_PATH,
) -> None:
    score_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    score_df.to_csv(score_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)


def run_modeling_pipeline(
    feature_data_path: Path = FEATURE_DATA_PATH,
    score_path: Path = MODEL_SCORE_PATH,
    prediction_path: Path = MODEL_PREDICTION_PATH,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | None = FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeling_df = load_modeling_data(feature_data_path)
    x, y = prepare_features_and_target(
        modeling_df,
        target_column=target_column,
        feature_columns=feature_columns,
    )
    x_train, x_test, y_train, y_test = split_dataset(x, y)

    score_df, prediction_df = train_and_evaluate_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )
    save_modeling_outputs(
        score_df=score_df,
        prediction_df=prediction_df,
        score_path=score_path,
        prediction_path=prediction_path,
    )
    return score_df, prediction_df


def main() -> None:
    score_df, _ = run_modeling_pipeline()

    print(score_df)


if __name__ == "__main__":
    main()
