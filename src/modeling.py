#
# 제작 목적: 모델링
# 제작 날짜: 2026-03-31
# 제작자: AI 3기 3반 박진
#


# 라이브러리
from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
VALID_SIZE = 0.2
CV_SPLITS = 5

FEATURE_DATA_PATH = Path("data/model_input.csv")
MODEL_SCORE_PATH = Path("outputs/model_outputs/model_scores.csv")
MODEL_PREDICTION_PATH = Path("outputs/model_outputs/model_predictions.csv")
BEST_MODEL_PATH = Path("outputs/model_outputs/best_model.pkl")


TARGET_COLUMN_CANDIDATES = ["cycle_life", "target"]
ID_COLUMNS = ["cell_id"]
BATCH_COLUMN = "batch"
GROUP_COLUMN = "charging_policy"
EXCLUDE_COLUMNS = {"split", "fold"}

TRAIN_BATCH_KEYWORDS = ("batch 1", "batch1", "2017-05-12")
TEST_BATCH_KEYWORDS = ("batch 2", "batch2", "2018-02-20")
ADDITIONAL_TEST_BATCH_KEYWORDS = ("batch 3", "batch3", "2018-04-12")

LINEAR_REGRESSION_PARAMS: dict[str, Any] = {}
RIDGE_REGRESSION_PARAMS: dict[str, Any] = {"alpha": 1.0}
ELASTIC_NET_PARAMS: dict[str, Any] = {
    "alpha": 0.03,
    "l1_ratio": 0.2,
    "max_iter": 10_000,
    "random_state": RANDOM_STATE,
}
RANDOM_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 8,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
XGBOOST_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 3,
    "min_child_weight": 4,
    "subsample": 0.75,
    "colsample_bytree": 0.70,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 8,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}


# 데이터 호출
def load_modeling_data(path: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"모델링 입력 데이터가 없습니다: {path}. "
            "feature_engineering.py에서 생성한 파일 경로로 수정해 주세요."
        )

    return pd.read_csv(path)


# 데이터 검증
def resolve_target_column(df: pd.DataFrame) -> str:
    for column in TARGET_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise KeyError(
        "타깃 컬럼을 찾을 수 없습니다. "
        f"후보 컬럼: {TARGET_COLUMN_CANDIDATES}"
    )


def validate_modeling_dataframe(df: pd.DataFrame, target_column: str) -> None:
    required_columns = {target_column, BATCH_COLUMN}
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "모델링 데이터에 필수 컬럼이 없습니다: "
            f"{missing_columns}. batch 기반 분할을 위해 필요한 컬럼을 확인해 주세요."
        )

    if df.empty:
        raise ValueError("모델링 데이터가 비어 있습니다.")

    if df[target_column].isna().any():
        raise ValueError("타깃 컬럼에 결측치가 있습니다. 전처리 단계에서 제거 또는 보정해 주세요.")


# 피처/타깃 준비
def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    excluded_columns = set(ID_COLUMNS) | {target_column, BATCH_COLUMN} | EXCLUDE_COLUMNS
    selected_features = [column for column in df.columns if column not in excluded_columns]

    if not selected_features:
        raise ValueError("사용 가능한 피처 컬럼이 없습니다. EXCLUDE_COLUMNS 설정을 확인해 주세요.")

    x = df[selected_features].copy()
    y = df[target_column].copy()
    return x, y, selected_features


# 배치 분리
def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def build_batch_mask(series: pd.Series, keywords: tuple[str, ...]) -> pd.Series:
    normalized = series.astype(str).map(normalize_text)
    return normalized.apply(lambda text: any(keyword in text for keyword in keywords))


def split_batch_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_source_df = df[build_batch_mask(df[BATCH_COLUMN], TRAIN_BATCH_KEYWORDS)].copy()
    test_df = df[build_batch_mask(df[BATCH_COLUMN], TEST_BATCH_KEYWORDS)].copy()
    additional_test_df = df[build_batch_mask(df[BATCH_COLUMN], ADDITIONAL_TEST_BATCH_KEYWORDS)].copy()

    if train_source_df.empty:
        raise ValueError(
            "학습용 Batch 1 데이터가 비어 있습니다. "
            f"{BATCH_COLUMN} 값과 TRAIN_BATCH_KEYWORDS 설정을 확인해 주세요."
        )
    if test_df.empty:
        raise ValueError(
            "테스트용 Batch 2 데이터가 비어 있습니다. "
            f"{BATCH_COLUMN} 값과 TEST_BATCH_KEYWORDS 설정을 확인해 주세요."
        )

    return train_source_df, test_df, additional_test_df


def split_train_valid(
    df: pd.DataFrame,
    valid_size: float = VALID_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if GROUP_COLUMN in df.columns and df[GROUP_COLUMN].nunique(dropna=True) >= 2:
        groups = df[GROUP_COLUMN].fillna("missing_group")
        splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
        train_idx, valid_idx = next(splitter.split(df, groups=groups))
        return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()

    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
    )
    return train_df.copy(), valid_df.copy()


# 전처리 파이프라인
class IdentityTransformer:
    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "IdentityTransformer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x


def build_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", Pipeline(steps=categorical_steps), categorical_columns),
        ],
        remainder="drop",
    )
    return transformer


# 모델링
# 1. 선형 모델
# 2. 트리 모델

def build_model_registry() -> dict[str, Any]:
    models: dict[str, Any] = {}

    # 1-1. 선형 회귀
    models["LinearRegression"] = LinearRegression(**LINEAR_REGRESSION_PARAMS)

    # 1-2. 릿지 회귀
    models["Ridge"] = Ridge(**RIDGE_REGRESSION_PARAMS)

    # 1-3. 엘라스틱 넷
    models["ElasticNet"] = ElasticNet(**ELASTIC_NET_PARAMS)

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


def build_model_pipeline(model_name: str, model: Any, x_train: pd.DataFrame) -> Pipeline:
    scale_numeric = model_name in {"LinearRegression", "Ridge", "ElasticNet"}
    preprocessor = build_preprocessor(x=x_train, scale_numeric=scale_numeric)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


# 모델 평가
# RMSE, R2, MAPE로 평가

def calculate_mape(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
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


def evaluate_regression(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)

    return {
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def make_cv_scorers() -> dict[str, Any]:
    return {
        "rmse": make_scorer(
            lambda y_true, y_pred: -float(np.sqrt(mean_squared_error(y_true, y_pred))),
            greater_is_better=True,
        ),
        "r2": make_scorer(r2_score),
        "mape": make_scorer(
            lambda y_true, y_pred: -calculate_mape(y_true, y_pred),
            greater_is_better=True,
        ),
    }


def cross_validate_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series | None,
) -> dict[str, float]:
    if groups is not None and groups.nunique(dropna=True) >= CV_SPLITS:
        cv = GroupKFold(n_splits=CV_SPLITS)
        cv_result = cross_validate(
            pipeline,
            x_train,
            y_train,
            groups=groups,
            cv=cv,
            scoring=make_cv_scorers(),
            n_jobs=None,
        )
    else:
        cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_result = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=make_cv_scorers(),
            n_jobs=None,
        )

    return {
        "rmse": float(-np.mean(cv_result["test_rmse"])),
        "r2": float(np.mean(cv_result["test_r2"])),
        "mape": float(-np.mean(cv_result["test_mape"])),
    }


# 학습 및 예측

def build_prediction_frame(
    model_name: str,
    split_name: str,
    source_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    prediction_df = pd.DataFrame(
        {
            "model_name": model_name,
            "data_split": split_name,
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
            "residual": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
        },
        index=source_df.index,
    )

    for column in [*ID_COLUMNS, BATCH_COLUMN, GROUP_COLUMN]:
        if column in source_df.columns:
            prediction_df[column] = source_df[column].values

    return prediction_df.reset_index(names="row_index")


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    additional_test_df: pd.DataFrame | None,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    score_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    best_model_name = ""
    best_model_mape = float("inf")
    best_pipeline: Pipeline | None = None

    x_train, y_train, _ = prepare_features_and_target(train_df, target_column=target_column)
    x_valid, y_valid, _ = prepare_features_and_target(valid_df, target_column=target_column)
    x_test, y_test, _ = prepare_features_and_target(test_df, target_column=target_column)

    groups = train_df[GROUP_COLUMN].fillna("missing_group") if GROUP_COLUMN in train_df.columns else None

    for model_name, model in build_model_registry().items():
        pipeline = build_model_pipeline(model_name=model_name, model=model, x_train=x_train)

        cv_metrics = cross_validate_model(
            pipeline=pipeline,
            x_train=x_train,
            y_train=y_train,
            groups=groups,
        )
        score_rows.append({"model_name": model_name, "stage": "Train (Batch 1 CV)", **cv_metrics})

        pipeline.fit(x_train, y_train)

        valid_predictions = pipeline.predict(x_valid)
        valid_metrics = evaluate_regression(y_valid, valid_predictions)
        score_rows.append({"model_name": model_name, "stage": "Valid (Batch 1 Hold-out)", **valid_metrics})
        prediction_frames.append(
            build_prediction_frame(
                model_name=model_name,
                split_name="valid_batch1",
                source_df=valid_df,
                y_true=y_valid,
                y_pred=valid_predictions,
            )
        )

        test_predictions = pipeline.predict(x_test)
        test_metrics = evaluate_regression(y_test, test_predictions)
        score_rows.append({"model_name": model_name, "stage": "Test (Batch 2)", **test_metrics})
        prediction_frames.append(
            build_prediction_frame(
                model_name=model_name,
                split_name="test_batch2",
                source_df=test_df,
                y_true=y_test,
                y_pred=test_predictions,
            )
        )

        train_valid_gap = valid_metrics["mape"] - cv_metrics["mape"]
        valid_test_gap = test_metrics["mape"] - valid_metrics["mape"]
        target_test_gap = test_metrics["mape"] - 9.1
        score_rows.append(
            {
                "model_name": model_name,
                "stage": "Gap (Train-Valid)",
                "rmse": np.nan,
                "r2": np.nan,
                "mape": float(train_valid_gap),
            }
        )
        score_rows.append(
            {
                "model_name": model_name,
                "stage": "Gap (Valid-Test)",
                "rmse": np.nan,
                "r2": np.nan,
                "mape": float(valid_test_gap),
            }
        )
        score_rows.append(
            {
                "model_name": model_name,
                "stage": "Gap (Target-Test)",
                "rmse": np.nan,
                "r2": np.nan,
                "mape": float(target_test_gap),
            }
        )

        if additional_test_df is not None and not additional_test_df.empty:
            x_additional, y_additional, _ = prepare_features_and_target(
                additional_test_df,
                target_column=target_column,
            )
            additional_predictions = pipeline.predict(x_additional)
            additional_metrics = evaluate_regression(y_additional, additional_predictions)
            score_rows.append(
                {"model_name": model_name, "stage": "Test (Batch 3)", **additional_metrics}
            )
            score_rows.append(
                {
                    "model_name": model_name,
                    "stage": "Gap (Batch2-Batch3)",
                    "rmse": np.nan,
                    "r2": np.nan,
                    "mape": float(additional_metrics["mape"] - test_metrics["mape"]),
                }
            )
            score_rows.append(
                {
                    "model_name": model_name,
                    "stage": "Gap (Target-Test Batch3)",
                    "rmse": np.nan,
                    "r2": np.nan,
                    "mape": float(additional_metrics["mape"] - 9.1),
                }
            )
            prediction_frames.append(
                build_prediction_frame(
                    model_name=model_name,
                    split_name="test_batch3",
                    source_df=additional_test_df,
                    y_true=y_additional,
                    y_pred=additional_predictions,
                )
            )

        if test_metrics["mape"] < best_model_mape:
            best_model_name = model_name
            best_model_mape = test_metrics["mape"]
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("학습 가능한 모델이 없습니다. 모델 레지스트리를 확인해 주세요.")

    score_df = pd.DataFrame(score_rows)
    score_df["model_rank"] = (
        score_df[score_df["stage"] == "Test (Batch 2)"]
        .set_index("model_name")["mape"]
        .rank(method="dense")
        .to_dict()
    )
    score_df["best_model_name"] = best_model_name

    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    return score_df, prediction_df, best_pipeline


# 결과 저장

def save_modeling_outputs(
    score_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    best_pipeline: Pipeline,
    score_path: Path = MODEL_SCORE_PATH,
    prediction_path: Path = MODEL_PREDICTION_PATH,
    model_path: Path = BEST_MODEL_PATH,
) -> None:
    score_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    score_df.to_csv(score_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)

    with model_path.open("wb") as file:
        pickle.dump(best_pipeline, file)


# 실행

def main() -> None:
    modeling_df = load_modeling_data()
    target_column = resolve_target_column(modeling_df)
    validate_modeling_dataframe(modeling_df, target_column=target_column)

    train_source_df, test_df, additional_test_df = split_batch_datasets(modeling_df)
    train_df, valid_df = split_train_valid(train_source_df)

    score_df, prediction_df, best_pipeline = train_and_evaluate_models(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        additional_test_df=additional_test_df if not additional_test_df.empty else None,
        target_column=target_column,
    )
    save_modeling_outputs(
        score_df=score_df,
        prediction_df=prediction_df,
        best_pipeline=best_pipeline,
    )

    summary_df = (
        score_df[score_df["stage"] == "Test (Batch 2)"]
        .sort_values("mape")
        .reset_index(drop=True)
    )
    print(summary_df[["model_name", "stage", "rmse", "r2", "mape"]])


if __name__ == "__main__":
    main()
