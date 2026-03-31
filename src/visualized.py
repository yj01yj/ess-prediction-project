#
# 제작 목적: 시각화
# 제작 날짜: 2026-03-31
# 제작자: AI 3기 3반 박진
#


# 라이브러리
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# 하이퍼 파라미터
PREDICTION_RESULT_PATH = Path("outputs/model_outputs/model_predictions.csv")
FIGURE_OUTPUT_DIR = Path("outputs/images/visualizing")

MODEL_COLUMN = "model_name"
ACTUAL_COLUMN = "actual"
PREDICTED_COLUMN = "predicted"
RESIDUAL_COLUMN = "residual"

FIGSIZE = (8, 6)
SCATTER_ALPHA = 0.7
HIST_BINS = 30
IMAGE_DPI = 150


# 최종 결과 데이터 호출
def load_prediction_results(path: Path = PREDICTION_RESULT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"최종 예측 결과 파일이 없습니다: {path}. "
            "modeling.py에서 저장한 결과 파일 경로로 수정해 주세요."
        )

    result_df = pd.read_csv(path)
    required_columns = {
        MODEL_COLUMN,
        ACTUAL_COLUMN,
        PREDICTED_COLUMN,
    }
    missing_columns = required_columns - set(result_df.columns)
    if missing_columns:
        raise KeyError(f"시각화에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

    if RESIDUAL_COLUMN not in result_df.columns:
        result_df[RESIDUAL_COLUMN] = (
            result_df[ACTUAL_COLUMN] - result_df[PREDICTED_COLUMN]
        )

    return result_df


def slugify_model_name(model_name: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip().lower())
    return safe_name.strip("_") or "model"


# 시각화
# 1. 모델별 실제 vs 예측값 산점도
def plot_actual_vs_predicted(
    model_df: pd.DataFrame,
    model_name: str,
    output_dir: Path = FIGURE_OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slugify_model_name(model_name)}_actual_vs_predicted.png"

    plt.figure(figsize=FIGSIZE)
    plt.scatter(
        model_df[ACTUAL_COLUMN],
        model_df[PREDICTED_COLUMN],
        alpha=SCATTER_ALPHA,
    )

    min_value = min(model_df[ACTUAL_COLUMN].min(), model_df[PREDICTED_COLUMN].min())
    max_value = max(model_df[ACTUAL_COLUMN].max(), model_df[PREDICTED_COLUMN].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.title(f"{model_name} Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


# 2. 모델별 잔차 플롯
def plot_residuals(
    model_df: pd.DataFrame,
    model_name: str,
    output_dir: Path = FIGURE_OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slugify_model_name(model_name)}_residual_plot.png"

    plt.figure(figsize=FIGSIZE)
    plt.scatter(
        model_df[PREDICTED_COLUMN],
        model_df[RESIDUAL_COLUMN],
        alpha=SCATTER_ALPHA,
    )
    plt.axhline(y=0, linestyle="--")

    plt.title(f"{model_name} Residual Plot")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


# 3. 모델별 예측값 분포
def plot_prediction_distribution(
    model_df: pd.DataFrame,
    model_name: str,
    output_dir: Path = FIGURE_OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slugify_model_name(model_name)}_prediction_distribution.png"

    plt.figure(figsize=FIGSIZE)
    plt.hist(model_df[PREDICTED_COLUMN], bins=HIST_BINS, alpha=SCATTER_ALPHA)

    plt.title(f"{model_name} Prediction Distribution")
    plt.xlabel("Predicted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=IMAGE_DPI)
    plt.close()
    return output_path


def create_all_visualizations(
    prediction_df: pd.DataFrame,
    model_column: str = MODEL_COLUMN,
    output_dir: Path = FIGURE_OUTPUT_DIR,
) -> list[Path]:
    generated_files: list[Path] = []

    for model_name, model_df in prediction_df.groupby(model_column):
        generated_files.append(
            plot_actual_vs_predicted(
                model_df=model_df,
                model_name=model_name,
                output_dir=output_dir,
            )
        )
        generated_files.append(
            plot_residuals(
                model_df=model_df,
                model_name=model_name,
                output_dir=output_dir,
            )
        )
        generated_files.append(
            plot_prediction_distribution(
                model_df=model_df,
                model_name=model_name,
                output_dir=output_dir,
            )
        )

    return generated_files


def run_visualization_pipeline(
    prediction_result_path: Path = PREDICTION_RESULT_PATH,
    output_dir: Path = FIGURE_OUTPUT_DIR,
) -> list[Path]:
    prediction_df = load_prediction_results(prediction_result_path)
    return create_all_visualizations(
        prediction_df,
        model_column=MODEL_COLUMN,
        output_dir=output_dir,
    )


def main() -> None:
    run_visualization_pipeline(output_dir=FIGURE_OUTPUT_DIR)


if __name__ == "__main__":
    main()