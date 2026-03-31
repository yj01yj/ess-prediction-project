from __future__ import annotations

import pandas as pd

from configs.config import DEFAULT_DATA_DIR, DEFAULT_FEATURE_CACHE_DIR
from features import build_and_save_feature_cache


def run_feature_engineering_pipeline(
    input_path=DEFAULT_DATA_DIR,
    output_path=DEFAULT_FEATURE_CACHE_DIR,
) -> pd.DataFrame:
    _, model_input_df = build_and_save_feature_cache(
        data_dir=input_path,
        feature_cache_dir=output_path,
    )
    return model_input_df


def main() -> None:
    run_feature_engineering_pipeline()


if __name__ == "__main__":
    main()
