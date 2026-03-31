from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import (
    BLOCK_MAX_FEATURES,
    DEFAULT_ALPHA_GRID,
    DEFAULT_FEATURE_CACHE_DIR,
    DEFAULT_L1_RATIO_GRID,
    DEFAULT_OUTPUT_DIR,
    FEATURE_BLOCKS,
    HIGH_CORR_THRESHOLD,
    VIF_ALERT_THRESHOLD,
)
from features import load_feature_tables


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def group_holdout_split(df, group_col="charging_policy", valid_ratio=0.2, random_state=42):
    groups = pd.Series(df[group_col].dropna().unique())
    shuffled = groups.sample(frac=1.0, random_state=random_state).tolist()
    n_valid_groups = max(1, int(np.ceil(len(shuffled) * valid_ratio)))
    valid_groups = set(shuffled[:n_valid_groups])
    train_df = df[~df[group_col].isin(valid_groups)].copy()
    valid_df = df[df[group_col].isin(valid_groups)].copy()
    return train_df, valid_df, valid_groups


def build_group_folds(df, group_col="charging_policy", n_splits=4, random_state=42):
    groups = pd.Series(df[group_col].dropna().unique()).sample(frac=1.0, random_state=random_state).tolist()
    n_splits = max(2, min(n_splits, len(groups)))
    fold_groups = [groups[i::n_splits] for i in range(n_splits)]
    folds = []
    for valid_groups in fold_groups:
        valid_groups = set(valid_groups)
        train_fold = df[~df[group_col].isin(valid_groups)].copy()
        valid_fold = df[df[group_col].isin(valid_groups)].copy()
        if len(train_fold) > 0 and len(valid_fold) > 0:
            folds.append((train_fold, valid_fold))
    return folds


def fit_preprocessor(train_df, feature_cols):
    medians = train_df[feature_cols].median(numeric_only=True)
    filled = train_df[feature_cols].fillna(medians)
    means = filled.mean()
    stds = filled.std().replace(0, 1.0).fillna(1.0)
    return medians, means, stds


def transform_features(df, feature_cols, medians, means, stds):
    x = df[feature_cols].copy().fillna(medians)
    x = (x - means) / stds
    return x.to_numpy(dtype=float)


def soft_threshold(value, penalty):
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def fit_elastic_net(x, y, alpha, l1_ratio, max_iter=5000, tol=1e-6):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = x.mean(axis=0)
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean

    n_samples, n_features = x_centered.shape
    coef = np.zeros(n_features, dtype=float)
    residual = y_centered.copy()
    col_norm = (x_centered ** 2).sum(axis=0) / n_samples

    for _ in range(max_iter):
        max_update = 0.0
        for idx in range(n_features):
            if col_norm[idx] == 0:
                continue
            residual = residual + x_centered[:, idx] * coef[idx]
            rho = np.dot(x_centered[:, idx], residual) / n_samples
            new_coef = soft_threshold(rho, alpha * l1_ratio) / (col_norm[idx] + alpha * (1.0 - l1_ratio))
            residual = residual - x_centered[:, idx] * new_coef
            max_update = max(max_update, abs(new_coef - coef[idx]))
            coef[idx] = new_coef
        if max_update < tol:
            break

    intercept = y_mean - x_mean @ coef
    return intercept, coef


def predict_linear(x, intercept, coef):
    return np.asarray(x, dtype=float) @ coef + intercept


def feature_target_corr(df: pd.DataFrame, feature_cols, target_col="cycle_life"):
    rows = []
    for feature in feature_cols:
        series = df[[feature, target_col]].dropna()
        corr = series[feature].corr(series[target_col]) if len(series) >= 3 else np.nan
        rows.append({"feature": feature, "corr_with_target": corr, "abs_corr": abs(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def build_batch_stability_report(feature_tables: dict, feature_cols, target_col="cycle_life"):
    rows = []
    for feature in feature_cols:
        batch_corrs = {}
        for batch_name, df in feature_tables.items():
            series = df[[feature, target_col]].dropna() if feature in df.columns else pd.DataFrame()
            corr = series[feature].corr(series[target_col]) if len(series) >= 3 else np.nan
            batch_corrs[batch_name] = corr

        valid_corrs = [value for value in batch_corrs.values() if pd.notna(value)]
        signs = {int(np.sign(value)) for value in valid_corrs if value != 0}
        abs_corrs = [abs(value) for value in valid_corrs]

        rows.append(
            {
                "feature": feature,
                "batch1_corr": batch_corrs.get("batch1"),
                "batch2_corr": batch_corrs.get("batch2"),
                "batch3_corr": batch_corrs.get("batch3"),
                "mean_abs_corr": float(np.mean(abs_corrs)) if abs_corrs else np.nan,
                "min_abs_corr": float(np.min(abs_corrs)) if abs_corrs else np.nan,
                "max_abs_corr": float(np.max(abs_corrs)) if abs_corrs else np.nan,
                "sign_consistent": len(signs) <= 1 and len(valid_corrs) == len(feature_tables),
                "stable_candidate": (
                    len(signs) <= 1
                    and len(valid_corrs) == len(feature_tables)
                    and len([value for value in abs_corrs if value >= 0.20]) >= 2
                    and (float(np.mean(abs_corrs)) if abs_corrs else 0.0) >= 0.25
                ),
                "stable_candidate_strict": (
                    len(signs) <= 1
                    and len(valid_corrs) == len(feature_tables)
                    and len([value for value in abs_corrs if value >= 0.30]) == len(feature_tables)
                    and (float(np.mean(abs_corrs)) if abs_corrs else 0.0) >= 0.35
                    and (float(np.min(abs_corrs)) if abs_corrs else 0.0) >= 0.25
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["stable_candidate_strict", "stable_candidate", "mean_abs_corr"],
        ascending=[False, False, False],
    )


def find_high_corr_pairs(df: pd.DataFrame, feature_cols, threshold=HIGH_CORR_THRESHOLD):
    corr = df[feature_cols].corr(numeric_only=True)
    rows = []
    for idx, left in enumerate(feature_cols):
        for right in feature_cols[idx + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"feature_1": left, "feature_2": right, "corr": float(value)})
    if not rows:
        return pd.DataFrame(columns=["feature_1", "feature_2", "corr"])
    return pd.DataFrame(rows).sort_values("corr", key=lambda s: s.abs(), ascending=False)


def compute_vif_table(df: pd.DataFrame, feature_cols):
    if not feature_cols:
        return pd.DataFrame(columns=["feature", "vif"])

    clean = df[feature_cols].copy()
    clean = clean.fillna(clean.median(numeric_only=True))
    rows = []
    for feature in feature_cols:
        others = [col for col in feature_cols if col != feature]
        if not others:
            rows.append({"feature": feature, "vif": 1.0})
            continue
        y = clean[feature].to_numpy(dtype=float)
        x = clean[others].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ coef
        denom = np.sum((y - y.mean()) ** 2)
        r2 = 0.0 if denom == 0 else 1.0 - np.sum((y - fitted) ** 2) / denom
        vif = np.inf if r2 >= 0.999999 else 1.0 / max(1e-9, 1.0 - r2)
        rows.append({"feature": feature, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def select_features_from_block(train_df: pd.DataFrame, block_name: str):
    available = [feature for feature in FEATURE_BLOCKS[block_name] if feature in train_df.columns]
    available = [feature for feature in available if train_df[feature].notna().sum() >= 3]
    if not available:
        return [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    corr_report = feature_target_corr(train_df, available)
    selected = []
    for feature in corr_report["feature"]:
        if len(selected) >= BLOCK_MAX_FEATURES[block_name]:
            break
        keep = True
        for kept in selected:
            pair = train_df[[feature, kept]].dropna()
            if len(pair) >= 3 and abs(pair[feature].corr(pair[kept])) >= HIGH_CORR_THRESHOLD:
                keep = False
                break
        if keep:
            selected.append(feature)

    pair_report = find_high_corr_pairs(train_df, available)
    vif_report = compute_vif_table(train_df, selected)
    return selected, corr_report, pair_report, vif_report


def select_feature_blocks(train_df: pd.DataFrame):
    selected_blocks = {}
    report_rows = []
    pair_frames = []
    vif_frames = []

    for block_name in FEATURE_BLOCKS:
        selected, corr_report, pair_report, vif_report = select_features_from_block(train_df, block_name)
        selected_blocks[block_name] = selected

        for rank, row in enumerate(corr_report.itertuples(index=False), start=1):
            report_rows.append(
                {
                    "block": block_name,
                    "feature": row.feature,
                    "corr_with_target": row.corr_with_target,
                    "abs_corr": row.abs_corr,
                    "selected": row.feature in selected,
                    "rank_in_block": rank,
                }
            )

        if not pair_report.empty:
            pair_report = pair_report.copy()
            pair_report["block"] = block_name
            pair_frames.append(pair_report)

        if not vif_report.empty:
            vif_report = vif_report.copy()
            vif_report["block"] = block_name
            vif_report["vif_alert"] = vif_report["vif"] >= VIF_ALERT_THRESHOLD
            vif_frames.append(vif_report)

    feature_report = pd.DataFrame(report_rows)
    pair_report = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    vif_report = pd.concat(vif_frames, ignore_index=True) if vif_frames else pd.DataFrame()
    return selected_blocks, feature_report, pair_report, vif_report


def build_paper_feature_sets(df: pd.DataFrame):
    available = set(df.columns)

    variance_model = [feature for feature in ["delta_q_log_variance"] if feature in available]
    discharge_model = [
        feature
        for feature in (
            FEATURE_BLOCKS["summary"]
            + FEATURE_BLOCKS["fade"]
            + FEATURE_BLOCKS["delta_q"]
            + ["delta_q_log_variance"]
        )
        if feature in available
    ]
    full_model = [
        feature
        for feature in (
            FEATURE_BLOCKS["summary"]
            + FEATURE_BLOCKS["charging"]
            + FEATURE_BLOCKS["fade"]
            + FEATURE_BLOCKS["delta_q"]
            + ["delta_q_log_variance"]
        )
        if feature in available
    ]
    return {
        "variance_model": variance_model,
        "discharge_model": discharge_model,
        "full_model": full_model,
    }


def restrict_feature_sets_to_stable(feature_sets: dict, stability_report: pd.DataFrame):
    stable_features = set(stability_report.loc[stability_report["stable_candidate"], "feature"])
    restricted = {}
    for set_name, features in feature_sets.items():
        stable_only = [feature for feature in features if feature in stable_features]
        if stable_only:
            restricted[f"{set_name}_stable"] = stable_only
    return restricted


def restrict_feature_sets_to_stable_strict(feature_sets: dict, stability_report: pd.DataFrame):
    strict_features = set(stability_report.loc[stability_report["stable_candidate_strict"], "feature"])
    restricted = {}
    for set_name, features in feature_sets.items():
        strict_only = [feature for feature in features if feature in strict_features]
        if strict_only:
            restricted[f"{set_name}_stable_strict"] = strict_only
    return restricted


def build_eda_feature_sets(selected_blocks: dict):
    summary = selected_blocks.get("summary", [])
    charging = selected_blocks.get("charging", [])
    fade = selected_blocks.get("fade", [])
    delta_q = selected_blocks.get("delta_q", [])
    return {
        "eda_summary_only": summary,
        "eda_summary_charging": summary + charging,
        "eda_summary_charging_fade": summary + charging + fade,
        "eda_summary_charging_deltaq": summary + charging + delta_q,
        "eda_all_pruned": summary + charging + fade + delta_q,
    }


def evaluate_feature_set(train_df, valid_df, test_df, batch3_df, feature_cols, alpha_grid, l1_ratio_grid, group_col="charging_policy"):
    cv_rows = []
    folds = build_group_folds(train_df, group_col=group_col, n_splits=4, random_state=42)
    for alpha in alpha_grid:
        for l1_ratio in l1_ratio_grid:
            fold_scores = []
            for fold_train, fold_valid in folds:
                medians, means, stds = fit_preprocessor(fold_train, feature_cols)
                x_train = transform_features(fold_train, feature_cols, medians, means, stds)
                x_valid = transform_features(fold_valid, feature_cols, medians, means, stds)
                y_train = np.log(fold_train["cycle_life"].to_numpy(dtype=float))
                y_valid = fold_valid["cycle_life"].to_numpy(dtype=float)
                intercept, coef = fit_elastic_net(x_train, y_train, alpha=alpha, l1_ratio=l1_ratio)
                pred_valid = np.exp(predict_linear(x_valid, intercept, coef))
                fold_scores.append(mape(y_valid, pred_valid))
            cv_rows.append({"alpha": alpha, "l1_ratio": l1_ratio, "cv_mape": float(np.mean(fold_scores))})

    cv_table = pd.DataFrame(cv_rows).sort_values("cv_mape").reset_index(drop=True)
    best_alpha = float(cv_table.iloc[0]["alpha"])
    best_l1_ratio = float(cv_table.iloc[0]["l1_ratio"])

    medians, means, stds = fit_preprocessor(train_df, feature_cols)
    x_train = transform_features(train_df, feature_cols, medians, means, stds)
    x_valid = transform_features(valid_df, feature_cols, medians, means, stds)
    x_test = transform_features(test_df, feature_cols, medians, means, stds)
    x_batch3 = transform_features(batch3_df, feature_cols, medians, means, stds)

    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    y_valid = valid_df["cycle_life"].to_numpy(dtype=float)
    y_test = test_df["cycle_life"].to_numpy(dtype=float)
    y_batch3 = batch3_df["cycle_life"].to_numpy(dtype=float)

    intercept, coef = fit_elastic_net(x_train, np.log(y_train), alpha=best_alpha, l1_ratio=best_l1_ratio)
    pred_train = np.exp(predict_linear(x_train, intercept, coef))
    pred_valid = np.exp(predict_linear(x_valid, intercept, coef))
    pred_test = np.exp(predict_linear(x_test, intercept, coef))
    pred_batch3 = np.exp(predict_linear(x_batch3, intercept, coef))

    valid_pred_log = predict_linear(x_valid, intercept, coef)
    valid_true_log = np.log(y_valid)
    calib_x = np.column_stack([np.ones(len(valid_pred_log)), valid_pred_log])
    calib_coef, *_ = np.linalg.lstsq(calib_x, valid_true_log, rcond=None)
    calib_intercept = float(calib_coef[0])
    calib_slope = float(calib_coef[1])

    calibrated_valid = np.exp(calib_intercept + calib_slope * valid_pred_log)
    calibrated_test = np.exp(calib_intercept + calib_slope * predict_linear(x_test, intercept, coef))
    calibrated_batch3 = np.exp(calib_intercept + calib_slope * predict_linear(x_batch3, intercept, coef))

    metrics = {
        "best_alpha": best_alpha,
        "best_l1_ratio": best_l1_ratio,
        "n_features": len(feature_cols),
        "n_nonzero_coef": int(np.sum(np.abs(coef) > 1e-8)),
        "train_cv_mape": float(cv_table.iloc[0]["cv_mape"]),
        "train_fit_mape": mape(y_train, pred_train),
        "valid_mape": mape(y_valid, pred_valid),
        "test_mape": mape(y_test, pred_test),
        "batch3_mape": mape(y_batch3, pred_batch3),
        "calibrated_valid_mape": mape(y_valid, calibrated_valid),
        "calibrated_test_mape": mape(y_test, calibrated_test),
        "calibrated_batch3_mape": mape(y_batch3, calibrated_batch3),
        "calibration_intercept": calib_intercept,
        "calibration_slope": calib_slope,
    }
    artifacts = {
        "cv_table": cv_table,
        "medians": medians,
        "means": means,
        "stds": stds,
        "intercept": intercept,
        "coef": coef,
        "feature_cols": feature_cols,
        "predictions": {
            "train": pd.DataFrame({"y_true": y_train, "y_pred": pred_train}, index=train_df["cell_id"]),
            "valid": pd.DataFrame({"y_true": y_valid, "y_pred": pred_valid}, index=valid_df["cell_id"]),
            "test": pd.DataFrame({"y_true": y_test, "y_pred": pred_test}, index=test_df["cell_id"]),
            "batch3": pd.DataFrame({"y_true": y_batch3, "y_pred": pred_batch3}, index=batch3_df["cell_id"]),
            "valid_calibrated": pd.DataFrame({"y_true": y_valid, "y_pred": calibrated_valid}, index=valid_df["cell_id"]),
            "test_calibrated": pd.DataFrame({"y_true": y_test, "y_pred": calibrated_test}, index=test_df["cell_id"]),
            "batch3_calibrated": pd.DataFrame({"y_true": y_batch3, "y_pred": calibrated_batch3}, index=batch3_df["cell_id"]),
        },
    }
    return metrics, artifacts


def run_model_search(feature_tables: dict, output_dir: Path, alpha_grid, l1_ratio_grid):
    batch1_full = feature_tables["batch1"].dropna(subset=["cycle_life", "charging_policy"]).copy()
    batch2_test = feature_tables["batch2"].dropna(subset=["cycle_life"]).copy()
    batch3_test = feature_tables["batch3"].dropna(subset=["cycle_life"]).copy()

    train_df, valid_df, valid_groups = group_holdout_split(
        batch1_full,
        group_col="charging_policy",
        valid_ratio=0.2,
        random_state=42,
    )

    selected_blocks, feature_report, pair_report, vif_report = select_feature_blocks(train_df)
    paper_feature_sets = build_paper_feature_sets(batch1_full)
    eda_feature_sets = build_eda_feature_sets(selected_blocks)
    feature_sets = {**paper_feature_sets, **eda_feature_sets}
    stability_candidates = sorted({feature for features in feature_sets.values() for feature in features})
    stability_report = build_batch_stability_report(feature_tables, stability_candidates)
    stable_feature_sets = restrict_feature_sets_to_stable(feature_sets, stability_report)
    strict_stable_feature_sets = restrict_feature_sets_to_stable_strict(feature_sets, stability_report)
    feature_sets = {**feature_sets, **stable_feature_sets, **strict_stable_feature_sets}

    print(f"[split] train={train_df.shape}, valid={valid_df.shape}, batch2_test={batch2_test.shape}, batch3_test={batch3_test.shape}", flush=True)
    print(f"[select] blocks={selected_blocks}", flush=True)

    model_rows = []
    artifacts_by_set = {}
    for set_name, features in feature_sets.items():
        features = [feature for feature in features if feature in batch1_full.columns]
        if not features:
            continue
        print(f"[model] evaluating {set_name} with {len(features)} features", flush=True)
        metrics, artifacts = evaluate_feature_set(
            train_df=train_df,
            valid_df=valid_df,
            test_df=batch2_test,
            batch3_df=batch3_test,
            feature_cols=features,
            alpha_grid=alpha_grid,
            l1_ratio_grid=l1_ratio_grid,
            group_col="charging_policy",
        )
        metrics["feature_set"] = set_name
        model_rows.append(metrics)
        artifacts_by_set[set_name] = artifacts
        print(
            f"[model] {set_name}: valid_mape={metrics['valid_mape']:.3f}, test_mape={metrics['test_mape']:.3f}, batch3_mape={metrics['batch3_mape']:.3f}, calibrated_test={metrics['calibrated_test_mape']:.3f}, calibrated_batch3={metrics['calibrated_batch3_mape']:.3f}, nonzero={metrics['n_nonzero_coef']}",
            flush=True,
        )

    model_search = pd.DataFrame(model_rows)
    model_search["robust_score"] = (
        0.20 * model_search["valid_mape"]
        + 0.40 * model_search["calibrated_test_mape"]
        + 0.40 * model_search["calibrated_batch3_mape"]
    )
    model_search = model_search.sort_values(
        ["robust_score", "calibrated_test_mape", "calibrated_batch3_mape", "valid_mape"]
    ).reset_index(drop=True)
    best_feature_set = model_search.iloc[0]["feature_set"]
    best_artifacts = artifacts_by_set[best_feature_set]

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_report.to_csv(output_dir / "feature_screen_report.csv", index=False)
    if not pair_report.empty:
        pair_report.to_csv(output_dir / "high_corr_pairs.csv", index=False)
    if not vif_report.empty:
        vif_report.to_csv(output_dir / "vif_report.csv", index=False)
    model_search.to_csv(output_dir / "model_search.csv", index=False)
    stability_report.to_csv(output_dir / "batch_stability_report.csv", index=False)

    selected_rows = []
    for block_name, features in selected_blocks.items():
        for feature in features:
            selected_rows.append({"block": block_name, "feature": feature})
    pd.DataFrame(selected_rows).to_csv(output_dir / "selected_features.csv", index=False)

    coef_table = pd.DataFrame(
        {"feature": best_artifacts["feature_cols"], "coef": best_artifacts["coef"]}
    ).sort_values("coef", key=lambda s: s.abs(), ascending=False)
    coef_table.to_csv(output_dir / "best_model_coefficients.csv", index=False)

    for split_name, pred_df in best_artifacts["predictions"].items():
        pred_df.reset_index(names="cell_id").to_csv(output_dir / f"predictions_{split_name}.csv", index=False)

    summary = {
        "best_feature_set": best_feature_set,
        "valid_groups": sorted(valid_groups),
        "selected_blocks": selected_blocks,
        "paper_feature_sets": paper_feature_sets,
        "eda_feature_sets": eda_feature_sets,
        "all_feature_sets": feature_sets,
        "stable_features": stability_report.loc[stability_report["stable_candidate"], "feature"].tolist(),
        "strict_stable_features": stability_report.loc[stability_report["stable_candidate_strict"], "feature"].tolist(),
        "best_metrics": model_search.iloc[0].to_dict(),
    }
    (output_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "batch2_test": batch2_test,
        "batch3_test": batch3_test,
        "selected_blocks": selected_blocks,
        "feature_report": feature_report,
        "pair_report": pair_report,
        "vif_report": vif_report,
        "stability_report": stability_report,
        "model_search": model_search,
        "best_feature_set": best_feature_set,
        "best_artifacts": best_artifacts,
    }


def main():
    feature_tables, combined = load_feature_tables(DEFAULT_FEATURE_CACHE_DIR)
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_name, table in feature_tables.items():
        table.to_csv(output_dir / f"features_{batch_name}.csv", index=False)
    combined.to_csv(output_dir / "features_all_batches.csv", index=False)

    result = run_model_search(
        feature_tables=feature_tables,
        output_dir=output_dir,
        alpha_grid=DEFAULT_ALPHA_GRID,
        l1_ratio_grid=DEFAULT_L1_RATIO_GRID,
    )

    print("Best feature set:", result["best_feature_set"])
    print(result["model_search"].head().to_string(index=False))


if __name__ == "__main__":
    main()
