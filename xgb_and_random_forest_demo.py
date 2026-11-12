"""Standalone synthetic drowsiness detection and prediction demo."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
import logging

logging.basicConfig(
    filename="outputs/logs/model.log",  # Log file path
    filemode="a", # 'w' to overwrite the log file each run, 'a' to append
    level=logging.INFO,  # Set the minimum level to capture (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define how the log looks
)

logging.info("Starting drowsiness_xgb_demo.py script.")

# Random Forest uses bagged trees; XGBoost uses gradient-boosted trees.
MODEL_MODE: Literal["RANDOM_FOREST", "XGBOOST"] = "XGBOOST"
MODEL_DISPLAY_NAMES = {
    "RANDOM_FOREST": "Random Forest",
    "XGBOOST": "XGBoost",
}
RANDOM_STATE = 42
N_DRIVERS = 20
MINUTES_PER_DRIVE = 150
PREDICTION_HORIZON_MIN = 5

FEATURE_NAMES = [
    "PERCLOS",
    "blink_rate",
    "slow_blink_freq",
    "avg_blink_duration_ms",
    "blink_duration_var",
    "rebased_delta_EAR",
    "rebased_delta_MAR",
    "yawn_rate",
    "head_nod_freq",
    "t_drive_min",
    "delta_PERCLOS_5min",
    "TPMA_alertness_A_t",
]

# The forecast uses trajectory and fatigue precursors rather than current-state eye snapshots.
PREDICTION_FEATURES = [
    "slow_blink_freq",
    "t_drive_min",
    "delta_PERCLOS_5min",
    "blink_duration_var",
    "TPMA_alertness_A_t",
]

IMPORTANCE_COLUMNS = {
    "total_gain": "Total gain",
    "gain": "Avg gain",
    "weight": "Splits",
    "total_cover": "Total cover",
    "cover": "Avg cover",
}


def generate_synthetic_data(seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate reproducible one-minute feature windows and both ground-truth labels."""
    rng = np.random.default_rng(seed)
    session_frames: list[pd.DataFrame] = []

    for driver_id in range(1, N_DRIVERS + 1):
        minute = np.arange(1, MINUTES_PER_DRIVE + 1, dtype=float)
        onset_minute = rng.uniform(80.0, 125.0)
        transition_width = rng.uniform(15.0, 24.0)
        circadian_phase = rng.uniform(0, 2 * np.pi)
        session_trend = rng.normal(0, 0.0018, size=MINUTES_PER_DRIVE).cumsum()
        circadian_effect = 0.025 * np.sin((minute / 150.0) * 2 * np.pi + circadian_phase)
        fatigue = np.clip(
            0.08
            + 0.84 / (1.0 + np.exp(-(minute - onset_minute) / transition_width))
            + session_trend
            + circadian_effect,
            0.03,
            0.97,
        )
        local_fatigue_slope = np.gradient(fatigue)
        instantaneous_ocular_noise = rng.normal(0, 0.95, MINUTES_PER_DRIVE)

        perclos_noise = np.empty(MINUTES_PER_DRIVE)
        perclos_noise[0] = rng.normal(0, 0.004)
        for index in range(1, MINUTES_PER_DRIVE):
            perclos_noise[index] = (
                0.88 * perclos_noise[index - 1] + rng.normal(0, 0.0015)
            )
        driver_perclos_offset = rng.normal(0, 0.02)
        perclos = np.clip(
            0.018
            + 0.43 * fatigue
            + driver_perclos_offset
            + perclos_noise,
            0.01,
            0.45,
        )
        blink_rate = np.clip(
            29.0 - 13.5 * fatigue + rng.normal(0, 2.6, MINUTES_PER_DRIVE),
            6.0,
            35.0,
        )
        slow_blink_freq = np.clip(
            0.15
            + 6.2 * fatigue**1.55
            + 0.55 * instantaneous_ocular_noise
            + rng.normal(0, 0.35)
            + rng.normal(0, 0.42, MINUTES_PER_DRIVE),
            0.0,
            8.0,
        )
        avg_blink_duration = np.clip(
            145.0 + 330.0 * fatigue + rng.normal(0, 36, MINUTES_PER_DRIVE),
            100.0,
            650.0,
        )
        blink_duration_var = np.clip(
            400.0
            + 1000.0 * fatigue**1.3
            + 80_000.0 * (local_fatigue_slope - 0.005)
            + rng.normal(0, 145, MINUTES_PER_DRIVE),
            50.0,
            2500.0,
        )
        rebased_delta_ear = np.clip(
            -0.006
            - 0.125 * fatigue
            - 0.014 * instantaneous_ocular_noise
            + rng.normal(0, 0.008, MINUTES_PER_DRIVE),
            -0.18,
            0.03,
        )
        rebased_delta_mar = np.clip(
            -0.008 + 0.24 * fatigue + rng.normal(0, 0.035, MINUTES_PER_DRIVE),
            -0.05,
            0.45,
        )
        yawn_rate = np.clip(
            0.04 + 1.65 * fatigue**1.65 + rng.normal(0, 0.25, MINUTES_PER_DRIVE),
            0.0,
            3.0,
        )
        head_nod_freq = np.clip(
            0.03 + 2.25 * fatigue**1.8 + rng.normal(0, 0.28, MINUTES_PER_DRIVE),
            0.0,
            5.0,
        )
        delta_perclos_5min = np.zeros(MINUTES_PER_DRIVE)
        delta_perclos_5min[PREDICTION_HORIZON_MIN:] = (
            perclos[PREDICTION_HORIZON_MIN:] - perclos[:-PREDICTION_HORIZON_MIN]
        )
        delta_perclos_5min = np.clip(delta_perclos_5min, -0.05, 0.20)
        tpma_alertness = np.clip(
            14.0
            - 10.0 * fatigue
            - 500.0 * local_fatigue_slope
            + rng.normal(0, 0.75, MINUTES_PER_DRIVE),
            3.0,
            16.0,
        )

        kss = np.clip(
            np.rint(
                1.0
                + 8.0 * fatigue
                + 0.65 * instantaneous_ocular_noise
                + rng.normal(0, 1.7, MINUTES_PER_DRIVE)
            ),
            1,
            9,
        ).astype(int)
        session = pd.DataFrame(
            {
                "driver_id": driver_id,
                "KSS": kss,
                "PERCLOS": perclos,
                "blink_rate": blink_rate,
                "slow_blink_freq": slow_blink_freq,
                "avg_blink_duration_ms": avg_blink_duration,
                "blink_duration_var": blink_duration_var,
                "rebased_delta_EAR": rebased_delta_ear,
                "rebased_delta_MAR": rebased_delta_mar,
                "yawn_rate": yawn_rate,
                "head_nod_freq": head_nod_freq,
                "t_drive_min": minute,
                "delta_PERCLOS_5min": delta_perclos_5min,
                "TPMA_alertness_A_t": tpma_alertness,
            }
        )
        session_frames.append(session)

    data = pd.concat(session_frames, ignore_index=True)
    data["y_detect"] = (data["KSS"] >= 7).astype(int)

    prediction_target = pd.Series(pd.NA, index=data.index, dtype="Int64")
    for start in range(0, len(data), MINUTES_PER_DRIVE):
        stop = start + MINUTES_PER_DRIVE
        session_kss = data.loc[start:stop - 1, "KSS"].to_numpy()
        for current_idx in range(MINUTES_PER_DRIVE - PREDICTION_HORIZON_MIN):
            future_kss = session_kss[
                current_idx + 1 : current_idx + PREDICTION_HORIZON_MIN + 1
            ]
            global_idx = start + current_idx
            if data.at[global_idx, "y_detect"] == 0:
                prediction_target.at[global_idx] = int(np.any(future_kss >= 7))

    data["y_predict_5min"] = prediction_target
    return data


def make_classifier(
    mode: Literal["RANDOM_FOREST", "XGBOOST"],
) -> RandomForestClassifier | XGBClassifier:
    if mode == "RANDOM_FOREST":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            max_features=0.7,
            max_samples=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if mode == "XGBOOST":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(
        f"Unsupported MODEL_MODE {mode!r}; choose 'RANDOM_FOREST' or 'XGBOOST'."
    )


def extract_importance_df(
    model: RandomForestClassifier | XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return native feature importance as percentage shares."""
    if isinstance(model, RandomForestClassifier):
        fitted_feature_names = model.feature_names_in_
        fitted_importance = pd.Series(
            model.feature_importances_,
            index=fitted_feature_names,
            dtype=float,
        )
        importance_values = (
            fitted_importance.reindex(feature_names, fill_value=0.0) * 100.0
        )
        importance = pd.DataFrame(
            {"Impurity importance": importance_values},
            index=feature_names,
        )
        return importance.sort_values("Impurity importance", ascending=False)

    booster = model.get_booster()
    importance = pd.DataFrame(index=feature_names, columns=IMPORTANCE_COLUMNS.values(), dtype=float)

    for importance_type, column_name in IMPORTANCE_COLUMNS.items():
        scores = booster.get_score(importance_type=importance_type)
        for position, feature_name in enumerate(feature_names):
            score = scores.get(feature_name)
            if score is None:
                score = scores.get(f"f{position}", 0.0)
            importance.at[feature_name, column_name] = float(score)

        total = importance[column_name].sum()
        if total <= 0:
            raise ValueError(
                f"XGBoost returned no {importance_type} importance; cannot normalize it."
            )
        importance[column_name] = importance[column_name] / total * 100.0

    return importance.sort_values("Total gain", ascending=False)


def print_metrics(
    label: str,
    model: RandomForestClassifier | XGBClassifier,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1]
    if target.nunique() < 2:
        raise ValueError(f"{label} test labels contain only one class; ROC-AUC is undefined.")

    logging.info(f"\n{label} test metrics")
    logging.info(f"  Accuracy : {accuracy_score(target, prediction):.3f}")
    logging.info(f"  Precision: {precision_score(target, prediction, zero_division=0):.3f}")
    logging.info(f"  Recall   : {recall_score(target, prediction, zero_division=0):.3f}")
    logging.info(f"  F1       : {f1_score(target, prediction, zero_division=0):.3f}")
    logging.info(f"  ROC-AUC  : {roc_auc_score(target, probability):.3f}")

def render_importance_heatmap(
    detection_importance: pd.DataFrame,
    prediction_importance: pd.DataFrame,
    output_path: Path,
    model_mode: Literal["RANDOM_FOREST", "XGBOOST"],
) -> None:
    """Save the two-panel importance heatmap using the detection ranking."""
    model_name = MODEL_DISPLAY_NAMES[model_mode]
    feature_order = detection_importance.index.tolist()
    importance_columns = detection_importance.columns.tolist()
    primary_importance = (
        "Impurity importance" if model_mode == "RANDOM_FOREST" else "Total gain"
    )
    left_values = detection_importance.loc[feature_order, importance_columns]
    right_values = pd.DataFrame(
        {
            f"Detection ({model_name})": detection_importance.loc[
                feature_order, primary_importance
            ],
            f"Prediction ({model_name})": prediction_importance[
                primary_importance
            ].reindex(
                feature_order
            ),
        },
        index=feature_order,
    )
    max_value = max(float(left_values.to_numpy().max()), float(right_values.to_numpy().max()))
    color_max = max(25.0, float(np.ceil(max_value / 5.0) * 5.0))

    sns.set_theme(style="white", font="DejaVu Sans")
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.1], wspace=0.35)
    left_ax = fig.add_subplot(grid[0, 0])
    right_ax = fig.add_subplot(grid[0, 1])
    fig.suptitle(
        f"What the {model_name} Drowsiness Pipeline Knows Best — "
        "Feature Importance Heatmap",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.94,
        (
            "300 trees · 12 features · Random Forest (Gini impurity importance)."
            if model_mode == "RANDOM_FOREST"
            else "300 trees · 12 features · binary:logistic. Gain = how much a feature "
            "improves predictions; Splits = how often it's used; Cover = samples it touches."
        ),
        ha="left",
        fontsize=10,
        color="dimgray",
    )

    sns.heatmap(
        left_values,
        ax=left_ax,
        cmap="Blues",
        vmin=0,
        vmax=color_max,
        annot=True,
        fmt=".1f",
        linewidths=1.5,
        linecolor="white",
        cbar=False,
        yticklabels=True,
    )
    left_ax.set_title(
        (
            "Stage 1 Detection Model — Impurity importance (%), sorted descending"
            if model_mode == "RANDOM_FOREST"
            else "Stage 1 Detection Model — Share of each importance metric (%), "
            "sorted by total gain"
        ),
        fontsize=11,
        pad=14,
    )
    left_ax.xaxis.tick_top()
    left_ax.xaxis.set_label_position("top")
    left_ax.set_xlabel("")
    left_ax.set_ylabel("")
    left_ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    left_ax.tick_params(axis="y", labelrotation=0, labelsize=9)

    sns.heatmap(
        right_values,
        ax=right_ax,
        cmap="Blues",
        vmin=0,
        vmax=color_max,
        annot=True,
        fmt=".1f",
        linewidths=1.5,
        linecolor="white",
        cbar=False,
        yticklabels=False,
    )
    right_ax.set_title(
        f"Detection vs. Prediction\n({primary_importance} %)",
        fontsize=11,
        pad=14,
    )
    right_ax.xaxis.tick_top()
    right_ax.xaxis.set_label_position("top")
    right_ax.set_xlabel("")
    right_ax.set_ylabel("")
    right_ax.tick_params(axis="x", labelrotation=0, labelsize=9)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.15)
    colorbar_ax = fig.add_axes([0.24, 0.065, 0.55, 0.025])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=0, vmax=color_max), cmap="Blues"),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label("% of metric total (darker = model relies on it more)", fontsize=10)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent/"outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    csv_path = output_dir / "synthetic_driver_drowsiness_data.csv"
    image_path = output_dir / f"{MODEL_MODE.lower()}_drowsiness_feature_importance.png"
    logging.info(f"Selected model family: {MODEL_DISPLAY_NAMES[MODEL_MODE]}")

    if csv_path.exists():
        logging.info(f"Loading existing CSV: {csv_path}")
        data = pd.read_csv(csv_path)
    else:
        logging.info(f"Generating synthetic data and saving to CSV: {csv_path}")
        data = generate_synthetic_data()
        data.to_csv(csv_path, index=False)

    # print(f"Synthetic data: {len(data):,} rows x {len(data.columns)} columns")
    # print(f"Saved CSV: {csv_path}")
    # print("\nFirst 10 synthetic rows:")
    # print(data.head(10).to_string(index=False))

    logging.info(f"Stage 1 class counts: {data['y_detect'].value_counts().sort_index().to_dict()}")

    logging.info(f"Stage 2 eligible class counts: {data['y_predict_5min'].value_counts(dropna=True).sort_index().to_dict()}")

    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)
    train_indices, test_indices = next(
        splitter.split(data[FEATURE_NAMES], data["y_detect"], groups=data["driver_id"])
    )
    train_driver_ids = set(data.iloc[train_indices]["driver_id"])
    test_driver_ids = set(data.iloc[test_indices]["driver_id"])
    detection_train = data.iloc[train_indices]
    detection_test = data.iloc[test_indices]
    detection_model = make_classifier(MODEL_MODE)
    detection_model.fit(
        detection_train[FEATURE_NAMES],
        detection_train["y_detect"],
    )

    print_metrics(
        "Stage 1 Detection",
        detection_model,
        detection_test[FEATURE_NAMES],
        detection_test["y_detect"],
    )

    detection_importance = extract_importance_df(detection_model, FEATURE_NAMES)

    eligible = data["y_predict_5min"].notna()
    prediction_train = data.loc[eligible & data["driver_id"].isin(train_driver_ids)]
    prediction_test = data.loc[eligible & data["driver_id"].isin(test_driver_ids)]
    if prediction_train["y_predict_5min"].nunique() < 2:
        raise ValueError("Stage 2 training labels contain only one class.")
    prediction_model = make_classifier(MODEL_MODE)
    prediction_model.fit(
        prediction_train[PREDICTION_FEATURES],
        prediction_train["y_predict_5min"].astype(int),
    )
    print_metrics(
        "Stage 2 5-minute Prediction",
        prediction_model,
        prediction_test[PREDICTION_FEATURES],
        prediction_test["y_predict_5min"].astype(int),
    )
    prediction_importance = extract_importance_df(prediction_model, FEATURE_NAMES)

    logging.info("Stage 1 feature importance (% of each metric total):")
    logging.info("\n" + detection_importance.to_string(float_format=lambda value: f"{value:6.2f}"))
    logging.info("Stage 2 feature importance (% of each metric total):")
    logging.info("\n" + prediction_importance.to_string(float_format=lambda value: f"{value:6.2f}"))

    primary_importance = (
        "Impurity importance" if MODEL_MODE == "RANDOM_FOREST" else "Total gain"
    )
    comparison = pd.DataFrame(
        {
            f"Detection ({MODEL_DISPLAY_NAMES[MODEL_MODE]})": detection_importance[
                primary_importance
            ],
            f"Prediction ({MODEL_DISPLAY_NAMES[MODEL_MODE]})": prediction_importance[
                primary_importance
            ],
        }
    ).reindex(detection_importance.index)
    logging.info(f"{primary_importance} comparison (%):")
    logging.info("\n" + comparison.to_string(float_format=lambda value: f"{value:6.2f}"))

    render_importance_heatmap(
        detection_importance,
        prediction_importance,
        image_path,
        MODEL_MODE,
    )
    logging.info(f"Saved feature importance heatmap: {image_path}")


if __name__ == "__main__":
    main()
