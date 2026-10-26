from pathlib import Path
import sys
import logging

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root)) # put the project root in the import path

logging.basicConfig(
    filename="outputs/logs/model.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from src.driver_monitoring_ml.models.random_forest import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from src.driver_monitoring_ml.evaluation.metrics import (
    accuracy_score,
    derive_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import numpy as np

logging.info("in random_forest_test.py")

df = pd.read_csv("data\synthetic\driver_monitoring_synthetic_test.csv")

feature_cols = [
    "eye_openness",
    "perclos",
    "blink_rate_per_min",
    "avg_blink_duration_ms",
    "blink_duration_variability_ms",
    "yawning_rate_per_min",
    "head_pitch_deg",
    "head_yaw_deg",
    "head_roll_deg",
]

label_col = "drowsiness_label"

# drop samples with incomplete data
df = df.dropna(subset=feature_cols + [label_col])

participants = df["participant_id"].unique()
rng = np.random.RandomState(0)
rng.shuffle(participants)

n_test = max(1, int(0.2 * len(participants)))
test_ids = participants[:n_test]
test_mask = df["participant_id"].isin(test_ids)

# set training and test dataset
train_df = df[~test_mask]
test_df = df[test_mask]

# seperate the features and labels for the training and test set
X_train = train_df[feature_cols].to_numpy()
y_train = train_df[label_col].to_numpy()
X_test = test_df[feature_cols].to_numpy()
y_test = test_df[label_col].to_numpy()


def log_metrics(label, y_true, predictions):
    tp_count, tn_count, fp_count, fn_count = derive_confusion_matrix(
        y_true, predictions
    )
    accuracy = accuracy_score(tp_count, tn_count, len(y_true))
    precision = precision_score(tp_count, fp_count)
    recall = recall_score(tp_count, fn_count)
    f1 = f1_score(precision, recall)

    logging.info(
        "%s | Accuracy: %.2f | Precision: %.2f | Recall: %.2f | F1 Score: %.2f",
        label,
        accuracy,
        precision,
        recall,
        f1,
    )


def main():
    logging.info("creating random forest classifier object")
    rf = RandomForestClassifier(
        number_of_trees=10,
        max_tree_depth=4,
        max_features=0.7,
        random_state=42
    )

    rf_scikit = SklearnRandomForestClassifier(
        n_estimators=10,
        max_depth=4,
        max_features=0.7,
        random_state=42,
    )

    logging.info("training custom random forest classifier")
    rf.fit(X_train, y_train)

    logging.info("training scikit-learn random forest classifier")
    rf_scikit.fit(X_train, y_train)

    logging.info("running inference using custom random forest classifier")
    custom_predictions = rf.predict(X_test)

    logging.info("running inference using scikit-learn random forest classifier")
    sklearn_predictions = rf_scikit.predict(X_test)

    log_metrics("Custom random forest metrics", y_test, custom_predictions)
    log_metrics("Scikit-learn random forest metrics", y_test, sklearn_predictions)

if __name__ == "__main__":
    main()
