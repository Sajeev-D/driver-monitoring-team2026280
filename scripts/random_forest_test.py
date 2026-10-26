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
from src.driver_monitoring_ml.evaluation.metrics import *
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

def main():
    logging.info("creating random forest classifier object")
    rf = RandomForestClassifier(
        number_of_trees=10,
        max_tree_depth=4,
        max_features=0.7,
        random_state=42
    )

    logging.info("training random forest classifier")
    rf.fit(X_train, y_train)

    logging.info("running inference using random forest classifier")
    outputs = rf.predict(X_test)

    logging.info(f"len of outputs is {len(outputs)} and len of y_test is {len(y_test)}")

    confusion_matrix_tuple = derive_confusion_matrix(y_test, outputs)
    tp_count = confusion_matrix_tuple[0]
    fp_count = confusion_matrix_tuple[2]
    tn_count = confusion_matrix_tuple[1]
    fn_count = confusion_matrix_tuple[3]

    acc = accuracy_score(tp_count, tn_count, len(y_test))
    prec = precision_score(tp_count, fp_count)
    recall = recall_score(tp_count, fn_count)
    f1 = f1_score(prec, recall)

    logging.info(
    "Evaluation Metrics | Accuracy: %.2f | Precision: %.2f | "
    "Recall: %.2f | F1 Score: %.2f",
    acc, prec, recall, f1
)

if __name__ == "__main__":
    main()
