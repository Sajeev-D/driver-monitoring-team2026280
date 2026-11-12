from pathlib import Path
import pandas as pd
import sys
import joblib

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root)) 

from src.driver_monitoring_ml.models.random_forest import RandomForestClassifier
from configs import config
import logging

logging.basicConfig(
    filename=config.LOG_PATH,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MODEL_PATH = project_root / "outputs" / "models" / "custom_random_forest.joblib"

def train_model(
    csv_path: Path,
    feature_columns: list[str] | None = None,
    label_column: str | None = None,
):
    """@brief Train and save a random forest classifier from a CSV file.

    @param csv_path Path to the training CSV file.
    @param feature_columns Feature column names; defaults to config.FEATURE_COLUMNS.
    @param label_column Target column name; defaults to config.LABEL_COLUMN.
    @return The fitted random forest classifier.
    @throws ValueError If a required column is missing or no complete rows remain.
    """
    if feature_columns is None:
        feature_columns = config.FEATURE_COLUMNS
    if label_column is None:
        label_column = config.LABEL_COLUMN

    logging.info("Calling train_model in rf_api.py")
    training_data = pd.read_csv(csv_path)

    # check if all the required columns are present in the CSV file
    for col in feature_columns + [label_column]:
        if col not in training_data.columns:
            logging.error(f"Column '{col}' not found in the CSV file.")
            raise ValueError(f"Column '{col}' not found in the CSV file.")

    # loc (from pandas library) - filters out columns that are not required
    # dropna (from pandas library) - filters out rows with empty cells
    complete_training_data = training_data.loc[:, feature_columns + [label_column]].dropna()

    if complete_training_data.empty:
        logging.error("training data is empty")
        raise ValueError("training data is empty")

    # when slicing a pandas dataframe, must use the .loc function
    features = complete_training_data.loc[:, feature_columns].to_numpy(dtype=float)
    labels = complete_training_data[label_column].to_numpy()

    model = RandomForestClassifier(
        number_of_trees=10,
        max_tree_depth=4,
        max_features=0.7,
        random_state=42
    )

    model.fit(features, labels)
    save_model(model)
    logging.info("Trained and saved random forest model")

    # return the trained model object so that it can be used for inference without having to reload it from disk
    return model

def save_model(model: RandomForestClassifier):
    """@brief Serialize a trained random forest classifier to MODEL_PATH.

    @param model The classifier to save.
    @return The path to the saved model file.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # serializes (saves) your trained Python model object from RAM onto your hard drive
    joblib.dump(model, MODEL_PATH)
    logging.info("Saved model to %s", MODEL_PATH)
    return MODEL_PATH

def run_inference(model: RandomForestClassifier, input_data: pd.DataFrame):
    """@brief Predict labels for the supplied feature rows.

    @param model The fitted random forest classifier.
    @param input_data DataFrame containing model features in training column order.
    @return A NumPy array containing one prediction per input row.
    """
    input_features = input_data.to_numpy(dtype=float)
    predictions = model.predict(input_features)

    # returns a numpy array of predictions (0 or 1) for each row in the input data
    return predictions 

def load_model():
    """@brief Load the classifier saved at MODEL_PATH.

    @return The deserialized random forest classifier.
    @throws FileNotFoundError If no saved model exists at MODEL_PATH.
    """
    if not MODEL_PATH.is_file():
        logging.error("No trained model found at %s. Call train_model first.", MODEL_PATH)
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Call train_model first.")

    # deserializes (loads) your trained Python model object from your hard drive into RAM
    return joblib.load(MODEL_PATH)

def main():
    logging.basicConfig(
        filename=config.LOG_PATH,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    training_csv = (
        project_root / "data" / "synthetic" / "driver_monitoring_synthetic_test.csv"
    )
    train_model(training_csv)
    model = load_model()
    X_test = pd.DataFrame([[0.5, 0.3, 15, 200, 50, 2, 10, 5, 0]])
    predictions = run_inference(model, X_test)
    logging.info("Predictions: %s", predictions)

if __name__== "__main__":
    main()