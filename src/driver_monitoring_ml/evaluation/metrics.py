import numpy as np
import logging

logging.basicConfig(
    filename="outputs/logs/model.log",
    filemode='a',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("in metrics.py")


# returns (TP, TN, FP, FN)
def derive_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    """
    True Positives (TP): Ground truth is 1 and model predicts 1
    True Negatives (TN): Ground truth is 0 and model predicts 0
    False Positives (FP): Ground truth is 0 but model predicts 1 (False Alarm)
    False Negatives (FN): Ground truth is 1 but model predicts 0 (Missed Drowsiness)
    """

    tp_count = np.sum((y_true == 1) & (y_pred == 1))
    tn_count = np.sum((y_true == 0) & (y_pred == 0))
    fp_count = np.sum((y_true == 0) & (y_pred == 1))
    fn_count = np.sum((y_true == 1) & (y_pred == 0))

    logging.info(f"tp count is {tp_count}")
    logging.info(f"tn count is {tn_count}")
    logging.info(f"fp count is {fp_count}")
    logging.info(f"fn count is {fn_count}")

    return (tp_count, tn_count, fp_count, fn_count)

# fraction of predictions that were correct
def accuracy_score(tp_count: int, tn_count: int, total_samples: int):
    if total_samples == 0:
        raise RuntimeError("there are no samples in dataset")
    
    res = (tp_count + tn_count) / float(total_samples)
    logging.info(f"accuracy is {res}")
    return res

# of the drowsy predictions, how many predictions were actually drowsy
def precision_score(tp_count: int, fp_count: int):
    if tp_count + fp_count == 0:
        raise RuntimeError("model triggered 0 drowsiness alarms")
    
    res = float(tp_count) / (tp_count + fp_count)
    logging.info(f"precision is {res}")
    return res

# of all the truly drowsy samples, how many did the model catch
def recall_score(tp_count: int, fn_count: int):
    if tp_count + fn_count == 0:
        raise RuntimeError("there are no positive labels in the dataset")
    
    res = float(tp_count) / (tp_count + fn_count)
    logging.info(f"recall is {res}")
    return res

# harmonic mean of precision and recall i.e. it punishes imbalance between the precision and recall score
def f1_score(precision_score: float, recall_score: float):
    if precision_score + recall_score == 0:
        raise RuntimeError("division by 0 error in f1_score function")

    res = ((precision_score*recall_score) / (precision_score + recall_score))*2
    logging.info(f"f1 score is {res}")
    return res

