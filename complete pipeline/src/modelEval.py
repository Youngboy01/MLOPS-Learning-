import logging
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)
import json

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.Logger("modelEval")
logger.setLevel("DEBUG")

console_hanlder = logging.StreamHandler()
console_hanlder.setLevel("DEBUG")
filepath = os.path.join(log_dir, "modelEval.log")
file_handler = logging.FileHandler(filepath)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_hanlder.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_hanlder)
logger.addHandler(file_handler)


def load_data(filepath: str):
    """Load data from csv"""
    try:
        df = pd.read_csv(filepath)
        logger.debug("data has been loaded from file %s", filepath)
        return df
    except pd.errors.ParserError as e:
        logger.error("error while parsing the file %s", e)
        raise
    except Exception as e:
        logger.error("error occurred while loading the data from the file %s", e)
        raise


def load_model(filepath: str):
    """Load the trained model from file"""
    try:
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from file successfully %s", filepath)
        return model
    except FileNotFoundError as e:
        logger.error("file not found %s", e)
        raise
    except Exception as e:
        logger.error("Error while loading the model %s", e)
        raise


def eval_matrices(classifier, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = classifier.predict(X_test)
        # Use probabilities for the positive class (label 1) and keep lengths consistent
        # predict_proba returns shape (n_samples, n_classes); for binary take column of class 1
        y_pred_proba = classifier.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred=y_pred)
        recall = recall_score(y_test, y_pred=y_pred)
        precision = precision_score(y_test, y_pred=y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "auc": auc,
            "f1_score": f1,
        }
        logger.debug("All metrics scores have been properly calculated")
        return metrics
    except Exception as e:
        logger.error("error encountered in eval method %s", e)
        raise


def save_metrics(metrics: dict, filepath: str):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug("metrics saved %s", filepath)
    except Exception as e:
        raise e


def main():
    try:
        classifier = load_model("./models/model.pkl")
        test_data = load_data("./data/processed/test_tfidf.csv")
        X_test = test_data.iloc[:, :-1].values
        y_test = np.array(test_data.iloc[:, -1].values)
        metrics = eval_matrices(classifier, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
    except Exception as e:
        logger.error("ERROR %s", e)
        print(f"error:{e}")


if __name__ == "__main__":
    main()
