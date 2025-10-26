import logging
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

logdir = "logs"
os.makedirs(logdir, exist_ok=True)

logger = logging.getLogger("modelBuilding")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

filepath = os.path.join(logdir, "modelBuilding.log")
file_handler = logging.FileHandler(filepath)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logger.debug("data loaded from %s", filepath)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error("unknown error occurred %s", e)
        raise


def train(X_train: np.ndarray, y_train: np.ndarray):
    try:
        logger.debug("RandomForest initialized")
        classifier = RandomForestClassifier()
        logger.debug("Training starting with .fit with %d params", X_train.shape[0])
        classifier.fit(X_train, y_train)
        logger.debug("Training completed")
        return classifier
    except Exception as e:
        logger.error("Some error occurred during model training %s", e)
        raise


def save_model(model, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("file saved %s", file_path)
    except Exception as e:
        logger.error("Error occured while saving the file %s", e)
        raise


def main():
    try:
        train_data = load_data("./data/processed/train_tfidf.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        classifier = train(X_train=X_train, y_train=y_train)
        save_path = "models/model.pkl"
        save_model(classifier, save_path)
    except Exception as e:
        logger.error("some error occurred in model building %s", e)
        raise
