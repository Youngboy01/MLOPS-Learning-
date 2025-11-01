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
    except FileNotFoundError as e:
        logger.error("file not found %s", e)
        raise
    except Exception as e:
        logger.error("unknown error occurred while loading data %s", e)
        raise


def train(
    X_train: np.ndarray, y_train: np.ndarray, params: dict
) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("NO of samples in X_train and y_train are not same")
        logger.debug("RandomForest initialized with params %s", params)
        classifier = RandomForestClassifier(
            n_estimators=params["n_estimators"], random_state=params["random_state"]
        )
        logger.debug("Training starting with .fit with %d params", X_train.shape[0])
        classifier.fit(X_train, y_train)
        logger.debug("Training completed")
        return classifier
    except ValueError as e:
        logger.error("Value error in training %s", e)
        raise
    except Exception as e:
        logger.error("Some error occurred during model training %s", e)
        raise


def save_model(model, file_path: str):
    """save the trained model"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("file saved at %s", file_path)
    except FileNotFoundError as e:
        logger.error("filepath not found %s", e)
        raise
    except Exception as e:
        logger.error("Error occured while saving the file %s", e)
        raise


def main():
    try:
        params = {"n_estimators": 50, "random_state": 2}
        train_data = load_data("./data/processed/train_tfidf.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = np.array(train_data.iloc[:, -1].values)
        classifier = train(X_train=X_train, y_train=y_train, params=params)
        save_path = "models/model.pkl"
        save_model(classifier, save_path)
    except Exception as e:
        logger.error("some error occurred in model building %s", e)
        raise


if __name__ == "__main__":
    main()
