import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("featureEngineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

filepath = os.path.join(log_dir, "featureEngineering.log")
file_handler = logging.FileHandler(filepath)
file_handler.setLevel("DEBUG")

Formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(Formatter)
file_handler.setFormatter(Formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(filepath: str) -> pd.DataFrame:
    """Loading Data"""
    try:
        df = pd.read_csv(filepath)
        df.fillna("", inplace=True)
        logger.debug("data loaded and nan filled from %s", filepath)
        return df
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error("unknown error occurred %s", e)
        raise


def applyTfIdf(
    train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int
) -> tuple:
    """Applying tfidf to the data"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train = train_data["text"].values
        y_train = train_data["target"].values
        X_test = test_data["text"].values
        y_test = test_data["target"].values
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.fit_transform(X_test)
        train_df = pd.DataFrame(X_train_bow.todense())
        train_df["label"] = y_train
        test_df = pd.DataFrame(X_test_bow.todense())
        test_df["label"] = y_test
        logger.debug("data transform + tfidf applied")
        return train_df, test_df

    except Exception as e:
        logger.error("unknown error occurred while applying tfidf %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save data to csv file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)
    except Exception as e:
        logger.error("unknown error occurred while saving the data: %s", e)
        raise


def main():
    try:
        max_features = 500
        train_data = load_data("./data/interim/train_processed.csv")
        test_data = load_data("./data/interim/test_processed.csv")
        train_df, test_df = applyTfIdf(train_data, test_data, max_features)
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error("Some error occurred in featureEngineering pipeline %s", e)
        raise


if __name__ == "__main__":
    main()
