import logging
import os
import pandas as pd

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)  # ensuring log_directory exists
logger = logging.getLogger(
    "DataIngestion"
)  # logging use karke ek logger object banaya jiska naam diya DataIngestion
logger.setLevel("DEBUG")  # set level to debug , this will show the other levels too
console_handler = logging.StreamHandler()  # made an object named console_handler using StreamHandler function, this helps printing our logs directly in the terminal. Logger->handler->console_handler
console_handler.setLevel("DEBUG")
filepath = os.path.join(
    log_dir, "Data_ingestion.log"
)  # Creating a file path for printing our logs in a file
file_handler = logging.FileHandler(
    filepath
)  # file_handler will print our logs directly in the file
file_handler.setLevel("DEBUG")

# handler done  now formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)  # its just the format in which my logs will be printed.
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# LOGGING PART DONE


def load_data(file_url: str) -> pd.DataFrame:
    """Load data from a csv file"""
    try:
        df = pd.read_csv(file_url)
        logger.debug("data loaded from %s", file_url)
        return df
    except pd.errors.ParserError as e:
        logger.debug("Had an error while parsing %s", e)
        raise
    except Exception as e:
        logger.debug("unexpected error occured %s", e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
        # Check for duplicates after renaming columns
        num_duplicates = df.duplicated().sum()
        print(f"Number of duplicates: {num_duplicates}")
        logger.debug(f"Number of duplicates: {num_duplicates}")
        # Drop duplicates to match notebook behavior
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates dropped")
        logger.debug("Preprocessing done")
        return df
    except KeyError as e:
        logger.debug("Had some missing key %s", e)
        raise
    except Exception as e:
        logger.debug("unexpected error occurred %s", e)
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_file_path: str
) -> None:
    try:
        data_path = os.path.join(data_file_path, "initial")
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"))
        test_data.to_csv(os.path.join(data_path, "test.csv"))
        logger.debug("Test and train csv files have been saved")

    except Exception as e:
        logger.debug("unexpected error occured %s", e)
        raise


def main():
    data_path = (
        r"D:\Learning MLOPS\MLOPS-Learning-\complete pipeline\experiments\spam.csv"
    )
    logger.debug("starting data ingestion")
    df = load_data(data_path)
    df = preprocess_data(df)
    logger.debug("Data shape after preprocessing: %s", str(df.shape))

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=0.2, random_state=2)

    save_data(train, test, data_file_path="./data")
    logger.debug("Data ingestion pipeline completed successfully")


if __name__ == "__main__":
    main()
