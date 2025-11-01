import nltk
import os
import logging
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pandas as pd
from sklearn.preprocessing import LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
file_path = os.path.join(logs_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def text_cleaning(text):
    """Input text converted to lower case,tokenized,stopwords and punctuations removed, stemming done"""
    text = text.lower()  # lowercase
    text = nltk.word_tokenize(text)  # tokenization
    text = [
        word for word in text if word.isalnum()
    ]  # remove non alpha numeric characters
    stop_words = set(stopwords.words("english"))
    text = [
        word
        for word in text
        if word not in stop_words and word not in string.punctuation
    ]  # remove stop words and puctutaions
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]  # applied stemming
    return " ".join(text)


def preprocessing(df: pd.DataFrame, text_column="text", target_column="target"):
    """encodes target col, removes duplicate, transform the text column"""
    try:
        logger.debug("Starting preprocessing of dataframe")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(
            df[target_column]
        )  # encoding the target column into 0/1
        logger.debug("Label encoding completed")
        duplicate = df.duplicated().sum()
        logger.debug('total duplicates %s',duplicate)
        # remove duplicates rows
        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicate rows removed and now len of df is %s",len(df))

        # text cleaning
        df.loc[:, text_column] = df[text_column].apply(text_cleaning)
        logger.debug("Text cleaning completed")
        return df
    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise


def main(text_column="text", target_column="target"):
    try:
        train_data = pd.read_csv("./data/initial/train.csv")
        test_data = pd.read_csv("./data/initial/test.csv")
        logger.debug("Train and test data loaded")

        train_transformed = preprocessing(train_data, text_column, target_column)
        test_transformed = preprocessing(test_data, text_column, target_column)
        logger.debug("Shape after preprocessing is: %s", str(train_transformed.shape))
        logger.debug("Shape after preprocessing is: %s", str(test_transformed.shape))
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        train_transformed.to_csv(
            os.path.join(data_path, "train_processed.csv"), index=False
        )
        test_transformed.to_csv(
            os.path.join(data_path, "test_processed.csv"), index=False
        )
        logger.debug("Preprocessed data saved to interim folder")
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)


if __name__ == "__main__":
    main()
