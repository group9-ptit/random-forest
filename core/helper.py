import math
import time
import uuid
from core.type import List
import pandas as pd
from sklearn import preprocessing, model_selection


def random_id(length=9):
    return uuid.uuid4().hex[:length]


def now():
    return time.time()


def log2(x: float) -> float:
    return math.log2(x) if x != 0 else 0


def entropy(probabilities: List[float]) -> float:
    h = 0
    for p in probabilities:
        h -= p * log2(p if p > 0 else 0)
    return h


def gini(probabilities: List[float]) -> float:
    g = 1
    for p in probabilities:
        g -= p ** 2
    return g


def read_csv(filepath: str, unique_rows=False):
    df = pd.read_csv(filepath)
    if unique_rows:
        df.drop_duplicates(inplace=True)
    return df


def write_csv(df: pd.DataFrame, filepath: str):
    df.to_csv(filepath, encoding="utf-8", index=False)


def encode_attributes(df: pd.DataFrame):
    for column in df.columns:
        values = df[column]
        if values.dtype == object or values.dtype == bool:
            encoder = preprocessing.LabelEncoder()
            df[column] = encoder.fit_transform(values)
    return df


def separate_dataset(df: pd.DataFrame, label: str):
    y = df[label]
    X = df.drop(columns=[label], axis=1)
    return X, y


def train_test_split(X: pd.DataFrame, y: pd.Series, train_size: float):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=train_size)
    return {
        "sklearn_input": (X_train, X_test, y_train, y_test),
        "my_input": (X_train.to_dict("records"), X_test.to_dict("records"), list(y_train), list(y_test))
    }


def train_test_split_with_multibranch(X: pd.DataFrame, y: pd.Series, train_size: float):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=train_size)
    return {
        "sklearn_input": (X_train, X_test, y_train, y_test),
        "my_input": (X_train.to_dict("records"), X_test.to_dict("records"), list(y_train), list(y_test)),
        "multibranch_input": (
            X_train.values.tolist(),
            X_test.values.tolist(),
            list(y_train),
            list(y_test),
            [i for i in range(X.shape[1])]
        )
    }
