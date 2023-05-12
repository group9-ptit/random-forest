import pandas as pd
from sklearn import preprocessing, model_selection


def read_csv(filepath: str):
    return pd.read_csv(filepath)


def encode_attributes(df: pd.DataFrame):
    for column in df.columns:
        values = df[column]
        if values.dtype == object or values.dtype == bool:
            encoder = preprocessing.LabelEncoder()
            df[column] = encoder.fit_transform(values)
    return df


def separate_dataset(df: pd.DataFrame, label: str):
    y = df[label]
    X = df.drop(labels=[label], axis=1)
    return X, y


def train_test_split(X: pd.DataFrame, y: pd.DataFrame, train_size: float):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, train_size=train_size)
    return {
        'sklearn_input': (X_train, X_test, y_train, y_test),
        'my_input': (X_train.to_dict('records'), X_test.to_dict('records'), list(y_train), list(y_test))
    }
