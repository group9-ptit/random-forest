import csv
import math
from copy import deepcopy
from src.type import CsvRow, CsvRowWithoutLabel, Record, List, Tuple
from sklearn.preprocessing import LabelEncoder


def log2(x: float) -> float:
    return math.log2(x) if x != 0 else 0


def entropy(probabilities: List[float]) -> float:
    h = 0
    for p in probabilities:
        h -= p * log2(p if p > 0 else 0)
    return h


def read_csv(filepath: str, delimiter=",") -> List[CsvRow]:
    with open(filepath, encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        return [row for row in reader]


def separate_dataset(data: List[CsvRow], label: str) -> Tuple[List[CsvRowWithoutLabel], List[str]]:
    """Phân chia dữ liệu thành hai phần thuộc tính và nhãn"""
    X, Y = [], []
    for row in deepcopy(data):
        _label = row.pop(label)
        X.append(row)
        Y.append(_label)
    return X, Y


def encode_attributes(data: List[CsvRowWithoutLabel]) -> List[Record]:
    """Mã hoá thuộc tính rời rạc thành liên tục"""
    attribute_values = {}

    for row in data:
        for key, value in row.items():
            values = attribute_values.get(key, [])
            values.append(value)
            attribute_values.update([(key, values)])

    for attribute, values in attribute_values.items():
        encoder = LabelEncoder()
        if type(values[0]) == str:
            encoded_values = list(encoder.fit_transform(values))
            attribute_values.update([(attribute, encoded_values)])

    encoded_rows = []
    for i in range(data.__len__()):
        encoded_row = {}
        for key, encoded_values in attribute_values.items():
            encoded_row[key] = encoded_values[i]
        encoded_rows.append(encoded_row)

    return encoded_rows
