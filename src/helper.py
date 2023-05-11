import csv
import math
from copy import deepcopy
from typing import List, Tuple
from src.type import CsvRow, CsvRowWithoutLabel, Record
from sklearn.preprocessing import LabelEncoder


def log2(x: float) -> float:
    return math.log2(x) if x != 0 else 0


def entropy(probabilities: List[float]) -> float:
    h = 0
    for p in probabilities:
        h -= p * log2(p if p > 0 else 0)
    return h


def read_csv(filepath: str) -> List[CsvRow]:
    with open(filepath, encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def separate_dataset(data: List[CsvRow], label_field: str) -> Tuple[List[CsvRowWithoutLabel], List[str]]:
    """Phân chia dữ liệu thành hai phần thuộc tính và nhãn"""
    X, Y = [], []
    for row in deepcopy(data):
        _label = row.pop(label_field)
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
        encoded_values = list(encoder.fit_transform(values))
        attribute_values.update([(attribute, encoded_values)])

    encoded_rows = []
    for i in range(data.__len__()):
        encoded_row = {}
        for key, encoded_values in attribute_values.items():
            encoded_row[key] = encoded_values[i]
        encoded_rows.append(encoded_row)

    return encoded_rows
