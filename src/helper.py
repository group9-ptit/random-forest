import csv
import math
from copy import deepcopy
from typing import List, Tuple
from src.type import CsvRow, CsvRowWithoutLabel


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
    X, Y = [], []
    for row in deepcopy(data):
        _label = row.pop(label_field)
        X.append(row)
        Y.append(_label)
    return X, Y
