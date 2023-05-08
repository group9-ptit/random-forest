from typing import List
import math


def log2(x: float) -> float:
    return math.log2(x) if x != 0 else 0


def entropy(probabilities: List[float]) -> float:
    h = 0
    for p in probabilities:
        h -= p * log2(1 / p if p > 0 else 0)
    return h
