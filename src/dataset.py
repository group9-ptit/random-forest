import copy
import csv
import math
import logging
import statistics
from typing import List, Tuple
from collections import Counter
from src.helper import entropy
from src.type import Record


class Dataset:
    def __init__(self, records: List[Record], labels: List[str]) -> None:
        self.records = records
        self.attributes = set(records[0].keys())
        self.labels = labels
        self.shape = (len(records), len(self.attributes))

    def best_splitter(self) -> Tuple[str, float, 'Dataset', 'Dataset']:
        """Tìm thuộc tính có khả năng phân loại tốt nhất và ngưỡng giá trị của nó"""
        ES = self.__entropy()
        _attribute, _threshold, _lte, _gt = None, 0, None, None
        max_gain = -math.inf

        for attribute in self.attributes:
            split_point, loss, lte_dataset, gt_dataset = self.__best_split_point(
                attribute)
            information_gain = ES - loss
            if information_gain > max_gain:
                _threshold = split_point
                _attribute = attribute
                _lte = lte_dataset
                _gt = gt_dataset

        logging.debug(f'Best splitter: IG[{_attribute}] = {max_gain}')

        return _attribute, _threshold, _lte, _gt

    def __entropy(self):
        freq = Counter(self.labels)
        n_row = self.shape[0]
        probabilities = [value / n_row for value in freq.values()]
        return entropy(probabilities)

    def __best_split_point(self, attribute: str) -> Tuple[float, float, 'Dataset', 'Dataset']:
        """Tìm ngưỡng phân chia tốt nhất với thuộc tính `attribute`"""
        split_point, min_loss, lte, gt = None, math.inf, None, None
        values = [record[attribute] for record in self.records]
        sorted_values = values.sort()

        for i in range(1, len(sorted_values) - 1):
            left = sorted_values[i]
            right = sorted_values[i + 1]

            if left == right:
                continue

            threshold = (left + right) / 2
            lte_dataset, gt_dataset = self.__split(attribute, threshold)
            loss = lte_dataset.__entropy() + gt_dataset.__entropy()

            if loss < min_loss:
                min_loss = loss
                split_point = threshold
                lte = lte_dataset
                gt = gt_dataset

        return split_point, min_loss, lte, gt

    def __split(self, attribute: str, threshold: float):
        """Chia dữ liệu thành hai tập dựa trên thuộc tính `attribute` và ngưỡng `threshold`"""
        n_row = self.shape[0]

        lte_records = []
        lte_labels = []

        gt_records = []
        gt_label = []

        for i in range(n_row):
            if self.records[i][attribute] <= threshold:
                lte_records.append(self.records[i])
                lte_labels.append(self.labels[i])
            else:
                gt_records.append(self.records[i])
                gt_label.append(self.labels[i])

        return Dataset(lte_records, lte_labels), Dataset(gt_records, gt_label)

    def same_class(self):
        return set(self.labels).__sizeof__() == 1

    def most_common_label(self):
        return statistics.mode(self.labels)
