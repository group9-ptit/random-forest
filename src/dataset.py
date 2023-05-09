from typing import List, Any, Set
from collections import Counter
from src.helper import entropy
from src.type import DatasetRow
import logging


class Dataset:
    def __init__(self, records: List[DatasetRow], labels: List[str]) -> None:
        self.records = records
        self.attributes = set(records[0].keys())
        self.labels = labels
        self.label_values = set(labels)
        self.shape = (len(records), len(self.attributes))

    def sub_dataset(self, attribute: str, value: Any) -> 'Dataset':
        """Tạo tập dữ liệu mới đối với thuộc tính `attribute` có giá trị `value`"""
        n_row = self.shape[0]

        _records = []
        _labels = []

        for i in range(n_row):
            if self.records[i][attribute] == value:
                clone_record = self.records[i].copy()
                clone_record.pop(attribute)
                _records.append(clone_record)
                _labels.append(self.labels[i])

        return Dataset(_records, _labels)

    def values(self, attribute: str) -> Set[str]:
        """Tìm tập giá trị của thuộc tính `attribute`"""
        values = set()
        for record in self.records:
            values.add(record[attribute])
        return values

    def entropy(self):
        freq = Counter(self.labels)
        n_row = self.shape[0]
        probabilities = [value / n_row for value in freq.values()]
        return entropy(probabilities)

    def best_splitter(self) -> str:
        """Tìm thuộc tính có khả năng phân loại tốt nhất"""
        ES = self.entropy()
        max_ig, max_attribute = 0, None

        for attribute in self.attributes:
            information_gain = ES

            for value in self.values(attribute):
                sub_dataset = self.sub_dataset(attribute, value)
                information_gain -= (sub_dataset.shape[0] /
                                     self.shape[0]) * sub_dataset.entropy()

            logging.debug(f'IG[{attribute}] = {information_gain}')

            if information_gain > max_ig:
                max_ig = information_gain
                max_attribute = attribute

        logging.debug(f'Best splitter: IG[{max_attribute}] = {max_ig}')

        return max_attribute

    def is_single_label(self):
        return len(self.label_values) == 1
