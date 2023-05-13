import json
import math
import logging
from collections import Counter
from core.helper import entropy
from core.type import Optional, Label, Dict, Any, List, Record, Tuple


class Dataset:
    def __init__(self, records: List[Record], labels: List[Label]) -> None:
        self.records = records
        self.attributes = set(records[0].keys())
        self.labels = labels
        self.samples = records.__len__()
        self.label_counter = Counter(labels)

    @property
    def entropy(self):
        samples = self.samples
        freq = self.label_counter.values()
        probabilities = [value / samples for value in freq]
        return entropy(probabilities)

    def best_splitter(self) -> Tuple[str, float, 'Dataset', 'Dataset']:
        """Tìm thuộc tính có khả năng phân loại tốt nhất và ngưỡng giá trị của nó"""
        ES = self.entropy
        _attribute, _threshold, _lte, _gt = None, 0, None, None
        max_gain = -math.inf

        for attribute in self.attributes:
            split_point, loss, lte_dataset, gt_dataset = self.__best_split_point(
                attribute)
            information_gain = ES - loss
            if information_gain > max_gain:
                max_gain = information_gain
                _threshold = split_point
                _attribute = attribute
                _lte = lte_dataset
                _gt = gt_dataset

        logging.debug(f'Best splitter: IG[{_attribute}] = {max_gain}')

        return _attribute, _threshold, _lte, _gt

    def __best_split_point(self, attribute: str) -> Tuple[float, float, 'Dataset', 'Dataset']:
        """Tìm ngưỡng phân chia tốt nhất với thuộc tính `attribute`"""
        split_point, min_loss, lte, gt = None, math.inf, None, None
        values = [record[attribute] for record in self.records]
        values.sort()

        for i in range(0, len(values) - 1):
            left = values[i]
            right = values[i + 1]

            if left == right:
                continue

            threshold = (left + right) / 2
            lte_dataset, gt_dataset = self.__split(attribute, threshold)
            left_loss = (lte_dataset.samples / self.samples) * \
                lte_dataset.entropy
            right_loss = (gt_dataset.samples / self.samples) * \
                gt_dataset.entropy
            loss = left_loss + right_loss

            if loss < min_loss:
                min_loss = loss
                split_point = threshold
                lte = lte_dataset
                gt = gt_dataset

        return split_point, min_loss, lte, gt

    def __split(self, attribute: str, threshold: float):
        """Chia dữ liệu thành hai tập dựa trên thuộc tính `attribute` và ngưỡng `threshold`"""
        n_row = self.samples

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
        return set(self.labels).__len__() == 1

    def most_common_label(self):
        most_common = self.label_counter.most_common()
        sorted_most_common = sorted(most_common, key=lambda x: x[0])
        return sorted_most_common[0][0]


class TreeNode:
    def __init__(
        self,
        dataset: Dataset,
        attribute: Optional[str] = None,
        threshold: Optional[float] = None,
        left: Optional['TreeNode'] = None,
        right: Optional['TreeNode'] = None,
        label: Optional[Label] = None
    ) -> None:
        self.dataset = dataset
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self) -> bool:
        return self.label is not None

    def to_dict(self) -> Dict[str, Any]:
        if self.is_leaf():
            return {
                'entropy': round(self.dataset.entropy, 3),
                'samples': self.dataset.samples,
                'value': self.dataset.label_counter,
                'label': self.label
            }
        return {
            'attribute': self.attribute,
            'threshold': round(self.threshold, 3),
            'entropy': round(self.dataset.entropy, 3),
            'samples': self.dataset.samples,
            'value': self.dataset.label_counter,
        }

    @property
    def children(self):
        rt = []
        if self.left is not None:
            rt.append(self.left)
        if self.right is not None:
            rt.append(self.right)
        return rt

    @property
    def value(self) -> str:
        return json.dumps(self.to_dict(), indent=2)