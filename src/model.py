from abc import ABC, abstractmethod
from copy import deepcopy
from src.dataset import Dataset
from src.type import DatasetRow
from typing import List, Dict, Optional


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X: List[DatasetRow], Y: List[str]) -> None:
        """Huấn luyện mô hình với tập dữ liệu X và nhãn tương ứng Y"""
        pass

    @abstractmethod
    def predict(self, X: List[DatasetRow]) -> List[str]:
        """Dự đoán nhãn của tập dữ liệu X"""
        pass


class TreeNode:
    def __init__(self, attribute: str, children: Dict[str, 'TreeNode'], label: Optional[str] = None) -> None:
        self.attribute = attribute
        self.children = children
        self.label = label

    def is_leaf(self) -> bool:
        return self.label is not None


class DecisionTree(Model):
    """Cây quyết định sử dụng thuật toán ID3 cho bài toán phân lớp"""

    def __init__(self) -> None:
        super().__init__()
        self.root = None

    def fit(self, X: List[DatasetRow], Y: List[str]) -> None:
        dataset = Dataset(X, Y)
        self.root = self.__build_tree(dataset)

    def __build_tree(self, dataset: Dataset, depth=0) -> TreeNode:
        split_attribute = dataset.best_splitter()
        attribute_values = dataset.values(split_attribute)

        children = {}
        for attribute_value in attribute_values:
            new_dataset = dataset.sub_dataset(split_attribute, attribute_value)
            child_node = None

            if new_dataset.is_single_label():
                clone_label_values = deepcopy(new_dataset.label_values)
                label = clone_label_values.pop()
                child_node = TreeNode(split_attribute, {}, label)
            else:
                child_node = self.__build_tree(new_dataset, depth + 1)

            children[attribute_value] = child_node

        return TreeNode(split_attribute, children)

    def predict(self, X: List[DatasetRow]) -> List[str]:
        return [self.predict_one(x) for x in X]

    def predict_one(self, x: DatasetRow) -> str:
        node = self.root
        while not node.is_leaf():
            attribute = node.attribute
            attribute_value = x[attribute]
            node = node.children[attribute_value]
        return node.label


class RandomForest(Model):
    def __init__(self, n_estimator=30) -> None:
        super().__init__()
        self.n_estimator = n_estimator

    def fit(self, X: List[DatasetRow], Y: List[str]) -> None:
        return super().fit(X, Y)

    def predict(self, X: List[DatasetRow]) -> List[str]:
        return super().predict(X)
