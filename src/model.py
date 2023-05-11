import json
from abc import ABC, abstractmethod
from typing import List, Optional
from src.dataset import Dataset
from src.type import Record


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X: List[Record], Y: List[str]) -> None:
        """Huấn luyện mô hình với tập dữ liệu X và nhãn tương ứng Y"""
        pass

    def predict(self, X: List[Record]) -> List[str]:
        """Dự đoán nhãn của tập dữ liệu X"""
        return [self.predict_one(x) for x in X]

    @abstractmethod
    def predict_one(self, x: Record) -> str:
        """Dự đoán nhãn của bản ghi x"""
        pass


class TreeNode:
    def __init__(
        self,
        attribute: Optional[str] = None,
        threshold: Optional[float] = None,
        left: Optional['TreeNode'] = None,
        right: Optional['TreeNode'] = None,
        label: Optional[str] = None
    ) -> None:
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self) -> bool:
        return self.label is not None


class DecisionTree(Model):
    """Cây quyết định sử dụng thuật toán ID3 cho bài toán phân lớp"""

    def __init__(self, max_depth: Optional[int] = None) -> None:
        super().__init__()
        self.root = None
        self.max_depth = max_depth

    def fit(self, X: List[Record], Y: List[str]) -> None:
        dataset = Dataset(X, Y)
        self.root = self.__build_tree(dataset)

    def __build_tree(self, dataset: Dataset, depth=0) -> TreeNode:
        if self.__can_stop(dataset, depth):
            return TreeNode(label=dataset.most_common_label())

        attribute, threshold, lte_dataset, gt_dataset = dataset.best_splitter()

        left = self.__build_tree(lte_dataset, depth + 1)
        right = self.__build_tree(gt_dataset, depth + 1)

        return TreeNode(attribute, threshold, left, right)

    def __can_stop(self, dataset: Dataset, current_depth: int) -> bool:
        if (self.max_depth is not None) and (self.max_depth == current_depth):
            return True
        return dataset.same_class()

    def predict_one(self, x: Record) -> str:
        node = self.root
        while not node.is_leaf():
            attribute = node.attribute
            threshold = node.threshold
            attribute_value = x[attribute]
            node = node.left if attribute_value <= threshold else node.right
        return node.label

    def to_json(self):
        return json.dumps(
            self.root,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=2
        )


class RandomForest(Model):
    def __init__(self, n_estimator=30) -> None:
        super().__init__()
        self.n_estimator = n_estimator

    def fit(self, X: List[Record], Y: List[str]) -> None:
        return super().fit(X, Y)

    def predict_one(self, x: Record) -> str:
        return super().predict_one(x)
