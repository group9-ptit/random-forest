import json
import statistics
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Dict
from src.dataset import Dataset
from src.type import Record, Label


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X: List[Record], Y: List[Label]) -> None:
        """Huấn luyện mô hình với tập dữ liệu X và nhãn tương ứng Y"""
        pass

    def predict(self, X: List[Record]) -> List[Label]:
        """Dự đoán nhãn của tập dữ liệu X"""
        return [self.predict_one(x) for x in X]

    @abstractmethod
    def predict_one(self, x: Record) -> Label:
        """Dự đoán nhãn của bản ghi x"""
        pass


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
        o = {
            'entropy': self.dataset.entropy,
            'value': self.dataset.label_counter,
            'samples': self.dataset.samples
        }
        if self.is_leaf():
            o['label'] = self.label
        else:
            o['threshold'] = self.threshold
            o['attribute'] = self.attribute
            o['left'] = self.left.to_dict()
            o['right'] = self.right.to_dict()

        return o


class DecisionTree(Model):
    """Cây quyết định sử dụng thuật toán ID3 cho bài toán phân lớp"""

    def __init__(self, max_depth: Optional[int] = None, min_samples_split: int = 2) -> None:
        super().__init__()
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: List[Record], Y: List[Label]) -> None:
        dataset = Dataset(X, Y)
        self.root = self.__build_tree(dataset)

    def __build_tree(self, dataset: Dataset, depth=0) -> TreeNode:
        # Kiểm tra điều kiện dừng xem có thể cắt cành luôn không
        if self.__can_stop(dataset, depth):
            return TreeNode(dataset, label=dataset.most_common_label())

        # Tìm thuộc tính và ngưỡng phân chia tối ưu nhất dựa trên IG
        attribute, threshold, lte_dataset, gt_dataset = dataset.best_splitter()

        # Xây cây con trái dựa trên tập dữ liệu nhỏ hơn hoặc bằng ngưỡng
        left = self.__build_tree(lte_dataset, depth + 1)

        # Xây cây con phải dựa trên tập dữ liệu lớn hơn ngưỡng
        right = self.__build_tree(gt_dataset, depth + 1)

        return TreeNode(dataset, attribute, threshold, left, right)

    def __can_stop(self, dataset: Dataset, current_depth: int) -> bool:
        if (self.max_depth is not None) and (self.max_depth == current_depth):
            return True
        if dataset.samples < self.min_samples_split:
            return True
        return dataset.same_class()

    def predict_one(self, x: Record) -> Label:
        node = self.root
        while not node.is_leaf():
            attribute = node.attribute
            threshold = node.threshold
            attribute_value = x[attribute]
            node = node.left if attribute_value <= threshold else node.right
        return node.label

    def write_tree(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(
                self.root,
                file,
                default=lambda o: o.to_dict(),
                sort_keys=True,
                indent=2
            )


class RandomForest(Model):
    def __init__(self, n_estimator=100, max_depth: Optional[int] = None, min_samples_split: int = 2, n_jobs=2) -> None:
        super().__init__()
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X: List[Record], Y: List[Label]) -> None:
        # pool = ThreadPoolExecutor(max_workers=self.n_jobs)
        # i = 0
        # while i < self.n_estimator:
        #     pool.submit()
        #     tree = self.__build_tree()
        #     self.trees.append(tree)
        #     i += 1
        pass

    def __build_tree(X: List[Record], Y: List[Label]) -> DecisionTree:
        pass

    def predict_one(self, x: Record) -> Label:
        labels = []
        for tree in self.trees:
            label = tree.predict_one(x)
            labels.append(label)
        return statistics.mode(labels)
