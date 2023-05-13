import logging
import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from src.datastructure import TreeNode, Dataset
from src.type import Record, Label, List, Optional, Union, Tuple


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


class DecisionTree(Model):
    """Cây quyết định sử dụng thuật toán ID3 cho bài toán phân lớp"""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2
    ) -> None:
        super().__init__()
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.depth = 0

    def fit(self, X: List[Record], Y: List[Label]) -> None:
        start_time = time.time()
        dataset = Dataset(X, Y)
        self.root = self.__build_tree(dataset)
        end_time = time.time()
        logging.debug(f'[DECISION_TREE]: Duration = {end_time - start_time}')

    def __build_tree(self, dataset: Dataset, depth=0) -> TreeNode:
        # Kiểm tra điều kiện dừng xem có thể cắt cành luôn không
        if self.__can_stop(dataset, depth):
            self.depth = depth if depth > self.depth else self.depth
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


class RandomForest(Model):
    """Rừng ngẫu nhiên xây dựng bởi cây cây quyết định sử dụng ID3"""

    def __init__(
        self,
        n_estimators=100,
        min_samples_split=2,
        n_jobs=2,
        max_depth: Optional[int] = None,
        max_samples: Optional[Union[int, float]] = None,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.trees = []

    def fit(self, X: List[Record], Y: List[Label]) -> None:
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            trees = executor.map(
                lambda args: self.__build_tree(*args),
                [(X, Y) for _ in range(self.n_estimators)]
            )
        self.trees = list(trees)
        end_time = time.time()
        logging.debug(f'[RANDOM_FOREST]: Duration = {end_time - start_time}')

    def __build_tree(self, X: List[Record], Y: List[Label]) -> DecisionTree:
        X_samples, Y_samples = self.__bootstrap_sampling(X, Y)
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        tree.fit(X_samples, Y_samples)
        return tree

    def __bootstrap_sampling(self, X: List[Record], Y: List[Label]) -> Tuple[List[Record], List[Label]]:
        X_samples = []
        Y_samples = []

        n_records = X.__len__()

        n_samples = n_records
        if type(self.max_samples) == int:
            n_samples = self.max_samples
        elif type(self.max_samples) == float:
            n_samples = int(self.max_samples * n_samples)

        for _ in range(n_samples):
            index = random.randint(0, n_records - 1)
            X_samples.append(X[index])
            Y_samples.append(Y[index])

        return X_samples, Y_samples

    def predict_one(self, x: Record) -> Label:
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            labels = executor.map(
                lambda args: self.__predict_one(*args),
                [(tree, x) for tree in self.trees]
            )
        sorted_most_common = sorted(
            Counter(labels).most_common(), key=lambda x: x[0])
        return sorted_most_common[0][0]

    def __predict_one(self, tree: DecisionTree, x: Record) -> Label:
        return tree.predict_one(x)
