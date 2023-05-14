from abc import ABC, abstractmethod
from pandas import DataFrame, Series
from prettytable import PrettyTable
from core.model import DecisionTree, RandomForest
from core.type import Optional, Union, Dict, Measure
from core.helper import train_test_split, now, random_id
from core.virtualization import virtualize_my_tree, virtualize_sklearn_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TestCase(ABC):
    def __init__(
        self,
        X: DataFrame,
        Y: Series,
        train_size=0.8,
        min_samples_split=2,
        max_depth: Optional[int] = None,
        criterion: Measure = "entropy"
    ) -> None:
        _input = train_test_split(X, Y, train_size)
        self.my_input = _input["my_input"]
        self.sklearn_input = _input["sklearn_input"]
        self.train_size = train_size
        self.meta_params = {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "criterion": criterion
        }
        self.id = random_id()
        self.my_result = {}
        self.sklearn_result = {}

    def run(self) -> None:
        print(f"-------TestID: {self.id}-------")
        self.run_my_model()
        self.run_sklearn_model()

    @abstractmethod
    def run_my_model(self):
        pass

    @abstractmethod
    def run_sklearn_model(self):
        pass

    def run_metrics(self, y_test, y_pred) -> Dict[str, float]:
        return {
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "F1": round(f1_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3)
        }

    def print_result(self):
        header = " | ".join([f"{key}={value}" for key, value in self.meta_params.items()])
        print(header)
        table = PrettyTable([
            "Model",
            "Train Duration",
            "Accuracy",
            "F1",
            "Precision",
            "Recall"
        ])
        table.add_rows([
            ["Mine", self.my_result["train_duration"], *self.my_result["metrics"].values()],
            ["Sklearn", self.sklearn_result["train_duration"], *self.sklearn_result["metrics"].values()]
        ])
        print(table)
        print()


class DecisionTreeTestCase(TestCase):
    def run_my_model(self):
        X_train, X_test, y_train, y_test = self.my_input
        tree = DecisionTree(**self.meta_params)
        start_train = now()
        tree.fit(X_train, y_train)
        end_train = now()
        y_pred = tree.predict(X_test)
        self.my_result = {
            "model": tree,
            "metrics": self.run_metrics(y_test, y_pred),
            "train_duration": round(end_train - start_train, 2),
        }

    def run_sklearn_model(self):
        X_train, X_test, y_train, y_test = self.sklearn_input
        tree = DecisionTreeClassifier(**self.meta_params)
        start_train = now()
        tree.fit(X_train, y_train)
        end_train = now()
        y_pred = tree.predict(X_test)
        self.sklearn_result = {
            "model": tree,
            "metrics": self.run_metrics(y_test, y_pred),
            "train_duration": round(end_train - start_train, 2),
        }

    def export_tree(self):
        my_tree = self.my_result["model"]
        sklearn_tree = self.sklearn_result["model"]
        virtualize_my_tree(my_tree, f"out/{self.id}.txt")
        virtualize_sklearn_tree(sklearn_tree, f"out/{self.id}.jpeg")


class RandomForestTestCase(TestCase):
    def __init__(
        self,
        X: DataFrame,
        Y: Series,
        criterion: Measure = "entropy",
        n_estimators=30,
        train_size=0.8,
        min_samples_split=2,
        n_jobs=2,
        max_samples: Optional[Union[int, float]] = None,
        max_depth: Optional[int] = None
    ) -> None:
        super().__init__(X, Y, train_size, min_samples_split, max_depth, criterion)
        self.meta_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "max_samples": max_samples,
            "n_jobs": n_jobs
        }

    def run_my_model(self):
        X_train, X_test, y_train, y_test = self.my_input
        forest = RandomForest(**self.meta_params)
        start_train = now()
        forest.fit(X_train, y_train)
        end_train = now()
        y_pred = forest.predict(X_test)
        self.my_result = {
            "model": forest,
            "metrics": self.run_metrics(y_test, y_pred),
            "train_duration": round(end_train - start_train, 2),
        }

    def run_sklearn_model(self):
        X_train, X_test, y_train, y_test = self.sklearn_input
        forest = RandomForestClassifier(
            max_features=None,
            **self.meta_params
        )
        start_train = now()
        forest.fit(X_train, y_train)
        end_train = now()
        y_pred = forest.predict(X_test)
        self.sklearn_result = {
            "model": forest,
            "metrics": self.run_metrics(y_test, y_pred),
            "train_duration": round(end_train - start_train, 2),
        }
