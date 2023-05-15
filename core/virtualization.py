from math import ceil
from PrettyPrint import PrettyPrintTree
from core.model import DecisionTree
from core.decisiontree import DecisionTreeID3
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def virtualize_my_tree(tree: DecisionTree, out: str):
    printer = PrettyPrintTree(
        lambda node: node.children,
        lambda node: node.to_json(tree.criterion),
        return_instead_of_print=True,
        color=None,
        border=True,
    )
    tree_str = printer(tree.root)
    with open(out, mode="w", encoding="utf-8") as file:
        file.write(tree_str)


def virtualize_multibranch_tree(tree: DecisionTreeID3, out: str):
    printer = PrettyPrintTree(
        lambda node: node.children,
        lambda node: node.to_json(),
        return_instead_of_print=True,
        color=None,
        border=True,
    )
    tree_str = printer(tree.root)
    with open(out, mode="w", encoding="utf-8") as file:
        file.write(tree_str)


def virtualize_sklearn_tree(tree: DecisionTreeClassifier, out: str):
    depth = tree.get_depth()
    width, height = ceil(depth * 5), ceil(depth * 3.5)
    classes = [str(_class) for _class in tree.classes_]
    plt.figure(figsize=(width, height))
    plot_tree(
        tree,
        feature_names=tree.feature_names_in_,
        filled=True,
        rounded=True,
        class_names=classes
    )
    plt.savefig(out)
    plt.close()


def show_confusion_matrix(y_true: list, y_pred: list):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()
