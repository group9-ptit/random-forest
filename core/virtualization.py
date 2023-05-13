from math import ceil
from PrettyPrint import PrettyPrintTree
from core.model import DecisionTree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt


def virtualize_my_tree(tree: DecisionTree, out: str):
    printer = PrettyPrintTree(
        lambda node: node.children,
        lambda node: node.value,
        return_instead_of_print=True,
        color=None,
        border=True,
    )
    tree_str = printer(tree.root)
    with open(out, mode='w') as file:
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
