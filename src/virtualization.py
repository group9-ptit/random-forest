from src.model import DecisionTree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from math import ceil

def virtualize_my_tree(tree: DecisionTree, out: str):
    return tree.write_tree(out)


def virtualize_sklearn_tree(tree: DecisionTreeClassifier, out: str):
    depth = tree.get_depth()
    width, height = ceil(depth * 5), ceil(depth * 3.5)
    classes = [str(_class) for _class in tree.classes_]
    print(classes)
    plt.figure(figsize=(width, height))
    plot_tree(
        tree,
        feature_names=tree.feature_names_in_,
        fontsize=14,
        filled=True,
        rounded=True,
        class_names=classes
    )
    plt.savefig(out)
