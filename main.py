import logging
from src import helper_pandas, virtualization, model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score

# logging.basicConfig(level=logging.DEBUG)

# Đọc file csv
df = helper_pandas.read_csv("datasets/diabetes.csv")
df = helper_pandas.encode_attributes(df)
X, y = helper_pandas.separate_dataset(df, label="diabetes")

flag = False

def compare(train_size: float, max_depth=None, min_samples_split=2):
    global flag
    print(
        f"---------Result with train_size={train_size} | max_depth={max_depth} | min_samples_split={min_samples_split}--------")
    splitted_data = helper_pandas.train_test_split(X, y, train_size)
    my_tree, my_predict = build_with_my_tree(splitted_data, max_depth, min_samples_split)
    sklearn_tree, sklearn_predict = build_with_sklearn_tree(splitted_data, max_depth, min_samples_split)

    # Draw tree
    same, diff = 0, 0
    for i in range(my_predict.__len__()):
        if my_predict[i] != sklearn_predict[i]:
            diff += 1
        else:
            same += 1

    if diff != 0 and flag == False:
        virtualization.virtualize_my_tree(my_tree, "out/my_tree.txt")
        virtualization.virtualize_sklearn_tree(sklearn_tree, "out/sklearn_tree.png")
        flag = True

    print("Same", same)
    print("Diff", diff)


def build_with_my_tree(splitted_data, max_depth=None, min_samples_split=2):
    X_train, X_test, y_train, y_test = splitted_data["my_input"]
    tree = model.DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    tree.fit(X_train, y_train)
    y_predict = tree.predict(X_test)
    print("My accuracy:", accuracy_score(y_test, y_predict))
    # print("My f1-score:", f1_score(y_test, y_predict))
    # print("My r2-score:", r2_score(y_test, y_predict))
    return tree, y_predict


def build_with_sklearn_tree(splitted_data, max_depth=None, min_samples_split=2):
    X_train, X_test, y_train, y_test = splitted_data["sklearn_input"]
    tree = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    tree.fit(X_train, y_train)
    y_predict = tree.predict(X_test)
    print("Sklearn accuracy:", accuracy_score(y_test, y_predict))
    # print("Sklearn f1-score:", f1_score(y_test, y_predict))
    # print("Sklearn r2-score:", r2_score(y_test, y_predict))
    return tree, list(y_predict)


sizes = [0.8]
max_depths = [None]
min_samples_splits = [4]
for size in sizes:
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            compare(size, max_depth, min_samples_split)
