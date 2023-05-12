import logging
from src import helper_pandas
from src.model import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score

# logging.basicConfig(level=logging.DEBUG)

# Đọc file csv
df = helper_pandas.read_csv("datasets/diabetes.csv")
df = helper_pandas.encode_attributes(df)
X, y = helper_pandas.separate_dataset(df, label='diabetes')


def compare(train_size: float):
    print(f"---------Result with train size {train_size}--------")
    splitted_data = helper_pandas.train_test_split(X, y, train_size)
    my_predict = build_with_my_tree(splitted_data)
    sklearn_predict = build_with_sklearn_tree(splitted_data)

    same, diff = 0, 0
    for i in range(my_predict.__len__()):
        if my_predict[i] != sklearn_predict[i]:
            diff += 1
        else:
            same += 1

    print('Same', same)
    print('Diff', diff)


def build_with_my_tree(splitted_data):
    X_train, X_test, y_train, y_test = splitted_data['my_input']
    tree = DecisionTree()
    tree.fit(X_train, y_train)
    y_predict = tree.predict(X_test)
    print("My accuracy:", accuracy_score(y_test, y_predict))
    print("My f1-score:", f1_score(y_test, y_predict))
    print("My r2-score:", r2_score(y_test, y_predict))
    return y_predict


def build_with_sklearn_tree(splitted_data):
    X_train, X_test, y_train, y_test = splitted_data['sklearn_input']
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(X_train, y_train)
    y_predict = tree.predict(X_test)
    print("Sklearn accuracy:", accuracy_score(y_test, y_predict))
    print("Sklearn f1-score:", f1_score(y_test, y_predict))
    print("Sklearn r2-score:", r2_score(y_test, y_predict))
    return list(y_predict)


sizes = [0.2, 0.4, 0.6, 0.8]
for size in sizes:
    compare(size)
