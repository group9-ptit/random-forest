import time
import logging
from src import helper_pandas, model
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# logging.basicConfig(level=logging.DEBUG)
N_JOBS = 6
MAX_SAMPLES = 0.8

df = helper_pandas.read_csv("datasets/diabetes.csv")
df = helper_pandas.encode_attributes(df)
X, y = helper_pandas.separate_dataset(df, label="diabetes")


def compare(
    train_size: float,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2
):
    print(
        f"train_size={train_size} | n_estimators={n_estimators} | max_depth={max_depth} | min_samples_split={min_samples_split}")
    splitted_data = helper_pandas.train_test_split(X, y, train_size)
    my_predict = build_with_my_forest(
        splitted_data,
        n_estimators,
        max_depth,
        min_samples_split
    )
    sklearn_predict = build_with_sklearn_forest(
        splitted_data,
        n_estimators,
        max_depth,
        min_samples_split
    )

    same, diff = 0, 0
    for i in range(my_predict.__len__()):
        if my_predict[i] != sklearn_predict[i]:
            diff += 1
        else:
            same += 1

    print('Same', same)
    print('Diff', diff)


def build_with_my_forest(
    splitted_data,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2
):
    X_train, X_test, y_train, y_test = splitted_data['my_input']
    rf = model.RandomForest(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        n_jobs=N_JOBS,
        max_samples=MAX_SAMPLES
    )
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    print("My accuracy:", accuracy_score(y_test, y_predict))
    # print("My f1-score:", f1_score(y_test, y_predict))
    # print("My r2-score:", r2_score(y_test, y_predict))
    return y_predict


def build_with_sklearn_forest(
    splitted_data,
    n_estimators=100,
    max_depth=None,
    min_samples_split=2
):
    X_train, X_test, y_train, y_test = splitted_data['sklearn_input']
    rf = RandomForestClassifier(
        criterion='entropy',
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        n_jobs=N_JOBS,
        max_samples=MAX_SAMPLES
    )
    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    logging.debug(f'[SKLEARN]: Duration = {end_time - start_time}')

    y_predict = rf.predict(X_test)
    print("Sklearn accuracy:", accuracy_score(y_test, y_predict))
    # print("Sklearn f1-score:", f1_score(y_test, y_predict))
    # print("Sklearn r2-score:", r2_score(y_test, y_predict))
    return list(y_predict)


estimators = [10, 20]
sizes = [0.8]
max_depths = [8]
min_samples_splits = [8]

for estimator in estimators:
    for size in sizes:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                compare(size, estimator, max_depth, min_samples_split)
