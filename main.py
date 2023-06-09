import logging
from core import helper, testcase
import json
# logging.basicConfig(level=logging.INFO)

N_JOBS = 6
MAX_SAMPLES = 0.8
df = helper.read_csv("datasets/phishing.csv")
df = helper.encode_attributes(df)
X, Y = helper.separate_dataset(df, label="class")

estimators = [10]
train_sizes = [0.5]
max_depths = [None]
min_samples_splits = [5]

results = []
for train_size in train_sizes:
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            ts = testcase.AllDecisionTreeTestCase(
                X=X,
                Y=Y,
                train_size=train_size,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion="entropy"
            )
            ts.run()
            # ts.print_result()
            results.append(ts.to_dict())
            ts.export_tree()


# for n_estimators in estimators:
#     for train_size in train_sizes:
#         for depth in max_depths:
#             for min_samples_split in min_samples_splits:
#                 ts = testcase.RandomForestTestCase(
#                     X=X,
#                     Y=Y,
#                     n_estimators=n_estimators,
#                     train_size=train_size,
#                     max_depth=depth,
#                     min_samples_split=min_samples_split,
#                     n_jobs=N_JOBS,
#                     max_samples=MAX_SAMPLES
#                 )
#                 ts.run()
#                 results.append(ts.to_dict())
                # ts.print_result()

with open("decision-tree.json", "w") as file:
    json.dump(results, file)
