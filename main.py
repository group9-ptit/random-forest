from core import helper, testcase

N_JOBS = 4
MAX_SAMPLES = 0.8

df = helper.read_csv("datasets/diabetes.csv")
df = helper.encode_attributes(df)
X, Y = helper.separate_dataset(df, label="diabetes")

estimators = [50]
train_sizes = [0.8]
max_depths = [8]
min_samples_splits = [20]

for train_size in train_sizes:
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            ts = testcase.DecisionTreeTestCase(
                X=X,
                Y=Y,
                train_size=train_size,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion='entropy'
            )
            ts.run()
            ts.print_result()
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
#                 ts.print_result()
