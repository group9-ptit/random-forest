from src.helper import read_csv, separate_dataset
from src.model import DecisionTree
import logging
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

data = read_csv("datasets/weather.csv")
X, Y = separate_dataset(data, "Label")

tree = DecisionTree()
tree.fit(X, Y)
# label = tree.predict_one({
#     "Outlook": "Rainy",
#     "Humidity": "High",
#     "Temp": "Hot",
#     "Windy": "FALSE"
# })
# print(label)
print(tree.to_json())
# df = pd.read_csv("datasets/weather.csv")
# label_encoder = LabelEncoder()
# Y = df['Label']
# X = df.drop(labels=['Label'], axis=1)
# for column in X.columns:
#     X[column] = label_encoder.fit_transform(X[column])
# Y = label_encoder.fit_transform(Y)
# tree = DecisionTreeClassifier(criterion='entropy')
# tree.fit(X, Y)
# plt.figure(figsize=(10, 8))
# plot_tree(tree)
# plt.show()