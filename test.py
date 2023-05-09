from src.helper import read_csv, separate_dataset
from src.model import DecisionTree
import logging

logging.basicConfig(level=logging.DEBUG)

data = read_csv("datasets/weather.csv")
X, Y = separate_dataset(data, "Label")

tree = DecisionTree()
tree.fit(X, Y)
label = tree.predict_one({
    "Outlook": "Rainy",
    "Humidity": "High",
    "Temp": "Hot",
    "Windy": "FALSE"
})
print(label)
