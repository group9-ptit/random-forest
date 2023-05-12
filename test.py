import logging
from src.helper import read_csv, separate_dataset, encode_attributes
from src.model import DecisionTree

logging.basicConfig(level=logging.DEBUG)

# Đọc file csv
rows = read_csv("datasets/diabetes.csv")

# Phân chia dữ liệu thành hai phần thuộc tính và nhãn
non_encoded_rows, labels = separate_dataset(rows, label="diabetes")

# Mã hoá thuộc tính rời rạc thành liên tục
encoded_rows = encode_attributes(non_encoded_rows)

# Huấn luyện mô hình
tree = DecisionTree()
tree.fit(encoded_rows, labels)

# In cây vào file json
tree.write_tree("tree.json")
