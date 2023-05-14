import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from decision_tree import DecisionTreeID3, virtualize_my_tree
from sklearn.metrics import accuracy_score


class DataProcess():
    def __init__(self, path: str):
        self.data = pd.read_csv(path)
        pass


if __name__ == '__main__':
    path_data = './dataset/phishing.csv'
    data = pd.read_csv(path_data).drop(labels = 'Index', axis=1)
    attributes = [col for col in data.columns][:-1]

    idx2attribute = {}
    for i, attribute in enumerate(attributes):
        idx2attribute.update({i: attribute})
    
    # print(idx2attribute, idx2attribute[0])

    samples = data.drop(labels = 'class', axis=1).to_numpy()
    # print(samples.shape)

    samples = samples.tolist()
    label_samples = list(data['class'])


    data_decision_tree,_ , data_decision_tree_labels, _labels = train_test_split(samples, label_samples, test_size=0.9, random_state=42)

    # print(np.array(data_decision_tree).shape)
    # print(np.array(data_decision_tree_labels).shape)
    # print(list(idx2attribute.keys()))

    train, test, train_labels, test_labels = train_test_split(data_decision_tree, data_decision_tree_labels, test_size=0.2,random_state=44)
    # print(np.array(train).shape, np.array(test).shape)
    for max_depth in range(1, 10):
        for min_samples_splits in [1,4,5]:
            tree = DecisionTreeID3(max_depth=max_depth, min_samples_splits=min_samples_splits, index2attribute=idx2attribute)
            tree.fit(train, train_labels, list(idx2attribute.keys()))
            pred_labels = tree.predict(test)
            print(accuracy_score(test_labels, pred_labels))
            virtualize_my_tree(tree, f"./out/tree_{max_depth}_{min_samples_splits}.txt")
            




    



    