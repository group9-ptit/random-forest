from typing import Optional
import math
import json


def entropy(freq: list) -> float:

    """
    có freq là tần suất xuất hiện của 1 attribute, tính entropy qua 2 bước:
        + Tính total_sample là tổng số mẫu
        + xác suất xuất hiện = tần suất / total_sample
        + entropy của 1 item trong freq = -pi * log2(pi) (ở vòng for mình chưa tính dấu trừ vì sau sẽ đặt dấu trừ ra ngoài)
    
    return entropy của list này
    """
    if len(freq) <= 1:
        return float(0)
    entropy = []
    
    total_sample = sum(freq)
    for item in freq:
        probability = item / total_sample
        entropy.append(probability * math.log2(probability))
    
    return -sum(entropy)
    

class TreeNode(object):
    def __init__(self, idxs: list, depth: int = 0, count_label: dict = {}) -> None:
        self.idxs = idxs #index của data trong node này
        self.entropy = entropy(list(count_label.values())) # diễn tả entropy của attribute
        self.depth = depth # diễn tả độ sâu của cây

        self.atribute = None # diễn tả thuộc tính được chọn làm node !!! Lưu attribute dưới dạng index
        self.value_attribute = None # VD cột label có yes và no thì label là attribute còn yes và no là value attribute tương ứng với cột label
        self.unselected_atributes = None # diễn tả thuộc tính chưa được chọn có thể làm node con sau nó
        self.children = [] # diễn tả node con của nó
        self.label = None # nhãn của Node đó nếu nó là lá

        self.father_attribute = None # Nhớ thuộc tính cha của nó là gì, chỉ dùng để tiện vẽ cây thui
        self.count_label = count_label

    
    def set_leaf(self) -> None:
        #tìm xem với node này thì nhãn nào xuất hiện nhiều nhất và gán bằng nhãn đó:
        item_max_value = max(self.count_label.items(), key=lambda x : x[1])
        self.label = item_max_value[0]
    

    def is_leaf(self) -> bool:
        return True if self.label is not None else False
    


    
    
    def to_dict(self) -> dict:
        if self.is_leaf():
            return {
                'sample': len(self.idxs),
                'value': self.count_label,
                'entropy': self.entropy,
                'value_attribute': self.value_attribute,
                'label': self.label
            }
        return {
            'sample': len(self.idxs),
            'value': self.count_label,
            'attribute': self.atribute,
            'entropy': self.entropy,
            'value': self.count_label,
            'value_attribute': self.value_attribute
        }

    
    @property
    def value(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    

    def display(self):
        print('--tree node---')
        # print('att', self.atribute)
        print('depth', self.depth)
        print('label', self.label)
        print('entropy', self.entropy)
        print('value attribute', self.value_attribute)
        # print('father attribute', self.father_attribute)
        
        for key, value in attribute2idx.items():
            if self.atribute == value:
                print('attribute', key)
                break
        
        for key, value in label2idx.items():
            if self.label == value:
                print('label', key)
                break

        for key, value in attribute2idx.items():
            if self.father_attribute == value:
                print('father attribute', key)
                break
        
        if self.father_attribute == 0:
            for key, value in outlook2idx.items():
                if value == self.value_attribute:
                    print('value attribute', key)
                    break
        elif self.father_attribute == 1:
            for key, value in temperature2idx.items():
                if value == self.value_attribute:
                    print('value attribute', key)
                    break
        elif self.father_attribute == 2:
            for key, value in humidity2idx.items():
                if value == self.value_attribute:
                    print('value attribute', key)
                    break
        elif self.father_attribute == 3:
            for key, value in wind2idx.items():
                if value == self.value_attribute:
                    print('value attribute', key)
                    break

class DecisionTreeID3(object):
    def __init__(self, max_depth: Optional[int] = None, min_samples_splits: int = 1) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_splits
        
        self.data = None
        self.labels = None
        self.root = None # khởi tạo node ban đầu bằng None
        

    def fit(self, data : list, labels : list, attributes: list) -> None:
        #attribute là list dạng index tương ứng các cột của các thuộc tính

        self.data = data
        self.labels = labels
        idx_items = [idx for idx in range(len(data))]
        dict_freq = self.__count_freq(idx_items=idx_items)
       

        self.root = TreeNode(
            idxs = idx_items, 
            depth = 0,
            count_label=dict_freq,
        )
        
        self.root.unselected_atributes = attributes

        queue = [self.root]

        while queue:
            node = queue.pop()

            if len(node.idxs) < self.min_samples_split:
                #here
                # print("node idx", node.idxs)
                # dict_freq = self.__count_freq(idx_items= node.idxs)

                # #tìm xem với node này thì nhãn nào xuất hiện nhiều nhất và gán bằng nhãn đó:
                # item_max_value = max(dict_freq.items(), key=lambda x : x[1])
                # node.label = item_max_value[0]
                # # print("hi: ", dict_freq)
                node.set_leaf()
                continue

            if self.max_depth is None or node.depth < self.max_depth:
                node.children = self.__findNextNode(node)
                queue += node.children
            else:
                node.set_leaf()
                # dict_freq = self.__count_freq(idx_items=node.idxs)
                
                # #tìm xem với node này thì nhãn nào xuất hiện nhiều nhất và gán bằng nhãn đó:
                # item_max_value = max(dict_freq.items(), key=lambda x : x[1])
                # node.label = item_max_value[0]

                pass
                

    def __divideAttribute(self, idx_items: list, attribute: int) -> dict:
        '''
        Input:
            items: list index data 
            attribute: index cột của thuộc tính cần chia thành các phần

        Chia các thuộc tính ra thành các phần theo giá trị của nó: 
        trả về dictionary gồm:
            + key là giá trị thuộc tính
            + value là index của các data chứa value thuộc tính tương ứng
        '''
        dict_attribute = {}
        for idx in idx_items:
            sample = self.data[idx]

            try:
                dict_attribute[sample[attribute]].append(idx)
            except:
                dict_attribute.update({sample[attribute]: [idx]})

        return dict_attribute
    
    def __count_freq(self, idx_items: list):
        """
        return tần suất xuất hiện các label trong tương ứng với các idx_items
        """
        freq_dict = {}

        for idx in idx_items:
            item = self.labels[idx]
            try:
                freq_dict[item] = int(freq_dict[item] + 1)
            except:
                freq_dict.update({item: 1})

        
        return freq_dict

        


    def __findNextNode(self, node : TreeNode) -> list:
        
        if node.label is not None:
            return [] # nếu node này mà có nhãn rồi thì nó là node lá rồi, ko có node con nữa nên return về  [] luôn
        
        idxs = node.idxs
        
        best_IG = 0
        best_attribute = None
        best_dict_attribute = None
        

        for attribute in node.unselected_atributes: # duyệt các attribute chưa được xét làm node
            dict_attribute = self.__divideAttribute(idx_items = idxs, attribute = attribute)
            
            I = 0
            for key, value in dict_attribute.items():
                freq_dict = self.__count_freq(value)
                entropy_freq = entropy(list(freq_dict.values()))
                I += len(value) * entropy_freq / len(idxs)
            
            
            IG = node.entropy - I
            # print(f'--------------{IG}------------------')
            if IG > best_IG:
                best_IG = IG
                best_attribute = attribute
                best_dict_attribute = dict_attribute
        
        node.atribute = best_attribute
        unselected_attributes = [attribute for attribute in node.unselected_atributes if attribute != best_attribute]
        children_node = []
        for key, value in best_dict_attribute.items():
            freq = self.__count_freq(value)
            child_node = TreeNode(
                idxs = value, 
                depth = node.depth + 1, 
                # entropy = entropy(list(
                #     freq.values()
                # )),
                count_label=freq
            )
            labels = list(freq.keys())
            if len(labels) == 1:
                child_node.label = labels[0]
            
            child_node.unselected_atributes = unselected_attributes
            child_node.value_attribute = key
            children_node.append(child_node)

            child_node.father_attribute = node.atribute

        return children_node


    def predictOneSample(self, new_data: list) -> str:
        """
        data có dạng là 1 list gồm các thuộc tính ừ attribute 0 -> n
        """
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.label is not None:
                return node.label
            attribute = node.atribute # lấy thuộc tính attributes trong node
           
            value_attribute = new_data[attribute] # đọc giá trị của thuộc tính tương ứng với attributes để xem vào child_node nào
           
            for child_node in node.children:
                if child_node.value_attribute == value_attribute:
                    queue.append(child_node)
        return None
    
    def predict(self, new_data):
        labels = []
        for data in new_data:
            labels.append(self.predictOneSample(data))
        
        return labels
    

    def display_tree(self):
        
        lst = [self.root]
        while lst:
            node = lst.pop()
            node.display()
            lst += node.children

 
import pandas as pd

class DataProcess():
    def __init__(self, path: str):
        self.data = pd.read_csv(path)


from PrettyPrint import PrettyPrintTree

def virtualize_my_tree(tree: DecisionTreeID3, out: str):
    printer = PrettyPrintTree(
        lambda node: node.children,
        lambda node: node.value,
        return_instead_of_print=True,
        color=None,
        border=True,
    )
    tree_str = printer(tree.root)
    with open(out, mode='w') as file:
        file.write(tree_str)

from sklearn.metrics import accuracy_score
attribute2idx = {'outlook': 0, 'temperature': 1, 'humidity': 2, 'wind': 3}
outlook2idx = {'sunny': 0, 'overcast' : 1, 'rainy': 2}
temperature2idx = {'hot': 0, 'mild': 1, 'cool': 2}
humidity2idx = {'high': 0, 'normal': 1}
wind2idx = {'strong': 0, 'weak': 1}
label2idx = {'yes': 1, 'no': 0}
test = [1, 2, 1, 1]
if __name__=='__main__':
    df = pd.read_csv('./weather.csv', index_col = 0, parse_dates = True)

    X = df.drop(labels = 'play', axis=1)
    X = X.values.tolist()
    y = list(df['play'])
    
    train = []
    for item in X:
        train.append([outlook2idx[item[0]],  temperature2idx[item[1]], humidity2idx[item[2]], wind2idx[item[3]]])

    labels = [label2idx[item] for item in y]
    
    tree = DecisionTreeID3(None,1)
    

    atts = [i for i in range(4)]

    # root = TreeNode(
    #         idxs = [i for i in range (len(labels))], 
    #         depth = 0, 
    #         entropy = entropy()
    #     )
        
        
    # root.unselected_atributes = atts
    tree.fit(train, labels, atts)
    
    pred_labels = tree.predict(train)
    print(pred_labels)
    print(labels)
    print(accuracy_score(labels, pred_labels))

    tree.display_tree()
    
    virtualize_my_tree(tree, "tree.txt")

    #display tree
    