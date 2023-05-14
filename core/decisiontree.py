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
    def __init__(
        self,
        idxs: list,
        depth: int = 0,
        count_label: dict = {},
        idx2attribute: Optional[dict] = None
    ) -> None:
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
        self.idx2attribute = idx2attribute

    def set_leaf(self) -> None:
        """
        tìm xem với node này thì nhãn nào xuất hiện nhiều nhất và gán bằng nhãn đó:
        """
        item_max_value = max(self.count_label.items(), key=lambda x : x[1])
        self.label = item_max_value[0]

    def is_leaf(self) -> bool:
        return True if self.label is not None else False

    def to_dict(self) -> dict:
        if self.idx2attribute is not None:
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
                'attribute': self.idx2attribute[self.atribute] if self.atribute is not None else self.atribute,
                'entropy': self.entropy,
                'value': self.count_label,
                'value_attribute': self.value_attribute
            }
            return
        
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
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class DecisionTreeID3(object):
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_splits: int = 1,
        index2attribute: Optional[list] = None
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_splits
        self.data = None
        self.labels = None
        self.root = None # khởi tạo node ban đầu bằng None
        self.index2attribute = index2attribute

    def fit(self, data : list, labels : list, attributes: list) -> None:
        #attribute là list dạng index tương ứng các cột của các thuộc tính

        self.data = data
        self.labels = labels
        idx_items = [idx for idx in range(len(data))]
        dict_freq = self.__count_freq(idx_items=idx_items)

        self.root = TreeNode(
            idxs = idx_items, 
            depth = 0,
            count_label = dict_freq,
            idx2attribute= self.index2attribute
        )
        
        self.root.unselected_atributes = attributes

        queue = [self.root]

        while queue:
            node = queue.pop()

            if len(node.idxs) < self.min_samples_split:
                node.set_leaf()
                continue

            if self.max_depth is None or node.depth < self.max_depth:
                node.children = self.__findNextNode(node)
                queue += node.children
            else:
                node.set_leaf()

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
        best_dict_attribute = {}
        

        for attribute in node.unselected_atributes: # duyệt các attribute chưa được xét làm node
            dict_attribute = self.__divideAttribute(idx_items = idxs, attribute = attribute)
            # print("dict_att", dict_attribute)
            I = 0
            for key, value in dict_attribute.items():
                freq_dict = self.__count_freq(value)
                entropy_freq = entropy(list(freq_dict.values()))
                I += len(value) * entropy_freq / len(idxs)
            
            
            IG = node.entropy - I
            # print(IG)
            # print(f'--------------{IG}------------------')
            if IG >= best_IG:
                best_IG = IG
                best_attribute = attribute
                best_dict_attribute = dict_attribute
        
        node.atribute = best_attribute
        unselected_attributes = [attribute for attribute in node.unselected_atributes if attribute != best_attribute]
        children_node = []
        
        # print('best_dict_attribute ', best_dict_attribute)

        for key, value in best_dict_attribute.items():
            freq = self.__count_freq(value)
            child_node = TreeNode(
                idxs = value, 
                depth = node.depth + 1, 
                count_label=freq,
                idx2attribute=self.index2attribute
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
