import fiftyone.zoo as foz
import numpy as np
import json
import pickle
from tqdm import tqdm

# 加载 COCO 数据集
ds = foz.load_zoo_dataset("coco-2017", split="train")

with open('labels.json', 'r') as f:
    data = json.load(f)

all_labels = data['all_labels']
label_dict = data['label_dict']
label_cnt = data['label_cnt']

# 初始化共现矩阵
cooccurrence_matrix = np.zeros((label_cnt, label_cnt), dtype=np.int32)

# 遍历数据集，计算标签共现
for sample in tqdm(ds):
    if sample.ground_truth is not None:
        labels = [detection.label for detection in sample.ground_truth.detections]
        label_indices = [label_dict[label] for label in labels]  # 将标签名称转换为索引

        for i in range(len(label_indices)):
            for j in range(len(label_indices)):
                if i != j:
                    cooccurrence_matrix[label_indices[i], label_indices[j]] += 1

# 打印共现矩阵
print(cooccurrence_matrix)

# 设置阈值并构建邻接矩阵
threshold = 5
adjacency_matrix = (cooccurrence_matrix > threshold).astype(np.float32)

# 统计每个标签的出现次数
label_counts = np.sum(cooccurrence_matrix, axis=1)

# 创建包含邻接矩阵和标签计数的字典
result = {
    'adj': adjacency_matrix,
    'nums': label_counts
}

# 保存到 .pkl 文件
with open('adjacency_file.pkl', 'wb') as f:
    pickle.dump(result, f)