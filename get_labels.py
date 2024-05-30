import json
import fiftyone as fo
import fiftyone.zoo as foz

ds = foz.load_zoo_dataset("coco-2017", split="train")
ds.persistent = True

label_dict = {}
all_labels = ds.distinct("ground_truth.detections.label")
label_id = 0
for label in all_labels:
    label_dict[label] = label_id
    label_id = label_id + 1

# 将列表和字典存储到文件
with open('labels.json', 'w') as f:
    json.dump({'all_labels': all_labels, 'label_dict': label_dict, 'label_cnt': label_id}, f)

