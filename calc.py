import fiftyone as fo
import fiftyone.zoo as foz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, average_precision_score
import json

ds = foz.load_zoo_dataset("coco-2017", split="validation")
ds.persistent = True

with open('labels.json', 'r') as f:
    data = json.load(f)

all_labels = data['all_labels']
label_dict = data['label_dict']
label_cnt = data['label_cnt']

with open('results.json', 'r') as f:
    data = json.load(f)

y_gt = data['y_gt']
y_pred = data['y_pred']

accuracy = 0
for label_gt, label_pred in zip(y_gt, y_pred):
    mat = 0
    all = 0
    for l1, l2 in zip(label_gt, label_pred):
        if l1 and l2:
            mat += 1
        if l1 or l2:
            all += 1
    if all > 0:
        accuracy += mat / all
    else:
        accuracy += 1
accuracy /= len(y_gt)
precision = precision_score(y_gt, y_pred, average='micro')
recall = recall_score(y_gt, y_pred, average='micro')
f1 = f1_score(y_gt, y_pred, average='micro')
hamming = hamming_loss(y_gt, y_pred)
mAP = average_precision_score(y_gt, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Hamming Loss: {hamming}")
print(f"mAP: {mAP}")

session = fo.launch_app(ds)

for labels, gfs in zip(y_pred, y_gt):
    label_id = 0
    out = []
    ans = []
    for val, boo in zip(labels, gfs):
        if val >= 0.5:
            out.append(all_labels[label_id])
        if boo:
            ans.append(all_labels[label_id])
        label_id = label_id + 1
    print(out)
    print(ans)
    input()