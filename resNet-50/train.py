import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F  # For handling expressions in matching and filtering
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchsummary import summary
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
torch.set_default_device(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = foz.load_zoo_dataset("coco-2017", split="train")
# ds = foz.load_zoo_dataset("quickstart")
ds.persistent = True

nds = ds[0:10000]

with open('labels.json', 'r') as f:
    data = json.load(f)

all_labels = data['all_labels']
label_dict = data['label_dict']
label_cnt = data['label_cnt']

# 定义自定义的 FiftyOne 数据集类，用于转换为 PyTorch 数据集
class FiftyOneTorchDataset(Dataset):
    def __init__(self, fiftyone_dataset):
        self.fiftyone_dataset = fiftyone_dataset

    def __len__(self):
        return len(self.fiftyone_dataset)

    def __getitem__(self, idx):
        sample = self.fiftyone_dataset[idx:].first()
        
        label = np.zeros(label_cnt)
        if sample.ground_truth is not None:
            for detection in sample.ground_truth.detections:
                label[label_dict[detection.label]] = 1

        image = sample["filepath"]
        image = Image.open(image).convert("RGB")
        image = preprocess(image)

        return image, torch.tensor(label, dtype=torch.float32)

train_dataset = FiftyOneTorchDataset(ds)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=torch.Generator(device='cuda'))

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.train()

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, label_cnt),
    nn.Sigmoid()
)

# 定义损失函数为二元交叉熵损失
criterion = nn.BCELoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    model.train() 
    
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

torch.save(model, 'model_complete.pth')