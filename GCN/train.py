import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchsummary import summary
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds = foz.load_zoo_dataset("coco-2017", split="train")
ds.persistent = True
nds = ds[0:10000]

with open('labels.json', 'r') as f:
    data = json.load(f)

all_labels = data['all_labels']
label_dict = data['label_dict']
label_cnt = data['label_cnt']

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

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int64)
    return torch.tensor(_adj, dtype=torch.float32, device=device)

# 定义图卷积层
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, adjacency_matrix):
        super(GraphConvolution, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        support = torch.mm(x, self.weight)
        output = torch.mm(support,self.adjacency_matrix)
        return output

# 定义包含 GCN 的模型
class GCNResNet(nn.Module):
    def __init__(self, num_classes, adjacency_matrix):
        super(GCNResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(2048, num_classes)  # 将 ResNet 输出调整为与标签数目一致
        self.gcn_layer = GraphConvolution(num_classes, num_classes, adjacency_matrix)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.resnet(x)
        features = self.fc(features)  # 调整特征维度
        output = self.gcn_layer(features)
        output = self.sigmoid(output)
        return output


# Load the adjacency matrix
adjacency_matrix = gen_A(label_cnt, 0.2, 'adjacency_file.pkl')

# Instantiate the model
model = GCNResNet(label_cnt, adjacency_matrix).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    model.train() 
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Save the model
torch.save(model, 'model_complete.pth')




