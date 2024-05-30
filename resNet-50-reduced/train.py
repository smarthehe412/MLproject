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
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

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
                if detection.label in label_dict:
                    label[label_dict[detection.label]] = 1

        image = sample["filepath"]
        image = Image.open(image).convert("RGB")
        image = preprocess(image)

        return image, torch.tensor(label, dtype=torch.float32)

train_dataset = FiftyOneTorchDataset(ds)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True, num_workers=12)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.train()

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, label_cnt),
    nn.Sigmoid()
)

model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
cpu_times = []
gpu_times = []

for epoch in tqdm(range(num_epochs)):
    model.train() 
    
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        torch.cuda.synchronize()

        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')
    torch.save(model.state_dict(), f'model_epoch{epoch+1}.pth')

torch.save(model.state_dict(), 'model_complete.pth')