import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
import numpy as np
import csv




class CustomDataset(Dataset):
    def __init__(self, iris, transforms=None):
        self.x = [i[0] for i in data]
        self.y = [i[1] for i in data]

    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx): # getter
        x = self.x[idx]
        y = self.y[idx]

        return x, y


data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1]] # sample data

train_dataset = CustomDataset(data, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
# batch_size를 무조건 키워서 한번에 돌리는 것보다 짜잘해게 많이 돌리는게 학습률이 높고 컴퓨터가 연산 견디기에도 좋음 cpu 딸리면 1로 하는 게 좋음ㅎ

# for x, y in train_loader:
    # print(x, y)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
        self.batch_norm1 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)


    def forward(self, x):
        print(x.shape)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)