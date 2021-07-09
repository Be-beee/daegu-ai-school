import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self): # 모델 정의
        super(Net, self).__init__()

        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 3)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.sig(x)

        return x



class CustomDataset(Dataset):
    def __init__(self, x_train, y_train, transforms=None):
        # data
        self.x = x_train
        self.y = y_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = np.array(x)
        y = np.array(y)

        return x, y


iris = load_iris()
iris_data = iris.data
iris_label = iris.target

x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.25, shuffle=True, stratify=iris_label, random_state=40)
# train / test 분리


train_dataset = CustomDataset(x_train, y_train, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

epoch = 100
total_loss = 0

model.train() # 모델을 학습모드로 변경
for i in range(epoch):
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.long().to(device)

        outputs = model(x)

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()


        total_loss += loss.item()
        outputs = outputs.detach().numpy()

        y = y.numpy()
    if i == 99:
        torch.save(model.state_dict(), f"iris_model_last_2.pth")
    print(f"epoch -> {i}      loss -- > ", total_loss / len(x_train))
    optimizer.step()
    total_loss = 0



model.eval() # 모델을 평가 모드로 변경
model.load_state_dict(torch.load('iris_model_last_2.pth'))
test_dataset = CustomDataset(x_test, y_test, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

accuracy = 0
for x, y in test_loader:
    x = x.float().to(device)
    y = y.long().to(device)
    outputs = model(x)
    outputs = outputs.detach().numpy()
    answer = y.numpy()[0]
    max_value_idx = 0
    for i, label in enumerate(outputs[0]):
        if label == max(outputs[0]):
            max_value_idx = i
    print(y.numpy(), outputs)
    if answer == max_value_idx:
        print("----> correct!")
        accuracy += 1
    else:
        print("----> not correct!")

print(f'accuracy: {accuracy/len(x_test) * 100}%')