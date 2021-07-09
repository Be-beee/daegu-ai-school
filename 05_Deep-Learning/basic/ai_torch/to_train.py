import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np



class Net(nn.Module):
    def __init__(self): # 모델 정의
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(3, 1)
        self.batch_norm1 = nn.BatchNorm1d(3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)
        self.sig = nn.Sigmoid()

    def forward(self, x):  # 실제 모델이 동작하는 부분
        # print(x.shape)
        x = self.fc1(x)
        x = self.batch_norm1(x)  # 정규화 -> batch_size를 웬만하면 짝수로 해서 서로간의 비교 대상이 있어야함
        # x = self.relu(x)
        # x = self.dropout(x)

        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.sig(x)

        return x


class CustomDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.x = [i[0] for i in data]
        self.y = [i[1] for i in data]

    def __len__(self): # dataset 데이터 갯수
        return len(self.x)

    def __getitem__(self, idx):
        x = [self.x[idx]]
        y = self.y[idx]
        x = np.array(x)
        y = np.array(y)

        return x, y


data = [[2, 4], [4, 8], [6, 12], [8, 16], [10, 20], [12, 24]]

train_dataset = CustomDataset(data, transforms=None)
print(train_dataset.x)
print(train_dataset.y)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
# drop_last: batchNorm의 사이즈에 어긋나는 텐서에 대해 (마지막 나머지로 들어오는 것) 버리겠다. 데이터 사이즈가 작으면 drop_last가 모델 학습에 치명적일 수 있음
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)
# 모델 정의


criterion = nn.MSELoss() # MSE 로스
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters() , lr=  0.001) # learning rate -> 보폭, step_size -> 걸음수


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1) # 1000스텝마다 0.1을 곱한다.
epoch = 1000
total_loss = 0


model.train() # 모델을 학습모드로 변경
for i in range(epoch):
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.float().to(device)

        # print(x,y)
        # print(x, y)
        outputs = model(x)
        # print(outputs)

        loss = criterion(outputs, y)
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 가중치와 편향에 대해 기울기 계산


        total_loss += loss.item() # 로스값 -> 텐서로 나옴
        outputs = outputs.detach().numpy() # 기울기 값을 떼어내줌
        y = y.numpy()
    if i == 999: # loss가 best일 때, last일 때를 저장함
        torch.save(model.state_dict(), f"weight/model_last.pth") # 파일로 저장. state_dict()에 학습 로그가 전부 남음, f"~.pth"는 경로
    print(f"epoch -> {i}      loss -- > ", total_loss / len(train_loader))
    optimizer.step() # 한 스텝을 쌓는다.
    total_loss = 0

model.eval() # 모델을 평가 모드로 변경
model.load_state_dict(torch.load('weight/model_last.pth'))
test_data = [[1, 2],[3, 6],[5, 10]]
test_dataset = CustomDataset(test_data, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # 평가만 하는 거라 셔플할 필요 없음
for x, y in test_loader:
    x = x.float().to(device)
    y = y.long().to(device)
    # print(x, y)
    outputs = model(x)
    print(x, y, outputs)
