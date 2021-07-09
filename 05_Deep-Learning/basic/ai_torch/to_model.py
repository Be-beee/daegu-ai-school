import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 다 주어지는 코드

        self.fc1 = nn.Linear(1, 3) # 1개의 값을 3개로 늘림
        self.fc2 = nn.Linear(3, 1) # 3개의 값을 1개의 결과값으로 표현
        self.batch_norm1 = nn.BatchNorm1d(3) # 학습할 때 정규화, batch_size가 1일 때는 정규화시 비교 대상이 없어서 에러남 튀는 값을 서로 비교해서 맞춰줌~~
        self.relu = nn.ReLU(inplace=True) # 음수 값 무시한다.
        self.dropout = nn.Dropout(0.3) # 연산이 좀 많을 때 0.3 확률로 랜덤 결과값을 죽이겠다. 컴퓨터가 연산을 힘들어하니까ㅇㅇ -> 근데 오히려 학습률이 높아질 때가 있음



    def forward(self, x):
        print(x.shape)
        print(x)
        x = self.fc1(x)
        # x = self.batch_norm1(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        # x = self.relu(x)
        print(x.shape)
        print(x)
        return x



class CustomDataset(Dataset):
  def __init__(self, data, transforms=None):
      self.x = [i[0] for i in data]
      self.y = [i[1] for i in data]


  def __len__(self):
    return len(self.x)


  def __getitem__(self, idx):
      x = [self.x[idx]]
      y = self.y[idx]
      x= np.array(x)


      return x, y



data = [[2,0], [4,0], [6,0], [8,1], [10,1], [12,1]]

train_dataset = CustomDataset(data, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# dataset 구축 완료

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Net().to(device)
# 여기 연산 전부 cpu로 쓸거임
# 모델 정의 완료

for x, y in train_loader:
    x = x.float().to(device) # 들어갈 때 GPU를 사용할 때를 고려
    y = y.float().to(device)
    # int32 -> float로 형변환: 모델 안에 데이터 넣을 때 float / long으로 넣음
    print(x)
    print("-"*20)
    outputs = model(x) # forward가 실행되는 지점
    print(outputs)
    outputs = outputs.detach().numpy() # 기울기 값 정보(grad_fn=<AddmmBackward>) 제거 detach()
    print(outputs)
    print(y)
    exit()