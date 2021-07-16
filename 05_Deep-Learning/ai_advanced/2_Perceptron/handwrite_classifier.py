import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


mnist = fetch_openml('mnist_784', version=1, as_frame=False)

mnist.target = mnist.target.astype(np.int8)
x = mnist.data / 255  # 0 ~ 255 값을 [0, 1] 구간으로 정규화
y = mnist.target


# plt.imshow(x[3].reshape(28, 28), cmap='gray')
# print('image target name: {:.0f}'.format(y[3]))


# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)



# print(x_train.shape)
# print(x_test)
# print(y_train)
# print(y_test)



# data -> tensor
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)



# dataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# dataloader
loader_train = DataLoader(train_set, batch_size=64, shuffle=True)
loader_test = DataLoader(test_set, batch_size=64, shuffle=False)



# multi-layer perceptron
from torch import nn
from torch import optim

model = nn.Sequential()
model.add_module("fc1", nn.Linear(28 * 28 * 1, 100))
model.add_module("relu1", nn.ReLU())
model.add_module("fc2", nn.Linear(100, 100))
model.add_module("relu2", nn.ReLU())
model.add_module("fc3", nn.Linear(100, 100))


# Loss Func
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train loop
def train(epoch):
    # 신경망을 학습 모드로 전환
    model.train()

    # DataLoader에서 미니배치를 하나씩 꺼내 학습 수행
    for data, targets in loader_train:
        #optimizer zero
        optimizer.zero_grad()
        output = model(data)

        # 출력과 훈련 데이터 정답간의 오차 계산
        loss = loss_fn(output, targets.long())
        loss.backward()
        optimizer.step()

    print(f"epoch {epoch} : 완료\n")



def test():
    # 신경망을 추론(평가) 모드로 전환
    model.eval()
    correct = 0

    # DataLoader에서 데이터를 꺼내 추론 실행
    with torch.no_grad():  # 추론 과정에는 미분이 필요 없음
        for data, targets in loader_test:
            output = model(data)   # 데이터를 입력하고 출력을 계산

            # 추론
            # 확률이 가장 높은 레이블이 무엇인지 계산
            _, pred = torch.max(output.data)
            correct += pred.eq(targets.data.view_as(pred)).sum()

    data_num = len(loader_test.dataset)
    print("\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%\n"
          .format(correct, data_num, 100. * correct / data_num))


for epoch in range(10):
    train(epoch)

test()


# 추론
index = 2007

model.eval()
data = x_test[index]

output = model(data)
_, pred = torch.max(output.data, 0)

print(f"예측 결과: {pred}")

x_test_show = (x_test[index]).numpy()
plt.imshow(x_test_show.reshape(28, 28), cmap='gray')

print("이 이미지 데이터의 정답 레이블은 {:.0f}입니다.".format(y_test[index]))