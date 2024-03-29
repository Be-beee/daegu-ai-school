# Pytorch nn.Linear and nn.Sigmoid로 로지스틱 회귀를 구현

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

torch.manual_seed(1)


# train data tensor
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

# train data -> Tensor
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

model(x_train)

optimizer = optim.SGD(model.parameters(), lr=0.1)
epoch_num = 1000


# train loop
for epoch in range(epoch_num + 1):

    # H(x) 계산
    hypothesis = model(x_train)

    # loss
    loss = f.binary_cross_entropy(hypothesis, y_train)

    # loss H(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print
    if epoch % 10 == 0:
        # 에측값이 0.5 넘으면 True로 간주
        prediction = hypothesis >= torch.FloatTensor([0.5])
        # 실제값과 일치하는 경우만 True로 간주
        correct_prediction = prediction.float() == y_train
        # 정확도 계산
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print("Epoch {:4d}/{} cost: {:.6f} Acc. {:2.2f}%"
              .format(epoch, epoch_num, loss.item(), accuracy*100))