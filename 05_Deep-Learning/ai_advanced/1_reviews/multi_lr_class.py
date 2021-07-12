# 다중 선형 클래스 구현
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

torch.manual_seed(777)

# data set
x_train = torch.FloatTensor([
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100],
    [73, 66, 70]
])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # 다중 선형 회귀이므로 input 3 output dim 1

    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)


# train loop
epoch_num = 6000
for epoch in range(epoch_num):
    # 가설
    prediction = model(x_train)


    # loss
    # F.mse_loss -> 파이토치에서 제공하는 평균 제곱 오차 함수
    loss = f.mse_loss(prediction, y_train)


    # loss 개선
    optimizer.zero_grad()  # 기울기를 0으로 초기화
    loss.backward()  # loss 함수를 미분하여 기울기 계산
    optimizer.step()  # w and b 업데이트



    # print
    # 100번 출력

    if epoch % 100 == 0:
        print("Epoch {:4d}/{} loss : {:.6f}".format(epoch, epoch_num, loss.item()))


new_var = torch.FloatTensor([[73, 82, 72]])
pred_y = model(new_var)
print(f"훈련 후 입력: {new_var}일 때의 예측값: {pred_y}")