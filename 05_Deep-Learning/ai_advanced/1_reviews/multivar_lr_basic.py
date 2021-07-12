# 다중선형 회귀
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(777)

# Practice 1
# 1.다중선형회귀실습withPyTorch구현
# 1) 가설:H(x)=w1x1+w2x2+w3x3+b
# 2) Train dataset 작성 torch.FloatTensor 형태
# 3) 가중치 W와 편향 b를 선언(다중 선형 회귀라서 가중치 W도 3개 선언)
# 4) optimizer 설정
# 5) 학습 코드 1000번을 반복하여 학습을 진행

# data
x1_train = torch.FloatTensor(([73], [93], [89], [96], [73]))
x2_train = torch.FloatTensor(([80], [88], [91], [98], [66]))
x3_train = torch.FloatTensor(([75], [93], [90], [100], [70]))



# 정답지
y_train = torch.FloatTensor(([152], [185], [180], [196], [142]))



# 가중치 w와 편향 b를 선언
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)


b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
epoch_num = 1000

for epoch in range(epoch_num + 1):
    # H(x) 게산
    # 가설을 선언한 부분
    # hypothersis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    hypothersis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b


    # loss
    loss = torch.mean((hypothersis - y_train) ** 2)


    # loss H(x) 계산
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if epoch % 100 == 0:
        print("Epoch {:4d}/{} w1 {:.3f} w2 {:.3f} w3 {:.3f} loss {:.6f}"
              .format(epoch, epoch_num, w1.item(), w2.item(), w3.item(), loss.item()))



# 2. PyTorch nn.Module 로 구현하는 선형회귀 (단순 선형, 다중 선형)
# 1) 파이토치에서 제공하는 선형 회귀모델 (nn.Linear(), 평균 제곱오차 nn.functional.mse_loss())