import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# linear regression
# 현재 실습하고 있는 파이썬 코드 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드 설정
torch.manual_seed(7777)


# 실습을 위한 기본 세팅 작업
# 훈련 데이터 x_train, y_train
x_train = torch.FloatTensor(([1], [2], [3]))
y_train = torch.FloatTensor(([2], [4], [7]))



# show x_train shape
print('x_train >> ', x_train.size()) # shape or size()
print('y_train >> ', y_train.shape)

# x_train >>  torch.Size([3, 1])
# y_train >>  torch.Size([3, 1])



# 가중치와 편향의 초기화
# 가중치 0으로 초기화하고 이 값을 출력 편향 b도 0으로 초기화
# requires_grad=True : 학습을 통해서 계속 값이 변경되는 변수
w = torch.zeros(1, requires_grad=True)
print("가중치 w", w)

b = torch.zeros(1, requires_grad=True)
print("편향 b", b)

# 가중치 w tensor([0.], requires_grad=True)
# 편향 b tensor([0.], requires_grad=True)




# 가설 선언
# 파이토치 코드 상으로 직선의 방정식에 해당되는 가설을 선언
hypothersis = x_train * w + b
print("가설: ", hypothersis)

# 가설:  tensor([[0.],
#         [0.],
#         [0.]], grad_fn=<AddBackward0>)





# Loss Function 선언
loss = torch.mean((hypothersis - y_train) ** 2)
print(loss)
# tensor(23., grad_fn=<MeanBackward0>)




# 경사하강법 구현
# input w b 가 sgd 입력이 되어야 합니다.
optimizer = optim.SGD([w, b], lr=0.01)




# 기울기 0으로 초기화
optimizer.zero_grad()


# loss function 미분하여 기울기 계산
loss.backward()


# w와 b값을 업데이트
optimizer.step()


# 학습 진행
epoch_num = 2000  # 원하는 만큼 경사하강법을 반복



# Train mode
# epoch -> 전체 훈련 데이터가 학습에 한 번 사용되는 주기
for epoch in range(epoch_num+1):
    # 가설 계산
    hypothersis = x_train * w + b

    # # CNN
    # out = model(input)

    # loss 계산
    loss = torch.mean((hypothersis - y_train) ** 2)


    # loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    # 100번마다 print
    if epoch % 100 == 0:
        print("Epoch {:4d}/{} w : {:.3f}, b :{:.3f} loss : {:.6f}"
              .format(epoch, epoch_num, w.item(), b.item(), loss.item()))





