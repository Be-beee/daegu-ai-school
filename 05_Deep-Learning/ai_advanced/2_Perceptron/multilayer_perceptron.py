# 파이토치로 다층 퍼셉트론 구현
import torch
import torch.nn as nn

# GPU 연산 가능하다면 Random seed
torch.manual_seed(777)

# GPU 사용 가능한가?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

if device == 'cuda':
    torch.cuda.manual_seed(777)

# cpu test code
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]


x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)
# print(x, y)


# references
# crossEntropy 경우에는 마지막 레이어 노드수가 2개 이상이어야 함
# 만약 마지막층 1개 output이라면 BCELoss
# BCELoss 사용할 경우 마지막 레이어의 값은 0 ~ 1로 조정 필요
# 마지막 레이어에 시그모이드 함수 적용

# 다층 퍼셉트론 설계
model = nn.Sequential(
    nn.Linear(2, 10, bias=True),  # input=2, hidden=10
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),  # input=10, hidden=10
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),  # input=10, hidden=10
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True),  # input=10, output=1
    nn.Sigmoid()
).to(device)

print(model)


criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)




# train loop
for epoch in range(100000):
    optimizer.zero_grad()

    output = model(x)

    # loss
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()



    # print
    if epoch % 10000 == 0:
        # loss test
        print(f"epoch {epoch}, loss {loss.item()}")



# 학습된 다층 퍼셉트론의 예측값 확인
with torch.no_grad():
    output = model(x)
    pred = (output > 0.5).float()
    acc = (pred == y).float().mean()
    print("model 츨력값 \n", output.detach().cpu().numpy())
    print("model 예측값 \n", pred.detach().cpu().numpy())
    print("실제값 (Y) \n", y.cpu().numpy())
    print("정확도:", acc.item())