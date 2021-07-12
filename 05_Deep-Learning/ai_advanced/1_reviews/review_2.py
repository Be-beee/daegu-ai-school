# 자동미분 실습
import torch

# 값이 2인 임의의 스칼라 텐서 w를 선언할 때 required_grad true
# 이는 이 텐서에 대한 기울기 저장을 의미

w = torch.tensor(2.0, requires_grad=True)

# 2w^2 + 5 -> 8
# 수식을 정의
y = w ** 2
z = 2 * y + 5


# 이제 해당 수식을 w에 대해서 미분 backward 호출하면 해당 수식의 w에 대한 기울기 계산
z.backward()
print(f"수식을 w로 미분한 값 : {w.grad}")
