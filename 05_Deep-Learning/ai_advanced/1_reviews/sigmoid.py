import numpy as np
import matplotlib.pyplot as plt

# [-10, 10]
t = np.linspace(-10, 10, 100)

# 시그모이드 공식
# Numpy np.exp() 함수는 밑이 자연상수 e인 지수함수로 변환해줍니다.
sig = 1 / (1 + np.exp(-t))

# t와 시그모이드 결과값을 plotting
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:") # y축 기준 0.5 실선 생성
plt.plot([-10, 10], [1, 1], "k:")  # y축 기준 1.0 실선 생성
plt.plot([0, 0], [-1.1, 1.1], "k-")  # center 기준 0.0 선 생성
plt.plot(t, sig, "r-", linewidth=2, label=r"$\sigma(t)= \frac{1}{1 + e^{-t}}$")
plt.xlabel("t") # x축 이름
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
plt.show()



# w 값 변화에 따른 경사도 변화
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


x = np.arange(-5.0, 5.0, 0.1)
# print(x)
y1 = sigmoid(0.5 * x)
y2 = sigmoid(x)
y3 = sigmoid(2 * x)

print("\n y1 sigmoid \n", y1)
print("\n y2 sigmoid \n", y2)
print("\n y3 sigmoid \n", y3)


plt.plot(x, y1, 'r', linestyle="--")  # w == 0.5
plt.plot(x, y2, 'g')  # w == 1
plt.plot(x, y3, 'b', linestyle="--")  # w == 2

plt.plot([0, 0], [1.0, 0.0])
plt.title("sigmoid")
plt.show()



# b 값의 변화에 따른 좌 우 이동

# def sigmoid(t):
#     return 1 / (1 + np.exp(-t))

x = np.arange(-5.0, 5.0, 0.1)

y1 = sigmoid(x + 0.5)
y2 = sigmoid(x + 1)
y3 = sigmoid(x + 1.5)

plt.plot(x, y1, 'r', linestyle="--")  # w == 0.5
plt.plot(x, y2, 'g')  # w == 1
plt.plot(x, y3, 'b', linestyle="--")  # w == 2

plt.plot([0, 0], [1.0, 0.0])
plt.title("sigmoid")
plt.show()