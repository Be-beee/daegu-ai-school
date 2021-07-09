import numpy as np
import matplotlib.pyplot as plt

data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1]]
x = [i[0] for i in data]
y = [i[1] for i in data]

# plt.figure(figsize=(8,5))
# plt.scatter(x, y)
# plt.show()
######################################################################3
a = 0
b = 0

lr = 0.03
epochs = 2001

def sigmoid(x):
    return 1 / (1+np.e ** (-x))

for i in range(epochs):
    for x_data, y_data in data:
        a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
        b_diff = sigmoid(a * x_data + b) - y_data

        a = a - lr * a_diff
        b = b - lr * b_diff

        if i % 100 == 0:
            print(f"epoch = {i}, diff = {a} bias = {b}")
#############################################################################

plt.scatter(x, y)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
x_range = (np.arange(0, 15, 0.1))
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a*x+b) for x in x_range]))
plt.show()
