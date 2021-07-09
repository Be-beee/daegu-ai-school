from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
import numpy as np

# linear

# x_train = [[1], [2], [3], [4]]
# y_train = [2, 4, 6, 8]
#
# plt.plot(x_train, y_train, 'o')
# plt.plot(x_train, y_train)
# plt.show()
#
# x_test = [[5], [6], [7], [8]]
# y_test = [10, 11, 14, 16]
#
# clf = LinearRegression()
# clf.fit(x_train, y_train)
# pred = clf.predict(x_test)
# print(pred)
# plt.plot(x_test, y_test, 'o')
# plt.plot(x_test, pred)
# plt.show()
# exit()


# logistic

x_train = [[2], [4], [6], [8], [10], [12]]
y_train = [0, 0, 0, 1, 1, 1]

plt.plot(x_train, y_train, 'o')
plt.show()

x_test = [[1], [3], [5], [9], [11], [13]]
y_test = [0, 1, 0, 1, 1, 1]

clf = LogisticRegression(max_iter=20000)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
plt.plot(x_test, y_test, 'o')
x_range = (np.arange(0, 15, 0.1))
x_range_test = x_range[:, np.newaxis]
print(x_range_test)

pred = clf.predict(x_range_test)
plt.plot(x_range, pred)
plt.show()