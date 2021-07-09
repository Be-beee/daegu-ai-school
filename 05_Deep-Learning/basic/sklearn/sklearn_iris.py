from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import numpy as np


iris = load_iris()
iris_data = iris.data
# print(iris_data[:5])
# print(iris.target[:5])
# print(iris.target_names)
# print(iris.feature_names)
# print(iris.filename)

iris_label = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.25, shuffle=True, stratify=iris_label, random_state=40)
plt.plot(x_train, y_train, 'o')
plt.title('train data')
plt.show()
# exit()

# LogisticRegression

clf = LogisticRegression(max_iter=20000)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)

plt.title('test data')
plt.plot(x_test, y_test, 'o')
plt.show()

plt.title('prediction data')
plt.plot(x_test, pred, 'o')
plt.show()

rig = 0
for p, t in zip(pred, y_test):
    if p == t:
        rig += 1

print(f"{rig/len(pred) * 100}%")
