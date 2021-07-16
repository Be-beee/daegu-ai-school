# 데이터 구조
# 4,921 개 학습용 데이터 / 테스트 1320 개

import itertools
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff # !pip install arff
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

file_path = "./dataset"
train_fn = "FordA_TRAIN.arff"
test_fn = "FordA_TEST.arff"


def read_ariff(path):
    raw_data, meta = arff.loadarff(path)
    cols = [x for x in meta]

    data2d = np.zeros([raw_data.shape[0], len(cols)])

    for index, col in zip(range(len(cols)), cols):
        data2d[:, index] = raw_data[col]

    return data2d


train_path = os.path.join(file_path, train_fn)
test_path = os.path.join(file_path, test_fn)
train = read_ariff(train_path)
test = read_ariff(test_path)
# print("train >> ", len(train))
# print("test >>", len(test))



x_train_temp = train[:,:-1]
y_train_temp = train[:,-1] # 마지막 컬럼이 레이블 값

x_test = test[:,:-1]
y_test = test[:,-1]

# print(x_test, y_test)




# 학습용 검증용 테스트용 데이터셋 나누기
normal_x = x_train_temp[y_train_temp==1] # train_x 테이터 중 정상 데이터
abnormal_x = x_train_temp[y_train_temp==-1] # train_x 데이터 중 비정상 데이터

normal_y = y_train_temp[y_train_temp==1]
abnormal_y = y_train_temp[y_train_temp==-1]

# 정상 데이터 8:2
# 정상 데이터를 8:2 나누기 위한 인덱스 설정
ind_x_normal = int(normal_x.shape[0]*0.8)
ind_y_normal = int(normal_y.shape[0]*0.8)
# 비정상 데이터 8:2
ind_x_abnoraml = int(abnormal_x.shape[0]*0.8)
ind_y_abnoraml = int(abnormal_y.shape[0]*0.8)


x_train = np.concatenate((normal_x[:ind_x_normal], abnormal_x[:ind_x_abnoraml]), axis=0) # 80
x_valid = np.concatenate((normal_x[ind_x_normal:], abnormal_x[ind_x_abnoraml:]), axis=0) # 20

y_train = np.concatenate((normal_y[:ind_y_normal], abnormal_y[:ind_y_abnoraml]), axis=0) # 80
y_valid = np.concatenate((normal_y[ind_y_normal:], abnormal_y[ind_y_abnoraml:]), axis=0)





# 데이터 확인
# print("x_tain" , len(x_train))
# print("x_valid" , len(x_valid))
# print("y_train" , len(y_train))
# print("y_valid" , len(y_valid))
# print("x_test", len(x_test))
# print("y_test", len(y_test))




# 시각화

# class 종류 정상 1 비정상 -1
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
# print(classes)

x = np.arange(len(classes)) # plot x 축 개수
lables = ["Abnormal", "Normal"] # plot x 축 이름

# train, valid , test
valuse_train = [(y_train == i ).sum() for i in classes]
valuse_valid = [(y_valid == i ).sum() for i in classes]
valuse_test = [(y_test == i ).sum() for i in classes]

# print(valuse_train, valuse_valid, valuse_test)


plt.figure(figsize = (8,4))
plt.subplot(1,3,1)
plt.title("Train_data")
plt.bar(x, valuse_train, width = 0.6, color=["red" , "blue"])
plt.ylim([0, 1500])
plt.xticks(x, lables)

plt.subplot(1,3,2)
plt.title("val_data")
plt.bar(x, valuse_valid, width = 0.6, color=["red" , "blue"])
plt.ylim([0, 1500])
plt.xticks(x, lables)

plt.subplot(1,3,3)
plt.title("test_data")
plt.bar(x, valuse_test, width = 0.6, color=["red" , "blue"])
plt.ylim([0, 1500])
plt.xticks(x, lables)

# plt.show()

# 시각화 특정 시간에서의 시계열 샘플을 플롯
import random

# 정상 : 1 비정상 : -1
labels = np.unique(np.concatenate((y_train, y_test), axis=0))
# print(labels)

plt.figure(figsize=(10, 4))

for c in labels:
    C_X_train = x_train[y_train == c]
    #     print(C_X_train)
    if c == -1: c = c + 1
    time_t = random.randint(0, C_X_train.shape[0])  # 0 ~ 1404 사이의 랜덤한 정수 특정 time t 가 됨
    plt.scatter(range(0, 500), C_X_train[time_t], label="class = " + str(int(c)),
                marker='o', s=5)

plt.legend(loc="lower right")
plt.xlabel("Sensor", fontsize=15)
plt.xlabel("Sensor", fontsize=15)
# plt.show()


# 특정 시간에서의 시계열 샘플을 (정상 비정상 샘플로 각각 출력)
def get_scatter_plot(c):
    time_t = random.randint(0, c_x_train.shape[0])
    # print("Random time number : ", time_t)

    plt.scatter(range(0, c_x_train.shape[1]), c_x_train[time_t],
                marker='o', s=5, c="r" if c == -1 else "b")
    plt.title(f"at time t_{time_t}", fontsize=20)
    plt.xlabel("Sensor", fontsize=15)
    plt.ylabel("Sensor value", fontsize=15)

    # plt.show()


labels = np.unique(np.concatenate((y_train, y_test)), axis=0)

for c in labels:
    c_x_train = x_train[y_train == c]

    if c == -1:
        # print("비정상 Label number data : ", len(c_x_train))
        get_scatter_plot(c)
    else:
        # print("정상 Label number data : ", len(c_x_train))
        get_scatter_plot(c)






# 시각화 임의의 센서 값의 시계열 show
sensor_number = random.randint(0, 500)
# print(f"random sensor number {sensor_number}")
plt.figure(figsize = (13,4))
plt.title(f"sensor number {sensor_number}", fontsize=20)
plt.plot(x_train[:, sensor_number])
plt.xlabel("time", fontsize=15)
plt.ylabel("Sensor Value" , fontsize=15)
# plt.show()





# 데이터 특성 파악
import matplotlib.cm as cm
from matplotlib.collections import EllipseCollection

df = pd.DataFrame(data=x_train, columns=["sensor_{}".format(label + 1)
                                         for label in range(x_train.shape[-1])])

# print(df)
data = df.corr()


# print(data)

def plot_corr_ellipes(data, ax=None, **kwargs):
    M = np.array(data)

    if not M.ndim == 2:
        return ValueError("data must be a 2D array")
    if ax is None:
        flg, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


fig, ax = plt.subplots(1, 1, figsize=(20, 20))
cmap = cm.get_cmap("jet", 31)
m = plot_corr_ellipes(data, ax=ax, cmap=cmap)
cb = fig.colorbar(m)
cb.set_label("Correlation coefficient")
plt.title("Correlation between Feature")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.tight_layout()
# plt.show()





from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler




"""
동일 시간 길이(3,600) 내 센서 값들이 상당히 넓은 범위로 퍼져 있을 뿐만 아니라, 변
수 간의 Scale이 서로 다르기 때문에, 데이터를 그대로 학습하는 것은 일반적으로 적
절하지 않다. 따라서 인풋 값들을 정규화(Normalization) 과정을 거치는데,
StandardScaler 또는 RobustScaler를 통해 진행한다.
흔히 공정 데이터에 이상치(Outlier)가 발생할 수 있는데 이에 강건한 정규화가 필요
할 때가 있다. 이때 RobustScaler를 사용한다. StandardScaler는 보다 더 일반적으
로 많이 사용하는 정규화 방법으로, 데이터를 단위 분산으로 조정함으로써 Outlier에
취약할 수 있는 반면, RobustScaler는 Feature 간 은 스케일을 갖게 되지만 평균과
분산 대신 중간 값(median)과 사분위값(quartile)을 사용함으로써, 극단값(Outlier)
에 영향을 받지 않는 특징이 있다.
"""

# Stander
stder = StandardScaler()
stder.fit(x_train)
x_train = stder.transform(x_train)
x_valid = stder.transform(x_valid)
# print(x_train, x_valid)

# RobustScaler
# rscaler = RobustScaler()
# rscaler.fit(x_train)
# x_train = rscaler.transform(x_train)
# x_valid = rscaler.transform(x_valid)
# print(x_train, x_valid)




from sklearn.linear_model import LogisticRegression
clf_lr_1 = LogisticRegression(
    penalty = 'l2',
    C=1,
    fit_intercept = True,
    intercept_scaling = 1,
    random_state = 2,
    solver='lbfgs',
    max_iter = 1000,
    multi_class = 'auto',
    verbose=0
)


# numpy 로 직접 구현
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.eps = 1e-10
        self.threshold = 0.5
        self.loss_history = list()

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h + self.eps) - (1 - y) * np.log(1 - h + self.eps)).mean()

    # fit() 학습 데이터로 model 을 학습하는 메서드
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            logit = np.dot(X, self.theta)
            hypothesis = self.__sigmoid(logit)
            gradint = np.dot(X.T, (hypothesis - y)) / y.size
            self.theta -= self.lr * gradint

            if self.verbose == True and i % 10 == 0:
                loss = self.__loss(hypothesis, y)
                print(f"epoch : {i} \t loss:{loss} \t")
                self.loss_history.append(loss)

        return self.loss_history

    # 학습 데이터로 학습된 모델을 바탕으로 테스트 데이터의 각 인스턴스의 정상일 확률을 도출하는 메서드
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

        # 학습 데이터로 학습된 모델을 바탕으로 테스트 데이터 라벨을 확인 하는 메서드

    def prdict(self, X):
        prdict_labels = np.where(self.predict_prob(X) > self.threshold, 1, 0)
        return prdict_labels

    # 모델 테스트 정확도 도출
    def eval(self, x, y):
        res_y = np.round(self.predict_prob(x), 0)
        accuracy = np.sum(res_y == y) / len(y)
        return accuracy





x_train_lr = np.concatenate((x_train, x_valid), axis=0)
y_train_lr = np.concatenate((y_train, y_valid), axis=0)




# sklearn 로지스틱 회귀 학습
clf_lr_1.fit(x_train_lr, y_train_lr)





# test
y_pred = clf_lr_1.predict(x_test)
score = clf_lr_1.score(x_test, y_test)
print("Logisic Regression Prediction Rate : ", round(score*100 , 2), "%")




clf_lr_2 = LogisticRegression(lr=0.01, num_iter=1000, verbose=True)
history_lr = clf_lr_2.fit(x_train_lr, y_train_lr)




score = clf_lr_2.eval(x_test, y_test)
print("Logisic Regression Prediction Rate : ", round(score*100 , 2), "%")