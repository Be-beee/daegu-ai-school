# -*- coding: utf-8 -*-
"""Matplotlib.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yhoNB5ZdGCK3CReqV19y5wtzfpWjMXSK

# Matplotlib

Matplotlib은 파이썬의 시각화 도구이다. Matplotlib은 NumPy 배열을 기반으로 만들어져 있으며 SciPy와 함께 사용하기 좋게 설게 되었다.
다양한 운영체계와 그래픽 백엔드에서도 잘 동작한다.
"""

import matplotlib as mpl

import matplotlib.pyplot as plt

"""# Style 적용하기"""

plt.style.use('classic')

"""![mat-1](images/mat-1.png)"""

import numpy as np

"""linspace( ) 함수는 파이썬의 numpy 모듈에 포함된 함수로서 1차원의 배열 만들기, 그래프 그리기에서 수평축의 간격 만들기 등에 매우 편리하게 사용할 수 있는 함수입니다. 이름에서 알 수 있듯이 Linearly Spaced의 줄임말인 것으로 생각되며, 이 시리즈의 [P026]편에서 간단히 언급한 적이 있는 함수입니다.

사용법은 numpy 모듈을 importing 한 후에 x=np.linspace(start, stop, num)과 같이 적으면 됩니다. start는 배열의 시작값, stop은 배열의 끝값이고, num은 start와 stop 사이를 몇 개의 일정한 간격으로 요소를 만들 것인지를 나타내는 것입니다. 만일 num을 생략하면 디폴트(Default)로 50개의 수열, 즉 1차원 배열을 만들어줍니다.
"""

x = np.linspace(0, 10, 100)

x

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')
fig.savefig('my_figure.png')

fig.canvas.get_supported_filetypes()

fig2 = plt.figure()

# subplot(rows, columns, panel number)
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x)) # 데이터의 시각화

"""# 간단한 라인 플룻

np.linspace()에 대한 참조 링크 ([https://m.blog.naver.com/choi_s_h/221730568009](https://m.blog.naver.com/choi_s_h/221730568009))<br>
linspace() 함수는 파이썬의 NumPy에 포함된 함수로 1차원의 배열 만들기 혹은 그래프 그리기에서 수평축의 간격 만들기 등에 매우 편리하게 사용할 수 있는 함수 있다. 사용방법은 x = np(staar, stop, num)과 같은 형태로 사용된다. 이때 num 파라메터를 생략하게 되면 기본적으로 50개의 수열을 생성해 준다.
"""

plt.style.use('seaborn-whitegrid')

fig3 = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

"""## 선색상과 스타일"""

plt.plot(x, np.sin(x-0), color='blue') # 색상을 이름으로 지정
plt.plot(x, np.sin(x-1), color='g') # 짧은 색상 코드
plt.plot(x, np.sin(x-2), color='0.5') # 0 ~ 1사이의 gray scale
plt.plot(x, np.sin(x-3), color='#ffdd44') # 16진수로 표현한 RGB 컬러코드
plt.plot(x, np.sin(x-4), color=(1.0, 0.2, 0.3)) # RGB를 튜플로 표현
plt.plot(x, np.sin(x-5), color='chartreuse') # html에서 정해진 색상 이름

plt.plot(x, x+0, linestyle='solid')
plt.plot(x, x+1, linestyle='dashed')
plt.plot(x, x+2, linestyle='dashdot')
plt.plot(x, x+3, linestyle='dotted')

plt.plot(x, x+4, linestyle='-')
plt.plot(x, x+5, linestyle='--')
plt.plot(x, x+6, linestyle='-.')
plt.plot(x, x+7, linestyle=':')

plt.plot(x, x+8, '-g')
plt.plot(x, x+9, '--c') # cyan
plt.plot(x, x+10, '-.k')
plt.plot(x, x+11, ':r')

"""## 플롯 조정하기: 축 경계

축 경계를 세밀하게 제어 할 때 사용할 수 있는 함수가 xlim(), ylim()이 있다.
"""

plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

plt.plot(x, np.sin(x))

plt.xlim(10, 0)
plt.ylim(1.2, -1.2) # 범위 순서 역순 -> 그래프도 뒤집혀서 출력

plt.plot(x, np.sin(x))

plt.axis('tight') # 그래프를 화면에 꽉차게 그리는 옵션

plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]) #axis([xlim, ylim])

plt.plot(x, np.sin(x))

plt.axis('equal')

"""## 플롯에 레이블 붙이기

제목과 x, y축에 레이블 그리고 범례를 붙이는 방법을 살펴본다.
"""

plt.plot(x, np.sin(x))
plt.title('A Sine Curve')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend()

"""## 산점도의 출력

라인 플룻에 이어서 점으로 표시하는 산점도(Scatter plot)는 점이나 다른 도형으로 표현한다.
"""

rng = np.random.RandomState(0) # 난수 생성

for marker in ['o', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker, label='marker='+marker)
    plt.legend()

rng.rand(5) # 호출할 때마다 랜덤 값

rng.rand(5)

plt.plot(x, y, '-p', color='gray', markersize=15, linewidth=4, 
         markerfacecolor='white', markeredgecolor='gray', markeredgewidth=2) # p = pentagon

"""# plt.scatter를 이용한 산점도의 표현"""

rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
color = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=color, s=sizes, alpha=0.3, cmap='viridis') # viridis
plt.colorbar()

color

"""# Iris Dataset

References: [Iris Dataset](https://bishwamittra.github.io/imli.html)
"""

from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data.T

plt.scatter(features[0], features[1], s=100 * features[3], alpha=0.2, c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

"""# 오차 시각화 하기"""

x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3)