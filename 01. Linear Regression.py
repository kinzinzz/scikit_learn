# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 1. Linear Regression
# ### 공부 시간에 따른 시험 점수

import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')

dataset.head()

X = dataset.iloc[:,:-1].values # row, column 처음부터 마지막 컬럼 직전까지의 데이터(독립 변수)
y = dataset.iloc[:,-1].values # 마지막 컬럼 데이터(종속 변수 - 결과)
X, y

from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X, y) # 학습 (모델 생성)

y_pred = reg.predict(X) # X에 대한 예측 값
y_pred

plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, y_pred, color='green')
plt.title('Score by hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

print('9시간 공부했을 때 예상 점수 :', reg.predict([[9]]))

reg.coef_ # 기울기(m)

reg.intercept_ # y절편(b)

# ### 데이터 세트 분리

import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv('LinearRegressionData.csv')
dataset

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # 훈련 80 : 테스트 20 으로 분리

X, len(X) # 전체 데이터 X, 개수

X_train, len(X_train) # 훈련 세트 X, 개수

X_test, len(X_test) # 테스트 세트 X, 개수

y, len(y) # 전체 데이터 y

y_train, len(y_train) # 훈련 세트 y

y_test, len(y_test) # 테스트 세트 y

# ### 분리된 데이터를 통한 모델링

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X_train, y_train) # 훈련 세트로 학습

# ### 데이터 시각화(훈련 세트)

plt.scatter(X_train, y_train, color='blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Score by hours(train data)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

# ### 데이터 시각화(테스트)

plt.scatter(X_test, y_test, color='blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color='green')
plt.title('Score by hours(test data)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

reg.coef_

reg.intercept_

# ### 모델 평가

reg.score(X_test, y_test) # 테스트 세트를 통한 모델 평가

reg.score(X_train, y_train)

# ## 경사 하강법(Gradient Descent)

# max_iter : 훈련 세트 반복 횟수( Epoch 횟수)
# eta0 : 학습률(learning rate)

from sklearn.linear_model import SGDRegressor # SGD : 확률적 경사 하강법
sr = SGDRegressor(max_iter=500, eta0=1e-4, random_state=0, verbose=1)
sr.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='blue') # 산점도
plt.plot(X_train, sr.predict(X_train), color='green')
plt.title('Score by hours(train data, SGD)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

sr.coef_, sr.intercept_

sr.score(X_test, y_test)

sr.score(X_train, y_train)
