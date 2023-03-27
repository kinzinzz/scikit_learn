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

# # 3. Polynomial Regression

# ### 공부 시간에 따른 시험 점수(우등생)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('PolynomialRegressionData.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# ### 3-1. 단순 선형 회귀(simple Linear Regression)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)

# ### 데이터 시각화(전체)

plt.scatter(X, y, color='blue') # 산점도
plt.plot(X, reg.predict(X), color='green') # 선 그래프
plt.title('Score by hours(geninus)') 
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

reg.score(X, y) # 모델 평가

# ## 3-2. 다항 회귀(Polynomial Regression)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # n차 다항식
X_poly = poly_reg.fit_transform(X)
X_poly[:5]

X[:5]

poly_reg.get_feature_names_out()

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y) # 변환된 X와 y를 가지고 모델 학습

# ### 데이터 시각화(변환된 X, y)

plt.scatter(X, y, color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='green')
plt.title('Score by hours(geninus)') 
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

X_range = np.arange(min(X),max(X), 0.1) # X를 0.1 단위로 데이터 생성
X_range

X_range.shape

X[:5]

X_range = X_range.reshape(-1, 1) # -1: row 갯수 자동 계산, column 갯수는 1개
X_range.shape

X_range[:5]

plt.scatter(X, y, color='blue')
plt.plot(X_range, lin_reg.predict(poly_reg.fit_transform(X_range)), color='green')
plt.title('Score by hours(geninus)') 
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

# ### 공부 시간에 따른 시험 성적 예측

reg.predict([[2]]) # 2시간 공부했을 때 선형 회귀 모델의 예측

lin_reg.predict(poly_reg.fit_transform([[2]])) # 2시간을 공부했을 때 다항회귀 모델의 예측

lin_reg.score(X_poly, y)
