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

# # 4. Logistic Regression

# ### 공부 시간에 따른 자격증 시험 합격 가능성

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('LogisticRegressionData.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# ## 데이터 분리

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# ## n시간 공부했을 때 예측?

classifier.predict([[6]])
# 결과 1 : 합격할 것으로 예측

classifier.predict_proba([[6]]) # 합격할 확률(불합격 확률, 합격할 확률)

classifier.predict([[4]])
# 결과 0 : 불합격할 것으로 예측

classifier.predict_proba([[4]])

# ### 분류 결과에 예측(테스트 세트)

y_pred = classifier.predict(X_test)
y_pred # 예측 값

y_test # 실제 값

X_test

classifier.score(X_test, y_test) # 모델 평가
# 전체 테스트 세트 4개 중에서 분류 예측을 올바로 맞힌 갯수 3/4 = 0.75

# ### 데이터 시각화(훈련 세트)

X_range = np.arange(min(X), max(X), 0.1)
X_range

p = 1 / (1 + np.exp(-(classifier.coef_ * X_range + classifier.intercept_))) # y = mx + b 
p

p.shape, X_range.shape

p = p.reshape(-1) # 1차원 배열 형태로 변경
p.shape

plt.scatter(X_train, y_train, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # 상수 함수
plt.title('Probability by hours')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()

# ### 데이터 시각화(테스트 세트)

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_range, p, color='green')
plt.plot(X_range, np.full(len(X_range), 0.5), color='red') # 상수 함수
plt.title('Probability by hours(test)')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()

classifier.predict_proba([[4.5]]) # 4.5 시간 공부했을 때 모델에서는 합격 예측, 실제로는 불학격

# ## 혼동 행렬(Confussion Matrix)

# +
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# (0,0)불합격 예측 : 실제 불합격 1개
# (0,1)합격 예측 : 실제 불합격 1개

# (1,0)불합격 예측 : 실제로 합격 0개
# (1,1)합격 예측 : 실제로 합격 2개

