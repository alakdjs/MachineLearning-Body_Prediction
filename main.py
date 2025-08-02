import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

# 데이터 로드
body = pd.read_csv('body.csv')

# 목표 변수와 특징 변수 선택
input = body[['age', 'height', 'weight']].to_numpy()
target = body[['muscle_mass', 'body_fat']].to_numpy()

# 다항 회귀 적용
poly = PolynomialFeatures(degree=10)
input_poly = poly.fit_transform(input)

# 데이터 분할
tr_in, te_in, tr_tar_m, te_tar_m = train_test_split(input_poly, target[:, 0], random_state=42)
_, _, tr_tar_f, te_tar_f = train_test_split(input_poly, target[:, 1], random_state=42)

# 스케일링
ss = StandardScaler()
ss.fit(tr_in)
tr_scaled = ss.transform(tr_in)
te_scaled = ss.transform(te_in)

# 다중회귀 모델 학습 및 평가 (골격근량 예측)
lr_m = LinearRegression()
lr_m.fit(tr_scaled, tr_tar_m)
train_score_m = lr_m.score(tr_scaled, tr_tar_m)
test_score_m = lr_m.score(te_scaled, te_tar_m)

# 결과 출력 (골격근량)
print(f"골격근량 훈련 세트 예측 값 : {train_score_m:.2f}")
print(f"골격근량 테스트 세트 예측 값 : {test_score_m:.2f}")

# 다중회귀 모델 학습 및 평가 (체지방량 예측)
lr_f = LinearRegression()
lr_f.fit(tr_scaled, tr_tar_f)
train_score_f = lr_f.score(tr_scaled, tr_tar_f)
test_score_f = lr_f.score(te_scaled, te_tar_f)

# 결과 출력 (체지방량)
print(f"체지방량 훈련 세트 에측 값 : {train_score_f:.2f}")
print(f"체지방량 테스트 세트 예측 값 : {test_score_f:.2f}")

# Ridge 규제 적용 및 최적의 alpha 값 찾기
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_scores_m = []
cv_scores_f = []

for alpha in alpha_values:
    ridge_cv_m = Ridge(alpha=alpha)
    ridge_cv_f = Ridge(alpha=alpha)

    scores_m = cross_val_score(ridge_cv_m, tr_scaled, tr_tar_m, cv=5)
    scores_f = cross_val_score(ridge_cv_f, tr_scaled, tr_tar_f, cv=5)

    cv_scores_m.append(scores_m.mean())
    cv_scores_f.append(scores_f.mean())

best_alpha_m = alpha_values[cv_scores_m.index(max(cv_scores_m))]
best_alpha_f = alpha_values[cv_scores_f.index(max(cv_scores_f))]

print(f"골격근량 최적의 alpha 값: {best_alpha_m}")
print(f"체지방량 최적의 alpha 값: {best_alpha_f}")

# Ridge 회귀 모델 학습 및 평가
ridge_m = Ridge(alpha=best_alpha_m)
ridge_m.fit(tr_scaled, tr_tar_m)
train_score_m_ridge = ridge_m.score(tr_scaled, tr_tar_m)
test_score_m_ridge = ridge_m.score(te_scaled, te_tar_m)

ridge_f = Ridge(alpha=best_alpha_f)
ridge_f.fit(tr_scaled, tr_tar_f)
train_score_f_ridge = ridge_f.score(tr_scaled, tr_tar_f)
test_score_f_ridge = ridge_f.score(te_scaled, te_tar_f)

# 결과 출력 (Ridge)
print(f"골격근량 훈련 세트 예측 값 (Ridge) : {train_score_m_ridge:.2f}")
print(f"골격근량 테스트 세트 예측 값 (Ridge) : {test_score_m_ridge:.2f}")
print(f"체지방량 훈련 세트 예측 값 (Ridge) : {train_score_f_ridge:.2f}")
print(f"체지방량 테스트 세트 예측 값 (Ridge) : {test_score_f_ridge:.2f}")

# 특성들과 타겟 변수의 이름
features = ['age', 'height', 'weight']
targets = ['muscle_mass', 'body_fat']

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# 산점도 그리기
for i, feature in enumerate(features):
    for j, target in enumerate(targets):
        axs[j, i].scatter(body[feature], body[target], alpha=0.5)
        axs[j, i].set_title(f"{feature} vs {target}", fontsize=8)
        axs[j, i].set_xlabel(feature, fontsize=8)
        axs[j, i].set_ylabel(target, fontsize=8)
        axs[j, i].grid(True)

# 새로운 데이터 예측
new_data = np.array([[26, 180, 73]])  # 예시 데이터
new_data_poly = poly.transform(new_data)
new_data_scaled = ss.transform(new_data_poly)

lr_prediction_m = lr_m.predict(new_data_scaled)
lr_prediction_f = lr_f.predict(new_data_scaled)
ridge_prediction_m = ridge_m.predict(new_data_scaled)
ridge_prediction_f = ridge_f.predict(new_data_scaled)

print(f"나이, 키, 몸무게 입력 데이터 : {new_data}")
print(f"골격근량(kg) 예측 값 (Linear Regression) : {lr_prediction_m[0]:.2f}")
print(f"체지방량(kg) 예측 값 (Linear Regression) : {lr_prediction_f[0]:.2f}")
print(f"골격근량(kg) 예측 값 (Ridge) : {ridge_prediction_m[0]:.2f}")
print(f"체지방량(kg) 예측 값 (Ridge) : {ridge_prediction_f[0]:.2f}")

# plt.show()
