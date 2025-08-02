# MachineLearning-Body_Prediction
머신러닝-신체스펙에 따른 건강 예측 모델(골격근량,체지방량). Python - 2024(3-1)


# 머신러닝: 신체 정보 기반 골격근량 및 체지방량 예측

이 프로젝트는 신체 데이터(나이, 키, 몸무게)를 바탕으로 골격근량과 체지방량을 예측하는 머신러닝 모델을 구현한 예제입니다.

## 📂 파일 구성
- `main.py` : 전체 머신러닝 모델 학습, 평가, 시각화 및 예측 코드
- `body.csv` : 입력 데이터 (age, height, weight, muscle_mass, body_fat)

## 📌 주요 내용
- 다항 회귀(Polynomial Regression) 적용
- StandardScaler로 정규화
- Ridge Regression + 교차검증을 통한 하이퍼파라미터 튜닝
- 입력값에 따른 예측 결과 출력 및 시각화

## ⚙️ 사용된 라이브러리
- numpy, pandas  
- scikit-learn  
- matplotlib

## 🧪 예측 예시
```python
입력값: [26세, 180cm, 73kg]
→ 골격근량 예측: 28.32kg (Ridge)
→ 체지방량 예측: 14.27kg (Ridge)
