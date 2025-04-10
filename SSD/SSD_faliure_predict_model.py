import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib  # 모델 저장/로드용 라이브러리

# 학습 데이터 준비
extract_path = "C:/Users/데이터텍/pythonfile/SSD_remain"
csv_files = [f for f in os.listdir(extract_path) if f.endswith(".csv")]

if not csv_files:
    print("CSV 파일을 찾을 수 없습니다.")
    exit(1)

csv_path = os.path.join(extract_path, csv_files[0])
df = pd.read_csv(csv_path)
df = shuffle(df)

# 중요 특성 선택
important_columns = [
    'capacity_bytes',
    'smart_9_raw',      # Power-On Hours
    'smart_241_raw',    # Total LBAs Written
    'smart_242_raw',    # Total LBAs Read
    'failure'           # 고장 여부 (target)
]

# 데이터 전처리
df_filtered = df[important_columns].copy()
df_filtered = df_filtered.fillna(0)
df_filtered = df_filtered.astype('float64')

# 특성 엔지니어링: 추가 특성 생성
df_filtered['write_intensity'] = df_filtered['smart_241_raw'] / (df_filtered['smart_9_raw'] + 1)
df_filtered['read_write_ratio'] = df_filtered['smart_242_raw'] / (df_filtered['smart_241_raw'] + 1)
df_filtered['usage_rate'] = df_filtered['smart_241_raw'] / df_filtered['capacity_bytes']

# 극단값 제한
df_filtered['write_intensity'] = np.clip(
    df_filtered['write_intensity'], 
    0, 
    np.percentile(df_filtered['write_intensity'], 95)
)
df_filtered['read_write_ratio'] = np.clip(
    df_filtered['read_write_ratio'], 
    0, 
    np.percentile(df_filtered['read_write_ratio'], 95)
)

# X (입력 특성), y (고장 여부) 설정
X = df_filtered.drop(columns=['failure'])
y = df_filtered['failure']

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 특성 스케일링 및 모델 파이프라인 설정
scaler = RobustScaler(quantile_range=(5, 95))  # 이상치의 영향을 더 줄임
base_svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='auto',
    probability=True,
    random_state=42,
    class_weight={0: 1, 1: 0.2}  # 고장 클래스의 가중치를 낮게 설정
)

# 캘리브레이션된 SVM 모델 생성
calibrated_svm = CalibratedClassifierCV(
    base_svm,
    cv=5,
    method='sigmoid'
)

# Pipeline 생성
model_pipeline = Pipeline([
    ('scaler', scaler),
    ('classifier', calibrated_svm)
])

# 모델 학습
model_pipeline.fit(X_train, y_train)

# 학습된 모델 저장
model_path = "C:/Users/데이터텍/pythonfile/SSD_remain/svm_failure_model.pkl"
joblib.dump(model_pipeline, model_path)
print(f"✅ 학습된 모델이 {model_path}에 저장되었습니다.")

# 모델 평가
def evaluate_model(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print("\n📊 모델 평가 결과:")
    print(f"  - 정확도 (Accuracy): {accuracy:.4f}")
    print(f"  - 정밀도 (Precision): {precision:.4f}")
    print(f"  - 재현율 (Recall): {recall:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    print(f"  - ROC AUC Score: {roc_auc:.4f}")

# 테스트 데이터에 대해 예측
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

# 평가 함수 호출
evaluate_model(y_test, y_pred, y_prob)
