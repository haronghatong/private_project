import os
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.utils import shuffle
import re  # 정규식 사용

# 1. 학습 데이터 준비 및 모델 학습
# CSV 파일 경로 설정
extract_path = "C:/Users/pythonfile/SSD_remain"  # 경로 변경

# CSV 파일 찾기
csv_files = [f for f in os.listdir(extract_path) if f.endswith(".csv")]

if not csv_files:
    print("CSV 파일을 찾을 수 없습니다.")
    exit(1)

# CSV 파일 읽기
csv_path = os.path.join(extract_path, csv_files[0])
df = pd.read_csv(csv_path)

# 데이터 섞기
df = shuffle(df)

# 실제로 사용할 수 있는 중요 컬럼만 선택
important_columns = [
    'capacity_bytes', 
    'failure',          # 수정: 이후 제거할 컬럼
    'smart_9_raw',      # Power-On Hours (전원 켜진 시간)
    'smart_241_raw',    # Total LBAs Written
    'smart_242_raw'     # Total LBAs Read
]

# 데이터 전처리
df_filtered = df[important_columns].copy()
df_filtered = df_filtered.fillna(0)  # 결측치 처리
df_filtered = df_filtered.astype('float64')  # 데이터 타입 변환

# 목표 변수 (남은 사용 가능 시간) 생성
max_usage_time = df_filtered["smart_9_raw"].max()
df_filtered["remaining_lifetime"] = (
    max_usage_time - df_filtered["smart_9_raw"]
) * (1 + np.random.uniform(0.9, 1.1, len(df_filtered)))  # 랜덤 스케일 범위 축소

# 랜덤 노이즈 추가 (데이터 다양성 증가)
df_filtered["smart_241_raw"] += np.random.normal(0, 0.05, len(df_filtered))
df_filtered["smart_242_raw"] += np.random.normal(0, 0.05, len(df_filtered))

# X (입력 데이터), y (출력 라벨) 설정
X = df_filtered.drop(columns=['failure', 'remaining_lifetime'])
y = df_filtered["remaining_lifetime"]

# 학습/테스트 데이터 분할 (랜덤 샘플링, 80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 데이터 스케일링 (표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet 모델 학습
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.7)
elastic_net.fit(X_train_scaled, y_train)

# 2. 모델 평가 함수 정의
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📌 {model_name} 모델 평가 결과:")
    print(f"  - MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  - RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  - R² Score: {r2:.4f}")

# 모델 평가
elastic_net_pred = elastic_net.predict(X_test_scaled)
evaluate_model(y_test, elastic_net_pred, "ElasticNet")

# 3. 디스크 정보 파일 예측
# 예측을 진행할 파일 경로
disk_info_path = "C:/Users/pythonfile/SSD_remain/disk_info_*.txt"  # 경로 변경
disk_info_files = glob.glob(disk_info_path)

if not disk_info_files:
    print("disk_info_*.txt 파일을 찾을 수 없습니다.")
    exit(1)

# disk_info_*.txt 파일에서 필요한 성분 추출 함수
def parse_disk_info_v2(file_path):
    """disk_info_*.txt 파일에서 필요한 성분 추출"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    disk_info = {}
    for line in lines:
        if ':' in line:
            key, value = line.strip().split(':', 1)
            disk_info[key.strip()] = value.strip()
    
    # 필요한 성분 추출
    capacity_gb = float(disk_info['total'].replace(' GB', ''))  # total -> capacity_bytes
    install_time = disk_info['install_time']  # 설치 날짜
    read_bytes_gb = float(disk_info['read_bytes'].replace(' GB', ''))  # read_bytes
    write_bytes_gb = float(disk_info['write_bytes'].replace(' GB', ''))  # write_bytes
    percent = float(disk_info['percent'])  # 사용률(%)
    
    return capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent


# 전처리 함수
def preprocess_disk_info_v2(capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent):
    """추출한 데이터를 모델 입력 형식에 맞게 전처리"""
    # total -> capacity_bytes (GB -> Bytes 변환)
    capacity_bytes = int(capacity_gb * (1024**3))
    
    # install_time (설치 날짜 -> 사용 시간 계산)
    try:
        # 날짜만 추출 (정규식으로 "YYYY년 MM월 DD일" 패턴 찾기)
        date_match = re.search(r'\d{4}년 \d{1,2}월 \d{1,2}일', install_time)
        if date_match:
            install_date = datetime.strptime(date_match.group(), '%Y년 %m월 %d일')
        else:
            raise ValueError(f"Unknown date format: {install_time}")
    except Exception as e:
        raise ValueError(f"Invalid date format in install_time: {install_time}, Error: {str(e)}")
    
    # 현재 날짜와 설치 날짜를 비교해 사용 시간 계산
    today = datetime.now()
    hours_used = (today - install_date).total_seconds() / 3600  # 사용 시간 (시간 단위)
    
    # read_bytes -> daily usage 계산
    daily_usage = (read_bytes_gb / (hours_used / 24)) * 1024  # 하루 평균 사용량 (MB)
    
    # years_used 계산
    years_used = hours_used / (24 * 365)
    
    return capacity_bytes, hours_used, daily_usage, years_used, install_date, percent


def predict_lifetime_v2(file_path, model, scaler):
    """디스크 정보 파일에서 수명 예측 및 결과 저장"""
    try:
        # 1) 파일에서 성분 추출
        capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent = parse_disk_info_v2(file_path)
        
        # 2) 성분 전처리
        capacity_bytes, hours_used, daily_usage, years_used, install_date, usage_percent = preprocess_disk_info_v2(
            capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent
        )
        
        # 3) 모델 입력 데이터 구성 (4개의 피처 포함)
        input_data = np.array([[capacity_bytes, hours_used, daily_usage, usage_percent]])
        input_scaled = scaler.transform(input_data)  # 스케일링
        
        # 4) 예측 실행
        predicted_hours = model.predict(input_scaled)[0]  # 잔여 수명 예측
        
        #총 예상 수명 = 잔여 수명 + 사용 시간
        remaining_lifetime = predicted_hours / (24 * 365) # 잔여 수명 (년 단위)
        total_lifetime = (predicted_hours + hours_used) / (24 * 365)  # 총 예상 수명 (년 단위)
        
        # remaining_lifetime이 0 이하로 떨어질 경우 0으로 고정
        if remaining_lifetime < 0:
            remaining_lifetime = 0
            total_lifetime = hours_used / (24 * 365)

        # JSON 데이터 생성
        prediction_data = {
            "remaining_lifetime": (remaining_lifetime),
            "daily_usage": daily_usage,
            "years_used": years_used,
            "total_lifetime": total_lifetime,
            "manufacture_date": install_date.strftime('%Y-%m-%d'),
            "usage_percent": usage_percent
        }
        
        print(f"\n📂 파일: {file_path}")
        print(json.dumps(prediction_data, indent=2, ensure_ascii=False))
        
        return prediction_data
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


# 모든 파일에 대해 예측 실행
results = []
for file_path in disk_info_files:
    result = predict_lifetime_v2(file_path, elastic_net, scaler)
    if result:
        results.append({"file": file_path, "prediction_data": result})
        
        # 결과를 JSON 파일로 저장
        output_file = file_path.replace('.txt', '_prediction.json')  # 파일 이름 형식 변경
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 예측 결과가 {output_file}에 저장되었습니다.")

# # 전체 결과도 저장 (필요한 경우)
# output_path = "C:/Users/Desktop/prediction_results.json"  # 경로 변경
# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ 모든 파일에 대한 예측이 완료되었습니다.")
# print(f"전체 결과가 {output_path}에 저장되었습니다.")

