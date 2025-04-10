import os
import glob
import json
import numpy as np
import joblib  # 모델 로드용 라이브러리
from datetime import datetime
import re

# 학습된 모델 로드
model_path = "C:/Users/pythonfile/SSD_remain/svm_failure_model.pkl"  # 올려주신 모델 파일 경로
if not os.path.exists(model_path):
    print(f"모델 파일 {model_path}을 찾을 수 없습니다. 올바른 경로를 확인하세요.")
    exit(1)

# 모델 로드
model_pipeline = joblib.load(model_path)
print(f"✅ 학습된 모델이 {model_path}에서 로드되었습니다.")

# 디스크 정보 파싱 함수
def parse_disk_info(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    disk_info = {}
    for line in lines:
        if ':' in line:
            key, value = line.strip().split(':', 1)
            disk_info[key.strip()] = value.strip()
    
    capacity_gb = float(disk_info['total'].replace(' GB', ''))
    install_time = disk_info['install_time']
    read_bytes_gb = float(disk_info['read_bytes'].replace(' GB', ''))
    write_bytes_gb = float(disk_info['write_bytes'].replace(' GB', ''))
    percent = float(disk_info['percent'])
    
    return capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent

# 디스크 정보 전처리 함수
def preprocess_disk_info(capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent):
    capacity_bytes = int(capacity_gb * (1024**3))
    
    try:
        date_match = re.search(r'\d{4}년 \d{1,2}월 \d{1,2}일', install_time)
        if date_match:
            install_date = datetime.strptime(date_match.group(), '%Y년 %m월 %d일')
        else:
            raise ValueError(f"Unknown date format: {install_time}")
    except Exception as e:
        raise ValueError(f"Invalid date format in install_time: {install_time}, Error: {str(e)}")
    
    today = datetime.now()
    hours_used = (today - install_date).total_seconds() / 3600
    
    # SMART 값 계산
    smart_9_raw = hours_used
    smart_241_raw = write_bytes_gb * (1024**3)
    smart_242_raw = read_bytes_gb * (1024**3)
    
    # 추가 특성 계산
    write_intensity = smart_241_raw / (smart_9_raw + 1)
    read_write_ratio = smart_242_raw / (smart_241_raw + 1)
    usage_rate = percent / 100
    
    # 사용 기간 계산
    total_years = smart_9_raw / (24 * 365)  # 총 사용 연도
    years = int(total_years)  # 정수 부분: 연도
    months = round((total_years - years) * 12)  # 소수점 부분을 반올림하여 개월로 변환
    if months == 12:
        years += 1
        months = 0
    
    return {
        'capacity_bytes': capacity_bytes,
        'smart_9_raw': smart_9_raw,
        'smart_241_raw': smart_241_raw,
        'smart_242_raw': smart_242_raw,
        'write_intensity': write_intensity,
        'read_write_ratio': read_write_ratio,
        'usage_rate': usage_rate,
        'usage_years': f"{years}년 {months}개월",
        'install_date': install_date
    }

# 예측 함수
def predict_failure_probability(file_path, model_pipeline):
    try:
        capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent = parse_disk_info(file_path)
        processed_data = preprocess_disk_info(capacity_gb, install_time, read_bytes_gb, write_bytes_gb, percent)
        
        input_data = np.array([[
            processed_data['capacity_bytes'],
            processed_data['smart_9_raw'],
            processed_data['smart_241_raw'],
            processed_data['smart_242_raw'],
            processed_data['write_intensity'],
            processed_data['read_write_ratio'],
            processed_data['usage_rate']
        ]])
        
        failure_prob = model_pipeline.predict_proba(input_data)[0][1]
        risk_level = (
            "낮음" if failure_prob < 0.01 else
            "중간" if failure_prob < 0.03 else
            "높음"
        )
        
        prediction_data = {
            "failure_probability": float(failure_prob),
            "failure_probability_percent": float(failure_prob * 100),
            "risk_level": risk_level,
            "usage_years": processed_data['usage_years'],
            "usage_rate": float(processed_data['usage_rate']),
            "capacity_gb": capacity_gb,
            "install_time": install_time,
            "read_bytes_gb": read_bytes_gb,
            "write_bytes_gb": write_bytes_gb,
            "percent_used": percent
        }
        
        return prediction_data
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# 모든 파일에 대해 예측 실행
def run_predictions(input_path, output_path):
    disk_info_files = glob.glob(input_path)

    if not disk_info_files:
        print("disk_info_*.txt 파일을 찾을 수 없습니다.")
        return []

    results = []
    for file_path in disk_info_files:
        result = predict_failure_probability(file_path, model_pipeline)
        if result:
            results.append({"file": file_path, "prediction_data": result})
            
            # 개별 파일로 저장
            output_file = os.path.join(output_path, os.path.basename(file_path).replace('.txt', '_failure_prediction.json'))
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ 예측 결과가 {output_file}에 저장되었습니다.")
    
    # 전체 결과를 통합하여 저장
    all_results_file = os.path.join(output_path, "all_failure_predictions.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 모든 예측 결과가 {all_results_file}에 저장되었습니다.")
    
    return results

# 실행 경로 설정
disk_info_path = "C:/Users/pythonfile/SSD_remain/input/disk_info_*.txt"  # 입력 파일 경로
output_dir = "C:/Users/pythonfile/SSD_remain/output"  # 출력 파일 저장 경로

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 예측 실행 및 결과 저장
all_results = run_predictions(disk_info_path, output_dir)

# 최종 결과 반환
print("\n📊 최종 예측 결과:")
print(json.dumps(all_results, indent=2, ensure_ascii=False))
