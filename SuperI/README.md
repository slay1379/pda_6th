# 이상 거래 탐지 테스트(Anomalous Transaction Detection)

## 1. 주제

이 프로젝트는 금융 데이터에서 비정상적인 거래를 탐지하는 시스템을 구현하고, 다양한 인스턴스 유형 및 아키텍처에서의 성능을 비교하는 데 목적이 있습니다.

---

## 2. 목적

- 이상 거래를 조기에 탐지하여 보안 강화 및 피해 최소화
- 다양한 AWS 인스턴스 유형 및 아키텍처에서 실행 성능 비교
- 비용 대비 효율적인 인프라 선택을 위한 기준 마련

---

## 3. 인스턴스 유형 list 및 아키텍처 소개

### 3.1 c: 컴퓨팅 최적화 (Compute-Optimized)
- `c6i.large`, `c6i.xlarge` (x86 기반)
- `c7g.large`, `c7g.xlarge` (Arm 기반 Graviton)

### 3.2 m: 범용 (General Purpose)
- `m6i.large`, `m6i.xlarge` (x86 기반)
- `m7g.large`, `m7g.xlarge` (Arm 기반 Graviton)

---

## 4. 테스트 코드 소개
test.csv
```js
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 설정값
n_total = 500_000
anomaly_ratio = 0.01  # 1%
n_anomaly = int(n_total * anomaly_ratio)  # 1,000
n_normal = n_total - n_anomaly           # 99,000

# 사용자 및 IP 초기 세팅
user_ids = [random.randint(1000, 9999) for _ in range(300)]
ip_pool = [f"192.168.0.{i}" for i in range(1, 51)]
user_ip_map = {uid: random.choice(ip_pool) for uid in user_ids}
start_time = datetime(2024, 1, 1)
data = []

# ✅ 정상 거래 생성
for i in range(n_normal):
    uid = random.choice(user_ids)
    amount = round(np.random.exponential(scale=30000), 2)
    t_type = random.choice(['deposit', 'withdrawal'])
    timestamp = start_time + timedelta(seconds=i*2)
    ip = user_ip_map[uid]
    data.append([i, uid, amount, t_type, timestamp, ip, 0])  # 정상 거래 (label=0)

# ✅ 이상 거래 생성 (고르게 분산해서 생성)
anomaly_id = n_normal
per_type = n_anomaly // 5

# ① 고액 + 새벽
for i in range(per_type):
    uid = random.choice(user_ids)
    amount = random.uniform(5_000_000, 10_000_000)
    timestamp = datetime(2024, 1, 2, random.randint(1, 3), random.randint(0, 59), 0)
    ip = user_ip_map[uid]
    data.append([anomaly_id, uid, amount, 'withdrawal', timestamp, ip, 1])
    anomaly_id += 1

# ② 반복 거래
repeat_user = random.choice(user_ids)
base_time = datetime(2024, 1, 3, 10, 0, 0)
for i in range(per_type):
    timestamp = base_time + timedelta(seconds=i*5)
    data.append([anomaly_id, repeat_user, 10000, 'withdrawal', timestamp, user_ip_map[repeat_user], 1])
    anomaly_id += 1

# ③ 동일 IP 다중 사용자
shared_ip = "10.0.0.99"
for i in range(per_type):
    uid = random.randint(9000, 9999)
    timestamp = datetime(2024, 1, 4, 11, i % 60, 0)
    data.append([anomaly_id, uid, 20000, 'deposit', timestamp, shared_ip, 1])
    anomaly_id += 1

# ④ 위치 불일치 (예: 평소 서울, 갑자기 부산)
for i in range(per_type):
    uid = random.choice(user_ids)
    timestamp = datetime(2024, 1, 5, 9, i % 60, 0)
    ip = user_ip_map[uid] + "_부산"  # 비정상 지역 표시
    data.append([anomaly_id, uid, 15000, 'withdrawal', timestamp, ip, 1])
    anomaly_id += 1

# ⑤ 기타 랜덤 이상 거래
for i in range(n_anomaly - per_type * 4):
    uid = random.choice(user_ids)
    amount = random.uniform(1_000_000, 5_000_000)
    timestamp = datetime(2024, 1, 6, random.randint(0, 23), random.randint(0, 59), 0)
    ip = user_ip_map[uid]
    data.append([anomaly_id, uid, amount, 'withdrawal', timestamp, ip, 1])
    anomaly_id += 1

# ✅ DataFrame 저장
df = pd.DataFrame(data, columns=['transaction_id', 'user_id', 'amount', 'type', 'timestamp', 'ip_address', 'label'])
df = df.sample(frac=1).reset_index(drop=True)  # 셔플
df.to_csv("transactions_with_1_percent_anomaly.csv", index=False)
print("✅: 이상 거래 1% 포함한 데이터 생성 완료!")
```

test.py
```js
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time
import platform
import os

def log_system_info():
	print(":돋보기: System Info")
	print(f"Machine: {platform.machine()}")
	print(f"Processor: {platform.processor()}")
	print(f"Platform: {platform.platform()}")
	print(f"CPU cores: {os.cpu_count()}")
	print("-" * 40)

def main():
	log_system_info()

start = time.time()
df = pd.read_csv("test.csv")

# 특성 선택 및 전처리
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["is_withdrawal"] = (df["type"] == "withdrawal").astype(int)
df["ip_encoded"] = df["ip_address"].astype("category").cat.codes
features = ["amount", "hour", "is_withdrawal", "ip_encoded"]
X = df[features]
y = df["label"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 모델 학습
clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
clf.fit(X_scaled)

# 예측 및 평가
y_pred = clf.predict(X_scaled)
y_pred = np.where(y_pred == -1, 1, 0)  # -1은 이상치
print(classification_report(y, y_pred, digits=4))
end = time.time()
print(f": Total Time: {end - start:.2f} seconds")

```

## 5. 결과 분석

측정 지표:  
- **CPU 처리량 (CPU utilization %)**
- **처리 시간 (Execution time in seconds)**
- **네트워크 입출력 (Network I/O, 입력/출력 MB)**


### 5.1 인스턴스 크기 비교 (`large` vs `xlarge`)

| 인스턴스     | 크기     | CPU 처리량 (%) | 처리 시간 (s) |
|--------------|----------|----------------|----------------|
| c6i.large     | large   | 1.48           | 6.47           |
| c6i.xlarge    | xlarge  | 0.80           | 6.72           |
| m6i.large     | large   | 7.28           | 6.83           |
| m6i.xlarge    | xlarge  | 3.89           | 6.70           |
| m7g.large     | large   | 1.34           | 6.19           |
| m7g.xlarge    | xlarge  | 0.76           | 6.14           |
| c7g.large     | large   | 2.63           | 6.15           |
| c7g.xlarge    | xlarge  | 1.13           | 6.12           |

<img width="668" alt="스크린샷 2025-05-07 오후 12 51 34" src="https://github.com/user-attachments/assets/1e4fc774-0ab9-4534-bafd-25917f69bec7" />

> ✅ xlarge 인스턴스들이 대체로 더 낮은 CPU 사용률을 보였으나, 처리 시간은 큰 차이가 없었습니다.

---

### 5.2 아키텍처 비교 (`x86` vs `Arm`)

| 인스턴스  | 아키텍처 | CPU 처리량 (%) | 처리 시간 (s) |
|-----------|----------|----------------|----------------|
| c6i.xlarge | x86      | 0.80           | 6.72           |
| c7g.xlarge | Arm      | 1.13           | 6.12           |
| m6i.xlarge | x86      | 3.89           | 6.70           |
| m7g.xlarge | Arm      | 0.76           | 6.14           |

<img width="668" alt="스크린샷 2025-05-07 오후 12 51 44" src="https://github.com/user-attachments/assets/90f73bfa-6077-4759-9413-d375f6d9cfee" />


> ✅ Arm 계열 인스턴스(m7g, c7g)가 x86 계열보다 전반적으로 더 빠른 처리 시간과 효율적인 CPU 사용량을 보였습니다.

---

### 5.3 인스턴스 패밀리 비교

| 패밀리 | 인스턴스       | CPU 처리량 (%) | 처리 시간 (s) |
|--------|----------------|----------------|----------------|
| c6i    | c6i.large      | 1.48           | 6.47           | 
| c6i    | c6i.xlarge     | 0.80           | 6.72           |
| c7g    | c7g.large      | 2.63           | 6.15           | 
| c7g    | c7g.xlarge     | 1.13           | 6.12           | 
| m6i    | m6i.large      | 7.28           | 6.83           | 
| m6i    | m6i.xlarge     | 3.89           | 6.70           | 
| m7g    | m7g.large      | 1.34           | 6.19           | 
| m7g    | m7g.xlarge     | 0.76           | 6.14           | 

<img width="668" alt="스크린샷 2025-05-07 오후 12 51 50" src="https://github.com/user-attachments/assets/e1ea8c56-6ac5-4bdc-8b46-c71aad03dcf0" />


> ✅ `m6i.large`는 가장 높은 CPU 사용률(7.28%)을 보이며 비효율적인 성능을 보였고, `m7g.xlarge`는 가장 낮은 CPU 사용률(0.76%)로 가장 효율적이었습니다.

---

### 5.4 실행 비용 비교

| 인스턴스     | 시간당 요금 (USD) | 처리 시간 (초) | 1회 실행 비용 (USD) |
|--------------|------------------|----------------|---------------------|
| c6i.large    | $0.0850          | 6.47           | $0.000152           |
| c6i.xlarge   | $0.1700          | 6.72           | $0.000317           |
| m6i.large    | $0.0960          | 6.83           | $0.000182           |
| m6i.xlarge   | $0.1920          | 6.70           | $0.000357           |
| m7g.large    | $0.0840          | 6.19           | $0.000144           |
| m7g.xlarge   | $0.1632          | 6.14           | $0.000278           |
| c7g.large    | $0.0725          | 6.15           | $0.000124           |
| c7g.xlarge   | $0.1450          | 6.12           | $0.000247           |

> 💡 `c7g.large` 인스턴스는 가장 낮은 실행 비용과 우수한 성능을 제공하여 이상 거래 탐지 워크로드에 적합한 선택입니다.

---

## 6. 결론

- **Arm 기반 인스턴스(c7g, m7g)**는 **x86 기반(c6i, m6i)**보다 전반적으로 처리 시간이 빠르고 CPU 사용량이 적었습니다.
- **Compute Optimized (c 계열)**은 **범용 (m 계열)**보다 네트워크 입출력 처리량이 높은 경향을 보였습니다.
- 가격 대비 성능을 고려했을 때, **c7g.large** 또는 **m7g.xlarge**가 추천됩니다.
- 실제 사용 시에는 워크로드의 특성과 네트워크 요구사항을 함께 고려해야 최적의 선택이 가능합니다.

