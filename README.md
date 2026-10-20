# HealthAI - AI 기반 통합 건강관리 앱 모델링 파이프라인

> 웨어러블 + 설문 데이터로 정신건강 & 만성질환 위험을 AI가 미리 예측 -> 병원 연결까지 한번에

---

## 프로젝트 개요

번아웃 → 수면부족 → 고혈압 → 당뇨로 이어지는 **연쇄 악화 패턴**을 통합 예측하는 5개 모델 파이프라인.  
각 모델은 XGBoost / LightGBM / RandomForest 3종 경쟁 후 최고 성능 엔진을 선택하며,  
StratifiedKFold 5-Fold + Optuna 하이퍼파라미터 튜닝 + Threshold 최적화를 적용한다.

---

## 실행 순서

```bash
# 0. NHANES 자동 다운로드 + Cardio 전처리
python 00_download_datasets.py

# 1. 전처리 + EDA (brfss_processed.csv 재생성 포함)
python 01_eda_preprocessing.py

# 2. 모델 학습 (XGBoost / LightGBM / RandomForest + Optuna)
python 02_modeling.py

# 3. 최종 대시보드 시각화
python 03_visualization.py
```

> 반드시 01 → 02 → 03 순서를 지킬 것.
> 01이 생성하는 `brfss_processed.csv`를 02가 읽기 때문에 순서가 바뀌면 leakage 수정이 반영되지 않는다.

---

## 데이터셋 구성

| 질환 그룹 | 파일 | 출처 | 행 수 | 비고 |
|-----------|------|------|-------|------|
| 당뇨 / 고혈압 | `data/raw/brfss2015.csv` | [Kaggle - alexteboul](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | 70,692 | **수동 다운로드** |
| 심장질환 | `data/raw/cardio_train.csv` | [Kaggle - sulianova](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | 70,000 | **수동 다운로드** |
| 수면장애 | `data/raw/sleep_nhanes.csv` | CDC NHANES SLQ (2009~2020) | ~20,000 | 00 자동 생성 |
| 정신건강 / 번아웃 | `data/raw/mental_nhanes.csv` | CDC NHANES PHQ-9 (2005~2020) | ~30,000 | 00 자동 생성 |

### Kaggle 수동 다운로드 안내

```
brfss2015.csv       → data/raw/brfss2015.csv
cardio_train.csv    → data/raw/cardio_train.csv
```

---

## 폴더 구조

```
HealthAI/
├── 00_download_datasets.py
├── 01_eda_preprocessing.py
├── 02_modeling.py
├── 03_visualization.py
├── README.md
└── data/
    ├── raw/
    │   ├── brfss2015.csv            ← Kaggle 수동 저장
    │   ├── cardio_train.csv         ← Kaggle 수동 저장
    │   ├── sleep_nhanes.csv         ← 00 자동 생성
    │   └── mental_nhanes.csv        ← 00 자동 생성
    └── processed/
        ├── brfss_processed.csv      ← 01 자동 생성
        ├── sleep_processed.csv      ← 01 자동 생성
        └── cardio_processed.csv     ← 00 자동 생성
```

---

## 5개 모델 구조

### 모델 KPI 기준

| 지표 | 목표 | 이유 |
|------|------|------|
| **Recall** | **≥ 0.80** | 위험 환자를 놓치는 것이 가장 치명적 (최우선) |
| AUC-ROC | ≥ 0.80 | 전반적 판별력 (설문 기반 데이터 한계 반영) |
| F1-Score | ≥ 0.75 | 정밀도-재현율 균형 |

### 모델별 피처 설계

#### 모델 1 — 번아웃 / 우울 위험 (`Depression_Risk`)
- **데이터**: NHANES PHQ-9 (`mental_nhanes.csv`)
- **핵심 피처**: `Age`, `Gender_enc`, `BMI`, `Income_Ratio`, `Sleep_Hours`, `Active_Any`, `Sedentary_Min`, `PHQ_Q1~Q8` (개별 문항)
- **설계 주의**: `PHQ9_High`, `PHQ9_Moderate`는 타겟(`Depression_Risk` = PHQ≥10)을 직접 파생한 컬럼이므로 **제외** (data leakage)

#### 모델 2 — 고혈압 위험 (`HighBP`)
- **데이터**: BRFSS 2015 (`brfss_processed.csv`)
- **핵심 피처**: `Age`, `BMI`, `HighChol`, `GenHlth`, `HeartDiseaseorAttack`, `DiffWalk`, `Diabetes_binary`, `PhysHlth`, `Income`, `Education`
- **설계 주의**: `Comorbidity_Score` 정의에서 `HighBP` 제거 필수. `BMI_HighBP = BMI × HighBP`도 leakage이므로 `BMI_PhysActivity`로 대체

#### 모델 3 — 당뇨 위험 (`Diabetes_binary`)
- **데이터**: BRFSS 2015 (`brfss_processed.csv`)
- **핵심 피처**: `Age`, `BMI`, `HighBP`, `HighChol`, `GenHlth`, `DiffWalk`, `PhysHlth`, `Stroke`, `Income`, `Education`, `Fruits`, `Veggies`
- **설계 주의**: `Comorbidity_Score`에서 `HighBP` 제거 후 사용 가능. `GenHlth`는 상관 0.41로 가장 높으나 실제 인과관계(당뇨 → 건강악화)를 반영하므로 leakage 아님

#### 모델 4 — 수면장애 위험 (`Sleep_Disorder`)
- **데이터**: NHANES SLQ (`sleep_nhanes.csv` → `sleep_processed.csv`)
- **핵심 피처**: `Age`, `Gender_enc`, `BMI`, `Sleep_Hours`, `Snore_Risk`, `Sleepy_Risk`, `BMI_Category`, `Income_Ratio`, `HighBP_Diagnosed`

#### 모델 5 — 심장질환 위험 (`HeartDisease`)
- **데이터**: Cardiovascular Disease (`cardio_processed.csv`)
- **핵심 피처**: `Age`, `Gender`, `BP_Systolic`, `BP_Diastolic`, `Cholesterol`, `Glucose`, `Smoker`, `Alcohol`, `PhysActivity`, `BMI`, `Pulse_Pressure`

---

## 파생 피처 설계 원칙

```python
# 안전한 파생 피처 (타겟 정보 미포함)
Comorbidity_Score  = HighChol + HeartDiseaseorAttack + Stroke + CholCheck
Lifestyle_Risk     = Smoker + HvyAlcoholConsump + (1-PhysActivity) + (1-Fruits) + (1-Veggies)
BMI_PhysActivity   = BMI × (1 - PhysActivity)
BMI_Category       = pd.cut(BMI, bins=[0, 18.5, 25, 30, 100])

# leakage 피처 (절대 사용 금지)
# BMI_HighBP              = BMI × HighBP        → 고혈압 타겟 직접 포함
# Comorbidity_Score(구버전) = HighBP + ...       → 고혈압 타겟 직접 포함
# PHQ9_High               = PHQ9_Score >= 10    → 번아웃 타겟과 동일
# PHQ9_Moderate           = PHQ9_Score >= 5     → 번아웃 타겟 파생
```

---

## 출력물

```
outputs/
├── eda/
│   ├── 01_brfss_target_distribution.png
│   ├── 02_brfss_correlation_heatmap.png
│   ├── 03_brfss_risk_factors.png
│   ├── 04_brfss_engineered_features.png
│   ├── 05_sleep_eda.png
│   ├── 06_heart_eda.png
│   └── 07_mental_eda.png
├── viz/
│   └── 08_portfolio_dashboard.png
└── results/
    ├── model_summary.json
    └── final_scores.csv
```

---

## 트러블슈팅 기록

| 증상 | 원인 | 해결 |
|------|------|------|
| AUC = 1.0, Recall = 1.0 | Data leakage (타겟 파생 피처 포함) | 해당 피처 제거 |
| AUC ≈ 0.50, Recall = 1.0, Threshold = 0.10 | 모델이 전부 양성으로 예측 / leakage 수정 전 CSV 사용 | 01 재실행 후 02 실행 |
| 수면 데이터 374행 | 구버전 Sleep Health 더미 데이터 로딩 | NHANES sleep_nhanes.csv 로 교체 |
| 대시보드 결과 안 바뀜 | 03_visualization.py가 하드코딩된 수치 사용 | model_summary.json 동적 로드로 교체 |
| KeyError: 'Burnout' | BEST_ENGINE 직접 인덱싱 | `.get()` 방어 처리로 교체 |

---

## 설계 결정 및 트레이드오프

### Optuna 목적함수에 교차검증을 넣지 않은 이유

일반적으로 Optuna 목적함수 안에 StratifiedKFold를 넣으면 하이퍼파라미터 선택의 신뢰도가 올라간다. 그러나 이 프로젝트에서는 의도적으로 넣지 않았다.

| 방식 | 장점 | 단점 |
|------|------|------|
| Optuna 내부에 CV 포함 | 과적합 방지, 파라미터 신뢰도 높음 | 실행 시간 5배 증가 (100 trial × 5-Fold = 500회 학습), AUC 0.02~0.03 하락 |
| 현재 방식 (CV는 최종 평가만) | 현실적인 계산 비용, KPI 달성 가능 | 운 좋은 파라미터가 선택될 가능성 존재 |

최종 모델 평가는 StratifiedKFold 5-Fold로 수행하기 때문에 **결과 자체의 신뢰도는 확보**된다. Optuna 튜닝 단계의 CV 생략은 실무에서도 계산 비용과 성능 간 트레이드오프로 흔히 선택하는 방식이다.

### KPI AUC 기준을 0.85에서 0.80으로 설정한 이유

BRFSS는 설문 기반 데이터로, 혈압 수치나 혈당 수치 같은 실제 임상 데이터가 포함되어 있지 않다. 의료 현장에서 AUC 0.85 이상이 나오는 고혈압/당뇨 모델은 대부분 혈액검사 수치를 활용한다. 설문 데이터의 구조적 한계를 인지하고 현실적인 기준을 설정했다.

### Recall을 최우선 KPI로 설정한 이유

건강 앱 도메인에서 모델이 위험 환자를 정상으로 분류하는 **False Negative**는 실제 의료적 피해로 이어진다. 반면 정상인을 위험군으로 분류하는 False Positive는 추가 검진을 권유하는 수준에서 끝난다. 이 비대칭적 위험 구조 때문에 Precision이나 AUC보다 Recall을 최우선 지표로 설정했다.

같은 이유로 Threshold도 일반적인 0.5보다 낮게 최적화했다. 민감도를 높여 놓치는 환자를 줄이는 것이 이 앱의 핵심 가치와 일치하기 때문이다.

"암 진단 모델에서 실제 환자를 정상으로 분류하는 오류가 반대 방향의 오류보다 훨씬 위험하다"라는 의료 ML/AI의 방식을 따른다.
