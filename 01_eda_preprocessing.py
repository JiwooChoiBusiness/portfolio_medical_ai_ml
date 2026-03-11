"""
HealthAI — 데이터 전처리 + EDA
================================
사용 데이터셋:
  1. BRFSS 2015       → 만성질환 (당뇨/고혈압/비만)
  2. Mental Health    → 정신건강 (번아웃/스트레스)
  3. Sleep Health     → 수면장애
  4. Heart Disease    → 심장질환

실행 순서:
  python 01_eda_preprocessing.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 (OS 자동 감지) ──
import platform
_os = platform.system()
if _os == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif _os == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False

# ── 출력 디렉토리 ──
os.makedirs("outputs/eda", exist_ok=True)

# ════════════════════════════════════════════════
# 0. 공통 유틸리티
# ════════════════════════════════════════════════

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def save_fig(name: str):
    path = f"outputs/eda/{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  → 저장: {path}")


# ════════════════════════════════════════════════
# 1. BRFSS 2015 — 만성질환 (당뇨 / 고혈압 / 비만)
# ════════════════════════════════════════════════
section("1. BRFSS 2015 로드 및 전처리")

# ── 1-1. 데이터 로드 ──
# Kaggle에서 다운로드:
# https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
# 파일명: diabetes_binary_5050split_health_indicators_BRFSS2015.csv

BRFSS_PATH = "data/raw/brfss2015.csv"

if os.path.exists(BRFSS_PATH):
    df_brfss = pd.read_csv(BRFSS_PATH)
    print(f"  로드 완료: {df_brfss.shape[0]:,}행 × {df_brfss.shape[1]}열")
else:
    # ── 샘플 데이터 생성 (파일 없을 때 테스트용) ──
    print("  [!] 파일 없음 → 샘플 데이터 생성 (테스트용)")
    np.random.seed(42)
    n = 70_000
    df_brfss = pd.DataFrame({
        "Diabetes_binary":    np.random.randint(0, 2, n),
        "HighBP":             np.random.randint(0, 2, n),
        "HighChol":           np.random.randint(0, 2, n),
        "CholCheck":          np.random.randint(0, 2, n),
        "BMI":                np.random.normal(28, 6, n).clip(15, 60).astype(int),
        "Smoker":             np.random.randint(0, 2, n),
        "Stroke":             np.random.binomial(1, 0.04, n),
        "HeartDiseaseorAttack": np.random.binomial(1, 0.09, n),
        "PhysActivity":       np.random.randint(0, 2, n),
        "Fruits":             np.random.randint(0, 2, n),
        "Veggies":            np.random.randint(0, 2, n),
        "HvyAlcoholConsump":  np.random.binomial(1, 0.06, n),
        "AnyHealthcare":      np.random.binomial(1, 0.93, n),
        "NoDocbcCost":        np.random.binomial(1, 0.10, n),
        "GenHlth":            np.random.randint(1, 6, n),
        "MentHlth":           np.random.randint(0, 31, n),
        "PhysHlth":           np.random.randint(0, 31, n),
        "DiffWalk":           np.random.binomial(1, 0.17, n),
        "Sex":                np.random.randint(0, 2, n),
        "Age":                np.random.randint(1, 14, n),
        "Education":          np.random.randint(1, 7, n),
        "Income":             np.random.randint(1, 9, n),
    })

# ── 1-2. 기본 정보 ──
print("\n[BRFSS 기본 정보]")
print(f"  Shape        : {df_brfss.shape}")
print(f"  결측치 합계  : {df_brfss.isnull().sum().sum()}")
print(f"  Target 분포  :\n{df_brfss['Diabetes_binary'].value_counts()}")

# ── 1-3. 피처 그룹 정의 ──
FEATURES_BODY     = ["BMI", "HighBP", "HighChol", "CholCheck"]
FEATURES_LIFESTYLE= ["Smoker", "PhysActivity", "HvyAlcoholConsump", "Fruits", "Veggies"]
FEATURES_HEALTH   = ["HeartDiseaseorAttack", "Stroke", "DiffWalk",
                     "GenHlth", "MentHlth", "PhysHlth"]
FEATURES_DEMO     = ["Age", "Sex", "Education", "Income"]
FEATURES_MEDICAL  = ["AnyHealthcare", "NoDocbcCost"]
ALL_FEATURES      = FEATURES_BODY + FEATURES_LIFESTYLE + FEATURES_HEALTH \
                  + FEATURES_DEMO + FEATURES_MEDICAL
TARGET_BRFSS      = "Diabetes_binary"

# ── 1-4. Feature Engineering ──
print("\n[Feature Engineering]")

df_brfss["BMI_HighBP"]          = df_brfss["BMI"] * df_brfss["HighBP"]
df_brfss["Age_PhysActivity"]    = df_brfss["Age"] * (1 - df_brfss["PhysActivity"])
df_brfss["GenHlth_MentHlth"]    = df_brfss["GenHlth"] * df_brfss["MentHlth"]
df_brfss["Lifestyle_Risk"]      = (df_brfss["Smoker"] + df_brfss["HvyAlcoholConsump"]
                                   + (1 - df_brfss["PhysActivity"])
                                   + (1 - df_brfss["Fruits"])
                                   + (1 - df_brfss["Veggies"]))
df_brfss["Comorbidity_Score"]   = (df_brfss["HighBP"] + df_brfss["HighChol"]
                                   + df_brfss["HeartDiseaseorAttack"]
                                   + df_brfss["Stroke"])
df_brfss["BMI_Category"]        = pd.cut(df_brfss["BMI"],
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=[0, 1, 2, 3]).astype(int)

ENG_FEATURES = ["BMI_HighBP", "Age_PhysActivity", "GenHlth_MentHlth",
                "Lifestyle_Risk", "Comorbidity_Score", "BMI_Category"]

print(f"  파생 피처 {len(ENG_FEATURES)}개 추가 완료")
print(f"  최종 피처 수: {len(ALL_FEATURES) + len(ENG_FEATURES)}개")

# ── 1-5. 이상치 처리 (IQR) ──
def remove_outliers_iqr(df, cols, factor=3.0):
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask &= (df[col] >= Q1 - factor * IQR) & (df[col] <= Q3 + factor * IQR)
    return df[mask]

df_brfss_clean = remove_outliers_iqr(df_brfss, ["BMI", "MentHlth", "PhysHlth"])
print(f"\n  이상치 제거 후: {df_brfss_clean.shape[0]:,}행")

# ════════════════════════════════════════════════
# 2. EDA 시각화 — BRFSS
# ════════════════════════════════════════════════
section("2. EDA 시각화 — BRFSS")

# ── 2-1. Target 분포 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("BRFSS 2015 — Target & Key Feature Distribution", fontsize=14, fontweight="bold")

counts = df_brfss[TARGET_BRFSS].value_counts()
axes[0].bar(["Normal (0)", "Diabetes/Pre (1)"], counts.values,
            color=["#3b8eff", "#ef4444"], alpha=0.85, edgecolor="white", linewidth=1.5)
axes[0].set_title("Target Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 200, f"{v:,}\n({v/len(df_brfss)*100:.1f}%)",
                 ha="center", fontsize=10, fontweight="bold")

# BMI 분포
axes[1].hist(df_brfss[df_brfss[TARGET_BRFSS]==0]["BMI"], bins=40,
             alpha=0.6, color="#3b8eff", label="Normal", density=True)
axes[1].hist(df_brfss[df_brfss[TARGET_BRFSS]==1]["BMI"], bins=40,
             alpha=0.6, color="#ef4444", label="Diabetes", density=True)
axes[1].set_title("BMI Distribution by Target")
axes[1].set_xlabel("BMI")
axes[1].legend()

# GenHlth 분포
axes[2].hist(df_brfss[df_brfss[TARGET_BRFSS]==0]["GenHlth"], bins=5,
             alpha=0.6, color="#3b8eff", label="Normal", density=True)
axes[2].hist(df_brfss[df_brfss[TARGET_BRFSS]==1]["GenHlth"], bins=5,
             alpha=0.6, color="#ef4444", label="Diabetes", density=True)
axes[2].set_title("General Health by Target")
axes[2].set_xlabel("GenHlth (1=Excellent ~ 5=Poor)")
axes[2].legend()

plt.tight_layout()
save_fig("01_brfss_target_distribution")

# ── 2-2. 상관관계 히트맵 ──
fig, ax = plt.subplots(figsize=(14, 10))
key_cols = ALL_FEATURES[:15] + [TARGET_BRFSS]
corr = df_brfss[key_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 8}, linewidths=0.5)
ax.set_title("BRFSS Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
save_fig("02_brfss_correlation_heatmap")

# ── 2-3. 위험 요인별 당뇨 발생률 ──
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Risk Factor vs Diabetes Rate", fontsize=14, fontweight="bold")
axes = axes.flatten()

risk_factors = ["HighBP", "HighChol", "Smoker", "PhysActivity",
                "HeartDiseaseorAttack", "Stroke", "DiffWalk", "HvyAlcoholConsump"]
colors_pos = "#ef4444"
colors_neg = "#3b8eff"

for i, feat in enumerate(risk_factors):
    rates = df_brfss.groupby(feat)[TARGET_BRFSS].mean() * 100
    bars = axes[i].bar(rates.index.astype(str), rates.values,
                       color=[colors_neg, colors_pos], alpha=0.85, edgecolor="white")
    axes[i].set_title(feat, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Diabetes Rate (%)")
    axes[i].set_ylim(0, 80)
    for bar, val in zip(bars, rates.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
save_fig("03_brfss_risk_factors")

# ── 2-4. Feature Engineering 피처 효과 확인 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Engineered Features — Distribution by Target", fontsize=13, fontweight="bold")

eng_viz = ["Lifestyle_Risk", "Comorbidity_Score", "BMI_HighBP"]
for i, feat in enumerate(eng_viz):
    df_brfss.boxplot(column=feat, by=TARGET_BRFSS, ax=axes[i],
                     patch_artist=True,
                     boxprops=dict(facecolor="#dceeff"),
                     medianprops=dict(color="#ef4444", linewidth=2))
    axes[i].set_title(feat)
    axes[i].set_xlabel("Diabetes (0=No, 1=Yes)")

plt.suptitle("")
plt.tight_layout()
save_fig("04_brfss_engineered_features")

print("  BRFSS EDA 시각화 완료 (4개 차트)")


# ════════════════════════════════════════════════
# 3. NHANES 수면 데이터 — 수면장애
# ════════════════════════════════════════════════
section("3. NHANES 수면 데이터 전처리")

SLEEP_PATH = "data/raw/sleep_nhanes.csv"

if os.path.exists(SLEEP_PATH):
    df_sleep = pd.read_csv(SLEEP_PATH)
    print(f"  로드 완료: {df_sleep.shape[0]:,}행 × {df_sleep.shape[1]}열")
else:
    print("  [!] sleep_nhanes.csv 없음 → 00_download_datasets.py 먼저 실행하세요")
    df_sleep = pd.DataFrame()

if not df_sleep.empty:
    print(f"  수면장애 비율: {df_sleep['Sleep_Disorder'].mean()*100:.1f}%")

    # ── EDA 차트 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("NHANES Sleep Dataset EDA", fontsize=13, fontweight="bold")

    # 수면 시간 분포
    axes[0].hist(df_sleep[df_sleep["Sleep_Disorder"]==0]["Sleep_Hours"],
                 alpha=0.6, color="#3b8eff", label="Normal", bins=15, density=True)
    axes[0].hist(df_sleep[df_sleep["Sleep_Disorder"]==1]["Sleep_Hours"],
                 alpha=0.6, color="#ef4444", label="Disorder", bins=15, density=True)
    axes[0].set_title("Sleep Hours by Disorder")
    axes[0].set_xlabel("Hours")
    axes[0].legend()

    # 나이 vs 수면 시간
    sc_colors = ["#ef4444" if d else "#3b8eff" for d in df_sleep["Sleep_Disorder"]]
    axes[1].scatter(df_sleep["Age"], df_sleep["Sleep_Hours"],
                    c=sc_colors, alpha=0.3, s=10)
    axes[1].set_title("Age vs Sleep Hours")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Sleep Hours")

    # BMI 분포
    bmi_0 = df_sleep[df_sleep["Sleep_Disorder"]==0]["BMI"].dropna()
    bmi_1 = df_sleep[df_sleep["Sleep_Disorder"]==1]["BMI"].dropna()
    axes[2].boxplot([bmi_0, bmi_1], labels=["Normal", "Disorder"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#dceeff"),
                    medianprops=dict(color="#ef4444", linewidth=2))
    axes[2].set_title("BMI by Sleep Disorder")
    axes[2].set_ylabel("BMI")

    plt.tight_layout()
    save_fig("05_sleep_eda")
    print("  Sleep EDA 완료")


# ════════════════════════════════════════════════
# 4. Cardiovascular Disease Dataset — 심장질환
# ════════════════════════════════════════════════
section("4. Cardiovascular Disease 전처리")

HEART_PATH = "data/processed/cardio_processed.csv"

if os.path.exists(HEART_PATH):
    df_heart = pd.read_csv(HEART_PATH)
    print(f"  로드 완료: {df_heart.shape[0]:,}행 × {df_heart.shape[1]}열")
else:
    print("  [!] cardio_processed.csv 없음 → 00_download_datasets.py 먼저 실행하세요")
    df_heart = pd.DataFrame()

if not df_heart.empty:
    print(f"  심장질환 비율: {df_heart['HeartDisease'].mean()*100:.1f}%")

    # ── EDA 차트 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Cardiovascular Disease Dataset EDA", fontsize=13, fontweight="bold")

    axes[0].hist(df_heart[df_heart["HeartDisease"]==0]["Age"],
                 alpha=0.6, color="#3b8eff", label="No Disease", bins=20, density=True)
    axes[0].hist(df_heart[df_heart["HeartDisease"]==1]["Age"],
                 alpha=0.6, color="#ef4444", label="Disease", bins=20, density=True)
    axes[0].set_title("Age Distribution by Heart Disease")
    axes[0].set_xlabel("Age")
    axes[0].legend()

    axes[1].scatter(df_heart["BP_Systolic"], df_heart["BMI"],
                    c=["#ef4444" if x else "#3b8eff" for x in df_heart["HeartDisease"]],
                    alpha=0.2, s=8)
    axes[1].set_title("BP Systolic vs BMI")
    axes[1].set_xlabel("Systolic BP")
    axes[1].set_ylabel("BMI")

    chol_0 = df_heart[df_heart["HeartDisease"]==0]["Cholesterol"].dropna()
    chol_1 = df_heart[df_heart["HeartDisease"]==1]["Cholesterol"].dropna()
    axes[2].boxplot([chol_0, chol_1], labels=["No Disease","Disease"],
                    patch_artist=True,
                    boxprops=dict(facecolor="#dceeff"),
                    medianprops=dict(color="#ef4444", linewidth=2))
    axes[2].set_title("Cholesterol by Heart Disease")
    axes[2].set_ylabel("Cholesterol Level")

    plt.tight_layout()
    save_fig("06_heart_eda")
    print("  Heart Disease EDA 완료")


# ════════════════════════════════════════════════
# 5. NHANES PHQ-9 — 우울/번아웃
# ════════════════════════════════════════════════
section("5. NHANES PHQ-9 정신건강 전처리")

MENTAL_PATH = "data/raw/mental_nhanes.csv"

if os.path.exists(MENTAL_PATH):
    df_mental = pd.read_csv(MENTAL_PATH)
    print(f"  로드 완료: {df_mental.shape[0]:,}행 × {df_mental.shape[1]}열")
else:
    print("  [!] mental_nhanes.csv 없음 → 00_download_datasets.py 먼저 실행하세요")
    df_mental = pd.DataFrame()

if not df_mental.empty:
    print(f"  우울 위험 비율(PHQ≥10): {df_mental['Depression_Risk'].mean()*100:.1f}%")
    print(f"  PHQ-9 평균 점수: {df_mental['PHQ9_Score'].mean():.2f}")

    # ── EDA 차트 ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("NHANES PHQ-9 Mental Health EDA", fontsize=13, fontweight="bold")

    # PHQ-9 점수 분포
    axes[0].hist(df_mental[df_mental["Depression_Risk"]==0]["PHQ9_Score"],
                 alpha=0.6, bins=20, color="#3b8eff", label="Low Risk", density=True)
    axes[0].hist(df_mental[df_mental["Depression_Risk"]==1]["PHQ9_Score"],
                 alpha=0.6, bins=20, color="#ef4444", label="High Risk", density=True)
    axes[0].set_title("PHQ-9 Score Distribution")
    axes[0].set_xlabel("PHQ-9 Score")
    axes[0].legend()

    # 나이 vs PHQ-9
    sc_c = ["#ef4444" if d else "#3b8eff" for d in df_mental["Depression_Risk"]]
    axes[1].scatter(df_mental["Age"], df_mental["PHQ9_Score"],
                    c=sc_c, alpha=0.2, s=8)
    axes[1].axhline(y=10, color="#f59e0b", linestyle="--", linewidth=1.5, label="PHQ≥10 threshold")
    axes[1].set_title("Age vs PHQ-9 Score")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("PHQ-9 Score")
    axes[1].legend()

    # 수면 시간 vs 우울 위험
    if "Sleep_Hours" in df_mental.columns:
        slp_0 = df_mental[df_mental["Depression_Risk"]==0]["Sleep_Hours"].dropna()
        slp_1 = df_mental[df_mental["Depression_Risk"]==1]["Sleep_Hours"].dropna()
        axes[2].boxplot([slp_0, slp_1], labels=["Low Risk","High Risk"],
                        patch_artist=True,
                        boxprops=dict(facecolor="#dceeff"),
                        medianprops=dict(color="#ef4444", linewidth=2))
        axes[2].set_title("Sleep Hours by Depression Risk")
        axes[2].set_ylabel("Sleep Hours")
    else:
        axes[2].text(0.5, 0.5, "Sleep data\nnot available",
                     ha="center", va="center", fontsize=11)
        axes[2].set_title("Sleep Hours by Depression Risk")

    plt.tight_layout()
    save_fig("07_mental_eda")
    print("  Mental Health EDA 완료")


# ════════════════════════════════════════════════
# 6. 공통 피처 매핑 테이블 출력
# ════════════════════════════════════════════════
section("6. 모델별 공통 피처 매핑 요약")

summary = {
    "공통 피처": ["Age", "BMI/BMI_enc", "Sleep_Duration",
                 "Physical_Activity", "Smoker/Smoke", "Stress_Score"],
    "모델 1 (번아웃)": ["Work_Hours", "Job_Satisfaction", "Social_Support",
                       "PHQ9_Score", "GAD7_Score", "Work_Sleep_Ratio"],
    "모델 2 (고혈압)": ["HighBP", "HighChol", "HeartDiseaseorAttack",
                       "BMI_HighBP", "Comorbidity_Score"],
    "모델 3 (당뇨)":   ["HighBP", "HighChol", "GenHlth",
                       "Lifestyle_Risk", "BMI_Category"],
    "모델 4 (수면장애)": ["Quality_of_Sleep", "BP_Systolic",
                         "Sleep_Stress_Ratio", "Active_Sleep_Score"],
    "모델 5 (심장질환)": ["ChestPain_enc", "MaxHR", "Oldpeak",
                          "HR_Reserve", "BP_Chol_Risk"],
}

print(f"\n{'모델':<20} {'피처'}")
print("-" * 70)
for model, feats in summary.items():
    print(f"  {model:<18} {', '.join(feats[:4])}{'...' if len(feats)>4 else ''}")


# ════════════════════════════════════════════════
# 7. 전처리된 데이터 저장
# ════════════════════════════════════════════════
section("7. 전처리 데이터 저장")

os.makedirs("data/processed", exist_ok=True)

df_brfss_clean.to_csv("data/processed/brfss_processed.csv", index=False)
if not df_sleep.empty:
    df_sleep.to_csv("data/processed/sleep_processed.csv", index=False)

print("  저장 완료:")
print("    data/processed/brfss_processed.csv")
if not df_sleep.empty:
    print("    data/processed/sleep_processed.csv")
print("    (cardio_processed.csv, mental_nhanes.csv → 00_download_datasets.py 생성)")

print("\n" + "="*60)
print("  ✅ 01_eda_preprocessing.py 완료")
print("  다음 단계: python 02_modeling.py")
print("="*60)
