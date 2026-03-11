"""
HealthAI — 데이터셋 자동 다운로드
===================================
실행: python 00_download_datasets.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[자동 다운로드]  CDC NHANES — 로그인 불필요
  • NHANES 수면        (SLQ) 2013~2020  ~20,000행
  • NHANES PHQ-9 우울  (DPQ) 2005~2020  ~30,000행  ← Mental Health 대체
  • NHANES 인구통계    (DEMO)
  • NHANES 신체측정    (BMX)
  • NHANES 혈압        (BPQ)
  • NHANES 신체활동    (PAQ)

[수동 다운로드]  Kaggle — 아래 링크에서 직접 받기
  ① BRFSS 2015 (당뇨/고혈압)
     https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
     → File: diabetes_binary_5050split_health_indicators_BRFSS2015.csv
     → 저장: data/raw/brfss2015.csv

  ② Cardiovascular Disease 70,000행
     https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
     → File: cardio_train.csv
     → 저장: data/raw/cardio_train.csv
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, urllib.request
import pandas as pd
import numpy as np

os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

G,Y,R,B,RESET,BOLD = "\033[92m","\033[93m","\033[91m","\033[94m","\033[0m","\033[1m"
def ok(m):   print(f"  {G}✓{RESET} {m}")
def warn(m): print(f"  {Y}⚠{RESET}  {m}")
def err(m):  print(f"  {R}✗{RESET} {m}")
def info(m): print(f"  {B}→{RESET} {m}")
def sec(t):  print(f"\n{BOLD}{'─'*60}\n  {t}\n{'─'*60}{RESET}")

def download(url, dest, desc=""):
    if os.path.exists(dest):
        ok(f"이미 존재: {os.path.basename(dest)}")
        return True
    print(f"  다운로드 중: {desc}")
    try:
        def prog(n, bs, total):
            if total > 0:
                p = min(n*bs/total*100, 100)
                b = "█"*int(p/5) + "░"*(20-int(p/5))
                print(f"\r    [{b}] {p:5.1f}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=prog)
        print()
        ok(f"완료: {os.path.basename(dest)}  ({os.path.getsize(dest)//1024:,} KB)")
        return True
    except Exception as e:
        print(); err(f"실패: {e}"); return False


# ══════════════════════════════════════════════════
# NHANES 다운로드 목록 정의
# ══════════════════════════════════════════════════

# (url, dest, desc)
NHANES_FILES = {

    "수면 (SLQ)": [
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_SLQ.XPT",  "data/raw/P_SLQ.XPT",      "수면 2017-2020"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.XPT",  "data/raw/SLQ_I_2015.XPT", "수면 2015-2016"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SLQ_H.XPT",  "data/raw/SLQ_H_2013.XPT", "수면 2013-2014"),
    ],

    "우울 PHQ-9 (DPQ)": [
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DPQ.XPT",  "data/raw/P_DPQ.XPT",      "PHQ-9 2017-2020"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DPQ_I.XPT",  "data/raw/DPQ_I_2015.XPT", "PHQ-9 2015-2016"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DPQ_H.XPT",  "data/raw/DPQ_H_2013.XPT", "PHQ-9 2013-2014"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DPQ_G.XPT",  "data/raw/DPQ_G_2011.XPT", "PHQ-9 2011-2012"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DPQ_F.XPT",  "data/raw/DPQ_F_2009.XPT", "PHQ-9 2009-2010"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DPQ_E.XPT",  "data/raw/DPQ_E_2007.XPT", "PHQ-9 2007-2008"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/DPQ_D.XPT",  "data/raw/DPQ_D_2005.XPT", "PHQ-9 2005-2006"),
    ],

    "인구통계 (DEMO)": [
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_DEMO.XPT",  "data/raw/P_DEMO.XPT",      "인구통계 2017-2020"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.XPT",  "data/raw/DEMO_I_2015.XPT", "인구통계 2015-2016"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.XPT",  "data/raw/DEMO_H_2013.XPT", "인구통계 2013-2014"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.XPT",  "data/raw/DEMO_G_2011.XPT", "인구통계 2011-2012"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DEMO_F.XPT",  "data/raw/DEMO_F_2009.XPT", "인구통계 2009-2010"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.XPT",  "data/raw/DEMO_E_2007.XPT", "인구통계 2007-2008"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/DEMO_D.XPT",  "data/raw/DEMO_D_2005.XPT", "인구통계 2005-2006"),
    ],

    "신체측정 (BMX)": [
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BMX.XPT",  "data/raw/P_BMX.XPT",      "신체측정 2017-2020"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BMX_I.XPT",  "data/raw/BMX_I_2015.XPT", "신체측정 2015-2016"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.XPT",  "data/raw/BMX_H_2013.XPT", "신체측정 2013-2014"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.XPT",  "data/raw/BMX_G_2011.XPT", "신체측정 2011-2012"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BMX_F.XPT",  "data/raw/BMX_F_2009.XPT", "신체측정 2009-2010"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BMX_E.XPT",  "data/raw/BMX_E_2007.XPT", "신체측정 2007-2008"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/BMX_D.XPT",  "data/raw/BMX_D_2005.XPT", "신체측정 2005-2006"),
    ],

    "혈압 설문 (BPQ)": [
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BPQ.XPT",  "data/raw/P_BPQ.XPT",      "혈압 2017-2020"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BPQ_I.XPT",  "data/raw/BPQ_I_2015.XPT", "혈압 2015-2016"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPQ_H.XPT",  "data/raw/BPQ_H_2013.XPT", "혈압 2013-2014"),
    ],

    "신체활동 (PAQ)": [
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_PAQ.XPT",  "data/raw/P_PAQ.XPT",      "신체활동 2017-2020"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/PAQ_I.XPT",  "data/raw/PAQ_I_2015.XPT", "신체활동 2015-2016"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAQ_H.XPT",  "data/raw/PAQ_H_2013.XPT", "신체활동 2013-2014"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAQ_G.XPT",  "data/raw/PAQ_G_2011.XPT", "신체활동 2011-2012"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/PAQ_F.XPT",  "data/raw/PAQ_F_2009.XPT", "신체활동 2009-2010"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/PAQ_E.XPT",  "data/raw/PAQ_E_2007.XPT", "신체활동 2007-2008"),
        ("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles/PAQ_D.XPT",  "data/raw/PAQ_D_2005.XPT", "신체활동 2005-2006"),
    ],
}

# ── NHANES 전체 다운로드 ──
for section_name, files in NHANES_FILES.items():
    sec(f"NHANES {section_name} — 자동 다운로드")
    for url, dest, desc in files:
        download(url, dest, desc)


# ══════════════════════════════════════════════════
# Kaggle 수동 다운로드 안내
# ══════════════════════════════════════════════════
sec("Kaggle 데이터셋 — 수동 다운로드 확인")

KAGGLE_FILES = [
    {
        "name":  "BRFSS 2015 (당뇨/고혈압)",
        "url":   "https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
        "file":  "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        "dest":  "data/raw/brfss2015.csv",
        "rows":  "253,680행  |  21 피처",
        "note":  "Download → 압축 해제 → brfss2015.csv 로 저장",
    },
    {
        "name":  "Cardiovascular Disease (심장질환)",
        "url":   "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset",
        "file":  "cardio_train.csv",
        "dest":  "data/raw/cardio_train.csv",
        "rows":  "70,000행   |  11 피처  |  50:50 균형",
        "note":  "Download → 압축 해제 → cardio_train.csv 로 저장",
    },
]

for ds in KAGGLE_FILES:
    if os.path.exists(ds["dest"]):
        ok(f"{ds['name']}")
    else:
        warn(f"{ds['name']}  ← 아직 없음")
        print(f"      크기 : {ds['rows']}")
        print(f"      링크 : {ds['url']}")
        print(f"      저장 : {ds['dest']}")
        print(f"      방법 : {ds['note']}")
        print()


# ══════════════════════════════════════════════════
# NHANES 병합 함수
# ══════════════════════════════════════════════════
def load_nhanes_cycles(file_pairs):
    """여러 사이클 XPT 파일을 SEQN 기준으로 병합 후 concat"""
    frames = []
    for files in file_pairs:
        base_f = files[0]
        if not os.path.exists(base_f):
            warn(f"없음 — 건너뜀: {os.path.basename(base_f)}")
            continue
        try:
            df = pd.read_sas(base_f, format="xport", encoding="utf-8")
            for extra_f in files[1:]:
                if extra_f and os.path.exists(extra_f):
                    extra = pd.read_sas(extra_f, format="xport", encoding="utf-8")
                    # 공통 컬럼만 조인 (SEQN 기준)
                    merge_cols = ["SEQN"] + [c for c in extra.columns if c != "SEQN"]
                    df = df.merge(extra[merge_cols], on="SEQN", how="left")
            frames.append(df)
            info(f"{os.path.basename(base_f)}: {len(df):,}명")
        except Exception as e:
            err(f"로드 실패 ({base_f}): {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ══════════════════════════════════════════════════
# NHANES 수면 병합 → sleep_nhanes.csv
# ══════════════════════════════════════════════════
sec("NHANES 병합 ① → sleep_nhanes.csv")

sleep_pairs = [
    ["data/raw/P_SLQ.XPT",      "data/raw/P_DEMO.XPT",      "data/raw/P_BMX.XPT",      "data/raw/P_BPQ.XPT"],
    ["data/raw/SLQ_I_2015.XPT", "data/raw/DEMO_I_2015.XPT", "data/raw/BMX_I_2015.XPT", "data/raw/BPQ_I_2015.XPT"],
    ["data/raw/SLQ_H_2013.XPT", "data/raw/DEMO_H_2013.XPT", "data/raw/BMX_H_2013.XPT", "data/raw/BPQ_H_2013.XPT"],
]
df_sleep = load_nhanes_cycles(sleep_pairs)

if not df_sleep.empty:
    r = {"RIDAGEYR":"Age","RIAGENDR":"Gender","INDFMPIR":"Income_Ratio",
         "BMXBMI":"BMI","BPQ020":"HighBP_Diagnosed",
         "SLD012":"Sleep_Hours_Wday","SLD013":"Sleep_Hours_Wend",
         "SLD010H":"Sleep_Hours_Old",
         "SLQ050":"Trouble_Sleeping","SLQ060":"Sleep_Disorder_Dx",
         "SLQ030":"Snore_Freq","SLQ040":"Snort_Freq","SLQ120":"Daytime_Sleepy"}
    df_sleep = df_sleep.rename(columns={k:v for k,v in r.items() if k in df_sleep.columns})

    if "Sleep_Hours_Wday" in df_sleep.columns:
        df_sleep["Sleep_Hours"] = df_sleep["Sleep_Hours_Wday"]
    elif "Sleep_Hours_Old" in df_sleep.columns:
        df_sleep["Sleep_Hours"] = df_sleep["Sleep_Hours_Old"]

    if "Gender" in df_sleep.columns:
        df_sleep["Gender_enc"] = (df_sleep["Gender"] == 1).astype(float)

    # ── Sleep_Disorder 타겟 재정의 ──
    # 아래 3가지 중 2개 이상 해당 시 수면장애로 정의 (신호 강화)
    # 1) Sleep_Hours < 6  (단수면)
    # 2) Snore_Freq >= 3  (거의 매일 코골이)
    # 3) Daytime_Sleepy >= 3  (낮에 자주 졸림)
    _s1 = (df_sleep["Sleep_Hours"] < 6).astype(float) if "Sleep_Hours" in df_sleep.columns else pd.Series(0, index=df_sleep.index)
    _s2 = (df_sleep["Snore_Freq"] >= 3).astype(float) if "Snore_Freq" in df_sleep.columns else pd.Series(0, index=df_sleep.index)
    _s3 = (df_sleep["Daytime_Sleepy"] >= 3).astype(float) if "Daytime_Sleepy" in df_sleep.columns else pd.Series(0, index=df_sleep.index)
    _score = _s1 + _s2 + _s3
    df_sleep["Sleep_Disorder"] = (_score >= 2).astype(float)
    # 기존 진단 레이블도 OR로 반영
    if "Sleep_Disorder_Dx" in df_sleep.columns:
        df_sleep["Sleep_Disorder"] = np.where(
            df_sleep["Sleep_Disorder_Dx"] == 1, 1.0, df_sleep["Sleep_Disorder"])

    if "Age" in df_sleep.columns:
        df_sleep = df_sleep[df_sleep["Age"] >= 18]
    core = [c for c in ["Sleep_Hours","Age","BMI","Sleep_Disorder"] if c in df_sleep.columns]
    df_sleep = df_sleep.dropna(subset=core)
    if "Sleep_Hours" in df_sleep.columns:
        df_sleep = df_sleep[df_sleep["Sleep_Hours"].between(2,14)]
    if "BMI" in df_sleep.columns:
        df_sleep = df_sleep[df_sleep["BMI"].between(10,80)]
    if "Snore_Freq" in df_sleep.columns:
        df_sleep["Snore_Risk"] = (df_sleep["Snore_Freq"] >= 3).astype(float)
    if "Daytime_Sleepy" in df_sleep.columns:
        df_sleep["Sleepy_Risk"] = (df_sleep["Daytime_Sleepy"] >= 3).astype(float)
    if "BMI" in df_sleep.columns:
        df_sleep["BMI_Category"] = pd.cut(df_sleep["BMI"],
                                           bins=[0,18.5,25,30,100],
                                           labels=[0,1,2,3]).astype(float)

    df_sleep.to_csv("data/raw/sleep_nhanes.csv", index=False)
    ok(f"sleep_nhanes.csv  →  {len(df_sleep):,}행 × {df_sleep.shape[1]}열")
    print(f"      수면장애 비율: {df_sleep['Sleep_Disorder'].mean()*100:.1f}%")


# ══════════════════════════════════════════════════
# NHANES PHQ-9 병합 → mental_nhanes.csv
# ══════════════════════════════════════════════════
sec("NHANES 병합 ② → mental_nhanes.csv  (Mental Health 대체)")

# PHQ-9 사이클별 파일 + DEMO + BMX + PAQ 조인
mental_pairs = [
    ["data/raw/P_DPQ.XPT",      "data/raw/P_DEMO.XPT",      "data/raw/P_BMX.XPT",      "data/raw/P_PAQ.XPT",      "data/raw/P_SLQ.XPT"],
    ["data/raw/DPQ_I_2015.XPT", "data/raw/DEMO_I_2015.XPT", "data/raw/BMX_I_2015.XPT", "data/raw/PAQ_I_2015.XPT", "data/raw/SLQ_I_2015.XPT"],
    ["data/raw/DPQ_H_2013.XPT", "data/raw/DEMO_H_2013.XPT", "data/raw/BMX_H_2013.XPT", "data/raw/PAQ_H_2013.XPT", "data/raw/SLQ_H_2013.XPT"],
    ["data/raw/DPQ_G_2011.XPT", "data/raw/DEMO_G_2011.XPT", "data/raw/BMX_G_2011.XPT", "data/raw/PAQ_G_2011.XPT"],
    ["data/raw/DPQ_F_2009.XPT", "data/raw/DEMO_F_2009.XPT", "data/raw/BMX_F_2009.XPT", "data/raw/PAQ_F_2009.XPT"],
    ["data/raw/DPQ_E_2007.XPT", "data/raw/DEMO_E_2007.XPT", "data/raw/BMX_E_2007.XPT", "data/raw/PAQ_E_2007.XPT"],
    ["data/raw/DPQ_D_2005.XPT", "data/raw/DEMO_D_2005.XPT", "data/raw/BMX_D_2005.XPT", "data/raw/PAQ_D_2005.XPT"],
]
df_mental = load_nhanes_cycles(mental_pairs)

if not df_mental.empty:
    # ── PHQ-9 문항 컬럼명 표준화 ──
    phq_rename = {
        "DPQ010":"PHQ_Q1_Interest",      # 흥미/즐거움 저하
        "DPQ020":"PHQ_Q2_Depressed",     # 우울/절망감
        "DPQ030":"PHQ_Q3_Sleep",         # 수면 문제
        "DPQ040":"PHQ_Q4_Tired",         # 피로감
        "DPQ050":"PHQ_Q5_Appetite",      # 식욕 변화
        "DPQ060":"PHQ_Q6_Failure",       # 자책감
        "DPQ070":"PHQ_Q7_Concentrate",   # 집중력 저하
        "DPQ080":"PHQ_Q8_Moving",        # 행동 변화
        "DPQ090":"PHQ_Q9_Suicide",       # 자해 생각
        "DPQ100":"PHQ_Q10_Difficulty",   # 일상 기능 장애
    }
    demo_rename = {
        "RIDAGEYR":"Age","RIAGENDR":"Gender","INDFMPIR":"Income_Ratio",
        "RIDRETH1":"Race","DMDEDUC2":"Education",
    }
    bmx_rename  = {"BMXBMI":"BMI","BMXWAIST":"Waist_cm"}
    paq_rename  = {
        "PAQ605":"Vigorous_Work",     # 격렬한 업무 활동
        "PAQ620":"Moderate_Work",     # 중간 업무 활동
        "PAQ650":"Vigorous_Rec",      # 격렬한 여가 활동
        "PAQ665":"Moderate_Rec",      # 중간 여가 활동
        "PAD680":"Sedentary_Min",     # 앉아있는 시간(분/일)
    }
    slq_rename = {"SLD012":"Sleep_Hours_Wday","SLD010H":"Sleep_Hours_Old"}

    all_renames = {**phq_rename, **demo_rename, **bmx_rename, **paq_rename, **slq_rename}
    df_mental = df_mental.rename(columns={k:v for k,v in all_renames.items() if k in df_mental.columns})

    # ── PHQ-9 총점 계산 ──
    phq_cols = [c for c in phq_rename.values() if c in df_mental.columns and "Q10" not in c]
    if len(phq_cols) >= 7:
        # 7~9 (거의 매일): 3점 부여 / 결측 포함 시 제외
        df_mental[phq_cols] = df_mental[phq_cols].apply(
            pd.to_numeric, errors="coerce")
        df_mental["PHQ9_Score"] = df_mental[phq_cols].sum(axis=1, min_count=7)

        # ── 타겟: PHQ-9 ≥ 10 → 우울/번아웃 위험 (임상 기준) ──
        df_mental["Depression_Risk"] = (df_mental["PHQ9_Score"] >= 10).astype(float)

        # 중증도 레이블 (참고용)
        df_mental["PHQ9_Severity"] = pd.cut(
            df_mental["PHQ9_Score"],
            bins=[-1, 4, 9, 14, 19, 27],
            labels=["None","Mild","Moderate","ModSevere","Severe"])

    # ── 수면 시간 통합 ──
    if "Sleep_Hours_Wday" in df_mental.columns:
        df_mental["Sleep_Hours"] = df_mental["Sleep_Hours_Wday"]
    elif "Sleep_Hours_Old" in df_mental.columns:
        df_mental["Sleep_Hours"] = df_mental["Sleep_Hours_Old"]

    # ── 성별 인코딩 ──
    if "Gender" in df_mental.columns:
        df_mental["Gender_enc"] = (df_mental["Gender"] == 1).astype(float)

    # ── 신체활동 종합 점수 ──
    act_cols = [c for c in ["Vigorous_Work","Vigorous_Rec"] if c in df_mental.columns]
    if act_cols:
        df_mental["Active_Any"] = df_mental[act_cols].apply(
            lambda r: 1.0 if any(r==1) else 0.0, axis=1)

    # ── 필터링 ──
    if "Age" in df_mental.columns:
        df_mental = df_mental[df_mental["Age"] >= 18]
    core = [c for c in ["PHQ9_Score","Age","Depression_Risk"] if c in df_mental.columns]
    df_mental = df_mental.dropna(subset=core)
    if "BMI" in df_mental.columns:
        df_mental = df_mental[df_mental["BMI"].between(10, 80)]

    # ── 파생 피처 ──
    if "PHQ9_Score" in df_mental.columns:
        df_mental["PHQ9_High"]     = (df_mental["PHQ9_Score"] >= 10).astype(float)
        df_mental["PHQ9_Moderate"] = (df_mental["PHQ9_Score"].between(5,9)).astype(float)
    if "Sedentary_Min" in df_mental.columns:
        df_mental["Sedentary_4h+"] = (df_mental["Sedentary_Min"] >= 240).astype(float)

    df_mental.to_csv("data/raw/mental_nhanes.csv", index=False)
    ok(f"mental_nhanes.csv  →  {len(df_mental):,}행 × {df_mental.shape[1]}열")
    print(f"      우울 위험 비율(PHQ≥10): {df_mental['Depression_Risk'].mean()*100:.1f}%")
    print(f"      PHQ-9 평균 점수      : {df_mental['PHQ9_Score'].mean():.2f}")
    if "PHQ9_Severity" in df_mental.columns:
        print(f"      중증도 분포:")
        for sev, cnt in df_mental["PHQ9_Severity"].value_counts().sort_index().items():
            print(f"        {sev:<12}: {cnt:,}명 ({cnt/len(df_mental)*100:.1f}%)")
else:
    warn("PHQ-9 파일 없음 — 다운로드 후 재실행")


# ══════════════════════════════════════════════════
# Cardiovascular 전처리 → cardio_processed.csv
# ══════════════════════════════════════════════════
sec("Cardiovascular 전처리 → cardio_processed.csv")

if os.path.exists("data/raw/cardio_train.csv"):
    try:
        try:
            df_c = pd.read_csv("data/raw/cardio_train.csv", sep=";")
            if df_c.shape[1] < 5:
                df_c = pd.read_csv("data/raw/cardio_train.csv")
        except:
            df_c = pd.read_csv("data/raw/cardio_train.csv")

        if "age" in df_c.columns and df_c["age"].mean() > 1000:
            df_c["age"] = (df_c["age"] / 365.25).round(1)

        df_c = df_c.rename(columns={
            "age":"Age","gender":"Gender","height":"Height_cm","weight":"Weight_kg",
            "ap_hi":"BP_Systolic","ap_lo":"BP_Diastolic",
            "cholesterol":"Cholesterol","gluc":"Glucose",
            "smoke":"Smoker","alco":"Alcohol","active":"PhysActivity",
            "cardio":"HeartDisease"})

        if "BP_Systolic" in df_c.columns:
            df_c = df_c[df_c["BP_Systolic"].between(80,200) & df_c["BP_Diastolic"].between(40,130)]
        if "Age" in df_c.columns:
            df_c = df_c[df_c["Age"].between(18,80)]
        if "Height_cm" in df_c.columns:
            df_c["BMI"] = (df_c["Weight_kg"] / (df_c["Height_cm"]/100)**2).round(1)
            df_c = df_c[df_c["BMI"].between(10,60)]
        if "BP_Systolic" in df_c.columns:
            df_c["Pulse_Pressure"] = df_c["BP_Systolic"] - df_c["BP_Diastolic"]
            df_c["HighBP"]         = ((df_c["BP_Systolic"]>=140)|(df_c["BP_Diastolic"]>=90)).astype(int)
        if "Cholesterol" in df_c.columns:
            df_c["HighChol"] = (df_c["Cholesterol"] >= 2).astype(int)
        if "BMI" in df_c.columns:
            df_c["Obese"] = (df_c["BMI"] >= 30).astype(int)

        df_c.to_csv("data/processed/cardio_processed.csv", index=False)
        ok(f"cardio_processed.csv  →  {len(df_c):,}행")
        print(f"      심장병 비율: {df_c['HeartDisease'].mean()*100:.1f}%")
    except Exception as e:
        err(f"전처리 실패: {e}")
else:
    warn("cardio_train.csv 없음 → Kaggle에서 수동 다운로드 필요")
    print(f"      {B}https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset{RESET}")


# ══════════════════════════════════════════════════
# 최종 현황 요약
# ══════════════════════════════════════════════════
sec("최종 데이터셋 현황 요약")

FINAL_FILES = {
    "NHANES 수면 (병합)":          ("data/raw/sleep_nhanes.csv",           "자동"),
    "NHANES PHQ-9 우울 (병합)":    ("data/raw/mental_nhanes.csv",          "자동"),
    "BRFSS 2015":                  ("data/raw/brfss2015.csv",               "수동"),
    "Cardiovascular (원본)":       ("data/raw/cardio_train.csv",            "수동"),
    "  └ Cardiovascular (전처리)": ("data/processed/cardio_processed.csv",  "자동"),
}

missing = []
for name, (path, method) in FINAL_FILES.items():
    if os.path.exists(path):
        size = os.path.getsize(path)//1024
        ok(f"{name:<32} {size:>8,} KB  [{method}]")
    else:
        warn(f"{name:<32}    없음      [{method}]")
        if method == "수동":
            missing.append(name)

print()
print(f"  {BOLD}모델별 사용 데이터셋:{RESET}")
print(f"  {'모델 1 번아웃/우울':<20} mental_nhanes.csv   (PHQ-9 기반, ~30,000행)")
print(f"  {'모델 2 고혈압':<20} brfss2015.csv       (BRFSS, 253,680행)")
print(f"  {'모델 3 당뇨':<20} brfss2015.csv       (BRFSS, 253,680행)")
print(f"  {'모델 4 수면장애':<20} sleep_nhanes.csv    (NHANES, ~20,000행)")
print(f"  {'모델 5 심장질환':<20} cardio_processed.csv (70,000행)")

print()
if not missing:
    print(f"  {G}{BOLD}✓ 모든 데이터 준비 완료!{RESET}")
    print(f"  {B}→ 다음 단계: python 01_eda_preprocessing.py{RESET}")
else:
    print(f"  {Y}수동 다운로드 필요 ({len(missing)}개):{RESET}")
    for m in missing:
        print(f"    • {m}")
    print(f"\n  NHANES 데이터만으로도 모델 1, 4는 바로 실행 가능합니다.")

