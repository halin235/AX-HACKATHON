import pandas as pd
import re
from datetime import datetime

CSV_PATH = r"C:\Temp\voc_multichannel.csv"
OUTPUT_PATH = r"C:\Temp\cleaned_voc.csv"

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str)
print(f"[로드] {len(df)}행 읽음")

# ─────────────────────────────────────────────
# 2-A. 날짜 정규화
# 포맷: YYYY-MM-DD / YYYY/MM/DD / N월 M일 등
# ─────────────────────────────────────────────
parse_errors = []

def normalize_date(raw):
    if pd.isna(raw):
        return pd.NaT
    s = str(raw).strip()
    # YYYY-MM-DD 또는 YYYY/MM/DD
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    # N월 M일 (연도 없으면 2026 가정)
    m = re.fullmatch(r"(\d{1,2})월\s*(\d{1,2})일", s)
    if m:
        return f"2026-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    parse_errors.append(s)
    return pd.NaT

df["date"] = df["date"].apply(normalize_date)

# ─────────────────────────────────────────────
# 2-B. 중복 제거 (전체 컬럼 기준, 첫 행 유지)
# ─────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(keep="first")
print(f"[중복 제거] {before - len(df)}행 제거 → {len(df)}행 남음")

# ─────────────────────────────────────────────
# 2-C. 결측치 처리
# ─────────────────────────────────────────────
df["channel"]   = df["channel"].fillna("N/A").replace("", "N/A")
df["user_type"] = df["user_type"].fillna("N/A").replace("", "N/A")

# ─────────────────────────────────────────────
# 3-A. 유형 분류
# 우선순위: 버그 > 기능요청 > 칭찬 > 일반문의
# 버그+기능요청 동시 해당 → 버그
# ─────────────────────────────────────────────
BUG_KW = [
    "안됩니다", "안돼", "오류", "에러", "버그", "먹통", "멈춥", "멈추", "안열", "안 열",
    "실패", "안뜨", "안 뜨", "깨져", "사라졌", "날아갔", "복구", "손실", "중단",
    "무한", "충돌", "안보", "보이지 않", "블랙", "죽", "튕겨", "튕기", "막혔",
    "로딩만", "안받", "받지 못", "뜨지 않", "되지 않", "되질 않", "반응이 없"
]
FEATURE_KW = [
    "추가해", "추가하", "지원해", "지원하", "기능이 없", "기능이 있으면", "기능을 원",
    "원합니다", "원해요", "요청", "바랍니다", "바래요", "이어주", "연동해", "연동하",
    "넣어주", "만들어주", "제공해", "생겼으면", "있었으면", "있으면 좋", "개선",
    "넣어 주", "해주세요", "해 주세요", "구현", "업그레이드", "확장", "적용해주",
    "할 수 있게", "할수있게"
]
PRAISE_KW = [
    "감사합니다", "감사해요", "고맙습니다", "만족합니다", "만족해요", "편리해",
    "좋습니다", "좋아요", "최고입니다", "훌륭합니다", "훌륭해요", "덕분에",
    "도움이 됩니다", "잘 됩니다", "잘됩니다", "도움이 되", "편해졌", "줄었습니다",
    "즐거워", "기뻐", "쾌적", "사랑합니다"
]

def classify_type(text):
    if pd.isna(text):
        return "일반문의"
    t = str(text)
    is_bug     = any(kw in t for kw in BUG_KW)
    is_feature = any(kw in t for kw in FEATURE_KW)
    is_praise  = any(kw in t for kw in PRAISE_KW)
    # 버그 우선 (버그+기능요청 동시 → 버그)
    if is_bug:
        return "버그"
    if is_feature:
        return "기능요청"
    if is_praise:
        return "칭찬"
    return "일반문의"

df["유형분류"] = df["content"].apply(classify_type)

# ─────────────────────────────────────────────
# 3-B. 감정 분류
# 부정 표현이 하나라도 있으면 '부정' 처리
# ─────────────────────────────────────────────
NEG_KW = [
    "불편", "불만", "문제", "오류", "에러", "버그", "안됩", "안되", "안받", "실패",
    "힘듭니다", "곤란", "어렵", "짜증", "답답", "너무 느", "느려서", "먹통",
    "손실", "날아갔", "사라졌", "복구", "먹히지", "안열", "안뜨", "깨져", "이상해",
    "이상합니다", "안됩니다", "않습니다", "못하고", "막혔", "중단", "무한 로딩",
    "로딩만", "튕겨", "충돌", "취소", "환불", "탈퇴", "해지", "신고", "항의",
    "고장", "빠져나갔", "초과", "오청구", "잘못", "이해할 수 없", "황당"
]
POS_KW = [
    "감사합니다", "감사해요", "고맙습니다", "만족합니다", "편리해", "좋습니다",
    "좋아요", "최고입니다", "훌륭합니다", "덕분에", "도움이 됩니다", "편해졌",
    "기쁩니다", "줄었습니다", "즐거워", "쾌적", "사랑합니다", "반했습니다",
    "칭찬", "완벽해", "감동"
]

def classify_sentiment(text):
    if pd.isna(text):
        return "중립"
    t = str(text)
    # 부정 표현이 하나라도 있으면 무조건 '부정'
    if any(kw in t for kw in NEG_KW):
        return "부정"
    if any(kw in t for kw in POS_KW):
        return "긍정"
    return "중립"

df["감정분류"] = df["content"].apply(classify_sentiment)

# ─────────────────────────────────────────────
# 3-C. 요약 (30자 이내 핵심 요약)
# content 앞부분을 잘라 마지막 문장 경계에서 끊음
# ─────────────────────────────────────────────
def summarize(text, max_len=30):
    if pd.isna(text):
        return ""
    t = str(text).strip()
    if len(t) <= max_len:
        return t
    # 문장 구분자(. ! ? 기준) 내에서 자르기
    truncated = t[:max_len]
    for sep in [".", "!", "?", "。"]:
        pos = truncated.rfind(sep)
        if pos > 10:
            return truncated[:pos + 1]
    return truncated + "…"

df["요약"] = df["content"].apply(summarize)

# ─────────────────────────────────────────────
# 4. 집계: 유형별 건수 및 비율
# ─────────────────────────────────────────────
type_agg = (
    df["유형분류"]
    .value_counts()
    .rename_axis("유형")
    .reset_index(name="건수")
)
type_agg["비율(%)"] = (type_agg["건수"] / len(df) * 100).round(1)
type_agg["순위"] = range(1, len(type_agg) + 1)

# ─────────────────────────────────────────────
# 5. 즉시 대응 TOP3
# severity_hint='높음' AND 감정='부정' 중 심각한 3건
# 심각도 순위: severity_hint 높음 > 보통 > 낮음, 그 다음 날짜 최근순
# ─────────────────────────────────────────────
SEVERITY_ORDER = {"높음": 0, "보통": 1, "낮음": 2, "N/A": 3}

urgent = df[
    (df["severity_hint"] == "높음") & (df["감정분류"] == "부정")
].copy()

urgent["_sev_rank"] = urgent["severity_hint"].map(SEVERITY_ORDER).fillna(3)
urgent_sorted = urgent.sort_values(["_sev_rank", "date"], ascending=[True, False])
top3 = urgent_sorted.head(3)[["id", "date", "channel", "content", "유형분류", "요약"]]

# ─────────────────────────────────────────────
# 6. 저장 및 출력
# ─────────────────────────────────────────────
df.drop(columns=["_sev_rank"] if "_sev_rank" in df.columns else [], errors="ignore")
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n[저장] {OUTPUT_PATH}\n")

# 날짜 파싱 이슈 보고
if parse_errors:
    print(f"[날짜 변환 실패] {len(parse_errors)}건: {parse_errors}\n")
else:
    print("[날짜 변환 이슈] 없음\n")

# ── 미리보기 10행 ──────────────────────────────
SHOW_COLS = ["id", "date", "channel", "user_type", "severity_hint", "유형분류", "감정분류", "요약"]
print("=" * 80)
print("▶ 최종 데이터프레임 상위 10행")
print("=" * 80)
pd.set_option("display.max_colwidth", 35)
pd.set_option("display.width", 120)
print(df[SHOW_COLS].head(10).to_string(index=False))

# ── 유형별 집계표 ─────────────────────────────
print("\n" + "=" * 40)
print("▶ 유형별 집계표")
print("=" * 40)
print(type_agg[["순위", "유형", "건수", "비율(%)"]].to_string(index=False))

# ── 즉시 대응 TOP3 ────────────────────────────
print("\n" + "=" * 80)
print("▶ 즉시 대응 필요 TOP3  (severity=높음 ∩ 감정=부정)")
print("=" * 80)
for i, (_, row) in enumerate(top3.iterrows(), 1):
    print(f"\n[{i}위] ID={row['id']}  날짜={row['date']}  채널={row['channel']}")
    print(f"     유형: {row['유형분류']}")
    print(f"     요약: {row['요약']}")
    print(f"     원문: {str(row['content'])[:80]}")
