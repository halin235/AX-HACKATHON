"""
app.py — VOC 자동 분류 파이프라인 Streamlit 웹 UI  (v2 · 파일 업로드 지원)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from voc_pipeline import PipelineConfig, PipelineResult, VOCPipeline  # noqa: E402

# ── 상수 ───────────────────────────────────────────────────────────────────────
DEFAULT_CSV   = str(ROOT / "final_submission" / "data" / "05_voc_multichannel.csv")
OUTPUT_DIR    = str(ROOT / "final_submission" / "output")
ENCODINGS     = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
REQUIRED_COLS = {"id", "date", "channel", "content", "user_type"}

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VOC 자동 분류 파이프라인",
    page_icon="📊",
    layout="wide",
)

# ── 세션 상태 초기화 ───────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None


# ── 유틸리티 ───────────────────────────────────────────────────────────────────
def read_csv_auto(source) -> pd.DataFrame:
    """파일 경로(str) 또는 UploadedFile 객체를 받아 인코딩 자동 감지 후 DataFrame 반환."""
    for enc in ENCODINGS:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(
                source,
                encoding=enc,
                dtype=str,
                keep_default_na=False,
                na_values=[""],
            )
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"CSV 인코딩 감지 실패. 지원 인코딩: {ENCODINGS}")


# ── 사이드바 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 파이프라인 설정")
    top_n      = st.slider("즉시대응 TOP-N", min_value=1, max_value=10, value=3)
    infer_year = st.text_input("날짜 추론 연도", value="2026")
    st.divider()
    st.markdown("**출력 경로**")
    st.code(
        "final_submission/output/\n"
        "  ├ cleaned_voc.csv\n"
        "  ├ report.json\n"
        "  └ report_summary.md",
        language="text",
    )
    st.divider()
    st.caption("VocPipeline v2.0 · Streamlit")


# ── 헤더 ───────────────────────────────────────────────────────────────────────
st.title("📊 VOC 자동 분류 파이프라인")
st.caption(
    "SaaS 고객 피드백을 자동 분류하고 비즈니스 임팩트 기반으로 "
    "즉시 대응 이슈를 선별합니다."
)
st.divider()


# ── 데이터 소스 선택 ────────────────────────────────────────────────────────────
st.subheader("📂 데이터 소스")

uploaded = st.file_uploader(
    label="CSV 파일 업로드 (생략 시 기본 샘플 데이터 사용)",
    type=["csv"],
    help=(
        f"필수 컬럼: {', '.join(sorted(REQUIRED_COLS))}  |  "
        "인코딩: UTF-8 / CP949 자동 감지"
    ),
)

file_valid = False  # 이 렌더링 사이클에서 업로드 파일이 유효한지

if uploaded is not None:
    try:
        preview_df = read_csv_auto(uploaded)
        missing = REQUIRED_COLS - set(preview_df.columns)
        if missing:
            st.error(
                f"❌ 필수 컬럼 누락: **{', '.join(sorted(missing))}**\n\n"
                f"필요한 컬럼: `{', '.join(sorted(REQUIRED_COLS))}`"
            )
        else:
            file_valid = True
            size_kb = uploaded.size / 1024
            st.success(
                f"✅ **{uploaded.name}** 업로드 완료 "
                f"({len(preview_df):,}행 · {size_kb:.1f} KB)"
            )
            with st.expander("미리보기 — 처음 5행", expanded=False):
                st.dataframe(
                    preview_df.head(5),
                    use_container_width=True,
                    hide_index=True,
                )
    except Exception as exc:
        st.error(f"❌ 파일 읽기 실패: {exc}")
else:
    st.info("📁 기본 샘플 데이터 사용: `05_voc_multichannel.csv` (300건)")

st.divider()


# ── 실행 버튼 ──────────────────────────────────────────────────────────────────
if st.button("🚀 파이프라인 실행", type="primary", use_container_width=True):
    st.session_state.result = None  # 이전 결과 초기화

    cfg      = PipelineConfig(top_n=top_n, infer_year=infer_year)
    pipeline = VOCPipeline(cfg)
    warns: List[str] = []

    try:
        with st.status("파이프라인 실행 중...", expanded=True) as status:

            # [1/5] 로드
            st.write("📂 [1/5] CSV 로딩")
            if file_valid and uploaded is not None:
                raw = read_csv_auto(uploaded)
                source_label = f"업로드 파일 ({uploaded.name})"
            else:
                raw = pipeline._load(DEFAULT_CSV, warns)
                source_label = "기본 샘플 데이터"
            st.write(f"　　→ {source_label}: {len(raw)}행")

            # [2/5] 정제
            st.write(f"🧹 [2/5] 정제: {len(raw)}행 입력")
            cleaned = pipeline.cleaner.clean(raw)
            dup = cleaned.attrs.get("duplicates_removed", 0)
            st.write(f"　　→ {len(cleaned)}행 확정 (중복 {dup}건 제거)")

            # [3/5] 분류
            st.write("🏷️ [3/5] 분류: 유형 · 감정 · 요약 컬럼 추가")
            classified = pipeline.classifier.apply(cleaned)
            classified = pipeline.priority.score_all(classified)

            # [4/5] 우선순위
            st.write(f"🎯 [4/5] 우선순위: Priority Score 산정 → TOP {top_n} 선정")
            top_n_df = pipeline.priority.get_top_n(classified)

            # [5/5] 저장
            st.write(f"💾 [5/5] 저장: {OUTPUT_DIR}/")
            out_path = Path(OUTPUT_DIR)
            out_path.mkdir(parents=True, exist_ok=True)
            pipeline._save(classified, top_n_df, out_path)

            result = PipelineResult(
                df=classified,
                agg=pipeline.reporter.aggregate(classified),
                top_n=top_n_df,
                output_dir=str(out_path.resolve()),
                warnings=warns,
            )
            st.session_state.result = result
            status.update(
                label=f"✅ 완료! {len(classified)}건 처리",
                state="complete",
                expanded=False,
            )

    except FileNotFoundError as exc:
        st.error(f"❌ 파일 없음: {exc}")
    except Exception as exc:
        st.exception(exc)


# ── 결과 표시 ──────────────────────────────────────────────────────────────────
result: Optional[PipelineResult] = st.session_state.result

if result is None:
    st.info("위의 **파이프라인 실행** 버튼을 클릭하면 결과가 여기에 표시됩니다.")

    with st.expander("파이프라인 구조 보기"):
        st.code(
            "*.csv 입력\n"
            "  └─▶ [1] Load     인코딩 자동 감지 (UTF-8 / UTF-8-BOM / CP949)\n"
            "  └─▶ [2] Clean    날짜 정규화 · 결측치 보정 · 중복 제거\n"
            "  └─▶ [3] Classify 규칙 기반 유형·감정 분류 (전략 패턴)\n"
            "  └─▶ [4] Score    Priority Score = Freq × U × S × E × C × Churn\n"
            "  └─▶ [5] Report   cleaned_voc.csv + report.json + report_summary.md",
            language="text",
        )

    st.markdown(
        "**Priority Score 핵심 가중치**\n\n"
        "| 차원 | 기업고객 | 유료 | 무료 |\n"
        "|---|---|---|---|\n"
        "| 고객 등급 | ×3.0 | ×2.0 | ×1.0 |\n\n"
        "| 버그 | 기능요청 | 부정 감정 | 이탈 위협 |\n"
        "|---|---|---|---|\n"
        "| ×1.3 | ×1.0 | ×1.5 | ×2.0 |"
    )

else:
    # 경고 메시지
    for w in result.warnings:
        st.warning(w)

    # ── 요약 지표 ──────────────────────────────────────────────────────────────
    type_counts = result.df["유형분류"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 처리 건수", f"{len(result.df)}건")
    c2.metric("버그",      f"{type_counts.get('버그', 0)}건")
    c3.metric("기능요청",  f"{type_counts.get('기능요청', 0)}건")
    c4.metric("즉시대응",  f"{len(result.top_n)}건")

    st.divider()

    # ── 유형별 분포 + 집계 테이블 ──────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("유형별 분포")
        st.bar_chart(result.agg.set_index("유형")["건수"])
    with col_r:
        st.subheader("집계 테이블")
        st.dataframe(result.agg, use_container_width=True, hide_index=True)

    st.divider()

    # ── 즉시대응 TOP-N ──────────────────────────────────────────────────────────
    st.subheader(f"🔴 즉시대응 TOP {len(result.top_n)}")
    top_cols = [
        c for c in
        ["id", "date", "channel", "user_type", "유형분류", "감정분류", "priority_score", "content"]
        if c in result.top_n.columns
    ]
    st.dataframe(
        result.top_n[top_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "priority_score": st.column_config.NumberColumn(
                "Priority Score", format="%.1f"
            ),
            "content": st.column_config.TextColumn("피드백 내용", width="large"),
        },
    )

    st.divider()

    # ── 전체 분류 결과 ──────────────────────────────────────────────────────────
    with st.expander(f"전체 분류 결과 ({len(result.df)}건)", expanded=False):
        st.dataframe(result.df, use_container_width=True, hide_index=True)

    st.divider()

    # ── 다운로드 버튼 ──────────────────────────────────────────────────────────
    st.subheader("📥 결과 파일 다운로드")
    out = Path(result.output_dir)
    d1, d2, d3 = st.columns(3)

    csv_path = out / "cleaned_voc.csv"
    if csv_path.exists():
        d1.download_button(
            label="📄 cleaned_voc.csv",
            data=csv_path.read_bytes(),
            file_name="cleaned_voc.csv",
            mime="text/csv",
            use_container_width=True,
        )

    json_path = out / "report.json"
    if json_path.exists():
        d2.download_button(
            label="📦 report.json",
            data=json_path.read_text(encoding="utf-8"),
            file_name="report.json",
            mime="application/json",
            use_container_width=True,
        )

    md_path = out / "report_summary.md"
    if md_path.exists():
        d3.download_button(
            label="📝 report_summary.md",
            data=md_path.read_text(encoding="utf-8"),
            file_name="report_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )
