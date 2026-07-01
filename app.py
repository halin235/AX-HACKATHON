"""
app.py — VOC 자동 분류 파이프라인 Streamlit 웹 UI
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from voc_pipeline import PipelineConfig, PipelineResult, VOCPipeline  # noqa: E402

# ── 경로 상수 ──────────────────────────────────────────────────────────────────
INPUT_CSV  = str(ROOT / "final_submission" / "data" / "05_voc_multichannel.csv")
OUTPUT_DIR = str(ROOT / "final_submission" / "output")

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VOC 자동 분류 파이프라인",
    page_icon="📊",
    layout="wide",
)

# ── 세션 상태 초기화 ───────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

# ── 사이드바 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 파이프라인 설정")
    top_n      = st.slider("즉시대응 TOP-N", min_value=1, max_value=10, value=3)
    infer_year = st.text_input("날짜 추론 연도", value="2026")
    st.divider()
    st.markdown("**입력 파일**")
    st.code("final_submission/data/\n  05_voc_multichannel.csv", language="text")
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

# ── 실행 버튼 ──────────────────────────────────────────────────────────────────
if st.button("🚀 파이프라인 실행", type="primary", use_container_width=True):
    cfg      = PipelineConfig(top_n=top_n, infer_year=infer_year)
    pipeline = VOCPipeline(cfg)
    warns: List[str] = []

    try:
        with st.status("파이프라인 실행 중...", expanded=True) as status:

            st.write("📂 [1/5] CSV 로딩 — 인코딩 자동 감지")
            raw = pipeline._load(INPUT_CSV, warns)

            st.write(f"🧹 [2/5] 정제: {len(raw)}행 입력")
            cleaned = pipeline.cleaner.clean(raw)
            dup = cleaned.attrs.get("duplicates_removed", 0)
            st.write(f"　　→ {len(cleaned)}행 확정 (중복 {dup}건 제거)")

            st.write("🏷️ [3/5] 분류: 유형 · 감정 · 요약 컬럼 추가")
            classified = pipeline.classifier.apply(cleaned)
            classified = pipeline.priority.score_all(classified)

            st.write(f"🎯 [4/5] 우선순위: Priority Score 산정 → TOP {top_n} 선정")
            top_n_df = pipeline.priority.get_top_n(classified)

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

    except FileNotFoundError as e:
        st.error(f"❌ 입력 파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        st.exception(e)

# ── 결과 표시 ──────────────────────────────────────────────────────────────────
result: PipelineResult | None = st.session_state.result

if result is None:
    # 초기 화면
    st.info(
        "사이드바에서 설정을 확인한 뒤 위의 **파이프라인 실행** 버튼을 클릭하세요."
    )

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
            "content": st.column_config.TextColumn(
                "피드백 내용", width="large"
            ),
        },
    )

    st.divider()

    # ── 전체 분류 결과 (접을 수 있음) ──────────────────────────────────────────
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
