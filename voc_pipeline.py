"""
voc_pipeline.py — VoC 자동 분석 파이프라인 엔진 v2.0
=======================================================

[사용법] CLI:
    python voc_pipeline.py --input raw_voc.csv --output ./output

[사용법] Python import:
    from voc_pipeline import VOCPipeline, PipelineConfig
    result = VOCPipeline().run("raw_voc.csv", "./output")
    print(result.summary())

[입력 (Input)]
    raw CSV 파일 — 컬럼: id, date, channel, content, user_type, severity_hint
    인코딩: UTF-8 / UTF-8-BOM / CP949 자동 감지

[출력 (Output)]  지정 output 디렉토리 아래 3개 파일
    cleaned_voc.csv      정제 + 분류 완료 데이터 (UTF-8 BOM)
    report.json          유형 집계 + TOP-N JSON
    report_summary.md    사람이 읽을 수 있는 요약 리포트 (Markdown)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1.  설정 (Configuration)
#     모든 하드코딩 값은 이 블록에서만 관리합니다.
#     키워드는 ClassificationRules.from_json()으로 외부 JSON에서 로드 가능합니다.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassificationRules:
    """유형·감정 분류 키워드 룰셋.

    DEC-01: 버그 키워드 최우선 — 복합 케이스에서 버그 과소집계 방지
    DEC-02: 부정 감정 절대 우선 — 이탈 선행 신호 누락 방지
    """

    bug_kw: List[str] = field(default_factory=lambda: [
        "안됩니다", "안되", "오류", "에러", "버그", "먹통", "멈춥", "깨져",
        "사라졌", "복구", "손실", "중단", "무한", "충돌", "튕겨", "실패",
        "강제종료", "빈화면", "빈 화면", "반응없", "반응 없",
        "응답없", "응답 없", "데이터가 날", "날아갔", "날아가서", "소실",
        "삭제됐", "두 번씩", "계속 멈", "계속멈",
        "막혔", "막혀서", "접속이 안", "이용 불가",  # 간접 장애 표현 (DEC-05 개선)
    ])
    feature_kw: List[str] = field(default_factory=lambda: [
        "추가해", "지원해", "기능이 없", "원합니다", "요청", "바랍니다",
        "넣어주", "해주세요", "구현", "있으면 좋", "있었으면",
        "되면 좋", "필요합니다", "필요해요", "원해요",
    ])
    praise_kw: List[str] = field(default_factory=lambda: [
        "감사합니다", "만족합니다", "편리해", "좋습니다", "최고입니다",
        "훌륭합니다", "덕분에", "도움이 됩니다", "도움이 됐", "정말 좋아요",
        "너무 좋아요", "최고예요", "훌륭해요", "만족해요",
        "완벽합니다", "놀랍습니다", "잘 만들",
    ])
    neg_kw: List[str] = field(default_factory=lambda: [
        "불편", "불만", "문제", "오류", "에러", "버그", "안됩", "실패",
        "손실", "사라졌", "복구", "충돌", "환불", "탈퇴", "해지", "화가",
        "답답", "짜증", "엉망", "최악", "말이 됩니까", "자격이 없",
        "큰 피해", "법적", "이게 뭡니까",
        "막혔", "막혀서",                            # 서비스 차단 표현 추가
    ])
    pos_kw: List[str] = field(default_factory=lambda: [
        "감사합니다", "만족합니다", "편리해", "좋습니다", "최고입니다",
        "덕분에", "도움이 됩니다", "정말 좋아요", "너무 좋아요",
        "최고예요", "훌륭해요", "만족해요", "완벽합니다", "놀랍습니다",
    ])

    @classmethod
    def from_json(cls, path: str) -> "ClassificationRules":
        """JSON 파일에서 키워드 룰을 로드합니다.

        JSON 스키마:
            {
                "bug_kw":     ["오류", ...],
                "feature_kw": ["추가해", ...],
                "praise_kw":  ["감사합니다", ...],
                "neg_kw":     ["불편", ...],
                "pos_kw":     ["감사합니다", ...]
            }
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass
class PipelineConfig:
    """파이프라인 전체 설정.

    다른 CSV나 분류 기준을 쓰려면 이 객체만 교체하면 됩니다.
    """

    rules: ClassificationRules = field(default_factory=ClassificationRules)

    summary_max_len:  int  = 30         # 요약 최대 글자 수
    urgent_severity:  str  = "높음"     # 즉시 대응 심각도 기준
    urgent_sentiment: str  = "부정"     # 즉시 대응 감정 기준
    top_n:            int  = 3          # 즉시 대응 TOP-N 개수
    infer_year:       str  = "2026"     # N월M일 형식 연도 추론값
    missing_placeholder: str = "N/A"   # 결측치 대체 값

    # 고객 등급 우선순위 (낮을수록 높은 우선순위) — DEC-05
    user_priority: Dict[str, int] = field(default_factory=lambda: {
        "기업고객": 0,
        "유료":     1,
        "무료":     2,
        "N/A":      3,
    })

    # 출력 파일명
    output_cleaned_csv: str = "cleaned_voc.csv"
    output_report_json: str = "report.json"
    output_report_md:   str = "report_summary.md"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  정제 (VocCleaner)
# ══════════════════════════════════════════════════════════════════════════════

class VocCleaner:
    """날짜 정규화 · 결측치 보정 · 중복 제거."""

    _DATE_RULES: List[Tuple[re.Pattern, object]] = [
        (
            re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"),
            lambda m, _: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
        ),
        (
            re.compile(r"(\d{1,2})월\s*(\d{1,2})일"),
            lambda m, yr: f"{yr}-{int(m.group(1)):02d}-{int(m.group(2)):02d}",
        ),
    ]

    def __init__(self, config: PipelineConfig):
        self.config = config

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)
        # content 기준 중복 제거: ID가 다르더라도 동일 피드백이면 낮은 ID를 원본으로 보존
        df = df.copy().sort_values("id").drop_duplicates(subset=["content"], keep="first")
        df.attrs["duplicates_removed"] = original_len - len(df)

        df["date"] = df["date"].apply(
            lambda x: self._normalize_date(x, self.config.infer_year)
        )
        ph = self.config.missing_placeholder
        df["channel"]   = df["channel"].replace("", ph).fillna(ph)
        df["user_type"] = df["user_type"].replace("", ph).fillna(ph)
        return df

    def _normalize_date(self, raw: str, yr: str) -> str:
        for pattern, fmt in self._DATE_RULES:
            m = pattern.search(str(raw))
            if m:
                return fmt(m, yr)
        return str(raw).strip()   # 파싱 실패 시 원본 보존


# ══════════════════════════════════════════════════════════════════════════════
# 3.  분류 (VocClassifier)
# ══════════════════════════════════════════════════════════════════════════════

class VocClassifier:
    """유형 분류 · 감정 분류 · 30자 요약."""

    def __init__(self, config: PipelineConfig):
        self.rules   = config.rules
        self.max_len = config.summary_max_len

    def classify_type(self, text: str) -> str:
        t = str(text)
        # DEC-01: 버그 최우선 → 조기 리턴
        if any(kw in t for kw in self.rules.bug_kw):     return "버그"
        if any(kw in t for kw in self.rules.feature_kw): return "기능요청"
        if any(kw in t for kw in self.rules.praise_kw):  return "칭찬"
        return "일반문의"

    def classify_sentiment(self, text: str) -> str:
        t = str(text)
        # DEC-02: 부정 절대 우선
        if any(kw in t for kw in self.rules.neg_kw): return "부정"
        if any(kw in t for kw in self.rules.pos_kw): return "긍정"
        return "중립"

    def summarize(self, text: str) -> str:
        t = str(text).strip()
        if len(t) <= self.max_len:
            return t
        for ch in (".", "?", "!", "。"):
            pos = t[: self.max_len].rfind(ch)
            if pos > self.max_len // 2:
                return t[: pos + 1]
        return t[: self.max_len - 1] + "…"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["유형분류"] = df["content"].apply(self.classify_type)
        df["감정분류"] = df["content"].apply(self.classify_sentiment)
        df["요약"]     = df["content"].apply(self.summarize)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  우선순위 산정 (PriorityEngine)
# ══════════════════════════════════════════════════════════════════════════════

class PriorityEngine:
    """즉시 대응 TOP-N 선정.

    선정 기준 (DEC-05):
        1차  severity_hint = urgent_severity  ∩  감정분류 = urgent_sentiment
        2차  고객 등급 (user_priority 낮을수록 우선)
        3차  날짜 내림차순 (최신 우선)

    유형분류를 기준에서 의도적으로 제외:
        분류 오류가 긴급도 판단 오류로 전파되는 것을 방지합니다.
    """

    CHURN_KEYWORDS: List[str] = ["해지", "탈퇴", "분쟁", "카드사"]
    _USER_W: Dict[str, float] = {"기업고객": 3.0, "유료": 2.0, "무료": 1.0}
    _SEV_W:  Dict[str, float] = {"높음": 3.0, "보통": 2.0, "낮음": 1.0}
    _SENT_W: Dict[str, float] = {"부정": 1.5, "중립": 1.0, "긍정": 0.5}
    _TYPE_W: Dict[str, float] = {"버그": 1.3, "기능요청": 1.0, "일반문의": 1.0, "칭찬": 0.0}

    def __init__(self, config: PipelineConfig):
        self.config = config

    def score_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """priority_score 컬럼을 전체 행에 추가합니다.

        priority_score = Frequency × (U × S × E × C) × ChurnMultiplier
        R-03~R-05 설계 원칙 구현체 — decisions.md 참조
        """
        freq_map = df["유형분류"].value_counts().to_dict()

        def _row_score(row: pd.Series) -> float:
            u     = self._USER_W.get(str(row["user_type"]), 1.0)
            s     = self._SEV_W.get(str(row.get("severity_hint", "")), 1.0)
            e     = self._SENT_W.get(str(row["감정분류"]), 1.0)
            c     = self._TYPE_W.get(str(row["유형분류"]), 1.0)
            churn = 2.0 if any(kw in str(row["content"]) for kw in self.CHURN_KEYWORDS) else 1.0
            freq  = float(freq_map.get(str(row["유형분류"]), 1))
            return round(freq * u * s * e * c * churn, 2)

        df = df.copy()
        df["priority_score"] = df.apply(_row_score, axis=1)
        return df

    def get_top_n(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        mask = (
            (df["severity_hint"] == cfg.urgent_severity) &
            (df["감정분류"]       == cfg.urgent_sentiment)
        )
        urgent = df[mask].copy()
        urgent["_rank"] = (
            urgent["user_type"].map(cfg.user_priority).fillna(99).astype(int)
        )
        return (
            urgent
            .sort_values(["_rank", "date"], ascending=[True, False])
            .head(cfg.top_n)
            .drop(columns=["_rank"])
            .reset_index(drop=True)
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  리포트 생성 (ReportGenerator)
# ══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """집계 · JSON · Markdown 리포트 생성."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df["유형분류"].value_counts().reset_index()
        agg.columns = ["유형", "건수"]
        agg["비율(%)"] = (agg["건수"] / len(df) * 100).round(1)
        return agg

    def to_json(self, df: pd.DataFrame, top_n: pd.DataFrame) -> dict:
        agg = self.aggregate(df)
        top_n_cols = [
            "id", "channel", "user_type", "date",
            "유형분류", "감정분류", "severity_hint", "content", "요약",
        ]
        present = [c for c in top_n_cols if c in top_n.columns]
        return {
            "total": len(df),
            "aggregation": agg.to_dict(orient="records"),
            "top_n": top_n[present].to_dict(orient="records"),
        }

    def to_markdown(self, df: pd.DataFrame, top_n: pd.DataFrame) -> str:
        agg = self.aggregate(df)
        lines = [
            "# VoC 분류 분석 요약 리포트",
            "",
            f"> 총 **{len(df)}건** 분석 완료  ",
            "> 파이프라인: VocPipeline v2.0  ",
            "> 분류 우선순위: 버그 > 기능요청 > 칭찬 > 일반문의 / 감정: 부정 절대 우선",
            "",
            "---",
            "",
            "## 유형별 빈도",
            "",
            "| 유형 | 건수 | 비율(%) |",
            "|---|:---:|:---:|",
        ]
        for _, r in agg.iterrows():
            lines.append(f"| {r['유형']} | {r['건수']} | {r['비율(%)']}% |")
        lines.append(f"| **합계** | **{len(df)}** | **100.0%** |")

        cfg = self.config
        lines += [
            "",
            f"## 즉시 대응 필요 TOP {cfg.top_n}",
            "",
            f"> 기준: `severity_hint={cfg.urgent_severity}` ∩ `감정분류={cfg.urgent_sentiment}`  ",
            "> 정렬: 고객 등급(기업고객 > 유료 > 무료) → 최신일 순",
            "",
        ]
        for i, (_, r) in enumerate(top_n.iterrows(), 1):
            ut   = r.get("user_type", "N/A")
            sev  = r.get("severity_hint", "-")
            lines += [
                f"### 🔴 TOP {i} — `{r['id']}` ({ut} / {r['channel']} / {r['date']})",
                "",
                "| 항목 | 내용 |",
                "|---|---|",
                f"| **원문** | {r['content']} |",
                f"| **심각도** | {sev} |",
                f"| **유형** | {r['유형분류']} |",
                f"| **감정** | {r['감정분류']} |",
                "",
            ]

        lines += [
            "---",
            "*Generated by VocPipeline v2.0*",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  파이프라인 결과 컨테이너 (PipelineResult)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    df:         pd.DataFrame   # 정제 + 분류된 전체 데이터
    agg:        pd.DataFrame   # 유형별 집계
    top_n:      pd.DataFrame   # 즉시 대응 TOP-N
    output_dir: str            # 결과 저장 경로
    warnings:   List[str]      # 처리 중 발생한 경고

    def summary(self) -> str:
        """터미널 출력용 요약."""
        W = 60
        lines = [
            "=" * W,
            "  VocPipeline v2.0 — 실행 결과",
            "=" * W,
            f"  처리 행 수  : {len(self.df)}건",
            f"  출력 경로   : {self.output_dir}",
        ]
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        lines += ["", "  [유형별 집계]"]
        for _, r in self.agg.iterrows():
            bar = "█" * max(1, int(r["비율(%)"] / 5))
            lines.append(
                f"    {r['유형']:6s}  {r['건수']:3d}건  {r['비율(%)']:5.1f}%  {bar}"
            )
        lines += ["", f"  [즉시 대응 TOP {len(self.top_n)}]"]
        for i, (_, r) in enumerate(self.top_n.iterrows(), 1):
            ut = r.get("user_type", "N/A")
            preview = str(r["content"])
            if len(preview) > 58:
                preview = preview[:57] + "…"
            lines += [
                f"    TOP{i} [{r['id']}] {ut} | {r['channel']} | {r['date']}",
                f"         {preview}",
            ]
        lines.append("=" * W)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  메인 파이프라인 (VOCPipeline)
# ══════════════════════════════════════════════════════════════════════════════

class VOCPipeline:
    """VoC 자동 분석 파이프라인 엔진.

    raw CSV → 정제 → 분류 → 우선순위 산정 → 리포트 저장
    전 과정을 단일 run() 호출로 수행합니다.

    예시:
        # 기본 설정으로 실행
        result = VOCPipeline().run("raw_voc.csv", "./output")

        # 설정 커스터마이즈
        config = PipelineConfig(top_n=5, infer_year="2025")
        result = VOCPipeline(config).run("raw_voc.csv", "./output")

        # 키워드 룰을 외부 JSON으로 관리
        rules  = ClassificationRules.from_json("my_rules.json")
        config = PipelineConfig(rules=rules)
        result = VOCPipeline(config).run("raw_voc.csv", "./output")

        # 결과 확인
        print(result.summary())
    """

    _ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config     = config or PipelineConfig()
        self.cleaner    = VocCleaner(self.config)
        self.classifier = VocClassifier(self.config)
        self.priority   = PriorityEngine(self.config)
        self.reporter   = ReportGenerator(self.config)

    # ── 공개 메서드 ──────────────────────────────────────────────────────

    def run(self, input_path: str, output_dir: str) -> PipelineResult:
        """파이프라인 전체 실행.

        Args:
            input_path : raw CSV 파일 경로 (인코딩 자동 감지)
            output_dir : 결과 저장 디렉토리 (없으면 자동 생성)

        Returns:
            PipelineResult

        Raises:
            FileNotFoundError : input_path가 존재하지 않을 때
            ValueError        : CSV 인코딩을 감지할 수 없을 때
        """
        warnings: List[str] = []

        print(f"[1/5] 로드       : {input_path}")
        raw = self._load(input_path, warnings)

        print(f"[2/5] 정제       : {len(raw)}행 → ", end="")
        cleaned = self.cleaner.clean(raw)
        dup = cleaned.attrs.get("duplicates_removed", 0)
        print(f"{len(cleaned)}행 (중복 {dup}건 제거)")
        if dup:
            warnings.append(f"중복 {dup}행 제거됨")

        print("[3/5] 분류       : 유형·감정·요약 컬럼 추가")
        classified = self.classifier.apply(cleaned)
        classified = self.priority.score_all(classified)

        print(f"[4/5] 우선순위   : TOP {self.config.top_n} 선정")
        top_n = self.priority.get_top_n(classified)

        print(f"[5/5] 저장       : {output_dir}/")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._save(classified, top_n, out)

        return PipelineResult(
            df=classified,
            agg=self.reporter.aggregate(classified),
            top_n=top_n,
            output_dir=str(out.resolve()),
            warnings=warnings,
        )

    # ── 내부 메서드 ──────────────────────────────────────────────────────

    def _load(self, path: str, warnings: List[str]) -> pd.DataFrame:
        for enc in self._ENCODINGS:
            try:
                return pd.read_csv(
                    path, encoding=enc, dtype=str,
                    keep_default_na=False, na_values=[""],
                )
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(
            f"CSV 인코딩 감지 실패: {path}\n지원 인코딩: {self._ENCODINGS}"
        )

    def _save(self, df: pd.DataFrame, top_n: pd.DataFrame, out: Path) -> None:
        cfg = self.config

        # ① cleaned_voc.csv
        df.to_csv(out / cfg.output_cleaned_csv, index=False, encoding="utf-8-sig")

        # ② report.json
        payload = self.reporter.to_json(df, top_n)
        (out / cfg.output_report_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ③ report_summary.md
        (out / cfg.output_report_md).write_text(
            self.reporter.to_markdown(df, top_n), encoding="utf-8"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  CLI 진입점
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voc_pipeline",
        description="VoC 자동 분석 파이프라인 — raw CSV를 넣으면 분류·리포트를 자동 생성합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행
  python voc_pipeline.py --input raw_voc.csv --output ./output

  # TOP-5, 연도 2025 기준
  python voc_pipeline.py --input raw_voc.csv --output ./output --top-n 5 --infer-year 2025

  # 외부 키워드 JSON 사용
  python voc_pipeline.py --input raw_voc.csv --output ./output --rules-json my_rules.json
        """,
    )
    p.add_argument("--input",       required=True,  help="입력 CSV 경로")
    p.add_argument("--output",      required=True,  help="출력 디렉토리 경로")
    p.add_argument("--top-n",       type=int, default=3, metavar="N",
                   help="즉시 대응 TOP-N 개수 (기본: 3)")
    p.add_argument("--infer-year",  default="2026", metavar="YYYY",
                   help="'N월M일' 형식 날짜의 기본 연도 (기본: 2026)")
    p.add_argument("--rules-json",  default=None,   metavar="PATH",
                   help="키워드 룰 JSON 경로 (없으면 내장 기본값 사용)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)

    rules = (
        ClassificationRules.from_json(args.rules_json)
        if args.rules_json
        else ClassificationRules()
    )
    config = PipelineConfig(
        rules=rules,
        top_n=args.top_n,
        infer_year=args.infer_year,
    )

    try:
        result = VOCPipeline(config).run(args.input, args.output)
    except FileNotFoundError as e:
        print(f"[오류] 파일 없음: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[오류] {e}", file=sys.stderr)
        return 1

    print(result.summary())

    cfg = config
    print("\n저장된 파일:")
    print(f"  📄  {args.output}/{cfg.output_cleaned_csv}")
    print(f"  📦  {args.output}/{cfg.output_report_json}")
    print(f"  📝  {args.output}/{cfg.output_report_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
