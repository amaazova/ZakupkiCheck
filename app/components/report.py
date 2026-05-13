"""PDF report generation via fpdf2 with Cyrillic-capable system font."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .schemas import FindingType, RiskReport

# Verdana на macOS, DejaVu в Docker. Arial Unicode исключён намеренно: его
# CJK-таблицы ломают fpdf2-расчёт горизонтального пространства.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    str(Path(__file__).parent.parent / "assets" / "DejaVuSans.ttf"),
]

_FLAG_LABELS_RU = {
    FindingType.BRAND_WITHOUT_EQUIVALENT: "Товарный знак без оговорки об эквиваленте",
    FindingType.INCOMPLETE_DESCRIPTION:   "Неполное описание объекта закупки",
    FindingType.RESTRICTIVE_REQUIREMENT:  "Ограничительные требования",
    FindingType.KTRU_MISMATCH:            "Несоответствие позиции КТРУ",
    FindingType.INCONSISTENT_DATES:       "Противоречивые сроки",
    FindingType.MISSING_ACCEPTANCE_TERMS: "Отсутствие критериев приёмки",
    FindingType.OTHER:                    "Иная проблема соответствия",
}

_VERDICT_RU = {
    "high":   "Высокий риск нарушений",
    "medium": "Средний риск",
    "low":    "Нарушений не обнаружено",
}

FAMILY = "App"


def flag_label(flag_type: FindingType | str) -> str:
    if isinstance(flag_type, FindingType):
        return _FLAG_LABELS_RU.get(flag_type, flag_type.value)
    try:
        return _FLAG_LABELS_RU[FindingType(flag_type)]
    except (KeyError, ValueError):
        return str(flag_type)


def _first_existing(paths: Iterable[str]) -> str | None:
    for p in paths:
        if Path(p).is_file():
            return p
    return None


def _new_pdf() -> tuple[FPDF, bool]:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    font_path = _first_existing(_FONT_CANDIDATES)
    unicode_ready = False
    if font_path:
        pdf.add_font(FAMILY, "", font_path)
        pdf.add_page()
        pdf.set_font(FAMILY, size=11)
        unicode_ready = True
    else:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
    return pdf, unicode_ready


def _safe(text: str, unicode_ready: bool) -> str:
    if unicode_ready:
        return text
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        return text.encode("latin-1", errors="replace").decode("latin-1")


def _set(pdf: FPDF, size: int, unicode_ready: bool) -> None:
    if unicode_ready:
        pdf.set_font(FAMILY, size=size)
    else:
        pdf.set_font("Helvetica", size=size)


def _line(pdf: FPDF, text: str, *, size: int, unicode_ready: bool, h: float = 5.5) -> None:
    _set(pdf, size, unicode_ready)
    pdf.multi_cell(
        0, h, _safe(text, unicode_ready),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )


def generate_pdf(report: RiskReport, filename: str) -> bytes:
    pdf, unicode_ready = _new_pdf()

    _line(pdf, "ZakupkiCheck — отчёт о проверке ТЗ", size=16, unicode_ready=unicode_ready, h=9)
    pdf.ln(2)
    _line(pdf, f"Файл: {filename}", size=11, unicode_ready=unicode_ready)
    _line(pdf, f"Уровень риска: {_VERDICT_RU.get(report.verdict, report.verdict)}",
          size=11, unicode_ready=unicode_ready)
    _line(pdf, f"Найдено замечаний: {len(report.findings)}",
          size=11, unicode_ready=unicode_ready)
    _line(pdf, f"Оценка вероятности нарушения: {report.risk_score:.0%}",
          size=11, unicode_ready=unicode_ready)
    pdf.ln(3)

    _line(pdf, "Выявленные замечания и рекомендации",
          size=14, unicode_ready=unicode_ready, h=7)
    pdf.ln(1)
    if not report.findings:
        _line(pdf, "Замечаний не обнаружено.",
              size=11, unicode_ready=unicode_ready)
    else:
        recs_by_idx = {r.finding_index: r for r in report.recommendations}
        for i, finding in enumerate(report.findings):
            rec = recs_by_idx.get(i)
            label = flag_label(finding.flag_type)
            _line(pdf,
                  f"{i + 1}. {label} — уверенность {finding.confidence:.0%}",
                  size=12, unicode_ready=unicode_ready, h=6)
            if finding.evidence_quote:
                _line(pdf, f"Фрагмент: «{finding.evidence_quote}»",
                      size=10, unicode_ready=unicode_ready, h=5)
            if rec is not None:
                _line(pdf, f"Рекомендация: {rec.fix}",
                      size=10, unicode_ready=unicode_ready, h=5)
                _line(pdf, f"Норма: {rec.law_ref}",
                      size=10, unicode_ready=unicode_ready, h=5)
                _line(pdf, rec.law_text,
                      size=9, unicode_ready=unicode_ready, h=4.5)
            pdf.ln(1)

    pdf.ln(2)
    _line(pdf, "Похожие решения ФАС",
          size=14, unicode_ready=unicode_ready, h=7)
    pdf.ln(1)
    if not report.similar_cases:
        _line(pdf, "Похожие решения не найдены.",
              size=11, unicode_ready=unicode_ready)
    else:
        for i, case in enumerate(report.similar_cases, 1):
            verdict = case.get("verdict") or ""
            verdict_ru = (
                "нарушение установлено" if verdict == "violation_established"
                else "нарушение не установлено" if verdict == "violation_not_established"
                else verdict or "—"
            )
            decision_id = case.get("decision_id") or "—"
            _line(pdf,
                  f"{i}. Решение ФАС №{decision_id} — {verdict_ru}",
                  size=12, unicode_ready=unicode_ready, h=6)
            reasoning = (case.get("key_reasoning") or "").strip()
            if reasoning:
                _line(pdf, reasoning[:1200],
                      size=10, unicode_ready=unicode_ready, h=5)
            pdf.ln(1)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
