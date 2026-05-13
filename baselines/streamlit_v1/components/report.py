"""PDF report generation via fpdf2.

The report includes:
  - Summary card (документ, риски, уровень)
  - Risk checklist (тип, цитата, confidence, норма)
  - Top-N FAS precedents (decision_id, verdict, key_reasoning)

Cyrillic support requires a TTF font; we probe several common system paths.
Arial Unicode is intentionally excluded — its CJK-heavy tables break fpdf2's
horizontal-space calculation. We use Verdana on macOS, DejaVu on Linux Docker.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

from fpdf import FPDF
from fpdf.enums import XPos, YPos

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
    "brand_without_equivalent": "Товарный знак без оговорки об эквиваленте",
    "incomplete_description":   "Неполное описание объекта закупки",
    "restrictive_requirement":  "Ограничительные требования",
    "ktru_mismatch":            "Несоответствие позиции КТРУ",
    "inconsistent_dates":       "Противоречивые сроки",
    "missing_acceptance_terms": "Отсутствие критериев приёмки",
    "other":                    "Иная проблема соответствия",
}

_FLAG_ARTICLES = {
    "brand_without_equivalent": "44-ФЗ, п. 1 ч. 1 ст. 33",
    "incomplete_description":   "44-ФЗ, ст. 33, ст. 42",
    "restrictive_requirement":  "44-ФЗ, ч. 1 ст. 33, ст. 8",
    "ktru_mismatch":            "44-ФЗ, ч. 5–6 ст. 23, ст. 33",
    "inconsistent_dates":       "44-ФЗ, ст. 42",
    "missing_acceptance_terms": "44-ФЗ, ст. 33, ст. 94",
    "other":                    "—",
}

FAMILY = "App"


def _first_existing(paths: Iterable[str]) -> str | None:
    for p in paths:
        if Path(p).is_file():
            return p
    return None


def _new_pdf() -> tuple[FPDF, bool]:
    """Create a PDF with Unicode font if available."""
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


def generate_pdf(extraction_result: dict, similar_cases: list[dict], filename: str) -> bytes:
    """Build a PDF as bytes."""
    parsed = (extraction_result or {}).get("parsed") or {}
    risk_flags = list(parsed.get("risk_flags") or [])

    from .extractor import risk_level
    level = risk_level(risk_flags)

    pdf, unicode_ready = _new_pdf()

    # Header
    _line(pdf, "ZakupkiCheck — отчёт о проверке ТЗ", size=16, unicode_ready=unicode_ready, h=9)
    pdf.ln(2)
    _line(pdf, f"Файл: {filename}", size=11, unicode_ready=unicode_ready)
    _line(pdf, "Тип документа: ТЗ", size=11, unicode_ready=unicode_ready)
    _line(pdf, f"Найдено рисков: {len(risk_flags)}", size=11, unicode_ready=unicode_ready)
    _line(pdf, f"Уровень риска: {level}", size=11, unicode_ready=unicode_ready)
    if extraction_result.get("model"):
        _line(pdf, f"Модель извлечения: {extraction_result['model']}",
              size=9, unicode_ready=unicode_ready, h=4.5)
    pdf.ln(3)

    # Risks
    _line(pdf, "Выявленные риски", size=14, unicode_ready=unicode_ready, h=7)
    pdf.ln(1)
    if not risk_flags:
        _line(pdf, "Рисков не обнаружено.", size=11, unicode_ready=unicode_ready)
    else:
        for i, flag in enumerate(risk_flags, 1):
            ftype = flag.get("flag_type") or "other"
            label = _FLAG_LABELS_RU.get(ftype, ftype)
            conf = float(flag.get("confidence") or 0.0)
            quote = (flag.get("evidence_quote") or "").strip()
            article = _FLAG_ARTICLES.get(ftype, "—")

            _line(pdf, f"{i}. {label} — уверенность {conf:.0%}",
                  size=12, unicode_ready=unicode_ready, h=6)
            if quote:
                _line(pdf, f"Цитата: «{quote}»",
                      size=10, unicode_ready=unicode_ready, h=5)
            _line(pdf, f"Норма: {article}", size=10, unicode_ready=unicode_ready, h=5)
            pdf.ln(1)

    pdf.ln(2)
    _line(pdf, "Прецеденты ФАС", size=14, unicode_ready=unicode_ready, h=7)
    pdf.ln(1)
    if not similar_cases:
        _line(pdf, "Похожие решения не найдены.", size=11, unicode_ready=unicode_ready)
    else:
        for i, case in enumerate(similar_cases, 1):
            verdict = case.get("verdict") or "—"
            sim = case.get("similarity_score")
            sim_s = f" (сходство {sim:.2f})" if isinstance(sim, (int, float)) else ""
            _line(pdf,
                  f"{i}. Решение {case.get('decision_id', '—')} — {verdict}{sim_s}",
                  size=12, unicode_ready=unicode_ready, h=6)
            reasoning = (case.get("key_reasoning") or "").strip()
            if reasoning:
                _line(pdf, reasoning[:1200], size=10, unicode_ready=unicode_ready, h=5)
            pdf.ln(1)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()
