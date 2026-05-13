"""Tests for the text highlighter: full quote then word-bounded fallback."""
from __future__ import annotations

from components.highlighter import highlight_text
from components.schemas import Finding, FindingType


def _finding(quote: str, conf: float = 0.8, flag: FindingType = FindingType.BRAND_WITHOUT_EQUIVALENT) -> Finding:
    return Finding(flag_type=flag, confidence=conf, evidence_quote=quote)


def test_highlight_full_quote_match():
    text = "Поставка системного блока марки HP ProDesk 600 без оговорки об эквиваленте."
    findings = [_finding("HP ProDesk 600")]
    html = highlight_text(text, findings)
    assert "<mark" in html
    assert "HP ProDesk 600" in html


def test_highlight_handles_whitespace_variation():
    text = "системного блока марки HP  ProDesk\n600 без оговорки."
    findings = [_finding("HP ProDesk 600")]
    html = highlight_text(text, findings)
    assert "<mark" in html


def test_highlight_uses_full_quote_then_word_prefix_fallback():
    text = "В документе встречается выражение «более предпочтительные производители» в разделе 3."
    long_quote = "Для минимизации расходов Заказчика могут быть указаны более предпочтительные производители оборудования согласно п. 3.1"
    findings = [_finding(long_quote, conf=0.7, flag=FindingType.RESTRICTIVE_REQUIREMENT)]
    html = highlight_text(text, findings)
    assert "<mark" not in html


def test_highlight_word_boundary_short_quote_match():
    text = "Указание товарного знака в разделе 2."
    findings = [_finding("товарного знака")]
    html = highlight_text(text, findings)
    assert "<mark" in html
    assert ">товарного знака<" in html


def test_highlight_does_not_overlap_marks():
    text = "Поставка HP ProDesk 600. Поставка HP ProDesk 600 еще раз."
    f1 = _finding("HP ProDesk 600", conf=0.9)
    f2 = _finding("HP ProDesk 600", conf=0.7)
    html = highlight_text(text, [f1, f2])
    assert html.count("<mark") <= 2
    assert "<mark><mark" not in html


def test_highlight_skips_when_quote_too_short():
    text = "Краткий текст без бренда."
    findings = [_finding("HP")]
    html = highlight_text(text, findings)
    assert "<mark" not in html


def test_highlight_colours_by_confidence():
    text = "Поставка системного блока марки HP ProDesk 600."
    high = highlight_text(text, [_finding("HP ProDesk 600", conf=0.9)])
    medium = highlight_text(text, [_finding("HP ProDesk 600", conf=0.6)])
    low = highlight_text(text, [_finding("HP ProDesk 600", conf=0.3)])
    assert "#ffcccc" in high
    assert "#fff3cd" in medium
    assert "#cce5ff" in low


def test_highlight_escapes_html_in_source():
    text = "Указано <script>alert(1)</script> HP ProDesk 600 в разделе."
    findings = [_finding("HP ProDesk 600")]
    html = highlight_text(text, findings)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_highlight_higher_confidence_wins_on_overlap():
    text = "Указан бренд HP ProDesk 600 в разделе."
    f_low = Finding(
        flag_type=FindingType.RESTRICTIVE_REQUIREMENT,
        confidence=0.4,
        evidence_quote="HP ProDesk 600",
    )
    f_high = Finding(
        flag_type=FindingType.BRAND_WITHOUT_EQUIVALENT,
        confidence=0.95,
        evidence_quote="HP ProDesk 600",
    )
    html = highlight_text(text, [f_low, f_high])
    assert "#ffcccc" in html
    assert "#cce5ff" not in html
