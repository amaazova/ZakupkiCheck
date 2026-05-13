"""Pass-2 rule engine: ExtractedFacts → list[Finding], deterministic Python."""
from __future__ import annotations

from typing import Callable

from .schemas import DocType, ExtractedFacts, Finding, FindingType

RuleFn = Callable[[ExtractedFacts], list[Finding]]

# Маркеры benchmark-формулировок: «процессор не хуже Intel Core i5», «уровня HD
# Graphics 4600». ФАС такие фразы трактует как эталон производительности, а не
# закрытое требование к марке, поэтому соответствующий бренд не флагуется.
BENCHMARK_MARKERS = (
    "не хуже",
    "не ниже",
    "не менее",
    "не слабее",
    "уровня",
    "класса",
    "категории",
)


def _is_benchmark_phrasing(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in BENCHMARK_MARKERS)


def _f(
    flag: FindingType,
    confidence: float,
    *,
    quote: str = "",
    description: str = "",
    extra: dict | None = None,
) -> Finding:
    return Finding(
        flag_type=flag,
        confidence=confidence,
        evidence_quote=quote,
        description=description,
        detector="rule_engine",
        extra=extra or {},
    )


def _phrase_overlaps_brand_with_equiv(phrase_text: str, extracted: ExtractedFacts) -> bool:
    pt = (phrase_text or "").lower()
    for brand in extracted.brands:
        if not brand.has_equivalent_clause:
            continue
        if brand.name and brand.name.lower() in pt:
            return True
    return False


# ст. 33 п. 1 ч. 1 44-ФЗ — товарный знак без оговорки «или эквивалент».
# Инвариант: brand.has_equivalent_clause=True → finding не эмитится никем.
def rule_brand_without_equivalent(extracted: ExtractedFacts) -> list[Finding]:
    out: list[Finding] = []
    for brand in extracted.brands:
        if brand.has_equivalent_clause:
            continue
        if _is_benchmark_phrasing(brand.quote):
            continue
        out.append(_f(
            FindingType.BRAND_WITHOUT_EQUIVALENT,
            confidence=0.95,
            quote=brand.quote,
            description=f"Указан товарный знак «{brand.name}» без оговорки «или эквивалент».",
            extra={"brand": brand.name},
        ))
    return out


# ст. 33 п. 1 ч. 1 — ограничительные формулировки.
# Anti-double-flag: фразы, совпадающие с брендом, у которого has_equivalent_clause=True,
# пропускаются.
def rule_restrictive_phrases(extracted: ExtractedFacts) -> list[Finding]:
    out: list[Finding] = []
    for ph in extracted.restrictive_phrases:
        text = ph.phrase or ""
        if not text:
            continue
        if _phrase_overlaps_brand_with_equiv(text, extracted):
            continue
        out.append(_f(
            FindingType.RESTRICTIVE_REQUIREMENT,
            confidence=0.7,
            quote=ph.quote or text,
            description=f"Ограничительная формулировка: «{text}»",
            extra={"phrase": text},
        ))
    return out


# ст. 33, ст. 94 — отсутствие критериев приёмки.
def rule_missing_acceptance(extracted: ExtractedFacts) -> list[Finding]:
    if not extracted.completeness.has_acceptance_criteria:
        return [_f(
            FindingType.MISSING_ACCEPTANCE_TERMS,
            confidence=0.85,
            description=(
                "Не указан порядок приёмки результата, требования к экспертизе "
                "и критерии соответствия."
            ),
        )]
    return []


# ст. 33 п. 1 ч. 1 — неполное описание объекта закупки.
def rule_incomplete_description(extracted: ExtractedFacts) -> list[Finding]:
    missing: list[str] = []
    c = extracted.completeness
    m = extracted.measurements
    if not c.has_functional_requirements:
        missing.append("функциональные характеристики")
    if not c.has_technical_requirements:
        missing.append("технические характеристики")
    if not m.has_units:
        missing.append("единицы измерения")
    for ch in m.missing_characteristics:
        if ch and ch not in missing:
            missing.append(ch)
    if not missing:
        return []
    return [_f(
        FindingType.INCOMPLETE_DESCRIPTION,
        confidence=0.8,
        description="Не указаны: " + ", ".join(missing),
        extra={"missing": missing},
    )]


# ст. 33 п. 4 ч. 1 / ч. 5–6 ст. 23 — КТРУ не указан или избыточные характеристики.
def rule_ktru_mismatch(extracted: ExtractedFacts) -> list[Finding]:
    out: list[Finding] = []
    if not extracted.standards.ktru_code:
        out.append(_f(
            FindingType.KTRU_MISMATCH,
            confidence=0.5,
            description="Код КТРУ/ОКПД2 не указан в документе.",
        ))
    elif extracted.standards.has_additional_beyond_ktru:
        out.append(_f(
            FindingType.KTRU_MISMATCH,
            confidence=0.65,
            description=(
                "В описании указаны характеристики сверх позиции КТРУ "
                "без явного обоснования."
            ),
            extra={"ktru_code": extracted.standards.ktru_code},
        ))
    return out


# ст. 34 ч. 4–8 — отсутствие штрафных санкций.
def rule_no_penalty_clause(extracted: ExtractedFacts) -> list[Finding]:
    if extracted.contract_specific.has_penalty_clause is False:
        return [_f(
            FindingType.MISSING_ACCEPTANCE_TERMS,
            confidence=0.9,
            description=(
                "Не установлены штрафные санкции за ненадлежащее исполнение "
                "обязательств (ст. 34 ч. 4–8 44-ФЗ)."
            ),
        )]
    return []


# ст. 96 — отсутствие обеспечения исполнения контракта.
def rule_no_guarantee(extracted: ExtractedFacts) -> list[Finding]:
    if extracted.contract_specific.has_guarantee_provisions is False:
        return [_f(
            FindingType.MISSING_ACCEPTANCE_TERMS,
            confidence=0.85,
            description=(
                "Не установлены требования к обеспечению исполнения контракта "
                "(ст. 96 44-ФЗ)."
            ),
        )]
    return []


# ст. 94 — порядок приёмки результата контракта.
def rule_no_acceptance_procedure(extracted: ExtractedFacts) -> list[Finding]:
    if extracted.contract_specific.has_acceptance_procedure is False:
        return [_f(
            FindingType.MISSING_ACCEPTANCE_TERMS,
            confidence=0.8,
            description=(
                "Не описан порядок приёмки результатов контракта и процедура "
                "экспертизы (ст. 94 44-ФЗ)."
            ),
        )]
    return []


# ст. 42 п. 2 — не указана НМЦК.
def rule_no_nmck(extracted: ExtractedFacts) -> list[Finding]:
    if extracted.notice_specific.has_nmck is False:
        return [_f(
            FindingType.INCOMPLETE_DESCRIPTION,
            confidence=0.95,
            description="Не указана начальная (максимальная) цена контракта (НМЦК).",
        )]
    return []


# ст. 42 — не указан срок подачи заявок.
def rule_no_submission_deadline(extracted: ExtractedFacts) -> list[Finding]:
    if extracted.notice_specific.has_submission_deadline is False:
        return [_f(
            FindingType.INCOMPLETE_DESCRIPTION,
            confidence=0.9,
            description="Не указан срок подачи заявок участников закупки.",
        )]
    return []


# ст. 42 — не указан способ закупки.
def rule_no_procurement_method(extracted: ExtractedFacts) -> list[Finding]:
    if extracted.notice_specific.procurement_method_stated is False:
        return [_f(
            FindingType.INCOMPLETE_DESCRIPTION,
            confidence=0.85,
            description="Не указан способ закупки (аукцион, конкурс, запрос котировок).",
        )]
    return []


RULES_BY_DOC_TYPE: dict[DocType, list[RuleFn]] = {
    DocType.TZ: [
        rule_brand_without_equivalent,
        rule_restrictive_phrases,
        rule_missing_acceptance,
        rule_incomplete_description,
        rule_ktru_mismatch,
    ],
    DocType.CONTRACT: [
        rule_brand_without_equivalent,
        rule_restrictive_phrases,
        rule_no_penalty_clause,
        rule_no_guarantee,
        rule_no_acceptance_procedure,
    ],
    DocType.NOTICE: [
        rule_no_nmck,
        rule_no_submission_deadline,
        rule_no_procurement_method,
        rule_restrictive_phrases,
    ],
    DocType.DOCUMENTATION: [
        rule_brand_without_equivalent,
        rule_restrictive_phrases,
        rule_missing_acceptance,
        rule_incomplete_description,
    ],
}


def evaluate(extracted: ExtractedFacts, doc_type: DocType | None = None) -> list[Finding]:
    doc_type = doc_type or extracted.doc_type
    rules = RULES_BY_DOC_TYPE.get(doc_type) or RULES_BY_DOC_TYPE[DocType.TZ]
    out: list[Finding] = []
    for rule in rules:
        out.extend(rule(extracted))
    return out
