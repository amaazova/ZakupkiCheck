"""Map FindingType → recommendation template (title, fix, law reference)."""
from __future__ import annotations

from .schemas import Finding, FindingType, Recommendation, RiskReport

_TEMPLATES: dict[FindingType, dict[str, str]] = {
    FindingType.BRAND_WITHOUT_EQUIVALENT: {
        "title": "Добавьте оговорку об эквиваленте",
        "fix": (
            'После указания товарного знака «{brand}» добавьте слова «или эквивалент» '
            'и сформулируйте параметры эквивалентности.'
        ),
        "law_ref": "п. 1 ч. 1 ст. 33 44-ФЗ",
        "law_text": (
            "В описание объекта закупки не должны включаться указания на товарные "
            "знаки, фирменные наименования, патенты, наименование места происхождения "
            "товара или наименование производителя, если они не сопровождаются "
            "словами «или эквивалент»."
        ),
    },
    FindingType.INCOMPLETE_DESCRIPTION: {
        "title": "Дополните описание объекта закупки",
        "fix": "Укажите {missing} для полного описания объекта закупки.",
        "law_ref": "п. 1 ч. 1 ст. 33 44-ФЗ",
        "law_text": (
            "В описании объекта закупки указываются функциональные, технические "
            "и качественные характеристики, эксплуатационные характеристики "
            "(при необходимости) объекта закупки."
        ),
    },
    FindingType.RESTRICTIVE_REQUIREMENT: {
        "title": "Устраните ограничительное требование",
        "fix": (
            "Формулировка «{phrase}» может ограничивать количество участников. "
            "Замените её на объективный измеримый критерий или удалите."
        ),
        "law_ref": "ч. 1 ст. 33, ст. 8 44-ФЗ",
        "law_text": (
            "Описание объекта закупки не должно включать требования или указания, "
            "которые влекут за собой ограничение количества участников закупки, "
            "за исключением случаев, прямо предусмотренных законом."
        ),
    },
    FindingType.KTRU_MISMATCH: {
        "title": "Проверьте соответствие позиции КТРУ",
        "fix": (
            "Характеристики объекта закупки должны соответствовать позиции "
            "каталога товаров, работ, услуг (КТРУ). {details}"
        ),
        "law_ref": "ч. 5–6 ст. 23, ст. 33 44-ФЗ",
        "law_text": (
            "Описание объекта закупки формируется с учётом информации, включённой "
            "в каталог товаров, работ, услуг для обеспечения государственных и "
            "муниципальных нужд."
        ),
    },
    FindingType.INCONSISTENT_DATES: {
        "title": "Согласуйте сроки в разных частях ТЗ",
        "fix": "Приведите к единому значению расходящиеся даты: {details}.",
        "law_ref": "ст. 42 44-ФЗ",
        "law_text": (
            "В извещении и документации о закупке указываются срок исполнения "
            "контракта и другие существенные условия, которые должны быть "
            "взаимно согласованы во всех частях документации."
        ),
    },
    FindingType.MISSING_ACCEPTANCE_TERMS: {
        "title": "Опишите порядок приёмки и экспертизы",
        "fix": (
            "Добавьте порядок приёмки результата, требования к экспертизе и "
            "критерии соответствия — без них заказчик не сможет обосновать отказ."
        ),
        "law_ref": "ст. 33, ст. 94 44-ФЗ",
        "law_text": (
            "При описании объекта закупки указываются требования к приёмке "
            "поставленных товаров, выполненных работ или оказанных услуг."
        ),
    },
    FindingType.OTHER: {
        "title": "Обратите внимание",
        "fix": "Рассмотрите замечание и приведите соответствующий пункт ТЗ в соответствие требованиям 44-ФЗ.",
        "law_ref": "44-ФЗ",
        "law_text": "Подробности см. в соответствующих статьях Федерального закона № 44-ФЗ.",
    },
}


def _safe_format(template: str, extra: dict[str, str], finding: Finding) -> str:
    defaults = {
        "brand": "(марка не определена)",
        "missing": (finding.description or "недостающие сведения").lstrip("Отсутствует:").strip(),
        "phrase": (finding.description or "указанная формулировка").strip(),
        "details": finding.description or "",
    }
    merged = {**defaults, **{k: str(v) for k, v in extra.items() if v}}
    try:
        return template.format(**merged)
    except KeyError:
        return template


def generate_recommendation(finding: Finding, index: int = -1) -> Recommendation:
    template = _TEMPLATES.get(finding.flag_type) or _TEMPLATES[FindingType.OTHER]
    extra = dict(finding.extra or {})
    return Recommendation(
        title=template["title"],
        fix=_safe_format(template["fix"], extra, finding),
        law_ref=template["law_ref"],
        law_text=template["law_text"],
        finding_index=index,
    )


def populate(report: RiskReport) -> RiskReport:
    recommendations = [
        generate_recommendation(f, index=i)
        for i, f in enumerate(report.findings)
    ]
    return report.model_copy(update={"recommendations": recommendations})
