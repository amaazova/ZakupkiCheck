"""Tests for the rule engine + anti-double-flag invariant."""
from __future__ import annotations

from components.rule_engine import (
    RULES_BY_DOC_TYPE,
    evaluate,
    rule_brand_without_equivalent,
    rule_no_guarantee,
    rule_no_nmck,
    rule_no_penalty_clause,
    rule_restrictive_phrases,
)
from components.schemas import (
    BrandMention,
    CompletenessFacts,
    ContractSpecificFacts,
    DocType,
    ExtractedFacts,
    FindingType,
    MeasurementFacts,
    NoticeSpecificFacts,
    RestrictivePhrase,
    StandardsFacts,
)


def _facts(doc_type=DocType.TZ, **kwargs) -> ExtractedFacts:
    return ExtractedFacts(doc_type=doc_type, **kwargs)


def test_brand_with_equivalent_is_never_flagged():
    facts = _facts(brands=[
        BrandMention(name="ASUS B85M-G", has_equivalent_clause=True,
                     equivalent_text="или эквивалент",
                     quote="системная плата (ASUS B85M-G или эквивалент)"),
    ])
    assert rule_brand_without_equivalent(facts) == []


def test_brand_without_equivalent_is_flagged():
    facts = _facts(brands=[
        BrandMention(name="Microsoft Windows", has_equivalent_clause=False,
                     quote="наличие OEM лицензии Microsoft Windows ..."),
    ])
    out = rule_brand_without_equivalent(facts)
    assert len(out) == 1
    assert out[0].flag_type == FindingType.BRAND_WITHOUT_EQUIVALENT
    assert "Microsoft Windows" in out[0].description


# ВКР §4.6.1: «не хуже», «уровня …» — эталон, не закрытая марка.
def test_benchmark_phrasing_is_not_flagged():
    facts = _facts(brands=[
        BrandMention(
            name="HD Graphics 4600",
            has_equivalent_clause=False,
            quote="интегрированное графическое ядро не хуже HD Graphics 4600",
        ),
        BrandMention(
            name="Intel Core i5",
            has_equivalent_clause=False,
            quote="процессор уровня Intel Core i5",
        ),
        BrandMention(
            name="Microsoft Windows",
            has_equivalent_clause=False,
            quote="наличие OEM лицензии Microsoft Windows",
        ),
    ])
    out = rule_brand_without_equivalent(facts)
    assert len(out) == 1
    assert out[0].extra.get("brand") == "Microsoft Windows"


def test_benchmark_markers_individually():
    markers = ["не хуже", "не ниже", "не менее", "не слабее", "уровня", "класса", "категории"]
    for marker in markers:
        facts = _facts(brands=[
            BrandMention(
                name="HP ProDesk",
                has_equivalent_clause=False,
                quote=f"видеокарта {marker} HP ProDesk-G в составе",
            ),
        ])
        assert rule_brand_without_equivalent(facts) == [], (
            f"marker «{marker}» should suppress the brand finding"
        )


def test_restrictive_drops_brand_with_equivalent_overlap():
    facts = _facts(
        brands=[
            BrandMention(name="ASUS B85M-G", has_equivalent_clause=True,
                         quote="ASUS B85M-G или эквивалент"),
        ],
        restrictive_phrases=[
            RestrictivePhrase(phrase="ASUS B85M-G",
                              quote="системная плата ASUS B85M-G или эквивалент"),
        ],
    )
    assert rule_restrictive_phrases(facts) == []


def test_restrictive_keeps_genuine_phrase():
    facts = _facts(
        restrictive_phrases=[
            RestrictivePhrase(
                phrase="могут быть указаны более предпочтительные производители",
                quote="Для минимизации расходов... могут быть указаны более предпочтительные производители",
            ),
        ],
    )
    out = rule_restrictive_phrases(facts)
    assert len(out) == 1
    assert out[0].flag_type == FindingType.RESTRICTIVE_REQUIREMENT


def test_tz_pipeline_toljatti_scenario():
    facts = _facts(
        brands=[
            BrandMention(name="ASUS B85M-G", has_equivalent_clause=True,
                         quote="ASUS B85M-G или эквивалент"),
            BrandMention(name="Intel Core i3", has_equivalent_clause=True,
                         quote="Intel Core i3 или эквивалент"),
            BrandMention(name="Seagate Barracuda", has_equivalent_clause=True,
                         quote="Seagate Barracuda или эквивалент"),
            BrandMention(name="Microsoft Windows", has_equivalent_clause=False,
                         quote="OEM лицензии Microsoft Windows для обновления"),
        ],
        restrictive_phrases=[
            RestrictivePhrase(phrase="более предпочтительные производители",
                              quote="могут быть указаны более предпочтительные производители"),
        ],
        completeness=CompletenessFacts(),
        measurements=MeasurementFacts(has_units=True, has_ranges=True),
        standards=StandardsFacts(ktru_code="26.20.11.110"),
    )
    findings = evaluate(facts, doc_type=DocType.TZ)
    types = [f.flag_type for f in findings]
    assert FindingType.BRAND_WITHOUT_EQUIVALENT in types
    assert FindingType.RESTRICTIVE_REQUIREMENT in types
    assert len(findings) == 2


def test_tz_missing_acceptance_triggers_finding():
    facts = _facts(completeness=CompletenessFacts(has_acceptance_criteria=False))
    findings = evaluate(facts, doc_type=DocType.TZ)
    assert any(f.flag_type == FindingType.MISSING_ACCEPTANCE_TERMS for f in findings)


def test_tz_incomplete_description_lists_missing():
    facts = _facts(
        completeness=CompletenessFacts(has_technical_requirements=False),
        measurements=MeasurementFacts(has_units=False, missing_characteristics=["IP-класс"]),
    )
    findings = evaluate(facts, doc_type=DocType.TZ)
    incomplete = [f for f in findings if f.flag_type == FindingType.INCOMPLETE_DESCRIPTION]
    assert len(incomplete) == 1
    desc = incomplete[0].description
    assert "технические характеристики" in desc
    assert "единицы измерения" in desc
    assert "IP-класс" in desc


def test_tz_ktru_missing_flags():
    facts = _facts(standards=StandardsFacts(ktru_code=None))
    findings = evaluate(facts, doc_type=DocType.TZ)
    assert any(f.flag_type == FindingType.KTRU_MISMATCH for f in findings)


def test_contract_no_penalty_triggers_finding():
    facts = _facts(
        doc_type=DocType.CONTRACT,
        contract_specific=ContractSpecificFacts(has_penalty_clause=False),
    )
    out = rule_no_penalty_clause(facts)
    assert len(out) == 1
    assert "штрафные санкции" in out[0].description


def test_contract_no_guarantee_triggers_finding():
    facts = _facts(
        doc_type=DocType.CONTRACT,
        contract_specific=ContractSpecificFacts(has_guarantee_provisions=False),
    )
    out = rule_no_guarantee(facts)
    assert len(out) == 1


def test_contract_null_state_does_not_trigger():
    facts = _facts(
        doc_type=DocType.CONTRACT,
        contract_specific=ContractSpecificFacts(
            has_penalty_clause=None,
            has_guarantee_provisions=None,
        ),
    )
    assert rule_no_penalty_clause(facts) == []
    assert rule_no_guarantee(facts) == []


def test_notice_no_nmck_triggers_finding():
    facts = _facts(
        doc_type=DocType.NOTICE,
        notice_specific=NoticeSpecificFacts(has_nmck=False),
    )
    out = rule_no_nmck(facts)
    assert len(out) == 1
    assert "НМЦК" in out[0].description


def test_notice_rule_set_does_not_include_brand_rules():
    rules = RULES_BY_DOC_TYPE[DocType.NOTICE]
    assert rule_brand_without_equivalent not in rules


def test_evaluate_picks_rule_set_by_doc_type():
    facts = _facts(
        doc_type=DocType.CONTRACT,
        contract_specific=ContractSpecificFacts(has_penalty_clause=False),
    )
    findings = evaluate(facts)
    assert any("штрафные санкции" in f.description for f in findings)
