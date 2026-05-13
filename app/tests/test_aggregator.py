"""Tests for the aggregator: features, dedupe, verdict thresholds."""
from __future__ import annotations

from components.aggregator import (
    aggregate,
    dedupe_findings,
    derive_verdict,
    extract_ml_features,
)
from components.schemas import (
    BrandMention,
    CompletenessFacts,
    DocType,
    ExtractedFacts,
    ExtractionResult,
    Finding,
    FindingType,
    MeasurementFacts,
    RestrictivePhrase,
    StandardsFacts,
)


def _finding(flag_type: FindingType, conf: float, quote: str, detector: str = "rule_engine") -> Finding:
    return Finding(flag_type=flag_type, confidence=conf, evidence_quote=quote, detector=detector)


def _facts(**kwargs) -> ExtractedFacts:
    return ExtractedFacts(doc_type=DocType.TZ, **kwargs)


def test_dedupe_collapses_same_quote():
    a = _finding(FindingType.BRAND_WITHOUT_EQUIVALENT, 0.7, "Karcher K5")
    b = _finding(FindingType.BRAND_WITHOUT_EQUIVALENT, 0.9, "Karcher K5")
    deduped = dedupe_findings([a, b])
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.9


def test_dedupe_keeps_distinct_quotes():
    a = _finding(FindingType.BRAND_WITHOUT_EQUIVALENT, 0.8, "Karcher K5 high-pressure cleaner")
    b = _finding(FindingType.BRAND_WITHOUT_EQUIVALENT, 0.8, "Bosch GBH 2-26 rotary hammer")
    deduped = dedupe_findings([a, b])
    assert len(deduped) == 2


def test_feature_extractor_seventeen_keys_from_facts():
    facts = _facts(
        brands=[
            BrandMention(name="HP ProDesk 600", has_equivalent_clause=False, quote="..."),
            BrandMention(name="Bosch", has_equivalent_clause=True, quote="..."),
        ],
        measurements=MeasurementFacts(
            has_units=True, has_ranges=True,
            units_found=["мм"], missing_characteristics=["IP-класс"],
        ),
        restrictive_phrases=[RestrictivePhrase(phrase="только", quote="...только...")],
        standards=StandardsFacts(ktru_code="26.20.11.110", has_additional_beyond_ktru=True),
        completeness=CompletenessFacts(
            has_functional_requirements=True,
            has_technical_requirements=True,
            has_quality_requirements=False,
            has_acceptance_criteria=False,
        ),
    )
    findings = [_finding(FindingType.BRAND_WITHOUT_EQUIVALENT, 0.95, "...")]
    feats = extract_ml_features(facts, findings, text_len=10_000)
    assert set(feats.keys()) == {
        "has_brand", "brand_count", "has_equivalent", "brands_without_equiv",
        "has_units", "has_ranges", "missing_char_count", "restrictive_count",
        "ktru_mentioned", "beyond_ktru", "has_functional", "has_technical",
        "has_quality", "risk_flag_count", "max_confidence", "mean_confidence",
        "tz_char_count",
    }
    assert feats["has_brand"] == 1
    assert feats["brand_count"] == 2
    assert feats["has_equivalent"] == 1
    assert feats["brands_without_equiv"] == 1
    assert feats["restrictive_count"] == 1
    assert feats["ktru_mentioned"] == 1
    assert feats["beyond_ktru"] == 1
    assert feats["missing_char_count"] == 1
    assert feats["has_quality"] == 0
    assert feats["tz_char_count"] == 10_000


def test_verdict_thresholds():
    assert derive_verdict(0.1, []) == "low"
    assert derive_verdict(0.8, [_finding(FindingType.OTHER, 0.5, "x")]) == "high"
    assert derive_verdict(0.3, [_finding(FindingType.OTHER, 0.9, "x")]) == "high"
    assert derive_verdict(0.5, [_finding(FindingType.OTHER, 0.5, "x")]) == "medium"
    assert derive_verdict(0.2, [_finding(FindingType.OTHER, 0.3, "x")]) == "low"


def test_aggregate_end_to_end():
    facts = _facts(
        brands=[BrandMention(name="HP ProDesk 600", has_equivalent_clause=False, quote="...")],
    )
    extraction = ExtractionResult(facts=facts, cost_usd=0.001, model="test")
    findings = [_finding(FindingType.BRAND_WITHOUT_EQUIVALENT, 0.95, "HP ProDesk 600")]
    text = "Поставка системного блока марки HP ProDesk 600."
    report = aggregate(
        extraction, findings, text,
        doc_hash="abc123", doc_type=DocType.TZ, similar_cases=[],
    )
    assert report.doc_hash == "abc123"
    assert report.doc_type == DocType.TZ
    assert report.doc_chars == len(text)
    assert len(report.findings) == 1
    assert 0.0 <= report.risk_score <= 1.0
    assert report.verdict in {"high", "medium", "low"}
    assert report.cost_usd == 0.001
    assert report.facts is not None
