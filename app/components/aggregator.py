"""Aggregate findings + ML risk score into RiskReport."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import joblib

from .schemas import (
    DocType,
    ExtractedFacts,
    ExtractionResult,
    Finding,
    FindingType,
    RiskReport,
)

MODEL_PATH = Path(__file__).parent.parent / "models" / "lr_model.joblib"

FEATURE_COLUMNS = [
    "has_brand", "brand_count", "has_equivalent", "brands_without_equiv",
    "has_units", "has_ranges", "missing_char_count", "restrictive_count",
    "ktru_mentioned", "beyond_ktru", "has_functional", "has_technical",
    "has_quality", "risk_flag_count", "max_confidence", "mean_confidence",
    "tz_char_count",
]


def _load_model() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def dedupe_findings(findings: Iterable[Finding]) -> list[Finding]:
    by_key: dict[tuple[str, str], Finding] = {}
    for f in findings:
        quote_key = re.sub(r"\s+", " ", f.evidence_quote or "").strip()[:60].lower()
        key = (f.flag_type.value, quote_key or f.description[:60].lower())
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = f
        elif f.confidence > existing.confidence:
            by_key[key] = f
    out = list(by_key.values())
    out.sort(key=lambda x: x.confidence, reverse=True)
    return out


def extract_ml_features(facts: ExtractedFacts, findings: list[Finding], text_len: int) -> dict[str, float]:
    brands_without_equiv = sum(1 for b in facts.brands if not b.has_equivalent_clause)
    brand_count = len(facts.brands)
    has_equivalent = 1 if any(b.has_equivalent_clause for b in facts.brands) else 0
    confs = [f.confidence for f in findings] or [0.0]
    return {
        "has_brand": 1 if brand_count else 0,
        "brand_count": brand_count,
        "has_equivalent": has_equivalent,
        "brands_without_equiv": brands_without_equiv,
        "has_units": 1 if facts.measurements.has_units else 0,
        "has_ranges": 1 if facts.measurements.has_ranges else 0,
        "missing_char_count": len(facts.measurements.missing_characteristics),
        "restrictive_count": len(facts.restrictive_phrases),
        "ktru_mentioned": 1 if facts.standards.ktru_code else 0,
        "beyond_ktru": 1 if facts.standards.has_additional_beyond_ktru else 0,
        "has_functional": 1 if facts.completeness.has_functional_requirements else 0,
        "has_technical": 1 if facts.completeness.has_technical_requirements else 0,
        "has_quality": 1 if facts.completeness.has_quality_requirements else 0,
        "risk_flag_count": len(findings),
        "max_confidence": max(confs),
        "mean_confidence": sum(confs) / len(confs),
        "tz_char_count": text_len,
    }


def compute_risk_score(features: dict[str, float]) -> float:
    payload = _load_model()
    if payload is None:
        n = features.get("risk_flag_count", 0)
        max_conf = features.get("max_confidence", 0.0)
        if n == 0:
            return 0.1
        return min(0.95, 0.3 + 0.15 * n + 0.3 * max_conf)
    model = payload["model"]
    feat_order = payload.get("feature_names", FEATURE_COLUMNS)
    row = [[float(features.get(name, 0.0)) for name in feat_order]]
    proba = model.predict_proba(row)[0][1]
    return float(proba)


# Один high-confidence finding поднимает verdict даже при низком score:
# LR обучен на широком корпусе и часто неуверен на явных нарушениях.
def derive_verdict(risk_score: float, findings: list[Finding]) -> str:
    if not findings:
        return "low"
    max_conf = max(f.confidence for f in findings)
    if risk_score >= 0.7 or max_conf >= 0.85 or len(findings) >= 4:
        return "high"
    if risk_score >= 0.4 or max_conf >= 0.55 or len(findings) >= 2:
        return "medium"
    return "low"


def aggregate(
    extraction: ExtractionResult,
    findings: list[Finding],
    text: str,
    *,
    doc_hash: str,
    doc_type: DocType,
    similar_cases: list[dict[str, Any]] | None = None,
) -> RiskReport:
    findings = dedupe_findings(findings)
    features = extract_ml_features(extraction.facts, findings, len(text))
    score = compute_risk_score(features)
    verdict = derive_verdict(score, findings)
    return RiskReport(
        doc_type=doc_type,
        findings=findings,
        recommendations=[],
        risk_score=score,
        verdict=verdict,
        similar_cases=similar_cases or [],
        doc_hash=doc_hash,
        doc_chars=len(text),
        facts=extraction.facts,
        extraction=extraction,
        cost_usd=round(extraction.cost_usd, 6),
    )
