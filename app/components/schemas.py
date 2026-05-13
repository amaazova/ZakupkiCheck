"""Pydantic schemas shared across extractor, rule engine, aggregator, and UI."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocType(str, Enum):
    TZ = "tz"
    CONTRACT = "contract"
    NOTICE = "notice"
    DOCUMENTATION = "documentation"


class FindingType(str, Enum):
    BRAND_WITHOUT_EQUIVALENT = "brand_without_equivalent"
    INCOMPLETE_DESCRIPTION = "incomplete_description"
    RESTRICTIVE_REQUIREMENT = "restrictive_requirement"
    KTRU_MISMATCH = "ktru_mismatch"
    INCONSISTENT_DATES = "inconsistent_dates"
    MISSING_ACCEPTANCE_TERMS = "missing_acceptance_terms"
    OTHER = "other"


class BrandMention(BaseModel):
    name: str
    has_equivalent_clause: bool = False
    equivalent_text: Optional[str] = None
    quote: str = ""


class MeasurementFacts(BaseModel):
    has_units: bool = False
    has_ranges: bool = False
    units_found: list[str] = Field(default_factory=list)
    missing_characteristics: list[str] = Field(default_factory=list)


class RestrictivePhrase(BaseModel):
    phrase: str
    quote: str = ""


class StandardsFacts(BaseModel):
    gost_references: list[str] = Field(default_factory=list)
    ktru_code: Optional[str] = None
    has_additional_beyond_ktru: bool = False


class CompletenessFacts(BaseModel):
    has_functional_requirements: bool = True
    has_technical_requirements: bool = True
    has_quality_requirements: bool = True
    has_acceptance_criteria: bool = True
    has_warranty_terms: bool = True
    has_delivery_terms: bool = True


class ContractSpecificFacts(BaseModel):
    has_penalty_clause: Optional[bool] = None
    has_guarantee_provisions: Optional[bool] = None
    has_acceptance_procedure: Optional[bool] = None
    payment_terms_defined: Optional[bool] = None


class NoticeSpecificFacts(BaseModel):
    has_nmck: Optional[bool] = None
    has_submission_deadline: Optional[bool] = None
    procurement_method_stated: Optional[bool] = None
    has_eligibility_criteria: Optional[bool] = None


class ExtractedFacts(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doc_type: DocType = DocType.TZ
    brands: list[BrandMention] = Field(default_factory=list)
    measurements: MeasurementFacts = Field(default_factory=MeasurementFacts)
    restrictive_phrases: list[RestrictivePhrase] = Field(default_factory=list)
    standards: StandardsFacts = Field(default_factory=StandardsFacts)
    completeness: CompletenessFacts = Field(default_factory=CompletenessFacts)
    contract_specific: ContractSpecificFacts = Field(default_factory=ContractSpecificFacts)
    notice_specific: NoticeSpecificFacts = Field(default_factory=NoticeSpecificFacts)


class Finding(BaseModel):
    flag_type: FindingType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str = ""
    description: str = ""
    detector: str = "rule_engine"
    extra: dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    title: str
    fix: str
    law_ref: str
    law_text: str
    finding_index: int = -1


class ExtractionResult(BaseModel):
    facts: ExtractedFacts
    elapsed_sec: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    model: str = ""


class RiskReport(BaseModel):
    doc_type: DocType = DocType.TZ
    findings: list[Finding] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0, default=0.0)
    verdict: str = "low"
    similar_cases: list[dict[str, Any]] = Field(default_factory=list)
    doc_hash: str
    doc_chars: int = 0
    facts: Optional[ExtractedFacts] = None
    extraction: Optional[ExtractionResult] = None
    cost_usd: float = 0.0
    cached: bool = False
