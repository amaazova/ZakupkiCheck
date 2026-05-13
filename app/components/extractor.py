"""Pass-1 LLM extractor: one call per document, returns ExtractedFacts."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .llm_client import (
    DEFAULT_MODEL as _DEFAULT_MODEL,
    call_llm_with_retries,
    get_client,
    parse_llm_json,
)
from .logging_config import get_logger
from .rate_limiter import TokenBucketLimiter
from .schemas import DocType, ExtractedFacts, ExtractionResult

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_hints_candidates = [
    Path(__file__).parent.parent.parent / "taxonomy" / "taxonomy_hints.json",
    Path(__file__).parent.parent / "taxonomy_hints.json",
]
TAXONOMY_HINTS_PATH: Optional[Path] = next(
    (p for p in _hints_candidates if p.is_file()), None
)
DEFAULT_MODEL = _DEFAULT_MODEL

_USER_PROMPT_FILES = {
    DocType.TZ: "tz_extraction.md",
    DocType.CONTRACT: "contract_extraction.md",
    DocType.NOTICE: "notice_extraction.md",
    DocType.DOCUMENTATION: "documentation_extraction.md",
}

_GENERIC_HINT = (
    "указание товарных знаков без оговорки «или эквивалент»; "
    "ограничительные формулировки; неполное описание объекта закупки; "
    "отсутствие порядка приёмки и КТРУ."
)

_SCHEMA_TEMPLATE = """{
  "doc_type": "tz | contract | notice | documentation",
  "brands": [
    {"name": "...", "has_equivalent_clause": true_or_false,
     "equivalent_text": "..." | null, "quote": "..."}
  ],
  "measurements": {
    "has_units": true_or_false, "has_ranges": true_or_false,
    "units_found": ["..."], "missing_characteristics": ["..."]
  },
  "restrictive_phrases": [
    {"phrase": "...", "quote": "..."}
  ],
  "standards": {
    "gost_references": ["..."], "ktru_code": "..." | null,
    "has_additional_beyond_ktru": true_or_false
  },
  "completeness": {
    "has_functional_requirements": true_or_false,
    "has_technical_requirements": true_or_false,
    "has_quality_requirements": true_or_false,
    "has_acceptance_criteria": true_or_false,
    "has_warranty_terms": true_or_false,
    "has_delivery_terms": true_or_false
  },
  "contract_specific": {
    "has_penalty_clause": true_or_false_or_null,
    "has_guarantee_provisions": true_or_false_or_null,
    "has_acceptance_procedure": true_or_false_or_null,
    "payment_terms_defined": true_or_false_or_null
  },
  "notice_specific": {
    "has_nmck": true_or_false_or_null,
    "has_submission_deadline": true_or_false_or_null,
    "procurement_method_stated": true_or_false_or_null,
    "has_eligibility_criteria": true_or_false_or_null
  }
}"""

_RATE_LIMITER = TokenBucketLimiter(max_requests=20, window_sec=60.0)
_logger = get_logger("extractor")

_system_prompt_cache: Optional[str] = None
_user_prompt_cache: dict[DocType, str] = {}
_taxonomy_hints_cache: Optional[dict[str, str]] = None


def _load_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        _system_prompt_cache = (PROMPTS_DIR / "system_prompt.md").read_text(encoding="utf-8").strip()
    return _system_prompt_cache


def _load_user_prompt(doc_type: DocType) -> str:
    if doc_type not in _user_prompt_cache:
        path = PROMPTS_DIR / _USER_PROMPT_FILES[doc_type]
        _user_prompt_cache[doc_type] = path.read_text(encoding="utf-8")
    return _user_prompt_cache[doc_type]


def _load_taxonomy_hints() -> dict[str, str]:
    global _taxonomy_hints_cache
    if _taxonomy_hints_cache is None:
        if TAXONOMY_HINTS_PATH is not None and TAXONOMY_HINTS_PATH.is_file():
            _taxonomy_hints_cache = json.loads(
                TAXONOMY_HINTS_PATH.read_text(encoding="utf-8")
            )
        else:
            _taxonomy_hints_cache = {}
    return _taxonomy_hints_cache


def select_taxonomy_hint(doc_type: DocType, cluster_id: Optional[int] = None) -> str:
    hints = _load_taxonomy_hints()
    if cluster_id is not None and str(cluster_id) in hints:
        return hints[str(cluster_id)]
    return _GENERIC_HINT


def extract(
    text: str,
    *,
    doc_type: DocType = DocType.TZ,
    cluster_id: Optional[int] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ExtractionResult:
    if not text or len(text) < 50:
        raise ValueError("text must be at least 50 characters long")

    t0 = time.monotonic()
    system_prompt = _load_system_prompt()
    user_template = _load_user_prompt(doc_type)
    hint = select_taxonomy_hint(doc_type, cluster_id)
    user_prompt = user_template.format(
        doc_text=text,
        taxonomy_hint=hint,
        json_schema=_SCHEMA_TEMPLATE,
    )

    _RATE_LIMITER.wait()
    client = get_client()
    result = call_llm_with_retries(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        use_json_response_format=True,
    )

    if result.status != "ok":
        _logger.warning("extract_http_failure", status=result.status, doc_type=doc_type.value)
        return ExtractionResult(
            facts=ExtractedFacts(doc_type=doc_type),
            elapsed_sec=time.monotonic() - t0,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            error=f"llm_{result.status}: {result.error[:160]}",
            model=model,
        )

    parsed, parse_status = parse_llm_json(result.text)
    if parse_status != "ok":
        strict_user = (
            user_prompt
            + "\n\nIMPORTANT: respond with ONLY valid JSON. "
              "No markdown fences, no prose. Just the JSON object."
        )
        retry = call_llm_with_retries(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=strict_user,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_response_format=True,
        )
        result.cost_usd += retry.cost_usd
        result.input_tokens += retry.input_tokens
        result.output_tokens += retry.output_tokens
        if retry.status == "ok":
            parsed, parse_status = parse_llm_json(retry.text)

    if parse_status != "ok" or not isinstance(parsed, dict):
        _logger.warning("extract_parse_failed", doc_type=doc_type.value)
        return ExtractionResult(
            facts=ExtractedFacts(doc_type=doc_type),
            elapsed_sec=time.monotonic() - t0,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            error="parse_failed",
            model=model,
        )

    parsed.setdefault("doc_type", doc_type.value)
    try:
        facts = ExtractedFacts.model_validate(parsed)
    except Exception as exc:
        _logger.warning("extract_validation_failed", error=str(exc), doc_type=doc_type.value)
        return ExtractionResult(
            facts=ExtractedFacts(doc_type=doc_type),
            elapsed_sec=time.monotonic() - t0,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            error=f"schema_validation: {exc!s}",
            model=model,
        )

    return ExtractionResult(
        facts=facts,
        elapsed_sec=time.monotonic() - t0,
        cost_usd=result.cost_usd,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model=model,
    )
