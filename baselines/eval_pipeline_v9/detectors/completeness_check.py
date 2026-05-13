"""completeness_check — отсутствие конкретных характеристик / объёмов / ед.изм."""
from __future__ import annotations

from .. import config
from ..llm_client import call_llm
from ._common import parse_detector_json, truncate


SYSTEM_PROMPT = (
    "Ты эксперт по документации госзакупок (44-ФЗ). Проверь техническое задание на "
    "полноту описания объекта закупки. Нарушение: в ТЗ отсутствуют конкретные "
    "характеристики товара/работ/услуг, не указаны объёмы поставки, единицы измерения, "
    "или требования к качеству.\n"
    "Это нарушение ст. 33 44-ФЗ (надлежащее описание объекта закупки) и ст. 42 (содержание "
    "извещения). Verdict='violation' только если действительно есть пропуск обязательных "
    "сведений, а не если ТЗ просто длинное или сложное. Если описание полное (характеристики "
    "указаны, объёмы указаны, единицы измерения присутствуют) → verdict='not_established'.\n"
    "Ответь строго в JSON: "
    '{"verdict": "violation"|"not_established", "confidence": 0..1, '
    '"evidence_quote": "цитата из ТЗ ≤300 символов или \\"\\" если нарушения нет", '
    '"reasoning": "1-2 предложения о том, что именно пропущено или почему всё ок"}'
)


def detect(tz_text: str, cluster_name: str, *, episode_id: str) -> dict:
    tz_for_prompt = truncate(tz_text, config.TZ_TEXT_CHAR_LIMIT)
    user_prompt = (
        f"Кластер ФАС (контекст подозрения): {cluster_name}\n\n"
        f"Текст ТЗ (укорочён до {len(tz_for_prompt)} символов):\n```\n{tz_for_prompt}\n```"
    )
    result = call_llm(
        SYSTEM_PROMPT,
        user_prompt,
        episode_id=episode_id,
        detector="completeness_check",
        response_format="json",
    )
    parsed = parse_detector_json(result.text)
    parsed["_tokens_in"] = result.input_tokens
    parsed["_tokens_out"] = result.output_tokens
    parsed["_cost_usd"] = result.cost_usd
    return parsed
