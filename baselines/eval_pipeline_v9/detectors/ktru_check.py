"""ktru_check — дополнительные характеристики сверх позиции КТРУ без обоснования."""
from __future__ import annotations

import re

from .. import config
from ..llm_client import call_llm
from ._common import parse_detector_json, truncate


# ОКПД2: dd.dd[.dd[.ddd]]
OKPD_RE = re.compile(r"\b(\d{2}\.\d{2}(?:\.\d{2}(?:\.\d{3})?)?)\b")
# КТРУ ID format like 26.20.16.110-00000001
KTRU_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{2}\.\d{3}-\d{8})\b")


SYSTEM_PROMPT = (
    "Ты эксперт по документации госзакупок (44-ФЗ) и каталогу КТРУ. В ТЗ указан код ОКПД2 "
    "{code_block}. Проверь:\n"
    "1. Содержит ли описание дополнительные характеристики сверх стандартных для этой "
    "категории товаров/работ/услуг?\n"
    "2. Если дополнительные характеристики есть — приведено ли в ТЗ обоснование "
    "(например: ссылка на технологическую необходимость, ГОСТ, регламент, нормативный акт)?\n"
    "Нарушение (ч. 5–6 ст. 23, ст. 33 44-ФЗ): дополнительные характеристики сверх КТРУ есть, "
    "обоснования нет. Если доп. характеристик нет ИЛИ они есть и обоснованы → "
    "verdict='not_established'.\n"
    "Ответь строго в JSON: "
    '{"verdict": "violation"|"not_established", "confidence": 0..1, '
    '"evidence_quote": "цитата ≤300 символов — конкретные доп. характеристики без обоснования", '
    '"reasoning": "1-2 предложения"}'
)


def _find_codes(text: str) -> tuple[list[str], list[str]]:
    okpd = []
    ktru = []
    seen_o = set()
    seen_k = set()
    # KTRU first since it's a superset of OKPD pattern
    for m in KTRU_RE.finditer(text):
        s = m.group(1)
        if s not in seen_k:
            seen_k.add(s)
            ktru.append(s)
    for m in OKPD_RE.finditer(text):
        s = m.group(1)
        # skip if it's part of a longer KTRU code already captured
        if any(k.startswith(s) for k in ktru):
            continue
        if s not in seen_o:
            seen_o.add(s)
            okpd.append(s)
    return okpd, ktru


def detect(tz_text: str, cluster_name: str, *, episode_id: str) -> dict:
    okpd_codes, ktru_codes = _find_codes(tz_text)
    if not okpd_codes and not ktru_codes:
        return {
            "verdict": "not_established",
            "confidence": 0.7,
            "evidence_quote": "",
            "reasoning": "regex_no_okpd2_or_ktru_code_found_in_tz",
        }
    code_block_parts = []
    if ktru_codes:
        code_block_parts.append("КТРУ: " + ", ".join(ktru_codes[:5]))
    if okpd_codes:
        code_block_parts.append("ОКПД2: " + ", ".join(okpd_codes[:5]))
    code_block = "; ".join(code_block_parts)

    system_prompt = SYSTEM_PROMPT.replace("{code_block}", code_block)
    tz_for_prompt = truncate(tz_text, config.TZ_TEXT_CHAR_LIMIT)
    user_prompt = (
        f"Кластер ФАС (контекст подозрения): {cluster_name}\n\n"
        f"Коды, найденные в ТЗ: {code_block}\n\n"
        f"Текст ТЗ (укорочён до {len(tz_for_prompt)} символов):\n```\n{tz_for_prompt}\n```"
    )
    result = call_llm(
        system_prompt,
        user_prompt,
        episode_id=episode_id,
        detector="ktru_check",
        response_format="json",
    )
    parsed = parse_detector_json(result.text)
    parsed["_tokens_in"] = result.input_tokens
    parsed["_tokens_out"] = result.output_tokens
    parsed["_cost_usd"] = result.cost_usd
    parsed["_codes"] = {"okpd": okpd_codes[:5], "ktru": ktru_codes[:5]}
    return parsed
