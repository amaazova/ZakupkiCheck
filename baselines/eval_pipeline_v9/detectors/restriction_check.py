"""restriction_check — ограничительное описание / описание под единственного поставщика."""
from __future__ import annotations

from .. import config
from ..llm_client import call_llm
from ._common import parse_detector_json, truncate


SYSTEM_PROMPT = (
    "Ты эксперт по документации госзакупок (44-ФЗ). Проверь техническое задание на "
    "наличие ограничительного описания объекта закупки. Нарушение: субъективные или "
    "неизмеримые критерии оценки, описание, ограничивающее конкуренцию под единственного "
    "поставщика через совокупность характеристик, чрезмерно узкие диапазоны параметров "
    "(например: «длина 1234.5 мм», «погрешность ровно 0.07»), нестандартные единицы "
    "измерения без обоснования, требования, на которые отвечает только один производитель.\n"
    "Это нарушение ч. 1 ст. 33 и ст. 8 (принцип конкуренции) 44-ФЗ. Verdict='violation' "
    "только если описание действительно ограничивает конкуренцию. Стандартные подробные "
    "ТЗ (с обычными ГОСТ, общеотраслевыми характеристиками, обоснованными требованиями) "
    "не являются нарушением → verdict='not_established'.\n"
    "Ответь строго в JSON: "
    '{"verdict": "violation"|"not_established", "confidence": 0..1, '
    '"evidence_quote": "цитата ≤300 символов с конкретными ограничительными формулировками", '
    '"reasoning": "1-2 предложения почему это (не) ограничивает конкуренцию"}'
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
        detector="restriction_check",
        response_format="json",
    )
    parsed = parse_detector_json(result.text)
    parsed["_tokens_in"] = result.input_tokens
    parsed["_tokens_out"] = result.output_tokens
    parsed["_cost_usd"] = result.cost_usd
    return parsed
