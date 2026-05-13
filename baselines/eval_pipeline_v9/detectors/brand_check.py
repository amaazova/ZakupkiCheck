"""brand_check — товарный знак / конкретная модель без «или эквивалент»."""
from __future__ import annotations

import re

from .. import config
from ..llm_client import call_llm
from ._common import parse_detector_json, truncate


# Capital-letter model strings like "HP M404", "EPSON L132", "Samsung X-100"
MODEL_RE = re.compile(r"\b([A-ZА-Я][A-ZА-Я0-9\-]{1,}(?:[\s\-]+\d+[A-ZА-Я0-9\-]*)?)\b")
# Latin or mixed acronym + digits combo
LAT_MODEL_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,}[\s\-]?\d+[A-Z0-9\-]*)\b")
# Quoted brand names
QUOTED_RE = re.compile(r"[«„\"]([^»\"„]{2,40})[»\"„]")
# Trademark sigils
SIGIL_RE = re.compile(r"[™®©]")

EQUIVALENT_RE = re.compile(
    r"или\s+эквивалент|или\s+аналог|эквивалент(?:а|ы|ность)?|аналог(?:а|и|ичн)",
    re.IGNORECASE,
)

# Common false positives — section headers, abbreviations
STOP_TOKENS = {
    "ОКПД", "ОКПД2", "ГОСТ", "СНиП", "СП", "КТРУ", "ТЗ", "НМЦК", "ФЗ", "РФ",
    "НДС", "ЕИС", "ЕСМ", "ОКЕИ", "ОКВЭД", "ОГРН", "ИНН", "КПП",
    "УТВЕРЖДАЮ", "СОГЛАСОВАНО", "ПРИЛОЖЕНИЕ", "РАЗДЕЛ", "ГЛАВА",
    "СТАТЬЯ", "ПУНКТ", "ЧАСТЬ", "ПОДПУНКТ", "АБЗАЦ",
}


def _find_candidates(text: str) -> list[tuple[str, int]]:
    """Find brand/model candidates. Returns list of (matched_string, position)."""
    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    for m in LAT_MODEL_RE.finditer(text):
        s = m.group(1).strip()
        if s in seen or s in STOP_TOKENS:
            continue
        seen.add(s)
        found.append((s, m.start()))
    for m in QUOTED_RE.finditer(text):
        s = m.group(1).strip()
        if len(s) < 3 or s in seen:
            continue
        # too generic single word
        if " " not in s and s.lower() in {"оригинал", "копия", "новый", "не бывший в употреблении"}:
            continue
        seen.add(s)
        found.append((s, m.start()))
    for m in SIGIL_RE.finditer(text):
        # capture surrounding token
        start = max(0, m.start() - 30)
        chunk = text[start: m.start() + 1].strip().split()[-1] if text[start: m.start() + 1].strip() else "[™]"
        if chunk and chunk not in seen:
            seen.add(chunk)
            found.append((chunk, m.start()))
    return found


def _context_has_equivalent(text: str, pos: int, window: int = 200) -> bool:
    chunk = text[max(0, pos - window): pos + window]
    return bool(EQUIVALENT_RE.search(chunk))


def _has_global_equivalent_clause(text: str) -> bool:
    """Some ТЗ open with a blanket «характеристики указаны с возможностью поставки
    эквивалентов» clause; if present, the document is not in violation."""
    head = text[:3000]
    return bool(EQUIVALENT_RE.search(head)) and ("указ" in head.lower() or "возможн" in head.lower())


SYSTEM_PROMPT = (
    "Ты эксперт по документации госзакупок (44-ФЗ). Тебе показан фрагмент технического "
    "задания и список подозрительных фрагментов, которые регулярка выделила как возможные "
    "товарные знаки или конкретные модели. Реши: указан ли в ТЗ товарный знак / "
    "конкретная модель БЕЗ слов «или эквивалент» (нарушение п. 1 ч. 1 ст. 33 44-ФЗ)?\n"
    "Если хотя бы один из подозрительных фрагментов — это действительно бренд/модель и "
    "рядом нет оговорки об эквивалентности → verdict='violation'. Если все фрагменты — "
    "это аббревиатуры, нормативные ссылки (ГОСТ, ОКПД, КТРУ), названия учреждений или "
    "юридических конструкций, либо оговорка «или эквивалент» присутствует в ТЗ "
    "глобально → verdict='not_established'.\n"
    "Ответь строго в JSON: "
    '{"verdict": "violation"|"not_established", "confidence": 0..1, '
    '"evidence_quote": "цитата ≤300 символов", "reasoning": "1-2 предложения"}'
)


def detect(tz_text: str, cluster_name: str, *, episode_id: str) -> dict:
    candidates = _find_candidates(tz_text)
    # Filter to candidates without local "или эквивалент"
    unguarded = [
        (s, pos) for (s, pos) in candidates
        if not _context_has_equivalent(tz_text, pos)
    ]
    if not unguarded:
        return {
            "verdict": "not_established",
            "confidence": 0.85,
            "evidence_quote": "",
            "reasoning": (
                f"regex_no_unguarded_candidates: total_candidates={len(candidates)}, "
                f"all_have_local_equivalent_clause"
            ),
        }
    if _has_global_equivalent_clause(tz_text):
        # global clause may cover everything — still ask LLM but signal it
        pass

    # Build a compact prompt with sample fragments and surrounding context
    sample_fragments: list[str] = []
    for s, pos in unguarded[:8]:
        ctx_start = max(0, pos - 80)
        ctx_end = min(len(tz_text), pos + 80)
        snippet = tz_text[ctx_start:ctx_end].replace("\n", " ")
        sample_fragments.append(f"«{s}» (контекст: …{snippet}…)")

    tz_for_prompt = truncate(tz_text, config.TZ_TEXT_CHAR_LIMIT)
    user_prompt = (
        f"Кластер ФАС: {cluster_name}\n\n"
        f"Подозрительные фрагменты ({len(unguarded)} найдено, показано до 8):\n"
        + "\n".join(f"- {f}" for f in sample_fragments)
        + f"\n\nТекст ТЗ (укорочён до {len(tz_for_prompt)} символов):\n```\n{tz_for_prompt}\n```"
    )

    result = call_llm(
        SYSTEM_PROMPT,
        user_prompt,
        episode_id=episode_id,
        detector="brand_check",
        response_format="json",
    )
    parsed = parse_detector_json(result.text)
    parsed["_tokens_in"] = result.input_tokens
    parsed["_tokens_out"] = result.output_tokens
    parsed["_cost_usd"] = result.cost_usd
    return parsed
