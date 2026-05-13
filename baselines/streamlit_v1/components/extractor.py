"""LLM extraction wrapper for the Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/Users/aza/Downloads/zakupki/last")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Lazy imports of project modules — kept inside functions where possible so
# the Streamlit page can still render error messages if config is missing.
from workspace.scripts import config as project_config  # noqa: E402
from workspace.scripts.extraction_runner import (  # noqa: E402
    call_llm_with_retries,
    get_client,
    parse_llm_json,
)

PROMPTS_DIR = PROJECT_ROOT / "workspace" / "eval" / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "tz_l1_system_prompt.md"
USER_TEMPLATE_PATH = PROMPTS_DIR / "tz_l1_user_prompt_template.md"

DEFAULT_MODEL = project_config.MODELS["v4flash"]  # deepseek/deepseek-v4-flash


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _format_user_prompt(template: str, doc_text: str) -> str:
    # Template expects {extra_instruction} and {doc_text}
    return template.format(extra_instruction="", doc_text=doc_text)


def extract_compliance(
    text: str,
    doc_hash: str,
    cache,
    *,
    filename: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Extract compliance features + risk_flags from a single document."""
    if doc_hash:
        cached = cache.get(doc_hash)
        if cached is not None:
            cached = dict(cached)
            cached["cached"] = True
            return cached

    system_prompt = _load_prompt(SYSTEM_PROMPT_PATH)
    user_prompt = _format_user_prompt(_load_prompt(USER_TEMPLATE_PATH), text)

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

    parsed: dict | None = None
    status = result.status
    if status == "ok":
        parsed_obj, parse_status = parse_llm_json(result.text)
        if parse_status == "ok" and isinstance(parsed_obj, dict):
            parsed = parsed_obj
        else:
            # One reprompt for strict JSON
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
            if retry.status == "ok":
                parsed_obj, parse_status = parse_llm_json(retry.text)
                if parse_status == "ok" and isinstance(parsed_obj, dict):
                    parsed = parsed_obj
                else:
                    status = "parse_failed"
            else:
                status = retry.status
            result.input_tokens += retry.input_tokens
            result.output_tokens += retry.output_tokens
            result.cost_usd += retry.cost_usd

    payload = {
        "parsed": parsed,
        "status": status,
        "error": result.error,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 6),
        "model": model,
        "cached": False,
    }

    if doc_hash and parsed is not None:
        cache.put(doc_hash, payload, filename=filename)

    return payload


def risk_level(risk_flags: list[dict]) -> str:
    """Aggregate risk level based on count + confidence.

    Высокий — есть flag с confidence >= 0.8 или >= 3 рисков.
    Средний — есть flag с confidence >= 0.5 или 1–2 рисков.
    Низкий — иначе.
    """
    if not risk_flags:
        return "Низкий"
    n = len(risk_flags)
    max_conf = max((float(f.get("confidence") or 0.0) for f in risk_flags), default=0.0)
    if n >= 3 or max_conf >= 0.8:
        return "Высокий"
    if max_conf >= 0.5 or n >= 1:
        return "Средний"
    return "Низкий"
