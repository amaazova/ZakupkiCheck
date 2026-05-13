"""OpenRouter client + retry/JSON-parse helpers (self-contained, no project deps)."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/amaazova/ZakupkiCheck",
    "X-Title": "ZakupkiCheck",
}
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

_client_cache: OpenAI | None = None


def get_client() -> OpenAI:
    global _client_cache
    if _client_cache is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Configure it via env var or Streamlit secrets."
            )
        _client_cache = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_DEFAULT_HEADERS,
        )
    return _client_cache


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def parse_llm_json(text: str) -> tuple[dict | list | None, str]:
    if not text:
        return None, "parse_failed"
    cleaned = text.strip()
    candidate = cleaned
    fenced = _JSON_FENCE_RE.search(cleaned)
    if fenced:
        candidate = fenced.group(1)
    else:
        bare = _JSON_BARE_RE.search(cleaned)
        if bare:
            candidate = bare.group(1)
    try:
        return json.loads(candidate), "ok"
    except json.JSONDecodeError:
        return None, "parse_failed"


def _retry_after_seconds(exc: Exception, default: float) -> float:
    resp = getattr(exc, "response", None)
    if resp is not None:
        ra = None
        try:
            ra = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
        except Exception:
            ra = None
        if ra:
            try:
                return float(ra)
            except (TypeError, ValueError):
                pass
    return default


@dataclass
class CallResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""
    attempt: int = 0
    status: str = "ok"
    error: str = ""


def _do_one_call(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    use_json_response_format: bool,
) -> CallResult:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_json_response_format:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        return CallResult(status="http_429", error=str(e)[:200])
    except APITimeoutError as e:
        return CallResult(status="timeout", error=str(e)[:200])
    except APIConnectionError as e:
        return CallResult(status="http_5xx", error=str(e)[:200])
    except APIError as e:
        code = getattr(e, "status_code", None)
        if code and 500 <= int(code) < 600:
            return CallResult(status="http_5xx", error=str(e)[:200])
        return CallResult(status="http_4xx", error=str(e)[:200])

    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

    # OpenRouter возвращает фактическую стоимость в response.usage.cost.
    # Fallback = 0.0 (точные тарифы держим в research/scripts/config.py).
    cost = 0.0
    try:
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        cost = float(((raw.get("usage") or {}).get("cost")) or 0.0)
    except Exception:
        cost = 0.0

    return CallResult(
        text=text,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
        finish_reason=choice.finish_reason or "",
        status="ok",
    )


def call_llm_with_retries(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    use_json_response_format: bool,
    max_http_retries: int = 3,
    log_callback=None,
) -> CallResult:
    backoffs = [2.0, 4.0, 8.0]
    last: CallResult | None = None
    for attempt in range(max_http_retries):
        result = _do_one_call(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_response_format=use_json_response_format,
        )
        result.attempt = attempt + 1
        if log_callback:
            log_callback(result)
        if result.status == "ok":
            return result
        last = result
        if result.status == "http_429":
            wait = _retry_after_seconds(Exception(), 30.0)
            time.sleep(wait)
            continue
        if result.status in ("timeout", "http_5xx"):
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
            continue
        return result
    if last is not None:
        last.status = "exhausted"
    return last or CallResult(status="exhausted", error="no attempts made")
