"""Thin wrapper around the openai SDK pointed at OpenRouter."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI
from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

from . import config


_client_lock = threading.Lock()
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                api_key=config.get_api_key(),
                base_url=config.OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://github.com/aza-zakupkicheck/local-eval",
                    "X-Title": "ZakupkiCheck v9 eval",
                },
            )
        return _client


_rate_lock = threading.Lock()
_recent_calls: list[float] = []


def _rate_limit() -> None:
    """Token-bucket: at most MAX_REQUESTS_PER_SECOND in any rolling 1-second window."""
    while True:
        with _rate_lock:
            now = time.monotonic()
            # drop calls older than 1 sec
            cutoff = now - 1.0
            while _recent_calls and _recent_calls[0] < cutoff:
                _recent_calls.pop(0)
            if len(_recent_calls) < config.MAX_REQUESTS_PER_SECOND:
                _recent_calls.append(now)
                return
            sleep_for = _recent_calls[0] + 1.0 - now
        if sleep_for > 0:
            time.sleep(sleep_for)


_log_lock = threading.Lock()


def _log_call(record: dict[str, Any]) -> None:
    with _log_lock:
        with config.LLM_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Fallback estimate when OpenRouter doesn't include usage.cost."""
    return (
        input_tokens * config.COST_PER_1M_INPUT_USD / 1_000_000
        + output_tokens * config.COST_PER_1M_OUTPUT_USD / 1_000_000
    )


class LLMCallResult:
    __slots__ = ("text", "input_tokens", "output_tokens", "cost_usd", "model_used", "raw_finish_reason")

    def __init__(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        model_used: str,
        raw_finish_reason: str,
    ):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.model_used = model_used
        self.raw_finish_reason = raw_finish_reason


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    episode_id: str,
    detector: str,
    max_tokens: int | None = None,
    response_format: str | None = None,
) -> LLMCallResult:
    """Make one LLM call. Returns LLMCallResult or raises after exhausting retries."""
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs: dict[str, Any] = {
        "model": config.MODEL,
        "messages": messages,
        "temperature": config.TEMPERATURE,
        "max_tokens": max_tokens or config.MAX_TOKENS,
    }
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        _rate_limit()
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            last_exc = e
            wait = config.RETRY_BACKOFF_BASE_SEC * (2 ** attempt)
            _log_call({
                "ts": _utc_now(),
                "episode_id": episode_id,
                "detector": detector,
                "model": config.MODEL,
                "attempt": attempt + 1,
                "status": "retry",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "wait_sec": wait,
            })
            time.sleep(wait)
            continue
        except APIError as e:
            last_exc = e
            # 4xx — non-retryable
            _log_call({
                "ts": _utc_now(),
                "episode_id": episode_id,
                "detector": detector,
                "model": config.MODEL,
                "attempt": attempt + 1,
                "status": "api_error",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            raise

        elapsed = time.monotonic() - t0
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

        # Try to read OpenRouter's usage.cost first; fall back to estimate
        cost = 0.0
        try:
            raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
            cost = float(((raw.get("usage") or {}).get("cost")) or 0.0)
        except Exception:
            cost = 0.0
        if cost <= 0:
            cost = _estimate_cost(in_tok, out_tok)

        _log_call({
            "ts": _utc_now(),
            "episode_id": episode_id,
            "detector": detector,
            "model": config.MODEL,
            "attempt": attempt + 1,
            "status": "ok",
            "elapsed_sec": round(elapsed, 3),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round(cost, 6),
            "finish_reason": choice.finish_reason or "",
        })
        return LLMCallResult(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            model_used=getattr(resp, "model", config.MODEL),
            raw_finish_reason=choice.finish_reason or "",
        )

    assert last_exc is not None
    _log_call({
        "ts": _utc_now(),
        "episode_id": episode_id,
        "detector": detector,
        "model": config.MODEL,
        "status": "exhausted_retries",
        "error": f"{type(last_exc).__name__}: {str(last_exc)[:200]}",
    })
    raise last_exc
