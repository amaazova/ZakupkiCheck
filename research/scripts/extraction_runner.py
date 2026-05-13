"""W0 universal extraction runner."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from tqdm import tqdm

from . import config


# Global lock for JSONL appends (output + log) and shared stats updates when
# run_extraction is called with max_workers > 1. Cheap to acquire when single-
# threaded so we always take it.
_io_lock = threading.Lock()


# JSONL utilities

def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(path: str | Path, record: dict) -> None:
    """Thread-safe JSONL append. Holds _io_lock for the duration of the write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _io_lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_ids(path: str | Path, key: str = "episode_id") -> set[str]:
    """Read a JSONL and return the set of already-seen episode_ids."""
    seen: set[str] = set()
    for rec in read_jsonl(path):
        v = rec.get(key)
        if v is not None:
            seen.add(str(v))
    return seen


# OpenRouter client

_client_cache: OpenAI | None = None


def get_client() -> OpenAI:
    global _client_cache
    if _client_cache is None:
        config.assert_api_key()
        _client_cache = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            default_headers=config.OPENROUTER_DEFAULT_HEADERS,
        )
    return _client_cache


# JSON parsing

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def parse_llm_json(text: str) -> tuple[dict | list | None, str]:
    """Try to extract a JSON object/array from an LLM response."""
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


# Retry policy

def _retry_after_seconds(exc: Exception, default: float) -> float:
    """Extract Retry-After (seconds) from a RateLimitError if present."""
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


# Single LLM call

@dataclass
class CallResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""
    attempt: int = 0
    status: str = "ok"  # ok | parse_failed | http_4xx | http_429 | http_5xx | timeout | exhausted
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
    """Make a single chat-completions call. Returns CallResult; never raises for HTTP errors —
    encodes them into status/error fields."""
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

    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        return CallResult(status="http_429", error=str(e)[:200])
    except APITimeoutError as e:
        return CallResult(status="timeout", error=str(e)[:200])
    except APIConnectionError as e:
        return CallResult(status="http_5xx", error=str(e)[:200])
    except APIError as e:
        # 4xx (non-429) — non-retryable
        code = getattr(e, "status_code", None)
        if code and 500 <= int(code) < 600:
            return CallResult(status="http_5xx", error=str(e)[:200])
        return CallResult(status="http_4xx", error=str(e)[:200])

    elapsed = time.monotonic() - t0
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)

    # OpenRouter cost (preferred) → fallback to PRICING estimate
    cost = 0.0
    try:
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        cost = float(((raw.get("usage") or {}).get("cost")) or 0.0)
    except Exception:
        cost = 0.0
    if cost <= 0:
        cost = config.estimate_cost(model, in_tok, out_tok)

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
    """Wrap _do_one_call with retry logic for HTTP failures."""
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
        # http_4xx — don't retry
        return result
    if last is not None:
        last.status = "exhausted"
    return last or CallResult(status="exhausted", error="no attempts made")


# Main entry point

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RunStats:
    processed: int = 0
    skipped_existing: int = 0
    parse_failures: int = 0
    http_failures: int = 0
    retries: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    by_status: dict[str, int] = field(default_factory=dict)


def _process_one(
    ep: dict,
    *,
    client: OpenAI,
    system_prompt: str,
    user_prompt_template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    use_json_response_format: bool,
    max_parse_retries: int,
    extra_keys: tuple[str, ...],
    output_path: Path,
    log_path: Path,
) -> tuple[dict | None, dict]:
    """Run the LLM on one episode and append the record to output_path."""
    summary: dict[str, Any] = {
        "processed": 0, "parse_failures": 0, "retries": 0,
        "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
        "status": "unknown", "http_failure": False,
    }
    episode_id = ep.get("episode_id")
    if episode_id is None:
        summary["status"] = "missing_id"
        return None, summary

    try:
        user_prompt = user_prompt_template.format(**ep)
    except KeyError as e:
        summary["status"] = "template_missing_key"
        append_jsonl(output_path, {
            "episode_id": episode_id, "model": model, "status": "template_missing_key",
            "error": f"missing template key: {e}", "ts_utc": _utc_now(),
        })
        summary["processed"] = 1
        summary["http_failure"] = True
        return None, summary

    attempts_http = 0
    attempts_parse = 0
    parsed = None
    result = CallResult()
    last_user_prompt = user_prompt
    for parse_attempt in range(max_parse_retries + 1):
        def _log(r: CallResult, _pa: int = parse_attempt) -> None:
            nonlocal attempts_http
            attempts_http += 1
            append_jsonl(log_path, {
                "ts_utc": _utc_now(),
                "episode_id": episode_id,
                "model": model,
                "attempt_http": r.attempt,
                "attempt_parse": _pa + 1,
                "status": r.status,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "finish_reason": r.finish_reason,
                "error": r.error,
            })
        result = call_llm_with_retries(
            client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=last_user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_json_response_format=use_json_response_format,
            log_callback=_log,
        )
        if result.attempt > 1:
            summary["retries"] += result.attempt - 1
        if result.status != "ok":
            break  # HTTP/transport failure → don't try to parse
        parsed, parse_status = parse_llm_json(result.text)
        attempts_parse = parse_attempt + 1
        if parse_status == "ok":
            result.status = "ok"
            break
        summary["parse_failures"] += 1
        last_user_prompt = (
            user_prompt
            + "\n\nIMPORTANT: respond with ONLY valid JSON. No markdown fences, "
              "no prose before or after. Just the JSON object."
        )
        result.status = "parse_failed"

    record = {k: ep[k] for k in extra_keys if k in ep}
    record.update({
        "episode_id": episode_id,
        "model": model,
        "response_text": result.text,
        "parsed": parsed,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 6),
        "status": result.status,
        "error": result.error,
        "attempts_http": attempts_http,
        "attempts_parse": attempts_parse,
        "ts_utc": _utc_now(),
    })
    append_jsonl(output_path, record)

    summary["processed"] = 1
    summary["status"] = result.status
    summary["cost_usd"] = result.cost_usd
    summary["input_tokens"] = result.input_tokens
    summary["output_tokens"] = result.output_tokens
    summary["http_failure"] = result.status not in ("ok",)
    return record, summary


def run_extraction(
    episodes: Iterable[dict],
    *,
    system_prompt: str,
    user_prompt_template: str,
    output_path: str | Path,
    model: str = config.DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    batch_delay: float = 0.1,
    use_json_response_format: bool = True,
    max_parse_retries: int = 2,
    extra_keys: tuple[str, ...] = ("notice_id", "decision_id", "family", "verdict", "stratum"),
    max_workers: int = 1,
) -> RunStats:
    """Iterate episodes, call LLM for each, append parsed results to output_path JSONL."""
    output_path = Path(output_path)
    log_path = output_path.with_name(output_path.stem + "_log.jsonl")

    completed = load_completed_ids(output_path)
    stats = RunStats()
    client = get_client()

    episodes_list = [ep for ep in episodes]
    todo_list: list[dict] = []
    for ep in episodes_list:
        eid = ep.get("episode_id")
        if eid is None:
            stats.by_status["missing_id"] = stats.by_status.get("missing_id", 0) + 1
            continue
        if str(eid) in completed:
            stats.skipped_existing += 1
            continue
        todo_list.append(ep)

    desc = f"extract:{model.split('/')[-1]}"
    if max_workers > 1:
        desc += f"x{max_workers}"
    pbar = tqdm(total=len(todo_list), desc=desc, unit="ep")

    def _fold(summary: dict) -> None:
        with _io_lock:
            stats.processed += summary["processed"]
            stats.cost_usd += summary["cost_usd"]
            stats.input_tokens += summary["input_tokens"]
            stats.output_tokens += summary["output_tokens"]
            stats.parse_failures += summary["parse_failures"]
            stats.retries += summary["retries"]
            if summary["http_failure"]:
                stats.http_failures += 1
            stats.by_status[summary["status"]] = stats.by_status.get(summary["status"], 0) + 1
            pbar.set_postfix(cost=f"${stats.cost_usd:.3f}",
                             fail=stats.http_failures,
                             parse_fail=stats.parse_failures,
                             skip=stats.skipped_existing)

    process_kwargs = dict(
        client=client,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        use_json_response_format=use_json_response_format,
        max_parse_retries=max_parse_retries,
        extra_keys=extra_keys,
        output_path=output_path,
        log_path=log_path,
    )

    if max_workers <= 1:
        for ep in todo_list:
            _, summary = _process_one(ep, **process_kwargs)
            _fold(summary)
            pbar.update(1)
            if batch_delay > 0:
                time.sleep(batch_delay)
    else:
        # Bounded concurrency. Each worker holds its own slot; batch_delay
        # is applied in-thread (i.e. spaced per-thread, not globally).
        def _worker(ep: dict) -> dict:
            try:
                _, summary = _process_one(ep, **process_kwargs)
            except Exception as e:  # defensive — should not normally fire
                summary = {"processed": 1, "parse_failures": 0, "retries": 0,
                           "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
                           "status": "worker_exception", "http_failure": True}
                append_jsonl(output_path, {
                    "episode_id": ep.get("episode_id"), "model": model,
                    "status": "worker_exception", "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "ts_utc": _utc_now(),
                })
            if batch_delay > 0:
                time.sleep(batch_delay)
            return summary

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_worker, ep) for ep in todo_list]
            for fut in as_completed(futures):
                _fold(fut.result())
                pbar.update(1)

    pbar.close()
    print(
        f"\n=== run_extraction summary ===\n"
        f"  output:           {output_path}\n"
        f"  processed:        {stats.processed}\n"
        f"  skipped_existing: {stats.skipped_existing}\n"
        f"  http_failures:    {stats.http_failures}\n"
        f"  parse_failures:   {stats.parse_failures}\n"
        f"  retries:          {stats.retries}\n"
        f"  tokens:           in={stats.input_tokens}  out={stats.output_tokens}\n"
        f"  cost:             ${stats.cost_usd:.4f}\n"
        f"  workers:          {max_workers}\n"
        f"  by_status:        {stats.by_status}"
    )
    return stats
