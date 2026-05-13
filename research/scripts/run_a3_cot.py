"""W4 ablation A3 — TZ extraction with chain-of-thought user prompt."""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import PROJECT_ROOT, MODELS
from .data_loaders import build_episode_data
from .extraction_runner import (
    CallResult,
    append_jsonl,
    call_llm_with_retries,
    get_client,
    load_completed_ids,
    parse_llm_json,
)


EVAL_DIR = PROJECT_ROOT / "workspace" / "eval"
PROMPTS_DIR = EVAL_DIR / "prompts"
TZ_CHAR_LIMIT = 150_000

CHUNKING_INSTRUCTION = (
    "Документ очень длинный. Сфокусируйся на описании объекта закупки, "
    "технических характеристиках и требованиях к товару/работе/услуге. "
    "Пропусти повторяющиеся таблицы спецификаций — достаточно одной репрезентативной."
)
THIN_INSTRUCTION = (
    "Документ короткий (1–3 тыс. символов). Не пытайся выдумать нарушения из ничего — "
    "если ТЗ выглядит корректным, верни пустой массив risk_flags."
)


def load_prompts() -> tuple[str, str]:
    system = (PROMPTS_DIR / "tz_l1_system_prompt.md").read_text(encoding="utf-8")
    user_tmpl = (PROMPTS_DIR / "tz_a3_cot_user_prompt_template.md").read_text(encoding="utf-8")
    return system, user_tmpl


def prepare_episodes(eval_df: pd.DataFrame) -> list[dict]:
    episodes = build_episode_data(eval_df, include_fas=False)
    tier_lookup = dict(zip(eval_df["episode_id"], eval_df["quality_tier"]))
    for ep in episodes:
        tier = tier_lookup.get(ep["episode_id"], "normal")
        if tier == "chunking_risk":
            ep["extra_instruction"] = CHUNKING_INSTRUCTION
        elif tier == "thin":
            ep["extra_instruction"] = THIN_INSTRUCTION
        else:
            ep["extra_instruction"] = ""
        if len(ep["doc_text"]) > TZ_CHAR_LIMIT:
            ep["doc_text"] = ep["doc_text"][:TZ_CHAR_LIMIT] + "\n\n…[ТЗ обрезан до лимита контекста]…"
        ep["quality_tier"] = tier
    return episodes


OUTPUT_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()


class SharedStats:
    __slots__ = ("processed", "http_failures", "parse_failures", "retries",
                 "cost_usd", "input_tokens", "output_tokens", "by_status")

    def __init__(self) -> None:
        self.processed = 0
        self.http_failures = 0
        self.parse_failures = 0
        self.retries = 0
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.by_status: dict[str, int] = {}

    def merge(self, **kw) -> None:
        with STATS_LOCK:
            for k, v in kw.items():
                if k == "by_status":
                    self.by_status[v] = self.by_status.get(v, 0) + 1
                else:
                    setattr(self, k, getattr(self, k) + v)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


EXTRA_KEYS = ("notice_id", "decision_id", "family", "verdict", "stratum", "quality_tier")


def process_episode(
    ep: dict, *, client, system_prompt: str, user_prompt_template: str,
    model: str, output_path: Path, log_path: Path,
    max_tokens: int, max_parse_retries: int, stats: SharedStats,
) -> dict[str, Any]:
    episode_id = ep["episode_id"]
    try:
        user_prompt = user_prompt_template.format(**ep)
    except KeyError as e:
        record = {"episode_id": episode_id, "model": model, "status": "template_missing_key",
                  "error": f"missing template key: {e}", "ts_utc": _utc_now()}
        with OUTPUT_LOCK:
            append_jsonl(output_path, record)
        stats.merge(by_status="template_missing_key", processed=1)
        return record

    attempts_http = 0
    attempts_parse = 0
    parsed = None
    last_user_prompt = user_prompt
    result = CallResult()

    for parse_attempt in range(max_parse_retries + 1):
        def _log(r: CallResult, _ep=episode_id, _pa=parse_attempt) -> None:
            nonlocal attempts_http
            attempts_http += 1
            entry = {
                "ts_utc": _utc_now(),
                "episode_id": _ep, "model": model,
                "attempt_http": r.attempt, "attempt_parse": _pa + 1,
                "status": r.status,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "finish_reason": r.finish_reason, "error": r.error,
            }
            with LOG_LOCK:
                append_jsonl(log_path, entry)

        result = call_llm_with_retries(
            client, model=model, system_prompt=system_prompt,
            user_prompt=last_user_prompt, temperature=0.0,
            max_tokens=max_tokens, use_json_response_format=True,
            log_callback=_log,
        )
        if result.attempt > 1:
            stats.merge(retries=result.attempt - 1)
        if result.status != "ok":
            break
        parsed, parse_status = parse_llm_json(result.text)
        attempts_parse = parse_attempt + 1
        if parse_status == "ok":
            break
        stats.merge(parse_failures=1)
        last_user_prompt = (
            user_prompt
            + "\n\nIMPORTANT: respond with ONLY valid JSON. No markdown fences, "
              "no prose before or after. Just the JSON object."
        )
        result.status = "parse_failed"

    record = {k: ep[k] for k in EXTRA_KEYS if k in ep}
    record.update({
        "episode_id": episode_id, "model": model,
        "response_text": result.text, "parsed": parsed,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 6),
        "status": result.status, "error": result.error,
        "attempts_http": attempts_http, "attempts_parse": attempts_parse,
        "ts_utc": _utc_now(),
    })
    with OUTPUT_LOCK:
        append_jsonl(output_path, record)

    stats.merge(processed=1, cost_usd=result.cost_usd,
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                by_status=result.status)
    if result.status != "ok":
        stats.merge(http_failures=1)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="run 8 random episodes to tz_features_A3_cot_smoke.jsonl")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    eval_df = pd.read_csv(EVAL_DIR / "eval_dataset_v10.csv", dtype={"notice_id": str})
    system, user_tmpl = load_prompts()

    if args.smoke:
        df = eval_df.sample(8, random_state=43)
        output_path = EVAL_DIR / "tz_features_A3_cot_smoke.jsonl"
        output_path.unlink(missing_ok=True)
        Path(str(output_path).replace(".jsonl", "_log.jsonl")).unlink(missing_ok=True)
    else:
        df = eval_df.head(args.limit) if args.limit > 0 else eval_df
        output_path = EVAL_DIR / "tz_features_A3_cot.jsonl"

    log_path = output_path.with_name(output_path.stem + "_log.jsonl")
    completed = load_completed_ids(output_path)
    print(f"[A3-cot] workers={args.workers}  output={output_path.name}  "
          f"already_completed={len(completed)}")

    episodes = prepare_episodes(df)
    pending = [ep for ep in episodes if ep["episode_id"] not in completed]
    print(f"[A3-cot] pending: {len(pending)}/{len(episodes)} episodes")
    if not pending:
        print("[A3-cot] nothing to do.")
        return

    client = get_client()
    stats = SharedStats()
    model = MODELS["v4flash"]
    t0 = time.monotonic()
    pbar = tqdm(total=len(pending), desc=f"A3-cot[{args.workers}w]", unit="ep")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_episode, ep,
                client=client, system_prompt=system, user_prompt_template=user_tmpl,
                model=model, output_path=output_path, log_path=log_path,
                max_tokens=args.max_tokens, max_parse_retries=2, stats=stats,
            ): ep["episode_id"]
            for ep in pending
        }
        for fut in as_completed(futures):
            eid = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"\n[A3-cot] worker error for {eid}: {type(e).__name__}: {e}")
            pbar.update(1)
            pbar.set_postfix(cost=f"${stats.cost_usd:.3f}",
                             fail=stats.http_failures, parse=stats.parse_failures)

    pbar.close()
    dt = time.monotonic() - t0
    print(
        f"\n=== run_a3_cot summary ===\n"
        f"  output:           {output_path}\n"
        f"  processed:        {stats.processed}\n"
        f"  http_failures:    {stats.http_failures}\n"
        f"  parse_failures:   {stats.parse_failures}\n"
        f"  retries:          {stats.retries}\n"
        f"  tokens:           in={stats.input_tokens}  out={stats.output_tokens}\n"
        f"  cost:             ${stats.cost_usd:.4f}\n"
        f"  elapsed:          {dt:.0f}s = {dt/60:.1f} min ({dt/3600:.2f}h)\n"
        f"  by_status:        {stats.by_status}"
    )


if __name__ == "__main__":
    main()
