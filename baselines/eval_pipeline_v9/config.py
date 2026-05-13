"""Pipeline configuration — loads .env, exposes API/model/cost settings."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path("/Users/aza/Downloads/zakupki/last")
ENV_FILE = PROJECT_ROOT / ".env"
EVAL_DIR = PROJECT_ROOT / "workspace" / "eval"
LOGS_DIR = EVAL_DIR / "logs"
PREDICTIONS_PATH = EVAL_DIR / "predictions_b4.jsonl"
LLM_LOG_PATH = LOGS_DIR / "llm_calls.jsonl"


def _load_env() -> dict[str, str]:
    """Parse .env (simple KEY=VALUE)."""
    env: dict[str, str] = {}
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV = _load_env()


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or _ENV.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found in env or .env")
    return key


# OpenRouter endpoint — used with the standard openai SDK by overriding base_url
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-chat"
TEMPERATURE = 0.0
MAX_TOKENS = 800

# OpenRouter pricing for deepseek/deepseek-chat (DeepSeek V3) at freeze date 2026-05-12.
# Anchored to public list price; actual cost is taken from OpenRouter's `usage.cost`
# field on each response — these constants are only used as a fallback estimate.
COST_PER_1M_INPUT_USD = 0.27
COST_PER_1M_OUTPUT_USD = 1.10

# Rate limiting + retries
MAX_REQUESTS_PER_SECOND = 10
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SEC = 1.0

# How much of the tz.md to pass to the LLM. Long ТЗ truncation budget — picked
# to keep input under ~10k tokens with safe margin for prompts.
TZ_TEXT_CHAR_LIMIT = 30000

LOGS_DIR.mkdir(parents=True, exist_ok=True)
