"""W0 config: project paths, OpenRouter endpoint, model IDs, pricing."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path("/Users/aza/Downloads/zakupki/last")

# Load .env if python-dotenv is installed; otherwise expect the env var to be set.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Common HTTP headers for OpenRouter analytics ranking (optional but recommended).
OPENROUTER_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/aza-zakupkicheck/v10",
    "X-Title": "ZakupkiCheck v10 Wave 0",
}

# Model IDs verified against OpenRouter /v1/models catalog on 2026-05-12.
# - deepseek-v4-flash: present.
# - claude-sonnet-4.6: catalog uses dotted form (not "4-6").
# - qwen3.6-plus: 1M-token context, slot for the qwen ablation per v10 design.
MODELS = {
    "v4flash": "deepseek/deepseek-v4-flash",
    "sonnet":  "anthropic/claude-sonnet-4.6",
    "qwen":    "qwen/qwen3.6-plus",
}

DEFAULT_MODEL = MODELS["v4flash"]

# ($/M input tokens, $/M output tokens). Used as a fallback estimate when
# the response does not include usage.cost. Keep in sync with the OpenRouter
# catalog; treat as authoritative only if the response omits cost.
PRICING = {
    "deepseek/deepseek-v4-flash":  (0.14, 0.28),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "qwen/qwen3.6-plus":           (0.325, 1.95),
}


def assert_api_key() -> None:
    """Raise if OPENROUTER_API_KEY is missing — call at the start of any script that hits the API."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env at PROJECT_ROOT or export it."
        )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Fallback cost estimate when response.usage.cost is absent."""
    rate = PRICING.get(model)
    if not rate:
        return 0.0
    in_rate, out_rate = rate
    return input_tokens * in_rate / 1_000_000 + output_tokens * out_rate / 1_000_000
