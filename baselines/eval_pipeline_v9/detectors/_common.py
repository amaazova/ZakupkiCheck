"""Shared helpers for detectors: JSON-of-verdict parsing, evidence truncation."""
from __future__ import annotations

import json
import re
from typing import Any


VALID_VERDICTS = {"violation", "not_established"}


def truncate(s: str, limit: int) -> str:
    if not s:
        return ""
    return s if len(s) <= limit else s[: limit - 1] + "…"


def coerce_verdict(raw: str) -> str:
    r = (raw or "").strip().lower()
    if "violation" in r and "not" not in r.replace("not_established", "").replace("not established", ""):
        # plain "violation"
        return "violation"
    if r in ("violation",):
        return "violation"
    if r in ("not_established", "not established", "compliant", "no_violation"):
        return "not_established"
    # default to not_established on unknowns — fail safe
    return "not_established"


def parse_detector_json(text: str) -> dict[str, Any]:
    """Extract a {verdict, confidence, evidence_quote, reasoning} dict from an LLM
    response. Tolerates fenced ```json blocks and trailing garbage. Always returns
    a well-formed dict; on parse failure returns not_established with low confidence.
    """
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # find the first top-level JSON object
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        candidate = m.group(0) if m else cleaned
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return {
            "verdict": "not_established",
            "confidence": 0.0,
            "evidence_quote": "",
            "reasoning": f"parse_error: could not decode JSON. raw={truncate(cleaned, 200)}",
        }
    verdict = coerce_verdict(str(data.get("verdict", "")))
    try:
        confidence = float(data.get("confidence", 0))
    except (ValueError, TypeError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    evidence = truncate(str(data.get("evidence_quote", "") or ""), 300)
    reasoning = truncate(str(data.get("reasoning", "") or ""), 600)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence_quote": evidence,
        "reasoning": reasoning,
    }
