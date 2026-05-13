"""Aggregator — runs all 4 detectors, combines into a pipeline_verdict."""
from __future__ import annotations

from typing import Any

from .detectors import brand_check, completeness_check, ktru_check, restriction_check


DETECTORS = {
    "brand_check": brand_check.detect,
    "completeness_check": completeness_check.detect,
    "restriction_check": restriction_check.detect,
    "ktru_check": ktru_check.detect,
}


def aggregate_verdict(detector_results: dict[str, dict]) -> str:
    """OR rule: pipeline says 'violation' if ANY detector says 'violation'."""
    for r in detector_results.values():
        if r.get("verdict") == "violation":
            return "violation"
    return "not_established"


def run(tz_text: str, cluster_name: str, *, episode_id: str) -> dict[str, Any]:
    detectors_out: dict[str, dict] = {}
    total_in = 0
    total_out = 0
    total_cost = 0.0
    for name, fn in DETECTORS.items():
        try:
            res = fn(tz_text, cluster_name, episode_id=episode_id)
        except Exception as e:
            res = {
                "verdict": "not_established",
                "confidence": 0.0,
                "evidence_quote": "",
                "reasoning": f"detector_error: {type(e).__name__}: {str(e)[:200]}",
            }
        total_in += int(res.pop("_tokens_in", 0) or 0)
        total_out += int(res.pop("_tokens_out", 0) or 0)
        total_cost += float(res.pop("_cost_usd", 0.0) or 0.0)
        # keep _codes if ktru_check produced one — informational
        codes = res.pop("_codes", None)
        if codes:
            res["okpd_codes"] = codes.get("okpd", [])
            res["ktru_codes"] = codes.get("ktru", [])
        detectors_out[name] = res

    pipeline_verdict = aggregate_verdict(detectors_out)
    return {
        "pipeline_verdict": pipeline_verdict,
        "detectors": detectors_out,
        "tokens_used": total_in + total_out,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_usd": round(total_cost, 6),
    }
