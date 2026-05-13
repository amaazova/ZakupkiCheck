"""W0 data loaders: read FAS decision text + parsed TZ from disk.

All paths are anchored at config.PROJECT_ROOT — never hardcode here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT

RAW_FAS_DIR = PROJECT_ROOT / "data" / "raw_fas"
PARSED_CLEAN_DIR = PROJECT_ROOT / "data" / "parsed_clean"

MIN_FAS_TEXT_CHARS = 100  # below this we treat FAS text as "not present"
NOTICE_ID_WIDTH = 19      # canonical EIS notice_id width; pad short ints with leading zeros


def load_fas_text(decision_id: str) -> str | None:
    """Return the FAS decision text. Priority: full_text > decision_text. None on miss."""
    p = RAW_FAS_DIR / f"{decision_id}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("full_text", "decision_text"):
        text = (d.get(key) or "").strip()
        if len(text) >= MIN_FAS_TEXT_CHARS:
            return text
    return None


def _normalize_notice_id(nid: str | int) -> str:
    """Pad notice_id with leading zeros to the canonical EIS width (19)."""
    s = str(nid)
    if s.isdigit() and len(s) < NOTICE_ID_WIDTH:
        return s.zfill(NOTICE_ID_WIDTH)
    return s


def load_tz(notice_id: str | int) -> str | None:
    """Return parsed TZ markdown for a notice. None if file missing or unreadable."""
    nid = _normalize_notice_id(notice_id)
    p = PARSED_CLEAN_DIR / nid / "tz.md"
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _episode_to_decision_id(episode_id: str) -> str:
    """Episode IDs are `{decision_id}_{episode_idx}` — strip the trailing index."""
    return episode_id.rsplit("_", 1)[0]


def build_episode_data(
    eval_df: pd.DataFrame,
    *,
    include_fas: bool = True,
    log_skips: bool = True,
) -> list[dict[str, Any]]:
    """Materialize a list of episode dicts ready for run_extraction()."""
    records: list[dict[str, Any]] = []
    skipped_no_tz: list[str] = []
    skipped_no_fas = 0

    for row in eval_df.to_dict(orient="records"):
        episode_id = row["episode_id"]
        notice_id = _normalize_notice_id(row["notice_id"])
        decision_id = _episode_to_decision_id(episode_id)
        doc_text = load_tz(notice_id)
        if doc_text is None:
            skipped_no_tz.append(episode_id)
            continue
        record: dict[str, Any] = {
            "episode_id": episode_id,
            "notice_id": notice_id,
            "decision_id": decision_id,
            "family": row.get("cluster_id"),
            "verdict": row.get("fas_verdict"),
            "stratum": row.get("stratum"),
            "doc_text": doc_text,
        }
        if include_fas:
            fas_text = load_fas_text(decision_id)
            if fas_text is None:
                skipped_no_fas += 1
            record["fas_text"] = fas_text
        records.append(record)

    if log_skips:
        import sys
        if skipped_no_tz:
            print(
                f"[build_episode_data] skipped {len(skipped_no_tz)} episodes with missing TZ; "
                f"first 3: {skipped_no_tz[:3]}",
                file=sys.stderr,
            )
        if include_fas and skipped_no_fas:
            print(
                f"[build_episode_data] {skipped_no_fas} episodes have no FAS text "
                f"(field set to None on the record)",
                file=sys.stderr,
            )
    return records
