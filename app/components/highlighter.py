"""Wrap finding evidence_quote in <mark> with word-bounded fallback."""
from __future__ import annotations

import html
import re

from .schemas import Finding

_WHITESPACE_RE = re.compile(r"\s+")


def _color_for(confidence: float) -> str:
    if confidence >= 0.8:
        return "#ffcccc"
    if confidence >= 0.5:
        return "#fff3cd"
    return "#cce5ff"


def _normalise(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s or "").strip()


def _build_pattern(words: list[str]) -> re.Pattern[str] | None:
    if len(words) < 1:
        return None
    parts = [re.escape(w) for w in words if w]
    if not parts:
        return None
    pattern = r"\s+".join(parts)
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


# Fallback: full quote → ~half → 5 words → 3 words. Никогда не режем по середине слова.
def _find_match(text: str, quote: str) -> re.Match[str] | None:
    norm = _normalise(quote)
    if len(norm) < 8:
        return None
    words = norm.split()
    if not words:
        return None

    candidate_lengths = []
    full = len(words)
    candidate_lengths.append(full)
    half = max(full // 2, 5)
    if half < full:
        candidate_lengths.append(half)
    for n in (5, 3):
        if n < (candidate_lengths[-1] if candidate_lengths else full):
            candidate_lengths.append(n)
    seen: set[int] = set()
    candidate_lengths = [n for n in candidate_lengths if not (n in seen or seen.add(n))]

    for n in candidate_lengths:
        n = max(1, min(n, len(words)))
        pattern = _build_pattern(words[:n])
        if pattern is None:
            continue
        match = pattern.search(text)
        if match:
            return match
    return None


def highlight_text(text: str, findings: list[Finding]) -> str:
    if not text:
        return ""

    ordered = sorted(
        findings,
        key=lambda f: (-(f.confidence or 0.0), -len(f.evidence_quote or "")),
    )

    spans: list[tuple[int, int, Finding]] = []
    occupied: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        for s, e in occupied:
            if start < e and s < end:
                return True
        return False

    for f in ordered:
        match = _find_match(text, f.evidence_quote)
        if match is None:
            continue
        if _overlaps(match.start(), match.end()):
            continue
        occupied.append((match.start(), match.end()))
        spans.append((match.start(), match.end(), f))

    spans.sort(key=lambda x: x[0])

    out: list[str] = []
    cursor = 0
    for start, end, f in spans:
        out.append(html.escape(text[cursor:start]))
        color = _color_for(f.confidence)
        title = html.escape(f"{f.flag_type.value} · {int(f.confidence * 100)}%")
        body = html.escape(text[start:end])
        out.append(
            f'<mark style="background-color:{color};padding:2px 4px;'
            f'border-radius:3px;" title="{title}">{body}</mark>'
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out)
