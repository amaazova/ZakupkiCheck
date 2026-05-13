"""B1 baseline: rule/regex check on tz.md.

Three rule families:
  R1 trademark_no_equiv  — товарный знак / модель без "или эквивалент / аналог"
                            в окне ±200 символов
  R2 incomplete_short_tz — короткий ТЗ (<2000 chars) без признаков спецификации
                            (единица измерения, количество, объём, ГОСТ, ТУ)
  R3 restrictive_keyword — "единственный" / "только" / "исключительно" в контексте
                            характеристик (рядом со словами характеристика, требование, поставка)

Verdict: violation if any rule triggered, else not_established.

Output: workspace/eval/predictions_b1.jsonl
Schema: {episode_id, notice_id, cluster_id, fas_verdict, pipeline_verdict, triggered_rules}
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LAST_ROOT, setup_logger  # noqa: E402

EVAL_CSV = LAST_ROOT / "workspace" / "eval" / "eval_dataset.csv"
OUTPUT = LAST_ROOT / "workspace" / "eval" / "predictions_b1.jsonl"

VERDICT_NORM = {
    "violation_established": "violation",
    "violation_not_established": "not_established",
}

# R1 — товарный знак / модель / ™ ®
TRADEMARK_QUOTED = re.compile(r"[«\"]([A-ZА-Я][A-Za-zА-Яа-я0-9 \-]{1,40})[»\"]")
TRADEMARK_MODEL = re.compile(r"\b[A-Z][A-Z0-9]{1,4}\d{2,}[A-Z0-9\-]*\b")
TRADEMARK_SYMBOL = re.compile(r"[™®©]")
EQUIV_PATTERN = re.compile(r"или\s+(эквивалент|аналог|аналогичн)", re.IGNORECASE)
EQUIV_WINDOW = 200

# R2 — incompleteness (very short TZ without spec markers)
SHORT_TZ_LIMIT = 2000
SPEC_MARKER = re.compile(
    r"единиц[аы]?\s+измерени[яй]|количеств|объ[её]м|ГОСТ\s*\d|ТУ\s*[0-9]|спецификаци",
    re.IGNORECASE,
)

# R3 — restrictive keywords next to characteristic/requirement/supply
RESTRICTIVE_KW = re.compile(r"\b(единственн[а-я]+|только|исключительно)\b", re.IGNORECASE)
CHAR_KW = re.compile(
    r"характеристик|требовани[ея]|поставк|производител|марк[аи]|модел[иь]",
    re.IGNORECASE,
)
RESTRICTIVE_WINDOW = 80


def rule_trademark(text: str) -> bool:
    """Trademark-like token without 'или эквивалент' within ±200 chars."""
    candidates = []
    for m in TRADEMARK_QUOTED.finditer(text):
        # Skip if quoted text is too generic (e.g. just a Russian word)
        inner = m.group(1)
        if not re.search(r"[A-Za-z0-9]", inner):
            continue
        candidates.append(m.span())
    for m in TRADEMARK_MODEL.finditer(text):
        candidates.append(m.span())
    for m in TRADEMARK_SYMBOL.finditer(text):
        candidates.append(m.span())
    for start, end in candidates:
        lo = max(0, start - EQUIV_WINDOW)
        hi = min(len(text), end + EQUIV_WINDOW)
        if not EQUIV_PATTERN.search(text[lo:hi]):
            return True
    return False


def rule_incomplete(text: str) -> bool:
    if len(text) >= SHORT_TZ_LIMIT:
        return False
    return not SPEC_MARKER.search(text)


def rule_restrictive(text: str) -> bool:
    for m in RESTRICTIVE_KW.finditer(text):
        lo = max(0, m.start() - RESTRICTIVE_WINDOW)
        hi = min(len(text), m.end() + RESTRICTIVE_WINDOW)
        if CHAR_KW.search(text[lo:hi]):
            return True
    return False


def evaluate(text: str) -> list[str]:
    triggered: list[str] = []
    if rule_trademark(text):
        triggered.append("R1_trademark_no_equiv")
    if rule_incomplete(text):
        triggered.append("R2_incomplete_short_tz")
    if rule_restrictive(text):
        triggered.append("R3_restrictive_keyword")
    return triggered


def main() -> int:
    log = setup_logger("eval_b1_regex")
    rows = list(csv.DictReader(EVAL_CSV.open(encoding="utf-8")))
    log.info("loaded %d episodes", len(rows))

    with OUTPUT.open("w", encoding="utf-8") as f:
        missing = 0
        for r in rows:
            tz_path = LAST_ROOT / r["tz_path"]
            if tz_path.is_file():
                text = tz_path.read_text(encoding="utf-8", errors="replace")
            else:
                text = ""
                missing += 1
            triggered = evaluate(text)
            pred = "violation" if triggered else "not_established"
            gold = VERDICT_NORM.get(r["fas_verdict"], r["fas_verdict"])
            f.write(json.dumps({
                "episode_id": r["episode_id"],
                "notice_id": r["notice_id"],
                "cluster_id": r["cluster_id"],
                "fas_verdict": gold,
                "pipeline_verdict": pred,
                "triggered_rules": triggered,
            }, ensure_ascii=False) + "\n")
    log.info("wrote %s; missing tz files: %d", OUTPUT, missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
