"""Memorization probe: stratified 50-episode sample with shifted entities.

For each sampled episode:
  - load tz.md
  - replace 19+ digit IDs (notice_id-like) with random same-length digits
  - replace dates (dd.mm.yyyy / dd.mm.yy / "dd <месяц> yyyy") with shifts of
    ±30..90 days
  - replace 10-12 digit INN-like with random
  - replace organization names: ОПФ-prefix + «...» → ОПФ + «<synthetic>»

Outputs:
  - workspace/eval/shifted_tz/{episode_id}_shifted.md
  - workspace/eval/memorization_probe_episodes.csv
"""
from __future__ import annotations

import csv
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LAST_ROOT, setup_logger  # noqa: E402

EVAL_CSV = LAST_ROOT / "workspace" / "eval" / "eval_dataset.csv"
SHIFTED_DIR = LAST_ROOT / "workspace" / "eval" / "shifted_tz"
PROBE_CSV = LAST_ROOT / "workspace" / "eval" / "memorization_probe_episodes.csv"

SEED = 42
SAMPLE_SIZE_PER_CLASS = 25

MONTHS_GEN = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
MONTH_NUM_TO_GEN = {v: k for k, v in MONTHS_GEN.items()}

ORG_OPF_PATTERN = re.compile(
    r"\b(ООО|ОАО|ЗАО|ПАО|АО|ФГУП|МУП|ГБУ|МБУ|ФГБУ|МКУ|ГБОУ|МБОУ|ИП|ФКУ|ФГБОУ|МАУ|ГАУ|МАОУ|МКОУ|МКП)\s*[«\"]([^»\"\n]{1,120})[»\"]",
    re.IGNORECASE,
)
ID_LONG_PATTERN = re.compile(r"\b\d{19,}\b")
INN_PATTERN = re.compile(r"\b\d{10}\b|\b\d{12}\b")
DATE_DOT_PATTERN = re.compile(r"\b(\d{1,2})[.](\d{1,2})[.](\d{4})\b")
DATE_DOT_SHORT_PATTERN = re.compile(r"\b(\d{1,2})[.](\d{1,2})[.](\d{2})\b")
DATE_TEXT_PATTERN = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(MONTHS_GEN.keys()) + r")\s+(\d{4})\b",
    re.IGNORECASE,
)

SYNTH_NAMES = ["Альфа", "Бета", "Гамма", "Дельта", "Сигма", "Омега", "Эпсилон", "Лямбда"]
SYNTH_SUFFIXES = ["центр", "сервис", "групп", "плюс", "проф", "техно", "система", "ресурс", "холдинг"]


def random_digits(rng: random.Random, n: int) -> str:
    # Avoid leading zero — most real IDs start with non-zero.
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


def shift_date(rng: random.Random, y: int, m: int, d: int) -> tuple[int, int, int]:
    try:
        base = datetime(y, m, d)
    except ValueError:
        return y, m, d
    sign = rng.choice([-1, 1])
    delta_days = sign * rng.randint(30, 90)
    new = base + timedelta(days=delta_days)
    return new.year, new.month, new.day


def synth_org_name(rng: random.Random) -> str:
    base = rng.choice(SYNTH_NAMES)
    suffix = rng.choice(SYNTH_SUFFIXES)
    return f"{base}-{suffix}"


def shift_text(text: str, rng: random.Random) -> tuple[str, dict[str, int]]:
    counts = {"ids": 0, "inn": 0, "dates": 0, "orgs": 0}

    # 1) Long IDs first (greedier than INN regex)
    def _sub_id(m: re.Match) -> str:
        counts["ids"] += 1
        return random_digits(rng, len(m.group(0)))
    text = ID_LONG_PATTERN.sub(_sub_id, text)

    # 2) INN (10 / 12 digits — exact)
    def _sub_inn(m: re.Match) -> str:
        counts["inn"] += 1
        return random_digits(rng, len(m.group(0)))
    text = INN_PATTERN.sub(_sub_inn, text)

    # 3) Dates dd.mm.yyyy
    def _sub_date_full(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ny, nm, nd = shift_date(rng, y, mo, d)
        counts["dates"] += 1
        return f"{nd:02d}.{nm:02d}.{ny:04d}"
    text = DATE_DOT_PATTERN.sub(_sub_date_full, text)

    # 4) Dates dd.mm.yy (assume 20yy)
    def _sub_date_short(m: re.Match) -> str:
        d, mo, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        ny, nm, nd = shift_date(rng, 2000 + yy, mo, d)
        counts["dates"] += 1
        return f"{nd:02d}.{nm:02d}.{ny % 100:02d}"
    text = DATE_DOT_SHORT_PATTERN.sub(_sub_date_short, text)

    # 5) Dates "dd <месяц> yyyy"
    def _sub_date_text(m: re.Match) -> str:
        d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        mo = MONTHS_GEN[mon]
        ny, nm, nd = shift_date(rng, y, mo, d)
        counts["dates"] += 1
        return f"{nd} {MONTH_NUM_TO_GEN[nm]} {ny}"
    text = DATE_TEXT_PATTERN.sub(_sub_date_text, text)

    # 6) Organization names — ОПФ «...»
    def _sub_org(m: re.Match) -> str:
        counts["orgs"] += 1
        opf = m.group(1)
        return f'{opf} «{synth_org_name(rng)}»'
    text = ORG_OPF_PATTERN.sub(_sub_org, text)

    return text, counts


def main() -> int:
    log = setup_logger("eval_memorization_probe")
    rng = random.Random(SEED)
    rows = list(csv.DictReader(EVAL_CSV.open(encoding="utf-8")))

    by_verdict: dict[str, list[dict]] = {"violation_established": [], "violation_not_established": []}
    for r in rows:
        if r["fas_verdict"] in by_verdict:
            by_verdict[r["fas_verdict"]].append(r)

    sampled: list[dict] = []
    for v, pool in by_verdict.items():
        rng.shuffle(pool)
        sampled.extend(pool[:SAMPLE_SIZE_PER_CLASS])
    rng.shuffle(sampled)
    log.info(
        "sampled %d episodes (violation=%d, not_established=%d)",
        len(sampled),
        sum(1 for s in sampled if s["fas_verdict"] == "violation_established"),
        sum(1 for s in sampled if s["fas_verdict"] == "violation_not_established"),
    )

    SHIFTED_DIR.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []

    total_counts = {"ids": 0, "inn": 0, "dates": 0, "orgs": 0}
    for r in sampled:
        tz_path = LAST_ROOT / r["tz_path"]
        if not tz_path.is_file():
            log.warning("missing tz_path: %s", tz_path)
            continue
        text = tz_path.read_text(encoding="utf-8", errors="replace")
        # Per-episode RNG seeded by episode_id for reproducibility.
        episode_rng = random.Random(SEED + abs(hash(r["episode_id"])) % (10 ** 8))
        shifted, counts = shift_text(text, episode_rng)
        for k in total_counts:
            total_counts[k] += counts[k]
        out_path = SHIFTED_DIR / f"{r['episode_id']}_shifted.md"
        out_path.write_text(shifted, encoding="utf-8")
        out_rows.append({
            "episode_id": r["episode_id"],
            "original_tz_path": r["tz_path"],
            "shifted_tz_path": str(out_path.relative_to(LAST_ROOT)),
            "fas_verdict": r["fas_verdict"],
            "shifts_ids": counts["ids"],
            "shifts_inn": counts["inn"],
            "shifts_dates": counts["dates"],
            "shifts_orgs": counts["orgs"],
        })

    fieldnames = ["episode_id", "original_tz_path", "shifted_tz_path", "fas_verdict",
                  "shifts_ids", "shifts_inn", "shifts_dates", "shifts_orgs"]
    with PROBE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    log.info("wrote %s and %d shifted files; totals=%s", PROBE_CSV, len(out_rows), total_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
