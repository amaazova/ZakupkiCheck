"""W1 Session B — L0 regex extraction for TZ features."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import PROJECT_ROOT
from .data_loaders import load_tz
from .extraction_runner import append_jsonl, load_completed_ids


# Patterns

# Brand markers — phrases that introduce a brand/model. Designed to fire only
# when the marker is followed by an actual candidate (quoted name or latin
# model code), not when the marker is itself part of a different phrase
# (e.g. "указанием производителя", "(изготовителя)" — the next token is then
# not a brand). The downstream filter rejects role/role-org words.
BRAND_MARKER_RE = re.compile(
    r"(?:торгов[аоы][йя]\s+марк[аи]|"
    r"товарн[ыоа][йяe]\s+знак[аи]?|"
    r"\bбренд[аы]?\b|"
    r"\bмодел[ьи]\b|"
    r"марк[аи]\s+товара|"
    r"производства\s+(?=[«\"A-ZА-Я])|"  # "производства «Samsung»" / "производства HP"
    r"производитель\s+(?=[«\"A-ZА-Я]))",
    re.IGNORECASE,
)

# Role / participant words that are NOT brands but commonly appear in TZ text.
ROLE_WORDS = {
    "поставщик", "поставщика", "поставщику", "поставщиков",
    "заказчик", "заказчика", "заказчику", "заказчиков",
    "подрядчик", "подрядчика", "исполнитель", "исполнителя",
    "получатель", "получателя", "грузополучатель", "грузополучателя",
    "грузоотправитель", "товар", "товара", "товары", "продукция",
    "упак", "упаковка", "коробка",
}

# Organization-name fragments to filter out of quoted candidates.
ORG_NAME_MARKERS = (
    "управлен", "учрежден", "служба", "служб", "минист", "адмиистр", "администр",
    "комитет", "департамент", "агентств", "фонд ", "фонда ", "школа", "детский сад",
    "больниц", "поликлин", "санитарн", "военн", "академ", "университет", "училищ",
    "техникум", "колледж", "лицей", "гимназ", "правительств", "дума",
    "ао ", "ао\"", "оао", "пао", "зао", "ооо", "ип ", "ип\"", "фгбу", "гбу", "мбу",
    "фгуп", "гуп", "муп", "фкп", "гкп",
    "поставк", "описани", "оказани", "выполнени", "приёмк", "приемк",
    "обоснован", "техническ", "график", "приложен", "извещен", "контракт",
    "договор", "товаров", "работ", "услуг", "капитальн",
)

# Quoted capitalized names: «Samsung», «Бош», "Philips". Limit length so this
# does not catch sentence-long quoted phrases.
QUOTED_NAME_RE = re.compile(
    r"[«„\"]([A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\s\-\.&\+/]{1,40})[»\"„]"
)

# Model codes: capital letters + digits, e.g. "M404", "L132", "X-200", "HP M404dw".
# Require at least one digit to filter out plain acronyms like "USB" / "HDMI" /
# "AUTH" / etc. The brand-name part (Samsung, Philips) lives in QUOTED_NAME_RE.
LATIN_MODEL_RE = re.compile(
    r"\b([A-Z]{2,}[\s\-]?\d{2,}[A-Z0-9\-]*|"     # HP 404, EPSON L132
    r"[A-Z]{1,}\-\d{2,}[A-Z0-9\-]*|"             # X-200, M-404
    r"[A-Z][a-z]{2,}\s*\d{2,}[A-Z0-9\-]*)\b"     # Samsung 220, Galaxy S22
)

# Trademark sigils.
SIGIL_RE = re.compile(r"[™®]")

# Equivalence clause.
EQUIVALENT_RE = re.compile(
    r"или\s+эквивалент|или\s+аналог|эквивалент(?:а|ы|ность|ам|ами|ах|у)?|"
    r"аналог(?:а|и|ичн|ам|ами|ах|у)?",
    re.IGNORECASE,
)

# Common false positives that the LATIN_MODEL_RE catches.
LATIN_STOP_TOKENS = {
    # Russian regulatory abbreviations
    "ОКПД", "ОКПД2", "ГОСТ", "ТУ", "СНИП", "СП", "КТРУ", "ТЗ", "НМЦК", "ФЗ",
    "РФ", "СССР", "НДС", "ЕИС", "ЕСМ", "ОКЕИ", "ОКВЭД", "ОГРН", "ИНН", "КПП",
    "УТВЕРЖДАЮ", "СОГЛАСОВАНО", "ПРИЛОЖЕНИЕ", "РАЗДЕЛ", "ГЛАВА", "СТАТЬЯ",
    "ПУНКТ", "ЧАСТЬ", "ПОДПУНКТ", "АБЗАЦ", "СОДЕРЖАНИЕ", "ОПИСАНИЕ",
    "ПЕРЕЧЕНЬ", "ТАБЛИЦА", "РИСУНОК", "СХЕМА", "СОСТАВ", "НАЛИЧИЕ",
    # File / formatting artefacts from docx/xlsx export
    "SHEET", "TABLE", "ROW", "CELL", "USD", "EUR", "RUR",
    "PDF", "DOC", "DOCX", "XLS", "XLSX", "TXT", "XML", "JSON",
    # Generic standards bodies
    "ISO", "DIN", "EN", "ANSI", "IEC", "ITU", "IETF", "IEEE", "ASTM", "BS",
    # Generic interface / connector names — NOT brands
    "USB", "HDMI", "VGA", "DVI", "RJ", "RJ45", "BNC", "GPS", "GLONASS",
    "LED", "OLED", "LCD", "LAN", "WAN", "PAN", "BT", "NFC",
    "ETHERNET", "WIFI", "WIRELESS", "BLUETOOTH",
    # Generic file/format/transport (not brands)
    "MS",  # "MS Word", "MS Office"
    "ROHS", "CE",
}

# Units of measurement.
UNIT_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*"
    r"(мм|см|дм|км|м|кг|мг|тонн|т|г|л|мл|шт|компл|компл\.|упак|"
    r"в|вт|квт|мвт|гц|кгц|мгц|ггц|°c|°с|кельвин|"
    r"%|м[²2³3]|см[²2³3]|мм[²2³3]|кв\.?\s*м|куб\.?\s*м|"
    r"дюйм[аы]?|шт\.|пар[аы]?|комплект[аы]?|упаковк[аи])\b",
    re.IGNORECASE,
)

# Ranges: "от X до Y", "не менее", "не более"
RANGE_RE = re.compile(
    r"(?:от\s+\d+(?:[.,]\d+)?\s*[а-яёa-z]*\s+до\s+\d+|"
    r"не\s+мене[ея]|не\s+бол[еье]|"
    r"в\s+пределах|"
    r"диапазон|"
    r"≥|≤|>=|<=)",
    re.IGNORECASE,
)

# GOSTs / regulations.
GOST_RE = re.compile(
    r"(?:ГОСТ(?:\s+Р)?\s*\d+[\.\-]?\d*[\.\-]?\d*|"
    r"ТУ\s*\d{2}[\.\-]\d+|"
    r"СНи[Пп]\s*\d+[\.\-]?\d*|"
    r"СП\s*\d+[\.\-]?\d*)",
    re.IGNORECASE,
)

# KTRU code: dd.dd.dd.ddd (the third level identifies a КТРУ position).
KTRU_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{2}\.\d{3})\b")

# Restrictive phrases.
RESTRICTIVE_RE = re.compile(
    r"\b(только\s+(?!для|если|после|при|по|в)\w|"
    r"исключительно\s+\w|"
    r"единственн[ыойаяе]\w*|"
    r"конкретн[ыойаяе]\w*|"
    r"именно\s+\w|"
    r"не\s+допускается\s+замен|"
    r"не\s+допускаются?\s+аналог)",
    re.IGNORECASE,
)


# Helpers

def _has_equivalent_in_window(text: str, pos: int, radius: int = 200) -> bool:
    """True if 'или эквивалент' or 'или аналог' is within ±radius chars of pos."""
    chunk = text[max(0, pos - radius): pos + radius]
    return bool(EQUIVALENT_RE.search(chunk))


def _quote_around(text: str, pos: int, before: int = 30, after: int = 70) -> str:
    """Return a short evidence quote: ~100 chars centered on pos."""
    start = max(0, pos - before)
    end = min(len(text), pos + after)
    s = text[start:end].replace("\n", " ").strip()
    return s


def _is_acronym_token(token: str) -> bool:
    return token.upper() in LATIN_STOP_TOKENS


_ROMAN_PREFIX_RE = re.compile(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)\b")


def _looks_like_brand_token(token: str) -> bool:
    """Heuristic filter on Latin model candidates."""
    if len(token) < 3:
        return False
    if _is_acronym_token(token):
        return False
    alpha = sum(1 for c in token if c.isalpha())
    if alpha < 2:
        return False
    # Roman numeral prefix → likely a normative reference (СНиП III-10-75, etc.)
    if _ROMAN_PREFIX_RE.match(token):
        return False
    return True


# Brand extraction

def _is_role_word(name: str) -> bool:
    return name.lower().strip(".,;:()[]«»\"") in ROLE_WORDS


def _is_org_name(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in ORG_NAME_MARKERS)


def _extract_brands(text: str) -> list[dict[str, Any]]:
    """Return list of {brand, has_equivalent_clause, position, source}."""
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    # 1) brand-marker contexts: capture the next QUOTED name or latin model
    #    after a marker. Russian role words and organisation names are rejected.
    for m in BRAND_MARKER_RE.finditer(text):
        tail = text[m.end(): m.end() + 80]
        qm = QUOTED_NAME_RE.search(tail)
        name = None
        rel_pos = None
        source = "marker"
        if qm:
            cand = qm.group(1).strip()
            if not _is_org_name(cand) and not _is_role_word(cand) and len(cand) >= 3:
                name = cand
                rel_pos = qm.start()
        if name is None:
            # Latin model token in the first 80 chars after the marker
            lm = LATIN_MODEL_RE.search(tail)
            if lm and _looks_like_brand_token(lm.group(1)):
                name = lm.group(1).strip()
                rel_pos = lm.start()
        if name is None:
            continue
        pos = m.end() + rel_pos
        key = (name, pos)
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "brand": name,
            "position": pos,
            "source": source,
            "has_equivalent_clause": _has_equivalent_in_window(text, pos),
        })

    # 2) Quoted capitalised names — only brand-shaped: single word, or two-word
    #    with leading Latin token. Reject phrases with verbs/prepositions or
    #    document section titles.
    for m in QUOTED_NAME_RE.finditer(text):
        name = m.group(1).strip()
        if not re.search(r"[A-Za-zА-Яа-я]", name):
            continue
        if _is_role_word(name) or _is_org_name(name):
            continue
        if name.upper() in LATIN_STOP_TOKENS:
            continue
        if len(name) < 3 or name.islower():
            continue
        # reject quoted document section titles (start with «Об ...», «Об утверждении ...»,
        # contain "программ", "формирован" etc.)
        if re.search(r"\bоб\s|программ|формирован|организац|условиях|настоящ", name, re.I):
            continue
        # brand-shape filter: single token, OR Latin Capitalized followed by digits/dash.
        words = name.split()
        looks_like_brand = (
            len(words) == 1
            and re.fullmatch(r"[A-ZА-ЯЁ][A-Za-zА-Яа-яЁё0-9\-]{2,}", name) is not None
        ) or (
            2 <= len(words) <= 3
            and bool(re.fullmatch(r"[A-Z][A-Za-z0-9\-]{1,}", words[0]))
            # second word should be a model code, not a Russian common noun
            and bool(re.match(r"[A-Z0-9][A-Za-z0-9\-]*$", words[1]))
        )
        if not looks_like_brand:
            continue
        pos = m.start()
        key = (name, pos)
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "brand": name,
            "position": pos,
            "source": "quoted",
            "has_equivalent_clause": _has_equivalent_in_window(text, pos),
        })

    # 3) Latin model patterns
    for m in LATIN_MODEL_RE.finditer(text):
        tok = m.group(1).strip()
        if not _looks_like_brand_token(tok):
            continue
        pos = m.start()
        key = (tok, pos)
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "brand": tok,
            "position": pos,
            "source": "latin_model",
            "has_equivalent_clause": _has_equivalent_in_window(text, pos),
        })

    # 4) Trademark sigils — record presence; brand inferred from preceding token
    for m in SIGIL_RE.finditer(text):
        # nearest preceding word
        head = text[max(0, m.start() - 40): m.start()].strip().split()
        if not head:
            continue
        tok = head[-1].strip(",.;:()[]«»\"")
        if not tok or len(tok) < 2 or _is_acronym_token(tok):
            continue
        pos = m.start() - len(tok)
        key = (tok, pos)
        if key in seen:
            continue
        seen.add(key)
        found.append({
            "brand": tok,
            "position": pos,
            "source": "sigil",
            "has_equivalent_clause": _has_equivalent_in_window(text, pos),
        })

    return found


# Main entry points

def extract_l0(text: str, episode_id: str) -> dict[str, Any]:
    """Run all L0 regex extractors on `text`. Returns a dict in the L0 schema."""
    brands = _extract_brands(text)
    brand_count = len(brands)

    # global equivalent presence
    equivalent_matches = EQUIVALENT_RE.findall(text)
    equivalent_count = len(equivalent_matches)
    has_equivalent_clause = equivalent_count > 0

    units_count = sum(1 for _ in UNIT_RE.finditer(text))
    has_ranges = bool(RANGE_RE.search(text))
    gost_count = sum(1 for _ in GOST_RE.finditer(text))
    has_ktru = bool(KTRU_RE.search(text))
    restrictive_count = sum(1 for _ in RESTRICTIVE_RE.finditer(text))

    risk_flags: list[dict[str, Any]] = []
    for b in brands:
        if not b["has_equivalent_clause"]:
            risk_flags.append({
                "flag_type": "brand_without_equivalent",
                "evidence_quote": _quote_around(text, b["position"]),
                "confidence": 1.0,
                "brand": b["brand"],
                "source": b["source"],
            })

    return {
        "episode_id": episode_id,
        "level": "L0",
        "features": {
            "brand_mentions": [
                {"brand": b["brand"],
                 "has_equivalent_clause": b["has_equivalent_clause"],
                 "position": b["position"],
                 "source": b["source"]}
                for b in brands
            ],
            "brand_count": brand_count,
            "has_equivalent_clause": has_equivalent_clause,
            "equivalent_count": equivalent_count,
            "units_count": units_count,
            "has_ranges": has_ranges,
            "gost_count": gost_count,
            "has_ktru": has_ktru,
            "restrictive_phrase_count": restrictive_count,
        },
        "risk_flags": risk_flags,
    }


def run_l0(eval_df: pd.DataFrame, output_path: str | Path) -> dict[str, int]:
    """Process every episode in eval_df. Skip episode_ids already in output_path."""
    output_path = Path(output_path)
    completed = load_completed_ids(output_path)
    stats = {
        "processed": 0,
        "skipped_existing": len(completed),
        "missing_tz": 0,
        "with_brand": 0,
        "with_brand_unguarded": 0,
        "with_ktru": 0,
        "with_restrictive": 0,
    }
    rows = eval_df.to_dict(orient="records")
    pbar = tqdm(rows, desc="L0:regex", unit="ep")
    for row in pbar:
        episode_id = row["episode_id"]
        if str(episode_id) in completed:
            continue
        text = load_tz(row["notice_id"])
        if text is None:
            stats["missing_tz"] += 1
            continue
        rec = extract_l0(text, episode_id)
        # carry forward useful keys for downstream joining
        rec["notice_id"] = row["notice_id"]
        rec["cluster_id"] = row.get("cluster_id")
        rec["stratum"] = row.get("stratum")
        rec["fas_verdict"] = row.get("fas_verdict")
        append_jsonl(output_path, rec)
        stats["processed"] += 1
        feats = rec["features"]
        if feats["brand_count"] > 0:
            stats["with_brand"] += 1
        if any(f["flag_type"] == "brand_without_equivalent" for f in rec["risk_flags"]):
            stats["with_brand_unguarded"] += 1
        if feats["has_ktru"]:
            stats["with_ktru"] += 1
        if feats["restrictive_phrase_count"] > 0:
            stats["with_restrictive"] += 1
    pbar.close()
    return stats


def main() -> None:
    eval_df = pd.read_csv(
        PROJECT_ROOT / "workspace" / "eval" / "eval_dataset_v10.csv",
        dtype={"notice_id": str},
    )
    out = PROJECT_ROOT / "workspace" / "eval" / "tz_features_L0.jsonl"
    stats = run_l0(eval_df, out)
    print("\n=== L0 summary ===")
    for k, v in stats.items():
        print(f"  {k:<25} {v}")
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
