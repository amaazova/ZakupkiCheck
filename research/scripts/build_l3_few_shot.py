"""Build the L3 few-shot example bundle."""
from __future__ import annotations

import json
import re
from pathlib import Path

import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from workspace.scripts.config import PROJECT_ROOT  # type: ignore
else:
    from .config import PROJECT_ROOT


PROMPTS_DIR = PROJECT_ROOT / "workspace" / "eval" / "prompts"
FEW_SHOT_JSON = PROMPTS_DIR / "few_shot_examples.json"
FILLED_TEMPLATE = PROMPTS_DIR / "tz_l3_user_prompt_template.filled.md"


def find_window(tz: str, quote: str, ctx: int = 200) -> str:
    """Return a ~quote-len + 2*ctx window around the quote in tz."""
    qnorm = " ".join(quote.split()).lower()
    tnorm = " ".join(tz.split()).lower()
    pos = tnorm.find(qnorm)
    if pos < 0:
        return quote  # fall back
    # Walk back in normalized space, but we need the position in original text.
    # Re-walk word-by-word from the start.
    rebuilt: list[str] = []
    char_pos = 0
    cursor = 0
    for tok in tz.split():
        nt = tok.lower()
        if cursor == pos:
            break
        char_pos += len(tok) + 1
        cursor += len(nt) + 1
    start = max(0, char_pos - ctx)
    end = min(len(tz), char_pos + len(quote) + ctx)
    snippet = tz[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(tz):
        snippet = snippet + "…"
    return snippet


EXAMPLES = [
    {
        "label": "brand_without_equivalent",
        "episode_id": "202600121671001478_1",
        "notice_id": "0319300327723000044",  # placeholder — looked up below
        "quote_seed": "1.1.Барабанная установка Yamaha DTX-402K",
        "expected": {
            "risk_flags": [
                {
                    "flag_type": "brand_without_equivalent",
                    "confidence": 0.9,
                    "evidence_quote": "1.1. Барабанная установка Yamaha DTX-402K"
                }
            ],
            "features": {
                "brand_mentions": [
                    {"brand": "Yamaha DTX-402K", "has_equivalent_clause": False,
                     "quote": "1.1. Барабанная установка Yamaha DTX-402K"}
                ]
            }
        }
    },
    {
        "label": "incomplete_description",
        "episode_id": "202400143237000941_2",
        "quote_seed": "Оказание услуг по осуществлению технического надзора",
        "expected": {
            "risk_flags": [
                {
                    "flag_type": "incomplete_description",
                    "confidence": 0.85,
                    "evidence_quote": "Оказание услуг по осуществлению технического надзора на объекте: Строительство спортзала размерами … (характеристики, объёмы, единицы измерения не указаны)"
                }
            ],
            "features": {
                "measurement_completeness": {
                    "has_units": False,
                    "has_ranges": False,
                    "missing_characteristics": [
                        "трудозатраты", "график проверок", "критерии приёмки услуги"
                    ]
                }
            }
        }
    },
    {
        "label": "restrictive_requirement",
        "episode_id": "202400132489021870_1",
        "quote_seed": "Полнозаходная винтовая резьба на крышке и пробирке",
        "expected": {
            "risk_flags": [
                {
                    "flag_type": "restrictive_requirement",
                    "confidence": 0.85,
                    "evidence_quote": "Конструктивные особенности: Полнозаходная винтовая резьба на крышке и пробирке"
                }
            ],
            "features": {
                "restrictive_language": [
                    {"phrase": "Полнозаходная винтовая резьба на крышке и пробирке",
                     "restriction_type": "overly_specific"}
                ]
            }
        }
    },
]


def main() -> None:
    import pandas as pd
    eval_df = pd.read_csv(PROJECT_ROOT / "workspace" / "eval" / "eval_dataset_v10.csv",
                          dtype={"notice_id": str})
    eval_lookup = {r["episode_id"]: r for r in eval_df.to_dict(orient="records")}

    examples_out: list[dict] = []
    for ex in EXAMPLES:
        eid = ex["episode_id"]
        if eid not in eval_lookup:
            raise SystemExit(f"episode {eid} not in eval_dataset_v10")
        row = eval_lookup[eid]
        notice_id = str(row["notice_id"])
        tz_path = PROJECT_ROOT / "data" / "parsed_clean" / notice_id / "tz.md"
        tz = tz_path.read_text(encoding="utf-8", errors="replace")
        snippet = find_window(tz, ex["quote_seed"], ctx=200)
        examples_out.append({
            "label": ex["label"],
            "episode_id": eid,
            "notice_id": notice_id,
            "cluster_id": int(row["cluster_id"]),
            "tz_excerpt": snippet,
            "expected_extraction": ex["expected"],
        })

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    FEW_SHOT_JSON.write_text(
        json.dumps(examples_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {FEW_SHOT_JSON} with {len(examples_out)} examples")

    # Build the FILLED template — we substitute the three TZ excerpts + the three
    # expected extractions right into the L3 template, leaving only {doc_text}
    # and {extra_instruction} placeholders for the runner.
    skeleton = (PROMPTS_DIR / "tz_l3_user_prompt_template.md").read_text(encoding="utf-8")

    def _fmt_extraction(d: dict) -> str:
        return json.dumps(d, ensure_ascii=False, indent=2)

    filled = skeleton
    filled = filled.replace("{example_1_tz_excerpt}", examples_out[0]["tz_excerpt"])
    filled = filled.replace("{example_1_extraction}", _fmt_extraction(examples_out[0]["expected_extraction"]))
    filled = filled.replace("{example_2_tz_excerpt}", examples_out[1]["tz_excerpt"])
    filled = filled.replace("{example_2_extraction}", _fmt_extraction(examples_out[1]["expected_extraction"]))
    filled = filled.replace("{example_3_tz_excerpt}", examples_out[2]["tz_excerpt"])
    filled = filled.replace("{example_3_extraction}", _fmt_extraction(examples_out[2]["expected_extraction"]))

    # str.format() doesn't tolerate stray { in the prefilled JSON, so escape all
    # remaining braces EXCEPT our two real placeholders.
    DOC = "\x00DOC\x00"
    EXTRA = "\x00EXTRA\x00"
    filled_safe = filled.replace("{doc_text}", DOC).replace("{extra_instruction}", EXTRA)
    filled_safe = filled_safe.replace("{", "{{").replace("}", "}}")
    filled_safe = filled_safe.replace(DOC, "{doc_text}").replace(EXTRA, "{extra_instruction}")

    FILLED_TEMPLATE.write_text(filled_safe, encoding="utf-8")
    print(f"wrote {FILLED_TEMPLATE}")


if __name__ == "__main__":
    main()
