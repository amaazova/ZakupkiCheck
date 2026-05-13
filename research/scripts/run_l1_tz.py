"""W1 Session B — L1 TZ open extraction runner.

Reads workspace/eval/eval_dataset_v10.csv, builds episodes (TZ only, no FAS),
attaches `extra_instruction` based on quality_tier, truncates very long ТЗ to
fit DeepSeek V4 Flash context, then calls run_extraction().

Usage:
    python -m workspace.scripts.run_l1_tz --limit 5 --test   # test 3 hand-picked + smoke 5
    python -m workspace.scripts.run_l1_tz --smoke            # smoke 5 random
    python -m workspace.scripts.run_l1_tz                    # full 781
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, MODELS
from .data_loaders import build_episode_data
from .extraction_runner import run_extraction


EVAL_DIR = PROJECT_ROOT / "workspace" / "eval"
PROMPTS_DIR = EVAL_DIR / "prompts"
TZ_CHAR_LIMIT = 150_000  # safe margin under DeepSeek V4 Flash 64K-token context

CHUNKING_INSTRUCTION = (
    "Документ очень длинный. Сфокусируйся на описании объекта закупки, "
    "технических характеристиках и требованиях к товару/работе/услуге. "
    "Пропусти повторяющиеся таблицы спецификаций — достаточно одной репрезентативной."
)
THIN_INSTRUCTION = (
    "Документ короткий (1–3 тыс. символов). Не пытайся выдумать нарушения из ничего — "
    "если ТЗ выглядит корректным, верни пустой массив risk_flags."
)


def load_prompts() -> tuple[str, str]:
    system = (PROMPTS_DIR / "tz_l1_system_prompt.md").read_text(encoding="utf-8")
    user_tmpl = (PROMPTS_DIR / "tz_l1_user_prompt_template.md").read_text(encoding="utf-8")
    return system, user_tmpl


def prepare_episodes(eval_df: pd.DataFrame) -> list[dict]:
    """Build episodes (TZ only), attach extra_instruction, truncate long TZ."""
    episodes = build_episode_data(eval_df, include_fas=False)
    tier_lookup = dict(zip(eval_df["episode_id"], eval_df["quality_tier"]))
    for ep in episodes:
        tier = tier_lookup.get(ep["episode_id"], "normal")
        if tier == "chunking_risk":
            ep["extra_instruction"] = CHUNKING_INSTRUCTION
        elif tier == "thin":
            ep["extra_instruction"] = THIN_INSTRUCTION
        else:
            ep["extra_instruction"] = ""
        if len(ep["doc_text"]) > TZ_CHAR_LIMIT:
            ep["doc_text"] = ep["doc_text"][:TZ_CHAR_LIMIT] + "\n\n…[ТЗ обрезан до лимита контекста]…"
        ep["quality_tier"] = tier
    return episodes


def pick_three_test_episodes(eval_df: pd.DataFrame) -> pd.DataFrame:
    """For initial sanity: one benchmarkable, one service_only, one thin."""
    bench = eval_df[(eval_df.stratum == "benchmarkable") & (eval_df.quality_tier == "normal")].head(1)
    svc = eval_df[(eval_df.stratum == "service_only") & (eval_df.quality_tier == "normal")].head(1)
    thin = eval_df[eval_df.quality_tier == "thin"].head(1)
    return pd.concat([bench, svc, thin], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="run 3 hand-picked episodes to test prompt; out=tz_features_L1_test.jsonl")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke 5 random episodes; out=tz_features_L1_smoke.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    eval_df = pd.read_csv(EVAL_DIR / "eval_dataset_v10.csv", dtype={"notice_id": str})
    system, user_tmpl = load_prompts()

    if args.test:
        sample = pick_three_test_episodes(eval_df)
        out = EVAL_DIR / "tz_features_L1_test.jsonl"
        out.unlink(missing_ok=True)
        episodes = prepare_episodes(sample)
        print(f"[test] running 3 hand-picked: {[e['episode_id'] for e in episodes]}")
        run_extraction(
            episodes=episodes,
            system_prompt=system,
            user_prompt_template=user_tmpl,
            output_path=out,
            model=MODELS["v4flash"],
            max_tokens=4096,
            batch_delay=0.1,
        )
        return

    if args.smoke:
        sample = eval_df.sample(5, random_state=42)
        out = EVAL_DIR / "tz_features_L1_smoke.jsonl"
        out.unlink(missing_ok=True)
        episodes = prepare_episodes(sample)
        print(f"[smoke] 5 random: {[e['episode_id'] for e in episodes]}")
        run_extraction(
            episodes=episodes,
            system_prompt=system,
            user_prompt_template=user_tmpl,
            output_path=out,
            model=MODELS["v4flash"],
            max_tokens=4096,
            batch_delay=0.1,
        )
        return

    df = eval_df.head(args.limit) if args.limit > 0 else eval_df
    out = EVAL_DIR / "tz_features_L1.jsonl"
    episodes = prepare_episodes(df)
    print(f"[full] {len(episodes)} episodes, model={MODELS['v4flash']}, out={out}")
    run_extraction(
        episodes=episodes,
        system_prompt=system,
        user_prompt_template=user_tmpl,
        output_path=out,
        model=MODELS["v4flash"],
        max_tokens=4096,
        batch_delay=0.1,
    )


if __name__ == "__main__":
    main()
