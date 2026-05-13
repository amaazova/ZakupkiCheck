"""W3 A5 — multi-model extraction (Sonnet + Qwen)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from workspace.scripts import config, extraction_runner  # type: ignore
    from workspace.scripts.run_w1_fas_extraction import (
        SYSTEM_PROMPT as FAS_SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE as FAS_USER_PROMPT_TEMPLATE,
        build_decision_list, fan_out, sanity_report, stop_gate,
    )
    from workspace.scripts.run_l1_tz_parallel import (
        load_prompts as load_tz_prompts,
        prepare_episodes as prepare_tz_episodes,
    )
else:
    from . import config, extraction_runner
    from .run_w1_fas_extraction import (
        SYSTEM_PROMPT as FAS_SYSTEM_PROMPT,
        USER_PROMPT_TEMPLATE as FAS_USER_PROMPT_TEMPLATE,
        build_decision_list, fan_out, sanity_report, stop_gate,
    )
    from .run_l1_tz_parallel import (
        load_prompts as load_tz_prompts,
        prepare_episodes as prepare_tz_episodes,
    )


EVAL = config.PROJECT_ROOT / "workspace" / "eval"
EVAL_CSV = EVAL / "eval_dataset_v10.csv"


# Output-path helpers

def paths_for(model_alias: str, task: str, *, smoke: bool) -> dict[str, Path]:
    suffix = "_smoke" if smoke else ""
    if task == "fas":
        return {
            "raw":    EVAL / f"_{model_alias}_fas_decisions_raw{suffix}.jsonl",
            "per_ep": EVAL / f"{model_alias}_fas_findings{suffix}.jsonl",
        }
    if task == "tz":
        return {
            "out": EVAL / f"{model_alias}_tz_features{suffix}.jsonl",
        }
    raise ValueError(task)


# FAS task

def run_fas(model_alias: str, model_id: str, *, workers: int, smoke: bool,
            smoke_n: int = 3, max_tokens: int = 2048) -> int:
    eval_df = pd.read_csv(EVAL_CSV, dtype={"notice_id": str})
    print(f"[A5/fas] model={model_id}  eval={len(eval_df)} episodes")

    if smoke:
        # 3 mixed-verdict decisions
        df = eval_df.copy()
        df["decision_id"] = df["episode_id"].str.rsplit("_", n=1).str[0]
        deciders = df.drop_duplicates("decision_id")
        ve = deciders[deciders["fas_verdict"] == "violation_established"].sample(
            n=min(2, (deciders["fas_verdict"] == "violation_established").sum()),
            random_state=43,
        )
        vn = deciders[deciders["fas_verdict"] == "violation_not_established"].sample(
            n=min(smoke_n - len(ve), (deciders["fas_verdict"] == "violation_not_established").sum()),
            random_state=43,
        )
        smoke_dids = set(pd.concat([ve, vn])["decision_id"].tolist())
        smoke_eval = df[df["decision_id"].isin(smoke_dids)]
        episodes = build_decision_list(smoke_eval, decision_subset=smoke_dids)
        eval_for_fanout = smoke_eval
        print(f"[A5/fas] SMOKE — {len(episodes)} unique decisions")
    else:
        episodes = build_decision_list(eval_df)
        eval_for_fanout = eval_df
        print(f"[A5/fas] FULL — {len(episodes)} unique decisions")

    p = paths_for(model_alias, "fas", smoke=smoke)
    raw_out = p["raw"]
    per_ep_out = p["per_ep"]

    t0 = time.monotonic()
    stats = extraction_runner.run_extraction(
        episodes=episodes,
        system_prompt=FAS_SYSTEM_PROMPT,
        user_prompt_template=FAS_USER_PROMPT_TEMPLATE,
        output_path=raw_out,
        model=model_id,
        temperature=0.0,
        max_tokens=max_tokens,
        batch_delay=0.0,
        extra_keys=("decision_id",),
        max_workers=workers,
    )
    print(f"[A5/fas] runner finished in {time.monotonic()-t0:.1f}s; cost=${stats.cost_usd:.4f}")

    fan_summary = fan_out(eval_for_fanout, raw_out, per_ep_out)
    print(f"[A5/fas] fan-out: {fan_summary}")

    if smoke:
        sanity = sanity_report(eval_for_fanout, per_ep_out)
        import json as _json
        print(f"[A5/fas/smoke] sanity: {_json.dumps(sanity, ensure_ascii=False, indent=2)}")
    return 0


# TZ task

def run_tz(model_alias: str, model_id: str, *, workers: int, smoke: bool,
           smoke_n: int = 3, max_tokens: int = 4096) -> int:
    eval_df = pd.read_csv(EVAL_CSV, dtype={"notice_id": str})
    print(f"[A5/tz] model={model_id}  eval={len(eval_df)} episodes")

    if smoke:
        df = eval_df.sample(n=smoke_n, random_state=43)
        print(f"[A5/tz] SMOKE — {len(df)} random episodes")
    else:
        df = eval_df

    episodes = prepare_tz_episodes(df)
    tz_system, tz_user_tmpl = load_tz_prompts()

    p = paths_for(model_alias, "tz", smoke=smoke)
    output_path = p["out"]

    t0 = time.monotonic()
    stats = extraction_runner.run_extraction(
        episodes=episodes,
        system_prompt=tz_system,
        user_prompt_template=tz_user_tmpl,
        output_path=output_path,
        model=model_id,
        temperature=0.0,
        max_tokens=max_tokens,
        batch_delay=0.0,
        extra_keys=("notice_id", "decision_id", "family", "verdict", "stratum", "quality_tier"),
        max_workers=workers,
    )
    print(f"[A5/tz] runner finished in {time.monotonic()-t0:.1f}s; cost=${stats.cost_usd:.4f}")
    if smoke:
        import json as _json
        recs = extraction_runner.read_jsonl(output_path)
        ok = sum(1 for r in recs if r.get("status") == "ok")
        parsed_dicts = sum(1 for r in recs if isinstance(r.get("parsed"), dict))
        print(f"[A5/tz/smoke] records={len(recs)} ok={ok} parsed_dict={parsed_dicts}")
    return 0


# CLI

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("sonnet", "qwen"), required=True,
                    help="Model alias from config.MODELS")
    ap.add_argument("--task", choices=("fas", "tz"), required=True,
                    help="Which extraction to run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="3-episode smoke; outputs to *_smoke.jsonl")
    ap.add_argument("--smoke-n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=None)
    args = ap.parse_args()

    config.assert_api_key()
    model_id = config.MODELS[args.model]
    if args.task == "fas":
        return run_fas(
            args.model, model_id,
            workers=args.workers, smoke=args.smoke,
            smoke_n=args.smoke_n,
            max_tokens=args.max_tokens if args.max_tokens is not None else 2048,
        )
    return run_tz(
        args.model, model_id,
        workers=args.workers, smoke=args.smoke,
        smoke_n=args.smoke_n,
        max_tokens=args.max_tokens if args.max_tokens is not None else 4096,
    )


if __name__ == "__main__":
    sys.exit(main())
