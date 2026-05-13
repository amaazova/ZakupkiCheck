"""Runner — read eval_dataset.csv, call aggregator on each episode, append to predictions_b4.jsonl."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from . import aggregator, config

PROJECT_ROOT = config.PROJECT_ROOT
EVAL_DATASET = config.EVAL_DIR / "eval_dataset.csv"


def load_existing_predictions() -> set[str]:
    seen: set[str] = set()
    if config.PREDICTIONS_PATH.is_file():
        with config.PREDICTIONS_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("episode_id"):
                        seen.add(rec["episode_id"])
                except json.JSONDecodeError:
                    continue
    return seen


def read_tz(rel_path: str) -> str:
    p = PROJECT_ROOT / rel_path
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="process at most N episodes (0=all)")
    parser.add_argument("--smoke", action="store_true",
                        help="smoke mode: writes to predictions_smoke.jsonl, prints full results")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore existing predictions and reprocess")
    args = parser.parse_args()

    out_path = (
        config.EVAL_DIR / "predictions_smoke.jsonl"
        if args.smoke else config.PREDICTIONS_PATH
    )
    if args.smoke and out_path.is_file():
        out_path.unlink()

    seen = set() if (args.no_resume or args.smoke) else load_existing_predictions()
    print(f"existing predictions to skip: {len(seen)} (out={out_path.name})")

    with EVAL_DATASET.open(encoding="utf-8") as f:
        episodes = list(csv.DictReader(f))
    if args.limit > 0:
        episodes = episodes[: args.limit]
    print(f"episodes to consider: {len(episodes)}")

    total_cost = 0.0
    total_tokens = 0
    processed = 0
    started = time.monotonic()

    for i, ep in enumerate(episodes):
        episode_id = ep["episode_id"]
        if episode_id in seen:
            continue
        tz_text = read_tz(ep["tz_path"])
        if not tz_text:
            print(f"[{i+1}/{len(episodes)}] SKIP {episode_id}: tz not readable")
            continue
        t0 = time.monotonic()
        result = aggregator.run(
            tz_text,
            ep["cluster_name"],
            episode_id=episode_id,
        )
        elapsed = time.monotonic() - t0

        record = {
            "episode_id": episode_id,
            "notice_id": ep["notice_id"],
            "cluster_id": int(ep["cluster_id"]),
            "cluster_name": ep["cluster_name"],
            "stratum": ep["stratum"],
            "fas_verdict": ep["fas_verdict"],
            "pipeline_verdict": result["pipeline_verdict"],
            "detectors": result["detectors"],
            "tokens_used": result["tokens_used"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": result["cost_usd"],
            "elapsed_sec": round(elapsed, 2),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_cost += result["cost_usd"]
        total_tokens += result["tokens_used"]
        processed += 1

        agree = record["pipeline_verdict"] == (
            "violation" if ep["fas_verdict"] == "violation_established" else "not_established"
        )
        print(
            f"[{i+1}/{len(episodes)}] {episode_id} "
            f"fas={ep['fas_verdict'][:14]:<14} pipe={record['pipeline_verdict']:<15} "
            f"agree={'✓' if agree else '✗'} t={elapsed:.1f}s tok={result['tokens_used']} "
            f"$={result['cost_usd']:.4f}"
        )

        if args.smoke:
            print(json.dumps(record, ensure_ascii=False, indent=2))

    dt = time.monotonic() - started
    print(f"\n=== run summary ===")
    print(f"processed: {processed}")
    print(f"elapsed:   {dt:.1f} sec ({dt/60:.1f} min)")
    print(f"tokens:    {total_tokens}")
    print(f"cost:      ${total_cost:.4f}")
    if processed:
        print(f"avg/ep:    ${total_cost/processed:.4f}  ({total_tokens/processed:.0f} tok)")
        full_run_ep = 949
        proj = total_cost / processed * full_run_ep
        print(f"projected for full {full_run_ep} eps: ${proj:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
