"""B4 baseline runner: parallelized wrapper around the SESSION_2 aggregator."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LAST_ROOT = Path("/Users/aza/Downloads/zakupki/last")
sys.path.insert(0, str(LAST_ROOT / "workspace" / "eval"))

from pipeline import aggregator, config as pcfg  # noqa: E402

EVAL_CSV = LAST_ROOT / "workspace" / "eval" / "eval_dataset.csv"
OUTPUT = LAST_ROOT / "workspace" / "eval" / "predictions_b4.jsonl"


def load_existing(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("episode_id"):
                    seen.add(rec["episode_id"])
            except json.JSONDecodeError:
                continue
    return seen


def read_tz(rel_path: str) -> str:
    p = LAST_ROOT / rel_path
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


_write_lock = threading.Lock()


def _norm_gold(v: str) -> str:
    return {"violation_established": "violation",
            "violation_not_established": "not_established"}.get(v, v)


def process_one(ep: dict, out_path: Path) -> dict | None:
    tz = read_tz(ep["tz_path"])
    if not tz:
        return None
    t0 = time.monotonic()
    try:
        result = aggregator.run(tz, ep["cluster_name"], episode_id=ep["episode_id"])
    except Exception as e:
        record = {
            "episode_id": ep["episode_id"],
            "notice_id": ep["notice_id"],
            "cluster_id": int(ep["cluster_id"]),
            "cluster_name": ep["cluster_name"],
            "stratum": ep["stratum"],
            "fas_verdict": _norm_gold(ep["fas_verdict"]),
            "pipeline_verdict": "not_established",
            "detectors": {},
            "tokens_used": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "failed": True,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }
    else:
        record = {
            "episode_id": ep["episode_id"],
            "notice_id": ep["notice_id"],
            "cluster_id": int(ep["cluster_id"]),
            "cluster_name": ep["cluster_name"],
            "stratum": ep["stratum"],
            "fas_verdict": _norm_gold(ep["fas_verdict"]),
            "pipeline_verdict": result["pipeline_verdict"],
            "detectors": result["detectors"],
            "tokens_used": result["tokens_used"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": result["cost_usd"],
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "failed": False,
        }
    with _write_lock:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", type=str, default=str(OUTPUT))
    ap.add_argument("--early-sanity-after", type=int, default=30)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(EVAL_CSV.open(encoding="utf-8")))
    seen = load_existing(out_path)
    todo = [r for r in rows if r["episode_id"] not in seen]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"[b4] total={len(rows)} already_done={len(seen)} todo={len(todo)} workers={args.workers}")
    if not todo:
        print("[b4] nothing to do")
        return 0

    completed: list[dict] = []
    failed = 0
    started = time.monotonic()
    aborted = False
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, ep, out_path): ep for ep in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            ep = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"[b4] WORKER_EXC {ep['episode_id']}: {type(e).__name__}: {e}")
                failed += 1
                continue
            if rec is None:
                continue
            completed.append(rec)
            if rec.get("failed"):
                failed += 1
            if i % 50 == 0:
                cost = sum(r["cost_usd"] for r in completed)
                tok = sum(r["input_tokens"] + r["output_tokens"] for r in completed)
                rate = i / max(time.monotonic() - started, 1.0)
                print(f"[b4] {i}/{len(todo)} done, cost=${cost:.3f} tok={tok} failed={failed} "
                      f"rate={rate:.2f}/s")
            if i == args.early_sanity_after:
                from collections import Counter
                c = Counter(r["pipeline_verdict"] for r in completed)
                if len([v for v in c.values() if v > 0]) <= 1:
                    print(f"[b4] EARLY SANITY FAIL: distribution={dict(c)} — aborting")
                    aborted = True
                    for f in futures:
                        f.cancel()
                    break

    elapsed = time.monotonic() - started
    cost = sum(r["cost_usd"] for r in completed)
    in_tok = sum(r["input_tokens"] for r in completed)
    out_tok = sum(r["output_tokens"] for r in completed)

    print(f"\n=== b4 summary ===")
    print(f"completed: {len(completed)}  failed: {failed}  aborted: {aborted}")
    print(f"elapsed:   {elapsed:.1f} sec ({elapsed/60:.1f} min)")
    print(f"tokens:    in={in_tok}  out={out_tok}  total={in_tok + out_tok}")
    print(f"cost:      ${cost:.4f}")

    all_records: list[dict] = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    from collections import Counter
    pdist = Counter(r["pipeline_verdict"] for r in all_records)
    gdist = Counter(r["fas_verdict"] for r in all_records)
    correct = sum(1 for r in all_records if r["pipeline_verdict"] == r["fas_verdict"])
    print(f"verdict_distribution: pred={dict(pdist)} gold={dict(gdist)}")
    print(f"accuracy: {correct}/{len(all_records)} = {correct/max(len(all_records),1):.4f}")

    if len([v for v in pdist.values() if v > 0]) <= 1 and len(all_records) >= 30:
        print("[b4] ⚠️ FINAL SANITY FAIL: degenerate distribution")
        return 2
    return 0 if not aborted else 2


if __name__ == "__main__":
    sys.exit(main())
