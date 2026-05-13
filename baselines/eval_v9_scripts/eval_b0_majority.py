"""B0 baseline: per-episode majority verdict in its cluster (leave-one-out)."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import LAST_ROOT, setup_logger  # noqa: E402

EVAL_CSV = LAST_ROOT / "workspace" / "eval" / "eval_dataset.csv"
OUTPUT = LAST_ROOT / "workspace" / "eval" / "predictions_b0.jsonl"

VERDICT_NORM = {
    "violation_established": "violation",
    "violation_not_established": "not_established",
}


def main() -> int:
    log = setup_logger("eval_b0_majority")
    rows = list(csv.DictReader(EVAL_CSV.open(encoding="utf-8")))
    log.info("loaded %d episodes", len(rows))

    cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        cluster_counts[r["cluster_id"]][r["fas_verdict"]] += 1

    with OUTPUT.open("w", encoding="utf-8") as f:
        for r in rows:
            cid = r["cluster_id"]
            counts = Counter(cluster_counts[cid])
            counts[r["fas_verdict"]] -= 1  # leave-one-out
            if counts[r["fas_verdict"]] <= 0:
                del counts[r["fas_verdict"]]
            if counts:
                # Tie-break: alphabetical for determinism.
                top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            else:
                # Singleton cluster — fall back to dataset-wide majority.
                top = max(
                    Counter(rr["fas_verdict"] for rr in rows).items(),
                    key=lambda x: (x[1], -ord(x[0][0])),
                )[0]
            pred = VERDICT_NORM.get(top, top)
            gold = VERDICT_NORM.get(r["fas_verdict"], r["fas_verdict"])
            f.write(json.dumps({
                "episode_id": r["episode_id"],
                "notice_id": r["notice_id"],
                "cluster_id": cid,
                "fas_verdict": gold,
                "pipeline_verdict": pred,
            }, ensure_ascii=False) + "\n")

    log.info("wrote %s", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
