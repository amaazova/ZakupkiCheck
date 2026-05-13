"""Memorization probe: run B3-style call on the 50 original tz.md AND on the
50 shifted tz files; compare per-episode verdicts and overall accuracy.

Inputs:
  - workspace/eval/memorization_probe_episodes.csv
  - workspace/eval/eval_dataset.csv (for cluster_name lookup)
  - family_document_requirements.csv

Outputs:
  - workspace/eval/memorization_results.csv
      columns: episode_id, cluster_id, cluster_name, fas_verdict,
               original_verdict, shifted_verdict,
               original_confidence, shifted_confidence,
               original_cost, shifted_cost
  - workspace/eval/memorization_report.md

Decision rule per task spec: if accuracy_orig - accuracy_shifted > 10 п.п.
→ memorization signal.
"""
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

from pipeline import config as pcfg  # noqa: E402
from pipeline.llm_client import call_llm  # noqa: E402
from pipeline.detectors._common import parse_detector_json  # noqa: E402

# Reuse B3 prompt machinery
sys.path.insert(0, str(LAST_ROOT / "scripts"))
from eval_b3_taxonomy import (  # noqa: E402
    SYSTEM_PROMPT, make_user_prompt, family_description, load_family_descriptions,
)


PROBE_CSV = LAST_ROOT / "workspace" / "eval" / "memorization_probe_episodes.csv"
EVAL_CSV = LAST_ROOT / "workspace" / "eval" / "eval_dataset.csv"
RESULTS_CSV = LAST_ROOT / "workspace" / "eval" / "memorization_results.csv"
REPORT_MD = LAST_ROOT / "workspace" / "eval" / "memorization_report.md"

VERDICT_NORM = {
    "violation_established": "violation",
    "violation_not_established": "not_established",
}


def load_eval_lookup() -> dict[str, dict[str, str]]:
    return {r["episode_id"]: r for r in csv.DictReader(EVAL_CSV.open(encoding="utf-8"))}


def read_text(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = LAST_ROOT / p
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


_lock = threading.Lock()


def call_one(ep_id: str, cluster_name: str, family_desc: str, tz: str, label: str) -> dict:
    tz = tz[: pcfg.TZ_TEXT_CHAR_LIMIT]
    user = make_user_prompt(cluster_name, family_desc, tz)
    t0 = time.monotonic()
    try:
        r = call_llm(SYSTEM_PROMPT, user, episode_id=f"{ep_id}::{label}",
                     detector="b3_memorization", response_format="json")
        parsed = parse_detector_json(r.text)
        return {
            "verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "evidence_quote": parsed["evidence_quote"],
            "reasoning": parsed["reasoning"],
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "failed": False,
        }
    except Exception as e:
        return {
            "verdict": "not_established",
            "confidence": 0.0,
            "evidence_quote": "",
            "reasoning": f"call_failed: {type(e).__name__}: {str(e)[:200]}",
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "failed": True,
        }


def process_pair(probe_row: dict, eval_lookup: dict[str, dict[str, str]],
                 family_meta: dict[int, dict[str, str]]) -> dict | None:
    ep_id = probe_row["episode_id"]
    eval_row = eval_lookup.get(ep_id)
    if not eval_row:
        return None
    try:
        cid = int(eval_row["cluster_id"])
    except ValueError:
        cid = -1
    cluster_name = eval_row["cluster_name"]
    fam = family_meta.get(cid, {})
    desc = family_description(fam, cluster_name)

    orig_text = read_text(probe_row["original_tz_path"])
    shft_text = read_text(probe_row["shifted_tz_path"])
    if not orig_text or not shft_text:
        return None

    orig = call_one(ep_id, cluster_name, desc, orig_text, "orig")
    shft = call_one(ep_id, cluster_name, desc, shft_text, "shifted")

    return {
        "episode_id": ep_id,
        "cluster_id": cid,
        "cluster_name": cluster_name,
        "fas_verdict": VERDICT_NORM.get(probe_row["fas_verdict"], probe_row["fas_verdict"]),
        "original_verdict": orig["verdict"],
        "shifted_verdict": shft["verdict"],
        "original_confidence": orig["confidence"],
        "shifted_confidence": shft["confidence"],
        "original_cost": orig["cost_usd"],
        "shifted_cost": shft["cost_usd"],
        "original_tokens_in": orig["input_tokens"],
        "shifted_tokens_in": shft["input_tokens"],
        "original_tokens_out": orig["output_tokens"],
        "shifted_tokens_out": shft["output_tokens"],
        "original_failed": orig["failed"],
        "shifted_failed": shft["failed"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    probe_rows = list(csv.DictReader(PROBE_CSV.open(encoding="utf-8")))
    if args.limit > 0:
        probe_rows = probe_rows[: args.limit]
    eval_lookup = load_eval_lookup()
    family_meta = load_family_descriptions()
    print(f"[mem] probe rows: {len(probe_rows)}, workers={args.workers}")

    results: list[dict] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_pair, r, eval_lookup, family_meta): r for r in probe_rows}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                pr = futures[fut]
                print(f"[mem] WORKER_EXC {pr['episode_id']}: {type(e).__name__}: {e}")
                continue
            if res is None:
                continue
            results.append(res)
            if i % 10 == 0:
                cost = sum(r["original_cost"] + r["shifted_cost"] for r in results)
                print(f"[mem] {i}/{len(probe_rows)} pairs done, cost=${cost:.3f}")

    fieldnames = list(results[0].keys()) if results else []
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    # Metrics
    n = len(results)
    correct_orig = sum(1 for r in results if r["original_verdict"] == r["fas_verdict"])
    correct_shft = sum(1 for r in results if r["shifted_verdict"] == r["fas_verdict"])
    flips_to_match = sum(
        1 for r in results
        if r["original_verdict"] != r["shifted_verdict"]
    )
    acc_orig = correct_orig / n if n else 0.0
    acc_shft = correct_shft / n if n else 0.0
    delta_pp = (acc_orig - acc_shft) * 100
    memorization_signal = delta_pp > 10.0

    total_cost = sum(r["original_cost"] + r["shifted_cost"] for r in results)
    total_tok_in = sum(r["original_tokens_in"] + r["shifted_tokens_in"] for r in results)
    total_tok_out = sum(r["original_tokens_out"] + r["shifted_tokens_out"] for r in results)
    fails_orig = sum(1 for r in results if r["original_failed"])
    fails_shft = sum(1 for r in results if r["shifted_failed"])

    elapsed = time.monotonic() - started

    report = [
        "# Memorization Probe Report",
        "",
        f"- pairs evaluated: **{n}**",
        f"- prompt mode: B3 (taxonomy-aware), DeepSeek V3 via OpenRouter",
        f"- elapsed: {elapsed:.1f} sec ({elapsed/60:.1f} min)",
        "",
        "## Accuracy",
        "",
        f"- accuracy on **original** tz: **{acc_orig:.4f}** ({correct_orig}/{n})",
        f"- accuracy on **shifted** tz: **{acc_shft:.4f}** ({correct_shft}/{n})",
        f"- delta (orig − shifted): **{delta_pp:+.2f} п.п.**",
        f"- verdict flips between orig and shifted: **{flips_to_match}** ({flips_to_match/max(n,1):.2%})",
        "",
        "## Memorization Signal",
        "",
        f"- threshold: > 10 п.п. delta → signal",
        f"- result: **{'SIGNAL DETECTED' if memorization_signal else 'no signal'}**",
        "",
        "## Cost & Failures",
        "",
        f"- API calls: {n * 2}",
        f"- input tokens: {total_tok_in}",
        f"- output tokens: {total_tok_out}",
        f"- total cost: ${total_cost:.4f}",
        f"- failed calls: orig={fails_orig}, shifted={fails_shft}",
        "",
        "## Notes",
        "",
        "- 'Memorization signal' here is a coarse indicator: a large drop in",
        "  accuracy when entity surface forms are perturbed suggests the model",
        "  was relying on memorized notice/contract IDs rather than reasoning",
        "  about the document's content.",
        "- Stratified design (25 violation_established / 25 violation_not_established)",
        "  controls for label imbalance.",
        "",
    ]
    REPORT_MD.write_text("\n".join(report), encoding="utf-8")

    print(f"\n=== mem summary ===")
    print(f"pairs: {n}")
    print(f"acc_orig: {acc_orig:.4f}  acc_shft: {acc_shft:.4f}  delta: {delta_pp:+.2f}pp")
    print(f"signal: {memorization_signal}")
    print(f"cost: ${total_cost:.4f}  tokens: in={total_tok_in} out={total_tok_out}")
    print(f"failures: orig={fails_orig} shifted={fails_shft}")
    print(f"wrote {RESULTS_CSV} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
