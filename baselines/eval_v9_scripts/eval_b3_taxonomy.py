"""B3 baseline: single LLM call per episode, prompted with the cluster name +
family description from family_document_requirements.csv."""
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

EVAL_CSV = LAST_ROOT / "workspace" / "eval" / "eval_dataset.csv"
FAMILY_CSV = LAST_ROOT / "family_document_requirements.csv"
OUTPUT = LAST_ROOT / "workspace" / "eval" / "predictions_b3.jsonl"

VERDICT_NORM = {
    "violation_established": "violation",
    "violation_not_established": "not_established",
}

SYSTEM_PROMPT = (
    "Ты эксперт по 44-ФЗ. Анализируешь техническое задание (ТЗ) на наличие "
    "конкретного типа нарушения, выделенного в практике ФАС. Отвечай строго JSON-ом."
)

USER_PROMPT_TEMPLATE = (
    "В практике ФАС выделяется следующий тип нарушения: «{cluster_name}».\n"
    "Описание: {family_description}\n\n"
    "Проверь, есть ли такое нарушение в этом ТЗ. Ответь строго в JSON:\n"
    '{{"verdict": "violation"|"not_established", '
    '"confidence": float (0..1), '
    '"evidence_quote": str, '
    '"reasoning": str}}\n\n'
    "Текст ТЗ:\n"
    "<<TZ>>\n"
    "{tz_text}\n"
    "<<END_TZ>>"
)


def load_family_descriptions() -> dict[int, dict[str, str]]:
    """Map cluster_id (int) → {family_name, notes}."""
    out: dict[int, dict[str, str]] = {}
    for row in csv.DictReader(FAMILY_CSV.open(encoding="utf-8")):
        fid = row["family_id"]
        if not fid.startswith("cluster_"):
            continue
        try:
            cid = int(fid.split("_", 1)[1])
        except ValueError:
            continue
        out[cid] = {
            "family_name": row.get("family_name", ""),
            "notes": row.get("notes", ""),
        }
    return out


def family_description(meta: dict[str, str], cluster_name_fallback: str) -> str:
    name = meta.get("family_name") or cluster_name_fallback
    notes = meta.get("notes") or ""
    if notes:
        return f"{name}. {notes}"
    return name


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


def make_user_prompt(cluster_name: str, family_desc: str, tz_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        cluster_name=cluster_name,
        family_description=family_desc,
        tz_text=tz_text,
    )


def process_one(ep: dict, family_meta: dict[int, dict[str, str]], out_path: Path) -> dict | None:
    tz = read_tz(ep["tz_path"])
    if not tz:
        return None
    tz = tz[: pcfg.TZ_TEXT_CHAR_LIMIT]
    try:
        cid = int(ep["cluster_id"])
    except ValueError:
        cid = -1
    fam = family_meta.get(cid, {})
    desc = family_description(fam, ep["cluster_name"])
    user = make_user_prompt(ep["cluster_name"], desc, tz)
    t0 = time.monotonic()
    try:
        r = call_llm(SYSTEM_PROMPT, user, episode_id=ep["episode_id"], detector="b3_taxonomy",
                     response_format="json")
    except Exception as e:
        record = {
            "episode_id": ep["episode_id"],
            "notice_id": ep["notice_id"],
            "cluster_id": cid,
            "cluster_name": ep["cluster_name"],
            "fas_verdict": VERDICT_NORM.get(ep["fas_verdict"], ep["fas_verdict"]),
            "pipeline_verdict": "not_established",
            "confidence": 0.0,
            "evidence_quote": "",
            "reasoning": f"call_failed: {type(e).__name__}: {str(e)[:200]}",
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "failed": True,
        }
    else:
        parsed = parse_detector_json(r.text)
        record = {
            "episode_id": ep["episode_id"],
            "notice_id": ep["notice_id"],
            "cluster_id": cid,
            "cluster_name": ep["cluster_name"],
            "fas_verdict": VERDICT_NORM.get(ep["fas_verdict"], ep["fas_verdict"]),
            "pipeline_verdict": parsed["verdict"],
            "confidence": parsed["confidence"],
            "evidence_quote": parsed["evidence_quote"],
            "reasoning": parsed["reasoning"],
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd,
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
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=str, default=str(OUTPUT))
    ap.add_argument("--early-sanity-after", type=int, default=30)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(EVAL_CSV.open(encoding="utf-8")))
    family_meta = load_family_descriptions()
    seen = load_existing(out_path)
    todo = [r for r in rows if r["episode_id"] not in seen]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"[b3] total={len(rows)} already_done={len(seen)} todo={len(todo)} workers={args.workers} "
          f"family_meta_loaded={len(family_meta)}")
    if not todo:
        print("[b3] nothing to do")
        return 0

    completed: list[dict] = []
    failed = 0
    started = time.monotonic()
    aborted = False
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, ep, family_meta, out_path): ep for ep in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            ep = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"[b3] WORKER_EXC {ep['episode_id']}: {type(e).__name__}: {e}")
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
                print(f"[b3] {i}/{len(todo)} done, cost=${cost:.3f} tok={tok} failed={failed} "
                      f"rate={rate:.2f}/s")
            if i == args.early_sanity_after:
                from collections import Counter
                c = Counter(r["pipeline_verdict"] for r in completed)
                if len([v for v in c.values() if v > 0]) <= 1:
                    print(f"[b3] EARLY SANITY FAIL: distribution={dict(c)} — aborting")
                    aborted = True
                    for f in futures:
                        f.cancel()
                    break

    elapsed = time.monotonic() - started
    cost = sum(r["cost_usd"] for r in completed)
    in_tok = sum(r["input_tokens"] for r in completed)
    out_tok = sum(r["output_tokens"] for r in completed)

    print(f"\n=== b3 summary ===")
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
        print("[b3] ⚠️ FINAL SANITY FAIL: degenerate distribution")
        return 2
    return 0 if not aborted else 2


if __name__ == "__main__":
    sys.exit(main())
