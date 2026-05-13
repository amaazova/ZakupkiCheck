"""W1 Session A — FAS findings extraction (per decision, fan-out per episode)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from workspace.scripts import config, extraction_runner, data_loaders  # type: ignore
else:
    from . import config, extraction_runner, data_loaders


EVAL_CSV = config.PROJECT_ROOT / "workspace" / "eval" / "eval_dataset_v10.csv"
RAW_OUT = config.PROJECT_ROOT / "workspace" / "eval" / "_fas_decisions_raw.jsonl"
PER_EP_OUT = config.PROJECT_ROOT / "workspace" / "eval" / "fas_findings.jsonl"
SMOKE_RAW = config.PROJECT_ROOT / "workspace" / "eval" / "_fas_decisions_raw_smoke.jsonl"
SMOKE_PER_EP = config.PROJECT_ROOT / "workspace" / "eval" / "fas_findings_test.jsonl"

VALID_FINDING_TYPES = {
    "brand_without_equivalent", "incomplete_description", "restrictive_requirement",
    "ktru_mismatch", "inconsistent_dates", "missing_acceptance_terms", "other",
}
VALID_SEVERITIES = {"established", "not_established", "partial"}
VALID_VERDICTS = {"violation_established", "violation_not_established"}

SYSTEM_PROMPT = """Ты эксперт по российским государственным закупкам (44-ФЗ) и анализу решений ФАС России.

Твоя задача — извлечь все findings (нарушения, рассмотренные комиссией) из текста решения ФАС и вернуть их как структурированный JSON.

Правила:
1. Отвечай ТОЛЬКО валидным JSON-объектом. Без markdown-фенсов, без пояснений до или после.
2. Извлекай ВСЕ findings, рассмотренные комиссией, не только главное.
3. Если нарушений нет — вернуть пустой список findings.
4. evidence_quote — это дословная цитата из документа закупки (а не из самого решения), на которую ссылается комиссия. Если комиссия не цитирует конкретный фрагмент документа закупки — оставь пустую строку.
5. cited_articles — номера статей 44-ФЗ в формате "33", "33.1", "33.1.1", "42", "51".
6. severity = "established" если комиссия признала нарушение; "not_established" если рассматривала и отклонила; "partial" если признала частично.
7. Текст решения может содержать markdown-таблицы с искажённым форматированием — извлекай по смыслу, не по форматированию.
8. Не выдумывай нарушения, которых нет в тексте. Лучше пустой список findings, чем галлюцинация.

finding_type — ОДИН из словаря (использовать только эти значения):
- brand_without_equivalent — указание бренда/модели/производителя без оговорки «или эквивалент» / «или аналог»
- incomplete_description — неполное описание объекта закупки (нет характеристик, параметров, объёмов)
- restrictive_requirement — ограничительное/избыточное требование, искусственно сужающее круг участников
- ktru_mismatch — несоответствие позиции/характеристик КТРУ
- inconsistent_dates — противоречивые/нереальные сроки исполнения, действия контракта, приёмки
- missing_acceptance_terms — отсутствие критериев приёмки или порядка экспертизы
- other — иное нарушение по ст. 33 / 42 / 51, не подходящее под выше"""

USER_PROMPT_TEMPLATE = """Извлеки структурированные findings из этого решения ФАС.

Схема ответа (СТРОГО, верни ровно эту структуру):
{{
  "findings": [
    {{
      "finding_type": "<один из словаря>",
      "description": "краткое описание своими словами (1-2 предложения)",
      "evidence_quote": "дословная цитата из документа закупки (или пустая строка)",
      "cited_articles": ["33.1.1"],
      "severity": "established | not_established | partial"
    }}
  ],
  "overall_verdict": "violation_established | violation_not_established",
  "key_reasoning": "одно предложение — главный аргумент комиссии"
}}

Текст решения:
<<DECISION>>
{fas_text}
<<END_DECISION>>"""


def episode_to_decision(eid: str) -> str:
    return eid.rsplit("_", 1)[0]


def build_decision_list(eval_df: pd.DataFrame, *, decision_subset: set[str] | None = None) -> list[dict]:
    """Return a list of unique-decision dicts ready for run_extraction."""
    eval_df = eval_df.copy()
    eval_df["decision_id"] = eval_df["episode_id"].apply(episode_to_decision)
    unique_decisions = sorted(eval_df["decision_id"].unique().tolist())
    if decision_subset is not None:
        unique_decisions = [d for d in unique_decisions if d in decision_subset]

    out: list[dict] = []
    skipped = 0
    for did in unique_decisions:
        text = data_loaders.load_fas_text(did)
        if not text:
            skipped += 1
            continue
        out.append({"episode_id": did, "decision_id": did, "fas_text": text})
    if skipped:
        print(f"[fas-extract] {skipped} decisions skipped (no FAS text)", file=sys.stderr)
    return out


def normalize_finding(f: dict) -> dict:
    """Coerce a raw finding dict into the W1 schema."""
    if not isinstance(f, dict):
        return {"finding_type": "other", "description": "", "evidence_quote": "",
                "cited_articles": [], "severity": "not_established",
                "_normalized_warning": "non_dict_finding"}
    ftype = str(f.get("finding_type", "")).strip()
    if ftype not in VALID_FINDING_TYPES:
        ftype = "other"
    sev = str(f.get("severity", "")).strip().lower()
    if sev not in VALID_SEVERITIES:
        sev = "not_established"
    arts_raw = f.get("cited_articles", [])
    if isinstance(arts_raw, str):
        arts_raw = [arts_raw]
    cited = [str(a).strip() for a in arts_raw if str(a).strip()]
    return {
        "finding_type": ftype,
        "description": str(f.get("description", ""))[:1000],
        "evidence_quote": str(f.get("evidence_quote", ""))[:1500],
        "cited_articles": cited,
        "severity": sev,
    }


def normalize_decision_record(parsed: Any) -> dict:
    """Coerce model output into the per-decision schema. Always returns a well-formed dict."""
    if not isinstance(parsed, dict):
        return {"findings": [], "overall_verdict": "violation_not_established",
                "key_reasoning": "(parse_failed)"}
    findings_raw = parsed.get("findings", [])
    if not isinstance(findings_raw, list):
        findings_raw = []
    findings = [normalize_finding(f) for f in findings_raw]
    verdict = str(parsed.get("overall_verdict", "")).strip()
    if verdict not in VALID_VERDICTS:
        # If any finding has severity=established → violation, else not
        verdict = "violation_established" if any(f["severity"] == "established" for f in findings) \
            else "violation_not_established"
    key = str(parsed.get("key_reasoning", ""))[:500]
    return {"findings": findings, "overall_verdict": verdict, "key_reasoning": key}


def fan_out(eval_df: pd.DataFrame, raw_path: Path, out_path: Path) -> dict:
    """Read per-decision raw output, write per-episode JSONL with normalized findings."""
    by_decision: dict[str, dict] = {}
    for rec in extraction_runner.read_jsonl(raw_path):
        did = str(rec.get("episode_id"))
        by_decision[did] = rec

    if out_path.is_file():
        out_path.unlink()

    written = 0
    skipped_no_decision = 0
    by_status_decision: Counter[str] = Counter()
    eval_df = eval_df.copy()
    eval_df["decision_id"] = eval_df["episode_id"].apply(episode_to_decision)

    for row in eval_df.to_dict(orient="records"):
        did = str(row["decision_id"])
        rec = by_decision.get(did)
        if rec is None:
            skipped_no_decision += 1
            continue
        status = rec.get("status", "unknown")
        by_status_decision[status] += 1
        norm = normalize_decision_record(rec.get("parsed"))
        out_rec = {
            "episode_id": row["episode_id"],
            "decision_id": did,
            "notice_id": str(row["notice_id"]),
            "cluster_id": int(row["cluster_id"]),
            "stratum": row.get("stratum", ""),
            "gold_fas_verdict": row.get("fas_verdict", ""),
            "model": rec.get("model", ""),
            "decision_status": status,  # "ok" | "parse_failed" | "http_*"
            "extraction_cost_usd": float(rec.get("cost_usd") or 0.0),
            "extraction_input_tokens": int(rec.get("input_tokens") or 0),
            "extraction_output_tokens": int(rec.get("output_tokens") or 0),
            **norm,
        }
        extraction_runner.append_jsonl(out_path, out_rec)
        written += 1
    return {"written": written, "skipped_no_decision": skipped_no_decision,
            "by_status": dict(by_status_decision)}


def sanity_report(eval_df: pd.DataFrame, per_ep_path: Path) -> dict:
    results = extraction_runner.read_jsonl(per_ep_path)
    n = len(results)
    if n == 0:
        return {"n": 0}
    eval_map = {row["episode_id"]: row["fas_verdict"] for _, row in eval_df.iterrows()}

    empty = [r for r in results if not r.get("findings")]
    empty_violations = [r for r in empty
                        if eval_map.get(r["episode_id"]) == "violation_established"]
    n_violations = sum(1 for v in eval_map.values() if v == "violation_established")

    type_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    for r in results:
        for f in r.get("findings", []):
            type_counts[f.get("finding_type", "MISSING")] += 1
            severity_counts[f.get("severity", "MISSING")] += 1

    agree = sum(1 for r in results if r.get("overall_verdict") == eval_map.get(r["episode_id"]))
    parse_failed = sum(1 for r in results if r.get("decision_status") != "ok")

    return {
        "n": n,
        "empty_findings": len(empty),
        "empty_findings_pct": len(empty) / n,
        "empty_violations": len(empty_violations),
        "n_violations_total": n_violations,
        "empty_violations_rate_among_violations": (
            len(empty_violations) / n_violations if n_violations else 0.0
        ),
        "verdict_agreement": agree,
        "verdict_agreement_pct": agree / n,
        "type_counts": dict(type_counts),
        "severity_counts": dict(severity_counts),
        "decision_status_failures": parse_failed,
    }


def stop_gate(s: dict, *, smoke: bool) -> str | None:
    """Return a stop reason string if a STOP gate is tripped."""
    # Empty-findings gate is the project's headline gate.
    if s["n_violations_total"] >= 5:
        rate = s["empty_violations_rate_among_violations"]
        if rate > 0.30:
            return (f"empty_findings on {rate:.1%} of violation episodes "
                    f"({s['empty_violations']}/{s['n_violations_total']}) — > 30% threshold")
    # Verdict agreement gate
    if s["n"] >= 5 and s["verdict_agreement_pct"] < 0.70:
        return (f"verdict agreement {s['verdict_agreement_pct']:.1%} < 70%")
    # Smoke-time soft gate: no findings at all
    if smoke and sum(s["type_counts"].values()) == 0:
        return "no findings extracted at all in smoke run"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run on 5 random unique decisions; write to *_test outputs")
    ap.add_argument("--smoke-n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4,
                    help="ThreadPoolExecutor max_workers (default 4)")
    ap.add_argument("--no-stop", action="store_true",
                    help="do not abort on STOP gate failures (still print warning)")
    args = ap.parse_args()

    config.assert_api_key()
    eval_df = pd.read_csv(EVAL_CSV, dtype={"notice_id": str})
    print(f"[fas-extract] eval rows: {len(eval_df)}, model: {config.MODELS['v4flash']}")

    if args.smoke:
        # pick 5 RANDOM unique decisions, mix of verdicts
        eval_df_smoke = eval_df.copy()
        eval_df_smoke["decision_id"] = eval_df_smoke["episode_id"].apply(episode_to_decision)
        deciders = eval_df_smoke.drop_duplicates("decision_id")
        # stratify rough: at least one violation_established and one not_established
        ve = deciders[deciders["fas_verdict"] == "violation_established"].sample(
            n=min(3, (deciders["fas_verdict"] == "violation_established").sum()), random_state=42)
        vn = deciders[deciders["fas_verdict"] == "violation_not_established"].sample(
            n=min(args.smoke_n - len(ve), (deciders["fas_verdict"] == "violation_not_established").sum()),
            random_state=42)
        smoke_dids = set(pd.concat([ve, vn])["decision_id"].tolist())
        smoke_eval = eval_df_smoke[eval_df_smoke["decision_id"].isin(smoke_dids)]
        episodes = build_decision_list(smoke_eval, decision_subset=smoke_dids)
        raw_out = SMOKE_RAW
        per_ep_out = SMOKE_PER_EP
        print(f"[fas-extract] SMOKE — {len(episodes)} unique decisions covering {len(smoke_eval)} episodes")
    else:
        episodes = build_decision_list(eval_df)
        raw_out = RAW_OUT
        per_ep_out = PER_EP_OUT
        print(f"[fas-extract] FULL — {len(episodes)} unique decisions covering {len(eval_df)} episodes")

    t0 = time.monotonic()
    stats = extraction_runner.run_extraction(
        episodes=episodes,
        system_prompt=SYSTEM_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        output_path=raw_out,
        model=config.MODELS["v4flash"],
        temperature=0.0,
        max_tokens=2048,
        batch_delay=0.0,
        extra_keys=("decision_id",),
        max_workers=args.workers,
    )
    print(f"[fas-extract] runner finished in {time.monotonic() - t0:.1f}s; cost=${stats.cost_usd:.4f}")

    fan_summary = fan_out(eval_df if not args.smoke else smoke_eval, raw_out, per_ep_out)
    print(f"[fas-extract] fan-out: {fan_summary}")

    sanity = sanity_report(eval_df if not args.smoke else smoke_eval, per_ep_out)
    print(f"[fas-extract] sanity: {json.dumps(sanity, ensure_ascii=False, indent=2)}")

    stop_reason = stop_gate(sanity, smoke=args.smoke)
    if stop_reason:
        msg = f"⛔ STOP GATE: {stop_reason}"
        print(msg, file=sys.stderr)
        if not args.no_stop:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
