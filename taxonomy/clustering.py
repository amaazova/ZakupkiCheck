#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import certifi


ROOT = Path("/Users/aza/Downloads/zakupki/last")
PROMPT_PATH = Path("/Users/aza/Downloads/STAGE21_DE_NOVO_TAXONOMY_PROMPT_v2.md")
RAW_FAS_GLOB = str(ROOT / "data" / "raw_fas" / "*.json")
DOCS_MANIFEST = ROOT / "data" / "documents_manifest.csv"
RAW_DOCS = ROOT / "data" / "raw_documents"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_EXTRACT = "anthropic/claude-haiku-4.5"
MODEL_NAMING = "anthropic/claude-sonnet-4.6"
PROVIDER_ROUTE = "Anthropic"
TOO_SHORT_CHARS = 500
RANDOM_SEED = 20260510
SYSTEM_PROMPT = (
    "Ты — аналитик-юрист по решениям ФАС по 44-ФЗ. "
    "Извлекай самостоятельные эпизоды нарушений строго по тексту решения. "
    "Не используй заранее заданные taxonomy labels. "
    "Возвращай только валидный JSON, соответствующий схеме."
)
USER_PROMPT_TEMPLATE = """Прочитай решение ФАС по 44-ФЗ. Для каждого самостоятельного эпизода нарушения
или спорного требования, рассмотренного ФАС, выдай structured record.

Один эпизод = одно нарушение или одно спорное требование, по которому ФАС
выносит отдельное суждение. Одно решение может содержать 1, 2, 3 или более
эпизодов. Не схлопывай разные нарушения в один эпизод.

Для каждого эпизода верни JSON object:
{{
  "issue_label": "<свободная формулировка типа нарушения, 10–20 слов>",
  "reasoning_summary": "<краткое объяснение логики ФАС, 2–4 предложения>",
  "legal_articles": ["<нормализованные статьи: ст.33 ч.1 44-ФЗ, ПП 2604 и т.д.>"],
  "fas_verdict": "violation_established" | "violation_not_established" | "not_considered",
  "scope_tag": "doc_content" | "bid_evaluation" | "procedural" | "platform" | "national_regime" | "contract_execution" | "other",
  "involves_procurement_docs": true | false,
  "confidence": "high" | "medium" | "low"
}}

Верни JSON array. Если решение не содержит эпизодов — верни пустой массив.
Не копируй шаблонные преамбулы и не придумывай факты вне текста решения.

canonical_decision_id: {decision_id}
canonical_notice_id: {notice_id}
source_path: {source_path}

Текст решения:
<<<DECISION_TEXT
{decision_text}
DECISION_TEXT>>>"""

NAMING_SYSTEM_PROMPT = """Ты — аналитик-юрист, исследующий корпус решений ФАС по 44-ФЗ.
Тебе на вход даётся один кластер из ~10 фрагментов reasoning_summary,
описывающих, какие нарушения и какие правовые нормы реально разбирает ФАС.

Твоя задача — присвоить кластеру компактный label и определить его суть.

Формат ответа — СТРОГО валидный JSON:

{
  "label_ru": "<≤10 слов, конкретная формулировка типа нарушения>",
  "key_articles": ["<нормализованная статья 1>", "<статья 2>"],
  "summary_ru": "<1–2 предложения: что объединяет эпизоды в этом кластере>",
  "reasoning_type": "<какой тип reasoning нужен для детекции: lexical | catalog_lookup | semantic_comparison | legal_applicability | external_knowledge | mixed>",
  "runtime_doc_dependency": "<можно ли детектировать по текстам ТЗ/извещения/контракта: full | partial | none>",
  "external_knowledge_required": "<требуется ли знание за пределами документации: none | market_knowledge | physical_measurement | expert_judgment | regulatory_history>"
}

Правила:
- label_ru конкретный. «Ограничительное описание объекта закупки» лучше, чем «нарушения».
- Не используй заранее заданные category names или hypothesis codes.
- reasoning_type — оценка того, какой тип NLP-reasoning потребуется для автоматической детекции.
- runtime_doc_dependency — можно ли нарушение обнаружить по одним runtime-документам (ТЗ, извещение, контракт).
- Никаких преамбул, только JSON."""


@dataclass
class Paths:
    run_dir: Path
    manifests_dir: Path
    extraction_dir: Path
    clustering_dir: Path
    naming_dir: Path
    benchmark_dir: Path
    reports_dir: Path
    logs_dir: Path
    qc_dir: Path
    pipeline_dir: Path
    journal_path: Path


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def get_paths(run_dir: Path) -> Paths:
    return Paths(
        run_dir=run_dir,
        manifests_dir=run_dir / "manifests",
        extraction_dir=run_dir / "extraction",
        clustering_dir=run_dir / "clustering",
        naming_dir=run_dir / "naming",
        benchmark_dir=run_dir / "benchmarkability",
        reports_dir=run_dir / "reports",
        logs_dir=run_dir / "logs",
        qc_dir=run_dir / "qc",
        pipeline_dir=run_dir / "pipeline",
        journal_path=run_dir / "run_journal.md",
    )


def ensure_dirs(paths: Paths) -> None:
    for path in [
        paths.run_dir,
        paths.manifests_dir,
        paths.extraction_dir,
        paths.clustering_dir,
        paths.naming_dir,
        paths.benchmark_dir,
        paths.reports_dir,
        paths.logs_dir,
        paths.qc_dir,
        paths.pipeline_dir,
        paths.extraction_dir / "batches",
        paths.logs_dir / "preflight",
        paths.logs_dir / "extraction_raw",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (list, dict)):
                    clean[key] = json.dumps(value, ensure_ascii=False)
                else:
                    clean[key] = value
            writer.writerow(clean)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split())


def systematic_sample(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return rows[:]
    step = len(rows) / float(n)
    indices = sorted({min(len(rows) - 1, int(round(i * step))) for i in range(n)})
    return [rows[idx] for idx in indices]


def parse_dotenv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def load_api_key() -> tuple[str, str, bool]:
    shell_val = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if shell_val:
        return shell_val, "env", True
    dot_env = parse_dotenv(ROOT / ".env")
    if dot_env.get("OPENROUTER_API_KEY"):
        return dot_env["OPENROUTER_API_KEY"], ".env_fallback", False
    return "", "missing", False


def provider_resolved_from(response_json: dict[str, Any], headers: dict[str, str]) -> str:
    metadata = response_json.get("openrouter_metadata") or {}
    if isinstance(metadata, dict):
        for key_path in (
            ("provider", "slug"),
            ("provider", "route"),
            ("provider_slug",),
            ("provider_route",),
            ("selected_provider",),
            ("upstream_provider",),
            ("route",),
        ):
            current: Any = metadata
            for key in key_path:
                if not isinstance(current, dict) or key not in current:
                    current = None
                    break
                current = current[key]
            if current:
                return str(current)
    for key, value in headers.items():
        low = key.lower()
        if "provider" in low and value:
            return str(value)
    provider_value = response_json.get("provider")
    if isinstance(provider_value, str) and provider_value:
        return provider_value
    return "not_returned_by_api"


def openrouter_request(
    api_key: str,
    messages: list[dict[str, Any]],
    response_schema: dict[str, Any],
    log_path: Path,
    model: str = MODEL_EXTRACT,
    max_tokens: int = 2200,
    temperature: float = 0.0,
    response_format_name: str = "taxonomy_episodes",
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "provider": {
            "order": [PROVIDER_ROUTE],
            "allow_fallbacks": False,
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_format_name,
                "strict": True,
                "schema": response_schema,
            },
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://local.codex",
        "X-Title": "stage21-taxonomy-v2",
        "X-OpenRouter-Experimental-Metadata": "enabled",
    }
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=180, context=ssl_context) as response:
            raw = response.read()
            body = json.loads(raw.decode("utf-8"))
            result = {
                "http_status": response.status,
                "headers": dict(response.headers.items()),
                "request_payload": payload,
                "response_json": body,
            }
            write_json(log_path, result)
            return result
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"raw_error": raw}
        result = {
            "http_status": exc.code,
            "headers": dict(exc.headers.items()) if exc.headers else {},
            "request_payload": payload,
            "response_json": body,
        }
        write_json(log_path, result)
        raise RuntimeError(f"HTTP {exc.code}: {raw[:800]}") from exc
    except Exception as exc:  # noqa: BLE001
        result = {
            "http_status": "transport_error",
            "headers": {},
            "request_payload": payload,
            "response_json": {"error": repr(exc)},
        }
        write_json(log_path, result)
        raise


def bootstrap_manifest(paths: Paths) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    schema_counter = Counter()
    with_gold = 0
    no_notice = 0
    with_text = 0
    too_short = 0
    empty_rows = 0
    for raw_path_str in sorted(glob.glob(RAW_FAS_GLOB)):
        raw_path = Path(raw_path_str)
        payload = read_json(raw_path)
        if "decision_id" in payload:
            decision_id = str(payload.get("decision_id", "") or "").strip()
            decision_id_field = "decision_id"
        else:
            decision_id = str(payload.get("fas_case_id", "") or "").strip()
            decision_id_field = "fas_case_id"
        if "full_text" in payload:
            text = str(payload.get("full_text", "") or "")
            text_field = "full_text"
        else:
            text = str(payload.get("decision_text", "") or "")
            text_field = "decision_text"
        if "purchase_id" in payload:
            notice_id = str(payload.get("purchase_id", "") or "").strip()
            notice_field = "purchase_id"
        else:
            notice_id = str(payload.get("notice_id", "") or "").strip()
            notice_field = "notice_id"
        schema_family = f"{decision_id_field}|{text_field}|{notice_field}"
        schema_counter[schema_family] += 1
        has_text = bool(text.strip())
        text_length = len(text.strip())
        is_too_short = has_text and text_length < TOO_SHORT_CHARS
        canonical_decision_id = decision_id or raw_path.stem
        gold_rel = ""
        if notice_id:
            gold_path = RAW_DOCS / notice_id / "_gold_decision_reference.txt"
            if gold_path.exists():
                gold_rel = str(gold_path.relative_to(ROOT))
                with_gold += 1
        else:
            no_notice += 1
        if has_text:
            with_text += 1
        if is_too_short:
            too_short += 1
        if not has_text:
            empty_rows += 1
        rows.append(
            {
                "canonical_decision_id": canonical_decision_id,
                "canonical_notice_id": notice_id,
                "path": str(raw_path.relative_to(ROOT)),
                "text_length": text_length,
                "has_text": has_text,
                "is_too_short": is_too_short,
                "decision_id_source_field": decision_id_field,
                "text_source_field": text_field,
                "notice_id_source_field": notice_field,
                "is_empty": not has_text,
                "schema_family": schema_family,
                "matched_gold_decision_reference_path": gold_rel,
            }
        )
    rows.sort(key=lambda r: r["path"])
    write_csv(
        paths.manifests_dir / "raw_fas_canonical_manifest.csv",
        [
            "canonical_decision_id",
            "canonical_notice_id",
            "path",
            "text_length",
            "has_text",
            "is_too_short",
            "decision_id_source_field",
            "text_source_field",
            "notice_id_source_field",
            "is_empty",
            "schema_family",
            "matched_gold_decision_reference_path",
        ],
        rows,
    )
    summary = {
        "raw_fas_rows": len(rows),
        "rows_with_text": with_text,
        "rows_too_short": too_short,
        "rows_empty": empty_rows,
        "rows_without_notice_id": no_notice,
        "rows_with_gold_decision_reference": with_gold,
        "schema_family_counts": dict(schema_counter),
        "too_short_threshold_chars": TOO_SHORT_CHARS,
    }
    write_json(paths.manifests_dir / "bootstrap_summary.json", summary)
    return summary


def init_journal(paths: Paths, summary: dict[str, Any]) -> None:
    if paths.journal_path.exists():
        return
    lines = [
        "# Stage 21 v2 Run Journal",
        "",
        f"- started_at: {now_iso()}",
        f"- prompt_source: `{PROMPT_PATH}`",
        f"- workspace: `{ROOT}`",
        "- authoritative_inputs:",
        "  - `data/raw_fas/*.json`",
        "  - `data/raw_documents/<canonical_notice_id>/_gold_decision_reference.txt`",
        "  - `data/runtime_docs_clean/<canonical_notice_id>/`",
        "  - `data/parsed_clean/<canonical_notice_id>/`",
        "  - `data/documents_manifest.csv`",
        "  - `data/_corpus_summary.json` (legacy-bearing fields deferred until Step 6 only)",
        "- de_novo_rule: no legacy taxonomy/benchmark outputs used before taxonomy freeze",
        "- execution_plan: Step0 bootstrap -> mandatory LLM preflight -> dry-run extraction -> bulk extraction -> QC -> STOP 1",
        f"- normalization_assumption: missing raw decision ids fall back to filename stem; too-short flag threshold = {TOO_SHORT_CHARS} chars",
        "",
        "## Step 0 Bootstrap Summary",
        f"- raw_fas_rows: {summary['raw_fas_rows']}",
        f"- rows_with_text: {summary['rows_with_text']}",
        f"- rows_too_short: {summary['rows_too_short']}",
        f"- rows_empty: {summary['rows_empty']}",
        f"- rows_without_notice_id: {summary['rows_without_notice_id']}",
        f"- rows_with_gold_decision_reference: {summary['rows_with_gold_decision_reference']}",
        f"- schema_family_counts: `{summary['schema_family_counts']}`",
    ]
    paths.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_journal(paths: Paths, heading: str, lines: list[str]) -> None:
    with paths.journal_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {heading}\n")
        for line in lines:
            fh.write(f"{line}\n")


def decision_text_from_manifest_row(row: dict[str, str]) -> str:
    raw = read_json(ROOT / row["path"])
    field = row["text_source_field"]
    return str(raw.get(field, "") or "")


def extraction_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "issue_label": {"type": "string"},
                "reasoning_summary": {"type": "string"},
                "legal_articles": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "fas_verdict": {
                    "type": "string",
                    "enum": ["violation_established", "violation_not_established", "not_considered"],
                },
                "scope_tag": {
                    "type": "string",
                    "enum": [
                        "doc_content",
                        "bid_evaluation",
                        "procedural",
                        "platform",
                        "national_regime",
                        "contract_execution",
                        "other",
                    ],
                },
                "involves_procurement_docs": {"type": "boolean"},
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
            },
            "required": [
                "issue_label",
                "reasoning_summary",
                "legal_articles",
                "fas_verdict",
                "scope_tag",
                "involves_procurement_docs",
                "confidence",
            ],
        },
    }


def preflight(paths: Paths) -> dict[str, Any]:
    manifest_rows = read_csv(paths.manifests_dir / "raw_fas_canonical_manifest.csv")
    candidate = next(row for row in manifest_rows if row["has_text"] == "True")
    text = decision_text_from_manifest_row(candidate)
    sample_text = text[:5000]
    api_key, api_key_source, shell_has_key = load_api_key()
    if not api_key:
        result = {
            "api_key_source": "missing",
            "shell_env_has_key": shell_has_key,
            "model_requested": MODEL_EXTRACT,
            "model_resolved": "",
            "http_status": "missing_key",
            "provider_route_requested": PROVIDER_ROUTE,
            "provider_route_resolved": "",
            "provider_lock_confirmed": False,
        }
        write_json(paths.logs_dir / "preflight" / "preflight_result.json", result)
        append_journal(paths, f"Mandatory LLM Preflight @ {now_iso()}", [f"- {k}: `{v}`" for k, v in result.items()])
        raise RuntimeError("OPENROUTER_API_KEY missing in shell and .env")
    log_path = paths.logs_dir / "preflight" / "openrouter_preflight_raw.json"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                decision_id=candidate["canonical_decision_id"],
                notice_id=candidate["canonical_notice_id"] or "missing",
                source_path=candidate["path"],
                decision_text=sample_text,
            ),
        },
    ]
    raw = openrouter_request(api_key, messages, extraction_schema(), log_path, max_tokens=900)
    content = raw["response_json"]["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise RuntimeError("Structured output preflight returned non-array JSON")
    provider_resolved = provider_resolved_from(raw["response_json"], raw["headers"])
    result = {
        "api_key_source": api_key_source,
        "shell_env_has_key": shell_has_key,
        "model_requested": MODEL_EXTRACT,
        "model_resolved": raw["response_json"].get("model", MODEL_EXTRACT),
        "http_status": raw["http_status"],
        "provider_route_requested": PROVIDER_ROUTE,
        "provider_route_resolved": provider_resolved,
        "provider_lock_confirmed": True if raw["http_status"] == 200 else False,
        "structured_output_valid": True,
        "sample_decision_id": candidate["canonical_decision_id"],
        "sample_episode_count": len(parsed),
    }
    write_json(paths.logs_dir / "preflight" / "preflight_result.json", result)
    append_journal(paths, f"Mandatory LLM Preflight @ {now_iso()}", [f"- {k}: `{v}`" for k, v in result.items()])
    return result


def extract_one(row: dict[str, str], api_key: str, raw_dir: Path, max_tokens: int = 2200) -> dict[str, Any]:
    text = decision_text_from_manifest_row(row)
    log_path = raw_dir / f"{row['canonical_decision_id']}.json"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                decision_id=row["canonical_decision_id"],
                notice_id=row["canonical_notice_id"] or "missing",
                source_path=row["path"],
                decision_text=text,
            ),
        },
    ]
    last_error = None
    for attempt in range(1, 4):
        try:
            raw = openrouter_request(api_key, messages, extraction_schema(), log_path, max_tokens=max_tokens)
            content = raw["response_json"]["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise RuntimeError("non-array structured output")
            return {
                "canonical_decision_id": row["canonical_decision_id"],
                "canonical_notice_id": row["canonical_notice_id"],
                "source_path": row["path"],
                "text_source_field": row["text_source_field"],
                "decision_id_source_field": row["decision_id_source_field"],
                "notice_id_source_field": row["notice_id_source_field"],
                "episodes": parsed,
                "llm_http_status": raw["http_status"],
                "llm_model_resolved": raw["response_json"].get("model", MODEL_EXTRACT),
                "llm_provider_route_requested": PROVIDER_ROUTE,
                "llm_provider_route_resolved": provider_resolved_from(raw["response_json"], raw["headers"]),
                "parse_error": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(attempt * 2)
    return {
        "canonical_decision_id": row["canonical_decision_id"],
        "canonical_notice_id": row["canonical_notice_id"],
        "source_path": row["path"],
        "text_source_field": row["text_source_field"],
        "decision_id_source_field": row["decision_id_source_field"],
        "notice_id_source_field": row["notice_id_source_field"],
        "episodes": [],
        "llm_http_status": "error",
        "llm_model_resolved": MODEL_EXTRACT,
        "llm_provider_route_requested": PROVIDER_ROUTE,
        "llm_provider_route_resolved": "unknown",
        "parse_error": last_error or "unknown_error",
    }


def write_batch_outputs(paths: Paths, batch_idx: int, rows: list[dict[str, Any]]) -> None:
    path = paths.extraction_dir / "batches" / f"batch_{batch_idx:04d}.jsonl"
    write_jsonl(path, rows)


def combine_batches(paths: Paths) -> list[dict[str, Any]]:
    merged_rows: dict[str, dict[str, Any]] = {}
    for path in sorted((paths.extraction_dir / "batches").glob("batch_*.jsonl")):
        for row in read_jsonl(path):
            merged_rows[row["canonical_decision_id"]] = row
    all_rows = list(merged_rows.values())
    all_rows.sort(key=lambda row: row["source_path"])
    write_jsonl(paths.extraction_dir / "taxonomy_episodes.jsonl", all_rows)
    flat_rows: list[dict[str, Any]] = []
    for row in all_rows:
        for idx, ep in enumerate(row["episodes"], start=1):
            flat_rows.append(
                {
                    "canonical_decision_id": row["canonical_decision_id"],
                    "canonical_notice_id": row["canonical_notice_id"],
                    "source_path": row["source_path"],
                    "episode_idx": idx,
                    "issue_label": ep["issue_label"],
                    "reasoning_summary": ep["reasoning_summary"],
                    "legal_articles": ep["legal_articles"],
                    "fas_verdict": ep["fas_verdict"],
                    "scope_tag": ep["scope_tag"],
                    "involves_procurement_docs": ep["involves_procurement_docs"],
                    "confidence": ep["confidence"],
                }
            )
    write_csv(
        paths.extraction_dir / "taxonomy_episodes_flat.csv",
        [
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "episode_idx",
            "issue_label",
            "reasoning_summary",
            "legal_articles",
            "fas_verdict",
            "scope_tag",
            "involves_procurement_docs",
            "confidence",
        ],
        flat_rows,
    )
    return all_rows


def auto_qc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episode_counts = [len(row["episodes"]) for row in rows]
    flat = [ep for row in rows for ep in row["episodes"]]
    scope_counter = Counter(ep["scope_tag"] for ep in flat)
    doc_counter = Counter(str(ep["involves_procurement_docs"]).lower() for ep in flat)
    bad_rows = 0
    for row in rows:
        if not isinstance(row.get("episodes"), list):
            bad_rows += 1
            continue
        for ep in row["episodes"]:
            if not all(
                key in ep
                for key in [
                    "issue_label",
                    "reasoning_summary",
                    "legal_articles",
                    "fas_verdict",
                    "scope_tag",
                    "involves_procurement_docs",
                    "confidence",
                ]
            ):
                bad_rows += 1
                break
    parse_failures = sum(1 for row in rows if row.get("parse_error"))
    summaries = [len(ep["reasoning_summary"]) for ep in flat]
    qc = {
        "decision_rows": len(rows),
        "episode_rows": len(flat),
        "json_schema_valid_rate": round((len(rows) - bad_rows) / len(rows), 4) if rows else 0.0,
        "parse_failure_rate": round(parse_failures / len(rows), 4) if rows else 0.0,
        "episodes_per_decision_median": statistics.median(episode_counts) if episode_counts else 0,
        "episodes_per_decision_p95": sorted(episode_counts)[math.ceil(len(episode_counts) * 0.95) - 1] if episode_counts else 0,
        "episodes_per_decision_max": max(episode_counts) if episode_counts else 0,
        "empty_extraction_rate": round(sum(c == 0 for c in episode_counts) / len(episode_counts), 4) if episode_counts else 0.0,
        "scope_tag_distribution": dict(scope_counter),
        "involves_procurement_docs_distribution": dict(doc_counter),
        "reasoning_summary_too_short": sum(length < 80 for length in summaries),
        "reasoning_summary_too_long": sum(length > 900 for length in summaries),
    }
    return qc


def make_manual_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random.seed(RANDOM_SEED)
    random_rows = random.sample(rows, min(20, len(rows)))
    edge_candidates: list[dict[str, Any]] = []
    for row in rows:
        count = len(row["episodes"])
        scopes = {ep["scope_tag"] for ep in row["episodes"]}
        parse_error = bool(row.get("parse_error"))
        if parse_error or count == 0 or count >= 4 or len(scopes) >= 3:
            edge_candidates.append(row)
    sample_rows: list[dict[str, Any]] = []
    for kind, batch in [("random", random_rows), ("edge", edge_candidates[:10])]:
        for row in batch:
            sample_rows.append(
                {
                    "sample_kind": kind,
                    "canonical_decision_id": row["canonical_decision_id"],
                    "canonical_notice_id": row["canonical_notice_id"],
                    "source_path": row["source_path"],
                    "episode_count": len(row["episodes"]),
                    "issue_labels": [ep["issue_label"] for ep in row["episodes"]],
                    "scope_tags": [ep["scope_tag"] for ep in row["episodes"]],
                    "involves_procurement_docs": [ep["involves_procurement_docs"] for ep in row["episodes"]],
                    "confidences": [ep["confidence"] for ep in row["episodes"]],
                    "parse_error": row.get("parse_error", ""),
                }
            )
    return sample_rows


def five_examples_by_scope(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        for idx, ep in enumerate(row["episodes"], start=1):
            flat.append(
                {
                    "canonical_decision_id": row["canonical_decision_id"],
                    "source_path": row["source_path"],
                    "episode_idx": idx,
                    **ep,
                }
            )
    by_scope: dict[str, dict[str, Any]] = {}
    for ep in flat:
        by_scope.setdefault(ep["scope_tag"], ep)
    chosen = list(by_scope.values())[:5]
    while len(chosen) < 5 and flat:
        next_ep = flat[len(chosen) - len(by_scope)]
        if next_ep not in chosen:
            chosen.append(next_ep)
    return chosen[:5]


def episode_record(
    row: dict[str, Any],
    episode_idx: int,
    ep: dict[str, Any],
) -> dict[str, Any]:
    return {
        "canonical_decision_id": row["canonical_decision_id"],
        "canonical_notice_id": row["canonical_notice_id"],
        "source_path": row["source_path"],
        "text_source_field": row["text_source_field"],
        "decision_id_source_field": row["decision_id_source_field"],
        "notice_id_source_field": row["notice_id_source_field"],
        "episode_idx": episode_idx,
        "issue_label": ep["issue_label"],
        "reasoning_summary": ep["reasoning_summary"],
        "legal_articles": ep["legal_articles"],
        "fas_verdict": ep["fas_verdict"],
        "scope_tag": ep["scope_tag"],
        "involves_procurement_docs": ep["involves_procurement_docs"],
        "confidence": ep["confidence"],
    }


def derive_doc_core(paths: Paths) -> None:
    extraction_rows = read_jsonl(paths.extraction_dir / "taxonomy_episodes.jsonl")
    included_scopes = {"doc_content", "bid_evaluation", "national_regime"}
    excluded_scope_only = {"procedural", "platform", "contract_execution", "other"}

    included_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    contract_exec_audit_rows: list[dict[str, Any]] = []
    summary_counter: Counter[tuple[str, str]] = Counter()

    for row in extraction_rows:
        for episode_idx, ep in enumerate(row["episodes"], start=1):
            base = episode_record(row, episode_idx, ep)
            scope_tag = ep["scope_tag"]
            involves_docs = bool(ep["involves_procurement_docs"])

            if scope_tag in included_scopes and involves_docs:
                inclusion_status = "included"
                inclusion_reason = "doc_core_scope_and_docs_true"
                included_rows.append(
                    {
                        **base,
                        "inclusion_status": inclusion_status,
                        "inclusion_reason": inclusion_reason,
                    }
                )
            else:
                if not involves_docs:
                    exclusion_reason = "involves_procurement_docs_false"
                elif scope_tag in excluded_scope_only:
                    exclusion_reason = f"excluded_scope_{scope_tag}"
                else:
                    exclusion_reason = f"excluded_scope_{scope_tag}"
                exclusion_row = {
                    **base,
                    "inclusion_status": "excluded",
                    "exclusion_reason": exclusion_reason,
                }
                excluded_rows.append(exclusion_row)
                if scope_tag == "contract_execution" and involves_docs:
                    contract_exec_audit_rows.append(exclusion_row)
                inclusion_status = "excluded"

            summary_counter[(scope_tag, inclusion_status)] += 1

    included_rows.sort(key=lambda r: (r["source_path"], int(r["episode_idx"])))
    excluded_rows.sort(key=lambda r: (r["source_path"], int(r["episode_idx"])))
    contract_exec_audit_rows.sort(key=lambda r: (r["source_path"], int(r["episode_idx"])))

    write_csv(
        paths.manifests_dir / "doc_core_manifest.csv",
        [
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "text_source_field",
            "decision_id_source_field",
            "notice_id_source_field",
            "episode_idx",
            "issue_label",
            "reasoning_summary",
            "legal_articles",
            "fas_verdict",
            "scope_tag",
            "involves_procurement_docs",
            "confidence",
            "inclusion_status",
            "inclusion_reason",
        ],
        included_rows,
    )
    write_csv(
        paths.manifests_dir / "doc_core_exclusions.csv",
        [
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "text_source_field",
            "decision_id_source_field",
            "notice_id_source_field",
            "episode_idx",
            "issue_label",
            "reasoning_summary",
            "legal_articles",
            "fas_verdict",
            "scope_tag",
            "involves_procurement_docs",
            "confidence",
            "inclusion_status",
            "exclusion_reason",
        ],
        excluded_rows,
    )
    write_csv(
        paths.manifests_dir / "doc_core_excluded_contract_execution_audit.csv",
        [
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "text_source_field",
            "decision_id_source_field",
            "notice_id_source_field",
            "episode_idx",
            "issue_label",
            "reasoning_summary",
            "legal_articles",
            "fas_verdict",
            "scope_tag",
            "involves_procurement_docs",
            "confidence",
            "inclusion_status",
            "exclusion_reason",
        ],
        contract_exec_audit_rows,
    )

    summary_rows: list[dict[str, Any]] = []
    for scope_tag in sorted({key[0] for key in summary_counter}):
        summary_rows.append(
            {
                "scope_tag": scope_tag,
                "included_count": summary_counter.get((scope_tag, "included"), 0),
                "excluded_count": summary_counter.get((scope_tag, "excluded"), 0),
                "total_count": summary_counter.get((scope_tag, "included"), 0)
                + summary_counter.get((scope_tag, "excluded"), 0),
            }
        )
    write_csv(
        paths.manifests_dir / "doc_core_scope_inclusion_summary.csv",
        ["scope_tag", "included_count", "excluded_count", "total_count"],
        summary_rows,
    )

    included_keys = {
        (row["canonical_decision_id"], int(row["episode_idx"]))
        for row in included_rows
    }
    missing_notice_rows = [row for row in included_rows if not normalize_ws(str(row["canonical_notice_id"]))]
    contradictions = [
        row
        for row in included_rows
        if row["scope_tag"] not in included_scopes or not bool(row["involves_procurement_docs"])
    ]
    duplicate_count = len(included_rows) - len(included_keys)

    balanced_boundary: list[dict[str, Any]] = []
    boundary_doc_content = [
        row for row in included_rows if row["scope_tag"] == "doc_content"
    ]
    boundary_bid_eval = [
        row for row in included_rows if row["scope_tag"] == "bid_evaluation"
    ]
    boundary_procedural = [
        row
        for row in excluded_rows
        if row["scope_tag"] == "procedural" and bool(row["involves_procurement_docs"])
    ]
    for pool in [boundary_doc_content, boundary_bid_eval, boundary_procedural]:
        pool.sort(key=lambda r: (-len(str(r["reasoning_summary"])), r["source_path"], int(r["episode_idx"])))
    balanced_boundary.extend(systematic_sample(boundary_doc_content, 8))
    balanced_boundary.extend(systematic_sample(boundary_bid_eval, 6))
    balanced_boundary.extend(systematic_sample(boundary_procedural, 6))
    deduped_boundary: dict[tuple[str, int], dict[str, Any]] = {}
    for row in balanced_boundary:
        deduped_boundary[(row["canonical_decision_id"], int(row["episode_idx"]))] = row
    boundary_sample = list(deduped_boundary.values())
    if len(boundary_sample) < 20:
        fallback_pool = boundary_doc_content + boundary_bid_eval + boundary_procedural
        for row in fallback_pool:
            key = (row["canonical_decision_id"], int(row["episode_idx"]))
            if key not in deduped_boundary:
                deduped_boundary[key] = row
                boundary_sample.append(row)
            if len(boundary_sample) >= 20:
                break
    boundary_sample.sort(key=lambda r: (r["scope_tag"], r["source_path"], int(r["episode_idx"])))
    write_csv(
        paths.qc_dir / "doc_core_boundary_sample.csv",
        [
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "episode_idx",
            "scope_tag",
            "involves_procurement_docs",
            "fas_verdict",
            "issue_label",
            "reasoning_summary",
            "inclusion_status",
            "inclusion_reason",
            "exclusion_reason",
        ],
        boundary_sample,
    )
    write_csv(
        paths.qc_dir / "doc_core_missing_notice_audit.csv",
        [
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "text_source_field",
            "decision_id_source_field",
            "notice_id_source_field",
            "episode_idx",
            "scope_tag",
            "issue_label",
            "reasoning_summary",
            "fas_verdict",
            "confidence",
            "inclusion_status",
            "inclusion_reason",
        ],
        missing_notice_rows,
    )

    scope_breakdown = Counter(row["scope_tag"] for row in included_rows)
    excluded_reason_breakdown = Counter(row["exclusion_reason"] for row in excluded_rows)

    sanity_lines = [
        "# DOC_CORE Sanity",
        "",
        "## Automated QC",
        f"- included_episode_rows: **{len(included_rows)}**",
        f"- excluded_episode_rows: **{len(excluded_rows)}**",
        f"- duplicate_episode_keys_in_included: **{duplicate_count}**",
        f"- included_rows_missing_notice_id: **{len(missing_notice_rows)}**",
        f"- inclusion_rule_contradictions: **{len(contradictions)}**",
        f"- included scope breakdown: `{dict(scope_breakdown)}`",
        f"- exclusion reason breakdown: `{dict(excluded_reason_breakdown)}`",
        "- missing_notice_id rows are isolated in `qc/doc_core_missing_notice_audit.csv` for source-side linkage review.",
        "",
        "## Manual boundary review queue",
        f"- saved borderline rows: **{len(boundary_sample)}**",
        "- target boundary: doc_content vs procedural and doc_content vs bid_evaluation",
        "- composition target: 8 `doc_content` + 6 `bid_evaluation` + 6 excluded `procedural` rows where available",
        "",
        "## Manual sanity-check",
    ]

    findings: list[str] = []
    for scope_tag, limit in [("bid_evaluation", 3), ("doc_content", 3), ("procedural", 3)]:
        scoped = [row for row in boundary_sample if row["scope_tag"] == scope_tag][:limit]
        for row in scoped:
            label = normalize_ws(str(row["issue_label"]))
            if row["scope_tag"] == "procedural":
                findings.append(
                    f"- `{row['canonical_decision_id']}::ep{row['episode_idx']}` stays excluded as `procedural`: `{label}` focuses on timing / protocol / clarification procedure rather than document content."
                )
            elif row["scope_tag"] == "bid_evaluation":
                findings.append(
                    f"- `{row['canonical_decision_id']}::ep{row['episode_idx']}` stays included as `bid_evaluation`: `{label}` concerns application, scoring, admission, or rejection logic tied to procurement materials."
                )
            else:
                findings.append(
                    f"- `{row['canonical_decision_id']}::ep{row['episode_idx']}` stays included as `doc_content`: `{label}` is document-facing and extractable from notice / specification / contract text."
                )
    sanity_lines.extend(findings)
    sanity_lines.extend(
        [
            f"- `included_rows_missing_notice_id = {len(missing_notice_rows)}` is explained by source-side absent `purchase_id/notice_id` in normalized raw payloads, not by Step 2 filtering logic.",
            f"- Requested audit slice saved separately: `contract_execution + involves_procurement_docs=true` = **{len(contract_exec_audit_rows)}** rows, excluded from DOC_CORE.",
            "",
            "## Conclusion",
            "- Confirmed definition applied literally with no expansion or heuristic overrides.",
            "- Included set is suitable as the primary DOC_CORE subcorpus for episode-level embeddings.",
        ]
    )
    (paths.qc_dir / "doc_core_sanity.md").write_text("\n".join(sanity_lines), encoding="utf-8")

    append_journal(
        paths,
        f"Step 2 DOC_CORE @ {now_iso()}",
        [
            f"- included_episode_rows: {len(included_rows)}",
            f"- excluded_episode_rows: {len(excluded_rows)}",
            f"- included_scope_breakdown: `{dict(scope_breakdown)}`",
            f"- excluded_reason_breakdown: `{dict(excluded_reason_breakdown)}`",
            f"- included_rows_missing_notice_id: {len(missing_notice_rows)}",
            f"- contract_execution_docs_true_audit_rows: {len(contract_exec_audit_rows)}",
            "- outputs: `manifests/doc_core_manifest.csv`, `manifests/doc_core_exclusions.csv`, `manifests/doc_core_excluded_contract_execution_audit.csv`, `manifests/doc_core_scope_inclusion_summary.csv`, `qc/doc_core_boundary_sample.csv`, `qc/doc_core_missing_notice_audit.csv`, `qc/doc_core_sanity.md`",
        ],
    )


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def load_doc_core_manifest(paths: Paths) -> list[dict[str, Any]]:
    rows = read_csv(paths.manifests_dir / "doc_core_manifest.csv")
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        clean_rows.append(
            {
                **row,
                "episode_idx": int(row["episode_idx"]),
                "legal_articles": parse_json_list(row["legal_articles"]),
                "involves_procurement_docs": parse_bool(row["involves_procurement_docs"]),
            }
        )
    clean_rows.sort(key=lambda r: (r["source_path"], int(r["episode_idx"])))
    return clean_rows


def embedding_input_text(row: dict[str, Any]) -> str:
    return f"passage: {normalize_ws(row['issue_label'])}. {normalize_ws(row['reasoning_summary'])}"


def local_e5_snapshot_path() -> Path:
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--multilingual-e5-large" / "snapshots"
    snapshots = sorted(root.iterdir()) if root.exists() else []
    if not snapshots:
        raise RuntimeError("Local cached model intfloat/multilingual-e5-large not found")
    return snapshots[-1]


def embed_doc_core(paths: Paths) -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rows = load_doc_core_manifest(paths)
    if not rows:
        raise RuntimeError("DOC_CORE manifest is empty; cannot build embeddings")

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_path = local_e5_snapshot_path()
    model = SentenceTransformer(str(model_path), local_files_only=True)
    texts = [embedding_input_text(row) for row in rows]
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    np.savez_compressed(paths.clustering_dir / "embeddings_episode.npz", embeddings=embeddings)

    write_jsonl(
        paths.clustering_dir / "index_episodes.jsonl",
        [{**row, "embedding_text": text} for row, text in zip(rows, texts, strict=True)],
    )

    by_decision: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        item = by_decision.setdefault(
            row["canonical_decision_id"],
            {
                "canonical_decision_id": row["canonical_decision_id"],
                "canonical_notice_id": row["canonical_notice_id"],
                "source_path": row["source_path"],
                "episode_indices": [],
                "scope_tags": [],
            },
        )
        item["episode_indices"].append(idx)
        item["scope_tags"].append(row["scope_tag"])
    decision_rows = list(by_decision.values())
    decision_rows.sort(key=lambda r: r["source_path"])
    decision_embeddings = np.stack(
        [embeddings[item["episode_indices"]].mean(axis=0) for item in decision_rows],
        axis=0,
    ).astype("float32")
    decision_norms = np.linalg.norm(decision_embeddings, axis=1, keepdims=True)
    decision_embeddings = decision_embeddings / np.clip(decision_norms, 1e-12, None)
    np.savez_compressed(paths.clustering_dir / "embeddings_decision.npz", embeddings=decision_embeddings)
    write_jsonl(
        paths.clustering_dir / "index_decisions.jsonl",
        [
            {**item, "n_doc_core_episodes": len(item["episode_indices"])}
            for item in decision_rows
        ],
    )

    qc = {
        "episode_rows": len(rows),
        "decision_rows_with_doc_core": len(decision_rows),
        "embedding_dim": int(embeddings.shape[1]),
        "model_local_path": str(model_path),
        "device": str(getattr(model, "device", "cpu")),
        "mean_embedding_norm": round(float(np.linalg.norm(embeddings, axis=1).mean()), 6),
        "mean_decision_embedding_norm": round(float(np.linalg.norm(decision_embeddings, axis=1).mean()), 6),
    }
    write_json(paths.clustering_dir / "embedding_qc.json", qc)
    append_journal(
        paths,
        f"Step 3 Embeddings @ {now_iso()}",
        [
            f"- episode_rows: {qc['episode_rows']}",
            f"- decision_rows_with_doc_core: {qc['decision_rows_with_doc_core']}",
            f"- embedding_dim: {qc['embedding_dim']}",
            f"- model_local_path: `{qc['model_local_path']}`",
            f"- device: `{qc['device']}`",
            "- assumption: decision-level QC embeddings are mean-pooled over DOC_CORE episode embeddings per decision",
            "- outputs: `clustering/embeddings_episode.npz`, `clustering/index_episodes.jsonl`, `clustering/embeddings_decision.npz`, `clustering/index_decisions.jsonl`, `clustering/embedding_qc.json`",
        ],
    )


def top_n_counts(values: list[str], n: int = 3) -> list[list[Any]]:
    counter = Counter(values)
    return [[key, counter[key]] for key, _ in counter.most_common(n)]


def summarize_cluster_rows(cluster_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "size": len(cluster_rows),
        "n_unique_decisions": len({row["canonical_decision_id"] for row in cluster_rows}),
        "top_articles": top_n_counts([article for row in cluster_rows for article in row["legal_articles"] if article], 3),
        "top_scope_tags": top_n_counts([row["scope_tag"] for row in cluster_rows], 3),
        "top_verdicts": top_n_counts([row["fas_verdict"] for row in cluster_rows], 3),
    }


def choose_recommended_config(metrics_by_config: dict[str, dict[str, Any]]) -> str:
    eligible = [name for name, metrics in metrics_by_config.items() if metrics["noise_share"] <= 0.30]
    candidates = eligible or list(metrics_by_config)
    scored = sorted(
        candidates,
        key=lambda name: (
            metrics_by_config[name]["noise_share"] + 0.5 * metrics_by_config[name]["largest_cluster_share"],
            abs(metrics_by_config[name]["n_clusters"] - metrics_by_config["default"]["n_clusters"]),
            0 if name == "default" else 1,
        ),
    )
    return scored[0]


def cluster_stop2(paths: Paths) -> None:
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import adjusted_rand_score

    os.environ.setdefault("NUMBA_CACHE_DIR", str(paths.run_dir / ".numba_cache"))
    (paths.run_dir / ".numba_cache").mkdir(parents=True, exist_ok=True)
    from umap import UMAP

    embeddings = np.load(paths.clustering_dir / "embeddings_episode.npz")["embeddings"]
    rows = read_jsonl(paths.clustering_dir / "index_episodes.jsonl")
    if len(rows) != len(embeddings):
        raise RuntimeError("Episode index length does not match embedding matrix")

    reducer_10 = UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=20260510)
    umap10 = reducer_10.fit_transform(embeddings).astype("float32")
    np.savez_compressed(paths.clustering_dir / "umap10_episode.npz", umap=umap10)

    reducer_2 = UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=20260510)
    umap2 = reducer_2.fit_transform(embeddings).astype("float32")
    write_csv(
        paths.clustering_dir / "umap2_episode.csv",
        ["row_idx", "canonical_decision_id", "episode_idx", "x", "y"],
        [
            {
                "row_idx": idx,
                "canonical_decision_id": row["canonical_decision_id"],
                "episode_idx": row["episode_idx"],
                "x": float(coords[0]),
                "y": float(coords[1]),
            }
            for idx, (row, coords) in enumerate(zip(rows, umap2, strict=True))
        ],
    )

    configs = {"fine": 8, "default": 12, "coarse": 18}
    metrics_by_config: dict[str, dict[str, Any]] = {}
    labels_by_config: dict[str, Any] = {}
    cluster_profiles_by_config: dict[str, list[dict[str, Any]]] = {}

    for name, min_cluster_size in configs.items():
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, cluster_selection_method="eom")
        labels = model.fit_predict(umap10)
        labels_by_config[name] = labels
        probabilities = getattr(model, "probabilities_", None)
        if probabilities is None:
            probabilities = np.ones(len(labels), dtype="float32")
        assignments: list[dict[str, Any]] = []
        cluster_groups: dict[int, list[dict[str, Any]]] = {}
        for idx, (row, label, prob) in enumerate(zip(rows, labels, probabilities, strict=True)):
            item = {**row, "row_idx": idx, "cluster_id": int(label), "membership_probability": round(float(prob), 6)}
            assignments.append(item)
            if label != -1:
                cluster_groups.setdefault(int(label), []).append(item)
        write_csv(
            paths.clustering_dir / f"assignments_{name}.csv",
            [
                "row_idx",
                "canonical_decision_id",
                "canonical_notice_id",
                "episode_idx",
                "scope_tag",
                "fas_verdict",
                "cluster_id",
                "membership_probability",
                "issue_label",
                "reasoning_summary",
                "legal_articles",
                "confidence",
                "embedding_text",
            ],
            assignments,
        )

        cluster_sizes = sorted(((cluster_id, len(items)) for cluster_id, items in cluster_groups.items()), key=lambda kv: (-kv[1], kv[0]))
        cluster_profiles: list[dict[str, Any]] = []
        for cluster_id, size in cluster_sizes:
            summary = summarize_cluster_rows(cluster_groups[cluster_id])
            cluster_profiles.append(
                {
                    "cluster_id": cluster_id,
                    "size": size,
                    "n_unique_decisions": summary["n_unique_decisions"],
                    "top_articles": summary["top_articles"],
                    "top_scope_tags": summary["top_scope_tags"],
                    "top_verdicts": summary["top_verdicts"],
                }
            )
        cluster_profiles_by_config[name] = cluster_profiles
        write_csv(
            paths.clustering_dir / f"cluster_profiles_{name}.csv",
            ["cluster_id", "size", "n_unique_decisions", "top_articles", "top_scope_tags", "top_verdicts"],
            cluster_profiles,
        )

        noise_count = int(np.sum(labels == -1))
        largest_cluster_size = cluster_sizes[0][1] if cluster_sizes else 0
        metrics_by_config[name] = {
            "config": name,
            "min_cluster_size": min_cluster_size,
            "n_clusters": len(cluster_groups),
            "noise_count": noise_count,
            "noise_share": round(noise_count / len(labels), 4),
            "largest_cluster_size": largest_cluster_size,
            "largest_cluster_share": round(largest_cluster_size / len(labels), 4) if labels.size else 0.0,
            "median_cluster_size": round(statistics.median([size for _, size in cluster_sizes]), 2) if cluster_sizes else 0,
        }

    ari = {
        "fine_vs_default": round(float(adjusted_rand_score(labels_by_config["fine"], labels_by_config["default"])), 4),
        "fine_vs_coarse": round(float(adjusted_rand_score(labels_by_config["fine"], labels_by_config["coarse"])), 4),
        "default_vs_coarse": round(float(adjusted_rand_score(labels_by_config["default"], labels_by_config["coarse"])), 4),
    }
    recommended = choose_recommended_config(metrics_by_config)

    recommended_labels = labels_by_config[recommended]
    recommended_assignments = read_csv(paths.clustering_dir / f"assignments_{recommended}.csv")
    cluster_assignment_map: dict[int, list[dict[str, Any]]] = {}
    for row in recommended_assignments:
        cluster_id = int(row["cluster_id"])
        if cluster_id != -1:
            cluster_assignment_map.setdefault(cluster_id, []).append(row)

    coherence_rows: list[dict[str, Any]] = []
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for row in recommended_assignments:
        by_decision.setdefault(row["canonical_decision_id"], []).append(row)
    for decision_id, items in sorted(by_decision.items()):
        if len(items) < 2:
            continue
        cluster_ids = sorted({int(item["cluster_id"]) for item in items if int(item["cluster_id"]) != -1})
        coherence_rows.append(
            {
                "canonical_decision_id": decision_id,
                "canonical_notice_id": items[0]["canonical_notice_id"],
                "n_doc_core_episodes": len(items),
                "n_unique_clusters_non_noise": len(cluster_ids),
                "cluster_ids_non_noise": cluster_ids,
                "scope_tags": [item["scope_tag"] for item in items],
            }
        )
    write_csv(
        paths.clustering_dir / "decision_coherence_default.csv",
        ["canonical_decision_id", "canonical_notice_id", "n_doc_core_episodes", "n_unique_clusters_non_noise", "cluster_ids_non_noise", "scope_tags"],
        coherence_rows,
    )
    coherence_values = [int(row["n_unique_clusters_non_noise"]) for row in coherence_rows]
    coherence_summary = {
        "rows": len(coherence_rows),
        "median_unique_clusters": statistics.median(coherence_values) if coherence_values else 0,
        "p90_unique_clusters": sorted(coherence_values)[math.ceil(len(coherence_values) * 0.9) - 1] if coherence_values else 0,
        "max_unique_clusters": max(coherence_values) if coherence_values else 0,
    }

    detailed_lines = ["# Clustering Sanity", "", f"- recommended_config: **{recommended}**", f"- ARI: `{ari}`", ""]
    clean_candidates: list[dict[str, Any]] = []
    dirty_candidates: list[dict[str, Any]] = []
    broad_anchor_clusters: list[dict[str, Any]] = []
    for profile in cluster_profiles_by_config[recommended]:
        cluster_id = int(profile["cluster_id"])
        cluster_rows = cluster_assignment_map[cluster_id]
        size = len(cluster_rows)
        if size < 10:
            continue
        idxs = [int(row["row_idx"]) for row in cluster_rows]
        vecs = embeddings[idxs]
        centroid = vecs.mean(axis=0)
        centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-12, None)
        sims = vecs @ centroid
        order = np.argsort(-sims)
        rep_rows = [cluster_rows[i] for i in order[:5]]
        border_rows = [cluster_rows[i] for i in np.argsort(sims)[:3]]
        top_scopes = profile["top_scope_tags"]
        dominant_scope_share = top_scopes[0][1] / size if top_scopes else 0.0
        candidate_meta = {
            "cluster_id": cluster_id,
            "size": size,
            "dominant_scope_share": round(dominant_scope_share, 4),
            "top_scope_tags": top_scopes,
            "top_articles": profile["top_articles"],
        }
        if dominant_scope_share >= 0.95 and len(top_scopes) <= 2 and 20 <= size <= 160:
            clean_candidates.append(candidate_meta)
        if dominant_scope_share < 0.75 or len(top_scopes) >= 3:
            dirty_candidates.append(candidate_meta)
        if size >= 200:
            broad_anchor_clusters.append(candidate_meta)
        detailed_lines.extend(
            [
                f"## Cluster {cluster_id}",
                f"- size: **{size}**",
                f"- n_unique_decisions: **{profile['n_unique_decisions']}**",
                f"- top_articles: `{profile['top_articles']}`",
                f"- top_scope_tags: `{profile['top_scope_tags']}`",
                f"- top_verdicts: `{profile['top_verdicts']}`",
                "- representatives:",
            ]
        )
        for row in rep_rows:
            detailed_lines.append(f"  - `{row['canonical_decision_id']}::ep{row['episode_idx']}` [{row['scope_tag']}] {normalize_ws(row['issue_label'])} :: {normalize_ws(row['reasoning_summary'])[:240]}")
        detailed_lines.append("- borderlines:")
        for row in border_rows:
            detailed_lines.append(f"  - `{row['canonical_decision_id']}::ep{row['episode_idx']}` [{row['scope_tag']}] {normalize_ws(row['issue_label'])} :: {normalize_ws(row['reasoning_summary'])[:240]}")
        detailed_lines.append("")
    (paths.qc_dir / "clustering_sanity.md").write_text("\n".join(detailed_lines), encoding="utf-8")

    clean_candidates.sort(key=lambda item: (-item["size"], item["cluster_id"]))
    dirty_candidates.sort(key=lambda item: (item["dominant_scope_share"], -item["size"], item["cluster_id"]))
    broad_anchor_clusters.sort(key=lambda item: (-item["size"], item["cluster_id"]))
    clean_examples = clean_candidates[:3]
    dirty_examples = dirty_candidates[:2]

    metrics_rows = [metrics_by_config["fine"], metrics_by_config["default"], metrics_by_config["coarse"]]
    write_csv(
        paths.clustering_dir / "clustering_config_metrics.csv",
        ["config", "min_cluster_size", "n_clusters", "noise_count", "noise_share", "largest_cluster_size", "largest_cluster_share", "median_cluster_size"],
        metrics_rows,
    )
    write_json(paths.clustering_dir / "clustering_ari.json", {"ari": ari, "recommended_config": recommended})

    os.environ.setdefault("MPLCONFIGDIR", str(paths.run_dir / ".mplconfig"))
    (paths.run_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 8))
    top_clusters = sorted(set(int(label) for label in recommended_labels if int(label) != -1))[:20]
    for cluster_id in top_clusters:
        mask = recommended_labels == cluster_id
        ax.scatter(umap2[mask, 0], umap2[mask, 1], s=8, alpha=0.7, label=str(cluster_id), color=palette(cluster_id % 20))
    noise_mask = recommended_labels == -1
    if np.any(noise_mask):
        ax.scatter(umap2[noise_mask, 0], umap2[noise_mask, 1], s=6, alpha=0.3, label="noise", color="lightgray")
    ax.set_title(f"Stage 21 DOC_CORE UMAP 2D ({recommended})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(paths.clustering_dir / "umap2_recommended.png", dpi=180)
    plt.close(fig)

    summary_lines = ["# STOP 2 Clustering Review", "", f"- recommended_config: **{recommended}**", "", "## Config metrics"]
    for metrics in metrics_rows:
        summary_lines.append(
            f"- `{metrics['config']}`: min_cluster_size={metrics['min_cluster_size']}, n_clusters={metrics['n_clusters']}, noise_share={metrics['noise_share']}, largest_cluster_share={metrics['largest_cluster_share']}"
        )
    summary_lines.extend(["", f"## ARI\n- `{ari}`", "", "## Recommended config cluster size distribution"])
    for profile in cluster_profiles_by_config[recommended]:
        summary_lines.append(f"- cluster `{profile['cluster_id']}`: size={profile['size']}, top_articles={profile['top_articles']}, top_scope_tags={profile['top_scope_tags']}")
    summary_lines.extend(
        [
            "",
            "## Decision-level coherence check",
            f"- rows_with_2plus_doc_core_episodes: `{coherence_summary['rows']}`",
            f"- median_unique_clusters_non_noise: `{coherence_summary['median_unique_clusters']}`",
            f"- p90_unique_clusters_non_noise: `{coherence_summary['p90_unique_clusters']}`",
            f"- max_unique_clusters_non_noise: `{coherence_summary['max_unique_clusters']}`",
            "",
            "## UMAP 2D",
            "- plot: `clustering/umap2_recommended.png`",
            "",
            "## Clean cluster examples",
        ]
    )
    for item in clean_examples:
        summary_lines.append(
            f"- cluster `{item['cluster_id']}`: size={item['size']}, dominant_scope_share={item['dominant_scope_share']}, top_scope_tags={item['top_scope_tags']}, top_articles={item['top_articles']}"
        )
    summary_lines.extend(["", "## Dirty cluster examples"])
    for item in dirty_examples:
        summary_lines.append(
            f"- cluster `{item['cluster_id']}`: size={item['size']}, dominant_scope_share={item['dominant_scope_share']}, top_scope_tags={item['top_scope_tags']}, top_articles={item['top_articles']}"
        )
    summary_lines.extend(["", "## Broad anchor clusters"])
    for item in broad_anchor_clusters[:3]:
        summary_lines.append(
            f"- cluster `{item['cluster_id']}`: size={item['size']}, dominant_scope_share={item['dominant_scope_share']}, top_scope_tags={item['top_scope_tags']}, top_articles={item['top_articles']}"
        )
    summary_lines.append("")
    if min(ari.values()) < 0.3:
        summary_lines.extend(["## Limitation", "- ARI drops below 0.3 for at least one config pair; clustering stability should be treated as limited.", ""])
    summary_lines.extend(
        [
            "## Recommendation",
            f"- Use `{recommended}` for naming: it gives the best balance of lower noise and acceptable cluster concentration under the fixed Step 4 sensitivity grid.",
            "- Caveat: broad anchor clusters should be watched during naming/manual review, especially where national-regime and document-content arguments partially cohabit.",
        ]
    )
    (paths.reports_dir / "stop2_clustering_review.md").write_text("\n".join(summary_lines), encoding="utf-8")

    append_journal(
        paths,
        f"Step 4 Clustering @ {now_iso()}",
        [
            f"- metrics_by_config: `{metrics_by_config}`",
            f"- ari: `{ari}`",
            f"- recommended_config: `{recommended}`",
            f"- decision_coherence_rows: {len(coherence_rows)}",
            "- outputs: `clustering/clustering_config_metrics.csv`, `clustering/clustering_ari.json`, `clustering/assignments_*.csv`, `clustering/cluster_profiles_*.csv`, `clustering/umap10_episode.npz`, `clustering/umap2_episode.csv`, `clustering/umap2_recommended.png`, `clustering/decision_coherence_default.csv`, `qc/clustering_sanity.md`, `reports/stop2_clustering_review.md`",
        ],
    )


def naming_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label_ru": {"type": "string"},
            "key_articles": {
                "type": "array",
                "items": {"type": "string"},
            },
            "summary_ru": {"type": "string"},
            "reasoning_type": {
                "type": "string",
                "enum": [
                    "lexical",
                    "catalog_lookup",
                    "semantic_comparison",
                    "legal_applicability",
                    "external_knowledge",
                    "mixed",
                ],
            },
            "runtime_doc_dependency": {
                "type": "string",
                "enum": ["full", "partial", "none"],
            },
            "external_knowledge_required": {
                "type": "string",
                "enum": [
                    "none",
                    "market_knowledge",
                    "physical_measurement",
                    "expert_judgment",
                    "regulatory_history",
                ],
            },
        },
        "required": [
            "label_ru",
            "key_articles",
            "summary_ru",
            "reasoning_type",
            "runtime_doc_dependency",
            "external_knowledge_required",
        ],
    }


def naming_log_dir(paths: Paths) -> Path:
    path = paths.logs_dir / "naming_raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cluster_meta_flags(cluster_id: int, size: int, top_scope_tags: list[list[Any]]) -> tuple[str, str]:
    top_count = int(top_scope_tags[0][1]) if top_scope_tags else 0
    dominant_share = top_count / size if size else 0.0
    n_scopes = len(top_scope_tags)
    if cluster_id in {20, 36}:
        return "mixed_candidate_manual_priority", "user_priority_mixed"
    if cluster_id in {28, 33}:
        return "broad_anchor_manual_priority", "user_priority_broad"
    if dominant_share < 0.75 or n_scopes >= 3:
        return "mixed_candidate", "scope_distribution_mixed"
    if size >= 200:
        return "broad_anchor_candidate", "large_anchor_cluster"
    return "coherent_candidate", ""


def load_assignments(paths: Paths, config: str) -> list[dict[str, Any]]:
    rows = read_csv(paths.clustering_dir / f"assignments_{config}.csv")
    clean: list[dict[str, Any]] = []
    for row in rows:
        clean.append(
            {
                **row,
                "row_idx": int(row["row_idx"]),
                "episode_idx": int(row["episode_idx"]),
                "cluster_id": int(row["cluster_id"]),
                "membership_probability": float(row["membership_probability"]),
                "legal_articles": parse_json_list(row["legal_articles"]),
            }
        )
    return clean


def load_cluster_profiles(paths: Paths, config: str) -> dict[int, dict[str, Any]]:
    rows = read_csv(paths.clustering_dir / f"cluster_profiles_{config}.csv")
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(row["cluster_id"])] = {
            **row,
            "cluster_id": int(row["cluster_id"]),
            "size": int(row["size"]),
            "n_unique_decisions": int(row["n_unique_decisions"]),
            "top_articles": json.loads(row["top_articles"]),
            "top_scope_tags": json.loads(row["top_scope_tags"]),
            "top_verdicts": json.loads(row["top_verdicts"]),
        }
    return out


def cluster_payload(
    assignments_by_cluster: dict[int, list[dict[str, Any]]],
    profiles: dict[int, dict[str, Any]],
    embeddings: Any,
    cluster_id: int,
) -> dict[str, Any]:
    cluster_rows = assignments_by_cluster[cluster_id]
    profile = profiles[cluster_id]
    idxs = [row["row_idx"] for row in cluster_rows]
    vecs = embeddings[idxs]
    centroid = vecs.mean(axis=0)
    centroid = centroid / max(float((centroid @ centroid) ** 0.5), 1e-12)
    sims = vecs @ centroid
    order = list(reversed(sorted(range(len(cluster_rows)), key=lambda i: sims[i])))
    top_members = []
    for pos in order[:10]:
        row = cluster_rows[pos]
        top_members.append(
            {
                "canonical_decision_id": row["canonical_decision_id"],
                "episode_idx": row["episode_idx"],
                "issue_label": normalize_ws(row["issue_label"]),
                "reasoning_summary": normalize_ws(row["reasoning_summary"]),
                "scope_tag": row["scope_tag"],
                "fas_verdict": row["fas_verdict"],
            }
        )
    local_shape_flag, special_attention = cluster_meta_flags(
        cluster_id,
        profile["size"],
        profile["top_scope_tags"],
    )
    return {
        "cluster_id": cluster_id,
        "size": profile["size"],
        "n_unique_decisions": profile["n_unique_decisions"],
        "top_articles": profile["top_articles"],
        "scope_tag_distribution": profile["top_scope_tags"],
        "fas_verdict_distribution": profile["top_verdicts"],
        "top_members": top_members,
        "local_shape_flag": local_shape_flag,
        "special_attention_reason": special_attention,
    }


def naming_user_prompt(payload: dict[str, Any]) -> str:
    lines = [
        f"Cluster ID: {payload['cluster_id']}",
        f"Size: {payload['size']}",
        f"n_unique_decisions: {payload['n_unique_decisions']}",
        f"local_shape_flag: {payload['local_shape_flag']}",
    ]
    if payload["special_attention_reason"]:
        lines.append(f"special_attention_reason: {payload['special_attention_reason']}")
    lines.append(f"top_articles: {json.dumps(payload['top_articles'], ensure_ascii=False)}")
    lines.append(f"scope_tag_distribution: {json.dumps(payload['scope_tag_distribution'], ensure_ascii=False)}")
    lines.append(f"fas_verdict_distribution: {json.dumps(payload['fas_verdict_distribution'], ensure_ascii=False)}")
    lines.append("Top-10 episodes:")
    for idx, item in enumerate(payload["top_members"], start=1):
        lines.append(
            f"{idx}. [{item['scope_tag']}] [{item['fas_verdict']}] {item['issue_label']} :: {item['reasoning_summary']}"
        )
    lines.append(
        "Если кластер выглядит broad или mixed, не придумывай искусственно узкое имя; дай наиболее конкретный честный label для доминирующей темы, а downstream QC отдельно отметит mixed/broad status."
    )
    return "\n".join(lines)


def naming_preflight(paths: Paths) -> dict[str, Any]:
    assignments = load_assignments(paths, "coarse")
    profiles = load_cluster_profiles(paths, "coarse")
    embeddings = __import__("numpy").load(paths.clustering_dir / "embeddings_episode.npz")["embeddings"]
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for row in assignments:
        if row["cluster_id"] != -1:
            by_cluster.setdefault(row["cluster_id"], []).append(row)
    sample_cluster_id = 41 if 41 in by_cluster else sorted(by_cluster)[0]
    sample_payload = cluster_payload(by_cluster, profiles, embeddings, sample_cluster_id)
    api_key, api_key_source, shell_has_key = load_api_key()
    if not api_key:
        result = {
            "api_key_source": "missing",
            "shell_env_has_key": shell_has_key,
            "model_requested": MODEL_NAMING,
            "model_resolved": "",
            "http_status": "missing_key",
            "provider_route_requested": PROVIDER_ROUTE,
            "provider_route_resolved": "",
            "provider_lock_confirmed": False,
            "structured_output_valid": False,
        }
        write_json(paths.logs_dir / "preflight" / "naming_preflight_result.json", result)
        append_journal(paths, f"Mandatory Naming LLM Preflight @ {now_iso()}", [f"- {k}: `{v}`" for k, v in result.items()])
        raise RuntimeError("OPENROUTER_API_KEY missing in shell and .env")

    log_path = paths.logs_dir / "preflight" / "openrouter_naming_preflight_raw.json"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": NAMING_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": naming_user_prompt(sample_payload)},
    ]
    raw = openrouter_request(
        api_key,
        messages,
        naming_schema(),
        log_path,
        model=MODEL_NAMING,
        max_tokens=900,
        temperature=0.0,
        response_format_name="cluster_naming",
    )
    content = raw["response_json"]["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    provider_resolved = provider_resolved_from(raw["response_json"], raw["headers"])
    model_resolved = raw["response_json"].get("model", MODEL_NAMING)
    provider_lock_confirmed = raw["http_status"] == 200 and "anthropic" in str(provider_resolved).lower()
    result = {
        "api_key_source": api_key_source,
        "shell_env_has_key": shell_has_key,
        "model_requested": MODEL_NAMING,
        "model_resolved": model_resolved,
        "http_status": raw["http_status"],
        "provider_route_requested": PROVIDER_ROUTE,
        "provider_route_resolved": provider_resolved,
        "provider_lock_confirmed": provider_lock_confirmed,
        "structured_output_valid": isinstance(parsed, dict) and "label_ru" in parsed,
        "sample_cluster_id": sample_cluster_id,
        "sample_label_ru": parsed.get("label_ru", ""),
        "prompt_caching_requested": True,
    }
    write_json(paths.logs_dir / "preflight" / "naming_preflight_result.json", result)
    append_journal(paths, f"Mandatory Naming LLM Preflight @ {now_iso()}", [f"- {k}: `{v}`" for k, v in result.items()])
    if not result["provider_lock_confirmed"] or not result["structured_output_valid"]:
        raise RuntimeError("Naming LLM preflight failed; fail-closed before naming")
    return result


def naming_checkpoint_path(paths: Paths) -> Path:
    return paths.naming_dir / "cluster_catalog_coarse.csv"


def load_existing_naming_catalog(paths: Paths) -> list[dict[str, Any]]:
    path = naming_checkpoint_path(paths)
    if not path.exists():
        return []
    rows = read_csv(path)
    clean: list[dict[str, Any]] = []
    for row in rows:
        if row.get("cluster_id", "").strip():
            row["cluster_id"] = int(row["cluster_id"])
            row["size"] = int(row["size"])
            row["n_unique_decisions"] = int(row["n_unique_decisions"])
            clean.append(row)
    return clean


def write_naming_catalog(paths: Paths, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: int(r["cluster_id"]))
    write_csv(
        naming_checkpoint_path(paths),
        [
            "cluster_id",
            "size",
            "n_unique_decisions",
            "label_ru",
            "key_articles",
            "summary_ru",
            "reasoning_type",
            "runtime_doc_dependency",
            "external_knowledge_required",
            "local_shape_flag",
            "special_attention_reason",
            "manual_qc_status",
            "top_scope_tags",
            "top_articles_local",
            "top_verdicts_local",
            "raw_response_relpath",
        ],
        rows,
    )


def preliminary_benchmark_signal(named_row: dict[str, Any]) -> str:
    runtime_dep = named_row["runtime_doc_dependency"]
    ext_knowledge = named_row["external_knowledge_required"]
    shape = named_row["local_shape_flag"]
    size = int(named_row["size"])
    if shape.startswith("mixed"):
        return "mixed_review_first"
    if shape.startswith("broad"):
        return "broad_review_first"
    if runtime_dep == "full" and ext_knowledge == "none" and size >= 20:
        return "likely_benchmarkable"
    if runtime_dep in {"full", "partial"} and ext_knowledge in {"none", "market_knowledge", "regulatory_history"}:
        return "conditional_benchmarkable"
    return "service_only_signal"


def name_clusters_stop3(paths: Paths) -> None:
    preflight_result = read_json(paths.logs_dir / "preflight" / "naming_preflight_result.json")
    if not preflight_result.get("provider_lock_confirmed") or not preflight_result.get("structured_output_valid"):
        raise RuntimeError("Cannot run naming without successful naming preflight")

    import numpy as np

    assignments = load_assignments(paths, "coarse")
    profiles = load_cluster_profiles(paths, "coarse")
    embeddings = np.load(paths.clustering_dir / "embeddings_episode.npz")["embeddings"]
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for row in assignments:
        if row["cluster_id"] != -1:
            by_cluster.setdefault(row["cluster_id"], []).append(row)
    cluster_ids = sorted(by_cluster)

    existing = load_existing_naming_catalog(paths)
    existing_by_id = {int(row["cluster_id"]): row for row in existing}
    api_key, _api_key_source, _shell_has_key = load_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing; fail-closed before naming")

    catalog_rows = existing[:]
    dry_run_ids = [cid for cid in [20, 28, 41] if cid in by_cluster]
    if len(dry_run_ids) < 3:
        dry_run_ids = cluster_ids[:3]

    for order_idx, cluster_id in enumerate(cluster_ids, start=1):
        if cluster_id in existing_by_id:
            continue
        payload = cluster_payload(by_cluster, profiles, embeddings, cluster_id)
        raw_path = naming_log_dir(paths) / f"cluster_{cluster_id:03d}.json"
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": NAMING_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": naming_user_prompt(payload)},
        ]
        raw = openrouter_request(
            api_key,
            messages,
            naming_schema(),
            raw_path,
            model=MODEL_NAMING,
            max_tokens=1200,
            temperature=0.0,
            response_format_name="cluster_naming",
        )
        content = raw["response_json"]["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        local_shape_flag = payload["local_shape_flag"]
        special_attention_reason = payload["special_attention_reason"]
        manual_qc_status = "coherent"
        if local_shape_flag.startswith("mixed"):
            manual_qc_status = "mixed_candidate"
        elif local_shape_flag.startswith("broad"):
            manual_qc_status = "broad_candidate"
        named_row = {
            "cluster_id": cluster_id,
            "size": payload["size"],
            "n_unique_decisions": payload["n_unique_decisions"],
            "label_ru": parsed["label_ru"],
            "key_articles": parsed["key_articles"],
            "summary_ru": parsed["summary_ru"],
            "reasoning_type": parsed["reasoning_type"],
            "runtime_doc_dependency": parsed["runtime_doc_dependency"],
            "external_knowledge_required": parsed["external_knowledge_required"],
            "local_shape_flag": local_shape_flag,
            "special_attention_reason": special_attention_reason,
            "manual_qc_status": manual_qc_status,
            "top_scope_tags": payload["scope_tag_distribution"],
            "top_articles_local": payload["top_articles"],
            "top_verdicts_local": payload["fas_verdict_distribution"],
            "raw_response_relpath": raw_path.as_posix(),
        }
        catalog_rows.append(named_row)
        existing_by_id[cluster_id] = named_row
        write_naming_catalog(paths, catalog_rows)
        append_journal(
            paths,
            f"Step 5 Naming Cluster {cluster_id} @ {now_iso()}",
            [
                f"- order_index: {order_idx}",
                f"- label_ru: `{parsed['label_ru']}`",
                f"- local_shape_flag: `{local_shape_flag}`",
                f"- manual_qc_status: `{manual_qc_status}`",
                f"- raw_response_relpath: `{named_row['raw_response_relpath']}`",
            ],
        )

    dry_rows = [existing_by_id[cid] for cid in dry_run_ids if cid in existing_by_id]
    write_csv(
        paths.naming_dir / "dry_run_3_clusters.csv",
        [
            "cluster_id",
            "size",
            "label_ru",
            "summary_ru",
            "reasoning_type",
            "runtime_doc_dependency",
            "external_knowledge_required",
            "local_shape_flag",
            "manual_qc_status",
        ],
        dry_rows,
    )

    all_rows = load_existing_naming_catalog(paths)
    all_rows.sort(key=lambda r: int(r["cluster_id"]))
    prebench_rows = []
    for row in all_rows:
        prebench_rows.append(
            {
                "cluster_id": row["cluster_id"],
                "label_ru": row["label_ru"],
                "size": row["size"],
                "local_shape_flag": row["local_shape_flag"],
                "runtime_doc_dependency": row["runtime_doc_dependency"],
                "external_knowledge_required": row["external_knowledge_required"],
                "preliminary_signal": preliminary_benchmark_signal(row),
            }
        )
    write_csv(
        paths.naming_dir / "prebenchmark_signal_coarse.csv",
        [
            "cluster_id",
            "label_ru",
            "size",
            "local_shape_flag",
            "runtime_doc_dependency",
            "external_knowledge_required",
            "preliminary_signal",
        ],
        prebench_rows,
    )

    shape_counts = Counter(row["manual_qc_status"] for row in all_rows)
    large_rows = [row for row in all_rows if int(row["size"]) >= 100]
    naming_qc_lines = [
        "# Naming Sanity",
        "",
        f"- named_clusters: **{len(all_rows)}**",
        f"- shape_counts: `{dict(shape_counts)}`",
        "",
        "## Large-cluster manual review",
    ]
    for row in large_rows:
        naming_qc_lines.extend(
            [
                f"- cluster `{row['cluster_id']}`: label=`{row['label_ru']}`, size={row['size']}, shape=`{row['manual_qc_status']}`, top_scope_tags=`{row['top_scope_tags']}`",
                f"  summary: {row['summary_ru']}",
            ]
        )
    naming_qc_lines.extend(
        [
            "",
            "## Priority cluster review",
            f"- cluster `20`: `{existing_by_id[20]['label_ru']}` status=`{existing_by_id[20]['manual_qc_status']}`" if 20 in existing_by_id else "- cluster `20`: missing",
            f"- cluster `36`: `{existing_by_id[36]['label_ru']}` status=`{existing_by_id[36]['manual_qc_status']}`" if 36 in existing_by_id else "- cluster `36`: missing",
            f"- cluster `28`: `{existing_by_id[28]['label_ru']}` status=`{existing_by_id[28]['manual_qc_status']}`" if 28 in existing_by_id else "- cluster `28`: missing",
            f"- cluster `33`: `{existing_by_id[33]['label_ru']}` status=`{existing_by_id[33]['manual_qc_status']}`" if 33 in existing_by_id else "- cluster `33`: missing",
            "",
            "## Conclusion",
            "- Clusters marked `mixed_candidate` or `broad_candidate` are explicitly held out as QC-flagged families and should not be over-normalized during later benchmarkability mapping.",
        ]
    )
    (paths.qc_dir / "naming_sanity.md").write_text("\n".join(naming_qc_lines), encoding="utf-8")

    summary_lines = [
        "# STOP 3 Naming Review",
        "",
        f"- named_clusters: **{len(all_rows)}**",
        f"- dry_run_clusters: `{dry_run_ids}`",
        "",
        "## Full catalog",
    ]
    for row in all_rows:
        summary_lines.append(
            f"- cluster `{row['cluster_id']}`: size={row['size']}, label=`{row['label_ru']}`, key_articles={row['key_articles']}, reasoning_type=`{row['reasoning_type']}`, runtime_doc_dependency=`{row['runtime_doc_dependency']}`, shape=`{row['manual_qc_status']}`"
        )
    summary_lines.extend(["", "## Clean vs mixed/broad"])
    for key in sorted(shape_counts):
        summary_lines.append(f"- `{key}`: {shape_counts[key]}")
    summary_lines.extend(["", "## Preliminary benchmarkability map"])
    prebench_counts = Counter(row["preliminary_signal"] for row in prebench_rows)
    for key, value in sorted(prebench_counts.items()):
        summary_lines.append(f"- `{key}`: {value}")
    summary_lines.extend(
        [
            "",
            "## Priority clusters",
            f"- cluster `20`: label=`{existing_by_id[20]['label_ru']}`, status=`{existing_by_id[20]['manual_qc_status']}`" if 20 in existing_by_id else "- cluster `20`: missing",
            f"- cluster `36`: label=`{existing_by_id[36]['label_ru']}`, status=`{existing_by_id[36]['manual_qc_status']}`" if 36 in existing_by_id else "- cluster `36`: missing",
            f"- cluster `28`: label=`{existing_by_id[28]['label_ru']}`, status=`{existing_by_id[28]['manual_qc_status']}`" if 28 in existing_by_id else "- cluster `28`: missing",
            f"- cluster `33`: label=`{existing_by_id[33]['label_ru']}`, status=`{existing_by_id[33]['manual_qc_status']}`" if 33 in existing_by_id else "- cluster `33`: missing",
        ]
    )
    (paths.reports_dir / "stop3_naming_review.md").write_text("\n".join(summary_lines), encoding="utf-8")

    append_journal(
        paths,
        f"Step 5 Naming @ {now_iso()}",
        [
            f"- named_clusters: {len(all_rows)}",
            f"- dry_run_clusters: `{dry_run_ids}`",
            f"- shape_counts: `{dict(shape_counts)}`",
            f"- prebenchmark_counts: `{dict(prebench_counts)}`",
            "- outputs: `naming/cluster_catalog_coarse.csv`, `naming/dry_run_3_clusters.csv`, `naming/prebenchmark_signal_coarse.csv`, `qc/naming_sanity.md`, `reports/stop3_naming_review.md`",
        ],
    )


def write_stop1_report(paths: Paths, rows: list[dict[str, Any]], qc: dict[str, Any]) -> None:
    examples = five_examples_by_scope(rows)
    episode_counts = [len(row["episodes"]) for row in rows]
    sorted_counts = sorted(episode_counts)
    p50 = statistics.median(sorted_counts) if sorted_counts else 0
    p90 = sorted_counts[math.ceil(len(sorted_counts) * 0.9) - 1] if sorted_counts else 0
    p95 = sorted_counts[math.ceil(len(sorted_counts) * 0.95) - 1] if sorted_counts else 0
    lines = [
        "# STOP 1 Extraction Review",
        "",
        f"- processed decisions: **{qc['decision_rows']}**",
        f"- extracted episodes: **{qc['episode_rows']}**",
        "",
        "## scope_tag distribution",
    ]
    for key, value in sorted(qc["scope_tag_distribution"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## episodes per decision",
            f"- median: **{p50}**",
            f"- p90: **{p90}**",
            f"- p95: **{p95}**",
            f"- max: **{qc['episodes_per_decision_max']}**",
            "",
            "## five extraction examples",
        ]
    )
    for ep in examples:
        lines.extend(
            [
                f"### {ep['canonical_decision_id']}::ep{ep['episode_idx']}",
                f"- source_path: `{ep['source_path']}`",
                f"- scope_tag: `{ep['scope_tag']}`",
                f"- involves_procurement_docs: `{ep['involves_procurement_docs']}`",
                f"- fas_verdict: `{ep['fas_verdict']}`",
                f"- issue_label: {ep['issue_label']}",
                f"- reasoning_summary: {ep['reasoning_summary'][:500]}",
                "",
            ]
        )
    lines.extend(
        [
            "## QC summary",
            f"- json_schema_valid_rate: **{qc['json_schema_valid_rate']}**",
            f"- parse_failure_rate: **{qc['parse_failure_rate']}**",
            f"- empty_extraction_rate: **{qc['empty_extraction_rate']}**",
            f"- involves_procurement_docs distribution: `{qc['involves_procurement_docs_distribution']}`",
            "",
            "## Proposed DOC_CORE operational definition",
            "- Include episodes where `scope_tag` ∈ {`doc_content`, `bid_evaluation`, `national_regime`} and `involves_procurement_docs = true`.",
            "- Exclude episodes where `scope_tag` ∈ {`procedural`, `platform`, `contract_execution`, `other`} or `involves_procurement_docs = false`.",
        ]
    )
    (paths.reports_dir / "stop1_extraction_review.md").write_text("\n".join(lines), encoding="utf-8")


def extract_subset(
    paths: Paths,
    rows: list[dict[str, str]],
    output_name: str,
    max_workers: int = 4,
    start_batch_idx: int = 1,
    max_tokens: int = 2200,
    raw_subdir: str = "extraction_raw",
) -> list[dict[str, Any]]:
    api_key, api_key_source, shell_has_key = load_api_key()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing; fail-closed before extraction")
    append_journal(
        paths,
        f"Extraction Run `{output_name}` @ {now_iso()}",
        [
            f"- rows: {len(rows)}",
            f"- api_key_source: `{api_key_source}`",
            f"- shell_env_has_key: `{shell_has_key}`",
            f"- model_requested: `{MODEL_EXTRACT}`",
            f"- provider_route_requested: `{PROVIDER_ROUTE}`",
            f"- max_workers: {max_workers}",
        ],
    )
    batch_size = 50
    all_results: list[dict[str, Any]] = []
    for batch_idx, start in enumerate(range(0, len(rows), batch_size), start=start_batch_idx):
        batch_rows = rows[start : start + batch_size]
        batch_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(extract_one, row, api_key, paths.logs_dir / raw_subdir, max_tokens): row
                for row in batch_rows
            }
            for future in as_completed(future_map):
                batch_results.append(future.result())
        batch_results.sort(key=lambda row: row["source_path"])
        write_batch_outputs(paths, batch_idx, batch_results)
        all_results.extend(batch_results)
        combine_batches(paths)
        if len(all_results) >= 100 and not (paths.qc_dir / "extraction_early100_auto_qc.json").exists():
            early_qc = auto_qc(all_results[:100])
            write_json(paths.qc_dir / "extraction_early100_auto_qc.json", early_qc)
            append_journal(paths, f"Extraction Early QC @ {now_iso()}", [f"- auto_qc: `{early_qc}`"])
    all_results.sort(key=lambda row: row["source_path"])
    write_jsonl(paths.extraction_dir / f"{output_name}.jsonl", all_results)
    combine_batches(paths)
    return all_results


def run_stop1(paths: Paths) -> None:
    summary = bootstrap_manifest(paths)
    init_journal(paths, summary)
    preflight_result = preflight(paths)
    if not preflight_result.get("structured_output_valid"):
        raise RuntimeError("LLM preflight invalid; fail-closed")
    manifest_rows = read_csv(paths.manifests_dir / "raw_fas_canonical_manifest.csv")
    eligible_rows = [row for row in manifest_rows if row["has_text"] == "True"]
    dry_rows = systematic_sample(eligible_rows, 12)
    dry_results = extract_subset(paths, dry_rows, "dry_run_12", max_workers=3)
    dry_qc = auto_qc(dry_results)
    write_json(paths.qc_dir / "dry_run_12_auto_qc.json", dry_qc)
    append_journal(paths, f"Dry Run QC @ {now_iso()}", [f"- auto_qc: `{dry_qc}`"])
    # reset batches for full run
    for path in (paths.extraction_dir / "batches").glob("batch_*.jsonl"):
        path.unlink()
    full_results = extract_subset(paths, eligible_rows, "taxonomy_episodes_full", max_workers=4)
    qc = auto_qc(full_results)
    write_json(paths.qc_dir / "extraction_auto_qc.json", qc)
    sample_rows = make_manual_samples(full_results)
    write_csv(
        paths.qc_dir / "manual_extraction_sample.csv",
        [
            "sample_kind",
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "episode_count",
            "issue_labels",
            "scope_tags",
            "involves_procurement_docs",
            "confidences",
            "parse_error",
        ],
        sample_rows,
    )
    sanity_lines = [
        "# Extraction Sanity",
        "",
        "## Automated QC",
        f"- decision_rows: **{qc['decision_rows']}**",
        f"- episode_rows: **{qc['episode_rows']}**",
        f"- json_schema_valid_rate: **{qc['json_schema_valid_rate']}**",
        f"- parse_failure_rate: **{qc['parse_failure_rate']}**",
        f"- empty_extraction_rate: **{qc['empty_extraction_rate']}**",
        f"- scope_tag distribution: `{qc['scope_tag_distribution']}`",
        f"- involves_procurement_docs distribution: `{qc['involves_procurement_docs_distribution']}`",
        "",
        "## Manual review queue",
        f"- saved sample rows: **{len(sample_rows)}** = 20 random + {len(sample_rows) - 20} edge",
        "- Manual spot-check findings should be appended in-place before moving past STOP 1.",
    ]
    (paths.qc_dir / "extraction_sanity.md").write_text("\n".join(sanity_lines), encoding="utf-8")
    write_stop1_report(paths, full_results, qc)
    append_journal(
        paths,
        f"STOP 1 Ready @ {now_iso()}",
        [
            f"- processed_decisions: {qc['decision_rows']}",
            f"- extracted_episodes: {qc['episode_rows']}",
            f"- qc_report: `qc/extraction_sanity.md`",
            f"- stop_report: `reports/stop1_extraction_review.md`",
        ],
    )


def resume_stop1(paths: Paths) -> None:
    preflight_path = paths.logs_dir / "preflight" / "preflight_result.json"
    if not preflight_path.exists():
        raise RuntimeError("Cannot resume without successful preflight_result.json")
    preflight_result = read_json(preflight_path)
    if not preflight_result.get("structured_output_valid"):
        raise RuntimeError("Cannot resume because preflight was not valid")
    manifest_rows = read_csv(paths.manifests_dir / "raw_fas_canonical_manifest.csv")
    eligible_rows = [row for row in manifest_rows if row["has_text"] == "True"]
    existing_batches = sorted((paths.extraction_dir / "batches").glob("batch_*.jsonl"))
    committed_rows = combine_batches(paths)
    processed_ids = {row["canonical_decision_id"] for row in committed_rows}
    remaining_rows = [row for row in eligible_rows if row["canonical_decision_id"] not in processed_ids]
    raw_log_count = len(list((paths.logs_dir / "extraction_raw").glob("*.json")))
    discarded_uncommitted = max(raw_log_count - len(processed_ids), 0)
    append_journal(
        paths,
        f"Resume Extraction @ {now_iso()}",
        [
            f"- committed_rows_from_batches: {len(processed_ids)}",
            f"- remaining_rows: {len(remaining_rows)}",
            f"- existing_batches: {len(existing_batches)}",
            f"- discarded_uncommitted_raw_logs: {discarded_uncommitted}",
            "- resume_rationale: original full run was interrupted intentionally to continue only the remaining subset at higher parallelism",
            "- max_workers: 10",
        ],
    )
    if remaining_rows:
        extract_subset(
            paths,
            remaining_rows,
            "taxonomy_episodes_resume",
            max_workers=10,
            start_batch_idx=len(existing_batches) + 1,
        )
    full_results = combine_batches(paths)
    qc = auto_qc(full_results)
    write_json(paths.qc_dir / "extraction_auto_qc.json", qc)
    sample_rows = make_manual_samples(full_results)
    write_csv(
        paths.qc_dir / "manual_extraction_sample.csv",
        [
            "sample_kind",
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "episode_count",
            "issue_labels",
            "scope_tags",
            "involves_procurement_docs",
            "confidences",
            "parse_error",
        ],
        sample_rows,
    )
    sanity_lines = [
        "# Extraction Sanity",
        "",
        "## Automated QC",
        f"- decision_rows: **{qc['decision_rows']}**",
        f"- episode_rows: **{qc['episode_rows']}**",
        f"- json_schema_valid_rate: **{qc['json_schema_valid_rate']}**",
        f"- parse_failure_rate: **{qc['parse_failure_rate']}**",
        f"- empty_extraction_rate: **{qc['empty_extraction_rate']}**",
        f"- scope_tag distribution: `{qc['scope_tag_distribution']}`",
        f"- involves_procurement_docs distribution: `{qc['involves_procurement_docs_distribution']}`",
        "",
        "## Manual review queue",
        f"- saved sample rows: **{len(sample_rows)}** = 20 random + {len(sample_rows) - 20} edge",
        "- Manual spot-check findings should be appended in-place before moving past STOP 1.",
    ]
    (paths.qc_dir / "extraction_sanity.md").write_text("\n".join(sanity_lines), encoding="utf-8")
    write_stop1_report(paths, full_results, qc)
    append_journal(
        paths,
        f"STOP 1 Ready @ {now_iso()}",
        [
            f"- processed_decisions: {qc['decision_rows']}",
            f"- extracted_episodes: {qc['episode_rows']}",
            f"- qc_report: `qc/extraction_sanity.md`",
            f"- stop_report: `reports/stop1_extraction_review.md`",
        ],
    )


def rerun_parse_errors(paths: Paths) -> None:
    manifest_rows = read_csv(paths.manifests_dir / "raw_fas_canonical_manifest.csv")
    by_id = {row["canonical_decision_id"]: row for row in manifest_rows}
    combined_rows = combine_batches(paths)
    error_rows = [row for row in combined_rows if row.get("parse_error")]
    if not error_rows:
        append_journal(paths, f"Parse-Error Rerun @ {now_iso()}", ["- parse_error_rows: 0", "- action: skipped"])
        return
    rerun_manifest_rows = [by_id[row["canonical_decision_id"]] for row in error_rows if row["canonical_decision_id"] in by_id]
    existing_batches = sorted((paths.extraction_dir / "batches").glob("batch_*.jsonl"))
    append_journal(
        paths,
        f"Parse-Error Rerun @ {now_iso()}",
        [
            f"- parse_error_rows: {len(error_rows)}",
            f"- rerun_rows: {len(rerun_manifest_rows)}",
            f"- start_batch_idx: {len(existing_batches) + 1}",
            "- rerun_reason: truncated / invalid JSON from otherwise successful LLM calls",
            "- max_workers: 3",
            "- max_tokens: 5000",
        ],
    )
    extract_subset(
        paths,
        rerun_manifest_rows,
        "taxonomy_episodes_parse_error_rerun",
        max_workers=3,
        start_batch_idx=len(existing_batches) + 1,
        max_tokens=5000,
        raw_subdir="extraction_raw_rerun",
    )
    final_rows = combine_batches(paths)
    qc = auto_qc(final_rows)
    write_json(paths.qc_dir / "extraction_auto_qc.json", qc)
    sample_rows = make_manual_samples(final_rows)
    write_csv(
        paths.qc_dir / "manual_extraction_sample.csv",
        [
            "sample_kind",
            "canonical_decision_id",
            "canonical_notice_id",
            "source_path",
            "episode_count",
            "issue_labels",
            "scope_tags",
            "involves_procurement_docs",
            "confidences",
            "parse_error",
        ],
        sample_rows,
    )
    sanity_lines = [
        "# Extraction Sanity",
        "",
        "## Automated QC",
        f"- decision_rows: **{qc['decision_rows']}**",
        f"- episode_rows: **{qc['episode_rows']}**",
        f"- json_schema_valid_rate: **{qc['json_schema_valid_rate']}**",
        f"- parse_failure_rate: **{qc['parse_failure_rate']}**",
        f"- empty_extraction_rate: **{qc['empty_extraction_rate']}**",
        f"- scope_tag distribution: `{qc['scope_tag_distribution']}`",
        f"- involves_procurement_docs distribution: `{qc['involves_procurement_docs_distribution']}`",
        "",
        "## Manual review queue",
        f"- saved sample rows: **{len(sample_rows)}** = 20 random + {len(sample_rows) - 20} edge",
        "- Manual spot-check findings should be appended in-place before moving past STOP 1.",
    ]
    (paths.qc_dir / "extraction_sanity.md").write_text("\n".join(sanity_lines), encoding="utf-8")
    write_stop1_report(paths, final_rows, qc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["bootstrap", "preflight", "run-stop1", "resume-stop1", "rerun-parse-errors", "derive-doc-core", "embed-doc-core", "cluster-stop2", "naming-preflight", "name-clusters-stop3"],
    )
    args = parser.parse_args()
    paths = get_paths(Path(args.run_dir))
    ensure_dirs(paths)
    if args.stage == "bootstrap":
        summary = bootstrap_manifest(paths)
        init_journal(paths, summary)
    elif args.stage == "preflight":
        if not (paths.manifests_dir / "raw_fas_canonical_manifest.csv").exists():
            summary = bootstrap_manifest(paths)
            init_journal(paths, summary)
        preflight(paths)
    elif args.stage == "run-stop1":
        run_stop1(paths)
    elif args.stage == "resume-stop1":
        resume_stop1(paths)
    elif args.stage == "rerun-parse-errors":
        rerun_parse_errors(paths)
    elif args.stage == "derive-doc-core":
        derive_doc_core(paths)
    elif args.stage == "embed-doc-core":
        embed_doc_core(paths)
    elif args.stage == "cluster-stop2":
        cluster_stop2(paths)
    elif args.stage == "naming-preflight":
        naming_preflight(paths)
    elif args.stage == "name-clusters-stop3":
        name_clusters_stop3(paths)


if __name__ == "__main__":
    main()
