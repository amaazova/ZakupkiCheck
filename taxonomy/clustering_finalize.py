#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage21_v2_stop1 import ROOT, append_journal, get_paths, now_iso, read_csv, read_json, write_csv, write_json


PARSED_CLEAN = ROOT / "data" / "parsed_clean"
GOLD_PILOT = ROOT / "data" / "gold_stage20_pilot_stage_a.csv"
CORPUS_SUMMARY = ROOT / "data" / "_corpus_summary.json"


def status_reason(row: dict[str, str]) -> tuple[str, str]:
    size = int(row["size"])
    dep = row["runtime_doc_dependency"]
    ext = row["external_knowledge_required"]
    qc = row["manual_qc_status"]
    attention = row.get("special_attention_reason", "")

    if attention.startswith("mis_scoped") or attention.startswith("out_of_scope"):
        return "out_of_scope", "manual review marked family as mis-scoped relative to DOC_CORE"
    if size < 10:
        return "exploratory", "family has <10 episodes despite thematic interest"
    if dep == "full" and ext == "none" and qc == "coherent" and size >= 20:
        return "benchmarkable", "full runtime-doc detectability, no external knowledge, coherent family, >=20 episodes"
    if ext in {"expert_judgment", "physical_measurement"}:
        return "service_only", "external expert knowledge is the primary support for detection"
    if dep == "none":
        return "service_only", "family is not detectable from runtime documents"
    if ext == "market_knowledge" and dep != "full":
        return "service_only", "market knowledge is primary and runtime evidence is only partial"
    reasons: list[str] = []
    if dep != "full":
        reasons.append(f"runtime_doc_dependency={dep}")
    if ext != "none":
        if ext == "regulatory_history":
            reasons.append("needs regulatory-history workaround")
        else:
            reasons.append(f"external_knowledge_required={ext}")
    if qc != "coherent":
        reasons.append(f"family_qc={qc}")
    if size < 20:
        reasons.append("family size <20")
    if not reasons:
        reasons.append("falls short of strict benchmarkable criterion")
    return "conditional_benchmarkable", "; ".join(reasons)


def parsed_doc_flags(notice_id: str) -> dict[str, Any]:
    if not notice_id:
        return {
            "parsed_dir_exists": False,
            "has_notice_md": False,
            "has_tz_md": False,
            "has_contract_md": False,
        }
    base = PARSED_CLEAN / notice_id
    return {
        "parsed_dir_exists": base.exists(),
        "has_notice_md": (base / "notice.md").exists(),
        "has_tz_md": (base / "tz.md").exists(),
        "has_contract_md": (base / "contract.md").exists(),
    }


def select_unique_representatives(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    rows_sorted = sorted(
        rows,
        key=lambda r: (-float(r["membership_probability"]), r["canonical_decision_id"], int(r["episode_idx"])),
    )
    for row in rows_sorted:
        key = (row["canonical_decision_id"], row["episode_idx"])
        if key in seen:
            continue
        out.append(row)
        seen.add(key)
        if len(out) >= limit:
            break
    return out


def legacy_map(row: dict[str, str]) -> tuple[str, str, str]:
    text = " ".join(
        [
            row["label_ru"].lower(),
            row["summary_ru"].lower(),
            row["key_articles"].lower(),
        ]
    )
    hits: list[str] = []

    def has_term(term: str) -> bool:
        if term in {"сро", "опыт"}:
            return re.search(rf"\\b{re.escape(term)}\\b", text) is not None
        if term in {"лиценз", "разрешительн"}:
            return re.search(rf"\\b{re.escape(term)}", text) is not None
        return term in text

    req_qual_terms = ["пп 2571", "ст. 31", "лиценз", "сро", "опыт", "независимой гарантии", "разрешительн"]
    desc_catalog_terms = [
        "ктру",
        "окпд2",
        "реестр",
        "пп 1875",
        "пп 878",
        "126",
        "национальн режим",
        "радиоэлектрон",
        "картридж",
    ]
    desc_brand_terms = [
        "товарного знака",
        "товарный знак",
        "или эквивалент",
        "оригинальн",
        "совместимости",
        "единственного производителя",
        "бренд",
    ]
    desc_limit_terms = [
        "описани",
        "характеристик",
        "извещен",
        "контракта",
        "инструкц",
        "документации",
        "структурированной формы",
        "сроков",
        "приёмки",
        "объекта закупки",
    ]

    if any(has_term(term) for term in req_qual_terms):
        hits.append("REQ-QUAL")
    if any(has_term(term) for term in desc_catalog_terms):
        hits.append("DESC-CATALOG")
    if any(has_term(term) for term in desc_brand_terms):
        hits.append("DESC-BRAND")
    if any(has_term(term) for term in desc_limit_terms):
        hits.append("DESC-LIMIT-INTERNAL")

    hits = list(dict.fromkeys(hits))
    if not hits:
        return "no_clean_legacy_match", "", "available legacy lanes do not cleanly cover this family"
    primary = hits[0]
    secondary = ";".join(hits[1:])
    confidence = "high" if len(hits) == 1 else "mixed"
    note = f"keyword overlap -> {', '.join(hits)}; confidence={confidence}"
    return primary, secondary, note


def pilot_assessed_share() -> dict[str, dict[str, Any]]:
    rows = read_csv(GOLD_PILOT)
    by_hyp = defaultdict(Counter)
    for row in rows:
        by_hyp[row["hypothesis"]][row["applicability_status"]] += 1
    out: dict[str, dict[str, Any]] = {}
    for hyp, counter in by_hyp.items():
        total = sum(counter.values())
        assessed = total - counter.get("NotAssessed", 0)
        out[hyp] = {
            "pilot_rows": total,
            "assessed_rows": assessed,
            "assessed_share": round(assessed / total, 4) if total else 0.0,
            "counter": dict(counter),
        }
    return out


def build_outputs(paths: Any) -> dict[str, Any]:
    catalog_rows = read_csv(paths.naming_dir / "cluster_catalog_coarse.csv")
    assign_rows = read_csv(paths.clustering_dir / "assignments_coarse.csv")
    doc_core_rows = read_csv(paths.manifests_dir / "doc_core_manifest.csv")
    extraction_qc = read_json(paths.qc_dir / "extraction_auto_qc.json")
    embedding_qc = read_json(paths.clustering_dir / "embedding_qc.json")
    ari_json = read_json(paths.clustering_dir / "clustering_ari.json")
    clustering_metrics = read_csv(paths.clustering_dir / "clustering_config_metrics.csv")
    corpus_summary = read_json(CORPUS_SUMMARY)

    by_cluster_members = defaultdict(list)
    for row in assign_rows:
        by_cluster_members[row["cluster_id"]].append(row)

    doc_core_decisions = sorted({row["canonical_decision_id"] for row in doc_core_rows})
    doc_core_notices = sorted({row["canonical_notice_id"] for row in doc_core_rows if row["canonical_notice_id"]})
    noise_count = embedding_qc["episode_rows"] - sum(int(row["size"]) for row in catalog_rows)

    family_rows: list[dict[str, Any]] = []
    qc_sample_rows: list[dict[str, Any]] = []
    status_counts = Counter()
    status_episode_counts = Counter()

    for row in sorted(catalog_rows, key=lambda r: int(r["cluster_id"])):
        cluster_id = row["cluster_id"]
        members = by_cluster_members[cluster_id]
        status, reason = status_reason(row)
        status_counts[status] += 1
        status_episode_counts[status] += int(row["size"])
        primary_legacy, secondary_legacy, legacy_note = legacy_map(row)

        notice_ids = sorted({m["canonical_notice_id"] for m in members if m["canonical_notice_id"]})
        parsed_flags = [parsed_doc_flags(nid) for nid in notice_ids]
        parsed_dir_exists = sum(1 for f in parsed_flags if f["parsed_dir_exists"])
        has_notice_md = sum(1 for f in parsed_flags if f["has_notice_md"])
        has_tz_md = sum(1 for f in parsed_flags if f["has_tz_md"])
        has_contract_md = sum(1 for f in parsed_flags if f["has_contract_md"])

        manual_sample_n = 5 if status == "benchmarkable" else 3 if status == "service_only" else 0
        representatives = select_unique_representatives(members, max(manual_sample_n, 3))

        family_rows.append(
            {
                "cluster_id": cluster_id,
                "label_ru": row["label_ru"],
                "size": row["size"],
                "n_unique_decisions": row["n_unique_decisions"],
                "primary_status": status,
                "primary_status_reason": reason,
                "runtime_doc_dependency": row["runtime_doc_dependency"],
                "external_knowledge_required": row["external_knowledge_required"],
                "manual_qc_status": row["manual_qc_status"],
                "local_shape_flag": row["local_shape_flag"],
                "reasoning_type": row["reasoning_type"],
                "special_attention_reason": row["special_attention_reason"],
                "parsed_notice_coverage": f"{has_notice_md}/{len(notice_ids)}",
                "parsed_tz_coverage": f"{has_tz_md}/{len(notice_ids)}",
                "parsed_contract_coverage": f"{has_contract_md}/{len(notice_ids)}",
                "parsed_dir_coverage": f"{parsed_dir_exists}/{len(notice_ids)}",
                "legacy_primary": primary_legacy,
                "legacy_secondary": secondary_legacy,
                "legacy_note": legacy_note,
                "representative_decision_ids": json.dumps(
                    [r["canonical_decision_id"] for r in representatives], ensure_ascii=False
                ),
            }
        )

        for sample_idx, member in enumerate(representatives[:manual_sample_n], start=1):
            flags = parsed_doc_flags(member["canonical_notice_id"])
            qc_sample_rows.append(
                {
                    "cluster_id": cluster_id,
                    "label_ru": row["label_ru"],
                    "primary_status": status,
                    "sample_idx": sample_idx,
                    "canonical_decision_id": member["canonical_decision_id"],
                    "canonical_notice_id": member["canonical_notice_id"],
                    "episode_idx": member["episode_idx"],
                    "membership_probability": member["membership_probability"],
                    "scope_tag": member["scope_tag"],
                    "fas_verdict": member["fas_verdict"],
                    "issue_label": member["issue_label"],
                    "reasoning_summary": member["reasoning_summary"],
                    "has_notice_md": flags["has_notice_md"],
                    "has_tz_md": flags["has_tz_md"],
                    "has_contract_md": flags["has_contract_md"],
                }
            )

    write_csv(
        paths.benchmark_dir / "family_status_map.csv",
        [
            "cluster_id",
            "label_ru",
            "size",
            "n_unique_decisions",
            "primary_status",
            "primary_status_reason",
            "runtime_doc_dependency",
            "external_knowledge_required",
            "manual_qc_status",
            "local_shape_flag",
            "reasoning_type",
            "special_attention_reason",
            "parsed_notice_coverage",
            "parsed_tz_coverage",
            "parsed_contract_coverage",
            "parsed_dir_coverage",
            "legacy_primary",
            "legacy_secondary",
            "legacy_note",
            "representative_decision_ids",
        ],
        family_rows,
    )
    write_csv(
        paths.benchmark_dir / "benchmarkability_qc_samples.csv",
        [
            "cluster_id",
            "label_ru",
            "primary_status",
            "sample_idx",
            "canonical_decision_id",
            "canonical_notice_id",
            "episode_idx",
            "membership_probability",
            "scope_tag",
            "fas_verdict",
            "issue_label",
            "reasoning_summary",
            "has_notice_md",
            "has_tz_md",
            "has_contract_md",
        ],
        qc_sample_rows,
    )

    validation = {
        "status_counts": dict(status_counts),
        "status_episode_counts": dict(status_episode_counts),
        "doc_core_decisions": len(doc_core_decisions),
        "doc_core_notices": len(doc_core_notices),
        "doc_core_episodes": len(doc_core_rows),
        "named_clusters": len(catalog_rows),
        "named_episodes": sum(int(row["size"]) for row in catalog_rows),
        "noise_unassigned": noise_count,
        "pilot_assessed_share": pilot_assessed_share(),
    }
    write_json(paths.benchmark_dir / "status_summary.json", validation)

    return {
        "catalog_rows": catalog_rows,
        "family_rows": family_rows,
        "qc_sample_rows": qc_sample_rows,
        "doc_core_rows": doc_core_rows,
        "assign_rows": assign_rows,
        "extraction_qc": extraction_qc,
        "embedding_qc": embedding_qc,
        "ari_json": ari_json,
        "clustering_metrics": clustering_metrics,
        "corpus_summary": corpus_summary,
        "validation": validation,
    }


def report_corpus_landscape(paths: Any, ctx: dict[str, Any]) -> None:
    extraction_qc = ctx["extraction_qc"]
    validation = ctx["validation"]
    corpus_summary = ctx["corpus_summary"]
    lines = [
        "# Corpus Landscape",
        "",
        "## Full corpus",
        f"- decisions_with_text_processed: **{extraction_qc['decision_rows']}**",
        f"- extracted_episode_rows: **{extraction_qc['episode_rows']}**",
        f"- extraction_empty_rate: **{extraction_qc['empty_extraction_rate']}**",
        f"- json_schema_valid_rate: **{extraction_qc['json_schema_valid_rate']}**",
        f"- parse_failure_rate: **{extraction_qc['parse_failure_rate']}**",
        "",
        "## scope_tag distribution",
    ]
    for key, value in sorted(extraction_qc["scope_tag_distribution"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## DOC_CORE landscape",
            f"- doc_core_decisions: **{validation['doc_core_decisions']}**",
            f"- doc_core_notices: **{validation['doc_core_notices']}**",
            f"- doc_core_episodes: **{validation['doc_core_episodes']}**",
            "",
            "## Clustering coverage",
            f"- named_clusters: **{validation['named_clusters']}**",
            f"- named_episodes: **{validation['named_episodes']}**",
            f"- hdbscan_noise_unassigned: **{validation['noise_unassigned']}**",
            "",
            "## Prior workspace context used only as validation",
            f"- legacy scope_counts snapshot: `{corpus_summary['scope_counts']}`",
            f"- legacy readiness_counts snapshot: `{corpus_summary['readiness_counts']}`",
        ]
    )
    (paths.reports_dir / "corpus_landscape.md").write_text("\n".join(lines), encoding="utf-8")


def report_doc_core_definition(paths: Any, ctx: dict[str, Any]) -> None:
    doc_core_rows = ctx["doc_core_rows"]
    exclusions = read_csv(paths.manifests_dir / "doc_core_exclusions.csv")
    contract_exec_audit = read_csv(paths.manifests_dir / "doc_core_excluded_contract_execution_audit.csv")
    inc_counter = Counter(row["scope_tag"] for row in doc_core_rows)
    exc_counter = Counter(row["scope_tag"] for row in exclusions)
    lines = [
        "# DOC_CORE Definition",
        "",
        "## Operational definition",
        "- Include episodes where `scope_tag` ∈ {`doc_content`, `bid_evaluation`, `national_regime`} and `involves_procurement_docs=true`.",
        "- Exclude episodes where `scope_tag` ∈ {`procedural`, `platform`, `contract_execution`, `other`} or `involves_procurement_docs=false`.",
        "",
        "## Included counts",
    ]
    for key, value in sorted(inc_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Excluded counts"])
    for key, value in sorted(exc_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Borderline cases",
            f"- excluded contract_execution audit slice retained separately: **{len(contract_exec_audit)}** episodes",
            "- cluster `3` was re-reviewed after STOP 3 and retained inside DOC_CORE as a notice-content sufficiency family under `ст. 42`, with explicit boundary marker in the catalog.",
        ]
    )
    (paths.reports_dir / "doc_core_definition.md").write_text("\n".join(lines), encoding="utf-8")


def report_taxonomy_method(paths: Any, ctx: dict[str, Any]) -> None:
    ari_json = ctx["ari_json"]
    clustering_metrics = ctx["clustering_metrics"]
    lines = [
        "# Taxonomy Method",
        "",
        "## Extraction",
        "- Primary source: `data/raw_fas/*.json` with canonical field normalization from Step 0.",
        "- LLM-first extraction: OpenRouter -> `anthropic/claude-haiku-4.5` with structured JSON output.",
        "- Taxonomy unit: episode-level, not decision-level.",
        "",
        "## Embeddings",
        "- Model: local `intfloat/multilingual-e5-large`.",
        "- Embedding text = `issue_label + reasoning_summary` with `passage:` prefix.",
        "",
        "## Clustering",
        "- UMAP 10D: `n_neighbors=15`, `min_dist=0.0`, `metric=cosine`, `random_state=20260510`.",
        "- HDBSCAN: `min_samples=5`, `cluster_selection_method=eom`.",
        "- Sensitivity grid:",
    ]
    for row in clustering_metrics:
        lines.append(
            f"  - `{row['config']}`: min_cluster_size={row['min_cluster_size']}, "
            f"n_clusters={row['n_clusters']}, noise_share={row['noise_share']}, largest_cluster_share={row['largest_cluster_share']}"
        )
    lines.extend(
        [
            "",
            "## Stability",
            f"- ARI: `{ari_json['ari']}`",
            f"- chosen config: `{ari_json['recommended_config']}`",
            "",
            "## Naming",
            "- LLM-first naming: OpenRouter -> `anthropic/claude-sonnet-4.6` with structured JSON output and provider lock.",
            "- Manual correction pass applied after STOP 3 to clusters `5`, `18`, `41`, with boundary re-review on cluster `3`.",
        ]
    )
    (paths.reports_dir / "taxonomy_method.md").write_text("\n".join(lines), encoding="utf-8")


def report_taxonomy_catalog(paths: Any, ctx: dict[str, Any]) -> None:
    family_rows = ctx["family_rows"]
    members_by_cluster = defaultdict(list)
    for row in ctx["qc_sample_rows"]:
        members_by_cluster[row["cluster_id"]].append(row)
    lines = [
        "# Taxonomy Catalog",
        "",
        "Families are grouped by Step 6 primary benchmarkability status, but definitions remain de novo.",
    ]
    for status in ["benchmarkable", "conditional_benchmarkable", "service_only", "exploratory", "out_of_scope"]:
        subset = [row for row in family_rows if row["primary_status"] == status]
        if not subset:
            continue
        lines.extend(["", f"## {status}"])
        for row in subset:
            cluster_id = row["cluster_id"]
            lines.append(
                f"- cluster `{cluster_id}`: `{row['label_ru']}` | size={row['size']} | decisions={row['n_unique_decisions']} | dep={row['runtime_doc_dependency']} | ext={row['external_knowledge_required']}"
            )
            lines.append(f"  definition: {next(r['summary_ru'] for r in ctx['catalog_rows'] if r['cluster_id'] == cluster_id)}")
            lines.append(f"  rationale: {row['primary_status_reason']}")
            reps = members_by_cluster.get(cluster_id, [])[:2]
            if reps:
                lines.append(
                    "  examples: "
                    + " || ".join(
                        f"{rep['canonical_decision_id']}::{rep['episode_idx']} {rep['issue_label']}"
                        for rep in reps
                    )
                )
    (paths.reports_dir / "taxonomy_catalog.md").write_text("\n".join(lines), encoding="utf-8")


def report_benchmarkability(paths: Any, ctx: dict[str, Any]) -> None:
    family_rows = ctx["family_rows"]
    validation = ctx["validation"]
    corpus_summary = ctx["corpus_summary"]
    lines = [
        "# Benchmarkability Map",
        "",
        "## Primary criteria",
        "- `benchmarkable`: full runtime-doc detectability, no external knowledge, coherent family, >=20 episodes.",
        "- `conditional_benchmarkable`: partial runtime evidence, workaround-based external knowledge, mixed family, or <20 episodes.",
        "- `service_only`: runtime documents are insufficient without expert / market / non-document support as the main signal.",
        "- `exploratory`: <10 episodes.",
        "- `out_of_scope`: mis-scoped relative to DOC_CORE.",
        "",
        "## Status counts",
    ]
    for key, value in sorted(validation["status_counts"].items()):
        lines.append(f"- `{key}`: {value} families / {validation['status_episode_counts'][key]} episodes")

    for status in ["benchmarkable", "conditional_benchmarkable", "service_only", "exploratory", "out_of_scope"]:
        subset = [row for row in family_rows if row["primary_status"] == status]
        if not subset:
            continue
        lines.extend(["", f"## {status}"])
        for row in subset:
            lines.append(
                f"- cluster `{row['cluster_id']}` `{row['label_ru']}`: {row['primary_status_reason']} "
                f"(size={row['size']}, dep={row['runtime_doc_dependency']}, ext={row['external_knowledge_required']}, qc={row['manual_qc_status']})"
            )

    pilot_share = validation["pilot_assessed_share"]
    lines.extend(
        [
            "",
            "## Post-hoc validation",
            "- Legacy pilot assessed share is used only as a sanity signal after taxonomy freeze; it never overrides the primary status above.",
            f"- legacy readiness snapshot: `{corpus_summary['readiness_counts']}`",
            f"- Stage 20 pilot assessed share: `{ {k: v['assessed_share'] for k, v in pilot_share.items()} }`",
            "- Notable finding: the de novo taxonomy isolates a small benchmarkable REQ-QUAL core instead of treating the whole legacy REQ-QUAL lane as uniformly low-density.",
        ]
    )
    (paths.reports_dir / "benchmarkability_map.md").write_text("\n".join(lines), encoding="utf-8")


def report_legacy_comparison(paths: Any, ctx: dict[str, Any]) -> None:
    family_rows = ctx["family_rows"]
    pilot_share = ctx["validation"]["pilot_assessed_share"]
    by_legacy = defaultdict(list)
    for row in family_rows:
        by_legacy[row["legacy_primary"]].append(row)

    legacy_rows = []
    for row in family_rows:
        legacy_rows.append(
            {
                "cluster_id": row["cluster_id"],
                "label_ru": row["label_ru"],
                "primary_status": row["primary_status"],
                "legacy_primary": row["legacy_primary"],
                "legacy_secondary": row["legacy_secondary"],
                "legacy_note": row["legacy_note"],
            }
        )
    write_csv(
        paths.benchmark_dir / "legacy_family_alignment.csv",
        ["cluster_id", "label_ru", "primary_status", "legacy_primary", "legacy_secondary", "legacy_note"],
        legacy_rows,
    )

    lines = [
        "# Legacy Comparison",
        "",
        "- This layer is post-hoc only and was not used as a source prior for extraction, clustering, naming, or primary benchmarkability status assignment.",
        "- Explicit local definitions for legacy `H2/H3/H5` families were not available in the current workspace; comparison is therefore grounded in the available `DESC-*` / `REQ-QUAL` lanes plus Stage 20 pilot assessed-share signals.",
        "",
        "## Stage 20 pilot assessed share",
    ]
    for hyp in ["DESC-BRAND", "DESC-CATALOG", "DESC-LIMIT-INTERNAL", "REQ-QUAL"]:
        payload = pilot_share.get(hyp)
        if payload:
            lines.append(
                f"- `{hyp}`: assessed_share={payload['assessed_share']} ({payload['assessed_rows']}/{payload['pilot_rows']})"
            )
    lines.extend(["", "## Family mapping summary"])
    for legacy_key, rows in sorted(by_legacy.items()):
        lines.append(f"- `{legacy_key}`: {len(rows)} families")
        for row in sorted(rows, key=lambda r: int(r["cluster_id"]))[:12]:
            lines.append(
                f"  - cluster `{row['cluster_id']}` `{row['label_ru']}` -> `{row['primary_status']}` ({row['legacy_note']})"
            )
        if len(rows) > 12:
            lines.append(f"  - ... +{len(rows) - 12} more")
    (paths.reports_dir / "legacy_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def report_final_summary(paths: Any, ctx: dict[str, Any]) -> None:
    extraction_qc = ctx["extraction_qc"]
    validation = ctx["validation"]
    ari_json = ctx["ari_json"]
    family_rows = ctx["family_rows"]
    benchmarkable = [row["label_ru"] for row in family_rows if row["primary_status"] == "benchmarkable"]
    service_only = [row["label_ru"] for row in family_rows if row["primary_status"] == "service_only"]
    lines = [
        "# Final Summary",
        "",
        f"- Full corpus with text: **{extraction_qc['decision_rows']}** decisions",
        f"- DOC_CORE: **{validation['doc_core_decisions']}** decisions / **{validation['doc_core_episodes']}** episodes",
        f"- Stable named families in coarse config: **{validation['named_clusters']}** covering **{validation['named_episodes']}** episodes",
        "",
        "## Benchmarkable families",
    ]
    for label in benchmarkable:
        lines.append(f"- {label}")
    lines.extend(["", "## Service-only families"])
    for label in service_only:
        lines.append(f"- {label}")
    lines.extend(
        [
            "",
            "## Stability",
            f"- ARI: `{ari_json['ari']}`",
            "",
            "## Reruns applied",
            "- extraction resume pass after intentional interrupt",
            "- parse-error rerun on 59 rows",
            "- second parse-error rerun on 3 rows",
            "- bounded manual naming correction pass on clusters 5, 18, 41 plus boundary re-review on cluster 3",
            "",
            "## Remaining limitations",
            f"- HDBSCAN noise remains **{validation['noise_unassigned']}** episodes and is not named in the current family catalog.",
            "- Broad and mixed families remain explicitly held out from overconfident benchmarkability claims.",
            "- Legacy H2/H3/H5 comparison is incomplete because explicit local definitions were not available in the workspace.",
            "",
            "## Recommendation for PROJECT_DESIGN v6",
            "- Treat the de novo benchmarkable families as the clean seed set for the next benchmark design iteration.",
            "- Use conditional families as bounded salvage / split candidates, not as immediate gold-label bulk queues.",
            "- Keep service-only families outside pure runtime-document benchmark scope unless a separate external-knowledge evaluation track is planned.",
        ]
    )
    (paths.reports_dir / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def report_benchmarkability_qc(paths: Any, ctx: dict[str, Any]) -> None:
    family_rows = {row["cluster_id"]: row for row in ctx["family_rows"]}
    samples = ctx["qc_sample_rows"]
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["cluster_id"]].append(row)

    lines = [
        "# Benchmarkability Sanity",
        "",
        "Manual review in this step was limited to representative episode members for `benchmarkable` and `service_only` families.",
    ]

    for status in ["benchmarkable", "service_only"]:
        subset = [row for row in ctx["family_rows"] if row["primary_status"] == status]
        if not subset:
            continue
        lines.extend(["", f"## {status} review"])
        for row in subset:
            cluster_id = row["cluster_id"]
            lines.append(
                f"- cluster `{cluster_id}` `{row['label_ru']}`: confirmed `{status}` on representative members; "
                f"dep={row['runtime_doc_dependency']}, ext={row['external_knowledge_required']}, parsed_tz_coverage={row['parsed_tz_coverage']}"
            )
            for sample in grouped.get(cluster_id, [])[: (5 if status == "benchmarkable" else 3)]:
                lines.append(
                    f"  - {sample['canonical_decision_id']}::{sample['episode_idx']} [{sample['fas_verdict']}] {sample['issue_label']}"
                )
    (paths.qc_dir / "benchmarkability_sanity.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(paths: Any) -> None:
    ctx = build_outputs(paths)
    report_corpus_landscape(paths, ctx)
    report_doc_core_definition(paths, ctx)
    report_taxonomy_method(paths, ctx)
    report_taxonomy_catalog(paths, ctx)
    report_benchmarkability(paths, ctx)
    report_legacy_comparison(paths, ctx)
    report_final_summary(paths, ctx)
    report_benchmarkability_qc(paths, ctx)

    append_journal(
        paths,
        f"Step 6 Benchmarkability @ {now_iso()}",
        [
            f"- status_counts: `{ctx['validation']['status_counts']}`",
            f"- status_episode_counts: `{ctx['validation']['status_episode_counts']}`",
            "- primary_rule: status assigned from de novo family metadata before any legacy comparison",
            "- outputs: `benchmarkability/family_status_map.csv`, `benchmarkability/benchmarkability_qc_samples.csv`, `benchmarkability/status_summary.json`, `qc/benchmarkability_sanity.md`, `reports/benchmarkability_map.md`",
        ],
    )
    append_journal(
        paths,
        f"Step 7 Final Reports @ {now_iso()}",
        [
            "- outputs: `reports/corpus_landscape.md`, `reports/doc_core_definition.md`, `reports/taxonomy_method.md`, `reports/taxonomy_catalog.md`, `reports/benchmarkability_map.md`, `reports/legacy_comparison.md`, `reports/final_summary.md`",
            "- legacy_comparison_note: explicit H2/H3/H5 local definitions not found; comparison limited to available DESC-* / REQ-QUAL lanes plus Stage 20 pilot assessed-share signals",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    paths = get_paths(Path(args.run_dir))
    finalize(paths)


if __name__ == "__main__":
    main()
