"""W4 — finalize: matching for all variants + comparison tables."""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .compute_metrics import (
    BOOTSTRAP_ALPHA, BOOTSTRAP_ITERS, cluster_bootstrap, episode_metrics, fact_pr,
    fmt_ci,
)
from .config import PROJECT_ROOT
from .extraction_runner import read_jsonl
from .matching_pipeline import (
    EMBED_MODEL, SPECIFIC_TYPES, collect_quotes, run_matching,
)

EVAL = PROJECT_ROOT / "workspace" / "eval"
EVAL_CSV = EVAL / "eval_dataset_v10.csv"
STRATA = ("benchmarkable", "conditional", "service_only")

# Knowledge ladder variants (all use V4 Flash FAS as gold side)
LADDER = [
    ("L0", "tz_features_L0.jsonl",     "matching_results_L0.jsonl",     "Regex"),
    ("L1", "tz_features_L1.jsonl",     "matching_results.jsonl",        "LLM open"),
    ("L2", "tz_features_L2.jsonl",     "matching_results_L2.jsonl",     "LLM + taxonomy hint"),
    ("L3", "tz_features_L3.jsonl",     "matching_results_L3.jsonl",     "LLM + few-shot"),
    ("A3", "tz_features_A3_cot.jsonl", "matching_results_A3.jsonl",     "LLM + CoT"),
]

# Multi-model variants (FAS_path, TZ_path, matching_path, label)
MODELS = [
    ("V4 Flash",     "fas_findings.jsonl",         "tz_features_L1.jsonl",     "matching_results.jsonl",         "v4flash"),
    ("Sonnet 4.6",   "sonnet_fas_findings.jsonl",  "sonnet_tz_features.jsonl", "matching_results_sonnet.jsonl",  "sonnet"),
    ("Qwen 3.6+",    "qwen_fas_findings.jsonl",    "qwen_tz_features.jsonl",   "matching_results_qwen.jsonl",    "qwen"),
]


# Shared quote embedding (across all FAS/TZ files)

def build_shared_quote_emb() -> dict[str, np.ndarray]:
    """Embed every unique FAS/TZ quote across all source files."""
    sources_fas = ["fas_findings.jsonl", "sonnet_fas_findings.jsonl", "qwen_fas_findings.jsonl"]
    sources_tz  = ["tz_features_L0.jsonl", "tz_features_L1.jsonl", "tz_features_L2.jsonl",
                   "tz_features_L3.jsonl", "tz_features_A3_cot.jsonl",
                   "sonnet_tz_features.jsonl", "qwen_tz_features.jsonl"]
    fas_recs = []
    tz_recs = []
    for f in sources_fas:
        fas_recs.extend(read_jsonl(EVAL / f))
    for f in sources_tz:
        tz_recs.extend(read_jsonl(EVAL / f))
    quotes = collect_quotes(fas_recs, tz_recs)
    print(f"[w4] unique quotes (union): {len(quotes)}")
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(quotes, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return {q: e for q, e in zip(quotes, embs)}


# Run the 6 missing matchings

def run_all_matchings(quote_emb: dict[str, np.ndarray]) -> None:
    # Knowledge ladder (uses V4 Flash FAS): L0, L2, L3, A3 (L1 already exists)
    for code, tz_file, out_file, _label in LADDER:
        out_path = EVAL / out_file
        if code == "L1":
            assert out_path.is_file(), f"missing W2 baseline: {out_path}"
            continue
        if out_path.is_file():
            print(f"[w4] {code}: {out_file} exists, skipping")
            continue
        run_matching(
            fas_path=EVAL / "fas_findings.jsonl",
            tz_path=EVAL / tz_file,
            eval_path=EVAL_CSV,
            output_path=out_path,
            quote_emb=quote_emb,
        )

    # Multi-model: Sonnet end-to-end, Qwen end-to-end
    for _label, fas_file, tz_file, out_file, alias in MODELS:
        if alias == "v4flash":
            continue
        out_path = EVAL / out_file
        if out_path.is_file():
            print(f"[w4] {alias}: {out_file} exists, skipping")
            continue
        run_matching(
            fas_path=EVAL / fas_file,
            tz_path=EVAL / tz_file,
            eval_path=EVAL_CSV,
            output_path=out_path,
            quote_emb=quote_emb,
        )


# Headline metrics per matching file

def filter_stratum(records, s):
    return records if s == "pooled" else [r for r in records if r["stratum"] == s]


def headline_metrics(matching_path: Path) -> dict:
    records = read_jsonl(matching_path)
    out = {}
    # Fact (specific_only) pooled with CI
    p, r, f1, m, g, pr = fact_pr(records, mode="specific")
    p_pt, p_lo, p_hi = cluster_bootstrap(records, lambda rs: fact_pr(rs, mode="specific")[0])
    r_pt, r_lo, r_hi = cluster_bootstrap(records, lambda rs: fact_pr(rs, mode="specific")[1])
    f_pt, f_lo, f_hi = cluster_bootstrap(records, lambda rs: fact_pr(rs, mode="specific")[2])
    out["fact_p"] = (p, p_lo, p_hi)
    out["fact_r"] = (r, r_lo, r_hi)
    out["fact_f1"] = (f1, f_lo, f_hi)
    out["fact_counts"] = (m, g, pr)
    # Episode strict pooled with CI
    em = episode_metrics(records, pred_key="episode_pred_strict")
    a_pt, a_lo, a_hi = cluster_bootstrap(records, lambda rs: episode_metrics(rs, pred_key="episode_pred_strict")["acc"])
    k_pt, k_lo, k_hi = cluster_bootstrap(records, lambda rs: episode_metrics(rs, pred_key="episode_pred_strict")["kappa"])
    out["ep_acc"] = (em["acc"], a_lo, a_hi)
    out["ep_kappa"] = (em["kappa"], k_lo, k_hi)
    out["ep_f1_macro"] = em["f1_macro"]
    out["ep_confusion"] = (em["tp"], em["fp"], em["fn"], em["tn"])
    # Stratified specific fact metrics
    strat = {}
    for s in STRATA:
        sub = filter_stratum(records, s)
        p2, r2, f12, m2, g2, pr2 = fact_pr(sub, mode="specific")
        strat[s] = {"p": p2, "r": r2, "f1": f12, "m": m2, "g": g2, "pr": pr2}
    out["fact_strat"] = strat
    return out


# Knowledge ladder

def build_knowledge_ladder() -> str:
    rows = []
    for code, _tz_file, match_file, label in LADDER:
        path = EVAL / match_file
        rows.append((code, label, headline_metrics(path)))

    lines: list[str] = []
    lines.append("# Knowledge ladder — L0 → L1 → L2 → L3 → A3\n")
    lines.append("_specific_only fact metrics on gold violation episodes; "
                 "episode-level metrics on all 781. Bootstrap (1000 iters, cluster by notice_id)._\n")
    lines.append("\n## Pooled\n")
    lines.append("| Level | Description | Fact P | Fact R | Fact F1 | Episode Acc | Episode κ |")
    lines.append("|---|---|---|---|---|---|---|")
    for code, label, m in rows:
        lines.append(
            f"| {code} | {label} | {fmt_ci(*m['fact_p'])} | {fmt_ci(*m['fact_r'])} | "
            f"{fmt_ci(*m['fact_f1'])} | {fmt_ci(*m['ep_acc'])} | "
            f"{fmt_ci(*m['ep_kappa'], pct=False)} |"
        )

    # Stratified
    lines.append("\n## Stratified (Fact F1, specific_only)\n")
    lines.append("| Level | benchmarkable | conditional | service_only |")
    lines.append("|---|---|---|---|")
    for code, _label, m in rows:
        s = m["fact_strat"]
        lines.append(
            f"| {code} | {s['benchmarkable']['f1']*100:.1f}% "
            f"({s['benchmarkable']['m']}/{s['benchmarkable']['g']}) | "
            f"{s['conditional']['f1']*100:.1f}% "
            f"({s['conditional']['m']}/{s['conditional']['g']}) | "
            f"{s['service_only']['f1']*100:.1f}% "
            f"({s['service_only']['m']}/{s['service_only']['g']}) |"
        )

    # Δ table (vs L1)
    base = next(m for c, _, m in rows if c == "L1")
    lines.append("\n## Δ vs L1 (pooled)\n")
    lines.append("| Level | Δ Fact F1 | Δ Episode Acc | Δ Episode κ |")
    lines.append("|---|---|---|---|")
    for code, _label, m in rows:
        if code == "L1":
            continue
        lines.append(
            f"| {code} | "
            f"{(m['fact_f1'][0]-base['fact_f1'][0])*100:+.1f} pp | "
            f"{(m['ep_acc'][0]-base['ep_acc'][0])*100:+.1f} pp | "
            f"{m['ep_kappa'][0]-base['ep_kappa'][0]:+.3f} |"
        )

    # Benchmarkable-recall central hypothesis check
    lines.append("\n## Central hypothesis — benchmarkable recall by level\n")
    lines.append("| Level | Fact Recall (benchmarkable) | Fact Recall (service_only) |")
    lines.append("|---|---|---|")
    for code, _label, m in rows:
        s = m["fact_strat"]
        b_r = s['benchmarkable']['r'] if s['benchmarkable']['g'] else 0.0
        so_r = s['service_only']['r'] if s['service_only']['g'] else 0.0
        lines.append(f"| {code} | {b_r*100:.1f}% | {so_r*100:.1f}% |")
    return "\n".join(lines) + "\n"


# Multi-model comparison

def build_multi_model_section() -> tuple[str, dict]:
    """Returns (markdown, model_match_records dict for downstream Fleiss/McNemar)."""
    model_match: dict[str, list[dict]] = {}
    for _label, _fas, _tz, match_file, alias in MODELS:
        model_match[alias] = read_jsonl(EVAL / match_file)

    # Headline numbers per model
    model_metrics: dict[str, dict] = {}
    for label, _fas, _tz, match_file, alias in MODELS:
        model_metrics[alias] = headline_metrics(EVAL / match_file)

    # Verdict-agreement (FAS extractor) reload from FAS files
    eval_df = pd.read_csv(EVAL_CSV, dtype={"notice_id": str})
    eval_map = dict(zip(eval_df["episode_id"], eval_df["fas_verdict"]))

    fas_summary: dict[str, dict] = {}
    fas_files = {
        "v4flash": "fas_findings.jsonl",
        "sonnet":  "sonnet_fas_findings.jsonl",
        "qwen":    "qwen_fas_findings.jsonl",
    }
    for alias, fname in fas_files.items():
        recs = read_jsonl(EVAL / fname)
        n = len(recs)
        verdict_agree = sum(1 for r in recs
                            if r.get("overall_verdict") == eval_map.get(r["episode_id"]))
        n_findings = sum(len(r.get("findings") or []) for r in recs)
        n_est = sum(1 for r in recs for f in (r.get("findings") or []) if f.get("severity") == "established")
        cost = sum(r.get("extraction_cost_usd", 0.0) for r in recs)
        fas_summary[alias] = dict(
            verdict_pct=verdict_agree / n,
            n_findings=n_findings,
            n_est=n_est,
            cost=cost,
        )

    tz_summary: dict[str, dict] = {}
    tz_files = {
        "v4flash": "tz_features_L1.jsonl",
        "sonnet":  "sonnet_tz_features.jsonl",
        "qwen":    "qwen_tz_features.jsonl",
    }
    for alias, fname in tz_files.items():
        recs = read_jsonl(EVAL / fname)
        flags = [f for r in recs if isinstance(r.get("parsed"), dict)
                 for f in (r["parsed"].get("risk_flags") or [])]
        cost = sum(r.get("cost_usd", 0.0) for r in recs)
        tz_summary[alias] = dict(
            n_flags=len(flags),
            cost=cost,
        )

    lines: list[str] = []
    lines.append("# Multi-model comparison — V4 Flash / Sonnet 4.6 / Qwen 3.6+\n")
    lines.append("_End-to-end: each model used for both FAS and TZ extraction. "
                 "Matching uses same pipeline; specific_only mode for fact metrics, "
                 "strict mode for episode metrics._\n")
    lines.append("\n## Headline table\n")
    lines.append("| Model | FAS verdict agr | FAS findings/ep | TZ flags/ep | Fact F1 | Episode Acc | Episode κ | Total cost |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for label, _fas, _tz, _mf, alias in MODELS:
        fs = fas_summary[alias]
        ts = tz_summary[alias]
        m = model_metrics[alias]
        lines.append(
            f"| {label} | {fs['verdict_pct']*100:.1f}% | "
            f"{fs['n_findings']/781:.2f} | {ts['n_flags']/781:.2f} | "
            f"{fmt_ci(*m['fact_f1'])} | {fmt_ci(*m['ep_acc'])} | "
            f"{fmt_ci(*m['ep_kappa'], pct=False)} | ${fs['cost']+ts['cost']:.2f} |"
        )

    # Fleiss kappa
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
    aliases = [m[-1] for m in MODELS]
    by_ep: dict[str, list[int]] = defaultdict(list)
    for alias in aliases:
        idx = {r["episode_id"]: r for r in model_match[alias]}
        for eid in eval_df["episode_id"]:
            r = idx.get(eid, {})
            by_ep[eid].append(1 if r.get("episode_pred_strict") == "violation_established" else 0)
    ratings = [by_ep[eid] for eid in eval_df["episode_id"]]
    # Need format: shape (n_subjects, n_categories); aggregate_raters converts.
    table, _cats = aggregate_raters(np.array(ratings))
    fleiss = fleiss_kappa(table, method="fleiss")

    lines.append("\n## Fleiss κ (3-model strict-mode agreement on episode prediction)\n")
    lines.append(f"**Fleiss κ = {fleiss:.3f}** over 781 episodes × 3 models. "
                 "Interpretation: κ < 0.20 poor; 0.21–0.40 fair; 0.41–0.60 moderate; "
                 "0.61–0.80 substantial.\n")

    # McNemar pairwise
    from statsmodels.stats.contingency_tables import mcnemar
    pair_idx = [("v4flash", "sonnet"), ("v4flash", "qwen"), ("sonnet", "qwen")]
    lines.append("\n## Pairwise McNemar tests (episode-level correctness)\n")
    lines.append("Cell counts: B = model A correct & model B wrong; C = model A wrong & model B correct. "
                 "Two-sided p; α = 0.05. Significant ⇒ models disagree on errors.\n")
    lines.append("| Pair (A vs B) | A correct/B wrong | A wrong/B correct | McNemar χ² | p-value |")
    lines.append("|---|---|---|---|---|")
    for a, b in pair_idx:
        idx_a = {r["episode_id"]: r for r in model_match[a]}
        idx_b = {r["episode_id"]: r for r in model_match[b]}
        b_only = 0  # A correct, B wrong
        c_only = 0  # A wrong, B correct
        for eid in eval_df["episode_id"]:
            gold = eval_map[eid]
            pa = idx_a[eid]["episode_pred_strict"] == gold
            pb = idx_b[eid]["episode_pred_strict"] == gold
            if pa and not pb:
                b_only += 1
            elif pb and not pa:
                c_only += 1
        ct = np.array([[0, b_only], [c_only, 0]])
        try:
            r_mc = mcnemar(ct, exact=False, correction=True)
            stat = r_mc.statistic
            pval = r_mc.pvalue
        except Exception as e:
            stat, pval = float("nan"), float("nan")
        sig = " **\\***" if pval < 0.05 else ""
        lines.append(f"| {a} vs {b} | {b_only} | {c_only} | {stat:.2f} | {pval:.4g}{sig} |")

    # Heatmap: model × finding_type fact recall
    finding_types = sorted({
        "brand_without_equivalent", "incomplete_description", "restrictive_requirement",
        "ktru_mismatch", "inconsistent_dates", "missing_acceptance_terms",
    })
    heatmap = np.zeros((len(MODELS), len(finding_types)))
    for i, (_label, fas_file, _tz, _mf, alias) in enumerate(MODELS):
        # Gold counts come from THIS model's FAS extraction (each row's own gold)
        fas_recs = read_jsonl(EVAL / fas_file)
        match_recs = model_match[alias]
        gold_by_type: Counter = Counter()
        for r in fas_recs:
            if eval_map.get(r["episode_id"]) != "violation_established":
                continue
            for f in r.get("findings") or []:
                if f.get("severity") == "established":
                    gold_by_type[f.get("finding_type")] += 1
        # Matched counts come from matching records: we need a per-type breakdown.
        # The matching_pipeline outputs aggregate counts only. We re-derive:
        # for each episode, find the per-type matched count by looking at
        # min(fas_type_count, tz_type_count) where types intersect (specific only).
        # That is the upper bound on type-matches under greedy 1-to-1.
        matched_by_type: Counter = Counter()
        fas_idx = {r["episode_id"]: r for r in fas_recs}
        tz_idx = {r["episode_id"]: r for r in read_jsonl(EVAL / MODELS[i][2])}
        from .matching_pipeline import extract_tz_flags as etf
        for mr in match_recs:
            if mr["fas_verdict"] != "violation_established":
                continue
            eid = mr["episode_id"]
            fas_rec = fas_idx.get(eid, {})
            tz_rec = tz_idx.get(eid, {})
            fas_est = [f for f in (fas_rec.get("findings") or [])
                       if f.get("severity") == "established"
                       and f.get("finding_type") in SPECIFIC_TYPES]
            tz_flags = [f for f in etf(tz_rec) if f.get("flag_type") in SPECIFIC_TYPES]
            fc = Counter(f.get("finding_type") for f in fas_est)
            tc = Counter(f.get("flag_type")    for f in tz_flags)
            for t in fc:
                matched_by_type[t] += min(fc[t], tc.get(t, 0))
        for j, t in enumerate(finding_types):
            g = gold_by_type.get(t, 0)
            heatmap[i, j] = matched_by_type[t] / g if g else 0.0

    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(heatmap, aspect="auto", vmin=0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(range(len(finding_types)))
    ax.set_xticklabels(finding_types, rotation=30, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[0] for m in MODELS])
    for i in range(len(MODELS)):
        for j in range(len(finding_types)):
            ax.text(j, i, f"{heatmap[i,j]*100:.0f}%",
                    ha="center", va="center",
                    color="white" if heatmap[i, j] > 0.5 else "black",
                    fontsize=9)
    ax.set_title("Fact recall by model × finding_type (specific_only, gold violations)")
    fig.colorbar(im, ax=ax, label="recall")
    fig.tight_layout()
    out_png = EVAL / "multi_model_heatmap.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    lines.append(f"\n## Heatmap — model × finding_type fact recall\n\n![heatmap]({out_png.name})\n")

    return "\n".join(lines) + "\n", model_match


# v9 vs v10 final table

def build_v9_vs_v10() -> str:
    # Best knowledge level by pooled episode accuracy (headline metric for this table)
    best_level = None
    best_level_acc = -1.0
    best_level_metrics = None
    for code, _tz, match_file, _label in LADDER:
        m = headline_metrics(EVAL / match_file)
        if m["ep_acc"][0] > best_level_acc:
            best_level_acc = m["ep_acc"][0]
            best_level = code
            best_level_metrics = m

    # Best model by pooled episode accuracy
    best_model = None
    best_model_acc = -1.0
    best_model_metrics = None
    for label, _fas, _tz, match_file, alias in MODELS:
        m = headline_metrics(EVAL / match_file)
        if m["ep_acc"][0] > best_model_acc:
            best_model_acc = m["ep_acc"][0]
            best_model = label
            best_model_metrics = m

    # IE + matching strict = v4flash baseline (matching_results.jsonl)
    v4 = headline_metrics(EVAL / "matching_results.jsonl")

    lines: list[str] = []
    lines.append("# v9 vs v10 — final comparison\n")
    lines.append("| Approach | Year | Accuracy | κ | Paradigm |")
    lines.append("|---|---|---|---|---|")
    lines.append("| B0 majority | v9 | 62.1% | 0.00 | naive (predict majority) |")
    lines.append("| B1 regex | v9 | 53.4% | — | pattern matching |")
    lines.append("| B2 zero-shot LLM | v9 | 50.7% | ~0.01 | direct classification |")
    lines.append("| B4 pipeline | v9 | 54.7% | ~0.09 | 4 detectors + OR |")
    lines.append("| ML on L1 features | v10 | 60.8% | 0.22 | extraction → sklearn |")
    lines.append(f"| **IE + strict matching (V4 Flash)** | v10 | **{v4['ep_acc'][0]*100:.1f}%** | "
                 f"**{v4['ep_kappa'][0]:.2f}** | IE + type match |")
    lines.append(f"| Best knowledge level ({best_level}) | v10 | "
                 f"{best_level_metrics['ep_acc'][0]*100:.1f}% | "
                 f"{best_level_metrics['ep_kappa'][0]:.2f} | IE (best prompt) + type match |")
    lines.append(f"| Best model ({best_model}) | v10 | "
                 f"{best_model_metrics['ep_acc'][0]*100:.1f}% | "
                 f"{best_model_metrics['ep_kappa'][0]:.2f} | IE (best model) + type match |")
    lines.append(
        "\n**Notes**: v9 numbers from `PROJECT_DESIGN_v9.md`; "
        "v10 numbers measured on 781-episode eval_dataset_v10 with strict "
        "episode prediction (≥1 type-matched specific finding). "
        "Bootstrap CIs in `knowledge_ladder.md` and `multi_model_comparison.md`."
    )
    return "\n".join(lines) + "\n"


# Budget summary

def sum_jsonl_cost(path: Path, key: str = "cost_usd") -> float:
    if not path.is_file():
        return 0.0
    total = 0.0
    for r in read_jsonl(path):
        # FAS files use extraction_cost_usd; TZ files use cost_usd
        total += r.get("cost_usd", 0.0) or r.get("extraction_cost_usd", 0.0)
    return total


def build_budget_summary() -> str:
    rows = [
        ("W1 FAS extraction (500 dec)", "V4 Flash",      "_fas_decisions_raw.jsonl"),
        ("W1 TZ L1 (open)",             "V4 Flash",      "tz_features_L1.jsonl"),
        ("W2 TZ L2 (guided)",           "V4 Flash",      "tz_features_L2.jsonl"),
        ("W2 TZ L3 (few-shot)",         "V4 Flash",      "tz_features_L3.jsonl"),
        ("A3 TZ CoT",                   "V4 Flash",      "tz_features_A3_cot.jsonl"),
        ("A5 Sonnet FAS",               "Sonnet 4.6",    "_sonnet_fas_decisions_raw.jsonl"),
        ("A5 Sonnet TZ",                "Sonnet 4.6",    "sonnet_tz_features.jsonl"),
        ("A5 Qwen FAS",                 "Qwen 3.6+",     "_qwen_fas_decisions_raw.jsonl"),
        ("A5 Qwen TZ",                  "Qwen 3.6+",     "qwen_tz_features.jsonl"),
    ]
    total = 0.0
    body = ["| Run | Model | Cost |", "|---|---|---|"]
    for label, model, fname in rows:
        c = sum_jsonl_cost(EVAL / fname)
        body.append(f"| {label} | {model} | ${c:.2f} |")
        total += c
    smoke = 1.5  # rough; smokes used ~$0.5 each
    body.append(f"| Smoke tests + misc | mixed | ~${smoke:.2f} |")
    total += smoke
    body.append(f"| **TOTAL** | | **${total:.2f}** |")
    body.append(f"| Budget | | $355.00 |")
    body.append(f"| Remaining | | ${355 - total:.2f} |")

    lines = ["# Budget summary\n",
             "_Costs read directly from each run's JSONL `cost_usd` / `extraction_cost_usd` fields._\n"]
    lines.extend(body)
    return "\n".join(lines) + "\n"


# Driver

def main() -> None:
    # Step 1: build shared embeddings (slow, one-shot)
    quote_emb = build_shared_quote_emb()

    # Step 2: matchings for L0, L2, L3, A3, Sonnet, Qwen
    run_all_matchings(quote_emb)

    # Step 3: knowledge ladder
    print("[w4] building knowledge_ladder.md...")
    kl = build_knowledge_ladder()
    (EVAL / "knowledge_ladder.md").write_text(kl, encoding="utf-8")

    # Step 4: multi-model comparison + Fleiss + McNemar + heatmap
    print("[w4] building multi_model_comparison.md...")
    mm, _ = build_multi_model_section()
    (EVAL / "multi_model_comparison.md").write_text(mm, encoding="utf-8")

    # Step 5: v9 vs v10
    print("[w4] building v9_vs_v10.md...")
    v9v10 = build_v9_vs_v10()
    (EVAL / "v9_vs_v10.md").write_text(v9v10, encoding="utf-8")

    # Step 6: budget summary
    print("[w4] building budget_summary.md...")
    budget = build_budget_summary()
    (EVAL / "budget_summary.md").write_text(budget, encoding="utf-8")

    # Step 7: final consolidated tables
    print("[w4] building final_eval_tables.md...")
    final = (
        "# v10 — final evaluation tables\n\n"
        "Consolidated report: knowledge ladder, multi-model, v9 vs v10, budget.\n\n"
        "---\n\n" + kl + "\n---\n\n" + mm + "\n---\n\n" + v9v10 + "\n---\n\n" + budget
    )
    (EVAL / "final_eval_tables.md").write_text(final, encoding="utf-8")
    print("[w4] done.")


if __name__ == "__main__":
    main()
