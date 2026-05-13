"""W3 Session A — Ablations A6, A7, A8."""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .compute_metrics import (
    BOOTSTRAP_ALPHA, BOOTSTRAP_ITERS, RNG_SEED, cluster_bootstrap, fmt_ci,
)
from .config import PROJECT_ROOT
from .extraction_runner import read_jsonl

EVAL = PROJECT_ROOT / "workspace" / "eval"


SPECIFIC_TYPES = frozenset({
    "brand_without_equivalent", "incomplete_description",
    "restrictive_requirement", "ktru_mismatch",
    "inconsistent_dates", "missing_acceptance_terms",
})

FUZZY_MAP = {
    "brand_without_equivalent": {"brand_without_equivalent", "restrictive_requirement"},
    "restrictive_requirement": {"restrictive_requirement", "brand_without_equivalent"},
    "incomplete_description": {"incomplete_description", "missing_acceptance_terms"},
    "ktru_mismatch": {"ktru_mismatch", "incomplete_description"},
    "inconsistent_dates": {"inconsistent_dates"},
    "missing_acceptance_terms": {"missing_acceptance_terms", "incomplete_description"},
    "other": {"other", "incomplete_description", "restrictive_requirement"},
}

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STRATA = ("benchmarkable", "conditional", "service_only")
EMBED_THRESHOLD_DEFAULT = 0.6
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


# Embed once

def collect_quotes(fas: list[dict], tz_l1: list[dict]) -> list[str]:
    quotes: set[str] = set()
    for r in fas:
        for f in r.get("findings") or []:
            q = (f.get("evidence_quote") or "").strip()
            if q:
                quotes.add(q)
            d = (f.get("description") or "").strip()
            if d:
                quotes.add(d)
    for r in tz_l1:
        p = r.get("parsed")
        if not isinstance(p, dict):
            continue
        for f in p.get("risk_flags") or []:
            q = (f.get("evidence_quote") or "").strip()
            if q:
                quotes.add(q)
    return list(quotes)


def encode_quotes(quotes: list[str]) -> dict[str, np.ndarray]:
    print(f"[ablations] embedding {len(quotes)} quotes...")
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(
        quotes, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return {q: e for q, e in zip(quotes, embs)}


# Strategy-based matching (returns per-episode counts)

def cos_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


def match_with_strategy(
    fas_findings: list[dict],
    tz_flags: list[dict],
    *,
    strategy: str,
    quote_emb: dict[str, np.ndarray],
    threshold: float = EMBED_THRESHOLD_DEFAULT,
) -> dict:
    """Returns dict with matched, n_fas_eligible (specific established), n_tz_eligible (specific)."""
    fas_eligible = [
        f for f in fas_findings
        if f.get("severity") == "established"
        and f.get("finding_type") in SPECIFIC_TYPES
    ]
    tz_eligible = [t for t in tz_flags if t.get("flag_type") in SPECIFIC_TYPES]

    used_tz: set[int] = set()
    matched = 0
    for ff in fas_eligible:
        ft = ff.get("finding_type")
        fq = (ff.get("evidence_quote") or "").strip()
        f_emb = quote_emb.get(fq)

        candidates: list[tuple[float, int]] = []  # (sem, ti)
        for ti, tf in enumerate(tz_eligible):
            if ti in used_tz:
                continue
            tt = tf.get("flag_type")
            if strategy == "exact_type":
                ok = (tt == ft)
            elif strategy == "fuzzy_type":
                ok = tt in FUZZY_MAP.get(ft, {ft})
            elif strategy == "embedding_match":
                tq = (tf.get("evidence_quote") or "").strip()
                t_emb = quote_emb.get(tq)
                ok = cos_sim(f_emb, t_emb) >= threshold
            else:
                raise ValueError(strategy)
            if not ok:
                continue
            tq = (tf.get("evidence_quote") or "").strip()
            t_emb = quote_emb.get(tq)
            sem = cos_sim(f_emb, t_emb)
            candidates.append((sem, ti))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        used_tz.add(candidates[0][1])
        matched += 1
    return {
        "matched": matched,
        "n_fas_eligible": len(fas_eligible),
        "n_tz_eligible": len(tz_eligible),
    }


# Episode-level summary per strategy

def per_episode_strategy_counts(
    fas: list[dict], tz_l1: list[dict],
    eval_df: pd.DataFrame, quote_emb: dict[str, np.ndarray],
    *, strategy: str, threshold: float = EMBED_THRESHOLD_DEFAULT,
) -> list[dict]:
    fas_by_ep = {r["episode_id"]: r for r in fas}
    l1_by_ep = {r["episode_id"]: r for r in tz_l1}
    out = []
    for _, row in eval_df.iterrows():
        eid = row["episode_id"]
        fas_rec = fas_by_ep.get(eid, {})
        l1_rec = l1_by_ep.get(eid, {})
        parsed = l1_rec.get("parsed") if isinstance(l1_rec.get("parsed"), dict) else {}
        tz_flags = (parsed or {}).get("risk_flags") or []
        r = match_with_strategy(
            fas_rec.get("findings") or [], tz_flags,
            strategy=strategy, quote_emb=quote_emb, threshold=threshold,
        )
        out.append({
            "episode_id": eid,
            "notice_id": row["notice_id"],
            "stratum": row["stratum"],
            "fas_verdict": row["fas_verdict"],
            "matched": r["matched"],
            "n_fas_eligible": r["n_fas_eligible"],
            "n_tz_eligible": r["n_tz_eligible"],
        })
    return out


def fact_pr_from_counts(records: list[dict]) -> tuple[float, float, float, int, int, int]:
    sum_m = sum_g = sum_p = 0
    for r in records:
        if r["fas_verdict"] != "violation_established":
            continue
        sum_m += r["matched"]
        sum_g += r["n_fas_eligible"]
        sum_p += r["n_tz_eligible"]
    p = sum_m / sum_p if sum_p else 0.0
    r_ = sum_m / sum_g if sum_g else 0.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    return p, r_, f1, sum_m, sum_g, sum_p


def filter_stratum(records, s):
    return records if s == "pooled" else [r for r in records if r["stratum"] == s]


# A6 — matching strategy comparison

def run_a6(fas, tz_l1, eval_df, quote_emb) -> str:
    lines: list[str] = []
    lines.append("# A6 — Matching strategy comparison\n")
    lines.append(f"_specific_only mode; bootstrap = {BOOTSTRAP_ITERS} iters, "
                 f"cluster by notice_id, α = {BOOTSTRAP_ALPHA}_\n")

    strategy_counts: dict[str, list[dict]] = {}
    for strategy in ("exact_type", "fuzzy_type", "embedding_match"):
        print(f"[A6] strategy={strategy}")
        strategy_counts[strategy] = per_episode_strategy_counts(
            fas, tz_l1, eval_df, quote_emb,
            strategy=strategy, threshold=EMBED_THRESHOLD_DEFAULT,
        )

    for strategy in ("exact_type", "fuzzy_type", "embedding_match"):
        cs = strategy_counts[strategy]
        title = strategy
        if strategy == "embedding_match":
            title += f" (threshold={EMBED_THRESHOLD_DEFAULT})"
        lines.append(f"\n## {title}\n")
        lines.append("| stratum | precision | recall | F1 | matched | gold est. | pred flags |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in list(STRATA) + ["pooled"]:
            sub = filter_stratum(cs, s)
            p, r_, f1, m, g, pr = fact_pr_from_counts(sub)
            p_pt, p_lo, p_hi = cluster_bootstrap(sub, lambda rs: fact_pr_from_counts(rs)[0])
            r_pt, r_lo, r_hi = cluster_bootstrap(sub, lambda rs: fact_pr_from_counts(rs)[1])
            f_pt, f_lo, f_hi = cluster_bootstrap(sub, lambda rs: fact_pr_from_counts(rs)[2])
            lines.append(f"| {s} | {fmt_ci(p, p_lo, p_hi)} | {fmt_ci(r_, r_lo, r_hi)} | "
                         f"{fmt_ci(f1, f_lo, f_hi)} | {m} | {g} | {pr} |")

    # Summary delta vs exact_type (pooled)
    lines.append("\n## Δ vs exact_type (pooled)\n")
    lines.append("| strategy | Δ precision | Δ recall | Δ F1 |")
    lines.append("|---|---|---|---|")
    base = fact_pr_from_counts(strategy_counts["exact_type"])
    for strategy in ("fuzzy_type", "embedding_match"):
        cur = fact_pr_from_counts(strategy_counts[strategy])
        lines.append(f"| {strategy} | {(cur[0]-base[0])*100:+.1f} pp | "
                     f"{(cur[1]-base[1])*100:+.1f} pp | {(cur[2]-base[2])*100:+.1f} pp |")

    return "\n".join(lines) + "\n", strategy_counts


# A7 — threshold sensitivity (uses matching_results.jsonl directly)

def run_a7(matching_records: list[dict]) -> tuple[str, dict]:
    """Filter existing exact-type matches by semantic_sim >= threshold."""
    rows: list[dict] = []

    # For each threshold, recompute fact P/R/F1 on records (specific_only)
    # We rebuild per-episode counts by walking semantic_sim_scores_specific.
    by_threshold: dict[float, dict] = {}
    for t in THRESHOLDS:
        per_ep = []
        for r in matching_records:
            sims = r["semantic_sim_scores_specific"]
            matched_kept = sum(1 for s in sims if s >= t)
            per_ep.append({
                "fas_verdict": r["fas_verdict"],
                "notice_id": r["notice_id"],
                "stratum": r["stratum"],
                "matched": matched_kept,
                "n_fas_eligible": r["n_fas_eligible_specific"],
                "n_tz_eligible": r["n_tz_eligible_specific"],
            })
        p, r_, f1, m, g, pr = fact_pr_from_counts(per_ep)
        p_pt, p_lo, p_hi = cluster_bootstrap(per_ep, lambda rs: fact_pr_from_counts(rs)[0])
        r_pt, r_lo, r_hi = cluster_bootstrap(per_ep, lambda rs: fact_pr_from_counts(rs)[1])
        f_pt, f_lo, f_hi = cluster_bootstrap(per_ep, lambda rs: fact_pr_from_counts(rs)[2])
        by_threshold[t] = dict(
            point=(p, r_, f1, m, g, pr),
            ci=((p_pt, p_lo, p_hi), (r_pt, r_lo, r_hi), (f_pt, f_lo, f_hi)),
        )

    lines: list[str] = []
    lines.append("# A7 — Semantic-similarity threshold sweep\n")
    lines.append("Starting from exact-type matches (specific_only), filter pairs where "
                 "`semantic_sim ≥ threshold`. As threshold rises, fewer pairs survive — "
                 "recall drops, precision rises (with respect to the original-mode denominators).\n")
    lines.append("| threshold | precision | recall | F1 | matched / gold | pred flags |")
    lines.append("|---|---|---|---|---|---|")
    for t in THRESHOLDS:
        info = by_threshold[t]
        p, r_, f1, m, g, pr = info["point"]
        (pp, plo, phi), (rp, rlo, rhi), (fp_, flo, fhi) = info["ci"]
        lines.append(f"| {t:.1f} | {fmt_ci(p, plo, phi)} | "
                     f"{fmt_ci(r_, rlo, rhi)} | {fmt_ci(f1, flo, fhi)} | {m}/{g} | {pr} |")
    lines.append("\nDenominators (gold est., pred flags) are constant across thresholds — "
                 "only the matched count changes.\n")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    p_arr = [by_threshold[t]["point"][0] for t in THRESHOLDS]
    r_arr = [by_threshold[t]["point"][1] for t in THRESHOLDS]
    f_arr = [by_threshold[t]["point"][2] for t in THRESHOLDS]
    ax.plot(THRESHOLDS, p_arr, marker="o", label="precision")
    ax.plot(THRESHOLDS, r_arr, marker="s", label="recall")
    ax.plot(THRESHOLDS, f_arr, marker="^", label="F1")
    ax.set_xlabel("semantic similarity threshold")
    ax.set_ylabel("score")
    ax.set_title("A7 — semantic threshold sweep (specific_only, gold violations)")
    ax.set_xticks(list(THRESHOLDS))
    ax.set_ylim(0, max(0.6, max(p_arr + r_arr + f_arr) * 1.1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = EVAL / "ablation_a7_threshold_plot.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    lines.append(f"![A7 plot]({plot_path.name})\n")
    return "\n".join(lines) + "\n", by_threshold


# A8 — TZ length sensitivity

def run_a8(matching_records: list[dict], eval_df: pd.DataFrame) -> str:
    # Per-episode fact_recall, only on gold violation_established
    by_ep = {r["episode_id"]: r for r in matching_records}
    tier_lookup = dict(zip(eval_df["episode_id"], eval_df["quality_tier"]))
    char_lookup = dict(zip(eval_df["episode_id"], eval_df["tz_char_count"]))

    points = []  # (char_count, recall, stratum, tier)
    for r in matching_records:
        if r["fas_verdict"] != "violation_established":
            continue
        g = r["n_fas_eligible_specific"]
        m = r["type_matches_specific"]
        if g == 0:
            continue
        rec = m / g
        points.append({
            "char_count": int(char_lookup.get(r["episode_id"]) or 0),
            "recall": rec,
            "stratum": r["stratum"],
            "tier": tier_lookup.get(r["episode_id"], "?"),
        })

    # Per-tier table
    by_tier: dict[str, list[float]] = defaultdict(list)
    for p in points:
        by_tier[p["tier"]].append(p["recall"])

    lines: list[str] = []
    lines.append("# A8 — TZ length sensitivity\n")
    lines.append(f"_n_violation_episodes_with_gold_findings = {len(points)}_  \n"
                 "Recall computed per episode = matched_specific / gold_specific.\n")
    lines.append("## Per quality_tier\n")
    lines.append("| tier | n | mean fact_recall | median | p25 | p75 | mean TZ chars |")
    lines.append("|---|---|---|---|---|---|---|")
    tier_chars: dict[str, list[int]] = defaultdict(list)
    for p in points:
        tier_chars[p["tier"]].append(p["char_count"])
    for t in ("thin", "normal", "chunking_risk", "?"):
        v = by_tier.get(t, [])
        if not v:
            continue
        vs = sorted(v)
        mean = sum(v) / len(v)
        median = vs[len(vs)//2]
        p25 = vs[len(vs)//4] if len(vs) >= 4 else vs[0]
        p75 = vs[len(vs)*3//4] if len(vs) >= 4 else vs[-1]
        chars_arr = tier_chars[t]
        mean_chars = int(sum(chars_arr) / max(len(chars_arr), 1))
        lines.append(f"| {t} | {len(v)} | {mean:.3f} | {median:.3f} | "
                     f"{p25:.3f} | {p75:.3f} | {mean_chars:,} |")

    # Scatter
    fig, ax = plt.subplots(figsize=(8, 5.5))
    color_map = {"benchmarkable": "#1f77b4", "conditional": "#ff7f0e", "service_only": "#2ca02c"}
    for s, color in color_map.items():
        xs = [p["char_count"] for p in points if p["stratum"] == s]
        ys = [p["recall"] for p in points if p["stratum"] == s]
        ax.scatter(xs, ys, c=color, label=s, alpha=0.55, s=22, edgecolor="none")
    # LOESS smoothing via statsmodels lowess on log10(char_count) bandwidth=0.5
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xs_all = np.array([max(p["char_count"], 1) for p in points], dtype=float)
        ys_all = np.array([p["recall"] for p in points], dtype=float)
        # Use log10(char_count) as predictor for visual readability, then map back.
        sm = lowess(ys_all, np.log10(xs_all), frac=0.5, return_sorted=True)
        ax.plot(10**sm[:, 0], sm[:, 1], color="black", linewidth=2.0, label="LOESS")
    except Exception as e:
        print(f"[A8] LOESS smoothing skipped: {e}")
    ax.set_xscale("log")
    ax.set_xlabel("TZ char_count (log scale)")
    ax.set_ylabel("fact_recall per episode (specific_only)")
    ax.set_title("A8 — TZ length vs fact recall (gold violations)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plot_path = EVAL / "ablation_a8_scatter.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    lines.append(f"\n## Scatter\n\n![A8 scatter]({plot_path.name})\n")

    # Top-line observation
    if "normal" in by_tier and "thin" in by_tier and "chunking_risk" in by_tier:
        lines.append("\n## Observation\n")
        means = {t: statistics.mean(v) for t, v in by_tier.items() if v}
        lines.append("Mean fact_recall by tier: " +
                     ", ".join(f"{t}={means[t]:.3f}" for t in ("thin", "normal", "chunking_risk")
                               if t in means) + ".")
    return "\n".join(lines) + "\n"


# Driver

def main() -> None:
    eval_df = pd.read_csv(EVAL / "eval_dataset_v10.csv", dtype={"notice_id": str})
    fas = read_jsonl(EVAL / "fas_findings.jsonl")
    tz_l1 = read_jsonl(EVAL / "tz_features_L1.jsonl")
    matching = read_jsonl(EVAL / "matching_results.jsonl")
    print(f"[ablations] eval={len(eval_df)} fas={len(fas)} tz_l1={len(tz_l1)} match={len(matching)}")

    quotes = collect_quotes(fas, tz_l1)
    quote_emb = encode_quotes(quotes)

    # A6
    print("[ablations] A6 — matching strategies")
    a6_md, _strategy_counts = run_a6(fas, tz_l1, eval_df, quote_emb)
    (EVAL / "ablation_a6_matching_strategies.md").write_text(a6_md, encoding="utf-8")

    # A7
    print("[ablations] A7 — threshold sweep")
    a7_md, _ = run_a7(matching)
    (EVAL / "ablation_a7_threshold_sweep.md").write_text(a7_md, encoding="utf-8")

    # A8
    print("[ablations] A8 — length sensitivity")
    a8_md = run_a8(matching, eval_df)
    (EVAL / "ablation_a8_length_sensitivity.md").write_text(a8_md, encoding="utf-8")

    # Summary
    print("[ablations] summary → ablation_results.md")
    summary = [
        "# W3 Session A — Ablations A6/A7/A8 summary\n",
        "See:\n",
        "- [A6 — matching strategies](ablation_a6_matching_strategies.md)\n",
        "- [A7 — threshold sweep](ablation_a7_threshold_sweep.md)\n",
        "- [A8 — length sensitivity](ablation_a8_length_sensitivity.md)\n",
    ]
    (EVAL / "ablation_results.md").write_text("".join(summary), encoding="utf-8")
    print("[ablations] done.")


if __name__ == "__main__":
    main()
