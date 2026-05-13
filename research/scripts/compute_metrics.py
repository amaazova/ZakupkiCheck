"""W2 Session A — primary metrics + cluster-bootstrap CIs."""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .extraction_runner import read_jsonl

EVAL = PROJECT_ROOT / "workspace" / "eval"
MATCH_PATH = EVAL / "matching_results.jsonl"
TABLES_PATH = EVAL / "eval_tables.md"
EVID_PATH = EVAL / "evidence_verification.md"

SPECIFIC_TYPES = frozenset({
    "brand_without_equivalent", "incomplete_description",
    "restrictive_requirement", "ktru_mismatch",
    "inconsistent_dates", "missing_acceptance_terms",
})
STRATA = ("benchmarkable", "conditional", "service_only")
BOOTSTRAP_ITERS = 1000
BOOTSTRAP_ALPHA = 0.05
RNG_SEED = 20260512


# Fact-level metrics

def fact_pr(records: list[dict], *, mode: str) -> tuple[float, float, float, int, int, int]:
    """Fact-level precision/recall/F1 over violation_established episodes only.

    mode = "all" or "specific".
    """
    if mode not in ("all", "specific"):
        raise ValueError(f"bad mode: {mode}")
    sum_matched = 0
    sum_gold = 0     # established FAS findings
    sum_pred = 0     # TZ flags emitted on these episodes
    for r in records:
        if r["fas_verdict"] != "violation_established":
            continue
        if mode == "all":
            sum_matched += r["type_matches_all"]
            sum_gold += r["n_fas_findings_established"]
            sum_pred += r["n_tz_flags"]
        else:
            sum_matched += r["type_matches_specific"]
            sum_gold += r["n_fas_eligible_specific"]
            sum_pred += r["n_tz_eligible_specific"]
    p = sum_matched / sum_pred if sum_pred else 0.0
    r_ = sum_matched / sum_gold if sum_gold else 0.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    return p, r_, f1, sum_matched, sum_gold, sum_pred


def fact_pr_l0(records: list[dict]) -> tuple[float, float, float, int, int, int]:
    """L0 ladder: only brand_without_equivalent matched flags."""
    sum_matched = 0
    sum_gold = 0
    sum_pred = 0
    for r in records:
        if r["fas_verdict"] != "violation_established":
            continue
        sum_matched += r["type_matches_l0_brand"]
        n_brand_gold = sum(
            1 for _ in range(r["type_matches_l0_brand"])  # placeholder; will be replaced below
        )
        # We need finer detail: count gold brand findings per episode.
        # That data isn't on the record — recompute below using the original fas file.
        sum_pred += r["n_l0_flags"]
    return None  # see below; this function shape is wrong — we replace it.


# Replacement: load fas + l0 to count per-type denominators directly.

def ladder_brand_pr(records: list[dict], fas_by_ep: dict[str, dict]) -> dict[str, dict]:
    """Per-stratum P/R for L0 brand_without_equivalent vs L1 brand_without_equivalent.

    Restricted to gold violation_established episodes (as elsewhere in fact metrics).
    """
    out: dict[str, dict] = {}
    keys = list(STRATA) + ["pooled"]
    for s in keys:
        out[s] = {
            "l0_matched": 0, "l1_matched": 0,
            "l0_pred": 0, "l1_pred": 0,
            "gold_brand": 0,
        }
    for r in records:
        if r["fas_verdict"] != "violation_established":
            continue
        s = r["stratum"]
        fas_rec = fas_by_ep.get(r["episode_id"], {})
        n_brand_gold = sum(
            1 for f in (fas_rec.get("findings") or [])
            if f.get("severity") == "established"
            and f.get("finding_type") == "brand_without_equivalent"
        )
        # Count L1 brand_without_equivalent flags emitted on this episode.
        # We need it directly — store via record? We don't have it precomputed.
        # The record has type_matches_specific (all specific types together) but
        # we only need the brand-type count. Compute via the raw L1 file once.
        for bucket in (s, "pooled"):
            out[bucket]["gold_brand"] += n_brand_gold
            out[bucket]["l0_matched"] += r["type_matches_l0_brand"]
            out[bucket]["l0_pred"] += r["n_l0_flags"]
    return out


# Episode-level metrics

def episode_metrics(records: list[dict], *, pred_key: str) -> dict:
    """Accuracy, macro-F1, per-class F1, Cohen's κ, confusion matrix.

    Binary: violation_established vs violation_not_established.
    """
    gold = [r["fas_verdict"] for r in records]
    pred = [r[pred_key] for r in records]
    n = len(gold)
    if n == 0:
        return dict(acc=0.0, f1_macro=0.0, f1_pos=0.0, f1_neg=0.0, kappa=0.0,
                    tp=0, fp=0, fn=0, tn=0)
    pos = "violation_established"
    neg = "violation_not_established"
    tp = sum(1 for g, p in zip(gold, pred) if g == pos and p == pos)
    fp = sum(1 for g, p in zip(gold, pred) if g == neg and p == pos)
    fn = sum(1 for g, p in zip(gold, pred) if g == pos and p == neg)
    tn = sum(1 for g, p in zip(gold, pred) if g == neg and p == neg)
    acc = (tp + tn) / n
    prec_pos = tp / (tp + fp) if (tp + fp) else 0.0
    rec_pos = tp / (tp + fn) if (tp + fn) else 0.0
    prec_neg = tn / (tn + fn) if (tn + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) else 0.0
    f1_macro = (f1_pos + f1_neg) / 2
    # Cohen's κ
    p_obs = acc
    p_gold_pos = (tp + fn) / n
    p_gold_neg = 1 - p_gold_pos
    p_pred_pos = (tp + fp) / n
    p_pred_neg = 1 - p_pred_pos
    p_exp = p_gold_pos * p_pred_pos + p_gold_neg * p_pred_neg
    kappa = (p_obs - p_exp) / (1 - p_exp) if (1 - p_exp) else 0.0
    return dict(
        acc=acc, f1_macro=f1_macro, f1_pos=f1_pos, f1_neg=f1_neg, kappa=kappa,
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


# Bootstrap (cluster by notice_id)

def cluster_bootstrap(
    records: list[dict],
    metric_fn: Callable[[list[dict]], float],
    *,
    n_iter: int = BOOTSTRAP_ITERS,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Cluster bootstrap: resample notice_ids with replacement; return (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    by_notice: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_notice[r["notice_id"]].append(r)
    notice_ids = list(by_notice.keys())
    point = metric_fn(records)
    estimates = []
    n = len(notice_ids)
    for _ in range(n_iter):
        sample_ids = rng.choice(notice_ids, size=n, replace=True)
        sample_records: list[dict] = []
        for nid in sample_ids:
            sample_records.extend(by_notice[nid])
        try:
            estimates.append(metric_fn(sample_records))
        except Exception:
            estimates.append(float("nan"))
    arr = np.array([x for x in estimates if not math.isnan(x)])
    if arr.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return point, lo, hi


# Helpers

def fmt_ci(point: float, lo: float, hi: float, pct: bool = True) -> str:
    if math.isnan(point):
        return "n/a"
    if pct:
        return f"{point*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


def filter_stratum(records: list[dict], stratum: str) -> list[dict]:
    if stratum == "pooled":
        return records
    return [r for r in records if r["stratum"] == stratum]


# Reports

def fmt_count_pct(num: int, den: int) -> str:
    if den == 0:
        return f"{num}/0"
    return f"{num}/{den} ({num/den*100:.1f}%)"


def build_eval_tables(records: list[dict], fas_by_ep: dict[str, dict],
                      l1_by_ep: dict[str, dict], l0_by_ep: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# W2 Session A — Evaluation tables\n")
    lines.append(f"_n_episodes = {len(records)}; bootstrap = {BOOTSTRAP_ITERS} iters, "
                 f"cluster by `notice_id`, α = {BOOTSTRAP_ALPHA}_\n")

    # ----- 1. Fact-level P/R/F1 (stratified) -----
    lines.append("## 1. Fact-level metrics — gold violation episodes only\n")
    lines.append("Each row pools all FAS-established findings vs TZ flags on those episodes.\n")
    for mode_label, mode in (("all_types (incl. `other`)", "all"),
                             ("specific_only (6 of 7 types)", "specific")):
        lines.append(f"\n### Mode: {mode_label}\n")
        lines.append("| stratum | precision | recall | F1 | matched | gold est. | pred flags |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in list(STRATA) + ["pooled"]:
            sub = filter_stratum(records, s)
            p, r_, f1, m, g, pr = fact_pr(sub, mode=mode)
            # CIs
            p_pt, p_lo, p_hi = cluster_bootstrap(sub, lambda rs: fact_pr(rs, mode=mode)[0])
            r_pt, r_lo, r_hi = cluster_bootstrap(sub, lambda rs: fact_pr(rs, mode=mode)[1])
            f_pt, f_lo, f_hi = cluster_bootstrap(sub, lambda rs: fact_pr(rs, mode=mode)[2])
            lines.append(f"| {s} | {fmt_ci(p, p_lo, p_hi)} | {fmt_ci(r_, r_lo, r_hi)} | "
                         f"{fmt_ci(f1, f_lo, f_hi)} | {m} | {g} | {pr} |")

    # ----- 2. Episode-level accuracy / F1 / κ -----
    lines.append("\n## 2. Episode-level metrics — all 781 episodes\n")
    for label, key in (("strict (≥1 type-matched specific finding → violation)", "episode_pred_strict"),
                       ("loose (≥1 any L1 TZ flag → violation)", "episode_pred_loose")):
        lines.append(f"\n### Mode: {label}\n")
        lines.append("| stratum | accuracy | F1 (macro) | F1 (violation) | F1 (no-violation) | κ | TP/FP/FN/TN |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in list(STRATA) + ["pooled"]:
            sub = filter_stratum(records, s)
            m = episode_metrics(sub, pred_key=key)
            acc_pt, acc_lo, acc_hi = cluster_bootstrap(sub, lambda rs: episode_metrics(rs, pred_key=key)["acc"])
            fm_pt, fm_lo, fm_hi = cluster_bootstrap(sub, lambda rs: episode_metrics(rs, pred_key=key)["f1_macro"])
            fp_pt, fp_lo, fp_hi = cluster_bootstrap(sub, lambda rs: episode_metrics(rs, pred_key=key)["f1_pos"])
            fn_pt, fn_lo, fn_hi = cluster_bootstrap(sub, lambda rs: episode_metrics(rs, pred_key=key)["f1_neg"])
            kp_pt, kp_lo, kp_hi = cluster_bootstrap(sub, lambda rs: episode_metrics(rs, pred_key=key)["kappa"])
            lines.append(f"| {s} | {fmt_ci(acc_pt, acc_lo, acc_hi)} | "
                         f"{fmt_ci(fm_pt, fm_lo, fm_hi)} | "
                         f"{fmt_ci(fp_pt, fp_lo, fp_hi)} | "
                         f"{fmt_ci(fn_pt, fn_lo, fn_hi)} | "
                         f"{fmt_ci(kp_pt, kp_lo, kp_hi, pct=False)} | "
                         f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} |")

    # ----- 3. Knowledge ladder L0 vs L1 (brand_without_equivalent) -----
    lines.append("\n## 3. Knowledge ladder — `brand_without_equivalent` only (L0 covers no other type)\n")
    lines.append("Per stratum, restricted to gold violation_established episodes.\n")
    lines.append("| stratum | L0 precision | L0 recall | L1 precision | L1 recall | Δ recall (L1−L0) |")
    lines.append("|---|---|---|---|---|---|")

    def brand_metric(rs: list[dict], *, level: str, kind: str) -> float:
        sum_matched = 0
        sum_pred = 0
        sum_gold = 0
        for r in rs:
            if r["fas_verdict"] != "violation_established":
                continue
            fas_rec = fas_by_ep.get(r["episode_id"], {})
            n_brand_gold = sum(
                1 for f in (fas_rec.get("findings") or [])
                if f.get("severity") == "established"
                and f.get("finding_type") == "brand_without_equivalent"
            )
            sum_gold += n_brand_gold
            if level == "l0":
                sum_matched += r["type_matches_l0_brand"]
                sum_pred += r["n_l0_flags"]
            else:
                l1_rec = l1_by_ep.get(r["episode_id"], {})
                parsed = l1_rec.get("parsed") if isinstance(l1_rec.get("parsed"), dict) else {}
                l1_brand_flags = [
                    f for f in (parsed or {}).get("risk_flags") or []
                    if f.get("flag_type") == "brand_without_equivalent"
                ]
                # Number of L1 brand_without_equivalent matched to FAS for this ep:
                # We need to recompute by reading record's matched lists.
                # Use type_matches_specific is across all types — we don't have a per-type
                # breakdown of matches in the record. Build approximate by computing fresh.
                # Cheap path: use min(n_brand_gold, n_brand_pred) as a rough upper bound,
                # but we want exact. We'll compute exact via a small inner matching.
                # For brand_without_equivalent, both sides have known type. Just compare
                # in numbers: each L1 brand flag can match at most one gold brand finding,
                # so the matched count is min(n_gold, n_pred). This corresponds to the
                # type-only Level-1 assignment used elsewhere.
                sum_matched += min(n_brand_gold, len(l1_brand_flags))
                sum_pred += len(l1_brand_flags)
        if kind == "precision":
            return sum_matched / sum_pred if sum_pred else 0.0
        if kind == "recall":
            return sum_matched / sum_gold if sum_gold else 0.0
        raise ValueError(kind)

    for s in list(STRATA) + ["pooled"]:
        sub = filter_stratum(records, s)
        l0_p_pt, l0_p_lo, l0_p_hi = cluster_bootstrap(sub, lambda rs: brand_metric(rs, level="l0", kind="precision"))
        l0_r_pt, l0_r_lo, l0_r_hi = cluster_bootstrap(sub, lambda rs: brand_metric(rs, level="l0", kind="recall"))
        l1_p_pt, l1_p_lo, l1_p_hi = cluster_bootstrap(sub, lambda rs: brand_metric(rs, level="l1", kind="precision"))
        l1_r_pt, l1_r_lo, l1_r_hi = cluster_bootstrap(sub, lambda rs: brand_metric(rs, level="l1", kind="recall"))
        delta = l1_r_pt - l0_r_pt
        lines.append(
            f"| {s} | {fmt_ci(l0_p_pt, l0_p_lo, l0_p_hi)} | "
            f"{fmt_ci(l0_r_pt, l0_r_lo, l0_r_hi)} | "
            f"{fmt_ci(l1_p_pt, l1_p_lo, l1_p_hi)} | "
            f"{fmt_ci(l1_r_pt, l1_r_lo, l1_r_hi)} | "
            f"{delta*100:+.1f} pp |"
        )

    lines.append("\n## 4. Comparison with v9 baselines (accuracy, pooled)\n")
    pooled_strict = episode_metrics(records, pred_key="episode_pred_strict")
    pooled_loose  = episode_metrics(records, pred_key="episode_pred_loose")
    lines.append("| model | accuracy |")
    lines.append("|---|---|")
    lines.append("| B0 majority | 62.1% |")
    lines.append("| B1 regex | 53.4% |")
    lines.append("| B2 zero-shot | 50.7% |")
    lines.append("| B4 pipeline (v9) | 54.7% |")
    lines.append(f"| **W2 strict** | **{pooled_strict['acc']*100:.1f}%** |")
    lines.append(f"| **W2 loose**  | **{pooled_loose['acc']*100:.1f}%** |")

    # Gate check
    gate_kappa = pooled_strict["kappa"]
    lines.append("\n## 5. Gate check\n")
    lines.append(f"Pooled κ (strict) = **{gate_kappa:.3f}**  → "
                 f"{'PASS (≥ 0.20)' if gate_kappa >= 0.20 else '**FAIL** (< 0.20)'}")
    return "\n".join(lines) + "\n"


def build_evidence_verification(records: list[dict], fas_by_ep: dict[str, dict],
                                l1_by_ep: dict[str, dict]) -> str:
    """Citation containment, hallucination, entity-overlap report."""
    lines: list[str] = []
    lines.append("# W2 Session A — Evidence verification\n")

    # FAS citation containment over ALL established findings
    flat_fas_cit: list[bool] = []
    n_fas_quote_empty = 0
    for r in records:
        flat_fas_cit.extend(r["fas_citation_containment"])
        fas_rec = fas_by_ep.get(r["episode_id"], {})
        for f in (fas_rec.get("findings") or []):
            if f.get("severity") == "established":
                if not (f.get("evidence_quote") or "").strip():
                    n_fas_quote_empty += 1
    n_fas_total = len(flat_fas_cit)
    n_fas_hit = sum(flat_fas_cit)

    # TZ citation containment over matched_all pairs (matched_all is the strictest count)
    flat_tz_cit_all = [b for r in records for b in r["tz_citation_containment_all"]]
    flat_tz_cit_spec = [b for r in records for b in r["tz_citation_containment_specific"]]
    n_tz_total = len(flat_tz_cit_all)
    n_tz_hit = sum(flat_tz_cit_all)

    # TZ-level: containment over ALL emitted L1 flags (not just matched ones)
    # — computed by re-checking the L1 risk_flags against tz.md isn't free here;
    # rather, we approximate by reporting matched-pairs containment as the main number,
    # which is the spec ("TZ evidence_quote реально есть в tz.md"). To get the global
    # rate over all emitted flags, we'd need to recompute over the L1 file — skip for now.

    # Hallucination = 1 - containment_rate
    fas_cont = n_fas_hit / n_fas_total if n_fas_total else 0.0
    tz_cont_all = n_tz_hit / n_tz_total if n_tz_total else 0.0
    tz_cont_spec = sum(flat_tz_cit_spec) / len(flat_tz_cit_spec) if flat_tz_cit_spec else 0.0

    # Entity overlap (mean Jaccard over matched pairs)
    all_entity = [s for r in records for s in r["entity_overlap_scores_all"]]
    spec_entity = [s for r in records for s in r["entity_overlap_scores_specific"]]
    all_sem = [s for r in records for s in r["semantic_sim_scores_all"]]
    spec_sem = [s for r in records for s in r["semantic_sim_scores_specific"]]

    lines.append("## 1. Citation containment (per finding/flag, evidence_quote vs source doc)\n")
    lines.append("| metric | n_total | n_contained | rate | hallucination |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| FAS established findings → fas.json | {n_fas_total} | {n_fas_hit} | "
                 f"{fas_cont*100:.1f}% | {(1-fas_cont)*100:.1f}% |")
    lines.append(f"| TZ flags (matched, all types) → tz.md | {n_tz_total} | {n_tz_hit} | "
                 f"{tz_cont_all*100:.1f}% | {(1-tz_cont_all)*100:.1f}% |")
    lines.append(f"| TZ flags (matched, specific) → tz.md | {len(flat_tz_cit_spec)} | "
                 f"{sum(flat_tz_cit_spec)} | {tz_cont_spec*100:.1f}% | "
                 f"{(1-tz_cont_spec)*100:.1f}% |")
    lines.append(f"\n_FAS established findings with empty evidence_quote: {n_fas_quote_empty}_\n")

    lines.append("\n## 2. Entity-overlap Jaccard (named entities in evidence_quote/description)\n")
    if all_entity:
        lines.append("| matched-pair set | n | mean | median | p10 | p90 |")
        lines.append("|---|---|---|---|---|---|")
        for label, arr in (("all_types", all_entity), ("specific_only", spec_entity)):
            if not arr:
                continue
            arr_s = sorted(arr)
            lines.append(f"| {label} | {len(arr)} | {statistics.mean(arr):.3f} | "
                         f"{statistics.median(arr):.3f} | {arr_s[len(arr)//10]:.3f} | "
                         f"{arr_s[len(arr)*9//10]:.3f} |")

    lines.append("\n## 3. Semantic similarity (multilingual MiniLM, cosine)\n")
    if all_sem:
        lines.append("| matched-pair set | n | mean | median | p10 | p90 | ≥0.5 | ≥0.6 | ≥0.7 | ≥0.8 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for label, arr in (("all_types", all_sem), ("specific_only", spec_sem)):
            if not arr:
                continue
            arr_s = sorted(arr)
            n = len(arr)
            ge = [sum(1 for x in arr if x >= t) for t in (0.5, 0.6, 0.7, 0.8)]
            lines.append(f"| {label} | {n} | {statistics.mean(arr):.3f} | "
                         f"{statistics.median(arr):.3f} | {arr_s[n//10]:.3f} | "
                         f"{arr_s[n*9//10]:.3f} | "
                         + " | ".join(f"{c} ({c/n*100:.0f}%)" for c in ge) + " |")

    # Per-type FAS containment breakdown
    type_cont: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        fas_rec = fas_by_ep.get(r["episode_id"], {})
        cits = r["fas_citation_containment"]
        i = 0
        for f in (fas_rec.get("findings") or []):
            if f.get("severity") == "established":
                type_cont[f.get("finding_type", "?")].append(cits[i] if i < len(cits) else False)
                i += 1
    lines.append("\n## 4. FAS citation containment by finding_type\n")
    lines.append("| finding_type | n_established | containment_rate |")
    lines.append("|---|---|---|")
    for t in sorted(type_cont, key=lambda k: -len(type_cont[k])):
        v = type_cont[t]
        rate = sum(v) / len(v) if v else 0.0
        lines.append(f"| {t} | {len(v)} | {rate*100:.1f}% |")

    return "\n".join(lines) + "\n"


# Main

def main() -> None:
    print(f"[metrics] reading {MATCH_PATH.name}...")
    records = read_jsonl(MATCH_PATH)
    fas = read_jsonl(EVAL / "fas_findings.jsonl")
    l1 = read_jsonl(EVAL / "tz_features_L1.jsonl")
    l0 = read_jsonl(EVAL / "tz_features_L0.jsonl")
    fas_by_ep = {r["episode_id"]: r for r in fas}
    l1_by_ep = {r["episode_id"]: r for r in l1}
    l0_by_ep = {r["episode_id"]: r for r in l0}
    print(f"[metrics] records={len(records)}")

    print("[metrics] building eval_tables.md...")
    txt = build_eval_tables(records, fas_by_ep, l1_by_ep, l0_by_ep)
    TABLES_PATH.write_text(txt, encoding="utf-8")
    print(f"[metrics] → {TABLES_PATH}")

    print("[metrics] building evidence_verification.md...")
    txt2 = build_evidence_verification(records, fas_by_ep, l1_by_ep)
    EVID_PATH.write_text(txt2, encoding="utf-8")
    print(f"[metrics] → {EVID_PATH}")

    # Print gate decision to stdout
    pooled = episode_metrics(records, pred_key="episode_pred_strict")
    print(f"\n=== GATE: pooled κ (strict) = {pooled['kappa']:.3f}  "
          f"({'PASS' if pooled['kappa'] >= 0.2 else 'FAIL — investigate'}) ===")


if __name__ == "__main__":
    main()
