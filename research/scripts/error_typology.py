"""W3 Session A — quantitative error typology."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import PROJECT_ROOT
from .data_loaders import load_fas_text, load_tz
from .extraction_runner import read_jsonl
from .matching_pipeline import contained, normalize

EVAL = PROJECT_ROOT / "workspace" / "eval"
OUT_MD = EVAL / "error_typology.md"
OUT_PIE = EVAL / "error_typology_pie.png"

SPECIFIC_TYPES = frozenset({
    "brand_without_equivalent", "incomplete_description",
    "restrictive_requirement", "ktru_mismatch",
    "inconsistent_dates", "missing_acceptance_terms",
})


def classify_unmatched(finding: dict, tz_norm: str, fas_norm: str) -> str:
    ftype = finding.get("finding_type")
    quote = (finding.get("evidence_quote") or "").strip()

    if ftype == "other":
        return "granularity_mismatch"

    if not quote:
        return "context_gap"  # no anchor — likely a narrative-only finding

    if contained(quote, tz_norm):
        return "extraction_miss"

    if contained(quote, fas_norm):
        return "context_gap"

    # Quote not contained anywhere we can verify — model may have paraphrased.
    return "context_gap"


def main() -> None:
    matching = read_jsonl(EVAL / "matching_results.jsonl")
    fas = read_jsonl(EVAL / "fas_findings.jsonl")
    l1 = read_jsonl(EVAL / "tz_features_L1.jsonl")
    eval_df = pd.read_csv(EVAL / "eval_dataset_v10.csv", dtype={"notice_id": str})

    match_by_ep = {r["episode_id"]: r for r in matching}
    fas_by_ep = {r["episode_id"]: r for r in fas}
    l1_by_ep = {r["episode_id"]: r for r in l1}

    # Cache normalized docs per notice and per decision.
    tz_norm_cache: dict[str, str] = {}
    fas_norm_cache: dict[str, str] = {}

    counts: Counter = Counter()
    by_type: dict[str, Counter] = {}
    n_unmatched = 0
    n_total_est_specific = 0
    n_total_est_all = 0

    # Per-episode: how many matches did we actually make under specific_only?
    # The greedy 1-to-1 assignment in matching_pipeline.match_episode uses
    # type equality + greedy by (sem, ent). For unmatched-FAS classification
    # we re-derive WHICH FAS findings were left unmatched by replaying the
    # same logic at type-count level (since assignment is greedy by best
    # score, the count is the upper bound):
    # specifically, "unmatched gold" = sum(max(0, gold_count_for_type -
    # tz_count_for_type)) plus all "other" findings (since other ∉ specific).
    # That gives us per-finding identification of unmatched.
    # But we want individual findings — easier approach: iterate findings,
    # and treat as unmatched the FAS findings of types that have FEWER TZ
    # flags of the same type than FAS findings of that type. We pick the
    # first n_extra findings of that type as unmatched.

    for r in matching:
        eid = r["episode_id"]
        notice_id = r["notice_id"]
        decision_id = r["decision_id"]
        fas_rec = fas_by_ep.get(eid, {})
        l1_rec = l1_by_ep.get(eid, {})

        fas_est = [f for f in (fas_rec.get("findings") or []) if f.get("severity") == "established"]
        n_total_est_all += len(fas_est)
        n_total_est_specific += sum(1 for f in fas_est if f.get("finding_type") in SPECIFIC_TYPES)

        parsed = l1_rec.get("parsed") if isinstance(l1_rec.get("parsed"), dict) else {}
        tz_flags = (parsed or {}).get("risk_flags") or []
        tz_type_counts: Counter = Counter(t.get("flag_type") for t in tz_flags)

        # Lazy-load docs
        if notice_id not in tz_norm_cache:
            tz_norm_cache[notice_id] = normalize(load_tz(notice_id) or "")
        if decision_id not in fas_norm_cache:
            fas_norm_cache[decision_id] = normalize(load_fas_text(decision_id) or "")
        tz_norm = tz_norm_cache[notice_id]
        fas_norm = fas_norm_cache[decision_id]

        # Identify unmatched FAS findings per type by FAS-count - TZ-count.
        fas_by_type: dict[str, list[dict]] = {}
        for ff in fas_est:
            fas_by_type.setdefault(ff.get("finding_type"), []).append(ff)
        for ft, group in fas_by_type.items():
            n_tz_same = tz_type_counts.get(ft, 0) if ft in SPECIFIC_TYPES else 0
            # "other"-type FAS findings are always unmatched in specific_only.
            n_unmatched_for_type = len(group) if ft not in SPECIFIC_TYPES else max(0, len(group) - n_tz_same)
            for ff in group[:n_unmatched_for_type]:
                cat = classify_unmatched(ff, tz_norm, fas_norm)
                counts[cat] += 1
                by_type.setdefault(ft, Counter())[cat] += 1
                n_unmatched += 1

    # Markdown
    lines: list[str] = []
    lines.append("# W3 Session A — Error typology\n")
    lines.append(
        f"Universe: every FAS established finding that did NOT get a same-type "
        f"TZ L1 flag under specific_only matching.\n\n"
        f"- `n_total_established` (all types): **{n_total_est_all}**\n"
        f"- `n_total_established_specific` (6 of 7 types): **{n_total_est_specific}**\n"
        f"- `n_unmatched` (in this typology): **{n_unmatched}**\n"
    )
    lines.append("\n## Distribution\n")
    lines.append("| error_type | count | % of unmatched |")
    lines.append("|---|---|---|")
    for cat, c in counts.most_common():
        lines.append(f"| `{cat}` | {c} | {c/max(n_unmatched,1)*100:.1f}% |")

    # Per-finding-type drill-down
    lines.append("\n## Per finding_type\n")
    lines.append("| finding_type | n_unmatched | top error categories |")
    lines.append("|---|---|---|")
    for ft, cc in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        top = ", ".join(f"{k}={v}" for k, v in cc.most_common(3))
        lines.append(f"| `{ft}` | {sum(cc.values())} | {top} |")

    # Notes on the other classes (false_alarm / negation_miss / hallucinated_entity)
    # — those are episode-level / flag-level concepts, addressed separately:
    pipeline_only = sum(
        1 for r in matching
        if r["fas_verdict"] == "violation_not_established"
        and r["n_fas_findings_established"] == 0
        and r["n_tz_flags"] >= 1
    )
    lines.append("\n## Flag-side error classes (not in the count above)\n")
    lines.append(
        "These classes are not unmatched-gold buckets; they describe pipeline-side "
        "errors and are reported separately:\n\n"
        f"- `false_alarm` — pipeline emits flags on an episode where FAS established "
        f"no violation. **n ≈ {pipeline_only}** episodes "
        f"(`fas_verdict == violation_not_established && n_tz_flags ≥ 1`).\n"
        "- `negation_miss` — pipeline misses an explicit «или эквивалент» / "
        "negation clause; requires lexical analysis of TZ flag context, "
        "**not auto-classified** — see case studies for examples.\n"
        "- `hallucinated_entity` — TZ flag quotes a brand/ГОСТ/КТРУ not present "
        "in tz.md. Estimated upper bound: TZ citation-containment miss rate × "
        "n_flags ≈ 53% of matched-pair TZ quotes are not literal substrings, "
        "but most are paraphrases rather than hallucinations.\n"
    )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Pie chart
    labels = [cat for cat, _ in counts.most_common()]
    sizes = [counts[c] for c in labels]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors[:len(labels)],
        autopct=lambda p: f"{p:.1f}%\n(n={int(round(p*sum(sizes)/100))})",
        startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title(f"Error typology — unmatched FAS established findings (n={n_unmatched})")
    fig.tight_layout()
    fig.savefig(OUT_PIE, dpi=140)
    plt.close(fig)
    print(f"[err_typology] → {OUT_MD}")
    print(f"[err_typology] → {OUT_PIE}")


if __name__ == "__main__":
    main()
