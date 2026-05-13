"""W3 Session A — per-cluster diagnostic table."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .compute_metrics import episode_metrics, fact_pr
from .config import PROJECT_ROOT
from .extraction_runner import read_jsonl

EVAL = PROJECT_ROOT / "workspace" / "eval"
OUT = EVAL / "per_family_diagnostic.md"


def main() -> None:
    eval_df = pd.read_csv(EVAL / "eval_dataset_v10.csv", dtype={"notice_id": str})
    matching = read_jsonl(EVAL / "matching_results.jsonl")

    cluster_lookup = dict(zip(eval_df["episode_id"], eval_df["cluster_id"]))
    name_lookup    = dict(zip(eval_df["cluster_id"], eval_df["cluster_name"]))
    stratum_lookup = dict(zip(eval_df["episode_id"], eval_df["stratum"]))

    by_cluster: dict[int, list[dict]] = defaultdict(list)
    for r in matching:
        c = int(cluster_lookup[r["episode_id"]])
        by_cluster[c].append(r)

    lines: list[str] = []
    lines.append("# W3 Session A — Per-family (cluster) diagnostic\n")
    lines.append(
        "_Appendix-level. No bootstrap CIs — most clusters have n < 50 and the CIs "
        "would be uninformatively wide. Fact metrics use specific_only mode on gold "
        "violation episodes; episode accuracy uses strict prediction over all episodes."
        "_\n"
    )
    lines.append("| cluster | name | stratum | N | fact P | fact R | fact F1 | episode acc (strict) | gold violations |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    # Sort by stratum then by cluster id for readability
    stratum_order = {"benchmarkable": 0, "conditional": 1, "service_only": 2}
    clusters = sorted(
        by_cluster.keys(),
        key=lambda c: (stratum_order.get(stratum_lookup[by_cluster[c][0]["episode_id"]], 9), c),
    )
    for c in clusters:
        recs = by_cluster[c]
        n = len(recs)
        s = stratum_lookup[recs[0]["episode_id"]]
        name = name_lookup.get(c, "?")
        n_viol = sum(1 for r in recs if r["fas_verdict"] == "violation_established")
        p, r_, f1, m, g, pr = fact_pr(recs, mode="specific")
        epi = episode_metrics(recs, pred_key="episode_pred_strict")
        # Trim cluster name to fit table
        nm = name if len(name) <= 65 else name[:65].rstrip() + "…"
        lines.append(
            f"| c_{c:02d} | {nm} | {s} | {n} | "
            f"{p*100:.1f}% | {r_*100:.1f}% | {f1*100:.1f}% | "
            f"{epi['acc']*100:.1f}% | {n_viol} |"
        )

    # Summary blocks
    lines.append("\n## Stratum totals\n")
    lines.append("| stratum | N | episodes / stratum |")
    lines.append("|---|---|---|")
    by_s: dict[str, int] = defaultdict(int)
    for r in matching:
        by_s[r["stratum"]] += 1
    for s in ("benchmarkable", "conditional", "service_only"):
        lines.append(f"| {s} | {by_s.get(s, 0)} | — |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[per_family] → {OUT}")


if __name__ == "__main__":
    main()
