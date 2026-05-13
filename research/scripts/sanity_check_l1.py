"""W1 Session B — sanity check for L1 (TZ open extraction)."""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

from .config import PROJECT_ROOT
from .extraction_runner import read_jsonl

EVAL = PROJECT_ROOT / "workspace" / "eval"


def main() -> None:
    l1 = read_jsonl(EVAL / "tz_features_L1.jsonl")
    if not l1:
        print("ERROR: tz_features_L1.jsonl empty or missing.")
        return
    l0 = read_jsonl(EVAL / "tz_features_L0.jsonl")
    l0_by_id = {r["episode_id"]: r for r in l0}

    n_total = len(l1)
    by_status = Counter(r.get("status") for r in l1)
    # Some models occasionally return a bare JSON array instead of an object.
    # Treat those as parsed-but-empty so the rest of the analysis is robust.
    for r in l1:
        if r.get("parsed") is not None and not isinstance(r["parsed"], dict):
            r["parsed"] = {}
    parsed = [r for r in l1 if isinstance(r.get("parsed"), dict)]
    n_parsed = len(parsed)

    print(f"=== L1 sanity ({n_total} episodes) ===")
    print(f"  status:           {dict(by_status)}")
    print(f"  parsed_ok:        {n_parsed}/{n_total} ({n_parsed/n_total*100:.1f}%)")

    cost_total = sum(r.get("cost_usd", 0.0) for r in l1)
    in_tok = sum(r.get("input_tokens", 0) for r in l1)
    out_tok = sum(r.get("output_tokens", 0) for r in l1)
    print(f"  tokens:           in={in_tok}  out={out_tok}")
    print(f"  cost:             ${cost_total:.4f} (avg ${cost_total/max(n_total,1):.4f}/ep)")

    with_flags = [r for r in parsed if (r["parsed"].get("risk_flags") or [])]
    print(f"\nWith risk_flags:    {len(with_flags)}/{n_parsed} ({len(with_flags)/max(n_parsed,1)*100:.1f}%)")

    types: Counter = Counter()
    confs: list[float] = []
    for r in parsed:
        for f in (r["parsed"].get("risk_flags") or []):
            t = f.get("flag_type", "MISSING")
            types[t] += 1
            if isinstance(f.get("confidence"), (int, float)):
                confs.append(float(f["confidence"]))
    print("\nFlag types (L1):")
    for t, c in types.most_common():
        print(f"  {t:<28} {c}")

    if confs:
        confs_sorted = sorted(confs)
        print(f"\nConfidence:")
        print(f"  mean={statistics.mean(confs):.3f}  median={statistics.median(confs):.3f}  "
              f"p10={confs_sorted[len(confs)//10]:.2f}  p90={confs_sorted[len(confs)*9//10]:.2f}")

    # L0 vs L1: brand detection
    l0_brand = sum(1 for r in l0 if (r.get("features") or {}).get("brand_count", 0) > 0)
    l0_unguarded = sum(1 for r in l0 if any(
        f.get("flag_type") == "brand_without_equivalent" for f in (r.get("risk_flags") or [])
    ))
    l1_brand_flag = sum(1 for r in parsed if any(
        f.get("flag_type") == "brand_without_equivalent" for f in (r["parsed"].get("risk_flags") or [])
    ))
    l1_brand_mentioned = sum(1 for r in parsed if (r["parsed"].get("features") or {})
                             .get("brand_mentions", []))
    print(f"\nBrand detection:")
    print(f"  L0 with brand_count>0:                   {l0_brand}")
    print(f"  L0 with brand_without_equivalent flag:   {l0_unguarded}")
    print(f"  L1 with brand_mentions list non-empty:   {l1_brand_mentioned}")
    print(f"  L1 with brand_without_equivalent flag:   {l1_brand_flag}")

    # Agreement L0 vs L1 on brand_without_equivalent
    l0_set = {r["episode_id"] for r in l0
              if any(f.get("flag_type") == "brand_without_equivalent" for f in (r.get("risk_flags") or []))}
    l1_set = {r["episode_id"] for r in parsed
              if any(f.get("flag_type") == "brand_without_equivalent" for f in (r["parsed"].get("risk_flags") or []))}
    both = l0_set & l1_set
    only_l0 = l0_set - l1_set
    only_l1 = l1_set - l0_set
    print(f"  Brand-flag agreement: both={len(both)}, only_L0={len(only_l0)}, only_L1={len(only_l1)}")

    # Stratum / cluster breakdown
    print("\nFlag-rate by stratum / cluster:")
    by_stratum: dict[str, list[int]] = {"benchmarkable": [], "conditional": [], "service_only": []}
    by_cluster: dict[int, list[int]] = {}
    by_tier: dict[str, list[int]] = {}
    for r in parsed:
        flags = r["parsed"].get("risk_flags") or []
        s = r.get("stratum", "?")
        c = r.get("family")
        t = r.get("quality_tier") or "?"
        by_stratum.setdefault(s, []).append(1 if flags else 0)
        by_cluster.setdefault(c, []).append(1 if flags else 0)
        by_tier.setdefault(t, []).append(1 if flags else 0)
    for s in ("benchmarkable", "conditional", "service_only"):
        v = by_stratum.get(s, [])
        if v:
            print(f"  {s:<14} {sum(v):>4}/{len(v):<4} ({sum(v)/len(v)*100:.0f}% with flags)")
    print("\nFlag-rate by cluster:")
    for c in sorted(by_cluster):
        v = by_cluster[c]
        print(f"  cluster_{c:02d}    {sum(v):>4}/{len(v):<4} ({sum(v)/len(v)*100:.0f}%)")

    # Average tokens by tier (useful for cost forecasting)
    tier_cost: dict[str, list[float]] = {}
    tier_in: dict[str, list[int]] = {}
    for r in l1:
        t = r.get("quality_tier") or "?"
        tier_cost.setdefault(t, []).append(r.get("cost_usd", 0.0))
        tier_in.setdefault(t, []).append(r.get("input_tokens", 0))
    print("\nMean cost / input-tokens by quality_tier:")
    for t in tier_cost:
        cs = tier_cost[t]
        ins = tier_in[t]
        print(f"  {t:<14} n={len(cs)}  mean_cost=${statistics.mean(cs):.4f}  "
              f"mean_in_tok={statistics.mean(ins):.0f}")


if __name__ == "__main__":
    main()
