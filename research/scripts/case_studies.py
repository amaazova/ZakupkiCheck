"""W3 Session A — case studies (12–15 episodes)."""
from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .data_loaders import _episode_to_decision_id, load_fas_text, load_tz
from .extraction_runner import read_jsonl
from .matching_pipeline import contained, normalize

EVAL = PROJECT_ROOT / "workspace" / "eval"
OUT = EVAL / "case_studies.md"
SEED = 20260512
random.seed(SEED)

SPECIFIC_TYPES = frozenset({
    "brand_without_equivalent", "incomplete_description",
    "restrictive_requirement", "ktru_mismatch",
    "inconsistent_dates", "missing_acceptance_terms",
})

ERROR_CATEGORIES = {
    "context_gap", "extraction_miss", "granularity_mismatch",
    "false_alarm", "negation_miss", "hallucinated_entity",
}


# Selection

def select_cases(matching: list[dict], fas_by_ep: dict[str, dict],
                 l1_by_ep: dict[str, dict]) -> dict[str, list[dict]]:
    """Return cases keyed by category."""
    high, partial, fas_only, pipeline_only, parse_fail = [], [], [], [], []

    for r in matching:
        eid = r["episode_id"]
        n_est = r["n_fas_findings_established"]
        n_tz = r["n_tz_flags"]
        n_match = r["type_matches_specific"]
        fas_status = fas_by_ep.get(eid, {}).get("decision_status")
        l1_status = l1_by_ep.get(eid, {}).get("status")
        if fas_status == "parse_failed" or l1_status == "parse_failed":
            parse_fail.append(r)
            continue
        if n_est >= 2 and n_match >= 2 and n_match / max(n_est, 1) >= 0.6:
            high.append(r)
            continue
        if n_est >= 3 and 1 <= n_match <= 2:
            partial.append(r)
            continue
        if n_est >= 1 and n_match == 0 and r["fas_verdict"] == "violation_established":
            fas_only.append(r)
            continue
        if n_est == 0 and n_tz >= 1 and r["fas_verdict"] == "violation_not_established":
            pipeline_only.append(r)
            continue

    def sample(lst: list[dict], n: int) -> list[dict]:
        if len(lst) <= n:
            return list(lst)
        return random.sample(lst, n)

    return {
        "high_match": sample(high, 3),
        "partial_match": sample(partial, 3),
        "fas_only": sample(fas_only, 3),
        "pipeline_only": sample(pipeline_only, 3),
        "parse_failed": sample(parse_fail, 2),
    }


# Heuristic error categorisation (per case)

def classify_episode(r: dict, fas_rec: dict, l1_rec: dict, tz_norm: str) -> str:
    """Return an error category for the case (whole-episode level)."""
    fas_findings = [f for f in (fas_rec.get("findings") or []) if f.get("severity") == "established"]
    parsed = l1_rec.get("parsed") if isinstance(l1_rec.get("parsed"), dict) else {}
    tz_flags = (parsed or {}).get("risk_flags") or []
    n_match = r["type_matches_specific"]
    # parse failed
    if l1_rec.get("status") == "parse_failed" or fas_rec.get("decision_status") == "parse_failed":
        return "extraction_miss"  # mechanical failure; closest existing bucket
    # pipeline_only: no FAS findings but pipeline emitted flags
    if not fas_findings and tz_flags:
        return "false_alarm"
    # fas_only: FAS has findings, pipeline matched nothing
    if fas_findings and n_match == 0:
        # If FAS quotes are CONTAINED in TZ → extraction_miss; otherwise context_gap.
        cit = r.get("fas_citation_containment") or []
        # If any FAS quote is contained AND the TZ also contains it → extraction_miss
        # (pipeline saw the text but failed to flag it).
        for ff, hit_in_fas in zip(fas_findings, cit):
            q = (ff.get("evidence_quote") or "").strip()
            if hit_in_fas and contained(q, tz_norm):
                return "extraction_miss"
        # Otherwise FAS evidence not in TZ → information lives outside TZ.
        return "context_gap"
    # Partial match
    if fas_findings and 0 < n_match < len(fas_findings):
        # Check if there are "other"-type FAS findings while TZ flags are specific.
        types_fas = {f.get("finding_type") for f in fas_findings}
        types_tz  = {t.get("flag_type")     for t in tz_flags}
        if "other" in types_fas and types_tz & SPECIFIC_TYPES:
            return "granularity_mismatch"
        return "context_gap"
    # high_match
    return "extraction_miss" if False else "context_gap"  # fall-through unused


def episode_label(r: dict) -> str:
    """Friendly bin label."""
    n_est = r["n_fas_findings_established"]
    n_tz = r["n_tz_flags"]
    n_match = r["type_matches_specific"]
    if n_est >= 2 and n_match >= 2 and n_match / max(n_est, 1) >= 0.6:
        return "high_match"
    if n_est >= 3 and 1 <= n_match <= 2:
        return "partial_match"
    if n_est >= 1 and n_match == 0 and r["fas_verdict"] == "violation_established":
        return "fas_only"
    if n_est == 0 and n_tz >= 1:
        return "pipeline_only"
    return "other"


# Markdown rendering

def truncate(s: str, n: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + "…"


def render_case(idx: int, category: str, r: dict, fas_rec: dict, l1_rec: dict,
                eval_df: pd.DataFrame, tz_norm: str) -> str:
    eid = r["episode_id"]
    erow = eval_df[eval_df["episode_id"] == eid].iloc[0]
    cluster_id = int(r["cluster_id"])
    cluster_name = erow["cluster_name"]
    stratum = r["stratum"]
    verdict = r["fas_verdict"]
    quality_tier = erow["quality_tier"]
    tz_chars = int(erow["tz_char_count"])
    fas_findings_all = fas_rec.get("findings") or []
    fas_est = [f for f in fas_findings_all if f.get("severity") == "established"]
    parsed = l1_rec.get("parsed") if isinstance(l1_rec.get("parsed"), dict) else {}
    tz_flags = (parsed or {}).get("risk_flags") or []
    fas_types = [f.get("finding_type") for f in fas_est]
    tz_types  = [t.get("flag_type")     for t in tz_flags]

    category_label = category
    err = classify_episode(r, fas_rec, l1_rec, tz_norm)

    lines: list[str] = []
    lines.append(f"### Case {idx}: `{eid}` — {category_label}\n")
    lines.append(f"- **Stratum**: {stratum}, cluster c_{cluster_id} (_{cluster_name}_)")
    lines.append(f"- **Quality tier**: {quality_tier} (TZ {tz_chars:,} chars)")
    lines.append(f"- **FAS verdict**: {verdict}")
    lines.append(f"- **FAS established findings** ({len(fas_est)}/{len(fas_findings_all)}): "
                 f"{', '.join(fas_types) or '—'}")
    lines.append(f"- **TZ risk flags** ({len(tz_flags)}): {', '.join(tz_types) or '—'}")
    lines.append(f"- **Type matches**: all={r['type_matches_all']}, "
                 f"specific={r['type_matches_specific']}, "
                 f"l0_brand={r['type_matches_l0_brand']}")

    if fas_est:
        lines.append("\n**FAS findings (sample):**")
        for ff in fas_est[:3]:
            lines.append(f"- `{ff.get('finding_type')}` — {truncate(ff.get('description'), 220)}")
            q = (ff.get("evidence_quote") or "").strip()
            if q:
                lines.append(f"  > _quote_: {truncate(q, 220)}")
    if tz_flags:
        lines.append("\n**TZ flags (sample):**")
        for tf in tz_flags[:3]:
            q = truncate(tf.get("evidence_quote"), 220)
            conf = tf.get("confidence")
            lines.append(f"- `{tf.get('flag_type')}` (conf={conf}) — _{q or 'no quote'}_")

    # Heuristic narrative
    note = build_note(category_label, r, fas_est, tz_flags, err)
    lines.append("\n**Analysis.** " + note)
    lines.append(f"\n**Error category**: `{err}`\n")
    return "\n".join(lines)


def build_note(category: str, r: dict, fas_est, tz_flags, err: str) -> str:
    n_est = len(fas_est)
    n_tz = len(tz_flags)
    n_match = r["type_matches_specific"]
    if category == "high_match":
        return (f"Pipeline matched {n_match}/{n_est} established findings on a "
                "type-by-type basis — the dominant flag types overlap with what FAS "
                "ruled on. Remaining unmatched are typically `other`-typed by FAS "
                "(narrative findings) or duplicate-typed where the greedy 1-to-1 "
                "assignment leaves the extra unmatched.")
    if category == "partial_match":
        types_fas = [f.get("finding_type") for f in fas_est]
        types_tz  = [t.get("flag_type")     for t in tz_flags]
        return (f"FAS produced {n_est} established findings ({sorted(set(types_fas))}); "
                f"pipeline flagged {n_tz} ({sorted(set(types_tz))}); only {n_match} "
                "specific types align. The misses are usually `other`-typed FAS "
                "findings that the open-extraction prompt couldn't categorise, or "
                "specific types whose evidence sits outside the TZ proper.")
    if category == "fas_only":
        if err == "extraction_miss":
            return ("FAS established a violation whose evidence_quote is present "
                    "in the TZ text, but the L1 extractor did not produce a flag of "
                    "that type. This is a recall failure attributable to the prompt "
                    "or model — the evidence is present in-context.")
        return ("FAS established a violation whose evidence does NOT appear in the TZ "
                "document — typically because the basis lives in the procurement notice, "
                "supplier registry, market analysis, or correspondence the pipeline "
                "never sees. This is a fundamental context_gap, not an extraction failure.")
    if category == "pipeline_only":
        return ("Pipeline produced flags but FAS established no violation — either "
                "FAS reviewed the same fact and dismissed it (legitimate disagreement) "
                "or the pipeline raised a false alarm on benign language. Without "
                "rater-level data we cannot distinguish, but the precision tax is "
                "real: every such flag is a false positive at the episode level.")
    if category == "parse_failed":
        return ("Either FAS or L1 extraction returned non-JSON; the episode is "
                "effectively excluded from matching. Mechanical failure, not "
                "modelling failure.")
    return "—"


# Driver

def main() -> None:
    matching = read_jsonl(EVAL / "matching_results.jsonl")
    fas = read_jsonl(EVAL / "fas_findings.jsonl")
    l1 = read_jsonl(EVAL / "tz_features_L1.jsonl")
    eval_df = pd.read_csv(EVAL / "eval_dataset_v10.csv", dtype={"notice_id": str})

    fas_by_ep = {r["episode_id"]: r for r in fas}
    l1_by_ep = {r["episode_id"]: r for r in l1}

    cases = select_cases(matching, fas_by_ep, l1_by_ep)
    print(f"[cases] selected: {[(k, len(v)) for k, v in cases.items()]}")

    out_lines = ["# W3 Session A — Case studies\n"]
    out_lines.append(f"_random seed = {SEED}; selection rules: see `case_studies.py`._\n")
    idx = 1
    for category, recs in cases.items():
        if not recs:
            continue
        out_lines.append(f"\n## Category: `{category}` ({len(recs)} cases)\n")
        for r in recs:
            eid = r["episode_id"]
            tz_raw = load_tz(r["notice_id"]) or ""
            tz_norm = normalize(tz_raw)
            md = render_case(
                idx, category, r,
                fas_rec=fas_by_ep.get(eid, {}),
                l1_rec=l1_by_ep.get(eid, {}),
                eval_df=eval_df, tz_norm=tz_norm,
            )
            out_lines.append(md)
            idx += 1

    OUT.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[cases] → {OUT}")


if __name__ == "__main__":
    main()
