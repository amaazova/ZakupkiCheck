"""Manual-review prep for W1 Session A: stratified sample of 30 episodes."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from workspace.scripts import config, extraction_runner, data_loaders  # type: ignore
else:
    from . import config, extraction_runner, data_loaders


EVAL_CSV = config.PROJECT_ROOT / "workspace" / "eval" / "eval_dataset_v10.csv"
PER_EP_OUT = config.PROJECT_ROOT / "workspace" / "eval" / "fas_findings.jsonl"
REVIEW_MD = config.PROJECT_ROOT / "workspace" / "eval" / "fas_manual_review.md"

SAMPLES_PER_STRATUM = 10
SEED = 42
FAS_EXCERPT_CHARS = 3000


def quote_in_text(quote: str, text: str) -> bool:
    if not quote.strip():
        return True  # empty quote = nothing to verify
    # tolerant: collapse whitespace
    q = " ".join(quote.split()).lower()
    t = " ".join(text.split()).lower()
    if len(q) < 12:  # too short to be a meaningful citation
        return True
    return q in t


def main() -> int:
    eval_df = pd.read_csv(EVAL_CSV, dtype={"notice_id": str})
    results = {r["episode_id"]: r for r in extraction_runner.read_jsonl(PER_EP_OUT)}
    if not results:
        print("[manual-review] fas_findings.jsonl is empty — run the full extraction first.",
              file=sys.stderr)
        return 1

    # Stratified sample: prefer episodes WITH findings so the review is informative.
    samples: list[dict] = []
    for stratum in ("benchmarkable", "conditional", "service_only"):
        sub = eval_df[(eval_df["stratum"] == stratum)
                      & (eval_df["episode_id"].isin(results.keys()))].copy()
        sub["_n_findings"] = sub["episode_id"].map(
            lambda eid: len((results[eid].get("findings") or []))
        )
        # Take a mix: prefer ≥1 finding, but include 1-2 with 0 to surface false negatives.
        with_findings = sub[sub["_n_findings"] > 0].sample(
            n=min(SAMPLES_PER_STRATUM - 2, len(sub[sub["_n_findings"] > 0])),
            random_state=SEED,
        ) if (sub["_n_findings"] > 0).sum() > 0 else sub.head(0)
        without = sub[sub["_n_findings"] == 0].sample(
            n=min(2, (sub["_n_findings"] == 0).sum()),
            random_state=SEED,
        ) if (sub["_n_findings"] == 0).sum() > 0 else sub.head(0)
        chosen = pd.concat([with_findings, without])
        if len(chosen) < SAMPLES_PER_STRATUM:
            need = SAMPLES_PER_STRATUM - len(chosen)
            extra = sub[~sub["episode_id"].isin(chosen["episode_id"])].sample(
                n=min(need, len(sub) - len(chosen)), random_state=SEED + 1)
            chosen = pd.concat([chosen, extra])
        chosen = chosen.head(SAMPLES_PER_STRATUM)
        samples.extend(chosen.to_dict(orient="records"))

    lines: list[str] = [
        "# W1 Session A — manual review (30 episodes)",
        "",
        "Stratified sample: 10 benchmarkable + 10 conditional + 10 service_only.",
        f"Random seed = {SEED}. FAS excerpts truncated to {FAS_EXCERPT_CHARS} chars.",
        "",
        "**Auto-flags pre-marked next to each issue.** Add ✓/✗/? after each entry as you review.",
        "",
        "Legend:",
        "- 🟢 verdict matches gold",
        "- 🔴 verdict mismatch",
        "- ⚠️ evidence_quote not found literally in FAS text (after whitespace normalization)",
        "- ∅ no findings extracted",
        "",
    ]

    for ix, row in enumerate(samples, 1):
        eid = row["episode_id"]
        rec = results[eid]
        decision_id = rec.get("decision_id", eid.rsplit("_", 1)[0])
        fas_text = data_loaders.load_fas_text(decision_id) or ""

        gold = row["fas_verdict"]
        pred = rec.get("overall_verdict", "")
        verdict_marker = "🟢" if pred == gold else "🔴"

        findings = rec.get("findings") or []
        empty_marker = " ∅" if not findings else ""

        lines += [
            "---",
            f"## {ix}. `{eid}` — stratum=`{row['stratum']}` cluster=c_{row['cluster_id']}{empty_marker}",
            "",
            f"- gold: **{gold}**",
            f"- pred: **{pred}** {verdict_marker}",
            f"- key_reasoning: _{rec.get('key_reasoning', '').strip()}_",
            f"- decision_id: `{decision_id}`  notice_id: `{row['notice_id']}`",
            f"- decision_status: `{rec.get('decision_status', '')}`",
            "",
            "### Extracted findings",
            "",
        ]
        if not findings:
            lines.append("_(none)_")
        for j, f in enumerate(findings, 1):
            quote = f.get("evidence_quote", "")
            quote_ok = quote_in_text(quote, fas_text)
            quote_marker = "" if quote_ok else "  ⚠️ quote not found"
            lines += [
                f"#### Finding {j}: `{f.get('finding_type')}` — severity=`{f.get('severity')}`{quote_marker}",
                "",
                f"- description: {f.get('description', '')}",
                f"- cited_articles: {f.get('cited_articles', [])}",
                f"- evidence_quote:",
                "",
                "  > " + (quote.replace("\n", " ").strip() or "_(empty)_"),
                "",
            ]
        lines += [
            "### FAS text (excerpt)",
            "",
            "```",
            fas_text[:FAS_EXCERPT_CHARS] + ("\n…(truncated)" if len(fas_text) > FAS_EXCERPT_CHARS else ""),
            "```",
            "",
            "### Reviewer note",
            "",
            "_(write findings_correct=✓/✗  finding_types_correct=✓/✗  evidence_quote_real=✓/✗  notes=…)_",
            "",
        ]

    REVIEW_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[manual-review] wrote {REVIEW_MD} ({len(samples)} episodes)")

    # Quick auto-stats
    matches = sum(1 for s in samples if results[s["episode_id"]].get("overall_verdict") == s["fas_verdict"])
    empties = sum(1 for s in samples if not (results[s["episode_id"]].get("findings") or []))
    n_quotes = 0
    n_quotes_ok = 0
    for s in samples:
        rec = results[s["episode_id"]]
        fas_text = data_loaders.load_fas_text(rec.get("decision_id", s["episode_id"].rsplit("_", 1)[0])) or ""
        for f in rec.get("findings", []):
            if (f.get("evidence_quote") or "").strip():
                n_quotes += 1
                if quote_in_text(f.get("evidence_quote", ""), fas_text):
                    n_quotes_ok += 1
    print(f"[manual-review] auto-stats: verdict_match={matches}/{len(samples)} "
          f"empty_findings={empties}/{len(samples)} "
          f"evidence_quotes_literal={n_quotes_ok}/{n_quotes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
