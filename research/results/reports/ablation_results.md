# W3 Session A — Ablations A6/A7/A8 summary

Pooled fact-level metrics, specific_only, gold violation episodes only.
Bootstrap CIs are cluster-bootstrapped over `notice_id` (1000 iters).

## Headline

| strategy / threshold | precision | recall | F1 | matched / gold |
|---|---|---|---|---|
| **A6 exact_type** (baseline, W2) | 20.4% [17.6, 23.2] | 47.4% [42.7, 52.6] | 28.6% [25.3, 31.7] | 286 / 603 |
| **A6 fuzzy_type** (synonym map) | 24.7% [21.6, 28.0] | 57.4% [52.3, 62.5] | 34.6% [31.0, 38.1] | 346 / 603 |
| A6 embedding_match (sim ≥ 0.6) | 4.9% [3.6, 6.4] | 11.3% [8.6, 14.5] | 6.8% [5.1, 8.7] | 68 / 603 |
| A7 sim ≥ 0.3 (post-filter) | 7.7% | 17.9% | 10.8% | 108 / 603 |
| A7 sim ≥ 0.5 (post-filter) | 2.9% | 6.6% | 4.0% | 40 / 603 |
| A7 sim ≥ 0.8 (post-filter) | 0.5% | 1.2% | 0.7% | 7 / 603 |

## A6 — matching strategy comparison

- **`fuzzy_type` is the best operating point we measured**: +6.0 pp F1 over the
  exact-type baseline (+10.0 pp recall at +4.3 pp precision). The synonym map
  is small and hand-curated — adding cross-mappings between
  `brand_without_equivalent ↔ restrictive_requirement`,
  `incomplete_description ↔ missing_acceptance_terms`,
  `ktru_mismatch ↔ incomplete_description` recovers ~60 additional matches
  without admitting noise.
- **`embedding_match` (ignore type, sim ≥ 0.6) is structurally bad**: F1 drops to
  6.8%. FAS evidence quotes paraphrase the legal violation; TZ evidence quotes
  are the literal procurement text. Their lexical surfaces differ enough that
  multilingual MiniLM cosine sits near 0.2 for most matched pairs — too low to
  act as a primary anchor.

## A7 — semantic similarity is not a useful gating threshold

Post-filtering exact-type matches by `semantic_sim ≥ t` monotonically destroys
recall without redeeming precision. At t=0.5 only 40 of 286 type-matched pairs
survive; at t=0.8 just 7. The semantic-similarity score is a confidence proxy
for individual pairs but not a re-ranker. **Recommendation**: keep semantic_sim
as a side channel (`confidence` boost when high), not a hard threshold.

See `ablation_a7_threshold_plot.png`.

## A8 — TZ-length sensitivity

Mean fact_recall by quality_tier:

| tier | n | mean recall | mean TZ chars |
|---|---|---|---|
| thin | 24 | 0.243 | 1,581 |
| normal | 308 | 0.545 | 31,725 |
| chunking_risk | 11 | 0.530 | 577,047 |

- **Thin TZs perform ~half as well** as normal/long. Short TZ documents either
  truly contain less to extract (FAS findings come from the notice or supplier
  registry) or the LLM produces over-cautious empty extractions when the input
  is sparse — both consistent with the W1 thin-tier instruction we added.
- **Chunking-risk TZs perform on par with normal**. The 150K-char truncation
  + chunking instruction added in W1 did not degrade recall on the
  ≥200K-character documents. No long-context penalty is visible.
- Scatter (`ablation_a8_scatter.png`) shows recall floor at very short TZs and
  a flat plateau from ~10K chars upward; LOESS curve confirms.

## Read across A6/A7/A8

The pipeline's primary lever is **type matching with a curated synonym map**.
Semantic similarity and length adjustments give small or negative returns from
this baseline. The next pp of F1 will come from prompt-level recall gains
(see error typology — `extraction_miss` is only 4% of unmatched, so the
ceiling for prompt-only improvement is limited) or from changing the unit of
analysis (cross-document evidence — but that requires a notice/registry
ingestion pipeline outside the scope of v10).

- [A6 details](ablation_a6_matching_strategies.md)
- [A7 details](ablation_a7_threshold_sweep.md) — [plot](ablation_a7_threshold_plot.png)
- [A8 details](ablation_a8_length_sensitivity.md) — [scatter](ablation_a8_scatter.png)
