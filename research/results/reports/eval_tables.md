# v10 — final evaluation tables

Consolidated report: knowledge ladder, multi-model, v9 vs v10, budget.

---

# Knowledge ladder — L0 → L1 → L2 → L3 → A3

_specific_only fact metrics on gold violation episodes; episode-level metrics on all 781. Bootstrap (1000 iters, cluster by notice_id)._


## Pooled

| Level | Description | Fact P | Fact R | Fact F1 | Episode Acc | Episode κ |
|---|---|---|---|---|---|---|
| L0 | Regex | 0.2% [0.1, 0.8] | 3.3% [1.4, 5.6] | 0.4% [0.1, 1.3] | 53.0% [48.6, 57.0] | 0.052 [0.022, 0.087] |
| L1 | LLM open | 20.4% [17.6, 23.2] | 47.4% [42.7, 52.6] | 28.6% [25.3, 31.7] | 74.4% [70.8, 78.0] | 0.486 [0.423, 0.549] |
| L2 | LLM + taxonomy hint | 22.8% [19.7, 26.3] | 59.2% [54.3, 64.3] | 32.9% [29.2, 36.7] | 78.6% [75.3, 82.1] | 0.572 [0.506, 0.640] |
| L3 | LLM + few-shot | 21.8% [18.0, 26.0] | 40.1% [34.6, 46.0] | 28.2% [24.1, 32.5] | 72.3% [68.4, 76.3] | 0.445 [0.374, 0.518] |
| A3 | LLM + CoT | 22.5% [19.6, 25.4] | 46.6% [41.7, 51.6] | 30.3% [27.0, 33.4] | 74.3% [70.9, 77.9] | 0.484 [0.419, 0.552] |

## Stratified (Fact F1, specific_only)

| Level | benchmarkable | conditional | service_only |
|---|---|---|---|
| L0 | 0.6% (8/141) | 0.2% (5/218) | 1.4% (7/244) |
| L1 | 28.7% (58/141) | 24.9% (105/218) | 32.6% (123/244) |
| L2 | 37.4% (85/141) | 28.3% (123/218) | 35.3% (149/244) |
| L3 | 27.5% (46/141) | 23.9% (79/218) | 32.5% (117/244) |
| A3 | 32.2% (64/141) | 27.1% (100/218) | 32.5% (117/244) |

## Δ vs L1 (pooled)

| Level | Δ Fact F1 | Δ Episode Acc | Δ Episode κ |
|---|---|---|---|
| L0 | -28.2 pp | -21.4 pp | -0.434 |
| L2 | +4.4 pp | +4.2 pp | +0.085 |
| L3 | -0.3 pp | -2.0 pp | -0.041 |
| A3 | +1.7 pp | -0.1 pp | -0.002 |

## Central hypothesis — benchmarkable recall by level

| Level | Fact Recall (benchmarkable) | Fact Recall (service_only) |
|---|---|---|
| L0 | 5.7% | 2.9% |
| L1 | 41.1% | 50.4% |
| L2 | 60.3% | 61.1% |
| L3 | 32.6% | 48.0% |
| A3 | 45.4% | 48.0% |

---

# Multi-model comparison — V4 Flash / Sonnet 4.6 / Qwen 3.6+

_End-to-end: each model used for both FAS and TZ extraction. Matching uses same pipeline; specific_only mode for fact metrics, strict mode for episode metrics._


## Headline table

| Model | FAS verdict agr | FAS findings/ep | TZ flags/ep | Fact F1 | Episode Acc | Episode κ | Total cost |
|---|---|---|---|---|---|---|---|
| V4 Flash | 84.0% | 1.76 | 3.42 | 28.6% [25.3, 31.7] | 74.4% [70.8, 78.0] | 0.486 [0.423, 0.549] | $2.18 |
| Sonnet 4.6 | 84.8% | 3.64 | 6.59 | 30.3% [27.9, 32.7] | 82.5% [79.3, 85.3] | 0.649 [0.586, 0.705] | $85.71 |
| Qwen 3.6+ | 84.9% | 3.91 | 3.12 | 39.2% [35.7, 42.6] | 77.7% [74.1, 80.9] | 0.554 [0.484, 0.616] | $20.33 |

## Fleiss κ (3-model strict-mode agreement on episode prediction)

**Fleiss κ = 0.609** over 781 episodes × 3 models. Interpretation: κ < 0.20 poor; 0.21–0.40 fair; 0.41–0.60 moderate; 0.61–0.80 substantial.


## Pairwise McNemar tests (episode-level correctness)

Cell counts: B = model A correct & model B wrong; C = model A wrong & model B correct. Two-sided p; α = 0.05. Significant ⇒ models disagree on errors.

| Pair (A vs B) | A correct/B wrong | A wrong/B correct | McNemar χ² | p-value |
|---|---|---|---|---|
| v4flash vs sonnet | 45 | 108 | 25.12 | 5.375e-07 **\*** |
| v4flash vs qwen | 71 | 97 | 3.72 | 0.05376 |
| sonnet vs qwen | 80 | 43 | 10.54 | 0.00117 **\*** |

## Heatmap — model × finding_type fact recall

![heatmap](multi_model_heatmap.png)


---

# v9 vs v10 — final comparison

| Approach | Year | Accuracy | κ | Paradigm |
|---|---|---|---|---|
| B0 majority | v9 | 62.1% | 0.00 | naive (predict majority) |
| B1 regex | v9 | 53.4% | — | pattern matching |
| B2 zero-shot LLM | v9 | 50.7% | ~0.01 | direct classification |
| B4 pipeline | v9 | 54.7% | ~0.09 | 4 detectors + OR |
| ML on L1 features | v10 | 60.8% | 0.22 | extraction → sklearn |
| **IE + strict matching (V4 Flash)** | v10 | **74.4%** | **0.49** | IE + type match |
| Best knowledge level (L2) | v10 | 78.6% | 0.57 | IE (best prompt) + type match |
| Best model (Sonnet 4.6) | v10 | 82.5% | 0.65 | IE (best model) + type match |

**Notes**: v9 numbers from `PROJECT_DESIGN_v9.md`; v10 numbers measured on 781-episode eval_dataset_v10 with strict episode prediction (≥1 type-matched specific finding). Bootstrap CIs in `knowledge_ladder.md` and `multi_model_comparison.md`.

---

# Budget summary

_Costs read directly from each run's JSONL `cost_usd` / `extraction_cost_usd` fields._

| Run | Model | Cost |
|---|---|---|
| W1 FAS extraction (500 dec) | V4 Flash | $0.52 |
| W1 TZ L1 (open) | V4 Flash | $1.32 |
| W2 TZ L2 (guided) | V4 Flash | $1.40 |
| W2 TZ L3 (few-shot) | V4 Flash | $1.32 |
| A3 TZ CoT | V4 Flash | $1.35 |
| A5 Sonnet FAS | Sonnet 4.6 | $20.64 |
| A5 Sonnet TZ | Sonnet 4.6 | $49.83 |
| A5 Qwen FAS | Qwen 3.6+ | $5.50 |
| A5 Qwen TZ | Qwen 3.6+ | $11.18 |
| Smoke tests + misc | mixed | ~$1.50 |
| **TOTAL** | | **$94.56** |
| Budget | | $355.00 |
| Remaining | | $260.44 |
