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

