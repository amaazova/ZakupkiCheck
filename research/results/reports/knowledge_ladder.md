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
