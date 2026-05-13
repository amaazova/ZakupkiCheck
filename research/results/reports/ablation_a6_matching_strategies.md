# A6 — Matching strategy comparison

_specific_only mode; bootstrap = 1000 iters, cluster by notice_id, α = 0.05_


## exact_type

| stratum | precision | recall | F1 | matched | gold est. | pred flags |
|---|---|---|---|---|---|---|
| benchmarkable | 22.1% [16.7, 28.4] | 41.1% [33.3, 50.0] | 28.7% [22.9, 35.1] | 58 | 141 | 263 |
| conditional | 16.8% [13.4, 21.0] | 48.2% [40.6, 56.0] | 24.9% [20.5, 30.1] | 105 | 218 | 625 |
| service_only | 24.1% [20.0, 28.3] | 50.4% [43.0, 58.3] | 32.6% [28.2, 37.4] | 123 | 244 | 511 |
| pooled | 20.4% [17.6, 23.2] | 47.4% [42.7, 52.6] | 28.6% [25.3, 31.7] | 286 | 603 | 1399 |

## fuzzy_type

| stratum | precision | recall | F1 | matched | gold est. | pred flags |
|---|---|---|---|---|---|---|
| benchmarkable | 26.2% [20.1, 33.3] | 48.9% [39.9, 59.0] | 34.2% [27.6, 41.4] | 69 | 141 | 263 |
| conditional | 21.6% [17.4, 26.3] | 61.9% [54.5, 69.6] | 32.0% [26.9, 37.5] | 135 | 218 | 625 |
| service_only | 27.8% [23.6, 32.2] | 58.2% [50.4, 67.2] | 37.6% [33.3, 42.4] | 142 | 244 | 511 |
| pooled | 24.7% [21.6, 28.0] | 57.4% [52.3, 62.5] | 34.6% [31.0, 38.1] | 346 | 603 | 1399 |

## embedding_match (threshold=0.6)

| stratum | precision | recall | F1 | matched | gold est. | pred flags |
|---|---|---|---|---|---|---|
| benchmarkable | 2.7% [1.0, 5.0] | 5.0% [1.9, 8.9] | 3.5% [1.4, 6.3] | 7 | 141 | 263 |
| conditional | 4.3% [2.9, 6.1] | 12.4% [8.3, 17.2] | 6.4% [4.3, 9.0] | 27 | 218 | 625 |
| service_only | 6.7% [4.0, 9.5] | 13.9% [8.7, 19.5] | 9.0% [5.6, 12.6] | 34 | 244 | 511 |
| pooled | 4.9% [3.6, 6.4] | 11.3% [8.6, 14.5] | 6.8% [5.1, 8.7] | 68 | 603 | 1399 |

## Δ vs exact_type (pooled)

| strategy | Δ precision | Δ recall | Δ F1 |
|---|---|---|---|
| fuzzy_type | +4.3 pp | +10.0 pp | +6.0 pp |
| embedding_match | -15.6 pp | -36.2 pp | -21.8 pp |
