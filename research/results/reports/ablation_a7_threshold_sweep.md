# A7 — Semantic-similarity threshold sweep

Starting from exact-type matches (specific_only), filter pairs where `semantic_sim ≥ threshold`. As threshold rises, fewer pairs survive — recall drops, precision rises (with respect to the original-mode denominators).

| threshold | precision | recall | F1 | matched / gold | pred flags |
|---|---|---|---|---|---|
| 0.3 | 7.7% [6.2, 9.6] | 17.9% [14.4, 21.9] | 10.8% [8.7, 13.2] | 108/603 | 1399 |
| 0.4 | 4.8% [3.7, 6.2] | 11.1% [8.5, 14.5] | 6.7% [5.2, 8.6] | 67/603 | 1399 |
| 0.5 | 2.9% [2.0, 4.0] | 6.6% [4.6, 9.4] | 4.0% [2.8, 5.5] | 40/603 | 1399 |
| 0.6 | 1.7% [1.0, 2.6] | 4.0% [2.3, 6.2] | 2.4% [1.4, 3.7] | 24/603 | 1399 |
| 0.7 | 0.9% [0.4, 1.6] | 2.2% [1.0, 3.6] | 1.3% [0.6, 2.1] | 13/603 | 1399 |
| 0.8 | 0.5% [0.1, 1.0] | 1.2% [0.3, 2.3] | 0.7% [0.2, 1.4] | 7/603 | 1399 |

Denominators (gold est., pred flags) are constant across thresholds — only the matched count changes.

![A7 plot](ablation_a7_threshold_plot.png)

