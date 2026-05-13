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
