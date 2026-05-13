# A8 — TZ length sensitivity

_n_violation_episodes_with_gold_findings = 343_  
Recall computed per episode = matched_specific / gold_specific.

## Per quality_tier

| tier | n | mean fact_recall | median | p25 | p75 | mean TZ chars |
|---|---|---|---|---|---|---|
| thin | 24 | 0.243 | 0.000 | 0.000 | 0.500 | 1,581 |
| normal | 308 | 0.545 | 0.500 | 0.000 | 1.000 | 31,725 |
| chunking_risk | 11 | 0.530 | 0.500 | 0.000 | 1.000 | 577,047 |

## Scatter

![A8 scatter](ablation_a8_scatter.png)


## Observation

Mean fact_recall by tier: thin=0.243, normal=0.545, chunking_risk=0.530.
