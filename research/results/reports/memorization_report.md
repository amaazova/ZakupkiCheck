# Memorization Probe Report

- pairs evaluated: **50**
- prompt mode: B3 (taxonomy-aware), DeepSeek V3 via OpenRouter
- elapsed: 358.9 sec (6.0 min)

## Accuracy

- accuracy on **original** tz: **0.5200** (26/50)
- accuracy on **shifted** tz: **0.4400** (22/50)
- delta (orig − shifted): **+8.00 п.п.**
- verdict flips between orig and shifted: **4** (8.00%)

## Memorization Signal

- threshold: > 10 п.п. delta → signal
- result: **no signal**

## Cost & Failures

- API calls: 100
- input tokens: 562192
- output tokens: 20131
- total cost: $0.1978
- failed calls: orig=0, shifted=0

## Notes

- 'Memorization signal' here is a coarse indicator: a large drop in
  accuracy when entity surface forms are perturbed suggests the model
  was relying on memorized notice/contract IDs rather than reasoning
  about the document's content.
- Stratified design (25 violation_established / 25 violation_not_established)
  controls for label imbalance.
