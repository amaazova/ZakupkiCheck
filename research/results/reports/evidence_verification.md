# W2 Session A — Evidence verification

## 1. Citation containment (per finding/flag, evidence_quote vs source doc)

| metric | n_total | n_contained | rate | hallucination |
|---|---|---|---|---|
| FAS established findings → fas.json | 1051 | 477 | 45.4% | 54.6% |
| TZ flags (matched, all types) → tz.md | 364 | 170 | 46.7% | 53.3% |
| TZ flags (matched, specific) → tz.md | 328 | 160 | 48.8% | 51.2% |

_FAS established findings with empty evidence_quote: 366_


## 2. Entity-overlap Jaccard (named entities in evidence_quote/description)

| matched-pair set | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| all_types | 364 | 0.033 | 0.000 | 0.000 | 0.000 |
| specific_only | 328 | 0.037 | 0.000 | 0.000 | 0.000 |

## 3. Semantic similarity (multilingual MiniLM, cosine)

| matched-pair set | n | mean | median | p10 | p90 | ≥0.5 | ≥0.6 | ≥0.7 | ≥0.8 |
|---|---|---|---|---|---|---|---|---|---|
| all_types | 364 | 0.228 | 0.205 | 0.000 | 0.542 | 47 (13%) | 28 (8%) | 14 (4%) | 7 (2%) |
| specific_only | 328 | 0.228 | 0.200 | 0.000 | 0.549 | 42 (13%) | 25 (8%) | 13 (4%) | 7 (2%) |

## 4. FAS citation containment by finding_type

| finding_type | n_established | containment_rate |
|---|---|---|
| other | 324 | 41.4% |
| incomplete_description | 319 | 50.2% |
| restrictive_requirement | 215 | 48.4% |
| ktru_mismatch | 114 | 35.1% |
| inconsistent_dates | 39 | 53.8% |
| brand_without_equivalent | 32 | 53.1% |
| missing_acceptance_terms | 8 | 12.5% |
