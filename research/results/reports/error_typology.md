# W3 Session A — Error typology

Universe: every FAS established finding that did NOT get a same-type TZ L1 flag under specific_only matching.

- `n_total_established` (all types): **1051**
- `n_total_established_specific` (6 of 7 types): **727**
- `n_unmatched` (in this typology): **723**


## Distribution

| error_type | count | % of unmatched |
|---|---|---|
| `context_gap` | 368 | 50.9% |
| `granularity_mismatch` | 324 | 44.8% |
| `extraction_miss` | 31 | 4.3% |

## Per finding_type

| finding_type | n_unmatched | top error categories |
|---|---|---|
| `other` | 324 | granularity_mismatch=324 |
| `incomplete_description` | 196 | context_gap=179, extraction_miss=17 |
| `restrictive_requirement` | 75 | context_gap=73, extraction_miss=2 |
| `ktru_mismatch` | 67 | context_gap=63, extraction_miss=4 |
| `inconsistent_dates` | 36 | context_gap=33, extraction_miss=3 |
| `brand_without_equivalent` | 17 | context_gap=12, extraction_miss=5 |
| `missing_acceptance_terms` | 8 | context_gap=8 |

## Flag-side error classes (not in the count above)

These classes are not unmatched-gold buckets; they describe pipeline-side errors and are reported separately:

- `false_alarm` — pipeline emits flags on an episode where FAS established no violation. **n ≈ 258** episodes (`fas_verdict == violation_not_established && n_tz_flags ≥ 1`).
- `negation_miss` — pipeline misses an explicit «или эквивалент» / negation clause; requires lexical analysis of TZ flag context, **not auto-classified** — see case studies for examples.
- `hallucinated_entity` — TZ flag quotes a brand/ГОСТ/КТРУ not present in tz.md. Estimated upper bound: TZ citation-containment miss rate × n_flags ≈ 53% of matched-pair TZ quotes are not literal substrings, but most are paraphrases rather than hallucinations.

