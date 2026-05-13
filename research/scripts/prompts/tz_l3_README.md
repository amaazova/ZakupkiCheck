# L3 (B_fewshot) — статус и зависимости

`tz_l3_user_prompt_template.md` — структура промпта готова. Заполнители:

| placeholder | что вставить |
|---|---|
| `{example_1_tz_excerpt}` | фрагмент ТЗ из эпизода 1 (cluster_15) — 200-400 символов, с конкретным брендом |
| `{example_1_extraction}` | ожидаемый JSON-extraction для этого фрагмента (только `risk_flags[0]` + minimal `features`) |
| `{example_2_tz_excerpt}` | фрагмент ТЗ из эпизода 2 (cluster_52 / cluster_53) — пример incomplete description |
| `{example_2_extraction}` | ожидаемый JSON |
| `{example_3_tz_excerpt}` | фрагмент ТЗ из эпизода 3 (cluster_22, cluster_29, cluster_28 — restrictive requirement) |
| `{example_3_extraction}` | ожидаемый JSON |
| `{extra_instruction}` | стандартное (chunking / thin) |
| `{doc_text}` | сам ТЗ |

## Что нужно от Session A

Чтобы заполнить примеры:

1. **Session A (FAS extraction)** должна закончить L1 на корпусе ФАС-решений.
2. Из её `fas_features_L1.jsonl` отобрать **3 эпизода**, где:
   - flag_type из FAS findings совпадает с одним из {brand_without_equivalent, incomplete_description, restrictive_requirement/ktru_mismatch};
   - confidence высокая (>=0.8);
   - `evidence_quote` из FAS можно сопоставить с фрагментом в `data/parsed_clean/<notice_id>/tz.md`.
3. Для каждого: вырезать **200-400-символьный фрагмент ТЗ**, содержащий findings, и записать **золотой** extraction вручную (не полагаться на L1 output, потому что L1 — это baseline без taxonomy hint).

## Решение о placement

L3 будет прогоняться в W2, после интеграции с Session A. Промпт-скелет лежит готовый и не зависит от выбора примеров — заполнение `.format(...)` через тот же `extraction_runner`. См. `run_l3_tz.py` (создать в W2).
