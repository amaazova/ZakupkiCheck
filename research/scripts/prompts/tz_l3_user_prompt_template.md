Проанализируй следующее техническое задание (ТЗ) госзакупки по 44-ФЗ и извлеки все потенциальные проблемы соответствия согласно описанной в system prompt схеме.

Ниже приведены 3 примера: фрагмент ТЗ → ожидаемый extraction result. Эти примеры показывают, какие проблемы и как нужно извлекать. Каждый пример соответствует подтверждённому нарушению, установленному решением ФАС по аналогичному кейсу.

---

**Пример 1.** flag_type = `brand_without_equivalent`.

ТЗ-фрагмент:
> {example_1_tz_excerpt}

Ожидаемый extraction (фрагмент):
```json
{example_1_extraction}
```

---

**Пример 2.** flag_type = `incomplete_description`.

ТЗ-фрагмент:
> {example_2_tz_excerpt}

Ожидаемый extraction (фрагмент):
```json
{example_2_extraction}
```

---

**Пример 3.** flag_type = `restrictive_requirement` (либо `ktru_mismatch`, в зависимости от итогового выбора в Session A).

ТЗ-фрагмент:
> {example_3_tz_excerpt}

Ожидаемый extraction (фрагмент):
```json
{example_3_extraction}
```

---

Теперь проанализируй следующий ТЗ. Не копируй цитаты из примеров — найди реальные цитаты в этом документе.

{extra_instruction}

Верни ТОЛЬКО валидный JSON.

--- TZ START ---
{doc_text}
--- TZ END ---
