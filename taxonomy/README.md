# `taxonomy/` — таксономия 56 семейств нарушений

Таксономия из ВКР: 56 «семейств» нарушений 44-ФЗ, индуцированных
HDBSCAN-кластеризацией по эмбеддингам решений ФАС.

## Файлы

| Файл | Что |
|---|---|
| `clustering.py` | UMAP-понижение (10 компонент) + HDBSCAN (eom, `min_samples=5`). Производит 56 кластеров из эмбеддингов `intfloat/multilingual-e5-large`. Перенесено из `runs/20260510T180119/pipeline/stage21_v2_stop1.py`. |
| `clustering_finalize.py` | Постобработка кластеров и формирование `family_status_map.csv`. Перенесено из `runs/20260510T180119/pipeline/stage21_v2_finalize.py`. |
| `family_document_requirements.csv` | 56 семейств × требуемые поля документации (нужны/опциональны/не применимо). Используется как карта benchmarkability. |
| `family_to_group_map.csv` | Маппинг семейство → группа (укрупнение для отчётов). |
| `taxonomy_56_families.csv` | Тот же список 56 семейств — удобная сводка для табличных приложений. |
| `taxonomy_hints.json` | Подсказки для L2 prompt (`research/scripts/prompts/tz_l2_user_prompt_template.md`). 15 ключевых семейств — те, что чаще всего встречаются в eval-выборке; передаются как `taxonomy_hint` в промпт извлечения L2. |

## Параметры алгоритма

```python
UMAP(n_components=10, n_neighbors=15, min_dist=0.0,
     metric="cosine", random_state=20260510)
HDBSCAN(min_samples=5, cluster_selection_method="eom")
```

## Запуск

`clustering.py` ожидает заранее посчитанные эмбеддинги — пайплайн не размещён
в репозитории целиком (он опирается на полный корпус решений ФАС, см.
`data/sample/README.md`). Для воспроизведения шага кластеризации потребуются:

1. Сырые решения ФАС → `research/scripts/run_w1_fas_extraction.py` для
   извлечения структурированных эпизодов.
2. Эмбеддинги решений (`intfloat/multilingual-e5-large`).
3. Запуск `clustering.py` на эмбеддингах.

Готовая таксономия (56 семейств с именами и описаниями) — в
`family_document_requirements.csv`.
