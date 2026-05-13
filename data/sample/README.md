# `data/sample/` — демо-выборка решений ФАС

В этой папке лежат 10 решений ФАС (`.json`), отобранных как minimal reproducible example для службы и проверки research-пайплайна:

- **4 решения** с вердиктом `violation_established` — покрывают три основных
  типа нарушений из таксономии (`brand_without_equivalent`,
  `incomplete_description`, `restrictive_requirement`) + одно «other».
- **6 решений** с вердиктом `violation_not_established` — из разных кластеров
  таксономии (cluster_id 13, 28, 33, 46, 51).

`manifest.json` содержит метаданные по каждому файлу: `decision_id`, `verdict`,
`cluster_id`, типы найденных нарушений, размер в байтах.

## Формат файла

Каждое решение хранится как JSON со схемой (выдержки):

```json
{
  "decision_id": "202300100161028454",
  "document_id": "1931053",
  "doc_type": "CHECK_RESULT",
  "complaint_number": "...",
  "purchase_id": "0158100015723000033",
  "case_number": "28/06/105-3022/2023",
  "articles": [...],
  "text_length": 5903,
  "freeze_date": "2026-04-30",
  "full_text": "РЕШЕНИЕ по делу № …"
}
```

## Полный корпус

**Скачать:** [ZakupkiCheck_corpus.tar.gz](https://drive.google.com/file/d/1yvD1QVAdlgBXb0HVgHcpUw34bo6dOqEA/view?usp=share_link) (~120 МБ)

Архив содержит:
- `raw_fas/` — 2 012 решений ФАС в JSON (§2.2 ВКР)
- `parsed_clean/` — распарсенные тексты ТЗ для extraction pipeline (§3.2.2)

Для воспроизведения таблиц ВКР (Табл. 7–16) корпус не требуется — результаты
заморожены в `research/results/data/*.jsonl`. Корпус нужен для повторного
запуска extraction pipeline с нуля.
