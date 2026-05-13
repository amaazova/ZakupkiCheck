# ZakupkiCheck — Streamlit app

Демонстрационное приложение для проверки технического задания (ТЗ) госзакупки
по 44-ФЗ. Принимает на вход документ (docx/pdf/txt/md), запускает LLM-извлечение
compliance-признаков и risk_flags, ищет похожие решения ФАС из локального
корпуса (ChromaDB или TF-IDF fallback) и формирует PDF-отчёт.

## Архитектура

```
workspace/streamlit/
├── app.py                       # Streamlit UI
├── components/
│   ├── parser.py                # .docx/.pdf/.txt → plain text
│   ├── extractor.py             # LLM extraction (V4 Flash, OpenRouter)
│   ├── retrieval.py             # ChromaDB / TF-IDF similar-case search
│   ├── cache.py                 # SQLite cache by sha256(text)
│   └── report.py                # PDF-отчёт (fpdf2)
├── Dockerfile
├── requirements.txt
└── .streamlit/config.toml
```

Переиспользуемые компоненты проекта (импортируются как есть, не модифицируются):

- `workspace/scripts/config.py` — API ключи, model IDs, pricing.
- `workspace/scripts/extraction_runner.py` — OpenRouter client, JSON parsing, retry.
- `workspace/eval/prompts/tz_l1_*.md` — system + user prompts L1.
- `workspace/eval/fas_findings.jsonl` — корпус решений ФАС для retrieval.

## Запуск локально

```bash
cd /Users/aza/Downloads/zakupki/last/workspace/streamlit
pip install -r requirements.txt

# В корне проекта (/Users/aza/Downloads/zakupki/last/.env) должна быть строка:
# OPENROUTER_API_KEY=sk-or-...

streamlit run app.py
```

UI откроется на http://localhost:8501.

## Запуск в Docker

```bash
cd /Users/aza/Downloads/zakupki/last/workspace/streamlit
docker build -t zakupkicheck .
docker run --rm -p 8501:8501 \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$(pwd)/../..":/data \
  zakupkicheck
```

При первом запуске retrieval-компонент построит ChromaDB индекс из
`workspace/eval/fas_findings.jsonl` (~781 решений ФАС). Это занимает 30–60 сек
на CPU. Индекс кешируется в `workspace/streamlit/data/chroma/`.

## Что приложение делает

1. Принимает файл .docx / .pdf / .txt / .md размером до 50 МБ.
2. Парсит документ в plain text (docx — через python-docx, pdf — через PyMuPDF).
3. Считает sha256 от текста. Если такой документ уже был проверен — берёт
   результат из SQLite-кеша. Иначе — вызывает LLM.
4. LLM: `deepseek/deepseek-v4-flash` через OpenRouter, system+user prompts L1
   из `workspace/eval/prompts/`. Ответ — JSON по схеме `{features, risk_flags}`.
5. Параллельно по тексту ТЗ ищется top-3 похожих решения ФАС из локального корпуса.
6. UI показывает: метрики (число рисков, уровень, стоимость), чек-лист рисков
   с цитатами и нормами, прецеденты ФАС.
7. По кнопке формируется PDF-отчёт (fpdf2 + DejaVu/Arial Unicode для Cyrillic).
8. Сайдбар показывает историю последних проверок.

## Ограничения

- Только ТЗ (без контрактов). Doc router отсутствует.
- Один документ за раз (batch/comparison mode не реализованы).
- Корпус retrieval — 781 эпизод из `fas_findings.jsonl`. Если файла нет
  или embeddings не загружаются, retrieval молча возвращает пустой список.

## Что НЕ модифицировано

- `workspace/scripts/extraction_runner.py`, `data_loaders.py`, `config.py`
- Любые eval-файлы (JSONL, CSV)
- ChromaDB по пути `zakupki_final/rag/chroma_db` — у этого приложения свой,
  отдельный, в `workspace/streamlit/data/chroma/`.
