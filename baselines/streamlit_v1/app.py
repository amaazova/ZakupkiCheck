"""ZakupkiCheck — Streamlit app for single-document compliance checking.

Run: `streamlit run app.py` (from this directory) or via Docker.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

# Make the project importable when launched from any cwd
PROJECT_ROOT = Path("/Users/aza/Downloads/zakupki/last")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.cache import CacheLayer
from components.extractor import DEFAULT_MODEL, extract_compliance, risk_level
from components.parser import parse_document
from components.report import _FLAG_ARTICLES, _FLAG_LABELS_RU, generate_pdf
from components.retrieval import find_similar_cases, retriever_backend

APP_DIR = Path(__file__).parent
CACHE_DB = APP_DIR / "data" / "cache.db"

st.set_page_config(
    page_title="ZakupkiCheck",
    page_icon="📋",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_cache() -> CacheLayer:
    return CacheLayer(CACHE_DB)


def _flag_label(flag_type: str) -> str:
    return _FLAG_LABELS_RU.get(flag_type, flag_type)


def _flag_article(flag_type: str) -> str:
    return _FLAG_ARTICLES.get(flag_type, "—")


def _level_color(level: str) -> str:
    return {"Высокий": "#d63031", "Средний": "#e17055", "Низкий": "#00b894"}.get(level, "#636e72")


def render_results(result: dict, cases: list[dict], filename: str) -> None:
    parsed = result.get("parsed") or {}
    risk_flags = list(parsed.get("risk_flags") or [])
    level = risk_level(risk_flags)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Найдено рисков", len(risk_flags))
    col2.metric("Уровень", level)
    col3.metric("Тип документа", "ТЗ")
    cost = result.get("cost_usd") or 0.0
    if result.get("cached"):
        col4.metric("Источник", "из кеша")
    else:
        col4.metric("Стоимость, USD", f"{cost:.4f}")

    st.markdown(
        f"<div style='padding:8px 12px;border-radius:6px;"
        f"background-color:{_level_color(level)};color:white;"
        f"display:inline-block;font-weight:600;'>"
        f"Итоговый уровень риска: {level}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Выявленные риски")
    if not risk_flags:
        st.success("Рисков не обнаружено — ТЗ выглядит корректным.")
    else:
        for i, flag in enumerate(risk_flags, 1):
            ftype = flag.get("flag_type") or "other"
            label = _flag_label(ftype)
            conf = float(flag.get("confidence") or 0.0)
            quote = (flag.get("evidence_quote") or "").strip()
            title = f"{i}. {label}  —  уверенность {conf:.0%}"
            with st.expander(title, expanded=(i == 1)):
                if quote:
                    st.markdown(f"**Цитата:** «{quote}»")
                else:
                    st.caption("Цитата не приведена моделью.")
                st.markdown(f"**Норма:** {_flag_article(ftype)}")
                st.markdown(f"**Тип:** `{ftype}`")

    features = parsed.get("features") or {}
    if features:
        with st.expander("Структурированные признаки (features)"):
            st.json(features)

    st.subheader("Прецеденты ФАС")
    if not cases:
        st.info("Похожие решения не найдены.")
    else:
        for case in cases:
            verdict = case.get("verdict") or "—"
            sim = case.get("similarity_score")
            sim_s = f" · сходство {sim:.2f}" if isinstance(sim, (int, float)) else ""
            header = f"Решение {case.get('decision_id', '—')} — {verdict}{sim_s}"
            with st.expander(header):
                reasoning = (case.get("key_reasoning") or "").strip()
                st.write(reasoning or "(без описания)")
                if case.get("notice_id"):
                    st.caption(f"notice_id: {case['notice_id']}")

    pdf_bytes = generate_pdf(result, cases, filename)
    st.download_button(
        "📥 Скачать отчёт PDF",
        data=pdf_bytes,
        file_name=f"zakupki_check_{Path(filename).stem}.pdf",
        mime="application/pdf",
    )


def render_history(cache: CacheLayer) -> None:
    st.sidebar.subheader("История проверок")
    rows = cache.history(limit=15)
    if not rows:
        st.sidebar.caption("Пока пусто.")
        return
    for row in rows:
        fname = row.get("filename") or "—"
        risks = row.get("risk_count") or 0
        st.sidebar.markdown(
            f"<small>📄 <b>{fname}</b><br/>"
            f"<span style='color:#888'>рисков: {risks}</span></small>",
            unsafe_allow_html=True,
        )


def main() -> None:
    cache = get_cache()

    st.title("📋 ZakupkiCheck")
    st.caption("Проверка технического задания госзакупки на соответствие 44-ФЗ")

    st.sidebar.header("Настройки")
    st.sidebar.markdown(f"**Модель:** `{DEFAULT_MODEL}`")
    backend = retriever_backend() if st.sidebar.checkbox("Показать диагностику", value=False) else None
    if backend is not None:
        st.sidebar.markdown(f"**Retrieval backend:** `{backend}`")

    uploaded = st.file_uploader(
        "Загрузите ТЗ (.docx, .pdf, .txt, .md)",
        type=["docx", "pdf", "txt", "md"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info(
            "Загрузите файл с техническим заданием. Поддерживаются форматы "
            "**.docx, .pdf, .txt, .md**. Анализ выполняется одним вызовом LLM "
            "и сопровождается поиском похожих решений ФАС из локального корпуса."
        )
        render_history(cache)
        return

    with st.spinner("Парсинг документа..."):
        text = parse_document(uploaded)

    if not text or len(text) < 50:
        st.error(
            "Не удалось извлечь текст из документа (или он слишком короткий). "
            "Проверьте, что файл не повреждён."
        )
        render_history(cache)
        return

    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    st.text_area(
        "Превью документа",
        value=text[:1000] + ("\n..." if len(text) > 1000 else ""),
        height=180,
    )
    st.caption(f"Длина: {len(text):,} символов · sha256: `{doc_hash[:12]}...`")

    if st.button("🔍 Анализировать", type="primary", use_container_width=False):
        with st.spinner("Извлечение compliance-признаков..."):
            try:
                result = extract_compliance(
                    text, doc_hash, cache, filename=uploaded.name
                )
            except RuntimeError as e:
                st.error(f"Ошибка конфигурации: {e}")
                render_history(cache)
                return
            except Exception as e:  # pragma: no cover
                st.error(f"Неожиданная ошибка при вызове LLM: {type(e).__name__}: {e}")
                render_history(cache)
                return

        if result.get("status") != "ok" or result.get("parsed") is None:
            st.error(
                f"Извлечение не удалось: статус `{result.get('status')}`. "
                f"{result.get('error', '')}"
            )
            render_history(cache)
            return

        with st.spinner("Поиск прецедентов ФАС..."):
            try:
                cases = find_similar_cases(text, top_k=3)
            except Exception as e:  # pragma: no cover
                st.warning(f"Не удалось получить прецеденты ФАС: {e}")
                cases = []

        render_results(result, cases, uploaded.name)

    render_history(cache)


if __name__ == "__main__":
    main()
