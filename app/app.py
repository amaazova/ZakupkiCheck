"""ZakupkiCheck v2 — Streamlit entrypoint."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path("/Users/aza/Downloads/zakupki/last")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components.aggregator import aggregate  # noqa: E402
from components.cache import CacheLayer  # noqa: E402
from components.extractor import DEFAULT_MODEL, extract  # noqa: E402
from components.highlighter import highlight_text  # noqa: E402
from components.logging_config import get_logger  # noqa: E402
from components.parser import parse_document  # noqa: E402
from components.recommendations import populate  # noqa: E402
from components.report import flag_label, generate_pdf  # noqa: E402
from components.retrieval import find_similar_cases, preload_retriever  # noqa: E402
from components.rule_engine import evaluate  # noqa: E402
from components.schemas import DocType, Finding, RiskReport  # noqa: E402

APP_DIR = Path(__file__).parent
CACHE_DB = APP_DIR / "data" / "cache.db"
LR_MODEL_PATH = APP_DIR / "models" / "lr_model.joblib"

DOC_TYPE_LABELS = {
    DocType.TZ: "Техническое задание",
    DocType.CONTRACT: "Проект контракта",
    DocType.NOTICE: "Извещение о закупке",
    DocType.DOCUMENTATION: "Документация о закупке",
}

VERDICT_LABEL = {
    "high": "Высокий риск нарушений",
    "medium": "Средний риск",
    "low": "Нарушений не обнаружено",
}
VERDICT_COLOR = {"high": "#e74c3c", "medium": "#f39c12", "low": "#27ae60"}

logger = get_logger("zakupkicheck.app")

st.set_page_config(
    page_title="ZakupkiCheck",
    page_icon="📋",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_cache() -> CacheLayer:
    return CacheLayer(CACHE_DB)


@st.cache_resource(show_spinner="Загрузка моделей...")
def preload() -> dict:
    import joblib

    backend = preload_retriever()
    lr_loaded = False
    if LR_MODEL_PATH.is_file():
        try:
            joblib.load(LR_MODEL_PATH)
            lr_loaded = True
        except Exception as exc:
            logger.warning("lr_preload_failed", error=str(exc))
    logger.info("preload_complete", retrieval_backend=backend, lr_loaded=lr_loaded)
    return {"retrieval_backend": backend, "lr_loaded": lr_loaded}


def render_verdict_bar(report: RiskReport) -> None:
    color = VERDICT_COLOR[report.verdict]
    label = VERDICT_LABEL[report.verdict]
    n = len(report.findings)
    st.markdown(
        f'<div style="background:{color};color:white;padding:14px 20px;'
        f'border-radius:8px;font-size:18px;font-weight:600;">'
        f'{label} — найдено замечаний: {n}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_findings(report: RiskReport) -> None:
    st.subheader("Замечания и рекомендации")
    if not report.findings:
        st.success("Замечаний не обнаружено — документ выглядит корректным.")
        return
    recs_by_idx = {r.finding_index: r for r in report.recommendations}
    for i, finding in enumerate(report.findings):
        rec = recs_by_idx.get(i)
        conf_pct = int(finding.confidence * 100)
        label = flag_label(finding.flag_type)
        title = f"{i + 1}. {label} — уверенность {conf_pct}%"
        with st.expander(title, expanded=(i == 0)):
            quote = (finding.evidence_quote or "").strip()
            if quote:
                st.markdown(f"**Фрагмент документа:** «{quote}»")
            elif finding.description:
                st.caption(finding.description)
            if rec is not None:
                st.markdown(f"**Рекомендация:** {rec.fix}")
                st.markdown(f"**Основание:** {rec.law_ref}")
                if rec.law_text:
                    st.caption(rec.law_text)


def render_document_with_highlights(text: str, findings: list[Finding]) -> None:
    st.subheader("Текст документа")
    html_text = highlight_text(text, findings)
    st.markdown(
        f'<div style="max-height:420px;overflow-y:auto;border:1px solid #ddd;'
        f'padding:16px;border-radius:8px;font-size:14px;line-height:1.55;'
        f'white-space:pre-wrap;font-family:Georgia,serif;">{html_text}</div>',
        unsafe_allow_html=True,
    )


def render_precedents(report: RiskReport) -> None:
    st.subheader("Похожие решения ФАС")
    if not report.similar_cases:
        st.info("Похожие решения не найдены.")
        return
    for case in report.similar_cases:
        verdict = case.get("verdict") or ""
        verdict_ru = (
            "нарушение установлено" if verdict == "violation_established"
            else "нарушение не установлено" if verdict == "violation_not_established"
            else verdict or "—"
        )
        decision_id = case.get("decision_id") or "—"
        with st.expander(f"Решение ФАС №{decision_id} — {verdict_ru}"):
            reasoning = (case.get("key_reasoning") or "").strip()
            st.write(reasoning or "(краткое обоснование отсутствует)")


def render_diagnostics(report: RiskReport, *, cached: bool, preload_info: dict) -> None:
    st.subheader("Диагностика")
    extraction = report.extraction
    st.json({
        "doc_type": report.doc_type.value,
        "ml_risk_score": round(report.risk_score, 4),
        "verdict": report.verdict,
        "n_findings": len(report.findings),
        "cached": cached,
        "cost_usd": report.cost_usd,
        "doc_chars": report.doc_chars,
        "retrieval_backend": preload_info.get("retrieval_backend"),
        "model": DEFAULT_MODEL,
        "extraction_elapsed_sec": round(extraction.elapsed_sec, 2) if extraction else None,
        "extraction_error": extraction.error if extraction else None,
    })
    if report.facts is not None:
        with st.expander("Извлечённые факты (Pass 1)"):
            st.json(report.facts.model_dump(mode="json"))


def render_history(cache: CacheLayer) -> None:
    st.sidebar.subheader("История проверок")
    rows = cache.history(limit=15)
    if not rows:
        st.sidebar.caption("Пока пусто.")
        return
    for row in rows:
        fname = row.get("filename") or "—"
        risks = int(row.get("risk_count") or 0)
        status = "нарушений не найдено" if risks == 0 else f"замечаний: {risks}"
        st.sidebar.markdown(
            f"<small><b>{fname}</b><br/>"
            f"<span style='color:#888'>{status}</span></small>",
            unsafe_allow_html=True,
        )


def _analyze(
    text: str,
    doc_hash: str,
    filename: str,
    doc_type: DocType,
    cache: CacheLayer,
) -> tuple[RiskReport, bool]:
    cache_key = f"{doc_hash}:{doc_type.value}"
    cached_payload = cache.get(cache_key)
    if cached_payload:
        try:
            report = RiskReport.model_validate(cached_payload)
            return report, True
        except Exception as exc:
            logger.warning("cache_validation_failed", error=str(exc), doc_hash=doc_hash[:12])

    with st.spinner("Извлечение фактов (один вызов LLM, около 20–40 сек)..."):
        extraction = extract(text, doc_type=doc_type)

    findings = evaluate(extraction.facts, doc_type=doc_type)

    with st.spinner("Поиск похожих решений ФАС..."):
        try:
            similar = find_similar_cases(text, top_k=3)
        except Exception as exc:
            logger.warning("retrieval_failed", error=str(exc))
            similar = []

    report = aggregate(
        extraction, findings, text,
        doc_hash=doc_hash, doc_type=doc_type, similar_cases=similar,
    )
    report = populate(report)
    try:
        cache.put(cache_key, report.model_dump(mode="json"), filename=filename,
                  doc_type=DOC_TYPE_LABELS[doc_type])
    except Exception as exc:
        logger.warning("cache_put_failed", error=str(exc))
    return report, False


def main() -> None:
    preload_info = preload()
    cache = get_cache()

    st.title("ZakupkiCheck")
    st.caption("Проверка закупочной документации на соответствие 44-ФЗ")

    st.sidebar.header("Настройки")
    diagnostics = st.sidebar.checkbox("Показать диагностику", value=False)
    if diagnostics:
        st.sidebar.markdown(f"**Модель:** `{DEFAULT_MODEL}`")
        st.sidebar.markdown(f"**Retrieval:** `{preload_info.get('retrieval_backend')}`")
        st.sidebar.markdown(f"**LR:** `{'loaded' if preload_info.get('lr_loaded') else 'fallback'}`")

    col_type, col_upload = st.columns([1, 2])
    with col_type:
        doc_type = st.selectbox(
            "Тип документа",
            options=list(DOC_TYPE_LABELS.keys()),
            format_func=lambda x: DOC_TYPE_LABELS[x],
            index=0,
        )
    with col_upload:
        uploaded = st.file_uploader(
            "Загрузите документ (.docx, .pdf, .txt, .md)",
            type=["docx", "pdf", "txt", "md"],
            accept_multiple_files=False,
        )

    if uploaded is None:
        st.info(
            "Выберите тип документа и загрузите файл. Анализ выполняется "
            "одним вызовом LLM (извлечение фактов) и набором детерминированных "
            "правил по статьям 44-ФЗ; параллельно ищутся похожие решения ФАС."
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
    st.caption(f"Длина: {len(text):,} символов")
    if diagnostics:
        st.caption(f"sha256: `{doc_hash[:12]}...`")

    if not st.button("Анализировать", type="primary"):
        render_history(cache)
        return

    try:
        report, was_cached = _analyze(text, doc_hash, uploaded.name, doc_type, cache)
    except RuntimeError as exc:
        st.error(f"Ошибка конфигурации: {exc}")
        render_history(cache)
        return
    except Exception as exc:  # pragma: no cover
        logger.error("pipeline_failure", error=str(exc), error_type=type(exc).__name__)
        st.error(f"Неожиданная ошибка: {type(exc).__name__}: {exc}")
        render_history(cache)
        return

    if report.extraction and report.extraction.error and not report.findings:
        st.warning(
            "Не удалось извлечь факты из документа: "
            f"{report.extraction.error}. Попробуйте загрузить другой файл."
        )

    render_verdict_bar(report)
    render_document_with_highlights(text, report.findings)
    render_findings(report)
    render_precedents(report)

    pdf_bytes = generate_pdf(report, uploaded.name)
    st.download_button(
        "Скачать отчёт PDF",
        data=pdf_bytes,
        file_name=f"zakupki_check_{Path(uploaded.name).stem}.pdf",
        mime="application/pdf",
    )

    if diagnostics:
        render_diagnostics(report, cached=was_cached, preload_info=preload_info)

    render_history(cache)


if __name__ == "__main__":
    main()
