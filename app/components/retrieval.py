"""Similar-case retrieval over fas_findings.jsonl: ChromaDB → TF-IDF fallback."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
_FINDINGS_CANDIDATES = [
    _REPO_ROOT / "research" / "results" / "data" / "fas_findings.jsonl",
    Path(__file__).parent.parent / "data" / "fas_findings.jsonl",
]
FINDINGS_PATH = next((p for p in _FINDINGS_CANDIDATES if p.is_file()), _FINDINGS_CANDIDATES[0])
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "fas_findings_v1"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _load_findings() -> list[dict]:
    out: list[dict] = []
    if not FINDINGS_PATH.is_file():
        return out
    with FINDINGS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _build_corpus_text(finding: dict) -> str:
    pieces: list[str] = []
    kr = (finding.get("key_reasoning") or "").strip()
    if kr:
        pieces.append(kr)
    for f in finding.get("findings") or []:
        desc = (f.get("description") or "").strip()
        if desc:
            pieces.append(desc)
    return "\n".join(pieces).strip()


class _Retriever:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self._backend: str = "none"
        self._collection: Any = None
        self._vectorizer: Any = None
        self._matrix: Any = None
        self._records: list[dict] = []

    def _try_chroma(self) -> bool:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except Exception:
            return False

        try:
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )

            existing = {c.name for c in client.list_collections()}
            if COLLECTION_NAME in existing:
                collection = client.get_collection(
                    name=COLLECTION_NAME, embedding_function=embed_fn
                )
                if collection.count() > 0:
                    self._collection = collection
                    self._backend = "chroma"
                    return True

            findings = _load_findings()
            if not findings:
                return False

            collection = client.get_or_create_collection(
                name=COLLECTION_NAME, embedding_function=embed_fn
            )
            ids, docs, metas = [], [], []
            for rec in findings:
                text = _build_corpus_text(rec)
                if not text:
                    continue
                ids.append(str(rec.get("episode_id") or rec.get("decision_id")))
                docs.append(text)
                metas.append({
                    "decision_id": str(rec.get("decision_id") or ""),
                    "notice_id": str(rec.get("notice_id") or ""),
                    "verdict": str(rec.get("overall_verdict") or rec.get("gold_fas_verdict") or ""),
                    "key_reasoning": (rec.get("key_reasoning") or "")[:2000],
                    "cluster_id": int(rec.get("cluster_id") or -1),
                })
            if not ids:
                return False

            batch = 64
            for i in range(0, len(ids), batch):
                collection.add(
                    ids=ids[i:i + batch],
                    documents=docs[i:i + batch],
                    metadatas=metas[i:i + batch],
                )
            self._collection = collection
            self._backend = "chroma"
            return True
        except Exception:
            return False

    def _try_tfidf(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return False

        findings = _load_findings()
        if not findings:
            return False

        records, texts = [], []
        for rec in findings:
            text = _build_corpus_text(rec)
            if not text:
                continue
            records.append(rec)
            texts.append(text)
        if not texts:
            return False

        self._vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        self._records = records
        self._backend = "tfidf"
        return True

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            if not self._try_chroma():
                self._try_tfidf()
            self._ready = True

    def query(self, text: str, top_k: int) -> list[dict]:
        self._ensure_ready()
        text = (text or "").strip()
        if not text:
            return []

        if self._backend == "chroma" and self._collection is not None:
            n = min(top_k, max(1, self._collection.count()))
            res = self._collection.query(query_texts=[text], n_results=n)
            ids = (res.get("ids") or [[]])[0]
            metadatas = (res.get("metadatas") or [[]])[0]
            distances = (res.get("distances") or [[]])[0]
            out: list[dict] = []
            for eid, meta, dist in zip(ids, metadatas, distances):
                meta = meta or {}
                # Chroma по умолчанию возвращает squared L2; маппим в similarity ∈ [0, 1].
                sim = max(0.0, 1.0 - float(dist or 0.0) / 2.0)
                out.append({
                    "decision_id": meta.get("decision_id", ""),
                    "notice_id": meta.get("notice_id", ""),
                    "verdict": meta.get("verdict", ""),
                    "key_reasoning": meta.get("key_reasoning", ""),
                    "similarity_score": round(sim, 3),
                    "episode_id": eid,
                })
            return out

        if self._backend == "tfidf" and self._matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            qv = self._vectorizer.transform([text])
            sims = cosine_similarity(qv, self._matrix).ravel()
            top_idx = np.argsort(-sims)[:top_k]
            out: list[dict] = []
            for i in top_idx:
                rec = self._records[int(i)]
                out.append({
                    "decision_id": str(rec.get("decision_id") or ""),
                    "notice_id": str(rec.get("notice_id") or ""),
                    "verdict": str(rec.get("overall_verdict") or rec.get("gold_fas_verdict") or ""),
                    "key_reasoning": rec.get("key_reasoning") or "",
                    "similarity_score": round(float(sims[int(i)]), 3),
                    "episode_id": str(rec.get("episode_id") or ""),
                })
            return out

        return []

    @property
    def backend(self) -> str:
        return self._backend


_GLOBAL_RETRIEVER: _Retriever | None = None


def _get_retriever() -> _Retriever:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = _Retriever()
    return _GLOBAL_RETRIEVER


def find_similar_cases(text: str, top_k: int = 3) -> list[dict]:
    candidates = _get_retriever().query(text, top_k=max(top_k * 3, top_k))
    seen: dict[str, dict] = {}
    for case in candidates:
        decision_id = str(case.get("decision_id") or case.get("episode_id") or "")
        if not decision_id:
            continue
        existing = seen.get(decision_id)
        if existing is None or case.get("similarity_score", 0) > existing.get("similarity_score", 0):
            seen[decision_id] = case
    unique = sorted(seen.values(), key=lambda c: c.get("similarity_score", 0), reverse=True)
    return unique[:top_k]


def preload_retriever() -> str:
    retriever = _get_retriever()
    retriever._ensure_ready()
    return retriever.backend


def retriever_backend() -> str:
    return _get_retriever().backend
