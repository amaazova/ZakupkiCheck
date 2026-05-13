"""W2 Session A — Matching pipeline (pure-Python, no API calls)."""
from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import PROJECT_ROOT
from .data_loaders import _episode_to_decision_id, load_fas_text, load_tz
from .extraction_runner import append_jsonl, read_jsonl

EVAL = PROJECT_ROOT / "workspace" / "eval"
OUT = EVAL / "matching_results.jsonl"

SPECIFIC_TYPES = frozenset({
    "brand_without_equivalent", "incomplete_description",
    "restrictive_requirement", "ktru_mismatch",
    "inconsistent_dates", "missing_acceptance_terms",
})
SEMANTIC_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FUZZY_THRESHOLD = 0.7
MIN_FUZZY_QUOTE_LEN = 30

# Containment Ratio threshold (ВКР §3.2.3). A pair (q_FAS, q_TZ) is considered
# evidence-aligned when CR ≥ CR_THRESHOLD, where
#     CR = |tokens(q_FAS) ∩ tokens(q_TZ)| / |tokens(q_FAS)|
# (i.e. share of FAS tokens covered by the TZ quote). CR is reported per pair
# in the matching output; it is also available as an optional matching gate
# via match_episode(..., cr_threshold=...).
CR_THRESHOLD = 0.3

# Type-synonym dictionary (ВКР §3.2.3). When the LLM extracted a TZ flag under
# a different but semantically adjacent type than the FAS finding, the pair
# should still be considered for matching. Strict equality is the default
# (preserves W1–W2 results); fuzzy mode is opt-in via match_episode(..., fuzzy_types=True).
TYPE_SYNONYMS: dict[str, frozenset[str]] = {
    "brand_without_equivalent": frozenset({
        "brand_without_equivalent", "restrictive_requirement",
    }),
    "restrictive_requirement": frozenset({
        "restrictive_requirement", "brand_without_equivalent", "incomplete_description",
    }),
    "incomplete_description": frozenset({
        "incomplete_description", "restrictive_requirement", "ktru_mismatch",
    }),
    "ktru_mismatch": frozenset({
        "ktru_mismatch", "incomplete_description",
    }),
    "missing_acceptance_terms": frozenset({
        "missing_acceptance_terms", "incomplete_description",
    }),
    "inconsistent_dates": frozenset({"inconsistent_dates"}),
    "other": frozenset({"other"}),
}


# Entity extraction

QUOTED_RE = re.compile(r'[«"„"]([^»""\n]{2,80})[»""]')
GOST_RE = re.compile(r'(?:ГОСТ|ISO|EN)\s*[\d\.\-]+(?:[A-ZА-Я\-]*\d+)?', re.IGNORECASE)
KTRU_RE = re.compile(r'\d{2}\.\d{2}\.\d{2}\.\d{3}')
LATIN_MODEL_RE = re.compile(r'\b[A-Z][A-Z0-9\-]{2,}\b')


def extract_entities(text: str) -> set[str]:
    if not text:
        return set()
    ents: set[str] = set()
    for m in QUOTED_RE.findall(text):
        s = m.strip().lower()
        if 2 <= len(s) <= 60:
            ents.add(("q", s))
    for m in GOST_RE.findall(text):
        ents.add(("gost", re.sub(r"\s+", "", m).lower()))
    for m in KTRU_RE.findall(text):
        ents.add(("ktru", m))
    for m in LATIN_MODEL_RE.findall(text):
        if any(c.isdigit() for c in m):
            ents.add(("lat", m.lower()))
    return ents


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# Citation containment

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    return _WS_RE.sub(" ", t).strip()


def contained(quote: str, doc_norm: str) -> bool:
    """True if `quote` is in `doc_norm` exactly, or fuzzily matches a window."""
    if not quote or not doc_norm:
        return False
    q = normalize(quote)
    if not q:
        return False
    if q in doc_norm:
        return True
    if len(q) < MIN_FUZZY_QUOTE_LEN:
        return False
    qlen = len(q)
    step = max(qlen // 2, 200)
    for i in range(0, len(doc_norm), step):
        window = doc_norm[i:i + 2 * qlen]
        if not window:
            break
        if difflib.SequenceMatcher(None, q, window).ratio() >= FUZZY_THRESHOLD:
            return True
    return False


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    """Lowercased word-token set used by containment_ratio."""
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def containment_ratio(q_fas: str, q_tz: str) -> float:
    """Containment Ratio (ВКР §3.2.3).

    CR = |tokens(q_FAS) ∩ tokens(q_TZ)| / |tokens(q_FAS)|

    Returns 0.0 when q_FAS is empty (i.e. nothing to cover).
    """
    fas_t = _tokens(q_fas)
    if not fas_t:
        return 0.0
    return len(fas_t & _tokens(q_tz)) / len(fas_t)


def types_match_fuzzy(fas_type: str, tz_type: str) -> bool:
    """Fuzzy type match using the TYPE_SYNONYMS dictionary (ВКР §3.2.3).

    Two types match if they are equal, or if `tz_type` is in the synonym set
    of `fas_type`. Used by match_episode when invoked with `fuzzy_types=True`.
    """
    if not fas_type or not tz_type:
        return False
    if fas_type == tz_type:
        return True
    return tz_type in TYPE_SYNONYMS.get(fas_type, frozenset())


# Matching (per episode)

def match_episode(
    fas_findings: list[dict],
    tz_flags: list[dict],
    *,
    tz_doc_norm: str,
    quote_emb: dict[str, np.ndarray],
    type_set: frozenset[str] | None = None,
    fuzzy_types: bool = False,
    cr_threshold: float | None = None,
) -> dict[str, Any]:
    """Greedy 1-to-1 assignment by type, scored by semantic_sim then entity_overlap.

    Args:
        fuzzy_types: when True, allow synonym-based type matching per
            TYPE_SYNONYMS (ВКР §3.2.3). Default False preserves W1–W2 results.
        cr_threshold: when set, drop candidate pairs whose
            containment_ratio(q_FAS, q_TZ) is below this value. The per-pair CR
            is always recorded in the output (`containment_ratio_scores`).

    Returns: dict with matched pairs, scores, per-pair containment + CR.
    """
    if type_set is not None:
        fas_eligible = [f for f in fas_findings if f.get("finding_type") in type_set]
        tz_eligible = [f for f in tz_flags if f.get("flag_type") in type_set]
    else:
        fas_eligible = list(fas_findings)
        tz_eligible = list(tz_flags)

    used_tz: set[int] = set()
    pairs: list[tuple[int, int]] = []  # (fas_idx, tz_idx)
    entity_overlap_scores: list[float] = []
    semantic_sim_scores: list[float] = []
    tz_citation_containment: list[bool] = []
    containment_ratio_scores: list[float] = []
    matched_specific = 0

    for fi, ff in enumerate(fas_eligible):
        ft = ff.get("finding_type")
        if fuzzy_types:
            candidates = [
                (ti, tf) for ti, tf in enumerate(tz_eligible)
                if ti not in used_tz and types_match_fuzzy(ft, tf.get("flag_type"))
            ]
        else:
            candidates = [
                (ti, tf) for ti, tf in enumerate(tz_eligible)
                if ti not in used_tz and tf.get("flag_type") == ft
            ]
        if not candidates:
            continue
        # Score each candidate; pick best.
        fq = (ff.get("evidence_quote") or "").strip()
        fdesc = (ff.get("description") or "").strip()
        f_ent = extract_entities(fq) | extract_entities(fdesc)
        f_emb = quote_emb.get(fq) if fq else None
        best = None
        best_score = (-1.0, -1.0)  # (sem, entity)
        for ti, tf in candidates:
            tq = (tf.get("evidence_quote") or "").strip()
            t_ent = extract_entities(tq)
            ent_j = jaccard(f_ent, t_ent)
            t_emb = quote_emb.get(tq) if tq else None
            if f_emb is not None and t_emb is not None:
                sem = float(np.dot(f_emb, t_emb))  # already L2-normalized
            else:
                sem = 0.0
            cr = containment_ratio(fq, tq)
            if cr_threshold is not None and cr < cr_threshold:
                continue  # below the ВКР CR gate
            score = (sem, ent_j)
            if score > best_score:
                best_score = score
                best = (ti, ent_j, sem, tq, cr)
        if best is None:
            continue
        ti, ent_j, sem, tq, cr = best
        used_tz.add(ti)
        pairs.append((fi, ti))
        entity_overlap_scores.append(ent_j)
        semantic_sim_scores.append(sem)
        tz_citation_containment.append(contained(tq, tz_doc_norm))
        containment_ratio_scores.append(cr)
        if ft in SPECIFIC_TYPES:
            matched_specific += 1

    return {
        "matched": len(pairs),
        "matched_specific": matched_specific,
        "pairs": pairs,
        "entity_overlap_scores": entity_overlap_scores,
        "semantic_sim_scores": semantic_sim_scores,
        "tz_citation_containment": tz_citation_containment,
        "containment_ratio_scores": containment_ratio_scores,
        "n_fas_eligible": len(fas_eligible),
        "n_tz_eligible": len(tz_eligible),
    }


# Schema-agnostic helpers

def extract_tz_flags(rec: dict) -> list[dict]:
    """Return the risk_flags list regardless of schema."""
    if "parsed" in rec:
        p = rec.get("parsed")
        raw = (p.get("risk_flags") or []) if isinstance(p, dict) else []
    else:
        raw = rec.get("risk_flags") or []
    return [f for f in raw if isinstance(f, dict)]


# Driver

def collect_quotes(fas: list[dict], tz: list[dict]) -> list[str]:
    quotes: set[str] = set()
    for r in fas:
        for f in r.get("findings") or []:
            if not isinstance(f, dict):
                continue
            q = (f.get("evidence_quote") or "").strip()
            if q:
                quotes.add(q)
            d = (f.get("description") or "").strip()
            if d:
                quotes.add(d)
    for r in tz:
        for f in extract_tz_flags(r):
            q = (f.get("evidence_quote") or "").strip()
            if q:
                quotes.add(q)
    return list(quotes)


def _load_st_model() -> "SentenceTransformer":
    print(f"[matching] loading {EMBED_MODEL}...")
    return SentenceTransformer(EMBED_MODEL)


def _encode(model: "SentenceTransformer", quotes: list[str]) -> dict[str, np.ndarray]:
    print(f"[matching] encoding {len(quotes)} quotes (L2-normalised)...")
    embs = model.encode(
        quotes, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return {q: e for q, e in zip(quotes, embs)}


def run_matching(
    *,
    fas_path: Path | str,
    tz_path: Path | str,
    eval_path: Path | str,
    output_path: Path | str,
    quote_emb: dict[str, np.ndarray] | None = None,
    include_l0_column: bool = False,
    l0_path: Path | str | None = None,
) -> Path:
    """Run matching for a given FAS × TZ pair. Writes per-episode JSONL."""
    fas_path = Path(fas_path)
    tz_path = Path(tz_path)
    eval_path = Path(eval_path)
    output_path = Path(output_path)

    print(f"[matching] FAS={fas_path.name}  TZ={tz_path.name}  → {output_path.name}")
    eval_df = pd.read_csv(eval_path, dtype={"notice_id": str})
    fas = read_jsonl(fas_path)
    tz = read_jsonl(tz_path)
    fas_by_ep = {r["episode_id"]: r for r in fas}
    tz_by_ep = {r["episode_id"]: r for r in tz}

    l0_by_ep: dict[str, dict] = {}
    if include_l0_column:
        if l0_path is None:
            raise ValueError("l0_path is required when include_l0_column=True")
        l0 = read_jsonl(Path(l0_path))
        l0_by_ep = {r["episode_id"]: r for r in l0}

    if quote_emb is None:
        quotes = collect_quotes(fas, tz)
        print(f"[matching] unique quotes: {len(quotes)}")
        model = _load_st_model()
        quote_emb = _encode(model, quotes)

    output_path.unlink(missing_ok=True)
    tz_doc_cache: dict[str, str] = {}
    fas_doc_cache: dict[str, str] = {}

    print("[matching] per-episode matching...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), unit="ep"):
        eid = row["episode_id"]
        notice_id = row["notice_id"]
        decision_id = _episode_to_decision_id(eid)
        verdict = row["fas_verdict"]
        stratum = row["stratum"]
        cluster_id = int(row["cluster_id"])

        f_rec = fas_by_ep.get(eid, {})
        tz_rec = tz_by_ep.get(eid, {})

        all_findings = f_rec.get("findings") or []
        fas_est = [f for f in all_findings if f.get("severity") == "established"]
        tz_flags = extract_tz_flags(tz_rec)

        if notice_id not in tz_doc_cache:
            tz_doc_cache[notice_id] = normalize(load_tz(notice_id) or "")
        if decision_id not in fas_doc_cache:
            fas_doc_cache[decision_id] = normalize(load_fas_text(decision_id) or "")
        tz_doc_norm = tz_doc_cache[notice_id]
        fas_doc_norm = fas_doc_cache[decision_id]

        m_all = match_episode(
            fas_est, tz_flags,
            tz_doc_norm=tz_doc_norm, quote_emb=quote_emb,
            type_set=None,
        )
        m_spec = match_episode(
            fas_est, tz_flags,
            tz_doc_norm=tz_doc_norm, quote_emb=quote_emb,
            type_set=SPECIFIC_TYPES,
        )
        m_l0 = None
        if include_l0_column:
            l0_rec = l0_by_ep.get(eid, {})
            l0_flags = l0_rec.get("risk_flags") or []
            m_l0 = match_episode(
                [f for f in fas_est if f.get("finding_type") == "brand_without_equivalent"],
                l0_flags,
                tz_doc_norm=tz_doc_norm, quote_emb=quote_emb,
                type_set=frozenset({"brand_without_equivalent"}),
            )

        fas_citation_containment: list[bool] = []
        for ff in fas_est:
            q = (ff.get("evidence_quote") or "").strip()
            fas_citation_containment.append(contained(q, fas_doc_norm) if q else False)

        episode_pred_strict = (
            "violation_established" if m_spec["matched_specific"] >= 1
            else "violation_not_established"
        )
        episode_pred_loose = (
            "violation_established" if len(tz_flags) >= 1
            else "violation_not_established"
        )

        record = {
            "episode_id": eid,
            "notice_id": notice_id,
            "decision_id": decision_id,
            "cluster_id": cluster_id,
            "stratum": stratum,
            "fas_verdict": verdict,
            "n_fas_findings_total": len(all_findings),
            "n_fas_findings_established": len(fas_est),
            "n_tz_flags": len(tz_flags),
            "type_matches_all": m_all["matched"],
            "type_matches_specific": m_spec["matched_specific"],
            "n_fas_eligible_specific": m_spec["n_fas_eligible"],
            "n_tz_eligible_specific": m_spec["n_tz_eligible"],
            "entity_overlap_scores_all": m_all["entity_overlap_scores"],
            "entity_overlap_scores_specific": m_spec["entity_overlap_scores"],
            "semantic_sim_scores_all": m_all["semantic_sim_scores"],
            "semantic_sim_scores_specific": m_spec["semantic_sim_scores"],
            "tz_citation_containment_all": m_all["tz_citation_containment"],
            "tz_citation_containment_specific": m_spec["tz_citation_containment"],
            "fas_citation_containment": fas_citation_containment,
            "episode_pred_strict": episode_pred_strict,
            "episode_pred_loose": episode_pred_loose,
        }
        if m_l0 is not None:
            record["type_matches_l0_brand"] = m_l0["matched"]
            record["n_l0_flags"] = len(l0_by_ep.get(eid, {}).get("risk_flags") or [])
        append_jsonl(output_path, record)

    print(f"[matching] done → {output_path}")
    return output_path


def main() -> None:
    """W2 baseline path: V4 Flash FAS × L1 TZ (with L0 brand column)."""
    run_matching(
        fas_path=EVAL / "fas_findings.jsonl",
        tz_path=EVAL / "tz_features_L1.jsonl",
        eval_path=EVAL / "eval_dataset_v10.csv",
        output_path=OUT,
        include_l0_column=True,
        l0_path=EVAL / "tz_features_L0.jsonl",
    )


if __name__ == "__main__":
    main()
