"""Tests for matching pipeline helpers (ВКР §3.2.3): CR, fuzzy types, match_episode."""
from __future__ import annotations

import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[2]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from scripts.matching_pipeline import (  # noqa: E402
    CR_THRESHOLD,
    TYPE_SYNONYMS,
    containment_ratio,
    match_episode,
    types_match_fuzzy,
)


def test_cr_threshold_is_documented_value():
    assert CR_THRESHOLD == 0.3


def test_containment_ratio_full_overlap():
    assert containment_ratio("alpha beta gamma", "alpha beta gamma delta") == 1.0


def test_containment_ratio_half_overlap():
    # FAS has 4 tokens, 2 of them are in TZ → CR = 2/4 = 0.5
    assert containment_ratio("alpha beta gamma delta", "alpha beta omega") == 0.5


def test_containment_ratio_no_overlap():
    assert containment_ratio("alpha beta", "gamma delta") == 0.0


def test_containment_ratio_empty_fas():
    """CR is undefined when q_FAS is empty; we return 0 (nothing to cover)."""
    assert containment_ratio("", "alpha") == 0.0


def test_containment_ratio_is_case_insensitive():
    assert containment_ratio("Alpha Beta", "alpha BETA") == 1.0


def test_containment_ratio_uses_tokens_not_characters():
    """The metric tokenises by word, so "брэнд" and "брэндо" are distinct."""
    assert containment_ratio("брэнд", "брэндо") == 0.0


def test_types_match_fuzzy_equality():
    assert types_match_fuzzy("brand_without_equivalent", "brand_without_equivalent")


def test_types_match_fuzzy_synonym():
    """ВКР §3.2.3: brand_without_equivalent ≈ restrictive_requirement."""
    assert types_match_fuzzy("brand_without_equivalent", "restrictive_requirement")
    assert types_match_fuzzy("restrictive_requirement", "brand_without_equivalent")


def test_types_match_fuzzy_unrelated():
    assert not types_match_fuzzy("brand_without_equivalent", "inconsistent_dates")


def test_types_match_fuzzy_empty():
    assert not types_match_fuzzy("", "brand_without_equivalent")
    assert not types_match_fuzzy("brand_without_equivalent", "")


def test_type_synonyms_covers_all_specific_types():
    """Every specific type in the matching pipeline must appear in the dict."""
    expected = {
        "brand_without_equivalent", "incomplete_description",
        "restrictive_requirement", "ktru_mismatch",
        "missing_acceptance_terms", "inconsistent_dates",
    }
    assert expected.issubset(set(TYPE_SYNONYMS.keys()))


def _fas(t: str, quote: str) -> dict:
    return {"finding_type": t, "evidence_quote": quote, "description": ""}


def _tz(t: str, quote: str) -> dict:
    return {"flag_type": t, "evidence_quote": quote}


def test_match_episode_strict_default():
    fas = [_fas("brand_without_equivalent", "марки HP ProDesk 600")]
    tz_flags = [_tz("restrictive_requirement", "марки HP ProDesk 600")]
    out = match_episode(fas, tz_flags, tz_doc_norm="марки hp prodesk 600", quote_emb={})
    # Strict mode (default) — types differ → no match.
    assert out["matched"] == 0


def test_match_episode_fuzzy_allows_synonym():
    fas = [_fas("brand_without_equivalent", "марки HP ProDesk 600")]
    tz_flags = [_tz("restrictive_requirement", "марки HP ProDesk 600")]
    out = match_episode(
        fas, tz_flags,
        tz_doc_norm="марки hp prodesk 600", quote_emb={},
        fuzzy_types=True,
    )
    assert out["matched"] == 1
    assert len(out["containment_ratio_scores"]) == 1
    assert out["containment_ratio_scores"][0] > 0


def test_match_episode_cr_threshold_filters_low_overlap():
    fas = [_fas("brand_without_equivalent", "alpha beta gamma delta")]
    tz_flags = [_tz("brand_without_equivalent", "omega")]
    out_no_gate = match_episode(
        fas, tz_flags, tz_doc_norm="omega", quote_emb={},
    )
    assert out_no_gate["matched"] == 1  # strict-type match, no CR gate

    out_with_gate = match_episode(
        fas, tz_flags, tz_doc_norm="omega", quote_emb={},
        cr_threshold=CR_THRESHOLD,
    )
    assert out_with_gate["matched"] == 0  # gated out by CR < 0.3


def test_match_episode_cr_records_per_pair_score():
    fas = [_fas("brand_without_equivalent", "alpha beta gamma delta")]
    tz_flags = [_tz("brand_without_equivalent", "alpha beta")]  # CR = 2/4 = 0.5
    out = match_episode(
        fas, tz_flags, tz_doc_norm="alpha beta", quote_emb={},
    )
    assert out["matched"] == 1
    assert abs(out["containment_ratio_scores"][0] - 0.5) < 1e-9


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
