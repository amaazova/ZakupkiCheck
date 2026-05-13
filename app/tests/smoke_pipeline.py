"""End-to-end smoke test: extract → rule engine → aggregate → retrieval → PDF."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from components.aggregator import aggregate  # noqa: E402
from components.extractor import extract  # noqa: E402
from components.recommendations import populate  # noqa: E402
from components.report import flag_label, generate_pdf  # noqa: E402
from components.retrieval import find_similar_cases  # noqa: E402
from components.rule_engine import evaluate  # noqa: E402
from components.schemas import DocType  # noqa: E402

DEFAULT_FIXTURE = APP_DIR / "tests" / "fixtures" / "test_tz_toljatti.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tz", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--doc-type", type=str, default="tz",
                        choices=[d.value for d in DocType])
    parser.add_argument("--out-pdf", type=Path, default=Path("/tmp/smoke_report.pdf"))
    args = parser.parse_args()

    if not args.tz.is_file():
        print(f"Document not found: {args.tz}", file=sys.stderr)
        return 2

    doc_type = DocType(args.doc_type)
    text = args.tz.read_text(encoding="utf-8")
    print(f"Document:  {args.tz}")
    print(f"Doc type:  {doc_type.value}")
    print(f"Length:    {len(text):,} characters")
    print()

    print("Pass 1 — LLM extraction...")
    t0 = time.monotonic()
    extraction = extract(text, doc_type=doc_type)
    t_extract = time.monotonic() - t0
    print(f"  elapsed:    {t_extract:.1f}s")
    print(f"  cost:       ${extraction.cost_usd:.5f}")
    print(f"  tokens:     in={extraction.input_tokens} out={extraction.output_tokens}")
    if extraction.error:
        print(f"  ERROR:      {extraction.error}")
        return 1

    facts = extraction.facts
    print(f"  brands:     {len(facts.brands)}  "
          f"(with equiv: {sum(1 for b in facts.brands if b.has_equivalent_clause)})")
    print(f"  restrict.:  {len(facts.restrictive_phrases)}")
    print(f"  units:      has_units={facts.measurements.has_units}  "
          f"has_ranges={facts.measurements.has_ranges}")
    print(f"  KTRU:       {facts.standards.ktru_code or '—'}")
    print(f"  accept:     {facts.completeness.has_acceptance_criteria}")
    print()

    print("Pass 2 — Rule engine...")
    findings = evaluate(facts, doc_type=doc_type)
    print(f"  findings:   {len(findings)}")
    print()

    similar = find_similar_cases(text, top_k=3)
    print(f"Retrieved {len(similar)} similar FAS decisions")
    print()

    doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    report = aggregate(
        extraction, findings, text,
        doc_hash=doc_hash, doc_type=doc_type, similar_cases=similar,
    )
    report = populate(report)

    print(f"Aggregated risk_score: {report.risk_score:.3f}")
    print(f"Verdict:               {report.verdict}")
    print(f"Findings (post-dedup): {len(report.findings)}")
    print(f"Recommendations:       {len(report.recommendations)}")
    print()
    print("Findings:")
    for i, f in enumerate(report.findings, 1):
        print(f"  {i}. {flag_label(f.flag_type)}  (conf={f.confidence:.0%})")
        if f.evidence_quote:
            print(f"     «{f.evidence_quote[:160]}»")
        else:
            print(f"     {f.description}")
    print()
    print("Similar decisions:")
    for c in report.similar_cases:
        print(f"  - {c.get('decision_id')}  {c.get('verdict')}  "
              f"sim={c.get('similarity_score')}")

    pdf_bytes = generate_pdf(report, args.tz.name)
    args.out_pdf.write_bytes(pdf_bytes)
    print()
    print(f"PDF written: {args.out_pdf} ({len(pdf_bytes):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
