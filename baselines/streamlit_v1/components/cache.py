"""SQLite cache for extraction results, keyed by sha256(document_text)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class CacheLayer:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extraction_cache (
                    doc_hash    TEXT PRIMARY KEY,
                    result      TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    filename    TEXT,
                    doc_type    TEXT,
                    risk_count  INTEGER
                )
                """
            )
            conn.commit()

    def get(self, doc_hash: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result FROM extraction_cache WHERE doc_hash = ?", (doc_hash,)
            ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["result"])
        except json.JSONDecodeError:
            return None

    def put(
        self,
        doc_hash: str,
        result: dict,
        filename: str,
        doc_type: str = "ТЗ",
    ) -> None:
        risk_count = len((result or {}).get("risk_flags", []) or [])
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO extraction_cache
                    (doc_hash, result, timestamp, filename, doc_type, risk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_hash, json.dumps(result, ensure_ascii=False), ts, filename, doc_type, risk_count),
            )
            conn.commit()

    def history(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_hash, filename, doc_type, risk_count, timestamp
                FROM extraction_cache
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
