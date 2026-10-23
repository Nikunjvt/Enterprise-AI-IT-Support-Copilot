import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()

def init_db() -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS ingest_audit (
            content_sha256 TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            chunks INTEGER NOT NULL,
            vectors INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    con.commit()
    con.close()

def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
    )
    con.commit()
    con.close()


def has_ingested_hash(content_sha256: str) -> bool:
    con = sqlite3.connect(settings.audit_db_path)
    row = con.execute(
        "SELECT 1 FROM ingest_audit WHERE content_sha256 = ? LIMIT 1",
        (content_sha256,),
    ).fetchone()
    con.close()
    return row is not None


def write_ingest_audit(
    content_sha256: str,
    original_filename: str,
    chunks: int,
    vectors: int,
) -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        "INSERT INTO ingest_audit(content_sha256, original_filename, chunks, vectors, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (content_sha256, original_filename, chunks, vectors, datetime.now(timezone.utc).isoformat()),
    )
    con.commit()
    con.close()
