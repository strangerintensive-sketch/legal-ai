import sqlite3
import json
from pathlib import Path

DB_PATH = Path("data/legal_ai.db")


def get_conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            raw_text TEXT,
            structured_fields TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS drafts (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            content TEXT,
            evidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS edits (
            id TEXT PRIMARY KEY,
            draft_id TEXT,
            original TEXT,
            edited TEXT,
            extracted_rules TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (draft_id) REFERENCES drafts(id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS preference_rules (
            id TEXT PRIMARY KEY,
            rule TEXT,
            source_edit_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
