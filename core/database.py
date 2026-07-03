"""
Very small SQLite wrapper used to persist every audited query.
"""

import sqlite3
import os
import json
import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audit_log.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            confidence REAL,
            hallucination_score REAL,
            hallucination_flag INTEGER,
            trust_score REAL,
            sources TEXT,
            user_feedback TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS redteam_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            prompt_id TEXT,
            category TEXT,
            prompt TEXT,
            response TEXT,
            attack_succeeded INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def log_query(query: str, answer: str, confidence: float, hallucination_score: float,
              hallucination_flag: bool, trust_score: float, sources: List[str]) -> int:
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO audit_log (timestamp, query, answer, confidence, hallucination_score,
                                hallucination_flag, trust_score, sources, user_feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            query,
            answer,
            confidence,
            hallucination_score,
            int(hallucination_flag),
            trust_score,
            json.dumps(sources),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def update_feedback(row_id: int, feedback: str):
    conn = get_connection()
    conn.execute("UPDATE audit_log SET user_feedback = ? WHERE id = ?", (feedback, row_id))
    conn.commit()
    conn.close()


def log_redteam_result(prompt_id: str, category: str, prompt: str, response: str, attack_succeeded: bool):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO redteam_log (timestamp, prompt_id, category, prompt, response, attack_succeeded)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.now().isoformat(timespec="seconds"),
            prompt_id,
            category,
            prompt,
            response,
            int(attack_succeeded),
        ),
    )
    conn.commit()
    conn.close()


def fetch_all_queries() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_all_redteam() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM redteam_log ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_calibration_pairs() -> List[Dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT confidence, user_feedback FROM audit_log WHERE user_feedback IS NOT NULL"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
