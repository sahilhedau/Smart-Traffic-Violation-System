import sqlite3
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from backend.app.config import config

logger = logging.getLogger("traffic_system.database")

_db_initialized = False

def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to SQLite with WAL enabled and row factory set."""
    global _db_initialized
    conn = sqlite3.connect(str(config.DATABASE_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    if not _db_initialized:
        _create_tables(conn)
        _db_initialized = True
    return conn

def _create_tables(conn: sqlite3.Connection):
    """Internal helper to ensure tables exist."""
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS violation_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            camera_id TEXT NOT NULL DEFAULT 'CAM_01',
            tracking_id INTEGER,
            confidence REAL NOT NULL DEFAULT 0.0,
            severity TEXT NOT NULL DEFAULT 'MEDIUM',
            status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
            evidence_path TEXT,
            crop_path TEXT,
            plate_text TEXT,
            plate_raw_text TEXT,
            plate_confidence REAL DEFAULT 0.0,
            details_json TEXT,
            reviewer_notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON violation_events(event_type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON violation_events(status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON violation_events(created_at);")

        conn.execute("""
        CREATE TABLE IF NOT EXISTS application_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)

        conn.execute("""
        CREATE TABLE IF NOT EXISTS processing_jobs (
            id TEXT PRIMARY KEY,
            source_filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'QUEUED',
            progress REAL NOT NULL DEFAULT 0.0,
            current_frame INTEGER DEFAULT 0,
            total_frames INTEGER DEFAULT 0,
            fps REAL DEFAULT 0.0,
            annotated_video_path TEXT,
            violations_count INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)

def init_db():
    """Initializes database tables if they do not exist."""
    conn = get_db_connection()
    try:
        _create_tables(conn)
        logger.info("Database schema initialized successfully.")
    finally:
        conn.close()

def save_setting(key: str, value: Any):
    """Store setting in SQLite as JSON."""
    conn = get_db_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        val_str = json.dumps(value)
        with conn:
            conn.execute("""
            INSERT INTO application_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """, (key, val_str, now))
    finally:
        conn.close()

def get_setting(key: str, default: Any = None) -> Any:
    """Retrieve setting from SQLite."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT value FROM application_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["value"])
        return default
    finally:
        conn.close()
