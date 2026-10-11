import pytest
import sqlite3
from backend.app.database.db import get_db_connection, init_db, save_setting, get_setting

def test_database_initialization():
    init_db()
    conn = get_db_connection()
    try:
        # Check tables exist
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row["name"] for row in cursor.fetchall()]
        assert "violation_events" in tables
        assert "application_settings" in tables
        assert "processing_jobs" in tables
    finally:
        conn.close()

def test_settings_save_and_retrieve():
    test_key = "test_calibration_key"
    test_val = {"line": [0.1, 0.2, 0.3, 0.4], "mode": "red"}
    save_setting(test_key, test_val)

    retrieved = get_setting(test_key)
    assert retrieved == test_val
    assert get_setting("non_existent_key", default="default_val") == "default_val"
