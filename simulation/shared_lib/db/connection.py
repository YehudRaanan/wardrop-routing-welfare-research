"""
Database connection management.

Provides get_connection() which auto-creates the DB file and tables
on first access. Uses WAL journal mode for concurrent read performance.

GUARDRAIL: All connections install a SQL authorizer that BLOCKS:
  - DELETE from any table (data is immutable once written)
  - DROP TABLE (schema is protected)
  - Deleting the DB file

To invalidate a run, use mark_run_invalid() which updates is_valid=0
with a mandatory explanation. This is the ONLY way to "remove" data.
"""

import os
import sqlite3
import time
from pathlib import Path

from . import schema


# ─────────────────────────────────────────────────────────────────────
# Robust connection helper (2026-05-12)
# ─────────────────────────────────────────────────────────────────────
# Multiprocessing pools with 14+ workers contend on the same SQLite file.
# Pre-fix symptom: `conn.execute("PRAGMA journal_mode=WAL")` raises
# `OperationalError: database is locked` → unhandled exception kills the
# entire Pool → 50+ queued sims for that VM are lost.
#
# Fix layers:
#   1. `sqlite3.connect(timeout=60.0)` — C-level busy wait before any PRAGMA.
#   2. PRAGMA busy_timeout=60000 set immediately after connect.
#   3. retry-with-backoff around the PRAGMA setup if both still hit BUSY.

CONNECT_TIMEOUT_SEC = 60.0
BUSY_TIMEOUT_MS = 60000
MAX_RETRIES = 8
INITIAL_BACKOFF_SEC = 0.5


def _connect_with_retry(path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with two layers of busy-handling + retry.

    Returns a connection with PRAGMAs set (journal_mode=WAL, busy_timeout,
    synchronous=NORMAL, foreign_keys=ON). Caller is responsible for schema
    creation and authorizer installation.
    """
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            conn = sqlite3.connect(str(path), timeout=CONNECT_TIMEOUT_SEC)
            conn.row_factory = sqlite3.Row
            # Set busy_timeout FIRST so subsequent PRAGMAs benefit.
            conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.OperationalError as e:
            last_err = e
            if "database is locked" not in str(e).lower() and "busy" not in str(e).lower():
                raise  # not a lock error — re-raise
            try:
                conn.close()  # noqa: F821
            except Exception:
                pass
            if attempt == MAX_RETRIES - 1:
                break
            backoff = INITIAL_BACKOFF_SEC * (2 ** attempt)
            time.sleep(backoff)
    raise sqlite3.OperationalError(
        f"Failed to acquire DB connection after {MAX_RETRIES} retries: {last_err}"
    )


# ─────────────────────────────────────────────────────────────────────
# SQL AUTHORIZER — blocks destructive operations at the engine level
# ─────────────────────────────────────────────────────────────────────

def _sql_authorizer(action, arg1, arg2, db_name, trigger_name):
    """SQLite authorizer callback that blocks DELETE and DROP.

    This is registered on every connection via set_authorizer().
    It runs before each SQL statement and can DENY destructive ops.

    Returns:
        sqlite3.SQLITE_OK     — allow the operation
        sqlite3.SQLITE_DENY   — block with error
        sqlite3.SQLITE_IGNORE — silently skip (for columns)
    """
    # Block DELETE FROM any table
    if action == sqlite3.SQLITE_DELETE:
        # arg1 = table name
        # Allow sqlite internal tables (sqlite_sequence, etc.)
        if arg1 and not arg1.startswith("sqlite_"):
            return sqlite3.SQLITE_DENY

    # Block DROP TABLE
    if action == sqlite3.SQLITE_DROP_TABLE:
        return sqlite3.SQLITE_DENY

    # Allow everything else (SELECT, INSERT, UPDATE, CREATE, etc.)
    return sqlite3.SQLITE_OK

# Database location — single source of truth in shared_lib.paths.
# Default is C:\WardropBackups\primary\simulation.db (NTFS) post-2026-05-06
# rebuild; overridable via WARDROP_PRIMARY_DB env var. See shared_lib/paths.py
# for the full rationale (Drive Stream partial-write corruption).
import sys as _sys
_THIS_DIR = Path(__file__).resolve().parent
_SHARED_LIB_DIR = _THIS_DIR.parent
if str(_SHARED_LIB_DIR.parent) not in _sys.path:
    _sys.path.insert(0, str(_SHARED_LIB_DIR.parent))
from shared_lib.paths import PRIMARY_DB_PATH as DB_PATH  # noqa: E402

DB_DIR = DB_PATH.parent

_connection: sqlite3.Connection | None = None


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get or create a SQLite connection.

    On first call, creates the database file and all tables if they
    don't exist. Subsequent calls return the same connection.

    Args:
        db_path: Override path for testing. Default: data/simulation.db

    Returns:
        sqlite3.Connection with row_factory=sqlite3.Row
    """
    global _connection

    if _connection is not None:
        return _connection

    path = Path(db_path) if db_path else DB_PATH

    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect_with_retry(path)

    # Create or migrate schema (BEFORE authorizer — schema ops need DELETE)
    current_version = schema.get_schema_version(conn)
    if current_version == 0:
        schema.create_tables(conn)
    elif current_version < schema.SCHEMA_VERSION:
        schema.migrate(conn)

    # Install authorizer AFTER schema setup (blocks DELETE/DROP from here on)
    conn.set_authorizer(_sql_authorizer)

    _connection = conn
    return conn


def close_connection() -> None:
    """Close the global connection if open."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def get_fresh_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a new connection (not cached). For use in multiprocessing workers.

    Each worker process needs its own connection — SQLite connections
    cannot be shared across processes. Auto-creates tables if needed.
    """
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect_with_retry(path)

    # Create or migrate schema (BEFORE authorizer)
    current_version = schema.get_schema_version(conn)
    if current_version == 0:
        schema.create_tables(conn)
    elif current_version < schema.SCHEMA_VERSION:
        schema.migrate(conn)

    # Install authorizer AFTER schema setup
    conn.set_authorizer(_sql_authorizer)

    return conn
