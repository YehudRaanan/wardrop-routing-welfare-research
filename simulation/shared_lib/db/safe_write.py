"""Safe-write helpers for the primary simulation DB.

Provides guards that prevent the 2026-04-30 wipe failure mode (filesystem-level
overwrite of the primary DB by a runaway script). The SQLite engine-level
authorizer in `connection.py` only protects against DELETE/DROP issued from a
SQL connection; it cannot stop a `cp` or a `Path.write_bytes()` from clobbering
the file. This module fills that gap.

Two layered defenses:

1. `verify_primary_intact()` — call before any code path that intends to write
   to the primary. Confirms the guard sentinel exists with the expected token
   AND the on-disk DB is above a sane minimum size. Raises PrimaryDBWipedError
   if either check fails.

2. `atomic_replace_primary(staging_path)` — atomically replaces the primary
   with a fully-validated staging DB. Performs SQLite integrity check and a
   row-count regression check before swapping. Takes a pre-merge snapshot.

3. `set_primary_readonly()` / `set_primary_writable()` — toggles the OS-level
   read-only flag around merge windows. On Windows this sets the read-only
   attribute; on POSIX it sets mode 0o444. Either way, a stray `cp` onto
   the primary fails with EACCES while the flag is set.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from shared_lib.paths import PRIMARY_DB_GUARD_PATH, PRIMARY_DB_PATH

GUARD_TOKEN: Final[str] = "PRIMARY_DB_GUARD_v1_DO_NOT_DELETE"
MIN_PRIMARY_DB_SIZE_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB


class PrimaryDBWipedError(RuntimeError):
    """Raised when a guard check detects the primary DB is missing, too small,
    or the sentinel token is gone. Callers should refuse to write."""


class IntegrityFailureError(RuntimeError):
    """Raised when a staging DB fails PRAGMA integrity_check before swap."""


class RowCountRegressionError(RuntimeError):
    """Raised when a staging DB has fewer valid+completed runs than the
    current primary. Prevents a wipe-equivalent merge from being installed."""


def verify_primary_intact(primary_path: Path = PRIMARY_DB_PATH) -> None:
    """Confirm the primary DB looks healthy. Raise if not.

    Checks:
        1. The guard sentinel file exists at the expected path with the
           expected token.
        2. The primary DB exists and is at least MIN_PRIMARY_DB_SIZE_BYTES.

    Either failure indicates the DB has been wiped or replaced with a
    schema-only file (the 2026-04-30 incident's exact failure mode).
    """
    if not PRIMARY_DB_GUARD_PATH.exists():
        raise PrimaryDBWipedError(
            f"Guard sentinel missing at {PRIMARY_DB_GUARD_PATH}. "
            "DB may have been wiped or moved; refuse to proceed."
        )
    token = PRIMARY_DB_GUARD_PATH.read_text(encoding="utf-8").strip()
    if token != GUARD_TOKEN:
        raise PrimaryDBWipedError(
            f"Guard sentinel token mismatch: got {token!r} expected "
            f"{GUARD_TOKEN!r}. Refuse to proceed."
        )
    if not primary_path.exists():
        raise PrimaryDBWipedError(f"Primary DB missing at {primary_path}.")
    size = primary_path.stat().st_size
    if size < MIN_PRIMARY_DB_SIZE_BYTES:
        raise PrimaryDBWipedError(
            f"Primary DB at {primary_path} is only {size:,} bytes — "
            f"below the {MIN_PRIMARY_DB_SIZE_BYTES:,} byte minimum. "
            "Likely wiped to schema-only; refuse to proceed."
        )


def count_valid_completed_runs(db_path: Path) -> int:
    """Count valid+completed runs in a DB. Used by the regression check."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM runs WHERE is_valid=1 AND status='completed'"
        ).fetchone()[0]


def atomic_replace_primary(
    staging_path: Path,
    primary_path: Path = PRIMARY_DB_PATH,
    expected_min_runs: int | None = None,
    snapshot_label: str = "merge",
) -> Path:
    """Atomically replace the primary DB with a staging DB.

    Sequence:
        1. PRAGMA integrity_check on staging — must return "ok"
        2. Row-count regression check on staging — must be >= current primary's
           valid+completed runs (or expected_min_runs if provided)
        3. Take pre-swap snapshot of primary
        4. os.replace(staging, primary) — atomic on a single filesystem

    Returns:
        Path to the pre-swap snapshot (kept for potential rollback).

    Raises:
        IntegrityFailureError: staging DB integrity_check returned non-ok
        RowCountRegressionError: staging has fewer rows than primary
    """
    with sqlite3.connect(f"file:{staging_path}?mode=ro", uri=True) as conn:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise IntegrityFailureError(
            f"Staging DB at {staging_path} failed integrity_check: {result!r}"
        )

    staging_runs = count_valid_completed_runs(staging_path)
    if primary_path.exists():
        primary_runs = count_valid_completed_runs(primary_path)
    else:
        primary_runs = 0
    threshold = max(primary_runs, expected_min_runs or 0)
    if staging_runs < threshold:
        raise RowCountRegressionError(
            f"Staging DB has {staging_runs} valid+completed runs but the "
            f"primary has {primary_runs} (threshold {threshold}). Refuse to "
            f"swap — this would lose data."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = primary_path.with_name(
        f"{primary_path.name}.pre-{snapshot_label}-{timestamp}"
    )
    if primary_path.exists():
        shutil.copy2(primary_path, snapshot_path)

    set_primary_writable(primary_path)
    os.replace(staging_path, primary_path)
    return snapshot_path


def set_primary_readonly(primary_path: Path = PRIMARY_DB_PATH) -> None:
    """Set the primary DB to read-only at the OS level.

    On Windows this sets the read-only attribute; on POSIX this sets mode
    0o444. A stray `cp`, `Path.write_bytes()`, or fopen-w against the primary
    fails with EACCES while the flag is set.
    """
    if not primary_path.exists():
        return
    primary_path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def set_primary_writable(primary_path: Path = PRIMARY_DB_PATH) -> None:
    """Restore the primary DB to read-write. Call only inside a sanctioned
    merge window."""
    if not primary_path.exists():
        return
    primary_path.chmod(
        stat.S_IREAD | stat.S_IWRITE | stat.S_IRGRP | stat.S_IROTH
    )


def write_guard_sentinel() -> None:
    """Create or refresh the guard sentinel file. Call once during setup."""
    PRIMARY_DB_GUARD_PATH.write_text(GUARD_TOKEN + "\n", encoding="utf-8")
