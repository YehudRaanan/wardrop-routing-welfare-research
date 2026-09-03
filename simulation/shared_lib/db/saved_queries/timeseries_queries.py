"""
Per-minute timeseries query functions.
Applicable to: ALL phases (unless noted otherwise).
"""

import sqlite3


def get_tt_timeseries(
    conn: sqlite3.Connection,
    run_id: int,
    post_warmup_only: bool = True,
) -> list[dict]:
    """Return per-minute avg travel times for both paths.

    Returns list of dicts with keys:
        minute, avg_tt_short, avg_tt_long, total_tt_sec, pct_short
    """
    sql = """
        SELECT minute, avg_tt_short, avg_tt_long, total_tt_sec, pct_short
        FROM timeseries
        WHERE run_id = ?
    """
    params: list = [run_id]
    if post_warmup_only:
        sql += " AND is_warmup = 0"
    sql += " ORDER BY minute"

    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_nstart_timeseries(
    conn: sqlite3.Connection,
    run_id: int,
    post_warmup_only: bool = True,
) -> list[dict]:
    """Return per-minute instantaneous vehicle counts on each path.

    Applicable to: Phases 2+ (when N_start is recorded).

    Returns list of dicts with keys: minute, n_on_short, n_on_long
    """
    sql = """
        SELECT minute, n_on_short, n_on_long
        FROM timeseries
        WHERE run_id = ?
    """
    params: list = [run_id]
    if post_warmup_only:
        sql += " AND is_warmup = 0"
    sql += " ORDER BY minute"

    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_wardrop_gap_timeseries(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict]:
    """Return per-minute Wardrop gap: |TT_short - TT_long| / avg_TT.

    Post-warmup only. Minutes with NULL avg_tt are excluded.
    Applicable to: Phases 2, 3, 4.

    Returns list of dicts with keys: minute, wardrop_gap, avg_tt_short, avg_tt_long
    """
    rows = conn.execute("""
        SELECT
            minute,
            avg_tt_short,
            avg_tt_long,
            CASE
                WHEN avg_tt_short IS NOT NULL AND avg_tt_long IS NOT NULL
                     AND (avg_tt_short + avg_tt_long) > 0
                THEN ABS(avg_tt_short - avg_tt_long) / ((avg_tt_short + avg_tt_long) / 2.0)
                ELSE NULL
            END AS wardrop_gap
        FROM timeseries
        WHERE run_id = ?
          AND is_warmup = 0
          AND avg_tt_short IS NOT NULL
          AND avg_tt_long IS NOT NULL
        ORDER BY minute
    """, (run_id,)).fetchall()

    return [dict(row) for row in rows]
