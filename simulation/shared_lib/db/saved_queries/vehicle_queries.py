"""
Per-vehicle query functions.
Applicable to: ALL phases (unless noted otherwise).

Every function takes a connection and returns structured results.

Travel-time metrics apply the full-path filter (see
``shared_lib.core.vehicle_filter``) to exclude vehicles spawned mid-path via
``departPos="free"``. Count queries do NOT apply the filter — path split
percentages are over ALL vehicles that made a route choice, regardless of
where they physically spawned.
"""

import sqlite3

from shared_lib.core.vehicle_filter import FULL_PATH_SQL_FILTER


def get_post_warmup_tt_by_path(
    conn: sqlite3.Connection,
    run_id: int,
) -> dict:
    """Return avg travel time per path, post-warmup FWD full-path vehicles.

    Excludes vehicles whose travel_time_sec is below the physical minimum
    for a full-path trip (``departPos="free"`` artifact). See
    ``shared_lib.core.vehicle_filter`` for details.

    Returns:
        {"avg_tt_short": float, "avg_tt_long": float,
         "n_short": int, "n_long": int, "total_tt": float}
    """
    row = conn.execute(f"""
        SELECT
            AVG(CASE WHEN chosen_path='short' THEN travel_time_sec END) AS avg_tt_short,
            AVG(CASE WHEN chosen_path='long'  THEN travel_time_sec END) AS avg_tt_long,
            SUM(CASE WHEN chosen_path='short' THEN 1 ELSE 0 END) AS n_short,
            SUM(CASE WHEN chosen_path='long'  THEN 1 ELSE 0 END) AS n_long,
            SUM(travel_time_sec) AS total_tt
        FROM vehicles
        WHERE run_id = ?
          AND is_warmup = 0
          AND direction = 'fwd'
          AND {FULL_PATH_SQL_FILTER}
    """, (run_id,)).fetchone()

    return dict(row) if row else {}


def get_path_split(
    conn: sqlite3.Connection,
    run_id: int,
) -> float | None:
    """Return fraction of post-warmup FWD vehicles that chose short path.

    Returns:
        Float between 0 and 1, or None if no vehicles.
    """
    row = conn.execute("""
        SELECT
            CAST(SUM(CASE WHEN chosen_path='short' THEN 1 ELSE 0 END) AS REAL)
            / COUNT(*) AS pct_short
        FROM vehicles
        WHERE run_id = ?
          AND is_warmup = 0
          AND direction = 'fwd'
    """, (run_id,)).fetchone()

    return row["pct_short"] if row else None


def get_wardrop_gap(
    conn: sqlite3.Connection,
    run_id: int,
) -> float | None:
    """Return Wardrop gap: |avg_TT_short - avg_TT_long| / avg_TT_network.

    Post-warmup FWD vehicles only.
    Returns None if either path has no vehicles.
    """
    tt = get_post_warmup_tt_by_path(conn, run_id)
    if not tt or tt["avg_tt_short"] is None or tt["avg_tt_long"] is None:
        return None

    avg_network = (
        (tt["avg_tt_short"] * tt["n_short"] + tt["avg_tt_long"] * tt["n_long"])
        / (tt["n_short"] + tt["n_long"])
    )
    if avg_network == 0:
        return None

    return abs(tt["avg_tt_short"] - tt["avg_tt_long"]) / avg_network


def get_vehicle_count_by_path(
    conn: sqlite3.Connection,
    run_id: int,
) -> dict:
    """Return vehicle counts by path (post-warmup FWD only).

    Returns:
        {"short": int, "long": int, "total": int}
    """
    rows = conn.execute("""
        SELECT chosen_path, COUNT(*) AS n
        FROM vehicles
        WHERE run_id = ? AND is_warmup = 0 AND direction = 'fwd'
        GROUP BY chosen_path
    """, (run_id,)).fetchall()

    result = {"short": 0, "long": 0, "total": 0}
    for row in rows:
        result[row["chosen_path"]] = row["n"]
    result["total"] = result["short"] + result["long"]
    return result
