"""
Cost function linearity query functions.
Used for TT vs N regression, breakpoint detection, cross-flow pooling.
"""

import sqlite3


def get_nstart_vs_tt_per_vehicle(
    conn: sqlite3.Connection,
    run_id: int,
    path: str = "short",
) -> list[dict]:
    """Return (N_start, travel_time) pairs for linearity regression.

    Filters to post-warmup FWD vehicles on the specified path.

    Args:
        run_id: Which run to query
        path: "short" or "long"

    Returns:
        List of dicts with keys: n_start, travel_time_sec
    """
    n_col = "n_start_short" if path == "short" else "n_start_long"

    rows = conn.execute(f"""
        SELECT {n_col} AS n_start, travel_time_sec
        FROM vehicles
        WHERE run_id = ?
          AND is_warmup = 0
          AND direction = 'fwd'
          AND chosen_path = ?
          AND {n_col} IS NOT NULL
        ORDER BY {n_col}
    """, (run_id, path)).fetchall()

    return [dict(row) for row in rows]


def get_avg_n_and_tt_per_run(
    conn: sqlite3.Connection,
    method: str | None = None,
    path: str = "short",
    valid_only: bool = True,
) -> list[dict]:
    """Return (avg_N, avg_TT, flow_rate) per run for cost function plotting.

    Aggregates post-warmup timeseries data. Useful for cross-flow
    cost function characterization.

    Returns:
        List of dicts with keys: run_id, flow_rate, avg_n, avg_tt
    """
    n_col = "n_on_short" if path == "short" else "n_on_long"
    tt_col = "avg_tt_short" if path == "short" else "avg_tt_long"

    sql = f"""
        SELECT
            r.run_id,
            r.flow_rate,
            AVG(t.{n_col}) AS avg_n,
            AVG(t.{tt_col}) AS avg_tt
        FROM runs r
        JOIN timeseries t ON r.run_id = t.run_id
        WHERE r.status = 'completed'
          AND t.is_warmup = 0
          AND t.{n_col} IS NOT NULL
          AND t.{tt_col} IS NOT NULL
    """
    params: list = []
    if valid_only:
        sql += " AND r.is_valid = 1"
    if method:
        sql += " AND r.route_choice_method = ?"
        params.append(method)

    sql += " GROUP BY r.run_id, r.flow_rate ORDER BY r.flow_rate"

    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_capacity_discovery_data(
    conn: sqlite3.Connection,
    path: str = "short",
    valid_only: bool = True,
) -> list[dict]:
    """Return capacity discovery results: flow vs avg_TT, avg_N, avg_speed.

    Queries runs with route_choice_method='none' (pure capacity test).

    Returns:
        List of dicts with keys: flow_rate, avg_n, avg_tt, n_blocked
    """
    tt_col = "avg_tt_short" if path == "short" else "avg_tt_long"
    n_col = "n_on_short" if path == "short" else "n_on_long"

    sql = f"""
        SELECT
            r.flow_rate,
            AVG(t.{n_col}) AS avg_n,
            AVG(t.{tt_col}) AS avg_tt,
            r.n_vehicles_blocked
        FROM runs r
        JOIN timeseries t ON r.run_id = t.run_id
        WHERE r.route_choice_method = 'none'
          AND r.status = 'completed'
          AND t.is_warmup = 0
          AND t.{n_col} IS NOT NULL
    """
    if valid_only:
        sql += " AND r.is_valid = 1"

    sql += " GROUP BY r.run_id, r.flow_rate ORDER BY r.flow_rate"

    return [dict(row) for row in conn.execute(sql).fetchall()]
