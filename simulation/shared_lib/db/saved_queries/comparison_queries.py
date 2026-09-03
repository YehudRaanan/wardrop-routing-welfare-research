"""
Cross-method comparison query functions.
Used for PoA analysis, method comparison, gap closing calculations.
"""

import sqlite3
from typing import Any


def get_total_tt_per_method_and_flow(
    conn: sqlite3.Connection,
    methods: list[str] | None = None,
    valid_only: bool = True,
) -> list[dict]:
    """Return total post-warmup travel time per method per flow.

    Joins runs + vehicles to compute aggregate TT.

    Returns list of dicts with keys:
        flow_rate, route_choice_method, total_tt, n_vehicles, avg_tt
    """
    sql = """
        SELECT
            r.flow_rate,
            r.route_choice_method,
            SUM(v.travel_time_sec) AS total_tt,
            COUNT(v.id) AS n_vehicles,
            AVG(v.travel_time_sec) AS avg_tt
        FROM runs r
        JOIN vehicles v ON r.run_id = v.run_id
        WHERE r.status = 'completed'
          AND v.is_warmup = 0
          AND v.direction = 'fwd'
    """
    params: list[Any] = []

    if valid_only:
        sql += " AND r.is_valid = 1"
    if methods:
        placeholders = ", ".join("?" for _ in methods)
        sql += f" AND r.route_choice_method IN ({placeholders})"
        params.extend(methods)

    sql += " GROUP BY r.flow_rate, r.route_choice_method"
    sql += " ORDER BY r.flow_rate, r.route_choice_method"

    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_so_total_tt_per_flow(
    conn: sqlite3.Connection,
    valid_only: bool = True,
) -> dict[int, float]:
    """Return the minimum total TT across all fixed-split runs per flow.

    This is the System Optimal (SO) baseline for PoA calculation.

    Returns:
        Dict mapping flow_rate -> minimum total_tt
    """
    sql = """
        SELECT
            r.flow_rate,
            MIN(sub.total_tt) AS so_total_tt
        FROM (
            SELECT
                r2.run_id,
                r2.flow_rate,
                SUM(v.travel_time_sec) AS total_tt
            FROM runs r2
            JOIN vehicles v ON r2.run_id = v.run_id
            WHERE r2.route_choice_method = 'fixed_split'
              AND r2.status = 'completed'
              AND v.is_warmup = 0
              AND v.direction = 'fwd'
    """
    if valid_only:
        sql += " AND r2.is_valid = 1"

    sql += """
            GROUP BY r2.run_id, r2.flow_rate
        ) sub
        JOIN runs r ON sub.run_id = r.run_id
        GROUP BY r.flow_rate
        ORDER BY r.flow_rate
    """

    result = {}
    for row in conn.execute(sql).fetchall():
        result[row["flow_rate"]] = row["so_total_tt"]
    return result
