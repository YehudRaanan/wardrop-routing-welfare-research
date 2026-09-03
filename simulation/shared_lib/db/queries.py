"""
Database write operations — insert runs, vehicles, timeseries, regression.

All insert functions use parameterized queries to prevent SQL injection.
Batch inserts use executemany for performance (~100x faster than individual).
"""

import sqlite3
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# RUNS
# ─────────────────────────────────────────────────────────────────────

def insert_run(conn: sqlite3.Connection, params: dict[str, Any]) -> int:
    """Insert a new run and return its run_id.

    Args:
        conn: Database connection
        params: Dict with keys matching runs table columns.
                Required: flow_rate, route_choice_method
                Optional: agent_mix, ema_alpha, toll_short_usd,
                         pct_short_fixed, demand_elasticity, random_seed,
                         sim_duration_sec, warmup_sec, step_length,
                         fwd_ratio, bwd_ratio, status, notes

    Returns:
        run_id of the newly inserted row
    """
    columns = [
        "flow_rate", "route_choice_method", "agent_mix",
        "ema_alpha", "logit_theta", "toll_short_usd", "pct_short_fixed",
        "demand_elasticity", "random_seed",
        "sim_duration_sec", "warmup_sec", "step_length",
        "fwd_ratio", "bwd_ratio", "status", "notes",
    ]

    # Filter to only columns present in params
    present = [(c, params[c]) for c in columns if c in params]
    col_names = ", ".join(c for c, _ in present)
    placeholders = ", ".join("?" for _ in present)
    values = tuple(v for _, v in present)

    cursor = conn.execute(
        f"INSERT INTO runs ({col_names}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    return cursor.lastrowid


def update_run_status(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    error_message: str | None = None,
    wall_clock_sec: float | None = None,
    n_vehicles_fwd: int | None = None,
    n_vehicles_bwd: int | None = None,
    n_vehicles_blocked: int | None = None,
) -> None:
    """Update run status and optional completion fields."""
    conn.execute(
        """UPDATE runs
           SET status = ?,
               error_message = ?,
               wall_clock_sec = ?,
               n_vehicles_fwd = ?,
               n_vehicles_bwd = ?,
               n_vehicles_blocked = ?
           WHERE run_id = ?""",
        (status, error_message, wall_clock_sec,
         n_vehicles_fwd, n_vehicles_bwd, n_vehicles_blocked, run_id),
    )
    conn.commit()


def mark_run_invalid(
    conn: sqlite3.Connection,
    run_id: int,
    reason: str,
) -> None:
    """Mark a run as invalid with mandatory explanation.

    GUARDRAIL: This is the ONLY way to "remove" data from the DB.
    Data is NEVER deleted — only marked as invalid with explanation.
    The is_valid=0 flag causes dedup and analysis queries to skip
    this run (unless valid_only=False is explicitly passed).

    Args:
        conn: Database connection
        run_id: The run to invalidate
        reason: MANDATORY explanation (cannot be empty). Must describe
                why this run is invalid (e.g., "entry-edge bug",
                "wrong alpha parameter", "simulation crashed mid-run").

    Raises:
        ValueError: If reason is empty or None
    """
    if not reason or not reason.strip():
        raise ValueError(
            "Cannot mark run as invalid without explanation. "
            "Provide a clear reason in the 'reason' parameter."
        )

    # Verify run exists
    existing = conn.execute(
        "SELECT run_id, is_valid FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if existing is None:
        raise ValueError(f"Run {run_id} does not exist")

    conn.execute(
        "UPDATE runs SET is_valid = 0, validity_reason = ? WHERE run_id = ?",
        (reason.strip(), run_id),
    )
    conn.commit()
    print(f"  Run {run_id} marked INVALID: {reason.strip()}")


# ─────────────────────────────────────────────────────────────────────
# VEHICLES (batch insert)
# ─────────────────────────────────────────────────────────────────────

# Canonical column order for vehicle inserts
VEHICLE_COLUMNS = [
    "run_id", "vid", "direction", "vehicle_type", "group_id",
    "assigned_kmh", "speed_factor", "agent_type", "chosen_path", "choice_type",
    "planned_departure_sec", "actual_departure_sec", "arrival_sec",
    "travel_time_sec", "buffer_delay_sec", "is_warmup",
    "n_start_short", "n_start_long",
    "ema_short_at_choice", "ema_long_at_choice",
    "davis_pred_short", "davis_pred_long",
    # 2026-05-06 rebuild: TraCI ground-truth TT (independent of method prediction)
    "traci_tt_short_at_choice", "traci_tt_long_at_choice",
    "traci_tt_short_at_departure", "traci_tt_long_at_departure",
    "mediator_signal", "signal_followed", "compliance_code",
    "mediator_mode", "ic_margin_at_choice", "ic_constrained",
    "vot_usd_hr", "income_gbp", "trip_purpose",
    "toll_paid_usd", "generalized_cost_sec", "reservation_price_usd",
    "p_short_at_choice", "cost_short_at_choice", "cost_long_at_choice",
    "mean_speed_kmh", "n_speed_samples",
    "trip_status",
]

_VEHICLE_SQL = (
    f"INSERT INTO vehicles ({', '.join(VEHICLE_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in VEHICLE_COLUMNS)})"
)


def insert_vehicles_batch(
    conn: sqlite3.Connection,
    run_id: int,
    records: list[dict[str, Any]],
) -> int:
    """Bulk insert vehicle records for a single run.

    Args:
        conn: Database connection
        run_id: The run these vehicles belong to
        records: List of dicts with keys from VEHICLE_COLUMNS
                 (run_id is auto-prepended, missing keys become NULL)

    Returns:
        Number of rows inserted
    """
    if not records:
        return 0

    rows = []
    for rec in records:
        row = tuple(
            run_id if col == "run_id" else rec.get(col)
            for col in VEHICLE_COLUMNS
        )
        rows.append(row)

    conn.executemany(_VEHICLE_SQL, rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────────────────────────────
# TIMESERIES (batch insert)
# ─────────────────────────────────────────────────────────────────────

TIMESERIES_COLUMNS = [
    "run_id", "minute", "is_warmup",
    "avg_tt_short", "avg_tt_long", "total_tt_sec",
    "n_arrived_short", "n_arrived_long",
    "n_chose_short", "n_chose_long", "pct_short",
    "n_on_short", "n_on_long", "n_blocked",
    "ema_short", "ema_long",
    "reg_intercept_short", "reg_slope_short",
    "reg_intercept_long", "reg_slope_long",
    "compliance_rate", "mediator_mode",
    "throughput_veh_per_min",
    "p10_speed_kmh", "p50_speed_kmh", "p90_speed_kmh",
    "n_speed_obs", "avg_speed_kmh",
    "dep_avg_tt_short", "dep_avg_tt_long",
    "n_departed_short", "n_departed_long",
]

_TIMESERIES_SQL = (
    f"INSERT INTO timeseries ({', '.join(TIMESERIES_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in TIMESERIES_COLUMNS)})"
)


def insert_timeseries_batch(
    conn: sqlite3.Connection,
    run_id: int,
    records: list[dict[str, Any]],
) -> int:
    """Bulk insert timeseries records for a single run."""
    if not records:
        return 0

    rows = []
    for rec in records:
        row = tuple(
            run_id if col == "run_id" else rec.get(col)
            for col in TIMESERIES_COLUMNS
        )
        rows.append(row)

    conn.executemany(_TIMESERIES_SQL, rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────────────────────────────
# RUN REGRESSION
# ─────────────────────────────────────────────────────────────────────

def insert_run_regression(
    conn: sqlite3.Connection,
    run_id: int,
    intercept_short: float | None = None,
    slope_short: float | None = None,
    intercept_long: float | None = None,
    slope_long: float | None = None,
    n_obs_short: int | None = None,
    n_obs_long: int | None = None,
    r_squared_short: float | None = None,
    r_squared_long: float | None = None,
) -> None:
    """Insert or replace regression results for a run."""
    conn.execute(
        """INSERT OR REPLACE INTO run_regression
           (run_id, intercept_short, slope_short, intercept_long, slope_long,
            n_obs_short, n_obs_long, r_squared_short, r_squared_long)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, intercept_short, slope_short, intercept_long, slope_long,
         n_obs_short, n_obs_long, r_squared_short, r_squared_long),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────
# READ HELPERS
# ─────────────────────────────────────────────────────────────────────

def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    """Fetch a single run by ID."""
    return conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()


def get_completed_runs(
    conn: sqlite3.Connection,
    method: str | None = None,
    agent_mix: str | None = None,
    valid_only: bool = True,
) -> list[sqlite3.Row]:
    """Fetch completed runs, optionally filtered."""
    sql = "SELECT * FROM runs WHERE status = 'completed'"
    params: list[Any] = []

    if valid_only:
        sql += " AND is_valid = 1"
    if method:
        sql += " AND route_choice_method = ?"
        params.append(method)
    if agent_mix:
        sql += " AND agent_mix = ?"
        params.append(agent_mix)

    sql += " ORDER BY flow_rate"
    return conn.execute(sql, params).fetchall()
