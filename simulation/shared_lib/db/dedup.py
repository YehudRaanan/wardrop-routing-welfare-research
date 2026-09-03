"""
Run deduplication — check if a simulation with identical parameters
already exists before running.

Uses NULL-safe comparison (IS instead of =) because SQLite's UNIQUE
constraint treats NULLs as distinct, but we want NULL == NULL for
deduplication purposes.
"""

import sqlite3
from typing import Any


def find_existing_run(
    conn: sqlite3.Connection,
    flow_rate: int,
    route_choice_method: str,
    agent_mix: str = "100B",
    ema_alpha: float | None = None,
    logit_theta: float | None = None,
    toll_short_usd: float = 0.0,
    pct_short_fixed: float | None = None,
    demand_elasticity: float | None = None,
    random_seed: int | None = None,
    sim_duration_sec: int = 7200,
    warmup_sec: int = 3600,
    valid_only: bool = True,
) -> int | None:
    """Check if a completed run with these exact parameters exists.

    Uses IS instead of = for nullable columns so that
    NULL IS NULL evaluates to TRUE (standard SQL behavior).

    Args:
        All dedup key columns from the runs table.
        valid_only: If True, only match runs where is_valid=1.

    Returns:
        run_id if found, None otherwise.
    """
    sql = """
        SELECT run_id FROM runs
        WHERE flow_rate = ?
          AND route_choice_method = ?
          AND agent_mix = ?
          AND ema_alpha IS ?
          AND logit_theta IS ?
          AND toll_short_usd = ?
          AND pct_short_fixed IS ?
          AND demand_elasticity IS ?
          AND random_seed IS ?
          AND sim_duration_sec = ?
          AND warmup_sec = ?
          AND status = 'completed'
    """
    params: list[Any] = [
        flow_rate, route_choice_method, agent_mix,
        ema_alpha, logit_theta, toll_short_usd, pct_short_fixed,
        demand_elasticity, random_seed,
        sim_duration_sec, warmup_sec,
    ]

    if valid_only:
        sql += " AND is_valid = 1"

    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    # Support both dict-like (Row) and tuple access
    return row["run_id"] if isinstance(row, sqlite3.Row) else row[0]


def find_existing_run_from_config(
    conn: sqlite3.Connection,
    config: Any,
    valid_only: bool = True,
) -> int | None:
    """Convenience wrapper that accepts a RunConfig dataclass.

    Args:
        config: A RunConfig instance (from shared_lib.runners.run_config)
        valid_only: If True, only match valid completed runs.

    Returns:
        run_id if found, None otherwise.
    """
    return find_existing_run(
        conn,
        flow_rate=config.flow_rate,
        route_choice_method=config.route_choice_method,
        agent_mix=config.agent_mix,
        ema_alpha=config.ema_alpha,
        logit_theta=config.logit_theta,
        toll_short_usd=config.toll_short_usd,
        pct_short_fixed=config.pct_short_fixed,
        demand_elasticity=config.demand_elasticity,
        random_seed=config.random_seed,
        sim_duration_sec=config.sim_duration_sec,
        warmup_sec=config.warmup_sec,
        valid_only=valid_only,
    )
