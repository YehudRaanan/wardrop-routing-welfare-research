"""
Welfare and economic query functions.
Applicable to: Phase 5+ (toll, VOT, generalized cost analysis) and the
CS pivot v2 stack (LOCKED 2026-05-26 — see
analyses/_synthesis/welfare_measure_dossier.md).
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from shared_lib.core.welfare_metrics import AgentMetrics


def get_toll_revenue_per_flow(
    conn: sqlite3.Connection,
    run_id: int,
) -> dict:
    """Return total toll revenue and avg toll per vehicle (post-warmup FWD).

    Returns:
        {"total_revenue": float, "avg_toll": float, "n_tolled": int}
    """
    row = conn.execute("""
        SELECT
            SUM(toll_paid_usd) AS total_revenue,
            AVG(CASE WHEN toll_paid_usd > 0 THEN toll_paid_usd END) AS avg_toll,
            SUM(CASE WHEN toll_paid_usd > 0 THEN 1 ELSE 0 END) AS n_tolled
        FROM vehicles
        WHERE run_id = ?
          AND is_warmup = 0
          AND direction = 'fwd'
    """, (run_id,)).fetchone()

    return dict(row) if row else {}


# ─────────────────────────────────────────────────────────────────────
# CS pivot v2 — load CS-eligible agents (methodology.md §0.5)
# ─────────────────────────────────────────────────────────────────────


def load_cs_eligible_vehicles(
    conn: sqlite3.Connection,
    run_id: int,
    w_start: float,
    w_end: float,
) -> list[AgentMetrics]:
    """Load natively logit-eligible vehicles from one run as AgentMetrics.

    Per methodology.md §0.5 / §1.1, only vehicles with populated
    ``p_short_at_choice`` (native logit choice — EMA, Davis-threshold,
    logit-eligible drivers in mixed populations) are returned by this
    helper. Two other sources of agents contribute to the CS aggregate
    but are NOT returned here:

      * **SO baseline** (``fixed_split_vot``) — fetch via
        :func:`analyses._shared.cs_backfill.backfill_so_run`.
      * **Queued-never-departed** — fetch via
        :func:`analyses._shared.cs_backfill.backfill_queued_for_run`.

    The full CS-eligible cohort for a cell is therefore the union of:
        ``load_cs_eligible_vehicles(...) + backfill_queued_for_run(...)``
    (for non-SO runs), or
        ``backfill_so_run(...) + backfill_queued_for_run(...)``
    (for SO runs).

    Args:
        conn: SQLite connection.
        run_id: Target run.
        w_start, w_end: Analysis-window endpoints (seconds).

    Returns:
        List of AgentMetrics ready for ``WelfareMetrics(agents=...)``.
    """
    rows = conn.execute(
        """
        SELECT vid, vot_usd_hr, travel_time_sec, toll_paid_usd, chosen_path,
               actual_departure_sec,
               p_short_at_choice, cost_short_at_choice, cost_long_at_choice,
               ema_short_at_choice, ema_long_at_choice
          FROM vehicles
         WHERE run_id = ?
           AND is_warmup = 0
           AND actual_departure_sec IS NOT NULL
           AND actual_departure_sec >= ?
           AND actual_departure_sec < ?
           AND p_short_at_choice IS NOT NULL
           AND cost_short_at_choice IS NOT NULL
           AND cost_long_at_choice IS NOT NULL
        """,
        (run_id, w_start, w_end),
    ).fetchall()

    agents: list[AgentMetrics] = []
    for r in rows:
        (vid, vot, tt, toll_paid, chosen_path, actual_dep,
         p_short, cost_short, cost_long, ema_short, ema_long) = r
        if vot is None or vot <= 0:
            continue
        agents.append(
            AgentMetrics(
                vid=str(vid),
                vot_usd_hr=float(vot),
                travel_time_sec=float(tt) if tt is not None else 0.0,
                toll_paid_usd=float(toll_paid) if toll_paid is not None else 0.0,
                chosen_path=str(chosen_path) if chosen_path else "",
                ema_short_at_choice=float(ema_short) if ema_short is not None else 0.0,
                ema_long_at_choice=float(ema_long) if ema_long is not None else 0.0,
                departure_time_sec=float(actual_dep),
                p_short_at_choice=float(p_short),
                cost_short_at_choice=float(cost_short),
                cost_long_at_choice=float(cost_long),
            )
        )
    return agents


def get_cs_coverage_for_run(
    conn: sqlite3.Connection,
    run_id: int,
    w_start: float,
    w_end: float,
) -> dict:
    """Return CS-eligibility decomposition for one run.

    Maps directly onto methodology.md §11's ``primary.csv`` columns
    ``n_veh_cs_eligible``, ``n_veh_logit_native``, ``n_veh_queued_cf``,
    ``n_veh_excluded_mixed``. The SO counterfactual count is run-level
    metadata; this helper does not infer SO-ness from method.

    Returns:
        {
          "n_veh_logit_native": int,
          "n_veh_queued_cf": int,
          "n_veh_excluded_mixed": int,
          "n_veh_total_in_W": int,
        }
    """
    row = conn.execute(
        """
        SELECT
            SUM(CASE
                WHEN p_short_at_choice IS NOT NULL THEN 1 ELSE 0 END) AS n_logit_native,
            SUM(CASE
                WHEN actual_departure_sec IS NULL AND planned_departure_sec IS NOT NULL
                THEN 1 ELSE 0 END) AS n_queued_cf,
            SUM(CASE
                WHEN p_short_at_choice IS NULL
                 AND actual_departure_sec IS NOT NULL
                THEN 1 ELSE 0 END) AS n_excluded_mixed,
            COUNT(*) AS n_total
        FROM vehicles
        WHERE run_id = ?
          AND is_warmup = 0
          AND ( (actual_departure_sec IS NOT NULL
                  AND actual_departure_sec >= ?
                  AND actual_departure_sec <  ?)
             OR (actual_departure_sec IS NULL
                  AND planned_departure_sec IS NOT NULL
                  AND planned_departure_sec >= ?
                  AND planned_departure_sec <  ?) )
        """,
        (run_id, w_start, w_end, w_start, w_end),
    ).fetchone()
    if row is None:
        return {
            "n_veh_logit_native": 0,
            "n_veh_queued_cf": 0,
            "n_veh_excluded_mixed": 0,
            "n_veh_total_in_W": 0,
        }
    n_logit, n_queued, n_excluded, n_total = row
    return {
        "n_veh_logit_native": int(n_logit or 0),
        "n_veh_queued_cf": int(n_queued or 0),
        "n_veh_excluded_mixed": int(n_excluded or 0),
        "n_veh_total_in_W": int(n_total or 0),
    }
