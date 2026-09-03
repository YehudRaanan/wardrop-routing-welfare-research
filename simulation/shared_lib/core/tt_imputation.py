"""
Travel-time imputation for missing vehicles.

==============================================================================
WHY THIS EXISTS
==============================================================================

At congested flows, SUMO may refuse vehicle insertion because the entry edge
is full (blocked), and near the end of the simulation, vehicles that would
have arrived after ``sim_duration_sec`` are truncated (their row is never
recorded because the trip didn't finish within the sim window).

Both cases produce the SAME bias: fewer "slow" vehicles are recorded than
the flow rate would predict. Raw ``AVG(travel_time_sec)`` is therefore
biased **downward** at congested flows.

Neither category is represented in the DB (``trip_status='completed'`` for
all rows). The vehicles are simply missing — no row to flag.

==============================================================================
THE IMPUTATION RULE
==============================================================================

For each run and each minute ``m`` in the post-warmup recording window:

  expected[m] = flow_rate * FWD_RATIO / 60      # from LOCKED calibration
  actual[m]   = count of recorded FWD vehicles departing in minute m
  missing[m]  = max(0, expected[m] - actual[m])

Each missing vehicle gets an imputed ``travel_time_sec`` equal to the mean
TT of the actual vehicles departing in the SAME minute. If the minute has
no actual vehicles (e.g., very end of sim), the nearest non-empty minute
in the same run is used.

Path assignment for missing vehicles mirrors the actual split at that
minute (sampled deterministically by rounding). If the minute is empty,
the nearest minute's split is used.

==============================================================================
OUTPUT
==============================================================================

``impute_missing(conn, run_id)`` returns a list of virtual rows with
``travel_time_sec``, ``chosen_path``, ``actual_departure_sec`` (set to
m*60 + 30, i.e., midpoint of the minute). These rows can be concatenated
with real rows before computing summary statistics.

Apply the full-path filter (shared_lib.core.vehicle_filter) to the REAL
rows as usual — imputed rows already represent full-path trips by
construction.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List


FWD_RATIO = 0.78  # from LOCKED calibration (simulation_shared.py)

# Recording window is [warmup_sec, sim_duration_sec) expressed in minutes.
# For our runs: warmup_sec = 10_800 s = 180 min, sim_duration_sec = 18_000 s = 300 min.


@dataclass(frozen=True)
class ImputedVehicle:
    run_id: int
    departure_minute: int
    chosen_path: str        # 'short' or 'long'
    travel_time_sec: float  # imputed mean from peer minute


def impute_missing(
    conn: sqlite3.Connection,
    run_id: int,
    warmup_sec: int = 10_800,
    sim_duration_sec: int = 18_000,
) -> List[ImputedVehicle]:
    """Return one ImputedVehicle per expected-but-missing vehicle in run_id.

    Only FWD vehicles are considered (BWD is not route-choosing and not
    persisted in the DB).
    """
    row = conn.execute(
        "SELECT flow_rate FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        return []
    flow_rate = row[0]
    expected_per_min = flow_rate * FWD_RATIO / 60.0

    warmup_min = warmup_sec // 60
    end_min = sim_duration_sec // 60

    # Actual counts and TT/path stats per departure minute.
    cur = conn.execute(
        """
        SELECT CAST(actual_departure_sec / 60 AS INTEGER) AS dep_min,
               COUNT(*) AS n,
               AVG(travel_time_sec) AS tt_mean,
               SUM(CASE WHEN chosen_path='short' THEN 1 ELSE 0 END) AS n_short
        FROM vehicles
        WHERE run_id = ?
          AND is_warmup = 0
          AND direction = 'fwd'
          AND trip_status = 'completed'
          AND actual_departure_sec IS NOT NULL
        GROUP BY dep_min
        """,
        (run_id,),
    )
    stats_by_min: dict[int, tuple[int, float, int]] = {}
    for r in cur:
        stats_by_min[r[0]] = (r[1], r[2], r[3])

    imputed: list[ImputedVehicle] = []
    minutes_in_window = list(range(warmup_min, end_min))
    for m in minutes_in_window:
        actual = stats_by_min.get(m, (0, None, 0))[0]
        missing = max(0, int(round(expected_per_min - actual)))
        if missing == 0:
            continue

        # Find a peer minute with data (prefer same minute; else nearest).
        peer_m = m if actual > 0 else _nearest_minute_with_data(m, stats_by_min, warmup_min, end_min)
        if peer_m is None:
            continue  # Run has no real data at all — skip.
        n_peer, tt_peer, n_short_peer = stats_by_min[peer_m]
        pct_short = n_short_peer / n_peer if n_peer > 0 else 0.5

        # Assign paths deterministically by rounding.
        n_imputed_short = int(round(missing * pct_short))
        for i in range(missing):
            path = "short" if i < n_imputed_short else "long"
            imputed.append(
                ImputedVehicle(
                    run_id=run_id,
                    departure_minute=m,
                    chosen_path=path,
                    travel_time_sec=float(tt_peer),
                )
            )

    return imputed


def _nearest_minute_with_data(
    m: int,
    stats: dict[int, tuple[int, float, int]],
    lo: int,
    hi: int,
) -> int | None:
    """Return the nearest minute in [lo, hi) with n>0 data, or None."""
    for d in range(1, hi - lo):
        for cand in (m - d, m + d):
            if lo <= cand < hi and stats.get(cand, (0,))[0] > 0:
                return cand
    return None
