"""
Full-path vehicle filter — corrects for the ``departPos="free"`` artifact.

==============================================================================
SIMULATION LIMITATION (READ BEFORE USING travel_time_sec IN ANALYSIS)
==============================================================================

The project injects vehicles with ``departPos="free"`` (see
`simulation_shared.py`, line ~111). This is a **deliberate calibration
choice** — it allows SUMO to spawn vehicles at any free position along the
entry edge, which avoids an artificial entry bottleneck and produces a
realistic steady-state density distribution.

**Consequence:** under congestion, some fraction of vehicles (1–6% at flows
4000–6000 veh/h) are spawned mid-path rather than at position 0. These
vehicles have travel times far below the physical minimum for a full-path
trip, which biases raw ``AVG(travel_time_sec)`` **downward at congested
flows**.

The bias is:
  * 0% at free-flow (positions are almost always free at position 0)
  * up to ~5% at flow 4000 veh/h
  * ~1–2% at flows 5000–6000 (most vehicles can't even enter)

**Fix:** exclude any vehicle whose ``travel_time_sec`` is below the physical
minimum (``path_length_m / SPEED_LIMIT_MS``) — such a trip is impossible
unless the vehicle started mid-path. Legitimate fastest drivers cluster
right at the threshold (141–144 km/h on empty road), so no real data is
lost.

This filter **must be applied whenever averaging ``travel_time_sec``**. It
is NOT needed for ``mean_speed_kmh`` (SUMO records that over the actual
segment the vehicle traversed, so it is immune to the artifact).

Do not change ``departPos="free"`` in the runner — the project's calibration
(2026-02-19 Van Aerde benchmarks) was fitted with this setting, so changing
it would invalidate the locked parameters.
==============================================================================
"""

# Path lengths, from shared_lib/network/thesis_dual_route_18022026.
SHORT_PATH_LENGTH_M: float = 7500.0
LONG_PATH_LENGTH_M: float = 9375.0

# Network speed limit (from calibration_config.SPEED_LIMIT_MS).
SPEED_LIMIT_MS: float = 40.0  # 144 km/h

# Physical minimum travel time for a full-path trip at the speed limit.
# A travel_time_sec below this value means the vehicle spawned mid-path
# (departPos="free" artifact), so the row should be excluded from averages.
SHORT_PATH_MIN_TT_SEC: float = SHORT_PATH_LENGTH_M / SPEED_LIMIT_MS  # 187.5 s
LONG_PATH_MIN_TT_SEC: float = LONG_PATH_LENGTH_M / SPEED_LIMIT_MS    # 234.375 s


# SQL snippet for filtering full-path vehicles. Assumes the vehicles row
# has columns ``chosen_path``, ``travel_time_sec``, and ``actual_departure_sec``.
#
# CS pivot v2 (LOCKED 2026-05-26): the third OR clause lets queued-never-departed
# vehicles (actual_departure_sec IS NULL) pass the filter so they can reach
# the §1.4 counterfactual-CS / V_FLOOR-projected C_soc step downstream.
# Without this clause they would be silently dropped by the mid-spawn artefact
# filter, defeating the symmetric-inclusion design in welfare_measure_dossier.md
# §1.4. See methodology.md §0.2.
FULL_PATH_SQL_FILTER: str = (
    f"((chosen_path = 'short' AND travel_time_sec >= {SHORT_PATH_MIN_TT_SEC})"
    f" OR (chosen_path = 'long'  AND travel_time_sec >= {LONG_PATH_MIN_TT_SEC})"
    f" OR (actual_departure_sec IS NULL))"
)


# SQL snippet for filtering CS-eligible vehicles (methodology.md §0.5).
# Pass: (a) native logit-eligible agents (p_short_at_choice populated), OR
#       (b) queued-never-departed vehicles (counterfactual CS via §1.4 projection).
# Excludes: habit / fixed-split-blind agents in mixed populations (they made a
# non-logit choice; p_short_at_choice is NULL by design). Their TT data still
# feeds the secondary C_soc diagnostic via FULL_PATH_SQL_FILTER above.
# SO-baseline vehicles (fixed_split_vot) have NULL p_short_at_choice but receive
# a counterfactual logsum at the same θ via cs_backfill.py — they are handled
# upstream of this filter by joining against runs.route_choice_method.
CS_ELIGIBLE_SQL_FILTER: str = (
    "(p_short_at_choice IS NOT NULL"
    " OR actual_departure_sec IS NULL)"
)


def is_full_path_vehicle(
    chosen_path: str,
    travel_time_sec: float,
    actual_departure_sec: float | None = None,
) -> bool:
    """True if the vehicle should pass the departPos-artefact filter.

    Returns True for:
      - full-path completed/in-transit trips (travel_time_sec >= path minimum)
      - queued-never-departed vehicles (actual_departure_sec is None)

    Returns False for mid-path-spawned trips (departPos="free" artefact).

    Args:
        chosen_path: "short" or "long".
        travel_time_sec: The vehicle's recorded travel time in seconds.
        actual_departure_sec: The vehicle's actual departure timestamp; None
            for queued-never-departed vehicles. Default None preserves
            backwards compatibility with v1 callers that pass two args.

    Returns:
        True if the row should be kept; False if it should be dropped.
    """
    # Queued-never-departed: pass through (counterfactual handled downstream).
    if actual_departure_sec is None:
        return True
    if chosen_path == "short":
        return travel_time_sec >= SHORT_PATH_MIN_TT_SEC
    if chosen_path == "long":
        return travel_time_sec >= LONG_PATH_MIN_TT_SEC
    # Unknown path — be conservative and keep the row.
    return True
