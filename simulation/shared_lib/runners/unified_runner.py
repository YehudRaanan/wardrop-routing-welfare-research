"""
Unified Simulation Runner
=========================
ONE entry point for all phases. Parameterized by RunConfig.

Supports route choice methods: ema, davis, mediated, fixed_split, thompson, none.
Records ALL columns (superset schema) — phase-specific columns are None when N/A.
Writes results to SQLite DB via shared_lib.db.

Usage:
    from shared_lib.runners.run_config import RunConfig
    from shared_lib.runners.unified_runner import run_simulation

    config = RunConfig(flow_rate=2000, route_choice_method='ema', ema_alpha=0.1)
    run_id = run_simulation(config)
"""

import os
import sys

# libsumo MUST be imported before numpy (numpy's MKL DLLs shadow libsumo's _libsumo.pyd)
import libsumo as traci
import sumolib

import time
import random
import numpy as np
from collections import defaultdict

# Ensure shared_lib is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "shared_lib"))

import path_setup
import simulation_shared as shared
from simulation_shared import get_route_edges, NSTART_EDGES
from calibration_config import FWD_RATIO, BWD_RATIO
from core.agent_logic import (
    GlobalEMA, GlobalNStartPredictor, GlobalDavisThreshold,
    GlobalThompsonSampling, GlobalMediator, GlobalHabitFixedAtStart,
)
from core.agent_types import TypeBAgent, TypeCAgent
from shared_lib.validation import validate_run_config, validate_routes
from shared_lib.db.connection import get_fresh_connection, DB_PATH
from shared_lib.db.queries import (
    insert_run, update_run_status, insert_vehicles_batch,
    insert_timeseries_batch, insert_run_regression,
)
from shared_lib.db.dedup import find_existing_run

# Network config
CONFIG_FILE = path_setup.get_network_path("thesis_dual_route_18022026.sumocfg")

# Path lengths in meters (for probe TT estimation)
PATH_LENGTH = {"short": 7500.0, "long": 9375.0}

# Probe edges for speed sampling
# REAL-TIME SUPPLEMENT: Every 60s, live vehicle speeds on these edges are
# sampled via traci.vehicle.getSpeed() and converted to TT estimates
# (path_length / avg_speed). These estimates are fed into the EMA at half
# weight (alpha * 0.5). This gives EMA partial real-time information beyond
# completed-trip feedback — it is NOT a purely lagging indicator.
# Requires >= 3 vehicles on the edge to fire. Falls back to BWD edges if
# FWD has fewer than 3 vehicles.
PROBE_EDGES_FWD = {"short": "short_fwd", "long": "long_fwd"}
PROBE_EDGES_BWD = {"short": "short_bwd", "long": "long_bwd"}

BUCKET_SEC = 60  # 1-minute time-series buckets


# ─────────────────────────────────────────────────────────────────────
# N_START COUNTING
# ─────────────────────────────────────────────────────────────────────

def count_nstart() -> tuple[int, int]:
    """Count vehicles on each path's body edges (FWD + BWD).
    Returns (n_short, n_long)."""
    n_short = sum(traci.edge.getLastStepVehicleNumber(e)
                  for e in NSTART_EDGES["short"])
    n_long = sum(traci.edge.getLastStepVehicleNumber(e)
                 for e in NSTART_EDGES["long"])
    return n_short, n_long


# ─────────────────────────────────────────────────────────────────────
# VOT THRESHOLD HELPER (for route_choice_method='fixed_split_vot')
# ─────────────────────────────────────────────────────────────────────

def _compute_vot_threshold(p_short: float, seed: int = 42, n_samples: int = 100_000) -> float:
    """Sample the population VOT distribution and return the (1 - p_short) quantile.

    Reproduces TypeBAgent.__post_init__ VOT-assignment logic with an isolated
    random.Random instance (so the run's main RNG is not perturbed). Agents
    with vot >= returned threshold should be assigned to the short path under
    the 'fixed_split_vot' rule; the rest go long. The (1 - p_short) quantile
    corresponds to the top p_short fraction by VOT.
    """
    import math as _math
    from calibration_config import (
        LN_MU, LN_SIGMA, EXCHANGE_RATE_USD_GBP,
        VOT_COEFF_BUSINESS, VOT_COEFF_COMMUTE, VOT_COEFF_OTHER,
    )
    rng = random.Random(seed)
    samples = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        log_income = rng.gauss(LN_MU, LN_SIGMA)
        income_gbp = _math.exp(log_income)
        hourly_usd = (income_gbp * EXCHANGE_RATE_USD_GBP) / 2000.0
        rand_p = rng.random()
        if rand_p < 0.333:
            coeff = VOT_COEFF_COMMUTE
        elif rand_p < 0.666:
            coeff = VOT_COEFF_BUSINESS
        else:
            coeff = VOT_COEFF_OTHER
        samples[i] = max(0.1, hourly_usd * coeff)
    # Top p_short by VOT means VOTs above the (1 - p_short) quantile.
    return float(np.quantile(samples, 1.0 - p_short))


# ─────────────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────

def run_simulation(config, force: bool = False, db_path=None) -> int:
    """Execute a simulation or return existing run_id if already done.

    Args:
        config: RunConfig instance
        force: If True, skip dedup check
        db_path: Override DB path (for testing)

    Returns:
        run_id of the completed (or cached) run
    """
    from .run_config import RunConfig

    conn = get_fresh_connection(db_path or DB_PATH)

    # ── Dedup check ──
    if not force:
        existing = find_existing_run(
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
        )
        if existing is not None:
            print(f"  CACHED: run_id={existing} for "
                  f"{config.route_choice_method}@{config.flow_rate}")
            conn.close()
            return existing

    # ── Validate ──
    errors = validate_run_config(config)
    if errors:
        print("  VALIDATION WARNINGS:")
        for e in errors:
            print(f"    - {e}")

    # Validate routes
    routes = {
        f"{p}_{d}": get_route_edges(p, d)
        for p in ["short", "long"]
        for d in ["fwd", "bwd"]
    }
    route_errors = validate_routes(routes)
    if route_errors:
        raise ValueError(f"Route validation failed: {route_errors}")

    # ── Insert run row (status='running') ──
    db_params = config.to_db_params()
    db_params["status"] = "running"
    run_id = insert_run(conn, db_params)
    print(f"  RUN: run_id={run_id} | {config.route_choice_method}@{config.flow_rate} veh/h")

    wall_start = time.time()

    try:
        vehicle_records, timeseries_records, regression_data, stats, inflight_snapshot = \
            _execute_sumo_loop(config)

        # ── Write to DB ──
        n_veh = insert_vehicles_batch(conn, run_id, vehicle_records)
        n_ts = insert_timeseries_batch(conn, run_id, timeseries_records)

        # 2026-05-08: persist inflight snapshot at sim end for proper C_soc.
        if inflight_snapshot:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inflight_at_window_end (
                    run_id        INTEGER NOT NULL,
                    vid           TEXT NOT NULL,
                    state         TEXT NOT NULL,  -- 'queued' or 'in_transit'
                    chosen_path   TEXT NOT NULL,
                    speed_ms      REAL,
                    dist_traveled_m REAL,
                    PRIMARY KEY (run_id, vid)
                )
            """)
            conn.executemany(
                "INSERT OR REPLACE INTO inflight_at_window_end "
                "(run_id, vid, state, chosen_path, speed_ms, dist_traveled_m) VALUES (?,?,?,?,?,?)",
                [(run_id, r["vid"], r["state"], r["chosen_path"],
                  r.get("speed_ms"), r.get("dist_traveled_m"))
                 for r in inflight_snapshot]
            )
            conn.commit()

        if regression_data:
            insert_run_regression(conn, run_id, **regression_data)

        wall_sec = time.time() - wall_start
        update_run_status(
            conn, run_id, "completed",
            wall_clock_sec=round(wall_sec, 1),
            n_vehicles_fwd=stats.get("n_fwd", 0),
            n_vehicles_bwd=stats.get("n_bwd", 0),
            n_vehicles_blocked=stats.get("n_blocked", 0),
        )

        print(f"  DONE: {n_veh} vehicles, {n_ts} timeseries rows, "
              f"{wall_sec:.1f}s wall-clock")

        conn.close()
        return run_id

    except Exception as e:
        wall_sec = time.time() - wall_start
        update_run_status(conn, run_id, "failed",
                          error_message=str(e),
                          wall_clock_sec=round(wall_sec, 1))
        conn.close()
        raise


def _execute_sumo_loop(config):
    """Run the SUMO/TraCI simulation loop.

    Returns:
        (vehicle_records, timeseries_records, regression_data, stats_dict)
    """
    method = config.route_choice_method
    flow = config.flow_rate

    if config.random_seed is not None:
        random.seed(config.random_seed)

    # ── Initialize route choice objects ──
    global_ema = None
    davis = None
    global_mediator = None

    if method in ("ema", "mediated"):
        global_ema = GlobalEMA(alpha=config.ema_alpha or 0.1)

    _davis_buf = getattr(config, "davis_buffer_size", None)

    if method in ("davis",):
        davis = GlobalNStartPredictor(
            baseline_tt_short=270.0,
            baseline_tt_long=337.5,
            buffer_size=_davis_buf,
        )

    # 2026-05-14 (P4): Davis Sec. V threshold algorithm extended to 2 paths
    # with self-calibrating T_ref. Drop-in compatible interface with the
    # regression Davis (same predict_tt signature), so the rest of the
    # runner needs no changes — agent_logic.py just calls predict_tt() and
    # picks the lower of the two synthetic-TT returns.
    if method == "davis_threshold":
        davis = GlobalDavisThreshold(
            baseline_tt_short=270.0,
            baseline_tt_long=337.5,
            buffer_size=_davis_buf,
        )

    if method == "mediated":
        preload = _compute_mediator_preload(config)
        global_mediator = GlobalMediator(warmup_sec=config.warmup_sec, preload=preload)
        # Mediated method uses Davis as agent belief model for IC checks
        if davis is None:
            davis = GlobalNStartPredictor(baseline_tt_short=270.0, baseline_tt_long=337.5,
                                          buffer_size=_davis_buf)

    # For cross-method observation: also create observers
    if method == "ema" and davis is None:
        davis = GlobalNStartPredictor(baseline_tt_short=270.0, baseline_tt_long=337.5,
                                      buffer_size=_davis_buf)
    if method == "davis" and global_ema is None:
        global_ema = GlobalEMA(alpha=config.ema_alpha or 0.1)

    # ── 2026-05-09: method_mix support (P2). If method_mix is set, every method
    # in the mix needs a state object. Habitual-type vehicles don't need one
    # but we still need the VOT threshold (computed below near fixed_split_vot).
    if config.method_mix:
        if "ema" in config.method_mix and global_ema is None:
            global_ema = GlobalEMA(alpha=config.ema_alpha or 0.1)
        if "davis" in config.method_mix and davis is None:
            davis = GlobalNStartPredictor(baseline_tt_short=270.0, baseline_tt_long=337.5,
                                          buffer_size=_davis_buf)
        # 2026-05-14 (P4 migration): davis_threshold as a mix component.
        # Uses the SAME `davis` slot — only one of {davis, davis_threshold}
        # is in any given mix; both flow through the same routing branch in
        # agent_logic.py via the `method in ("davis","davis_threshold")` test.
        if "davis_threshold" in config.method_mix and davis is None:
            davis = GlobalDavisThreshold(
                baseline_tt_short=270.0,
                baseline_tt_long=337.5,
                buffer_size=_davis_buf,
            )

    # ── Start SUMO ──
    # --max-depart-delay 7200: SUMO retains pending vehicles in its insertion
    #   queue for up to 2h while waiting for entry-edge space. Replaces the
    #   pre-2026-05-06 silent-drop behavior (max-depart-delay default = -1
    #   gave INFINITE wait but we never polled, so vehicles silently piled up
    #   in SUMO's pending list and were lost at sim end). With our new
    #   pending_submission tracking + end-of-sim persistence, queued vehicles
    #   are recorded as trip_status='queued_at_sim_end'.
    sumo_bin = sumolib.checkBinary("sumo")
    cmd = [
        sumo_bin, "-c", CONFIG_FILE, "--start",
        "--step-length", str(config.step_length),
        "--end", str(config.sim_duration_sec),
        "--max-depart-delay", "7200",
        "--no-warnings", "true",
        "--no-step-log", "true",
    ]
    traci.start(cmd)

    vtype_car, vtype_truck = shared.setup_vehicle_types()

    # ── Define routes (body edges only — no entry edges) ──
    for path in ["short", "long"]:
        for direction in ["fwd", "bwd"]:
            route_id = f"route_{path}_{direction}"
            edges = get_route_edges(path, direction)
            traci.route.add(route_id, edges)

    # ── Injection rates ──
    fwd_flow = config.fwd_flow
    bwd_flow = config.bwd_flow
    veh_per_step_fwd = (fwd_flow / 3600.0) * config.step_length
    veh_per_step_bwd = (bwd_flow / 3600.0) * config.step_length
    acc_fwd = 0.0
    acc_bwd = 0.0
    veh_counter = 0

    # ── Tracking structures ──
    active_agents = {}       # vid -> TypeBAgent or TypeCAgent (FWD)
    active_bwd = {}          # vid -> {"depart_step": int, "path": str}
    fwd_depart_steps = {}    # vid -> depart_step
    fwd_nstart_at_choice = {}  # vid -> (n_short, n_long)

    # ── 2026-05-06 rebuild: pre-entrance FIFO queue tracking ──
    # SUMO holds pending vehicles internally (--max-depart-delay 7200), but
    # we maintain our own per-path counters and recent-departure-delay buffer
    # so we can (a) report queue stats, (b) feed composite TT predictions,
    # (c) persist still-queued vehicles at sim end.
    pending_per_path = {"short": 0, "long": 0}
    # recent buffer-delays (sim_time, delay_sec) tuples per path, last 60s used
    # to estimate the queue-wait an agent making a route choice would face.
    recent_departures = {"short": [], "long": []}
    QUEUE_WAIT_LOOKBACK_SEC = 60.0

    # Per-minute bins
    minute_tt_short = defaultdict(list)
    minute_tt_long = defaultdict(list)
    minute_choices = defaultdict(lambda: {"short": 0, "long": 0})
    minute_nstart = {}       # minute -> (n_short, n_long)
    minute_n_blocked = defaultdict(int)
    minute_compliance = defaultdict(lambda: {"followed": 0, "total": 0})  # mediator compliance
    # Departure-based TT: keyed by departure minute, filled on arrival
    minute_dep_tt_short = defaultdict(list)
    minute_dep_tt_long = defaultdict(list)

    # Vehicle records and EMA trace
    vehicle_records = []
    ema_trace = {}           # minute -> (ema_short, ema_long)

    n_fwd_injected = 0
    n_bwd_injected = 0
    n_blocked = 0

    steps_per_bucket = int(BUCKET_SEC / config.step_length)
    speed_sample_interval = 10  # sample every 10 steps = 1s (matching Phase 1)

    # ── Speed sampling (Phase 1 compatible) ──
    # Per-vehicle speed observations: vid -> [speed_kmh, ...]
    # Sampled every 1s on measurement edges, post-warmup only
    veh_speed_obs = defaultdict(list)
    # Per-minute speed bins: minute -> [speed_kmh, ...]
    minute_speed_obs = defaultdict(list)
    # Determine measurement edges based on target_path
    measurement_edges = set()
    if config.target_path == "short" or config.target_path is None:
        measurement_edges.add("short_fwd")
    if config.target_path == "long" or config.target_path is None:
        measurement_edges.add("long_fwd")

    # ── Fixed-split accumulator ──
    fixed_split_acc = 0.0

    # ── VOT-sorted fixed-split threshold (only when method='fixed_split_vot') ──
    # Top p_short fraction by VOT goes short, rest go long. Threshold is the
    # (1 - p_short) quantile of the population VOT distribution, sampled with
    # a fixed seed so the threshold is deterministic per p_short value.
    vot_threshold: float | None = None
    if method == "fixed_split_vot" or (config.method_mix and "fixed_split_vot" in config.method_mix):
        vot_threshold = _compute_vot_threshold(
            config.pct_short_fixed if config.pct_short_fixed is not None else 0.5,
            seed=42,
        )
        print(f"  fixed_split_vot: p_short={config.pct_short_fixed:.3f} -> VOT threshold = ${vot_threshold:.2f}/hr")

    # ── 2026-05-09: per-vehicle method dispatch RNG (P2 method_mix). ──
    # Use a stable RNG seeded from the run-level seed so the per-vehicle
    # method assignment is deterministic and reproducible.
    mix_rng = None
    if config.method_mix:
        mix_rng = random.Random((config.random_seed or 0) ^ 0xA17F1B57)
        mix_methods_list = list(config.method_mix.keys())
        mix_weights_list = list(config.method_mix.values())
        # normalize
        wsum = sum(mix_weights_list)
        if wsum > 0:
            mix_weights_list = [w / wsum for w in mix_weights_list]
        print(f"  method_mix dispatch active: {dict(zip(mix_methods_list, [round(w,3) for w in mix_weights_list]))}")

    # ═══════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════
    for step in range(config.sim_steps):
        traci.simulationStep()
        sim_time = step * config.step_length

        # ── 2026-05-06 rebuild: FIFO-queue departure detection ──
        # Vehicles that just left the holding queue (entered the road this step).
        # SUMO emits each freshly-departed vid in getDepartedIDList(); we use it
        # to (a) record actual_dep on the agent, (b) decrement the per-path
        # pending counter, (c) push the vehicle's buffer_delay into the
        # rolling recent_departures buffer used by composite-TT prediction,
        # (d) capture traci.edge.getTraveltime at the actual departure moment.
        for dep_vid in traci.simulation.getDepartedIDList():
            if dep_vid not in active_agents:
                continue
            agent = active_agents[dep_vid]
            try:
                actual_dep = traci.vehicle.getDeparture(dep_vid)
                if actual_dep < 0:
                    actual_dep = sim_time
            except traci.TraCIException:
                actual_dep = sim_time
            planned_dep = agent.departure_time  # set at submit time
            buffer_delay = max(0.0, actual_dep - planned_dep)
            agent.actual_departure_time = actual_dep
            agent.buffer_delay_sec_observed = buffer_delay
            # Capture TraCI ground-truth TTs at actual departure
            try:
                agent.traci_tt_short_at_departure = traci.edge.getTraveltime("short_fwd")
                agent.traci_tt_long_at_departure  = traci.edge.getTraveltime("long_fwd")
            except traci.TraCIException:
                agent.traci_tt_short_at_departure = None
                agent.traci_tt_long_at_departure = None
            # Update FIFO accounting
            chosen = getattr(agent, "chosen_path", None)
            if chosen in pending_per_path and pending_per_path[chosen] > 0:
                pending_per_path[chosen] -= 1
            if chosen in recent_departures:
                recent_departures[chosen].append((sim_time, buffer_delay))
                # Trim to the lookback window
                cutoff = sim_time - QUEUE_WAIT_LOOKBACK_SEC
                while recent_departures[chosen] and recent_departures[chosen][0][0] < cutoff:
                    recent_departures[chosen].pop(0)

        # ── FWD injection ──
        acc_fwd += veh_per_step_fwd
        while acc_fwd >= 1.0:
            acc_fwd -= 1.0
            veh_counter += 1
            vid = f"fwd_{flow}_{veh_counter}"

            success, props = shared.inject_vehicle_deferred(
                vid, vtype_car, vtype_truck
            )
            if not success:
                continue

            # ── Create agent BEFORE route choice (need VOT for toll) ──
            if method == "mediated":
                agent = TypeCAgent(
                    vid=vid,
                    group_id=props["group_id"],
                    assigned_kmh=props["assigned_kmh"],
                    assigned_tau=props["assigned_tau"],
                )
            else:
                agent = TypeBAgent(
                    vid=vid,
                    group_id=props["group_id"],
                    assigned_kmh=props["assigned_kmh"],
                    assigned_tau=props["assigned_tau"],
                )

            # ── Compute agent-specific toll penalty ──
            toll_penalty_sec = 0.0
            if config.toll_short_usd > 0:
                toll_penalty_sec = config.toll_short_usd * 3600.0 / max(0.1, agent.vot)

            # ── Snapshot EMA/Davis state before suppression check ──
            # Needed both for elastic demand decision and for logging
            # perceived vs actual cost (phantom suppression detection).
            n_short, n_long = count_nstart()
            ema_s_snap, ema_l_snap = (
                global_ema.get_values() if global_ema else (None, None)
            )
            davis_s_snap, davis_l_snap = (
                davis.predict_tt(n_short, n_long) if davis else (None, None)
            )

            # Initialize reservation_gc_sec to be stored for all vehicle outcomes
            reservation_gc_sec = None

            # ── Elastic demand: per-agent trip suppression ──
            # Model: Correa & Stier-Moses reservation price framework
            # + Wardman (2016) heterogeneous VOT + Verhoef (2006) ε≈-0.4
            # Agent suppresses trip if current GC > multiplier × free-flow GC
            # demand_elasticity field stores the reservation multiplier (e.g., 3.5)
            if config.demand_elasticity is not None:
                reservation_mult = config.demand_elasticity
                # Current GC estimate: EMA travel time + toll penalty
                if ema_s_snap is not None:
                    gc_current = min(ema_s_snap, ema_l_snap) + toll_penalty_sec
                else:
                    gc_current = 308.0 + toll_penalty_sec  # weighted FF fallback
                # Free-flow GC for this agent
                gc_freeflow = 270.0 + toll_penalty_sec
                reservation_gc_sec = reservation_mult * gc_freeflow
                if gc_current > reservation_gc_sec:
                    # Trip suppressed — record agent profile, then skip
                    vehicle_records.append({
                        "vid": vid,
                        "direction": "fwd",
                        "vehicle_type": props.get("type", "car"),
                        "group_id": agent.group_id,
                        "assigned_kmh": round(agent.assigned_kmh, 2),
                        "speed_factor": round(props.get("speed_factor", 1.0), 4),
                        "agent_type": "C" if method == "mediated" else "B",
                        "chosen_path": "short",  # placeholder (never traveled)
                        "choice_type": "suppressed",
                        "planned_departure_sec": round(sim_time, 2),
                        "actual_departure_sec": None,
                        "arrival_sec": 0,
                        "travel_time_sec": 0,
                        "buffer_delay_sec": 0,
                        "is_warmup": 1 if sim_time < config.warmup_sec else 0,
                        "n_start_short": n_short, "n_start_long": n_long,
                        "ema_short_at_choice": round(ema_s_snap, 2) if ema_s_snap is not None else None,
                        "ema_long_at_choice": round(ema_l_snap, 2) if ema_l_snap is not None else None,
                        "davis_pred_short": round(davis_s_snap, 2) if davis_s_snap is not None else None,
                        "davis_pred_long": round(davis_l_snap, 2) if davis_l_snap is not None else None,
                        "mediator_signal": None, "signal_followed": None,
                        "compliance_code": None, "mediator_mode": None,
                        "ic_margin_at_choice": None, "ic_constrained": None,
                        "vot_usd_hr": round(agent.vot, 2),
                        "income_gbp": round(agent.income_gbp, 2) if hasattr(agent, "income_gbp") else None,
                        "trip_purpose": getattr(agent, "trip_purpose", None),
                        "toll_paid_usd": 0.0,
                        "generalized_cost_sec": round(gc_current, 2), # This is the perceived GC at choice time
                        "reservation_price_usd": round(reservation_gc_sec, 2),
                        "mean_speed_kmh": None, "n_speed_samples": 0,
                        "trip_status": "suppressed",
                    })
                    minute_num = int(sim_time / 60)
                    minute_n_blocked[minute_num] += 1
                    n_blocked += 1
                    continue
                else:
                    # Trip is NOT suppressed. Store reservation price on the agent to be saved on arrival.
                    agent.reservation_price_usd = reservation_gc_sec

            # ── Composite-TT inputs (2026-05-06 rebuild) ──
            # Per-path queue-wait estimate from recent observed buffer_delays.
            # This is added to the on-network TT predicted by EMA / Davis so
            # the agent's choice reflects total expected door-to-door cost
            # (queue wait at entry + on-network travel time).
            qw_short = _estimate_queue_wait(
                "short", recent_departures, sim_time, QUEUE_WAIT_LOOKBACK_SEC,
            )
            qw_long = _estimate_queue_wait(
                "long", recent_departures, sim_time, QUEUE_WAIT_LOOKBACK_SEC,
            )

            # ── Per-vehicle method assignment (2026-05-09: P2 method_mix) ──
            if mix_rng is not None:
                effective_method = mix_rng.choices(mix_methods_list, weights=mix_weights_list, k=1)[0]
            else:
                effective_method = method
            agent.routing_method = effective_method  # stored for choice_type bookkeeping

            # ── Route choice ──
            choice_details = None
            if effective_method == "fixed_split_vot":
                chosen_path = "short" if agent.vot >= vot_threshold else "long"
                mediator_details = None
            else:
                # Build a per-vehicle config view that overrides route_choice_method
                # so _choose_route dispatches on the agent's effective method.
                if effective_method != method:
                    class _ConfigView:
                        __slots__ = ("_orig", "route_choice_method")
                        def __init__(self, orig, method):
                            self._orig = orig
                            self.route_choice_method = method
                        def __getattr__(self, k):
                            return getattr(self._orig, k)
                    cfg_for_choice = _ConfigView(config, effective_method)
                else:
                    cfg_for_choice = config
                chosen_path, choice_details = _choose_route(
                    cfg_for_choice, props, global_ema, davis, global_mediator,
                    n_short, n_long, veh_counter, sim_time,
                    toll_penalty_sec=toll_penalty_sec,
                    queue_wait_short=qw_short, queue_wait_long=qw_long,
                )

            route_id = f"route_{chosen_path}_fwd"

            try:
                traci.vehicle.add(
                    vid, route_id, typeID=props["type"],
                    depart="now", departSpeed="max", departPos="free",
                )
                traci.vehicle.setTau(vid, props["assigned_tau"])
                traci.vehicle.setImperfection(vid, props["assigned_sigma"])
                traci.vehicle.setSpeedFactor(vid, props["speed_factor"])
                traci.vehicle.setColor(vid, props["color"])

                # Store mediator or choice details
                if method == "mediated" and choice_details:
                    agent.mediator_signal = choice_details.get("signal")
                    agent.signal_followed = choice_details.get("signal_followed", True)
                    agent.compliance_code = choice_details.get("compliance_code", "followed")
                    agent.mediator_mode = choice_details.get("mode")
                    agent.pred_short = choice_details.get("pred_short")
                    agent.pred_long = choice_details.get("pred_long")
                    agent.gap_physical = choice_details.get("gap_physical")
                    agent.ic_margin_at_choice = choice_details.get("ic_margin")
                    agent.ic_constrained = choice_details.get("ic_constrained", False)
                elif choice_details:
                    agent.p_short_at_choice = choice_details.get("p_short")
                    agent.cost_short = choice_details.get("cost_short")
                    agent.cost_long = choice_details.get("cost_long")

                agent.departure_time = sim_time
                agent.chosen_path = chosen_path

                # Store EMA/Davis state at choice time (reuse pre-suppression snapshot)
                if ema_s_snap is not None:
                    agent.ema_short_at_choice = ema_s_snap
                    agent.ema_long_at_choice = ema_l_snap
                if davis_s_snap is not None:
                    agent._davis_pred_short = davis_s_snap
                    agent._davis_pred_long = davis_l_snap

                active_agents[vid] = agent
                fwd_depart_steps[vid] = step
                fwd_nstart_at_choice[vid] = (n_short, n_long)

                # 2026-05-06 rebuild: register in pre-entrance FIFO counter.
                # Vehicle is now in SUMO's pending list; departure-detection
                # block at top of next step decrements when it actually leaves.
                if chosen_path in pending_per_path:
                    pending_per_path[chosen_path] += 1

                # Capture TraCI ground-truth TTs at choice time (independent
                # reference for analysis: prediction quality + queue impact).
                try:
                    agent.traci_tt_short_at_choice = traci.edge.getTraveltime("short_fwd")
                    agent.traci_tt_long_at_choice  = traci.edge.getTraveltime("long_fwd")
                except traci.TraCIException:
                    agent.traci_tt_short_at_choice = None
                    agent.traci_tt_long_at_choice = None

                minute_num = int(sim_time / 60)
                minute_choices[minute_num][chosen_path] += 1
                n_fwd_injected += 1

                # Track mediator compliance
                if method == "mediated" and hasattr(agent, "signal_followed"):
                    minute_compliance[minute_num]["total"] += 1
                    if agent.signal_followed:
                        minute_compliance[minute_num]["followed"] += 1

            except traci.TraCIException:
                # Record blocked vehicle profile
                vehicle_records.append({
                    "vid": vid,
                    "direction": "fwd",
                    "vehicle_type": props.get("type", "car"),
                    "group_id": agent.group_id,
                    "assigned_kmh": round(agent.assigned_kmh, 2),
                    "speed_factor": round(props.get("speed_factor", 1.0), 4),
                    "agent_type": "C" if method == "mediated" else "B",
                    "chosen_path": chosen_path,
                    "choice_type": "blocked",
                    "planned_departure_sec": round(sim_time, 2),
                    "actual_departure_sec": None,
                    "arrival_sec": 0,
                    "travel_time_sec": 0,
                    "buffer_delay_sec": 0,
                    "is_warmup": 1 if sim_time < config.warmup_sec else 0,
                    "n_start_short": n_short, "n_start_long": n_long,
                    "ema_short_at_choice": None, "ema_long_at_choice": None,
                    "davis_pred_short": None, "davis_pred_long": None,
                    "mediator_signal": None, "signal_followed": None,
                    "compliance_code": None, "mediator_mode": None,
                    "ic_margin_at_choice": None, "ic_constrained": None,
                    "vot_usd_hr": round(agent.vot, 2) if hasattr(agent, "vot") else None,
                    "income_gbp": round(agent.income_gbp, 2) if hasattr(agent, "income_gbp") else None,
                    "trip_purpose": getattr(agent, "trip_purpose", None),
                    "toll_paid_usd": 0.0,
                    "reservation_price_usd": round(reservation_gc_sec, 2) if reservation_gc_sec is not None else None,
                    "generalized_cost_sec": None,
                    "mean_speed_kmh": None, "n_speed_samples": 0,
                    "trip_status": "blocked",
                })
                n_blocked += 1
                minute_num = int(sim_time / 60)
                minute_n_blocked[minute_num] += 1

        # ── BWD injection (50/50 split, no route choice) ──
        acc_bwd += veh_per_step_bwd
        while acc_bwd >= 1.0:
            acc_bwd -= 1.0
            veh_counter += 1
            vid = f"bwd_{flow}_{veh_counter}"

            bwd_path = "short" if veh_counter % 2 == 0 else "long"
            route_id = f"route_{bwd_path}_bwd"

            success, props = shared.inject_vehicle(
                vid, route_id, vtype_car, vtype_truck
            )
            if success:
                active_bwd[vid] = {"depart_step": step, "path": bwd_path}
                n_bwd_injected += 1

        # ── Process arrivals ──
        for vid in traci.simulation.getArrivedIDList():
            if vid in active_agents:
                agent = active_agents.pop(vid)
                depart_step = fwd_depart_steps.pop(vid)
                nstart_at_choice = fwd_nstart_at_choice.pop(vid, (None, None))
                path = agent.chosen_path

                # 2026-05-06 rebuild: actual_dep is captured at the moment
                # the vehicle actually leaves the holding queue (departure
                # detection block at top of main loop). buffer_delay is
                # computed there and stored on the agent. Fall back to
                # querying TraCI if for some reason the agent didn't pass
                # through the departure-detection path (e.g., vehicle
                # departed and arrived in the same step).
                planned_dep = agent.departure_time
                actual_dep = getattr(agent, "actual_departure_time", None)
                if actual_dep is None:
                    try:
                        actual_dep = traci.vehicle.getDeparture(vid)
                        if actual_dep < 0:
                            actual_dep = planned_dep
                    except traci.TraCIException:
                        actual_dep = planned_dep

                buffer_delay = getattr(agent, "buffer_delay_sec_observed",
                                       max(0.0, actual_dep - planned_dep))
                arrival_sec = sim_time
                tt_sec = arrival_sec - actual_dep  # real on-road travel time

                # 2026-05-10: feed every initialised tracker on trip completion,
                # regardless of whether the run is single-method, mediated, or mixed.
                # Pre-fix this block was gated on method=="ema"/"davis"/"mediated",
                # so mixed-mode runs never received the terminal-TT signal —
                # methods were left relying solely on the per-minute in-flight
                # stream, which under-states completion-time congestion.
                if global_ema and nstart_at_choice[0] is not None:
                    global_ema.update(path, tt_sec)
                if davis and nstart_at_choice[0] is not None:
                    davis.record_trip(
                        nstart_at_choice[0], nstart_at_choice[1],
                        path, tt_sec,
                    )
                if method == "mediated" and global_mediator:
                    ema_at_choice = (
                        agent.ema_short_at_choice if path == "short"
                        else agent.ema_long_at_choice
                    ) if hasattr(agent, "ema_short_at_choice") else None
                    n_at = nstart_at_choice[0] if path == "short" else nstart_at_choice[1]
                    global_mediator.record_trip(path, n_at, tt_sec, ema_at_choice)

                # Record arrival (arrival-based grouping)
                arrival_min = int(sim_time / 60)
                if path == "short":
                    minute_tt_short[arrival_min].append(tt_sec)
                else:
                    minute_tt_long[arrival_min].append(tt_sec)

                # Record departure-based TT (Wardrop-correct grouping)
                dep_min = int(actual_dep / 60)
                if path == "short":
                    minute_dep_tt_short[dep_min].append(tt_sec)
                else:
                    minute_dep_tt_long[dep_min].append(tt_sec)

                # Warmup flag based on ACTUAL departure time (when
                # vehicle entered road, not when add() was called).
                is_warmup = 1 if actual_dep < config.warmup_sec else 0

                vehicle_records.append({
                    "vid": vid,
                    "direction": "fwd",
                    "vehicle_type": props.get("type", "car"),
                    "group_id": agent.group_id,
                    "assigned_kmh": round(agent.assigned_kmh, 2),
                    "speed_factor": round(props.get("speed_factor", 1.0), 4),
                    "agent_type": "C" if method == "mediated" else "B",
                    "chosen_path": path,
                    "choice_type": getattr(agent, "choice_type", None) or _get_choice_type(getattr(agent, "routing_method", method)),
                    "planned_departure_sec": round(planned_dep, 2),
                    "actual_departure_sec": round(actual_dep, 2),
                    "arrival_sec": round(arrival_sec, 2),
                    "travel_time_sec": round(tt_sec, 2),
                    "buffer_delay_sec": round(buffer_delay, 2),
                    "is_warmup": is_warmup,
                    "n_start_short": nstart_at_choice[0],
                    "n_start_long": nstart_at_choice[1],
                    "ema_short_at_choice": (
                        round(agent.ema_short_at_choice, 2)
                        if hasattr(agent, "ema_short_at_choice")
                           and agent.ema_short_at_choice else None
                    ),
                    "ema_long_at_choice": (
                        round(agent.ema_long_at_choice, 2)
                        if hasattr(agent, "ema_long_at_choice")
                           and agent.ema_long_at_choice else None
                    ),
                    "davis_pred_short": (
                        round(agent._davis_pred_short, 2)
                        if hasattr(agent, "_davis_pred_short") else None
                    ),
                    "davis_pred_long": (
                        round(agent._davis_pred_long, 2)
                        if hasattr(agent, "_davis_pred_long") else None
                    ),
                    # 2026-05-06 rebuild: TraCI ground-truth TTs (independent
                    # reference for prediction-quality + queue-impact analysis)
                    "traci_tt_short_at_choice": (
                        round(agent.traci_tt_short_at_choice, 2)
                        if getattr(agent, "traci_tt_short_at_choice", None) else None
                    ),
                    "traci_tt_long_at_choice": (
                        round(agent.traci_tt_long_at_choice, 2)
                        if getattr(agent, "traci_tt_long_at_choice", None) else None
                    ),
                    "traci_tt_short_at_departure": (
                        round(agent.traci_tt_short_at_departure, 2)
                        if getattr(agent, "traci_tt_short_at_departure", None) else None
                    ),
                    "traci_tt_long_at_departure": (
                        round(agent.traci_tt_long_at_departure, 2)
                        if getattr(agent, "traci_tt_long_at_departure", None) else None
                    ),
                    # Mediator details (Phase 4)
                    "mediator_signal": getattr(agent, "mediator_signal", None),
                    "signal_followed": 1 if getattr(agent, "signal_followed", None) else (0 if hasattr(agent, "signal_followed") and agent.signal_followed is not None else None),
                    "compliance_code": getattr(agent, "compliance_code", None),
                    "mediator_mode": getattr(agent, "mediator_mode", None),
                    "ic_margin_at_choice": round(agent.ic_margin_at_choice, 2) if getattr(agent, "ic_margin_at_choice", None) else None,
                    "ic_constrained": 1 if getattr(agent, "ic_constrained", None) else (0 if hasattr(agent, "ic_constrained") and agent.ic_constrained is not None else None),
                    "vot_usd_hr": round(agent.vot, 2) if hasattr(agent, "vot") else None,
                    "income_gbp": round(agent.income_gbp, 2) if hasattr(agent, "income_gbp") else None,
                    "trip_purpose": getattr(agent, "trip_purpose", None),
                    "toll_paid_usd": config.toll_short_usd if path == "short" else 0.0,
                    "generalized_cost_sec": (
                        round(tt_sec + (config.toll_short_usd * 3600.0 / max(0.1, agent.vot)), 2)
                        if path == "short" and config.toll_short_usd > 0 and hasattr(agent, "vot")
                        else round(tt_sec, 2) if tt_sec else None
                    ),
                    "reservation_price_usd": (
                        round(agent.reservation_price_usd, 2)
                        if hasattr(agent, "reservation_price_usd") and agent.reservation_price_usd is not None
                        else None
                    ),
                    "p_short_at_choice": (
                        round(agent.p_short_at_choice, 4)
                        if hasattr(agent, "p_short_at_choice")
                           and agent.p_short_at_choice is not None else None
                    ),
                    "cost_short_at_choice": (
                        round(agent.cost_short, 2)
                        if hasattr(agent, "cost_short")
                           and agent.cost_short is not None else None
                    ),
                    "cost_long_at_choice": (
                        round(agent.cost_long, 2)
                        if hasattr(agent, "cost_long")
                           and agent.cost_long is not None else None
                    ),
                    "mean_speed_kmh": (
                        round(float(np.mean(veh_speed_obs[vid])), 2)
                        if vid in veh_speed_obs and veh_speed_obs[vid] else None
                    ),
                    "n_speed_samples": (
                        len(veh_speed_obs[vid])
                        if vid in veh_speed_obs else 0
                    ),
                    "trip_status": "completed",
                })

            elif vid in active_bwd:
                active_bwd.pop(vid)

        # ── In-transit per-minute sampling (replaces 2026-pre-rebuild probe) ──
        # Both EMA and Davis consume the SAME observation stream per the
        # 2026-05-06 rebuild methodology. Once a minute, for each currently-
        # driving vehicle on a forward path, we synthesise an observation:
        #
        #   n_start  = current N_start on the path (same for all vehicles
        #              sampled at this tick)
        #   tt_est   = remaining_distance / current_speed  (per-vehicle
        #              projection; treats vehicle as if it'll continue at
        #              current speed for the rest of the trip)
        #
        # These tuples are pushed to BOTH global_ema (which exponentially
        # smooths the TT) and davis (which appends to its 500-tuple rolling
        # buffer for regression). At low flow few in-transit vehicles → few
        # tuples; at high flow many → many. Symmetric data, asymmetric
        # formula = the methodological cleanly-comparable design.
        if (step > 0 and step % steps_per_bucket == 0 and (global_ema or davis)
                and not getattr(config, "disable_intransit_feed", False)):
            n_short_now, n_long_now = count_nstart()
            for path_name, fwd_edge in PROBE_EDGES_FWD.items():
                n_now = n_short_now if path_name == "short" else n_long_now
                path_length = PATH_LENGTH[path_name]
                for sample_vid in traci.edge.getLastStepVehicleIDs(fwd_edge):
                    spd = traci.vehicle.getSpeed(sample_vid)
                    if spd <= 1.0:
                        continue
                    # Remaining-distance variant (option (b) per locked spec)
                    try:
                        lane_pos = traci.vehicle.getLanePosition(sample_vid)
                    except traci.TraCIException:
                        lane_pos = 0.0
                    remaining_dist = max(0.0, path_length - lane_pos)
                    if remaining_dist < 1.0:
                        continue
                    est_tt = remaining_dist / spd
                    if global_ema:
                        global_ema.update(path_name, est_tt)
                    if davis:
                        davis.record_observation(path_name, n_now, est_tt)

        # ── Speed sampling (every 1s = 10 steps, post-warmup) ──
        # Matches Phase 1 methodology: cross-sectional instantaneous speeds
        # on measurement edges, per-vehicle mean computed at arrival
        if step >= config.warmup_steps and step % speed_sample_interval == 0:
            minute_num = int(sim_time / 60)
            for edge in measurement_edges:
                try:
                    for svid in traci.edge.getLastStepVehicleIDs(edge):
                        if svid.startswith("fwd_"):
                            spd_kmh = traci.vehicle.getSpeed(svid) * 3.6
                            if spd_kmh > 1.0:
                                veh_speed_obs[svid].append(spd_kmh)
                                minute_speed_obs[minute_num].append(spd_kmh)
                except traci.TraCIException:
                    pass

        # ── Per-minute snapshots ──
        if step > 0 and step % steps_per_bucket == 0:
            minute_num = int(sim_time / 60)
            n_s, n_l = count_nstart()
            minute_nstart[minute_num] = (n_s, n_l)
            if global_ema:
                ema_s, ema_l = global_ema.get_values()
                ema_trace[minute_num] = (round(ema_s, 2), round(ema_l, 2))
            # Mediator refit every 60s
            if method == "mediated" and global_mediator:
                n_total = n_s + n_l
                global_mediator.refit(sim_time, n_total)

    # ═══════════════════════════════════════════════════════════════
    # 2026-05-06 rebuild: END-OF-SIM PERSISTENCE for queued / in-transit
    # ═══════════════════════════════════════════════════════════════
    # Any vid still in active_agents at sim end either (a) is still in SUMO's
    # holding queue (never departed) OR (b) departed but didn't arrive
    # before sim end. Both must be persisted so n_vehicles is comparable
    # across methods at the same flow rate.
    # NOTE on trip_status: the existing CHECK constraint (added in v4)
    # restricts trip_status to {'completed','suppressed','blocked'} and
    # SQLite cannot extend a CHECK constraint via ALTER. Recreating the
    # 38 GB vehicles table is forbidden by the DB protection rules. So
    # we map both end-of-sim end-states to 'blocked'; the queued vs
    # in_transit distinction is recovered in analysis via:
    #   - actual_departure_sec IS NULL  ⇒ queued (never departed)
    #   - actual_departure_sec IS NOT NULL AND travel_time_sec = 0
    #                                    ⇒ in_transit (departed but didn't arrive)
    sim_end_sec = config.sim_duration_sec
    # 2026-05-08: Inflight snapshot at sim end for proper C_soc per dossier §1.1.
    # For vehicles still in transit (departed in window, didn't arrive), we
    # capture (remaining_distance_m, current_speed_mps, chosen_path) so the
    # welfare query can compute realized + projected-remaining contribution.
    inflight_snapshot: list[dict] = []  # populated below
    for vid, agent in list(active_agents.items()):
        planned_dep = agent.departure_time
        actual_dep = getattr(agent, "actual_departure_time", None)
        path = getattr(agent, "chosen_path", "short")
        status = "blocked"
        if actual_dep is None:
            # Never departed — still in pre-entrance FIFO at sim end
            buffer_delay = max(0.0, sim_end_sec - planned_dep)
            is_warmup = 1 if planned_dep < config.warmup_sec else 0
            # Full path remaining (vehicle never even hit the road)
            inflight_snapshot.append({"vid": vid, "remaining_m": None, "speed_ms": 0.0,
                                      "chosen_path": path, "state": "queued"})
        else:
            # Departed but didn't arrive — query traci for position
            buffer_delay = getattr(agent, "buffer_delay_sec_observed",
                                   max(0.0, actual_dep - planned_dep))
            is_warmup = 1 if actual_dep < config.warmup_sec else 0
            try:
                lane_pos = float(traci.vehicle.getLanePosition(vid))
                speed_ms = float(traci.vehicle.getSpeed(vid))
                # remaining = edge_len - lane_pos. We don't know edge length here
                # without a query; getRoadID + lane.getLength is the path. Use
                # traci.vehicle.getDistance for total distance traveled instead,
                # which works regardless of multi-edge routes.
                dist_traveled = float(traci.vehicle.getDistance(vid))
            except (traci.TraCIException, Exception):
                lane_pos = 0.0
                speed_ms = 0.0
                dist_traveled = 0.0
            inflight_snapshot.append({"vid": vid, "remaining_m": None, "speed_ms": speed_ms,
                                      "chosen_path": path, "state": "in_transit",
                                      "dist_traveled_m": dist_traveled})

        vehicle_records.append({
            "vid": vid,
            "direction": "fwd",
            "vehicle_type": "car",
            "group_id": getattr(agent, "group_id", "unknown"),
            "assigned_kmh": round(getattr(agent, "assigned_kmh", 0.0), 2),
            "speed_factor": 1.0,
            "agent_type": "C" if method == "mediated" else "B",
            "chosen_path": path,
            "choice_type": getattr(agent, "choice_type", None) or _get_choice_type(getattr(agent, "routing_method", method)),
            "planned_departure_sec": round(planned_dep, 2),
            "actual_departure_sec": round(actual_dep, 2) if actual_dep is not None else None,
            "arrival_sec": 0,           # NOT NULL — never arrived
            "travel_time_sec": 0,       # NOT NULL — never completed
            "buffer_delay_sec": round(buffer_delay, 2),
            "is_warmup": is_warmup,
            "n_start_short": None,
            "n_start_long": None,
            "ema_short_at_choice": getattr(agent, "ema_short_at_choice", None),
            "ema_long_at_choice": getattr(agent, "ema_long_at_choice", None),
            "davis_pred_short": getattr(agent, "_davis_pred_short", None),
            "davis_pred_long": getattr(agent, "_davis_pred_long", None),
            "traci_tt_short_at_choice": getattr(agent, "traci_tt_short_at_choice", None),
            "traci_tt_long_at_choice": getattr(agent, "traci_tt_long_at_choice", None),
            "traci_tt_short_at_departure": getattr(agent, "traci_tt_short_at_departure", None),
            "traci_tt_long_at_departure": getattr(agent, "traci_tt_long_at_departure", None),
            "vot_usd_hr": round(getattr(agent, "vot", 0.0), 2) if hasattr(agent, "vot") else None,
            "income_gbp": round(agent.income_gbp, 2) if hasattr(agent, "income_gbp") else None,
            "trip_purpose": getattr(agent, "trip_purpose", None),
            "toll_paid_usd": 0.0,           # never traveled, no toll paid
            "generalized_cost_sec": None,
            "p_short_at_choice": (
                round(agent.p_short_at_choice, 4)
                if hasattr(agent, "p_short_at_choice")
                   and agent.p_short_at_choice is not None else None
            ),
            "cost_short_at_choice": (
                round(agent.cost_short, 2)
                if hasattr(agent, "cost_short")
                   and agent.cost_short is not None else None
            ),
            "cost_long_at_choice": (
                round(agent.cost_long, 2)
                if hasattr(agent, "cost_long")
                   and agent.cost_long is not None else None
            ),
            "trip_status": status,
        })
    # Now safe to clear active_agents (we've recorded everything outstanding)
    active_agents.clear()

    traci.close()

    # ═══════════════════════════════════════════════════════════════
    # BUILD TIMESERIES
    # ═══════════════════════════════════════════════════════════════
    all_minutes = sorted(set(
        list(minute_tt_short.keys()) +
        list(minute_tt_long.keys()) +
        list(minute_nstart.keys())
    ))

    timeseries_records = []
    for m in all_minutes:
        tt_s = minute_tt_short.get(m, [])
        tt_l = minute_tt_long.get(m, [])
        choices = minute_choices.get(m, {"short": 0, "long": 0})
        n_arr_s = len(tt_s)
        n_arr_l = len(tt_l)
        n_total = n_arr_s + n_arr_l

        ema_vals = ema_trace.get(m, (None, None))
        nstart_vals = minute_nstart.get(m, (None, None))

        # Davis regression snapshot
        reg_data = {}
        if davis and hasattr(davis, "slope_short"):
            reg_data = {
                "reg_intercept_short": round(davis.intercept_short, 4) if davis.intercept_short else None,
                "reg_slope_short": round(davis.slope_short, 6) if davis.slope_short else None,
                "reg_intercept_long": round(davis.intercept_long, 4) if davis.intercept_long else None,
                "reg_slope_long": round(davis.slope_long, 6) if davis.slope_long else None,
            }

        timeseries_records.append({
            "minute": m,
            "is_warmup": 1 if m < (config.warmup_sec / 60) else 0,
            "avg_tt_short": round(float(np.mean(tt_s)), 2) if tt_s else None,
            "avg_tt_long": round(float(np.mean(tt_l)), 2) if tt_l else None,
            "total_tt_sec": round(sum(tt_s) + sum(tt_l), 2),
            "n_arrived_short": n_arr_s,
            "n_arrived_long": n_arr_l,
            "n_chose_short": choices["short"],
            "n_chose_long": choices["long"],
            "pct_short": round(n_arr_s / n_total * 100, 2) if n_total > 0 else None,
            "n_on_short": nstart_vals[0],
            "n_on_long": nstart_vals[1],
            "n_blocked": minute_n_blocked.get(m, 0),
            "ema_short": ema_vals[0],
            "ema_long": ema_vals[1],
            "throughput_veh_per_min": n_total,
            # Speed percentiles from instantaneous edge-speed sampling
            "p10_speed_kmh": (
                round(float(np.percentile(minute_speed_obs[m], 10)), 2)
                if m in minute_speed_obs and len(minute_speed_obs[m]) >= 3 else None
            ),
            "p50_speed_kmh": (
                round(float(np.percentile(minute_speed_obs[m], 50)), 2)
                if m in minute_speed_obs and len(minute_speed_obs[m]) >= 3 else None
            ),
            "p90_speed_kmh": (
                round(float(np.percentile(minute_speed_obs[m], 90)), 2)
                if m in minute_speed_obs and len(minute_speed_obs[m]) >= 3 else None
            ),
            "n_speed_obs": len(minute_speed_obs.get(m, [])),
            "avg_speed_kmh": (
                round(float(np.mean(minute_speed_obs[m])), 2)
                if m in minute_speed_obs and minute_speed_obs[m] else None
            ),
            # Departure-minute grouped TT (Wardrop-correct)
            "dep_avg_tt_short": (
                round(float(np.mean(minute_dep_tt_short[m])), 2)
                if m in minute_dep_tt_short and minute_dep_tt_short[m] else None
            ),
            "dep_avg_tt_long": (
                round(float(np.mean(minute_dep_tt_long[m])), 2)
                if m in minute_dep_tt_long and minute_dep_tt_long[m] else None
            ),
            **reg_data,
            # Mediator compliance
            "compliance_rate": (
                round(minute_compliance[m]["followed"] / minute_compliance[m]["total"] * 100, 1)
                if m in minute_compliance and minute_compliance[m]["total"] > 0 else None
            ),
            "mediator_mode": (
                "active" if global_mediator and hasattr(global_mediator, "model_fitted") else None
            ),
        })

    # ── Regression data ──
    regression_data = None
    if davis and hasattr(davis, "slope_short") and davis.slope_short is not None:
        # 2026-05-06 rebuild: buffer_short/buffer_long are the new mixed-source
        # rolling buffers. history_short/long was the pre-rebuild attribute.
        regression_data = {
            "intercept_short": davis.intercept_short,
            "slope_short": davis.slope_short,
            "intercept_long": davis.intercept_long,
            "slope_long": davis.slope_long,
            "n_obs_short": (
                len(davis.buffer_short) if hasattr(davis, "buffer_short")
                else (len(davis.history_short) if hasattr(davis, "history_short") else 0)
            ),
            "n_obs_long": (
                len(davis.buffer_long) if hasattr(davis, "buffer_long")
                else (len(davis.history_long) if hasattr(davis, "history_long") else 0)
            ),
        }

    stats = {
        "n_fwd": n_fwd_injected,
        "n_bwd": n_bwd_injected,
        "n_blocked": n_blocked,
    }

    return vehicle_records, timeseries_records, regression_data, stats, inflight_snapshot


# ─────────────────────────────────────────────────────────────────────
# ROUTE CHOICE DISPATCH
# ─────────────────────────────────────────────────────────────────────

def _compute_mediator_preload(config) -> dict | None:
    """Compute pre-loaded mediator parameters from existing SO baseline data.

    Queries the DB for 50/50 fixed_split runs at this flow rate and fits
    regression + IC margins from the stable post-warmup window.
    Returns None if no suitable data exists (mediator will learn online).
    """
    from scipy import stats as spstats

    try:
        conn = get_fresh_connection()
        rows = conn.execute("""
            SELECT t.n_on_short, t.n_on_long, t.avg_tt_short, t.avg_tt_long
            FROM runs r JOIN timeseries t ON r.run_id = t.run_id
            WHERE r.route_choice_method = 'fixed_split'
              AND r.pct_short_fixed = 0.50
              AND r.flow_rate = ?
              AND r.status = 'completed' AND r.is_valid = 1
              AND t.minute >= (r.warmup_sec / 60)
              AND t.avg_tt_short IS NOT NULL AND t.avg_tt_long IS NOT NULL
            ORDER BY r.sim_duration_sec DESC
        """, (config.flow_rate,)).fetchall()
        conn.close()
    except Exception:
        return None

    if len(rows) < 10:
        return None

    n_s = np.array([r[0] for r in rows], dtype=float)
    n_l = np.array([r[1] for r in rows], dtype=float)
    tt_s = np.array([r[2] for r in rows], dtype=float)
    tt_l = np.array([r[3] for r in rows], dtype=float)

    # Fit regression TT = a + b*N per path
    if np.std(n_s) > 0.1:
        b_short, a_short, _, _, _ = spstats.linregress(n_s, tt_s)
    else:
        b_short, a_short = 0.01, float(np.mean(tt_s))
    if np.std(n_l) > 0.1:
        b_long, a_long, _, _, _ = spstats.linregress(n_l, tt_l)
    else:
        b_long, a_long = 0.01, float(np.mean(tt_l))

    # Enforce non-negative slopes
    if b_short < 0:
        b_short = 0.01
    if b_long < 0:
        b_long = 0.01

    # IC margins: 2*std of TT (covers ~95% of variation at steady state)
    ic_short = max(10.0, float(np.std(tt_s) * 2))
    ic_long = max(10.0, float(np.std(tt_l) * 2))

    # SO target: n_s_so = (a_l - a_s + 2*b_l*N_total) / (2*(b_s + b_l))
    n_total = float(np.mean(n_s + n_l))
    denom = 2.0 * (b_short + b_long)
    if denom > 0:
        n_so = (a_long - a_short + 2.0 * b_long * n_total) / denom
        n_so = max(0.0, min(n_total, n_so))
    else:
        n_so = n_total / 2.0

    print(f"  MEDIATOR PRELOAD @{config.flow_rate}: a_s={a_short:.1f} b_s={b_short:.4f} "
          f"a_l={a_long:.1f} b_l={b_long:.4f} ic_s={ic_short:.1f} ic_l={ic_long:.1f} "
          f"n_so={n_so:.0f}/{n_total:.0f}")

    return {
        "a_short": a_short, "b_short": b_short,
        "a_long": a_long, "b_long": b_long,
        "ic_margin_short": ic_short, "ic_margin_long": ic_long,
        "n_short_so": n_so,
        "n_total_ref": n_total,
    }


def _estimate_queue_wait(path: str, recent_departures: dict,
                         sim_time: float,
                         lookback_sec: float = 60.0) -> float:
    """Estimate the wait an agent would face in the path's pre-entrance FIFO.

    Uses the empirical mean of recently-observed buffer_delays as the proxy:
    "what did vehicles that just left the queue actually wait?". Robust and
    methodology-clean (no model assumption beyond "recent past predicts
    near future"). Returns 0 if no recent departures.

    A potential extension would multiply (current pending count) × (recent
    insertion rate) for a model-based estimate, but the recent-mean approach
    avoids the need for rate calibration.
    """
    recent = recent_departures.get(path, [])
    if not recent:
        return 0.0
    cutoff = sim_time - lookback_sec
    delays = [d for (t, d) in recent if t >= cutoff]
    if not delays:
        return 0.0
    return sum(delays) / len(delays)


def _choose_route(config, props, global_ema, davis, global_mediator,
                   n_short, n_long, veh_counter, sim_time=0.0,
                   toll_penalty_sec: float = 0.0,
                   queue_wait_short: float = 0.0,
                   queue_wait_long: float = 0.0):
    """Choose route based on method in config.

    Args:
        toll_penalty_sec: Agent-specific toll penalty in seconds
            (toll_usd * 3600 / agent.vot). Passed by caller after
            creating the agent, so each agent uses their own VOT.
        queue_wait_short / queue_wait_long: Composite-TT contribution
            (2026-05-06 rebuild). Added to each path's predicted on-network
            TT so the choice reflects total door-to-door expected cost.

    Returns:
        (chosen_path, mediator_details) — mediator_details is None for non-mediated methods.
    """
    method = config.route_choice_method

    if method == "none":
        return config.target_path or "short", None

    if method == "fixed_split":
        # 2026-05-11: per-vehicle Bernoulli draw at pct_short_fixed (matches
        # the locked RQ §4.2 habit rule — each driver draws once at injection,
        # never updates). Pre-fix was counter-based (veh_counter % 100 < int(pct*100))
        # which produced a deterministic mod-100 cycle visible at sub-100-vehicle
        # bin widths (e.g. 1-min split histograms at λ=3,835 → ~64 veh/min < 100).
        # The Bernoulli rule yields the same long-run rate but a smooth per-min
        # signal. fixed_split is VOT-blind and does NOT respond to queue-wait or tolls.
        pct = config.pct_short_fixed or 0.5
        return ("short" if random.random() < pct else "long"), None

    if method == "habit":
        # Habit-based routing: fixed-at-start draw with p_short_habit default 0.55.
        p_short = config.pct_short_fixed if config.pct_short_fixed is not None else 0.55
        return ("short" if random.random() < p_short else "long"), None

    if method == "ema":
        if global_ema:
            import math
            ema_s, ema_l = global_ema.get_values()
            cost_s = ema_s + queue_wait_short + toll_penalty_sec
            cost_l = ema_l + queue_wait_long
            
            theta = config.logit_theta if getattr(config, "logit_theta", None) is not None else 0.05
            delta = theta * (cost_s - cost_l)
            delta = max(-50.0, min(50.0, delta))
            p_short = 1.0 / (1.0 + math.exp(delta))
            chosen = "short" if random.random() < p_short else "long"
            details = {
                "p_short": p_short,
                "cost_short": cost_s,
                "cost_long": cost_l,
            }
            return chosen, details
        return "short", None

    if method in ("davis", "davis_threshold"):
        if davis:
            import math
            pred_s, pred_l = davis.predict_tt(n_short, n_long)
            cost_s = pred_s + queue_wait_short + toll_penalty_sec
            cost_l = pred_l + queue_wait_long
            
            theta = config.logit_theta if getattr(config, "logit_theta", None) is not None else 0.05
            delta = theta * (cost_s - cost_l)
            delta = max(-50.0, min(50.0, delta))
            p_short = 1.0 / (1.0 + math.exp(delta))
            chosen = "short" if random.random() < p_short else "long"
            details = {
                "p_short": p_short,
                "cost_short": cost_s,
                "cost_long": cost_l,
            }
            return chosen, details
        return "short", None

    if method == "thompson":
        # NOTE: GlobalThompsonSampling exists in agent_logic.py but is not
        # wired up here. This fallback is random 50/50 — not Bayesian.
        return ("short" if random.random() < 0.5 else "long"), None

    if method == "mediated":
        if global_mediator and davis:
            ema_s, ema_l = global_ema.get_values() if global_ema else (270.0, 337.5)
            signal, details = global_mediator.get_signal(
                n_short, n_long, config.flow_rate, sim_time, ema_s, ema_l
            )
            # IC check using Davis predictions (stable, not oscillating EMA)
            davis_pred_s, davis_pred_l = davis.predict_tt(n_short, n_long)
            ic_margin = details.get("ic_margin", 0.0)
            if signal == "short":
                davis_penalty = davis_pred_s - davis_pred_l
            else:
                davis_penalty = davis_pred_l - davis_pred_s

            if davis_penalty <= ic_margin:
                chosen = signal
                followed = True
                code = "followed"
            else:
                # Defect to Davis choice (not EMA)
                chosen = "short" if davis_pred_s <= davis_pred_l else "long"
                followed = False
                code = "defected"

            mediator_details = {
                **details,
                "signal": signal,
                "signal_followed": followed,
                "compliance_code": code,
            }
            return chosen, mediator_details
        # Fallback: EMA-only
        if global_ema:
            ema_s, ema_l = global_ema.get_values()
            return ("short" if ema_s <= ema_l else "long"), None
        return "short", None

    return "short", None


def _get_choice_type(method: str) -> str:
    """Map route choice method to choice_type string."""
    return {
        "ema": "ema",
        "davis": "nstart_anticipation",
        "mediated": "mediated",
        "fixed_split": "fixed_split",
        "fixed_split_vot": "fixed_split_vot",
        "thompson": "thompson",
        "none": "fixed_split",
    }.get(method, method)
