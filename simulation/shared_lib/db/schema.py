"""
SQLite schema for simulation results.

Tables:
    runs            — Run metadata + deduplication key
    vehicles        — Per-vehicle fact table (superset of all phases)
    timeseries      — Per-minute aggregate snapshots
    run_regression  — Final regression state per run (Davis/Mediator)

Schema versioning via PRAGMA user_version. Migrations use ALTER TABLE
ADD COLUMN for backwards-compatible extensions.
"""

import sqlite3

SCHEMA_VERSION = 8

# ─────────────────────────────────────────────────────────────────────
# TABLE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────

CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime')),

    -- DEDUPLICATION KEY (these columns form the UNIQUE constraint)
    flow_rate               INTEGER NOT NULL,
    route_choice_method     TEXT NOT NULL,
    agent_mix               TEXT NOT NULL DEFAULT '100B',
    ema_alpha               REAL,
    logit_theta             REAL,
    toll_short_usd          REAL NOT NULL DEFAULT 0.0,
    pct_short_fixed         REAL,
    demand_elasticity       REAL,
    random_seed             INTEGER,

    -- Simulation envelope (LOCKED values, stored for verification)
    sim_duration_sec    INTEGER NOT NULL DEFAULT 7200,
    warmup_sec          INTEGER NOT NULL DEFAULT 3600,
    step_length         REAL    NOT NULL DEFAULT 0.1,
    fwd_ratio           REAL    NOT NULL DEFAULT 0.78,
    bwd_ratio           REAL    NOT NULL DEFAULT 0.22,

    -- Run lifecycle
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK(status IN ('running','completed','failed')),
    error_message       TEXT,
    wall_clock_sec      REAL,
    n_vehicles_fwd      INTEGER,
    n_vehicles_bwd      INTEGER,
    n_vehicles_blocked  INTEGER,

    -- Validity tracking (can be updated post-hoc)
    is_valid            INTEGER DEFAULT 1,
    validity_reason     TEXT,

    notes               TEXT,

    UNIQUE(flow_rate, route_choice_method, agent_mix, ema_alpha, logit_theta,
           toll_short_usd, pct_short_fixed, demand_elasticity, random_seed,
           sim_duration_sec, warmup_sec)
);
"""

CREATE_VEHICLES = """
CREATE TABLE IF NOT EXISTS vehicles (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  INTEGER NOT NULL REFERENCES runs(run_id),

    -- Identity
    vid                     TEXT NOT NULL,
    direction               TEXT NOT NULL CHECK(direction IN ('fwd','bwd')),
    vehicle_type            TEXT NOT NULL,
    group_id                TEXT NOT NULL,
    assigned_kmh            REAL NOT NULL,
    speed_factor            REAL NOT NULL,

    -- Agent
    agent_type              TEXT NOT NULL DEFAULT 'B'
                            CHECK(agent_type IN ('A','B','C','none')),
    chosen_path             TEXT NOT NULL CHECK(chosen_path IN ('short','long')),
    choice_type             TEXT,

    -- Timing
    planned_departure_sec   REAL NOT NULL,
    actual_departure_sec    REAL,
    arrival_sec             REAL NOT NULL,
    travel_time_sec         REAL NOT NULL,
    buffer_delay_sec        REAL DEFAULT 0.0,
    is_warmup               INTEGER NOT NULL,

    -- Network state at choice time (ALL phases)
    n_start_short           INTEGER,
    n_start_long            INTEGER,

    -- EMA state (Phases 2, 4)
    ema_short_at_choice     REAL,
    ema_long_at_choice      REAL,

    -- Davis predictions (Phases 3, 4)
    davis_pred_short        REAL,
    davis_pred_long         REAL,

    -- Mediator (Phase 4)
    mediator_signal         TEXT,
    signal_followed         INTEGER,
    compliance_code         TEXT,
    mediator_mode           TEXT,
    ic_margin_at_choice     REAL,
    ic_constrained          INTEGER,

    -- Economic (Phase 5+)
    vot_usd_hr              REAL,
    income_gbp              REAL,
    trip_purpose            TEXT,
    toll_paid_usd           REAL DEFAULT 0.0,
    generalized_cost_sec    REAL,
    reservation_price_usd   REAL,

    -- Discrete Choice Modeling (Phase 5+ Stochastic Logit)
    p_short_at_choice       REAL,
    cost_short_at_choice    REAL,
    cost_long_at_choice     REAL,

    -- Speed observations (v2: edge-speed sampling for calibration comparison)
    mean_speed_kmh          REAL,
    n_speed_samples         INTEGER,

    -- TraCI ground-truth travel times (v7: rebuild — independent reference for analysis)
    -- traci.edge.getTraveltime(edge) = edge_length / mean_observed_speed_on_edge
    -- Captured per vehicle to enable choice-quality and queue-impact analysis.
    traci_tt_short_at_choice    REAL,    -- TraCI estimate when agent made route choice
    traci_tt_long_at_choice     REAL,
    traci_tt_short_at_departure REAL,    -- TraCI estimate at actual departure (post-queue)
    traci_tt_long_at_departure  REAL,

    -- Trip status — values constrained by v4 CHECK; SQLite cannot extend a
    -- CHECK via ALTER, so post-rebuild we encode end-of-sim states by
    -- mapping to 'blocked' and discriminating in analysis via
    -- actual_departure_sec / travel_time_sec:
    --   completed                                                  — full trip
    --   suppressed                                                  — elastic-demand-suppressed (Phase 6)
    --   blocked AND actual_departure_sec IS NOT NULL                — was previously: TraCI exception
    --                AND travel_time_sec > 0                            (rare; this path is essentially dead with --max-depart-delay)
    --   blocked AND actual_departure_sec IS NULL                    — queued at sim end
    --                                                                  (in pre-entrance FIFO when sim ended)
    --   blocked AND actual_departure_sec IS NOT NULL                — in-transit at sim end
    --                AND travel_time_sec = 0                            (departed but didn't arrive)
    trip_status             TEXT DEFAULT 'completed'
                            CHECK(trip_status IN ('completed','suppressed','blocked'))
);
"""

CREATE_VEHICLES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_veh_run ON vehicles(run_id);
CREATE INDEX IF NOT EXISTS idx_veh_run_path ON vehicles(run_id, chosen_path);
CREATE INDEX IF NOT EXISTS idx_veh_run_warmup ON vehicles(run_id, is_warmup);
"""

CREATE_TIMESERIES = """
CREATE TABLE IF NOT EXISTS timeseries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(run_id),
    minute          INTEGER NOT NULL,
    is_warmup       INTEGER NOT NULL,

    -- Travel time aggregates
    avg_tt_short    REAL,
    avg_tt_long     REAL,
    total_tt_sec    REAL,
    n_arrived_short INTEGER DEFAULT 0,
    n_arrived_long  INTEGER DEFAULT 0,

    -- Departures
    n_chose_short   INTEGER DEFAULT 0,
    n_chose_long    INTEGER DEFAULT 0,
    pct_short       REAL,

    -- Network state
    n_on_short      INTEGER,
    n_on_long       INTEGER,
    n_blocked       INTEGER DEFAULT 0,

    -- EMA (Phases 2, 4)
    ema_short       REAL,
    ema_long        REAL,

    -- Davis regression (Phases 3, 4)
    reg_intercept_short REAL,
    reg_slope_short     REAL,
    reg_intercept_long  REAL,
    reg_slope_long      REAL,

    -- Mediator (Phase 4)
    compliance_rate REAL,
    mediator_mode   TEXT,

    -- Throughput
    throughput_veh_per_min REAL,

    -- Edge speed percentiles (v2: instantaneous speed sampling, km/h)
    p10_speed_kmh       REAL,
    p50_speed_kmh       REAL,
    p90_speed_kmh       REAL,
    n_speed_obs         INTEGER,
    avg_speed_kmh       REAL,

    -- Departure-based TT (v5: Wardrop equilibrium requires departure-cohort comparison)
    dep_avg_tt_short    REAL,
    dep_avg_tt_long     REAL,
    n_departed_short    INTEGER DEFAULT 0,
    n_departed_long     INTEGER DEFAULT 0,

    UNIQUE(run_id, minute)
);
"""

CREATE_TIMESERIES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_ts_run ON timeseries(run_id);
"""

CREATE_RUN_REGRESSION = """
CREATE TABLE IF NOT EXISTS run_regression (
    run_id              INTEGER PRIMARY KEY REFERENCES runs(run_id),
    intercept_short     REAL,
    slope_short         REAL,
    intercept_long      REAL,
    slope_long          REAL,
    n_obs_short         INTEGER,
    n_obs_long          INTEGER,
    r_squared_short     REAL,
    r_squared_long      REAL
);
"""

# ─────────────────────────────────────────────────────────────────────
# SCHEMA MANAGEMENT
# ─────────────────────────────────────────────────────────────────────

def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't exist."""
    conn.executescript(CREATE_RUNS)
    conn.executescript(CREATE_VEHICLES)
    conn.executescript(CREATE_VEHICLES_INDEXES)
    conn.executescript(CREATE_TIMESERIES)
    conn.executescript(CREATE_TIMESERIES_INDEX)
    conn.executescript(CREATE_RUN_REGRESSION)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return current schema version."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> None:
    """Apply any pending schema migrations.

    Each migration block checks the current version and applies
    ALTER TABLE ADD COLUMN statements as needed. Existing rows
    get NULL for new columns, which is correct behavior.
    """
    current = get_schema_version(conn)

    if current == 0:
        # Fresh database — create all tables
        create_tables(conn)
        return

    if current < 2:
        # v2: Add speed observation columns for calibration comparison
        conn.execute("ALTER TABLE vehicles ADD COLUMN mean_speed_kmh REAL")
        conn.execute("ALTER TABLE vehicles ADD COLUMN n_speed_samples INTEGER")
        conn.execute("ALTER TABLE timeseries ADD COLUMN p10_speed_kmh REAL")
        conn.execute("ALTER TABLE timeseries ADD COLUMN p50_speed_kmh REAL")
        conn.execute("ALTER TABLE timeseries ADD COLUMN p90_speed_kmh REAL")
        conn.execute("ALTER TABLE timeseries ADD COLUMN n_speed_obs INTEGER")
        conn.execute("ALTER TABLE timeseries ADD COLUMN avg_speed_kmh REAL")
        conn.execute("PRAGMA user_version = 2")
        conn.commit()

    if current < 3:
        # v3: Add sim_duration_sec and warmup_sec to UNIQUE constraint.
        # SQLite cannot ALTER constraints, so recreate the table.
        # Disable FK checks during table swap.
        conn.executescript("""
            PRAGMA foreign_keys = OFF;

            CREATE TABLE runs_new (
                run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime')),
                flow_rate               INTEGER NOT NULL,
                route_choice_method     TEXT NOT NULL,
                agent_mix               TEXT NOT NULL DEFAULT '100B',
                ema_alpha               REAL,
                toll_short_usd          REAL NOT NULL DEFAULT 0.0,
                pct_short_fixed         REAL,
                demand_elasticity       REAL,
                random_seed             INTEGER,
                sim_duration_sec    INTEGER NOT NULL DEFAULT 7200,
                warmup_sec          INTEGER NOT NULL DEFAULT 3600,
                step_length         REAL    NOT NULL DEFAULT 0.1,
                fwd_ratio           REAL    NOT NULL DEFAULT 0.78,
                bwd_ratio           REAL    NOT NULL DEFAULT 0.22,
                status              TEXT NOT NULL DEFAULT 'running'
                                    CHECK(status IN ('running','completed','failed')),
                error_message       TEXT,
                wall_clock_sec      REAL,
                n_vehicles_fwd      INTEGER,
                n_vehicles_bwd      INTEGER,
                n_vehicles_blocked  INTEGER,
                is_valid            INTEGER DEFAULT 1,
                validity_reason     TEXT,
                notes               TEXT,
                UNIQUE(flow_rate, route_choice_method, agent_mix, ema_alpha,
                       toll_short_usd, pct_short_fixed, demand_elasticity, random_seed,
                       sim_duration_sec, warmup_sec)
            );

            INSERT INTO runs_new SELECT * FROM runs;
            DROP TABLE runs;
            ALTER TABLE runs_new RENAME TO runs;

            PRAGMA foreign_keys = ON;
            PRAGMA user_version = 3;
        """)
        conn.commit()

    if current < 4:
        # v4: Add trip_status column to track suppressed and blocked vehicles
        conn.execute("""
            ALTER TABLE vehicles ADD COLUMN trip_status TEXT DEFAULT 'completed'
        """)
        conn.execute("PRAGMA user_version = 4")
        conn.commit()

    if current < 5:
        # v5: Departure-based TT columns for Wardrop equilibrium measurement
        conn.execute("ALTER TABLE timeseries ADD COLUMN dep_avg_tt_short REAL")
        conn.execute("ALTER TABLE timeseries ADD COLUMN dep_avg_tt_long REAL")
        conn.execute("ALTER TABLE timeseries ADD COLUMN n_departed_short INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE timeseries ADD COLUMN n_departed_long INTEGER DEFAULT 0")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()

    if current < 6:
        # v6: Add reservation_price_usd to vehicles for elastic demand (Phase 6)
        conn.execute("ALTER TABLE vehicles ADD COLUMN reservation_price_usd REAL")
        conn.execute("PRAGMA user_version = 6")
        conn.commit()

    if current < 7:
        # v7 (2026-05-06 rebuild): Add TraCI ground-truth TT columns for
        # independent choice-quality / queue-impact analysis. SQLite ALTER
        # cannot extend a CHECK constraint, so trip_status's set of allowed
        # values is enforced only in newly-created tables (CREATE_VEHICLES);
        # for migrated databases, the new value 'queued_at_sim_end' is
        # accepted because the existing column has no CHECK after v4's ALTER.
        conn.execute("ALTER TABLE vehicles ADD COLUMN traci_tt_short_at_choice REAL")
        conn.execute("ALTER TABLE vehicles ADD COLUMN traci_tt_long_at_choice REAL")
        conn.execute("ALTER TABLE vehicles ADD COLUMN traci_tt_short_at_departure REAL")
        conn.execute("ALTER TABLE vehicles ADD COLUMN traci_tt_long_at_departure REAL")
        conn.execute("PRAGMA user_version = 7")
        conn.commit()

    if current < 8:
        # v8: Add logit columns to vehicles, and recreate runs table to include logit_theta in UNIQUE constraint
        conn.execute("ALTER TABLE vehicles ADD COLUMN p_short_at_choice REAL")
        conn.execute("ALTER TABLE vehicles ADD COLUMN cost_short_at_choice REAL")
        conn.execute("ALTER TABLE vehicles ADD COLUMN cost_long_at_choice REAL")

        conn.executescript("""
            PRAGMA foreign_keys = OFF;

            CREATE TABLE runs_new (
                run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime')),
                flow_rate               INTEGER NOT NULL,
                route_choice_method     TEXT NOT NULL,
                agent_mix               TEXT NOT NULL DEFAULT '100B',
                ema_alpha               REAL,
                logit_theta             REAL,
                toll_short_usd          REAL NOT NULL DEFAULT 0.0,
                pct_short_fixed         REAL,
                demand_elasticity       REAL,
                random_seed             INTEGER,
                sim_duration_sec    INTEGER NOT NULL DEFAULT 7200,
                warmup_sec          INTEGER NOT NULL DEFAULT 3600,
                step_length         REAL    NOT NULL DEFAULT 0.1,
                fwd_ratio           REAL    NOT NULL DEFAULT 0.78,
                bwd_ratio           REAL    NOT NULL DEFAULT 0.22,
                status              TEXT NOT NULL DEFAULT 'running'
                                    CHECK(status IN ('running','completed','failed')),
                error_message       TEXT,
                wall_clock_sec      REAL,
                n_vehicles_fwd      INTEGER,
                n_vehicles_bwd      INTEGER,
                n_vehicles_blocked  INTEGER,
                is_valid            INTEGER DEFAULT 1,
                validity_reason     TEXT,
                notes               TEXT,
                UNIQUE(flow_rate, route_choice_method, agent_mix, ema_alpha, logit_theta,
                       toll_short_usd, pct_short_fixed, demand_elasticity, random_seed,
                       sim_duration_sec, warmup_sec)
            );

            INSERT INTO runs_new (
                run_id, created_at, flow_rate, route_choice_method, agent_mix,
                ema_alpha, toll_short_usd, pct_short_fixed, demand_elasticity,
                random_seed, sim_duration_sec, warmup_sec, step_length,
                fwd_ratio, bwd_ratio, status, error_message, wall_clock_sec,
                n_vehicles_fwd, n_vehicles_bwd, n_vehicles_blocked, is_valid,
                validity_reason, notes
            )
            SELECT 
                run_id, created_at, flow_rate, route_choice_method, agent_mix,
                ema_alpha, toll_short_usd, pct_short_fixed, demand_elasticity,
                random_seed, sim_duration_sec, warmup_sec, step_length,
                fwd_ratio, bwd_ratio, status, error_message, wall_clock_sec,
                n_vehicles_fwd, n_vehicles_bwd, n_vehicles_blocked, is_valid,
                validity_reason, notes
            FROM runs;

            DROP TABLE runs;
            ALTER TABLE runs_new RENAME TO runs;

            PRAGMA foreign_keys = ON;
            PRAGMA user_version = 8;
        """)
        conn.commit()
