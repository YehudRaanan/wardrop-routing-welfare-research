"""Tests for shared_lib/db/ — schema creation, queries, deduplication."""

import sys
import os
import sqlite3
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared_lib.db.schema import create_tables, get_schema_version, SCHEMA_VERSION
from shared_lib.db.connection import get_fresh_connection, _sql_authorizer
from shared_lib.db.queries import (
    insert_run, update_run_status, insert_vehicles_batch,
    insert_timeseries_batch, get_run, get_completed_runs,
    mark_run_invalid,
)
from shared_lib.db.dedup import find_existing_run
import pytest


def _get_test_db(with_authorizer: bool = False) -> sqlite3.Connection:
    """Create a fresh in-memory database for testing.

    Args:
        with_authorizer: If True, install the SQL authorizer that blocks
                        DELETE/DROP. Default False for backwards compat.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    if with_authorizer:
        conn.set_authorizer(_sql_authorizer)
    return conn


# ─────────────────────────────────────────────────────────────────────
# Schema Tests
# ─────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_tables_created(self):
        conn = _get_test_db()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {t["name"] for t in tables}
        assert "runs" in names
        assert "vehicles" in names
        assert "timeseries" in names
        assert "run_regression" in names

    def test_schema_version_set(self):
        conn = _get_test_db()
        assert get_schema_version(conn) == SCHEMA_VERSION

    def test_idempotent_creation(self):
        conn = _get_test_db()
        create_tables(conn)  # Should not raise on second call
        assert get_schema_version(conn) == SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────
# Query Tests
# ─────────────────────────────────────────────────────────────────────

class TestQueries:
    def test_insert_and_get_run(self):
        conn = _get_test_db()
        run_id = insert_run(conn, {
            "flow_rate": 2000,
            "route_choice_method": "ema",
            "ema_alpha": 0.1,
            "status": "running",
        })
        assert run_id is not None
        run = get_run(conn, run_id)
        assert run["flow_rate"] == 2000
        assert run["route_choice_method"] == "ema"
        assert run["status"] == "running"

    def test_update_run_status(self):
        conn = _get_test_db()
        run_id = insert_run(conn, {
            "flow_rate": 1000,
            "route_choice_method": "davis",
        })
        update_run_status(conn, run_id, "completed", wall_clock_sec=120.5)
        run = get_run(conn, run_id)
        assert run["status"] == "completed"
        assert run["wall_clock_sec"] == 120.5

    def test_mark_invalid(self):
        conn = _get_test_db()
        run_id = insert_run(conn, {
            "flow_rate": 1000,
            "route_choice_method": "ema",
            "ema_alpha": 0.1,
            "status": "completed",
        })
        mark_run_invalid(conn, run_id, "entry-edge bug")
        run = get_run(conn, run_id)
        assert run["is_valid"] == 0
        assert run["validity_reason"] == "entry-edge bug"

    def test_get_completed_runs_filters(self):
        conn = _get_test_db()
        insert_run(conn, {"flow_rate": 1000, "route_choice_method": "ema",
                          "ema_alpha": 0.1, "status": "completed"})
        insert_run(conn, {"flow_rate": 2000, "route_choice_method": "davis",
                          "status": "completed"})
        insert_run(conn, {"flow_rate": 3000, "route_choice_method": "ema",
                          "ema_alpha": 0.1, "status": "failed"})

        all_completed = get_completed_runs(conn)
        assert len(all_completed) == 2

        ema_only = get_completed_runs(conn, method="ema")
        assert len(ema_only) == 1
        assert ema_only[0]["flow_rate"] == 1000

    def test_insert_vehicles_batch(self):
        conn = _get_test_db()
        run_id = insert_run(conn, {
            "flow_rate": 1000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })

        vehicles = [
            {
                "vid": "fwd_1000_1", "direction": "fwd",
                "vehicle_type": "car", "group_id": "q3",
                "assigned_kmh": 100.0, "speed_factor": 1.0,
                "agent_type": "B", "chosen_path": "short",
                "choice_type": "ema",
                "planned_departure_sec": 100.0,
                "arrival_sec": 400.0,
                "travel_time_sec": 300.0,
                "is_warmup": 0,
            },
            {
                "vid": "fwd_1000_2", "direction": "fwd",
                "vehicle_type": "car", "group_id": "q4",
                "assigned_kmh": 110.0, "speed_factor": 1.1,
                "agent_type": "B", "chosen_path": "long",
                "choice_type": "ema",
                "planned_departure_sec": 100.5,
                "arrival_sec": 440.0,
                "travel_time_sec": 339.5,
                "is_warmup": 0,
            },
        ]

        n = insert_vehicles_batch(conn, run_id, vehicles)
        assert n == 2

        count = conn.execute(
            "SELECT COUNT(*) FROM vehicles WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        assert count == 2

    def test_insert_timeseries_batch(self):
        conn = _get_test_db()
        run_id = insert_run(conn, {
            "flow_rate": 1000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })

        ts = [
            {"minute": 60, "is_warmup": 0, "avg_tt_short": 275.0,
             "avg_tt_long": 340.0, "n_arrived_short": 5, "n_arrived_long": 3},
            {"minute": 61, "is_warmup": 0, "avg_tt_short": 280.0,
             "avg_tt_long": 338.0, "n_arrived_short": 6, "n_arrived_long": 4},
        ]

        n = insert_timeseries_batch(conn, run_id, ts)
        assert n == 2


# ─────────────────────────────────────────────────────────────────────
# Dedup Tests
# ─────────────────────────────────────────────────────────────────────

class TestDedup:
    def test_finds_exact_match(self):
        conn = _get_test_db()
        insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        found = find_existing_run(conn, 2000, "ema", ema_alpha=0.1)
        assert found is not None

    def test_misses_different_method(self):
        conn = _get_test_db()
        insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        found = find_existing_run(conn, 2000, "davis")
        assert found is None

    def test_misses_different_flow(self):
        conn = _get_test_db()
        insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        found = find_existing_run(conn, 3000, "ema", ema_alpha=0.1)
        assert found is None

    def test_null_safe_comparison(self):
        """NULL ema_alpha should match NULL ema_alpha (IS instead of =)."""
        conn = _get_test_db()
        insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "davis",
            "status": "completed",
        })
        found = find_existing_run(conn, 2000, "davis")
        assert found is not None

    def test_skips_failed_runs(self):
        conn = _get_test_db()
        insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "failed",
        })
        found = find_existing_run(conn, 2000, "ema", ema_alpha=0.1)
        assert found is None

    def test_skips_invalid_runs(self):
        conn = _get_test_db()
        run_id = insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        mark_run_invalid(conn, run_id, "test invalidation")

        found = find_existing_run(conn, 2000, "ema", ema_alpha=0.1, valid_only=True)
        assert found is None

        # But should find if valid_only=False
        found2 = find_existing_run(conn, 2000, "ema", ema_alpha=0.1, valid_only=False)
        assert found2 is not None


# ─────────────────────────────────────────────────────────────────────
# DB Guardrail Tests — DELETE/DROP protection
# ─────────────────────────────────────────────────────────────────────

class TestDBGuardrails:
    def test_delete_from_runs_blocked(self):
        """DELETE FROM runs must be blocked by authorizer."""
        conn = _get_test_db(with_authorizer=True)
        insert_run(conn, {
            "flow_rate": 1000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM runs WHERE flow_rate = 1000")

    def test_delete_from_vehicles_blocked(self):
        """DELETE FROM vehicles must be blocked by authorizer."""
        conn = _get_test_db(with_authorizer=True)
        run_id = insert_run(conn, {
            "flow_rate": 1000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        insert_vehicles_batch(conn, run_id, [{
            "vid": "test_1", "direction": "fwd", "vehicle_type": "car",
            "group_id": "q3", "assigned_kmh": 100.0, "speed_factor": 1.0,
            "agent_type": "B", "chosen_path": "short",
            "planned_departure_sec": 100.0, "arrival_sec": 400.0,
            "travel_time_sec": 300.0, "is_warmup": 0,
        }])
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM vehicles WHERE run_id = ?", (run_id,))

    def test_delete_from_timeseries_blocked(self):
        """DELETE FROM timeseries must be blocked by authorizer."""
        conn = _get_test_db(with_authorizer=True)
        run_id = insert_run(conn, {
            "flow_rate": 1000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM timeseries WHERE run_id = ?", (run_id,))

    def test_drop_table_blocked(self):
        """DROP TABLE must be blocked by authorizer."""
        conn = _get_test_db(with_authorizer=True)
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DROP TABLE runs")

    def test_update_is_valid_allowed(self):
        """UPDATE is_valid via mark_run_invalid must work (not blocked)."""
        conn = _get_test_db(with_authorizer=True)
        run_id = insert_run(conn, {
            "flow_rate": 1000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        # This should succeed — UPDATE is allowed
        mark_run_invalid(conn, run_id, "test reason: entry-edge bug")
        run = get_run(conn, run_id)
        assert run["is_valid"] == 0
        assert run["validity_reason"] == "test reason: entry-edge bug"

    def test_mark_invalid_requires_reason(self):
        """mark_run_invalid must reject empty reason."""
        conn = _get_test_db(with_authorizer=True)
        run_id = insert_run(conn, {
            "flow_rate": 1000, "route_choice_method": "ema",
            "ema_alpha": 0.1, "status": "completed",
        })
        with pytest.raises(ValueError, match="without explanation"):
            mark_run_invalid(conn, run_id, "")
        with pytest.raises(ValueError, match="without explanation"):
            mark_run_invalid(conn, run_id, "   ")

    def test_mark_invalid_requires_existing_run(self):
        """mark_run_invalid must reject non-existent run_id."""
        conn = _get_test_db(with_authorizer=True)
        with pytest.raises(ValueError, match="does not exist"):
            mark_run_invalid(conn, 9999, "some reason")

    def test_insert_still_works(self):
        """INSERT operations must not be blocked."""
        conn = _get_test_db(with_authorizer=True)
        run_id = insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "davis",
            "status": "completed",
        })
        assert run_id is not None
        run = get_run(conn, run_id)
        assert run["flow_rate"] == 2000

    def test_select_still_works(self):
        """SELECT operations must not be blocked."""
        conn = _get_test_db(with_authorizer=True)
        insert_run(conn, {
            "flow_rate": 2000, "route_choice_method": "davis",
            "status": "completed",
        })
        runs = get_completed_runs(conn)
        assert len(runs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
