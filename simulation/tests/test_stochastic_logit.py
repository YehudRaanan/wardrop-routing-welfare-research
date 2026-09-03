"""
Smoke Test: Verify Stochastic Logit Route Choice Behavior.

This script runs short simulations to verify:
1. Pure Random Split (theta = 0.0): Choices should be close to 50/50 short/long.
2. Moderate Exploration (theta = 0.05): At free-flow, choices should be biased towards short path (~91%).
3. Deterministic Limit (theta = 1000.0): Choices should match deterministic cost optimization.

Uses a temporary local database to prevent contaminating the production primary database.
"""

import os
import sys
import sqlite3

# Setup paths
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "shared_lib"))

from shared_lib.runners.run_config import RunConfig
from shared_lib.runners.unified_runner import run_simulation
from shared_lib.db.connection import get_fresh_connection

import numpy as np

TEST_DB_PATH = "test_temp_logit.db"

def run_test_case(theta: float, name: str):
    print(f"\n--- Running: {name} (theta = {theta}) ---")
    
    # We use a short simulation (600s duration, 300s warmup) to keep it fast.
    # Note: validation warnings will print due to standard mismatch, which is normal for tests.
    config = RunConfig(
        flow_rate=3000,  # Lower flow rate to keep network at free-flow
        route_choice_method="ema",
        ema_alpha=0.1,
        logit_theta=theta,
        sim_duration_sec=600,
        warmup_sec=300,
        random_seed=42,  # Fixed seed for reproducibility
        notes=f"logit smoke test: {name}",
    )
    
    run_id = run_simulation(config, force=True, db_path=TEST_DB_PATH)
    
    # Query database to get vehicle choice distribution
    conn = get_fresh_connection(TEST_DB_PATH)
    try:
        # Get count of vehicles choosing short vs long
        rows = conn.execute(
            """
            SELECT chosen_path, COUNT(*) as count 
            FROM vehicles 
            WHERE run_id = ? AND trip_status = 'completed' AND is_warmup = 0
            GROUP BY chosen_path
            """,
            (run_id,)
        ).fetchall()
        
        counts = {r["chosen_path"]: r["count"] for r in rows}
        n_short = counts.get("short", 0)
        n_long = counts.get("long", 0)
        n_total = n_short + n_long
        
        pct_short = (n_short / n_total * 100.0) if n_total > 0 else 0.0
        
        # Also let's print statistics of p_short_at_choice and costs
        stats_row = conn.execute(
            """
            SELECT 
                AVG(p_short_at_choice) as avg_p,
                AVG(cost_short_at_choice) as avg_cost_s,
                AVG(cost_long_at_choice) as avg_cost_l
            FROM vehicles
            WHERE run_id = ? AND trip_status = 'completed' AND is_warmup = 0
            """,
            (run_id,)
        ).fetchone()
        
        avg_p = stats_row["avg_p"]
        avg_cost_s = stats_row["avg_cost_s"]
        avg_cost_l = stats_row["avg_cost_l"]
        
        print(f"Results for run_id={run_id}:")
        print(f"  Total vehicles: {n_total}")
        print(f"  Short path choices: {n_short} ({pct_short:.1f}%)")
        print(f"  Long path choices:  {n_long} ({100.0 - pct_short:.1f}%)")
        print(f"  Avg p_short_at_choice: {avg_p:.4f}" if avg_p is not None else "  Avg p_short_at_choice: N/A")
        print(f"  Avg cost_short: {avg_cost_s:.2f} s" if avg_cost_s is not None else "  Avg cost_short: N/A")
        print(f"  Avg cost_long:  {avg_cost_l:.2f} s" if avg_cost_l is not None else "  Avg cost_long: N/A")
        
        return {
            "n_total": n_total,
            "n_short": n_short,
            "n_long": n_long,
            "pct_short": pct_short,
            "avg_p": avg_p,
            "avg_cost_s": avg_cost_s,
            "avg_cost_l": avg_cost_l,
        }
    finally:
        conn.close()

def main():
    print("=" * 70)
    print("LOGIT MODEL SMOKE TEST")
    print("=" * 70)
    
    # Remove existing test DB if any
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass
            
    # Run test cases
    results = {}
    results["theta_0"] = run_test_case(0.0, "Pure Random (theta = 0.0)")
    results["theta_05"] = run_test_case(0.05, "Moderate Exploration (theta = 0.05)")
    results["theta_1000"] = run_test_case(1000.0, "Deterministic Limit (theta = 1000.0)")
    
    print("\n" + "=" * 70)
    print("LOGIT VERIFICATION SUMMARY")
    print("=" * 70)
    
    # Test 1: Theta = 0.0 -> statistical 50/50 split
    pct_0 = results["theta_0"]["pct_short"]
    t0_ok = 40.0 <= pct_0 <= 60.0
    print(f"Test 1 (theta = 0.0): Split is {pct_0:.1f}% short vs {100.0 - pct_0:.1f}% long")
    print(f"  Expected: ~50% each (within random noise [40%, 60%]) -> {'PASS' if t0_ok else 'FAIL'}")
    
    # Test 2: Theta = 0.05 -> approx 91% short
    pct_05 = results["theta_05"]["pct_short"]
    t05_ok = 85.0 <= pct_05 <= 95.0
    print(f"Test 2 (theta = 0.05): Split is {pct_05:.1f}% short vs {100.0 - pct_05:.1f}% long")
    print(f"  Expected: ~91% short (free-flow bias, within [85%, 95%]) -> {'PASS' if t05_ok else 'FAIL'}")
    
    # Test 3: Theta = 1000.0 -> deterministic cost-choice (short path has lower cost, should be 100% short)
    pct_1000 = results["theta_1000"]["pct_short"]
    t1000_ok = pct_1000 == 100.0
    print(f"Test 3 (theta = 1000.0): Split is {pct_1000:.1f}% short vs {100.0 - pct_1000:.1f}% long")
    print(f"  Expected: 100% short (cost of short path ~270s < long path ~338s) -> {'PASS' if t1000_ok else 'FAIL'}")
    
    # Clean up test DB
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass
            
    print("\n" + "=" * 70)
    if t0_ok and t05_ok and t1000_ok:
        print("ALL LOGIT SMOKE TESTS PASSED!")
    else:
        print("SOME LOGIT SMOKE TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
