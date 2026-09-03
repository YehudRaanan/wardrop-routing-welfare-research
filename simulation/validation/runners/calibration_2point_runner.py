"""
2-Point Calibration Runner (Phase 2)
-------------------------------------
Fast iterative calibration using 2-point Van Aerde methodology:
  - 1000 veh/h (free-flow) + 6000 veh/h (capacity)
  - 30 min simulation (15 min warmup + 15 min recording)
  - Short path only, bidirectional (78/22 split)

Methodology:
  - Cross-sectional instantaneous speeds (traci.vehicle.getSpeed)
  - P10/P50/P90 = np.percentile() over all speed observations
  - Slope = (cap_speed - ff_speed) / 1.95  (per 1000 main-dir PCU, Van Aerde Model II)
  - Tolerance: ±5% on all 6 metrics

Usage:
  python calibration_2point_runner.py
"""

import os
import sys

# libsumo MUST be imported before numpy (numpy loads conflicting MKL DLLs)
import libsumo as traci
import sumolib

import csv
import random
import datetime
import argparse
import json
import numpy as np
from collections import defaultdict

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import path_setup

import simulation_shared as shared
from calibration_config import (
    BENCHMARKS, FWD_RATIO, BWD_RATIO,
    FLOW_FREE_FLOW, FLOW_CONGESTED, EDGE_FLOW_FACTOR,
    validate_result,
    MU_KMH, SIGMA_KMH, SPEED_OFFSET_KMH,
    TAU_MIN, TAU_MAX, TAU_GRADIENT,
    TRUCK_SPEED_MEAN, TRUCK_SPEED_STD, TRUCK_SPEED_MAX, TRUCK_PROBABILITY,
    LC_OPPOSITE, LC_SPEED_GAIN, LC_ASSERTIVE, LC_IMPATIENCE,
)
from project_guardrails import save_parameter_snapshot

# ============================================================
# PARAMETERS
# ============================================================

CONFIG_FILE = path_setup.get_network_path("thesis_dual_route_18022026.sumocfg")
FLOW_STEPS_VEH_H = [FLOW_FREE_FLOW, FLOW_CONGESTED]  # 1000, 6000

# Simulation timing (full 120 min for stable statistics)
SIM_DURATION_SEC = 7200       # 120 min
WARMUP_SEC = 3600             # 60 min
STEP_LENGTH = 0.2
SIM_STEPS = int(SIM_DURATION_SEC / STEP_LENGTH)
WARMUP_STEPS = int(WARMUP_SEC / STEP_LENGTH)
BUCKET_SEC = 60
STEPS_PER_BUCKET = int(BUCKET_SEC / STEP_LENGTH)

# Routes (short path only — spawn directly on main edge, no entry paths)
ROUTE_FWD_EDGES = ["short_fwd"]
ROUTE_BWD_EDGES = ["short_bwd"]
MEASUREMENT_EDGE = "short_fwd"

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Tolerance
TOLERANCE_PCT = 5.0

# Slope denominator (Van Aerde Model II: main-direction PCU per 1000)
FLOW_DIFF_PER_1000 = (FLOW_CONGESTED - FLOW_FREE_FLOW) * EDGE_FLOW_FACTOR / 1000.0  # 1.95

# Full parameter snapshot for documentation
RUN_PARAMETERS = {
    "MU_KMH": MU_KMH,
    "SIGMA_KMH": SIGMA_KMH,
    "SPEED_OFFSET_KMH": SPEED_OFFSET_KMH,
    "TAU_MIN": TAU_MIN,
    "TAU_MAX": TAU_MAX,
    "TAU_GRADIENT": TAU_GRADIENT,
    "TRUCK_SPEED_MEAN": TRUCK_SPEED_MEAN,
    "TRUCK_SPEED_STD": TRUCK_SPEED_STD,
    "TRUCK_SPEED_MAX": TRUCK_SPEED_MAX,
    "TRUCK_PROBABILITY": TRUCK_PROBABILITY,
    "LC_OPPOSITE": LC_OPPOSITE,
    "LC_SPEED_GAIN": LC_SPEED_GAIN,
    "LC_ASSERTIVE": LC_ASSERTIVE,
    "LC_IMPATIENCE": LC_IMPATIENCE,
    "FWD_RATIO": FWD_RATIO,
    "BWD_RATIO": BWD_RATIO,
    "FLOW_FREE_FLOW": FLOW_FREE_FLOW,
    "FLOW_CONGESTED": FLOW_CONGESTED,
    "SIM_DURATION_SEC": 7200,
    "WARMUP_SEC": 3600,
}

# Calibration iteration log path
CAL_LOG_PATH = os.path.join(RESULTS_DIR, "calibration_iteration_log.csv")


# ============================================================
# ITERATION LOGGING
# ============================================================

def log_iteration(timestamp, rows, change_description=""):
    """Append this run's parameters and results to the calibration iteration log."""
    write_header = not os.path.exists(CAL_LOG_PATH)

    log_row = {
        "timestamp": timestamp,
        "SPEED_OFFSET": SPEED_OFFSET_KMH,
        "TAU_MIN": TAU_MIN,
        "TAU_MAX": TAU_MAX,
        "TAU_GRADIENT": TAU_GRADIENT,
        "TRUCK_MEAN": TRUCK_SPEED_MEAN,
        "TRUCK_MAX": TRUCK_SPEED_MAX,
        "FLOW_FF": FLOW_FREE_FLOW,
        "FLOW_CAP": FLOW_CONGESTED,
    }

    # Add results for each percentile
    for r in rows:
        pct = r["percentile"]
        log_row[f"{pct}_ff"] = r["ff_speed"]
        log_row[f"{pct}_ff_dev"] = r["ff_dev_pct"]
        log_row[f"{pct}_ff_pass"] = r["ff_pass"]
        log_row[f"{pct}_slope"] = r["slope"]
        log_row[f"{pct}_slope_dev"] = r["slope_dev_pct"]
        log_row[f"{pct}_slope_pass"] = r["slope_pass"]

    log_row["all_pass"] = all(r["ff_pass"] and r["slope_pass"] for r in rows)
    log_row["change_description"] = change_description

    fieldnames = list(log_row.keys())
    with open(CAL_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(log_row)

    print(f"\nIteration logged to {os.path.basename(CAL_LOG_PATH)}")


# ============================================================
# MAIN RUNNER
# ============================================================

def run_calibration(use_gui=False, change_description=""):
    """Run 2-point calibration."""
    binary_name = "sumo-gui" if use_gui else "sumo"
    try:
        sumo_bin = sumolib.checkBinary(binary_name)
    except Exception:
        sumo_bin = binary_name

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("2-Point Calibration (Van Aerde ±5%)")
    print(f"Flows: {FLOW_FREE_FLOW} (FF) + {FLOW_CONGESTED} (Cap) veh/h")
    print(f"Simulation: {SIM_DURATION_SEC}s total, {WARMUP_SEC}s warmup")
    print(f"Slope denominator: {FLOW_DIFF_PER_1000:.1f}")
    print("=" * 70)

    # Save parameter snapshot
    save_parameter_snapshot(RUN_PARAMETERS, RESULTS_DIR, run_name="cal2pt")

    # Print current parameters
    print(f"\nCurrent Parameters:")
    print(f"  MU={MU_KMH}, SIGMA={SIGMA_KMH}, OFFSET={SPEED_OFFSET_KMH}")
    print(f"  TAU: min={TAU_MIN}, max={TAU_MAX}, gradient={TAU_GRADIENT}")
    print(f"  TRUCK: mean={TRUCK_SPEED_MEAN}, std={TRUCK_SPEED_STD}, max={TRUCK_SPEED_MAX}, prob={TRUCK_PROBABILITY}")
    print(f"  LC: opposite={LC_OPPOSITE}, speedGain={LC_SPEED_GAIN}, assertive={LC_ASSERTIVE}, impatience={LC_IMPATIENCE}")
    print(f"  FLOW: FF={FLOW_FREE_FLOW}, Cap={FLOW_CONGESTED}, denominator={FLOW_DIFF_PER_1000:.1f}")

    results = {}

    for flow in FLOW_STEPS_VEH_H:
        label = "FF" if flow == FLOW_FREE_FLOW else "Cap"
        print(f"\n{'='*50}")
        print(f"Flow: {flow} veh/h ({label}) - {flow*FWD_RATIO:.0f} FWD + {flow*BWD_RATIO:.0f} BWD")
        print(f"{'='*50}")

        speeds = run_single_flow(sumo_bin, flow, use_gui=use_gui)
        results[flow] = speeds

        print(f"  P10={speeds['P10']:.1f}, P50={speeds['P50']:.1f}, P90={speeds['P90']:.1f} km/h")
        print(f"  Observations: {speeds['n_obs']}")

    # Compute slopes and validate
    ff = results[FLOW_FREE_FLOW]
    cap = results[FLOW_CONGESTED]

    print(f"\n{'='*70}")
    print("CALIBRATION RESULTS")
    print(f"{'='*70}")

    all_pass = True
    rows = []

    for pct in ["P10", "P50", "P90"]:
        ff_speed = ff[pct]
        cap_speed = cap[pct]
        slope = (cap_speed - ff_speed) / FLOW_DIFF_PER_1000

        bench_ff = BENCHMARKS[pct]["free_flow_speed"]
        bench_slope = BENCHMARKS[pct]["slope"]

        ff_dev = abs((ff_speed - bench_ff) / bench_ff) * 100
        slope_dev = abs((slope - bench_slope) / bench_slope) * 100

        ff_pass = ff_dev <= TOLERANCE_PCT
        slope_pass = slope_dev <= TOLERANCE_PCT

        if not ff_pass or not slope_pass:
            all_pass = False

        ff_status = "PASS" if ff_pass else "FAIL"
        slope_status = "PASS" if slope_pass else "FAIL"

        print(f"\n  {pct}:")
        print(f"    FF Speed: {ff_speed:.1f} vs {bench_ff:.1f} ({ff_dev:.1f}%) [{ff_status}]")
        print(f"    Slope:    {slope:.1f} vs {bench_slope:.1f} ({slope_dev:.1f}%) [{slope_status}]")

        rows.append({
            "percentile": pct,
            "ff_speed": round(ff_speed, 2),
            "cap_speed": round(cap_speed, 2),
            "slope": round(slope, 2),
            "bench_ff": bench_ff,
            "bench_slope": bench_slope,
            "ff_dev_pct": round(ff_dev, 2),
            "slope_dev_pct": round(slope_dev, 2),
            "ff_pass": ff_pass,
            "slope_pass": slope_pass,
        })

    print(f"\n{'='*70}")
    if all_pass:
        print("  *** ALL 6 BENCHMARKS PASSED (±5%) ***")
    else:
        print("  SOME BENCHMARKS FAILED — parameter tuning needed")
    print(f"{'='*70}")

    # Save results
    csv_path = os.path.join(RESULTS_DIR, f"cal2pt_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved: {os.path.basename(csv_path)}")

    # Log iteration with parameters and results
    log_iteration(timestamp, rows, change_description)

    return all_pass


# ============================================================
# SINGLE FLOW SIMULATION
# ============================================================

def run_single_flow(sumo_bin, flow, use_gui=False):
    """Run simulation for a single flow rate. Returns P10/P50/P90 speeds."""
    fwd_flow = flow * FWD_RATIO
    bwd_flow = flow * BWD_RATIO

    cmd = [
        sumo_bin, "-c", CONFIG_FILE, "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(SIM_DURATION_SEC),
        "--no-warnings", "true",
        "--no-step-log", "true"
    ]
    if use_gui:
        cmd.extend(["--delay", "50"])  # 50ms delay per step for visual observation
    traci.start(cmd)

    if use_gui:
        import time
        time.sleep(2)  # Give GUI time to initialize

    vtype_car, vtype_truck = shared.setup_vehicle_types()

    traci.route.add("route_short_fwd", ROUTE_FWD_EDGES)
    traci.route.add("route_short_bwd", ROUTE_BWD_EDGES)

    # Injection accumulators
    veh_per_step_fwd = (fwd_flow / 3600.0) * STEP_LENGTH
    veh_per_step_bwd = (bwd_flow / 3600.0) * STEP_LENGTH
    acc_fwd = 0.0
    acc_bwd = 0.0
    veh_counter = 0

    # Vehicle tracking
    active_fwd = set()  # Track FWD vehicle IDs
    veh_speeds = defaultdict(list)  # Per-vehicle speed observations (post-warmup)

    for step in range(SIM_STEPS):
        traci.simulationStep()

        # --- Forward injection ---
        acc_fwd += veh_per_step_fwd
        while acc_fwd >= 1.0:
            acc_fwd -= 1.0
            veh_counter += 1
            vid = f"fwd_{flow}_{veh_counter}"
            success, props = shared.inject_vehicle(vid, "route_short_fwd", vtype_car, vtype_truck)
            if success:
                active_fwd.add(vid)

        # --- Backward injection ---
        acc_bwd += veh_per_step_bwd
        while acc_bwd >= 1.0:
            acc_bwd -= 1.0
            veh_counter += 1
            vid = f"bwd_{flow}_{veh_counter}"
            shared.inject_vehicle(vid, "route_short_bwd", vtype_car, vtype_truck)

        # --- Remove arrived vehicles ---
        for vid in traci.simulation.getArrivedIDList():
            active_fwd.discard(vid)

        # --- Collect speeds (post-warmup, every 10 steps = 1s interval) ---
        if step >= WARMUP_STEPS and step % 10 == 0:
            for vid in traci.vehicle.getIDList():
                if vid in active_fwd:
                    try:
                        road = traci.vehicle.getRoadID(vid)
                        if road == MEASUREMENT_EDGE:
                            speed_kmh = traci.vehicle.getSpeed(vid) * 3.6
                            if speed_kmh > 1.0:
                                veh_speeds[vid].append(speed_kmh)
                    except traci.TraCIException:
                        pass

    traci.close()

    # Per-vehicle mean speeds (each vehicle counts once)
    per_veh_means = [float(np.mean(speeds)) for speeds in veh_speeds.values() if len(speeds) > 0]

    if len(per_veh_means) < 10:
        print(f"  WARNING: Only {len(per_veh_means)} vehicles observed!")
        return {"P10": 0, "P50": 0, "P90": 0, "n_obs": len(per_veh_means)}

    print(f"  Unique vehicles: {len(per_veh_means)}")
    return {
        "P10": float(np.percentile(per_veh_means, 10)),
        "P50": float(np.percentile(per_veh_means, 50)),
        "P90": float(np.percentile(per_veh_means, 90)),
        "n_obs": len(per_veh_means),
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch with SUMO-GUI")
    parser.add_argument("--flow-only", type=int, help="Run only this flow rate (e.g. 7000)")
    parser.add_argument("--desc", type=str, default="", help="Change description for iteration log")
    args = parser.parse_args()
    if args.flow_only:
        # Override flow steps to run a single flow
        FLOW_STEPS_VEH_H[:] = [args.flow_only]
    run_calibration(use_gui=args.gui, change_description=args.desc)
