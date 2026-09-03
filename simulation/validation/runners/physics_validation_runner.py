"""
Physics Validation Runner (Phase 2)
------------------------------------
Re-validates network physics with extended simulation (120 min, 60 min warmup)
across 7 flow rates on the SHORT PATH ONLY of the thesis network.

Methodology:
    - Short path only (entry_short_fwd → short_fwd), bidirectional with opposing traffic
    - Forward/opposing split: 78/22 (Van Aerde calibration)
    - Per-minute speed sampling: P10/P50/P90 of FWD vehicles on short_fwd edge
    - Per-vehicle travel time tracking (depart → arrive)
    - Van Aerde benchmark comparison (free-flow speed + slope)

Outputs per flow rate:
    - physics_timeseries_flow{FLOW}_{timestamp}.csv  (per-minute P10/P50/P90)
    - physics_per_vehicle_flow{FLOW}_{timestamp}.csv  (individual vehicle records)

Combined outputs:
    - physics_summary_{timestamp}.csv  (all flows, post-warmup averages + slopes)
    - speed_vs_time_flow{FLOW}_{timestamp}.png  (P10/P50/P90, warmup shaded)
    - slope_vs_flow_summary_{timestamp}.png  (measured vs benchmark)
"""

import os
import sys

# libsumo MUST be imported before numpy (numpy's MKL DLLs shadow libsumo's _libsumo.pyd)
import libsumo as traci
import sumolib

import csv
import random
import datetime
import numpy as np
import scipy.stats as spstats
from collections import defaultdict

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import path_setup

import simulation_shared as shared
from calibration_config import (
    BENCHMARKS, FWD_RATIO, BWD_RATIO,
    validate_result, print_validation_report
)
from project_guardrails import require_approval, save_parameter_snapshot

# ============================================================
# PARAMETERS
# ============================================================

CONFIG_FILE = path_setup.get_network_path("thesis_dual_route_18022026.sumocfg")
FLOW_STEPS_VEH_H = [500, 1000, 1500, 2000, 2500, 3000, 3500]

# Simulation timing (METHODOLOGICAL_DECISIONS #3)
SIM_DURATION_SEC = 7200       # 120 min
WARMUP_SEC = 3600             # 60 min
STEP_LENGTH = 0.2             # seconds per step (LOCKED 2026-04-16, was 0.1)
SIM_STEPS = int(SIM_DURATION_SEC / STEP_LENGTH)   # 36000
WARMUP_STEPS = int(WARMUP_SEC / STEP_LENGTH)      # 18000
BUCKET_SEC = 60               # 1-minute buckets
STEPS_PER_BUCKET = int(BUCKET_SEC / STEP_LENGTH)   # 300

# Routes (short path only: direct spawn on main edge, matching calibration)
ROUTE_FWD_EDGES = ["short_fwd"]
ROUTE_BWD_EDGES = ["short_bwd"]

# Main measurement edge (exclude entry queue from speed measurements)
MEASUREMENT_EDGE = "short_fwd"

# Output directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

RUN_PARAMETERS = {
    "Script": "physics_validation_runner.py",
    "FLOW_STEPS": FLOW_STEPS_VEH_H,
    "SIM_DURATION_SEC": SIM_DURATION_SEC,
    "WARMUP_SEC": WARMUP_SEC,
    "STEP_LENGTH": STEP_LENGTH,
    "BUCKET_SEC": BUCKET_SEC,
    "FWD_RATIO": FWD_RATIO,
    "BWD_RATIO": BWD_RATIO,
    "MU_KMH": shared.MU_KMH,
    "SIGMA_KMH": shared.SIGMA_KMH,
    "Path": "Short only",
}


# ============================================================
# MAIN RUNNER
# ============================================================

def run_physics_validation():
    """Run physics validation across all flow rates."""
    try:
        sumo_bin = sumolib.checkBinary("sumo")
    except Exception:
        sumo_bin = "sumo"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("Phase 2: Physics Re-Validation (Short Path Only)")
    print(f"Simulation: {SIM_DURATION_SEC}s total, {WARMUP_SEC}s warmup, {STEP_LENGTH}s step")
    print(f"Flow rates: {FLOW_STEPS_VEH_H}")
    print("=" * 70)

    all_summaries = []
    all_timeseries = {}

    for flow in FLOW_STEPS_VEH_H:
        print(f"\n{'='*50}")
        print(f"Flow: {flow} veh/h ({flow*FWD_RATIO:.0f} FWD + {flow*BWD_RATIO:.0f} BWD)")
        print(f"{'='*50}")

        ts_records, veh_records = run_single_flow(sumo_bin, flow)
        all_timeseries[flow] = ts_records

        # Save per-flow CSVs
        ts_path = os.path.join(RESULTS_DIR, f"physics_timeseries_flow{flow}_{timestamp}.csv")
        save_csv(ts_records, ts_path)

        veh_path = os.path.join(RESULTS_DIR, f"physics_per_vehicle_flow{flow}_{timestamp}.csv")
        save_csv(veh_records, veh_path)

        # Compute post-warmup summary
        summary = compute_summary(ts_records, flow)
        all_summaries.append(summary)

        print(f"  Post-warmup: P10={summary['P10_speed']:.1f}, P50={summary['P50_speed']:.1f}, P90={summary['P90_speed']:.1f} km/h")

    # Compute slopes across flow rates and save combined summary
    summary_path = os.path.join(RESULTS_DIR, f"physics_summary_{timestamp}.csv")
    compute_and_save_slopes(all_summaries, summary_path)

    # Generate plots
    generate_speed_vs_time_plots(all_timeseries, RESULTS_DIR, timestamp)
    generate_slope_vs_flow_plot(all_summaries, RESULTS_DIR, timestamp)

    # Validate against Van Aerde benchmarks
    validate_against_benchmarks(all_summaries)

    print(f"\nPhase 2 COMPLETE. Results in {RESULTS_DIR}/")


# ============================================================
# SINGLE FLOW SIMULATION
# ============================================================

def run_single_flow(sumo_bin, flow):
    """Run simulation for a single flow rate.

    Returns:
        (timeseries_records, vehicle_records)
    """
    fwd_flow = flow * FWD_RATIO
    bwd_flow = flow * BWD_RATIO

    cmd = [
        sumo_bin, "-c", CONFIG_FILE, "--start",
        "--step-length", str(STEP_LENGTH),
        "--end", str(SIM_DURATION_SEC),
        "--no-warnings", "true"
    ]
    traci.start(cmd)

    vtype_car, vtype_truck = shared.setup_vehicle_types()

    # Define routes (short path only)
    traci.route.add("route_short_fwd", ROUTE_FWD_EDGES)
    traci.route.add("route_short_bwd", ROUTE_BWD_EDGES)

    # Injection accumulators
    veh_per_step_fwd = (fwd_flow / 3600.0) * STEP_LENGTH
    veh_per_step_bwd = (bwd_flow / 3600.0) * STEP_LENGTH
    acc_fwd = 0.0
    acc_bwd = 0.0
    veh_counter = 0

    # Vehicle tracking
    active_props = {}  # vid -> props dict
    vehicle_records = []  # Completed vehicle records

    # Per-minute speed collection
    minute_speeds = defaultdict(list)  # minute_number -> [speed_kmh, ...]

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
                props["direction"] = "Fwd"
                props["depart_step"] = step
                active_props[vid] = props

        # --- Backward injection ---
        acc_bwd += veh_per_step_bwd
        while acc_bwd >= 1.0:
            acc_bwd -= 1.0
            veh_counter += 1
            vid = f"bwd_{flow}_{veh_counter}"
            success, props = shared.inject_vehicle(vid, "route_short_bwd", vtype_car, vtype_truck)
            if success:
                props["direction"] = "Bwd"
                props["depart_step"] = step
                active_props[vid] = props

        # --- Record arrived vehicles ---
        for vid in traci.simulation.getArrivedIDList():
            if vid in active_props:
                props = active_props.pop(vid)
                tt_sec = (step - props["depart_step"]) * STEP_LENGTH
                vehicle_records.append({
                    "vid": vid,
                    "group_id": props["group_id"],
                    "assigned_kmh": round(props["assigned_kmh"], 2),
                    "travel_time_sec": round(tt_sec, 2),
                    "departure_min": round(props["depart_step"] * STEP_LENGTH / 60, 2),
                    "arrival_min": round(step * STEP_LENGTH / 60, 2),
                    "direction": props["direction"],
                    "is_warmup": props["depart_step"] < WARMUP_STEPS
                })

        # --- Per-minute speed sampling ---
        if step > 0 and step % STEPS_PER_BUCKET == 0:
            minute_num = int(step * STEP_LENGTH / 60)

            # Collect instantaneous speeds of FWD vehicles on measurement edge
            for vid in traci.vehicle.getIDList():
                if vid in active_props and active_props[vid]["direction"] == "Fwd":
                    try:
                        road = traci.vehicle.getRoadID(vid)
                        if road == MEASUREMENT_EDGE:
                            speed_kmh = traci.vehicle.getSpeed(vid) * 3.6
                            if speed_kmh > 1.0:
                                minute_speeds[minute_num].append(speed_kmh)
                    except traci.TraCIException:
                        pass

    traci.close()

    # Build timeseries records
    timeseries = []
    warmup_min = WARMUP_SEC / 60
    for minute_num in sorted(minute_speeds.keys()):
        speeds = minute_speeds[minute_num]
        if len(speeds) >= 3:
            timeseries.append({
                "minute": minute_num,
                "n_vehicles": len(speeds),
                "avg_speed_kmh": round(float(np.mean(speeds)), 2),
                "P10_speed_kmh": round(float(np.percentile(speeds, 10)), 2),
                "P50_speed_kmh": round(float(np.percentile(speeds, 50)), 2),
                "P90_speed_kmh": round(float(np.percentile(speeds, 90)), 2),
                "is_warmup": minute_num <= warmup_min
            })

    print(f"  Timeseries: {len(timeseries)} min, Vehicles completed: {len(vehicle_records)}")
    return timeseries, vehicle_records


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def save_csv(records, path):
    """Save list of dicts to CSV."""
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def compute_summary(timeseries, flow):
    """Compute post-warmup average P10/P50/P90 speeds."""
    warmup_min = WARMUP_SEC / 60
    post_warmup = [r for r in timeseries if r["minute"] > warmup_min]

    if not post_warmup:
        return {"flow": flow, "P10_speed": 0, "P50_speed": 0, "P90_speed": 0}

    return {
        "flow": flow,
        "P10_speed": round(float(np.mean([r["P10_speed_kmh"] for r in post_warmup])), 2),
        "P50_speed": round(float(np.mean([r["P50_speed_kmh"] for r in post_warmup])), 2),
        "P90_speed": round(float(np.mean([r["P90_speed_kmh"] for r in post_warmup])), 2),
    }


def compute_and_save_slopes(summaries, path):
    """Compute slopes (km/h per 1000 veh/h) via linear regression across flow rates."""
    flows = np.array([s["flow"] for s in summaries])

    for pct in ["P10", "P50", "P90"]:
        speeds = np.array([s[f"{pct}_speed"] for s in summaries])
        valid = speeds > 10
        if np.sum(valid) >= 2:
            slope, intercept, r, p, se = spstats.linregress(flows[valid], speeds[valid])
            slope_per_1000 = round(float(slope * 1000), 2)
            intercept_val = round(float(intercept), 2)
        else:
            slope_per_1000 = 0.0
            intercept_val = 0.0

        for s in summaries:
            s[f"{pct}_slope"] = slope_per_1000
            s[f"{pct}_intercept"] = intercept_val

    save_csv(summaries, path)

    print(f"\nSummary saved to {os.path.basename(path)}")
    print(f"\n{'='*60}")
    print("SLOPE COMPARISON (km/h per 1000 veh/h)")
    print(f"{'='*60}")
    for pct in ["P10", "P50", "P90"]:
        measured = summaries[0].get(f"{pct}_slope", 0)
        benchmark = BENCHMARKS[pct]["slope"]
        print(f"  {pct}: Measured={measured:.1f}, Benchmark={benchmark:.1f}")


# ============================================================
# PLOTTING
# ============================================================

def generate_speed_vs_time_plots(all_timeseries, results_dir, timestamp):
    """Generate speed vs time plots with warmup shading."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        return

    warmup_min = WARMUP_SEC / 60

    for flow, ts_records in all_timeseries.items():
        if not ts_records:
            continue

        minutes = [r["minute"] for r in ts_records]
        p10 = [r["P10_speed_kmh"] for r in ts_records]
        p50 = [r["P50_speed_kmh"] for r in ts_records]
        p90 = [r["P90_speed_kmh"] for r in ts_records]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Warmup shading
        ax.axvspan(0, warmup_min, alpha=0.15, color='gray', label='Warmup')
        ax.axvline(warmup_min, color='gray', linestyle='--', alpha=0.5)

        # Speed lines
        ax.plot(minutes, p90, 'g-', label='P90 (Fast)', linewidth=1.5)
        ax.plot(minutes, p50, 'b-', label='P50 (Median)', linewidth=2)
        ax.plot(minutes, p10, 'r-', label='P10 (Slow)', linewidth=1.5)

        # Benchmark reference lines
        ax.axhline(BENCHMARKS["P90"]["free_flow_speed"], color='g', linestyle=':', alpha=0.4)
        ax.axhline(BENCHMARKS["P50"]["free_flow_speed"], color='b', linestyle=':', alpha=0.4)
        ax.axhline(BENCHMARKS["P10"]["free_flow_speed"], color='r', linestyle=':', alpha=0.4)

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title(f"Speed vs Time - {flow} veh/h (Short Path, FWD)")
        ax.legend(loc='lower right')
        ax.set_xlim(0, SIM_DURATION_SEC / 60)
        ax.set_ylim(0, 160)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(results_dir, f"speed_vs_time_flow{flow}_{timestamp}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Plot: {os.path.basename(plot_path)}")


def generate_slope_vs_flow_plot(summaries, results_dir, timestamp):
    """Generate slope vs flow summary plot (Van Aerde comparison)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        return

    flows = [s["flow"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Speed vs Flow (post-warmup averages)
    colors = {"P90": "green", "P50": "blue", "P10": "red"}
    markers = {"P90": "^", "P50": "o", "P10": "v"}
    for pct in ["P90", "P50", "P10"]:
        speeds = [s[f"{pct}_speed"] for s in summaries]
        ax1.plot(flows, speeds, marker=markers[pct], color=colors[pct],
                 linestyle='-', label=pct, linewidth=1.5, markersize=6)
        ax1.axhline(BENCHMARKS[pct]["free_flow_speed"],
                     color=colors[pct], linestyle=':', alpha=0.4)

    ax1.set_xlabel("Flow (veh/h)")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Post-Warmup Speed vs Flow")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 160)

    # Right: Measured slopes vs benchmarks (bar chart)
    pcts = ["P10", "P50", "P90"]
    measured_slopes = [summaries[0].get(f"{p}_slope", 0) for p in pcts]
    benchmark_slopes = [BENCHMARKS[p]["slope"] for p in pcts]

    x = np.arange(len(pcts))
    width = 0.35
    ax2.bar(x - width/2, measured_slopes, width, label='Measured', color='steelblue')
    ax2.bar(x + width/2, benchmark_slopes, width, label='Van Aerde', color='coral')
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("Slope (km/h per 1000 veh/h)")
    ax2.set_title("Slope: Measured vs Benchmark")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pcts)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    plot_path = os.path.join(results_dir, f"slope_vs_flow_summary_{timestamp}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot: {os.path.basename(plot_path)}")


# ============================================================
# VALIDATION
# ============================================================

def validate_against_benchmarks(summaries):
    """Validate post-warmup results against Van Aerde benchmarks."""
    # Free-flow reference = lowest flow rate (500 veh/h)
    ff_summary = summaries[0]

    results_dict = {}
    for pct in ["P10", "P50", "P90"]:
        results_dict[pct] = {
            "free_flow_speed": ff_summary[f"{pct}_speed"],
            "slope": summaries[0].get(f"{pct}_slope", 0)
        }

    print_validation_report(results_dict)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    require_approval(
        goal="Phase 2: Physics re-validation (short path, 7 flows, 120 min each)",
        params=RUN_PARAMETERS,
        expected_output="physics_timeseries, physics_per_vehicle, physics_summary CSVs + speed_vs_time plots"
    )
    save_parameter_snapshot(RUN_PARAMETERS, RESULTS_DIR, run_name="physics_validation")

    run_physics_validation()
