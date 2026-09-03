"""
30-Run Validation Script
========================
Runs 30 FF and 15 Cap simulations to assess stochastic variability.
Uses 90-min simulation (60 min warmup + 30 min recording) for faster runs.

Outputs:
  1. Scatter: FF speed (30 obs) vs benchmark for P10/P50/P90
  2. Scatter: Cap speed (15 obs) vs benchmark
  3. Slope distribution: Monte Carlo (avg-of-3 FF × 1 Cap) with benchmarks

Usage:
  python validation_30runs.py
"""

import os
import sys
import json
import datetime
import csv
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import path_setup

from calibration_config import (
    BENCHMARKS, FLOW_FREE_FLOW, FLOW_CONGESTED, EDGE_FLOW_FACTOR,
)

# Override simulation duration BEFORE importing run_single_flow
# 90 min total = 60 min warmup + 30 min recording
import calibration_2point_runner as runner_mod
runner_mod.SIM_DURATION_SEC = 5400       # 90 min
runner_mod.SIM_STEPS = int(5400 / 0.1)   # 54000 steps
# WARMUP_STEPS stays at 36000 (60 min) — unchanged

from calibration_2point_runner import run_single_flow

import sumolib

# Constants
N_RUNS_FF = 30
N_RUNS_CAP = 15
N_BOOTSTRAP = 10000
FLOW_DIFF_PER_1000 = (FLOW_CONGESTED - FLOW_FREE_FLOW) * EDGE_FLOW_FACTOR / 1000.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PCTS = ["P10", "P50", "P90"]


def run_with_retry(sumo_bin, flow, max_retries=3):
    """Run a single flow with retry logic for transient SUMO errors."""
    for attempt in range(max_retries):
        try:
            result = run_single_flow(sumo_bin, flow)
            return result
        except Exception as e:
            print(f"  ERROR (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Brief pause before retry
            else:
                raise


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    sumo_bin = sumolib.checkBinary("sumo")

    # ================================================================
    # Run 30 FF simulations
    # ================================================================
    ff_results = []
    print("=" * 60)
    print(f"PHASE 1: {N_RUNS_FF} Free-Flow simulations ({FLOW_FREE_FLOW} veh/h)")
    print(f"  Sim: 90 min (60 warmup + 30 recording)")
    print("=" * 60)

    for i in range(N_RUNS_FF):
        print(f"\n--- FF Run {i+1}/{N_RUNS_FF} ---")
        speeds = run_with_retry(sumo_bin, FLOW_FREE_FLOW)
        ff_results.append(speeds)
        print(f"  P10={speeds['P10']:.1f}, P50={speeds['P50']:.1f}, P90={speeds['P90']:.1f}, n={speeds['n_obs']}")

    # ================================================================
    # Run 15 Cap simulations
    # ================================================================
    cap_results = []
    print("\n" + "=" * 60)
    print(f"PHASE 2: {N_RUNS_CAP} Capacity simulations ({FLOW_CONGESTED} veh/h)")
    print(f"  Sim: 90 min (60 warmup + 30 recording)")
    print("=" * 60)

    for i in range(N_RUNS_CAP):
        print(f"\n--- Cap Run {i+1}/{N_RUNS_CAP} ---")
        speeds = run_with_retry(sumo_bin, FLOW_CONGESTED)
        cap_results.append(speeds)
        print(f"  P10={speeds['P10']:.1f}, P50={speeds['P50']:.1f}, P90={speeds['P90']:.1f}, n={speeds['n_obs']}")

    # ================================================================
    # Save raw data
    # ================================================================
    raw_path = os.path.join(RESULTS_DIR, f"validation_30runs_{timestamp}.json")
    with open(raw_path, "w") as f:
        json.dump({"ff_results": ff_results, "cap_results": cap_results}, f, indent=2)
    print(f"\nRaw data saved: {raw_path}")

    csv_path = os.path.join(RESULTS_DIR, f"validation_30runs_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "flow_type", "P10", "P50", "P90", "n_obs"])
        for i, r in enumerate(ff_results):
            writer.writerow([i+1, "FF", r["P10"], r["P50"], r["P90"], r["n_obs"]])
        for i, r in enumerate(cap_results):
            writer.writerow([i+1, "Cap", r["P10"], r["P50"], r["P90"], r["n_obs"]])
    print(f"CSV saved: {csv_path}")

    # ================================================================
    # Print summary statistics
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for pct in PCTS:
        ff_vals = [r[pct] for r in ff_results]
        cap_vals = [r[pct] for r in cap_results]
        bench_ff = BENCHMARKS[pct]["free_flow_speed"]

        print(f"\n  {pct} FF Speed (bench={bench_ff:.1f}):")
        print(f"    Mean={np.mean(ff_vals):.1f}, Std={np.std(ff_vals):.2f}, "
              f"Min={np.min(ff_vals):.1f}, Max={np.max(ff_vals):.1f}")

        print(f"  {pct} Cap Speed:")
        print(f"    Mean={np.mean(cap_vals):.1f}, Std={np.std(cap_vals):.2f}, "
              f"Min={np.min(cap_vals):.1f}, Max={np.max(cap_vals):.1f}")

    # ================================================================
    # Monte Carlo slope distribution
    # ================================================================
    print("\n" + "=" * 60)
    print(f"MONTE CARLO SLOPE DISTRIBUTION ({N_BOOTSTRAP} samples)")
    print("=" * 60)

    slopes = {pct: [] for pct in PCTS}

    for _ in range(N_BOOTSTRAP):
        # Pick 3 random FF runs, average their speeds
        ff_idx = np.random.choice(N_RUNS_FF, size=3, replace=False)
        # Pick 1 random Cap run
        cap_idx = np.random.randint(N_RUNS_CAP)

        for pct in PCTS:
            ff_avg = np.mean([ff_results[i][pct] for i in ff_idx])
            cap_speed = cap_results[cap_idx][pct]
            slope = (cap_speed - ff_avg) / FLOW_DIFF_PER_1000
            slopes[pct].append(slope)

    for pct in PCTS:
        bench_slope = BENCHMARKS[pct]["slope"]
        s = slopes[pct]
        print(f"\n  {pct} Slope (bench={bench_slope:.1f}):")
        print(f"    Mean={np.mean(s):.2f}, Std={np.std(s):.2f}")
        print(f"    5th={np.percentile(s,5):.2f}, Median={np.median(s):.2f}, 95th={np.percentile(s,95):.2f}")

    # ================================================================
    # Generate plots
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Validation: Van Aerde Calibration (30 FF + 15 Cap)", fontsize=14, fontweight="bold")

        colors = {"P10": "#2196F3", "P50": "#4CAF50", "P90": "#FF9800"}

        # Row 1: FF and Cap speed scatter
        for col, pct in enumerate(PCTS):
            ax = axes[0, col]
            ff_vals = [r[pct] for r in ff_results]
            cap_vals = [r[pct] for r in cap_results]
            bench_ff = BENCHMARKS[pct]["free_flow_speed"]

            ff_runs = np.arange(1, N_RUNS_FF + 1)
            cap_runs = np.arange(1, N_RUNS_CAP + 1)
            ax.scatter(ff_runs, ff_vals, color=colors[pct], alpha=0.7, label=f"FF ({np.mean(ff_vals):.1f}±{np.std(ff_vals):.1f})", zorder=3)
            ax.scatter(cap_runs, cap_vals, color=colors[pct], alpha=0.4, marker="s", label=f"Cap ({np.mean(cap_vals):.1f}±{np.std(cap_vals):.1f})", zorder=3)
            ax.axhline(bench_ff, color="red", linestyle="--", linewidth=1.5, label=f"Bench FF={bench_ff}")
            ax.fill_between([0, N_RUNS_FF+1], bench_ff*0.95, bench_ff*1.05, alpha=0.1, color="red")
            ax.set_title(f"{pct} Speeds", fontweight="bold")
            ax.set_xlabel("Run #")
            ax.set_ylabel("Speed (km/h)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, N_RUNS_FF + 1)

        # Row 2: Slope distributions (histograms)
        for col, pct in enumerate(PCTS):
            ax = axes[1, col]
            bench_slope = BENCHMARKS[pct]["slope"]
            s = slopes[pct]

            ax.hist(s, bins=50, color=colors[pct], alpha=0.7, edgecolor="white", density=True)
            ax.axvline(bench_slope, color="red", linestyle="--", linewidth=2, label=f"Bench={bench_slope:.1f}")
            ax.axvline(np.mean(s), color="black", linestyle="-", linewidth=1.5, label=f"Mean={np.mean(s):.2f}")
            ax.set_title(f"{pct} Slope Distribution (n={N_BOOTSTRAP})", fontweight="bold")
            ax.set_xlabel("Slope (km/h per 1000 veh/h)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"validation_30runs_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved: {plot_path}")
        plt.close()

    except ImportError:
        print("\nWARNING: matplotlib not available, skipping plots")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
