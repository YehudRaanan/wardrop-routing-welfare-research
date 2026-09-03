"""
Calibration Configuration
=========================
Single source of truth for all calibration parameters and benchmarks.

Source: Van Aerde & Yagar 1983, Site 400S1
Reference: calibration_guidelines.md

This module is read by:
- project_guardrails.py (for validation)
- simulation_shared.py (for parameter values)
- calibration_runner_open.py (for benchmarks)
- phase1a_runner.py (for benchmarks)

==============================================================================
LOCKED CALIBRATION - 2026-03-19
==============================================================================
Parameters below have been calibrated and LOCKED. DO NOT MODIFY without
explicit user approval and full regression testing.

Network: thesis_dual_route_18022026
Runner: 03_PhysicsValidation/runners/calibration_2point_runner.py
Method: Per-vehicle mean speeds (each vehicle counted once)

Final Benchmark Results (500/5000 veh/h, slope denominator 1.755):
┌────────┬─────────────────────────────────┬─────────────────────────────────┐
│        │ Free-Flow Speed                 │ Slope (km/h per 1000 veh/h)    │
│        │ Benchmark → Measured (Error)    │ Benchmark → Measured (Error)   │
├────────┼─────────────────────────────────┼─────────────────────────────────┤
│ P10    │ 89.1 → 88.1 km/h (-1.1%) PASS  │ -5.5 → -5.9 (+7.6%)           │
│ P50    │ 99.5 → 100.4 km/h (+0.9%) PASS │ -9.1 → -8.8 (-3.7%) PASS      │
│ P90    │ 112.4 → 109.4 km/h (-2.6%) PASS│ -13.4 → -10.7 (-20.3%)        │
└────────┴─────────────────────────────────┴─────────────────────────────────┘

4/6 benchmarks pass (±5%). All FF speeds pass. P50 slope passes.
P10 slope slightly steep (7.6%), P90 slope too shallow (20.3%).
Acceptable for thesis work — FF speed accuracy is excellent.
==============================================================================
"""

import math

# =============================================================================
# BENCHMARK TARGETS (Van Aerde & Yagar 1983, Site 400S1)
# =============================================================================

BENCHMARKS = {
    "P10": {"free_flow_speed": 89.1, "slope": -5.5},
    "P50": {"free_flow_speed": 99.5, "slope": -9.1},
    "P90": {"free_flow_speed": 112.4, "slope": -13.4},
}

# Tolerance for validation (percentage deviation allowed)
TOLERANCE_PERCENT = 10.0  # ±10% from benchmark is acceptable

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Traffic Split (Forward/Opposing)
FWD_RATIO = 0.78
BWD_RATIO = 0.22

# Discrete Choice Modeling (Logit) Parameter
LOGIT_THETA = 0.05  # Default scale parameter (1/seconds)

# Car Speed Distribution [LOCKED 2026-03-19]
MU_KMH = 99.5        # Mean speed (km/h)
SIGMA_KMH = 10.0     # Standard deviation (km/h) - matches Van Aerde benchmark spread
SPEED_OFFSET_KMH = 5.0  # LOCKED 2026-05-08 (was 10.0 prior to smoke 3)

# Following Time (Tau) - Speed-Dependent (Van Aerde calibrated) [LOCKED 2026-03-19]
# Fast drivers (P90) need low TAU to maintain higher cap speeds
# Slow drivers (P10) need high TAU to create congestion
TAU_MIN = 0.0        # Safety clamp only — never triggered with current params (min tau ≈ 1.0 at 120 km/h)
# 2026-05-08 SMOKE TEST: shallowing slopes to match Van Aerde benchmark.
# Locked values were TAU_MAX=3.2, TAU_GRADIENT=0.055 (LOCKED 2026-03-19).
# Heuristic (slope ∝ tau): aim P10 slope -7.53→-5.5, P50 -12.94→-9.1, P90 -16.84→-13.4.
TAU_MAX = 3.2        # LOCKED (2026-05-08 re-confirmed at smoke 5 values)
TAU_GRADIENT = 0.055 # LOCKED (2026-05-08 re-confirmed at smoke 5 values)

def calculate_tau(speed_kmh):
    """
    Calculate tau (following time) based on speed.

    Formula: TAU = TAU_MAX - TAU_GRADIENT * (speed_kmh - 80)
    - Fast drivers (high speed) -> lower tau (follow closer)
    - Slow drivers (low speed) -> higher tau (more spacing)
    """
    tau = TAU_MAX - TAU_GRADIENT * (speed_kmh - 80)
    return max(TAU_MIN, min(TAU_MAX, tau))

# Lane Change Parameters - Aggressive overtaking (iter 26: restore davis values)
LC_OPPOSITE = 1.0    # Opposite-lane overtaking enabled
LC_SPEED_GAIN = 10.0 # High — actively seek overtaking
LC_ASSERTIVE = 5.0   # Aggressive
LC_IMPATIENCE = 1.0  # Impatient

# Truck Parameters — iter 26: slow trucks as moving bottlenecks (Van Aerde 3%)
TRUCK_SPEED_MEAN = 90   # LOCKED 2026-05-08 (was 85 prior to smoke 5)
TRUCK_SPEED_STD = 5     # km/h
TRUCK_SPEED_MAX = 95    # LOCKED 2026-05-08 (was 90 prior to smoke 5)
TRUCK_PROBABILITY = 0.03  # 3% (Van Aerde)

# =============================================================================
# PHASE 2: VOT & ECONOMICS (UK 2025 Data)
# =============================================================================

# UK Income Distribution (2025 Provisional, Gross Annual GBP)
UK_INCOME_MEDIAN_GBP = 37430
UK_INCOME_P90_GBP = 72150

# Log-Normal Parameters (Derived)
# Mu = ln(Median)
LN_MU = math.log(UK_INCOME_MEDIAN_GBP)  # ~10.530
# Sigma = (ln(P90) - Mu) / 1.28155
LN_SIGMA = (math.log(UK_INCOME_P90_GBP) - LN_MU) / 1.28155  # ~0.512

# Currency conversion
EXCHANGE_RATE_USD_GBP = 1.34  # Forecast 31/12/2025

# Value of Time Coefficients (Wardman et al., 2016)
# Multipliers on Gross Hourly Income (Annual / 2000)
VOT_COEFF_BUSINESS = 1.20
VOT_COEFF_COMMUTE = 0.45
VOT_COEFF_OTHER = 0.35

# =============================================================================
# SLOPE METHODOLOGY
# =============================================================================

FLOW_FREE_FLOW = 500     # veh/h (free-flow measurement point) — iter 8: back to 500, using per-vehicle means for stability
FLOW_CONGESTED = 5000    # veh/h (network breakpoint — phase transition at ~5000 veh/h)

EDGE_FLOW_FACTOR = FWD_RATIO / 2  # 0.39 — accounts for directional throughput at capacity

def calculate_slope(ff_speed, congested_speed):
    """Calculate slope in km/h per 1000 veh/h (Van Aerde Model II, main-direction PCU)."""
    flow_diff = (FLOW_CONGESTED - FLOW_FREE_FLOW) * EDGE_FLOW_FACTOR / 1000.0  # 1.755
    return (congested_speed - ff_speed) / flow_diff

# =============================================================================
# NETWORK GEOMETRY (Phase 0)
# =============================================================================

EDGE_LENGTH_M = 3000     # meters
LANES_PER_DIRECTION = 1  # 2-lane highway (1 per direction)
SPEED_LIMIT_MS = 40      # m/s (144 km/h)
LANE_WIDTH_M = 3.5       # meters

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_result(percentile, metric, measured_value):
    """
    Validate a measured value against the benchmark.
    
    Args:
        percentile: "P10", "P50", or "P90"
        metric: "free_flow_speed" or "slope"
        measured_value: The measured value to validate
    
    Returns:
        dict with {valid: bool, benchmark: float, deviation_percent: float}
    """
    if percentile not in BENCHMARKS:
        return {"valid": False, "error": f"Unknown percentile: {percentile}"}
    
    benchmark = BENCHMARKS[percentile][metric]
    
    if benchmark == 0:
        deviation_percent = 0 if measured_value == 0 else float('inf')
    else:
        deviation_percent = abs((measured_value - benchmark) / benchmark) * 100
    
    is_valid = deviation_percent <= TOLERANCE_PERCENT
    
    return {
        "valid": is_valid,
        "benchmark": benchmark,
        "measured": measured_value,
        "deviation_percent": round(deviation_percent, 2),
        "tolerance_percent": TOLERANCE_PERCENT
    }


def validate_all_results(results_dict):
    """
    Validate a full set of results against all benchmarks.
    
    Args:
        results_dict: {
            "P10": {"free_flow_speed": X, "slope": Y},
            "P50": {"free_flow_speed": X, "slope": Y},
            "P90": {"free_flow_speed": X, "slope": Y}
        }
    
    Returns:
        dict with validation summary and all_valid flag
    """
    validations = {}
    all_valid = True
    
    for pct in ["P10", "P50", "P90"]:
        if pct not in results_dict:
            validations[pct] = {"error": "Missing percentile data"}
            all_valid = False
            continue
        
        validations[pct] = {}
        for metric in ["free_flow_speed", "slope"]:
            if metric not in results_dict[pct]:
                validations[pct][metric] = {"error": "Missing metric"}
                all_valid = False
                continue
            
            result = validate_result(pct, metric, results_dict[pct][metric])
            validations[pct][metric] = result
            if not result.get("valid", False):
                all_valid = False
    
    return {
        "all_valid": all_valid,
        "validations": validations
    }


def print_validation_report(results_dict):
    """Print a formatted validation report to console."""
    validation = validate_all_results(results_dict)
    
    print("\n" + "="*70)
    print("  BENCHMARK VALIDATION REPORT")
    print("  Source: Van Aerde & Yagar 1983, Site 400S1")
    print("="*70)
    
    for pct in ["P10", "P50", "P90"]:
        print(f"\n  {pct}:")
        if pct not in validation["validations"]:
            print(f"    ERROR: Missing data")
            continue
        
        for metric, result in validation["validations"][pct].items():
            if "error" in result:
                status = "ERROR"
                detail = result["error"]
            elif result["valid"]:
                status = "PASS"
                detail = f"{result['measured']:.1f} vs {result['benchmark']:.1f} ({result['deviation_percent']:.1f}%)"
            else:
                status = "FAIL"
                detail = f"{result['measured']:.1f} vs {result['benchmark']:.1f} ({result['deviation_percent']:.1f}% > {result['tolerance_percent']}%)"
            
            label = "FF Speed" if metric == "free_flow_speed" else "Slope"
            print(f"    {label:12}: {status} - {detail}")
    
    print("\n" + "-"*70)
    if validation["all_valid"]:
        print("  OVERALL: ALL BENCHMARKS PASSED")
    else:
        print("  OVERALL: SOME BENCHMARKS FAILED")
    print("="*70 + "\n")
    
    return validation["all_valid"]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing calibration_config.py...\n")
    
    # Test tau calculation
    print("1. Tau Calculation:")
    for speed in [80, 90, 100, 110, 120]:
        tau = calculate_tau(speed)
        print(f"   Speed {speed} km/h → Tau {tau:.2f}s")
    
    # Test slope calculation
    print("\n2. Slope Calculation:")
    slope = calculate_slope(99.5, 90.4)
    print(f"   FF=99.5, Cap=90.4 → Slope={slope:.2f}")
    
    # Test Phase 2 Parameters
    print("\n3. Phase 2 Parameters (UK 2025):")
    print(f"   Median Income: £{UK_INCOME_MEDIAN_GBP}")
    print(f"   Log-Normal Mu: {LN_MU:.3f}")
    print(f"   Log-Normal Sigma: {LN_SIGMA:.3f}")
    print(f"   Exchange Rate: {EXCHANGE_RATE_USD_GBP} USD/GBP")
