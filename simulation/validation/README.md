# Phase 2: Physics Re-Validation

Re-validates network physics with extended simulation (120 min, 60 min warmup) across 7 flow rates on the short path only.

## Flow Rates
500, 1000, 1500, 2000, 2500, 3000, 3500 veh/h

## Measurements
- Speed vs time (P10/P50/P90 per minute)
- Slope (speed-flow sensitivity via linear regression)
- Van Aerde benchmark comparison

## Runner
`runners/physics_validation_runner.py`

## Results (2026-03-14)

### Post-Warmup Speed (km/h, average of minutes 61-120)

| Flow | P10 | P50 | P90 |
|------|-----|-----|-----|
| 500 | 120.8 | 132.4 | 143.9 |
| 1000 | 94.5 | 110.2 | 142.6 |
| 1500 | 92.4 | 96.5 | 126.4 |
| 2000 | 89.7 | 91.4 | 113.2 |
| 2500 | 89.7 | 91.0 | 113.3 |
| 3000 | 89.6 | 90.7 | 109.7 |
| 3500 | 89.6 | 90.6 | 110.6 |

### Slopes (km/h per 1000 veh/h)

| Percentile | Measured | Benchmark | Deviation | Status |
|------------|----------|-----------|-----------|--------|
| P10 | -7.6 | -5.5 | 38.0% | High |
| P50 | -12.1 | -9.1 | 33.3% | High |
| P90 | -12.8 | -13.4 | 4.7% | PASS |

### Free-Flow Speed (500 veh/h)

| Percentile | Measured | Benchmark | Deviation |
|------------|----------|-----------|-----------|
| P10 | 120.8 | 89.1 | +35.6% |
| P50 | 132.4 | 99.5 | +33.0% |
| P90 | 143.9 | 112.4 | +28.0% |

### Interpretation

1. **Free-flow speeds higher than benchmarks**: Expected. Van Aerde benchmarks are from Ontario 2-lane highways with real-world friction. The thesis network geometry (7.5km semicircle, 40 m/s speed limit) allows higher free-flow speeds. Calibration was done on a separate 3km single-edge network.

2. **Speed-flow relationship is valid**: Speeds decrease monotonically with increasing flow. P10/P50 converge to ~90 km/h at 2000+ veh/h, showing congestion onset. P90 retains more speed (fast drivers overtake).

3. **P90 slope passes benchmark**: The speed sensitivity of fast drivers (-12.8 vs -13.4 benchmark) is realistic, within 5%.

4. **P10/P50 slopes steeper than benchmark**: Slow and median drivers lose speed faster with increasing flow than the benchmark predicts. This suggests the network geometry (curves, single lane with overtaking) creates more interaction effects for slower drivers.

5. **Network capacity**: Speed remains stable at 2500-3500 veh/h (no gridlock), confirming the network can handle the flow range needed for Phases 3-8.

### Output Files
- `physics_timeseries_flow{FLOW}_*.csv` — per-minute P10/P50/P90 speeds
- `physics_per_vehicle_flow{FLOW}_*.csv` — individual vehicle records
- `physics_summary_*.csv` — combined summary with slopes
- `speed_vs_time_flow{FLOW}_*.png` — speed time-series (warmup shaded)
- `slope_vs_flow_summary_*.png` — slope comparison chart
