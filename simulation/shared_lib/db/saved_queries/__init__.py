"""
Reusable query library.

All DB queries used by analysis scripts MUST be saved here as functions.
Before writing a new query, search this package for existing matches.

Modules:
    vehicle_queries     — Per-vehicle aggregations (avg TT, path split, etc.)
    timeseries_queries  — Per-minute aggregations (gap convergence, EMA trace)
    comparison_queries  — Cross-method comparisons (PoA, gap closing)
    linearity_queries   — Cost function regression, breakpoint detection
    welfare_queries     — VOT, toll, generalized cost (Phase 5+)
"""
