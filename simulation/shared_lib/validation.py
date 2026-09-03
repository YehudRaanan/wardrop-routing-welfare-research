"""
Automated pre-run validation checks.

All validation functions return lists of error/warning strings.
Empty list = all checks passed.

These are called by project_guardrails.require_approval() BEFORE
showing the approval prompt. If validation fails, the detailed
explanation is shown and the user can choose to override.

IMPORTANT: Validation does NOT abort automatically. It returns errors
and the caller decides how to handle them (show + ask for override).
"""

from typing import Any

# ─────────────────────────────────────────────────────────────────────
# SIMULATION STANDARDS (LOCKED)
# ─────────────────────────────────────────────────────────────────────
# These values are the project standard. Any deviation is flagged.
# Source: paper/main.tex Methodology and calibration_config.py.

LOCKED_SIM_DURATION_SEC = 7200
LOCKED_WARMUP_SEC = 3600
LOCKED_STEP_LENGTH = 0.2  # 2026-04-16: 0.1→0.2 with libsumo migration
LOCKED_BUCKET_SEC = 60
LOCKED_FWD_RATIO = 0.78
LOCKED_BWD_RATIO = 0.22

# Known valid body edges (from network thesis_dual_route_18022026)
VALID_BODY_EDGES = frozenset({
    "short_fwd", "long_fwd", "short_bwd", "long_bwd",
})

# Entry edges that MUST NOT appear in route definitions
BANNED_ENTRY_EDGES = frozenset({
    "entry_short_fwd", "entry_long_fwd",
    "entry_short_bwd", "entry_long_bwd",
})

# Valid route choice methods
VALID_METHODS = frozenset({
    "ema", "davis", "davis_threshold", "mediated", "fixed_split", "fixed_split_vot", "thompson", "none",
    "habit",
    "mixed",  # 2026-05-09: P2 method-mix populations (per-vehicle dispatch via config.method_mix)
})


# ─────────────────────────────────────────────────────────────────────
# ROUTE VALIDATION
# ─────────────────────────────────────────────────────────────────────

def validate_routes(routes: dict[str, list[str]]) -> list[str]:
    """Validate route edge definitions.

    Checks:
    1. No entry edges (entry_short_fwd, etc.) — these create bottlenecks
    2. All edges are known valid body edges
    3. Each route has at least one edge

    Args:
        routes: Dict mapping route_id -> list of edge IDs

    Returns:
        List of error strings (empty if valid)
    """
    errors = []

    for route_id, edges in routes.items():
        if not edges:
            errors.append(f"Route '{route_id}' is empty — must have at least one edge")
            continue

        for edge in edges:
            if edge in BANNED_ENTRY_EDGES:
                errors.append(
                    f"Route '{route_id}' contains BANNED entry edge '{edge}'. "
                    f"Entry edges create artificial 2→1 lane merge bottlenecks. "
                    f"Use body edge only: {VALID_BODY_EDGES}"
                )
            elif edge not in VALID_BODY_EDGES:
                errors.append(
                    f"Route '{route_id}' contains unknown edge '{edge}'. "
                    f"Valid edges: {VALID_BODY_EDGES}"
                )

    return errors


# ─────────────────────────────────────────────────────────────────────
# RUN CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────

def validate_run_config(config: Any) -> list[str]:
    """Validate a RunConfig against locked simulation standards.

    Checks locked parameters against paper/main.tex and calibration_config.py.
    Checks method-specific requirements (e.g., EMA needs alpha).

    Args:
        config: A RunConfig dataclass or dict with the same keys

    Returns:
        List of error strings (empty if valid)
    """
    errors = []

    # Support both dataclass and dict
    def _get(key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    # Check locked simulation envelope
    sim_dur = _get("sim_duration_sec", LOCKED_SIM_DURATION_SEC)
    if sim_dur != LOCKED_SIM_DURATION_SEC:
        errors.append(
            f"sim_duration_sec={sim_dur} differs from LOCKED standard "
            f"{LOCKED_SIM_DURATION_SEC}. Source: paper/main.tex Methodology."
        )

    warmup = _get("warmup_sec", LOCKED_WARMUP_SEC)
    if warmup != LOCKED_WARMUP_SEC:
        errors.append(
            f"warmup_sec={warmup} differs from LOCKED standard "
            f"{LOCKED_WARMUP_SEC}. Source: paper/main.tex Methodology."
        )

    step = _get("step_length", LOCKED_STEP_LENGTH)
    if abs(step - LOCKED_STEP_LENGTH) > 1e-6:
        errors.append(
            f"step_length={step} differs from LOCKED standard "
            f"{LOCKED_STEP_LENGTH}. Source: paper/main.tex Methodology."
        )

    fwd = _get("fwd_ratio", LOCKED_FWD_RATIO)
    if abs(fwd - LOCKED_FWD_RATIO) > 0.001:
        errors.append(
            f"fwd_ratio={fwd} differs from LOCKED standard "
            f"{LOCKED_FWD_RATIO}. Source: calibration_config.py."
        )

    bwd = _get("bwd_ratio", LOCKED_BWD_RATIO)
    if abs(bwd - LOCKED_BWD_RATIO) > 0.001:
        errors.append(
            f"bwd_ratio={bwd} differs from LOCKED standard "
            f"{LOCKED_BWD_RATIO}. Source: calibration_config.py."
        )

    # Check method validity
    method = _get("route_choice_method")
    if method and method not in VALID_METHODS:
        errors.append(
            f"route_choice_method='{method}' is not a valid method. "
            f"Valid: {VALID_METHODS}"
        )

    # Method-specific checks
    if method == "ema" and _get("ema_alpha") is None:
        errors.append("EMA method requires ema_alpha to be set")

    if method in ("fixed_split", "fixed_split_vot") and _get("pct_short_fixed") is None:
        errors.append(f"{method} method requires pct_short_fixed to be set")

    # Flow rate basic check
    flow = _get("flow_rate")
    if flow is not None:
        if flow <= 0:
            errors.append(f"flow_rate={flow} must be positive")
        if flow % 100 != 0:
            errors.append(
                f"flow_rate={flow} should be a multiple of 100 veh/h"
            )

    return errors


# ─────────────────────────────────────────────────────────────────────
# FLOW RATE VALIDATION
# ─────────────────────────────────────────────────────────────────────

def validate_flow_rates(
    flows: list[int],
    context: str = "",
) -> list[str]:
    """Validate a list of flow rates for a sweep.

    This is a WARNING-level check, not an error — flow ranges
    are per-goal and may legitimately vary.

    Checks:
    - All flows are positive multiples of 100
    - Step size is consistent (all same delta)
    - No duplicates

    Args:
        flows: List of flow rates in veh/h
        context: Description of the sweep goal (for warning messages)

    Returns:
        List of warning strings (empty if valid)
    """
    warnings = []

    if not flows:
        warnings.append(f"Empty flow list{f' for {context}' if context else ''}")
        return warnings

    # Check multiples of 100
    non_100 = [f for f in flows if f % 100 != 0]
    if non_100:
        warnings.append(
            f"Flows not multiples of 100: {non_100}"
        )

    # Check positive
    non_positive = [f for f in flows if f <= 0]
    if non_positive:
        warnings.append(f"Non-positive flows: {non_positive}")

    # Check duplicates
    if len(flows) != len(set(flows)):
        warnings.append(f"Duplicate flows detected in {flows}")

    # Check consistent step size
    if len(flows) >= 2:
        sorted_flows = sorted(flows)
        deltas = [sorted_flows[i+1] - sorted_flows[i] for i in range(len(sorted_flows)-1)]
        unique_deltas = set(deltas)
        if len(unique_deltas) > 1:
            warnings.append(
                f"Inconsistent step sizes: {unique_deltas}. "
                f"Expected uniform 500 or 100 veh/h steps."
            )

    return warnings
