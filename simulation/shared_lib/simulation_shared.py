"""
Simulation Shared Logic
-----------------------
Goal:
    Centralizes vehicle parameters, type definitions, and injection logic to ensure consistency
    across Calibration, Free-Flow Visualization, and Capacity Visualization.

Core Definitions:
    - Speeds: Normal Distribution N(MU, SIGMA) from calibration_config.py
    - Groups: 5 Car Quintiles (Very Slow to Very Fast) + 1 Truck Group.
    - Colors: Mapped to groups for GUI visualization (Red->Green, White).
    - Lane Changing: Dynamic 'KeepRight' and 'Cooperative' logic based on speed.
"""

import os
import sys

# SUMO can be installed system-wide or exposed through SUMO_HOME.

import libsumo as traci
import random
import scipy.stats as stats

# Import parameters from central config (single source of truth)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration_config import (
    MU_KMH, SIGMA_KMH, SPEED_OFFSET_KMH,
    TAU_MIN, TAU_MAX, calculate_tau,
    LC_OPPOSITE, LC_SPEED_GAIN, LC_ASSERTIVE, LC_IMPATIENCE,
    TRUCK_SPEED_MEAN, TRUCK_SPEED_STD, TRUCK_SPEED_MAX, TRUCK_PROBABILITY,
    SPEED_LIMIT_MS
)

# Derived constant for speed factor calculation
EDGE_SPEED_LIMIT_KMH = SPEED_LIMIT_MS * 3.6  # 144 km/h

# Quintile Definitions (0-20%, 20-40%, etc.)
QUINTILES = {
    "q1": {"name": "Very Slow", "range": (0.0, 0.20), "color": (255, 0, 0)},      # Red
    "q2": {"name": "Slow",      "range": (0.20, 0.40), "color": (255, 128, 0)},    # Orange
    "q3": {"name": "Medium",    "range": (0.40, 0.60), "color": (255, 255, 0)},    # Yellow
    "q4": {"name": "Fast",      "range": (0.60, 0.80), "color": (128, 255, 0)},    # Lime
    "q5": {"name": "Very Fast", "range": (0.80, 1.00), "color": (0, 255, 0)}       # Green
}

# Truck Definition
TRUCK_GROUP = {"name": "Trucks", "color": (255, 255, 255)} # White

def setup_vehicle_types():
    """
    Defines the vehicle types used in the simulation.
    These parameters are SHARED across calibration and visualization.
    """
    # 1. Base Car Type
    base_type = "base_driver"
    traci.vehicletype.copy("DEFAULT_VEHTYPE", base_type)
    traci.vehicletype.setLength(base_type, 5.0)
    traci.vehicletype.setMinGap(base_type, 1.5)
    traci.vehicletype.setImperfection(base_type, 0.3)  # Reduced for FF speed calibration
    traci.vehicletype.setSpeedDeviation(base_type, 0.0)  # Disabled - we control speed via speedFactor
    traci.vehicletype.setMaxSpeed(base_type, 70.0) # 252 km/h approx
    
    # Lane Change Parameters - Very Aggressive Overtaking
    traci.vehicletype.setParameter(base_type, "lcOpposite", "1.0")
    traci.vehicletype.setParameter(base_type, "lcStrategic", "1.0")
    traci.vehicletype.setParameter(base_type, "lcLookaheadLeft", "10.0")  # Look far ahead
    traci.vehicletype.setParameter(base_type, "lcKeepRight", "0.0")  # No pressure to keep right
    traci.vehicletype.setParameter(base_type, "lcCooperative", "0.0")  # Not cooperative = selfish
    traci.vehicletype.setParameter(base_type, "lcSpeedGain", "10.0")  # Aggressive overtaking (iter 26)
    traci.vehicletype.setParameter(base_type, "lcAssertive", "5.0")  # Aggressive
    traci.vehicletype.setParameter(base_type, "lcOvertakeDeltaSpeedFactor", "0.2")  # Low threshold to overtake

    # 2. Truck Type
    truck_type = "truck_driver"
    traci.vehicletype.copy("DEFAULT_VEHTYPE", truck_type)
    traci.vehicletype.setLength(truck_type, 12.0)
    traci.vehicletype.setMinGap(truck_type, 2.5)
    traci.vehicletype.setImperfection(truck_type, 0.3)  # Reduced for FF speed calibration
    traci.vehicletype.setSpeedDeviation(truck_type, 0.0)  # Disabled - we control speed via speedFactor
    traci.vehicletype.setMaxSpeed(truck_type, 25.0) # 90 km/h
    
    # Trucks do NOT overtake via opposite lane
    traci.vehicletype.setParameter(truck_type, "lcOpposite", "0.0") 
    traci.vehicletype.setParameter(truck_type, "lcKeepRight", "1.0") 

    return base_type, truck_type

def get_group_from_speed_percentile(p_val):
    if p_val <= 0.2: return "q1"
    if p_val <= 0.4: return "q2"
    if p_val <= 0.6: return "q3"
    if p_val <= 0.8: return "q4"
    return "q5"

def inject_vehicle(vid, route, vtype_car, vtype_truck):
    """
    Injects a vehicle and returns its assigned properties for analysis.
    Returns: dict with keys {group_id, assigned_kmh, assigned_tau, assigned_sigma}
    
    Uses parameters from calibration_config.py (single source of truth).
    """
    try:
        # Determine Type: Truck probability from config
        is_truck = random.random() < TRUCK_PROBABILITY
        my_type = vtype_truck if is_truck else vtype_car
        
        # Add vehicle to simulation
        traci.vehicle.add(vid, route, typeID=my_type, depart="now", departSpeed="max", departPos="free")
        
        props = {}

        if is_truck:
            # --- Truck Logic (using config values) ---
            assigned_kmh = random.normalvariate(TRUCK_SPEED_MEAN, TRUCK_SPEED_STD)
            assigned_kmh += SPEED_OFFSET_KMH  # LOCKED 2026-05-08: same offset as cars
            if assigned_kmh > TRUCK_SPEED_MAX: assigned_kmh = TRUCK_SPEED_MAX
            if assigned_kmh < 60: assigned_kmh = 60

            s_factor = assigned_kmh / EDGE_SPEED_LIMIT_KMH  # Same formula as cars
            assigned_tau = 1.3
            assigned_sigma = 0.3

            props = {
                "group_id": "truck",
                "assigned_kmh": assigned_kmh,
                "assigned_tau": assigned_tau,
                "assigned_sigma": assigned_sigma,
                "type": "truck"
            }

            # Set Color
            traci.vehicle.setColor(vid, TRUCK_GROUP["color"])

        else:
            # --- Car Logic ---
            # Sample Percentile (raw value for later P10/P50/P90 analysis)
            p_val = random.random()
            
            # Map percentile to Normal Distribution Speed
            # ppf = Percent Point Function (Inverse CDF)
            assigned_kmh = stats.norm.ppf(p_val, loc=MU_KMH, scale=SIGMA_KMH)
            
            # Apply speed offset to compensate for simulation-to-reality gap
            assigned_kmh += SPEED_OFFSET_KMH
            
            # Clipping (minimum reasonable speed)
            if assigned_kmh < 60: assigned_kmh = 60  # Reverted to original
            
            # Store raw percentile for post-simulation P10/P50/P90 calculation
            group_id = get_group_from_speed_percentile(p_val)  # Keep for color coding
            
            s_factor = assigned_kmh / EDGE_SPEED_LIMIT_KMH
            
            # Dynamic Tau using centralized config formula
            assigned_tau = calculate_tau(assigned_kmh)
            
            assigned_sigma = 0.3
            
            # All drivers use same lane-changing behavior (no special threshold)

            props = {
                "group_id": group_id,
                "assigned_kmh": assigned_kmh,
                "assigned_tau": assigned_tau,
                "assigned_sigma": assigned_sigma,
                "type": "car"
            }

            # Set Color
            traci.vehicle.setColor(vid, QUINTILES[group_id]["color"])

        # Apply parameters
        # IMPORTANT: setTau/setImperfection reset speedFactor to type default (1.0)
        # in SUMO's TraCI. setSpeedFactor MUST be called LAST.
        traci.vehicle.setTau(vid, assigned_tau)
        traci.vehicle.setImperfection(vid, props["assigned_sigma"])
        traci.vehicle.setSpeedFactor(vid, s_factor)
        
        # Enable aggressive overtaking (using config values)
        traci.vehicle.setParameter(vid, "lcOpposite", str(LC_OPPOSITE))
        traci.vehicle.setParameter(vid, "lcSpeedGain", str(LC_SPEED_GAIN))
        traci.vehicle.setParameter(vid, "lcAssertive", str(LC_ASSERTIVE))
        traci.vehicle.setParameter(vid, "lcImpatience", str(LC_IMPATIENCE))
        
        return True, props
        
    except traci.TraCIException:
        return False, None


def inject_vehicle_deferred(vid, vtype_car, vtype_truck):
    """
    Generates vehicle properties WITHOUT inserting to TraCI.
    Used by Phase 1B where route is determined by agent logic.
    
    Returns: (success, props_dict)
        props_dict includes: group_id, assigned_kmh, assigned_tau, assigned_sigma, type, speed_factor
    """
    try:
        # Determine Type: Truck probability from config
        is_truck = random.random() < TRUCK_PROBABILITY
        my_type = vtype_truck if is_truck else vtype_car
        
        props = {}

        if is_truck:
            assigned_kmh = random.normalvariate(TRUCK_SPEED_MEAN, TRUCK_SPEED_STD)
            assigned_kmh += SPEED_OFFSET_KMH  # LOCKED 2026-05-08: same offset as cars
            if assigned_kmh > TRUCK_SPEED_MAX: assigned_kmh = TRUCK_SPEED_MAX
            if assigned_kmh < 60: assigned_kmh = 60

            s_factor = assigned_kmh / EDGE_SPEED_LIMIT_KMH  # Same formula as cars
            assigned_tau = 1.3
            assigned_sigma = 0.3
            
            props = {
                "group_id": "truck",
                "assigned_kmh": assigned_kmh,
                "assigned_tau": assigned_tau,
                "assigned_sigma": assigned_sigma,
                "type": my_type,
                "speed_factor": s_factor,
                "color": TRUCK_GROUP["color"]
            }

        else:
            p_val = random.random()
            assigned_kmh = stats.norm.ppf(p_val, loc=MU_KMH, scale=SIGMA_KMH)
            assigned_kmh += SPEED_OFFSET_KMH
            if assigned_kmh < 60: assigned_kmh = 60
            
            group_id = get_group_from_speed_percentile(p_val)
            s_factor = assigned_kmh / EDGE_SPEED_LIMIT_KMH
            assigned_tau = calculate_tau(assigned_kmh)
            assigned_sigma = 0.3

            props = {
                "group_id": group_id,
                "assigned_kmh": assigned_kmh,
                "assigned_tau": assigned_tau,
                "assigned_sigma": assigned_sigma,
                "type": my_type,
                "speed_factor": s_factor,
                "color": QUINTILES[group_id]["color"]
            }

        return True, props
        
    except Exception:
        return False, None


# =============================================================================
# ROUTE DEFINITIONS — DIRECT INSERTION (matching calibration)
# =============================================================================
# All vehicles spawn directly on main path edges.
# DO NOT use entry edges (entry_short_fwd, etc.) — they create artificial
# 2-to-1 lane merge bottlenecks not present in the calibrated model.
# See: rebuild branch commit history for bug documentation.

ROUTES_FWD = {
    "short": "short_fwd",
    "long": "long_fwd",
}

ROUTES_BWD = {
    "short": "short_bwd",
    "long": "long_bwd",
}

# Edges used for N_start counting (instantaneous vehicle counts)
NSTART_EDGES = {
    "short": ["short_fwd", "short_bwd"],
    "long": ["long_fwd", "long_bwd"],
}


def get_route_edges(path: str, direction: str) -> list[str]:
    """Return route edge list for SUMO route.add().

    Args:
        path: "short" or "long"
        direction: "fwd" or "bwd"

    Returns:
        Single-element list with the body edge ID.
        NEVER includes entry edges.
    """
    if direction == "fwd":
        return [ROUTES_FWD[path]]
    return [ROUTES_BWD[path]]

