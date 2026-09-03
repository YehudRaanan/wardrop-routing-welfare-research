"""
Agent Types Module
==================

Implements different agent types for mixed-population simulations:
- TypeAAgent: Static agents with fixed route assignment (never changes)
- TypeBAgent: Selfish agents with pluggable route-choice (imported from agent_logic.py)

The route-choice method for TypeBAgent is determined by agent_logic.py,
which supports EMA, Thompson Sampling, and Davis N_start anticipation.

Type A Behavior:
- Pre-assigned to a route at creation (50% short, 50% long)
- Never changes route regardless of toll or congestion
- Still has VOT for welfare calculations (same distribution as Type B)

Usage:
    from agent_types import TypeAAgent, AgentFactory
    factory = AgentFactory(type_a_pct=40)
    agent = factory.create_agent(vid="v1", group_id="q3", assigned_kmh=100, assigned_tau=2.0)
"""

import random
import math
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Union
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import calibration_config as config
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    import calibration_config as config

# Import TypeBAgent from existing module
# Note: path_setup.py adds 00_Config/core to sys.path, so import directly
from agent_logic import TypeBAgent, GlobalEMA, GlobalMediator


@dataclass
class TypeAAgent:
    """
    Type A agent with static (fixed) route assignment.

    Behavior:
    - Route is pre-assigned at creation and NEVER changes
    - Does not respond to tolls, congestion, or EMA updates
    - Still has VOT for welfare calculations (same distribution as Type B)

    Attributes:
        vid: Vehicle ID
        group_id: Driver group (q1-q5, truck) - Speed Group
        assigned_kmh: Target speed
        assigned_tau: Following time
        assigned_route: Fixed route ("short" or "long") - NEVER changes
        vot: Value of Time (USD/hr) - for welfare calculations
    """
    vid: str
    group_id: str
    assigned_kmh: float
    assigned_tau: float
    assigned_route: Literal["short", "long"]

    # Economic properties (assigned in __post_init__)
    vot: float = field(init=False)
    income_gbp: float = field(init=False)
    trip_purpose: str = field(init=False)

    # Trip tracking
    departure_time: Optional[float] = None
    path_entry_time: Optional[float] = None  # When entering main path (after junction)
    chosen_path: Optional[str] = field(init=False)
    choice_type: str = field(init=False, default="static_fixed")

    # Deferred route choice tracking (phase1d common entry)
    route_decided: bool = False
    direction: Optional[str] = None  # "fwd" or "bwd"
    default_route: Optional[str] = None  # Default route assigned at spawn

    # Toll tracking (set when route is chosen)
    toll_paid: float = 0.0

    def __post_init__(self):
        """
        Assign economic properties (same distribution as Type B).
        Set chosen_path to the fixed assigned_route.
        """
        # 1. Draw Income from Log-Normal Distribution
        log_income = random.gauss(config.LN_MU, config.LN_SIGMA)
        self.income_gbp = math.exp(log_income)

        # 2. Convert to Hourly USD
        hourly_income_usd = (self.income_gbp * config.EXCHANGE_RATE_USD_GBP) / 2000.0

        # 3. Assign Trip Purpose (1/3 each)
        rand_p = random.random()
        if rand_p < 0.333:
            self.trip_purpose = "Commute"
            coeff = config.VOT_COEFF_COMMUTE
        elif rand_p < 0.666:
            self.trip_purpose = "Business"
            coeff = config.VOT_COEFF_BUSINESS
        else:
            self.trip_purpose = "Other"
            coeff = config.VOT_COEFF_OTHER

        # 4. Calculate VOT
        self.vot = max(0.1, hourly_income_usd * coeff)

        # 5. Set chosen_path to the fixed route
        self.chosen_path = self.assigned_route
        self.choice_type = "static_fixed"

    def choose_route(self, global_ema: GlobalEMA, toll_short_usd: float = 0.0) -> Literal["short", "long"]:
        """
        Return the fixed assigned route (never changes).

        Type A agents ignore EMA and tolls - they always take their assigned route.

        Args:
            global_ema: Ignored (Type A doesn't use EMA)
            toll_short_usd: Used only to track toll_paid if on short path

        Returns:
            The pre-assigned route ("short" or "long")
        """
        # Record toll if on short path
        if self.assigned_route == "short":
            self.toll_paid = toll_short_usd
        else:
            self.toll_paid = 0.0

        # Always return the fixed route
        self.chosen_path = self.assigned_route
        return self.assigned_route

    def to_log_dict(self, arrival_time: float) -> dict:
        """Generate log entry for this vehicle."""
        travel_time = arrival_time - self.departure_time if self.departure_time else 0

        return {
            "vid": self.vid,
            "agent_type": "A",
            "group_id": self.group_id,
            "assigned_kmh": round(self.assigned_kmh, 2),
            "assigned_tau": round(self.assigned_tau, 3),
            "income_gbp": round(self.income_gbp, 0),
            "trip_purpose": self.trip_purpose,
            "vot_usd_hr": round(self.vot, 2),
            "toll_paid_usd": self.toll_paid,
            "assigned_route": self.assigned_route,
            "chosen_path": self.chosen_path,
            "choice_type": self.choice_type,
            "departure_time": round(self.departure_time, 2) if self.departure_time else 0,
            "arrival_time": round(arrival_time, 2),
            "travel_time": round(travel_time, 2),
        }


@dataclass
class TypeCAgent:
    """
    Type C agent — mediated routing with IC verification.

    Combines private EMA beliefs with centralized mediator guidance.
    Follows mediator recommendation if it falls within the agent's
    uncertainty margin (IC check). Otherwise, reverts to selfish EMA choice.
    """
    vid: str
    group_id: str
    assigned_kmh: float
    assigned_tau: float

    # Economic properties (assigned in __post_init__)
    vot: float = field(init=False)
    income_gbp: float = field(init=False)
    trip_purpose: str = field(init=False)

    # Trip tracking
    departure_time: Optional[float] = None
    chosen_path: Optional[str] = None
    choice_type: Optional[str] = None
    ema_short_at_choice: Optional[float] = None
    ema_long_at_choice: Optional[float] = None

    # Mediator tracking
    mediator_signal: Optional[str] = None       # "short" or "long"
    signal_followed: Optional[bool] = None
    trust_broken: Optional[bool] = None
    compliance_code: Optional[str] = None       # "followed" / "defected" / "passthrough"
    mediator_mode: Optional[str] = None         # "passthrough" / "active"
    pred_short: Optional[float] = None          # Oracle prediction at departure
    pred_long: Optional[float] = None
    gap_physical: Optional[float] = None
    ic_margin_at_choice: Optional[float] = None
    ic_constrained: Optional[bool] = None

    # N_start at departure
    n_start_short_at_choice: Optional[int] = None
    n_start_long_at_choice: Optional[int] = None

    # Toll (for compatibility)
    toll_paid: float = 0.0

    def __post_init__(self):
        log_income = random.gauss(config.LN_MU, config.LN_SIGMA)
        self.income_gbp = math.exp(log_income)
        hourly_income_usd = (self.income_gbp * config.EXCHANGE_RATE_USD_GBP) / 2000.0
        rand_p = random.random()
        if rand_p < 0.333:
            self.trip_purpose = "Commute"
            coeff = config.VOT_COEFF_COMMUTE
        elif rand_p < 0.666:
            self.trip_purpose = "Business"
            coeff = config.VOT_COEFF_BUSINESS
        else:
            self.trip_purpose = "Other"
            coeff = config.VOT_COEFF_OTHER
        self.vot = max(0.1, hourly_income_usd * coeff)

    def choose_route(self, global_ema: GlobalEMA,
                     mediator_signal: str,
                     ic_margin: float) -> Literal["short", "long"]:
        """
        Three-step IC verification of mediator recommendation.

        1. Receive signal from mediator
        2. Compare recommended path's EMA with alternative
        3. Follow if within IC margin; otherwise defect to selfish choice

        Args:
            global_ema: Shared EMA state
            mediator_signal: "short" or "long" from mediator
            ic_margin: IC margin for the recommended path (seconds)

        Returns:
            Chosen path ("short" or "long")
        """
        ema_short, ema_long = global_ema.get_values()
        self.ema_short_at_choice = ema_short
        self.ema_long_at_choice = ema_long
        self.mediator_signal = mediator_signal

        # EMA of recommended vs alternative
        if mediator_signal == "long":
            ema_recommended = ema_long
            ema_alternative = ema_short
        else:
            ema_recommended = ema_short
            ema_alternative = ema_long

        # IC verification: recommended path should not look much worse
        ema_penalty = ema_recommended - ema_alternative

        if ema_penalty <= ic_margin:
            # Agent trusts mediator
            self.chosen_path = mediator_signal
            self.signal_followed = True
            self.trust_broken = False
            self.choice_type = "mediated"
            self.compliance_code = "followed"
        else:
            # Trust break — revert to selfish EMA choice
            self.chosen_path = "short" if ema_short <= ema_long else "long"
            self.signal_followed = False
            self.trust_broken = True
            self.choice_type = "defected_ema"
            self.compliance_code = "defected"

        return self.chosen_path


class AgentFactory:
    """
    Factory for creating mixed populations of Type A and Type B agents.

    Type A agents are assigned routes with 50/50 split (constant across all tolls).
    Type B agents make rational cost-based decisions.

    Usage:
        factory = AgentFactory(type_a_pct=40, toll_usd=0.25)
        agents = factory.create_agents(n=1000, group_distribution=group_dist)
    """

    def __init__(
        self,
        type_a_pct: float = 0.0,
        type_a_short_pct: float = 50.0,
    ):
        """
        Initialize the agent factory.

        Args:
            type_a_pct: Percentage of agents that are Type A (0-100)
            type_a_short_pct: Percentage of Type A agents assigned to short path (default 50%)
        """
        self.type_a_pct = max(0.0, min(100.0, type_a_pct))
        self.type_a_short_pct = max(0.0, min(100.0, type_a_short_pct))

    def create_agent(
        self,
        vid: str,
        group_id: str,
        assigned_kmh: float,
        assigned_tau: float,
    ) -> Union[TypeAAgent, TypeBAgent]:
        """
        Create a single agent (Type A or Type B based on factory settings).

        Args:
            vid: Vehicle ID
            group_id: Speed group
            assigned_kmh: Target speed
            assigned_tau: Following time

        Returns:
            Either TypeAAgent or TypeBAgent
        """
        # Determine agent type
        if random.random() * 100 < self.type_a_pct:
            # Create Type A agent
            # Assign route: 50/50 split
            if random.random() * 100 < self.type_a_short_pct:
                assigned_route = "short"
            else:
                assigned_route = "long"

            return TypeAAgent(
                vid=vid,
                group_id=group_id,
                assigned_kmh=assigned_kmh,
                assigned_tau=assigned_tau,
                assigned_route=assigned_route,
            )
        else:
            # Create Type B agent
            return TypeBAgent(
                vid=vid,
                group_id=group_id,
                assigned_kmh=assigned_kmh,
                assigned_tau=assigned_tau,
            )

    def is_type_a(self, agent: Union[TypeAAgent, TypeBAgent]) -> bool:
        """Check if agent is Type A."""
        return isinstance(agent, TypeAAgent)

    def is_type_b(self, agent: Union[TypeAAgent, TypeBAgent]) -> bool:
        """Check if agent is Type B."""
        return isinstance(agent, TypeBAgent)


def get_agent_type_str(agent: Union[TypeAAgent, TypeBAgent, TypeCAgent]) -> str:
    """Return agent type as string ("A", "B", or "C")."""
    if isinstance(agent, TypeAAgent):
        return "A"
    elif isinstance(agent, TypeCAgent):
        return "C"
    elif isinstance(agent, TypeBAgent):
        return "B"
    else:
        return "Unknown"
