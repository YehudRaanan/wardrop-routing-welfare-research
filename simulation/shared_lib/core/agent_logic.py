"""
Agent Logic Module
------------------
Implements Type B agent route choice with multiple available methods.

Available Route Choice Methods (active as of 2026-05-25):
1. GlobalEMA — Exponential Moving Average over per-path TT.
   Route decision uses STOCHASTIC LOGIT choice (logit_theta, default θ=0.05)
   over generalised cost (time + toll / VOT). Pre-stochastic deterministic
   (argmin) variants have been retired and their runs marked is_valid=0.
2. GlobalDavisThreshold — Davis (2009) Sec. V threshold anticipation with
   self-calibrated T_ref. Route decision uses STOCHASTIC LOGIT choice
   (logit_theta, default θ=0.05) over generalised cost. This is the canonical
   Davis variant for the thesis.
3. GlobalThompsonSampling — Bayesian belief updating; implemented but not used
   in the current thesis runs (no rows in DB).

Retired:
- GlobalNStartPredictor (vanilla N_start linear regression) — superseded by
  GlobalDavisThreshold. All pre-existing runs marked is_valid=0 in the DB.

The active method is chosen by the runner script (route_choice_method string),
NOT hardcoded here. All methods share a common interface: update() and get_values().

Economic Properties (VOT & Tolls):
- Agents have heterogeneous Value of Time (VOT)
- Income drawn from UK 2025 Log-Normal distribution
- Route choice uses Generalized Cost: Time + (Toll / VOT)
- Income is independent of speed

Standards Compliance:
- Uses parameters from simulation_shared.py and calibration_config.py
- Logs all decisions for analysis
"""

import random
import math
from dataclasses import dataclass, field
from typing import Literal, Optional
import csv
import os
import sys
import numpy as np

# Add project root to path to import calibration_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import calibration_config as config
except ImportError:
    # Fallback if running from a different directory context
    sys.path.append(os.path.join(os.getcwd(), '..'))
    import calibration_config as config

# =============================================================================
# THOMPSON SAMPLING CONFIGURATION
# =============================================================================
# Prior beliefs: Start with free-flow estimates and high uncertainty
PRIOR_MEAN_SHORT = 270.0      # seconds (free-flow estimate for 7.5km)
PRIOR_MEAN_LONG = 337.5       # seconds (free-flow estimate for 9.375km, ratio 1.25)
PRIOR_VARIANCE = 10000.0      # High variance = uncertain (std dev ~100s)

# Observation noise: How much variance in individual trip observations
OBSERVATION_VARIANCE = 900.0  # (30 seconds std dev)^2 - accounts for traffic variability

# =============================================================================
# EMA CONFIGURATION
# =============================================================================
EMA_ALPHA = 0.1
EXPLORATION_EPSILON = 0.0  # Set > 0 to enable epsilon-greedy exploration

# Initial EMA values (used if GlobalEMA is still used)
INITIAL_EMA_SHORT = 270.0     # seconds (free-flow TT for 7.5km at ~100km/h)
INITIAL_EMA_LONG = 337.5      # seconds (free-flow TT for 9.375km, ratio 1.25)

# Long/Short ratio for cross-update estimation
RATIO_LONG_TO_SHORT = 1.25  # 9375m / 7500m

# Path lengths (meters) - development-scale network
PATH_LENGTH_SHORT = 7500.0   # meters (7.5 km)
PATH_LENGTH_LONG = 9375.0    # meters (9.375 km)


class GlobalEMA:
    """
    Tracks global (shared) exponential moving average of travel times.
    All Type B agents read from this to make decisions.
    
    BIDIRECTIONAL UPDATE: When any vehicle completes, we use its average
    speed to estimate travel time for BOTH paths and update both EMAs.
    This ensures symmetric feedback and prevents EMA staleness.
    """
    
    def __init__(self, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.ema_short = INITIAL_EMA_SHORT
        self.ema_long = INITIAL_EMA_LONG
        self.update_count_short = 0
        self.update_count_long = 0
    
    def update(self, path: Literal["short", "long"], actual_travel_time: float):
        """
        Updates only the observed path with actual travel time.

        Cross-path updates were disabled 2026-02-03 because ratio-based
        propagation prevented agents from discovering an empty alternate path.
        """

        if path == "short":
            self.ema_short = self.alpha * actual_travel_time + (1 - self.alpha) * self.ema_short
            self.update_count_short += 1
        else:
            self.ema_long = self.alpha * actual_travel_time + (1 - self.alpha) * self.ema_long
            self.update_count_long += 1
    
    def get_optimal_path(self) -> Literal["short", "long"]:
        """Returns the path with lower EMA travel time."""
        return "short" if self.ema_short <= self.ema_long else "long"
    
    def get_values(self) -> tuple[float, float]:
        """Returns (ema_short, ema_long)."""
        return self.ema_short, self.ema_long


class GlobalThompsonSampling:
    """
    Thompson Sampling for route choice with Bayesian belief updating.

    Maintains Gaussian beliefs (mean, variance) for travel time on each path.
    Route choice uses Thompson Sampling: sample from each belief, choose lowest.

    This naturally balances exploration (uncertain paths get more variable samples)
    vs exploitation (well-known paths get consistent samples near mean).

    Academic foundation:
    - Thompson (1933): "On the likelihood that one unknown probability exceeds another"
    - Russo et al. (2018): "A Tutorial on Thompson Sampling"
    - Applied to route choice: Bayesian updating with Gaussian distributions
    """

    def __init__(self,
                 prior_mean_short: float = PRIOR_MEAN_SHORT,
                 prior_mean_long: float = PRIOR_MEAN_LONG,
                 prior_variance: float = PRIOR_VARIANCE,
                 obs_variance: float = OBSERVATION_VARIANCE):
        # Beliefs for short path
        self.mean_short = prior_mean_short
        self.var_short = prior_variance

        # Beliefs for long path
        self.mean_long = prior_mean_long
        self.var_long = prior_variance

        # Observation noise (assumed constant)
        self.obs_variance = obs_variance

        # Tracking
        self.update_count_short = 0
        self.update_count_long = 0

    def update(self, path: Literal["short", "long"], observed_tt: float):
        """
        Bayesian update of beliefs after observing a travel time.

        Uses precision-weighted averaging (conjugate Gaussian update):
        posterior_precision = prior_precision + observation_precision
        posterior_mean = (prior_mean * prior_prec + obs * obs_prec) / posterior_prec
        """
        if path == "short":
            prior_prec = 1.0 / self.var_short
            obs_prec = 1.0 / self.obs_variance
            post_prec = prior_prec + obs_prec

            self.mean_short = (self.mean_short * prior_prec + observed_tt * obs_prec) / post_prec
            self.var_short = 1.0 / post_prec
            self.update_count_short += 1
        else:
            prior_prec = 1.0 / self.var_long
            obs_prec = 1.0 / self.obs_variance
            post_prec = prior_prec + obs_prec

            self.mean_long = (self.mean_long * prior_prec + observed_tt * obs_prec) / post_prec
            self.var_long = 1.0 / post_prec
            self.update_count_long += 1

    def sample_and_choose(self, toll_penalty_short: float = 0.0) -> tuple[Literal["short", "long"], str]:
        """
        Thompson Sampling: sample from each belief, choose path with lowest sample.

        Args:
            toll_penalty_short: Toll penalty in seconds (toll_usd * 3600 / vot)

        Returns:
            (chosen_path, choice_type) where choice_type is "thompson_short_lower" or "thompson_long_lower"
        """
        # Sample travel times from belief distributions
        sample_short = random.gauss(self.mean_short, math.sqrt(self.var_short))
        sample_long = random.gauss(self.mean_long, math.sqrt(self.var_long))

        # Add toll penalty to short path cost
        cost_short = sample_short + toll_penalty_short
        cost_long = sample_long

        # Choose path with lower sampled cost
        if cost_short <= cost_long:
            return "short", "thompson"
        else:
            return "long", "thompson"

    def get_values(self) -> tuple[float, float]:
        """Returns (mean_short, mean_long) for compatibility with EMA interface."""
        return self.mean_short, self.mean_long

    def get_beliefs(self) -> dict:
        """Returns full belief state for logging/debugging."""
        return {
            "mean_short": round(self.mean_short, 2),
            "std_short": round(math.sqrt(self.var_short), 2),
            "mean_long": round(self.mean_long, 2),
            "std_long": round(math.sqrt(self.var_long), 2),
            "n_obs_short": self.update_count_short,
            "n_obs_long": self.update_count_long,
        }


# =============================================================================
# DAVIS (2009) N_START ANTICIPATION
# =============================================================================
# Implements the "anticipation" approach from:
# Davis, L.C. (2009). "Realizing Wardrop Equilibria with Real-Time Traffic Information."
# Physica A, 388(18), 4459-4474.
#
# KEY INSIGHT: N_start (vehicle count at departure) is a LEADING indicator of
# future travel time, while TT itself is a LAGGING indicator.
#
# NOTE: Requires centralized real-time vehicle count information.
# =============================================================================

class GlobalNStartPredictor:
    """Davis (2009)-style N_start-based TT prediction (rebuilt 2026-05-06).

    Maintains a rolling buffer of (N_start, TT_observation) tuples per path.
    Buffer is fed from TWO sources by the runner (mixed-source design):

      Source A — vehicle ARRIVAL: (N_start_at_THIS_vehicle's_departure,
                                   actual_TT_observed_at_arrival)
      Source B — IN-TRANSIT SAMPLING (every minute): for each currently-driving
                                   vehicle on the path,
                                   (current_N_start, remaining_dist / current_speed)

    Refits TT = intercept + slope × N_start on every prediction call (cheap
    for ~500 points). Floor at baseline FF TT so prediction can't go below
    physical minimum.

    Methodology contract (rebuild 2026-05-06):
      Both EMA and Davis consume the SAME stream of observations (same data
      window). The methodological difference is purely in how each
      aggregates: EMA exponential-decay-averages the TTs; Davis conditions
      the TTs on N_start via linear regression and extrapolates to the
      current N_start at choice time.
    """

    BUFFER_SIZE = 500  # rolling-buffer cap per path (count-based; default)
    MIN_OBS_FOR_REGRESSION = 10

    def __init__(self,
                 baseline_tt_short: float = PRIOR_MEAN_SHORT,
                 baseline_tt_long: float = PRIOR_MEAN_LONG,
                 buffer_size: int | None = None):
        from collections import deque

        self.baseline_short = baseline_tt_short
        self.baseline_long = baseline_tt_long

        # 2026-05-11: buffer_size param added for matched-N runs (e.g., 199 to
        # match EMA α=0.01 effective observation count). None preserves the
        # class default (500), so existing runners are unaffected.
        self._buffer_size = buffer_size if buffer_size is not None else self.BUFFER_SIZE

        # Rolling buffer of (n_start, tt_estimate) tuples per path.
        self.buffer_short: deque = deque(maxlen=self._buffer_size)
        self.buffer_long: deque = deque(maxlen=self._buffer_size)

        # Cached regression coefficients (refit lazily on prediction).
        self.intercept_short = baseline_tt_short
        self.slope_short = 0.0
        self.intercept_long = baseline_tt_long
        self.slope_long = 0.0
        self._dirty_short = True
        self._dirty_long = True

    # ─────────────────────────────────────────────────────────────────
    # Observation API
    # ─────────────────────────────────────────────────────────────────
    def record_observation(self, path: str, n_start: int, tt_estimate: float) -> None:
        """Append a new (N_start, TT) tuple to the rolling buffer.

        Args:
            path:         "short" or "long"
            n_start:      vehicles on this path at the observation moment
                          (arrival: at THIS vehicle's departure;
                           in-transit: current N_start at sampling time)
            tt_estimate:  observed or projected TT (seconds)
                          (arrival: actual_TT;
                           in-transit: remaining_dist / current_speed)
        """
        if path == "short":
            self.buffer_short.append((n_start, tt_estimate))
            self._dirty_short = True
        elif path == "long":
            self.buffer_long.append((n_start, tt_estimate))
            self._dirty_long = True
        else:
            raise ValueError(f"path must be 'short' or 'long', got {path!r}")

    # Backwards-compatibility shim for existing call sites that pass
    # both n_start values. Only the n_start corresponding to `path` is used.
    def record_trip(self, n_start_short: int, n_start_long: int,
                    path: str, actual_tt: float) -> None:
        n_start = n_start_short if path == "short" else n_start_long
        self.record_observation(path, n_start, actual_tt)

    # ─────────────────────────────────────────────────────────────────
    # Regression fit (lazy)
    # ─────────────────────────────────────────────────────────────────
    def _refit_short(self) -> None:
        if not self._dirty_short:
            return
        if len(self.buffer_short) < self.MIN_OBS_FOR_REGRESSION:
            self._dirty_short = False
            return

        n_arr = np.array([n for n, _ in self.buffer_short], dtype=float)
        tt_arr = np.array([tt for _, tt in self.buffer_short], dtype=float)
        n_mean, tt_mean = n_arr.mean(), tt_arr.mean()

        denom = np.sum((n_arr - n_mean) ** 2)
        if denom > 0:
            slope = np.sum((n_arr - n_mean) * (tt_arr - tt_mean)) / denom
            slope = max(0.0, slope)  # non-negative: more vehicles can't reduce TT
            self.slope_short = slope
            self.intercept_short = tt_mean - slope * n_mean
        self._dirty_short = False

    def _refit_long(self) -> None:
        if not self._dirty_long:
            return
        if len(self.buffer_long) < self.MIN_OBS_FOR_REGRESSION:
            self._dirty_long = False
            return

        n_arr = np.array([n for n, _ in self.buffer_long], dtype=float)
        tt_arr = np.array([tt for _, tt in self.buffer_long], dtype=float)
        n_mean, tt_mean = n_arr.mean(), tt_arr.mean()

        denom = np.sum((n_arr - n_mean) ** 2)
        if denom > 0:
            slope = np.sum((n_arr - n_mean) * (tt_arr - tt_mean)) / denom
            slope = max(0.0, slope)
            self.slope_long = slope
            self.intercept_long = tt_mean - slope * n_mean
        self._dirty_long = False

    # ─────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────
    def predict_tt(self, n_start_short: int, n_start_long: int) -> tuple:
        """Predict TT for each path: TT = intercept + slope × current_N_start.

        Floored at baseline FF TT.
        """
        self._refit_short()
        self._refit_long()
        pred_short = max(self.intercept_short + self.slope_short * n_start_short,
                         self.baseline_short)
        pred_long = max(self.intercept_long + self.slope_long * n_start_long,
                        self.baseline_long)
        return pred_short, pred_long

    # ─────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────
    def get_diagnostics(self) -> dict:
        return {
            "buffer_size_short": len(self.buffer_short),
            "buffer_size_long": len(self.buffer_long),
            "intercept_short": round(self.intercept_short, 2),
            "slope_short": round(self.slope_short, 4),
            "intercept_long": round(self.intercept_long, 2),
            "slope_long": round(self.slope_long, 4),
        }


class GlobalDavisThreshold:
    """Davis (2009) Sec. V threshold anticipation — extended to 2 congestible paths.

    Faithful re-implementation of the paper's algorithm, adapted to a network
    where both paths are congestion-dependent (paper assumed main + fixed-TT
    bypass). The original threshold rule is preserved by introducing a
    self-calibrated reference T_ref instead of the paper's fixed T_bypass.

    Algorithm
    ---------
    Per-path rolling buffer of (N_start, TT) pairs, pre-filled with
    free-flow pseudo-observations so the buffer is never empty:

        buffer_short ← [(0, FF_short)] * BUFFER_SIZE
        buffer_long  ← [(0, FF_long)]  * BUFFER_SIZE

    On every observation (arrival or in-transit) the corresponding buffer
    appends the new (N_start, TT) and the oldest entry drops off (deque
    with maxlen). Pseudo-observations are gradually flushed by real data.

    Reference T_ref is the running mean TT across BOTH buffers' contents:

        T_ref = mean({tt for (_, tt) in buffer_short ∪ buffer_long})

    Initial value with all pseudo-fill: (FF_short + FF_long) / 2 ≈ 211 s.
    Converges toward the empirical Wardrop TT (~340 s in this network) as
    real observations replace pseudos.

    Per-path classification + midpoint:

        N<_path = mean N_start over buffer entries with TT ≤ T_ref ("good")
        N>_path = mean N_start over buffer entries with TT  > T_ref ("bad")
        midpoint_path = ½(N<_path + N>_path)
        (single-class fallback: midpoint = the non-empty class's mean)

    Decision (uniform rule — no FF tiebreak):

        short_residual = N_start_short - midpoint_short
        long_residual  = N_start_long  - midpoint_long
        → take path with smaller residual

    This subsumes the paper's 4-case truth table into one comparison:
      - both-good → pick path *deeper* below its midpoint
      - both-bad  → pick path with less overshoot
      - mixed     → "good" path wins (residual negative vs positive)
    No FF advantage applied — under Wardrop targeting, asymmetric tiebreaks
    would prevent the equilibrium split from settling.

    Interface compatibility
    -----------------------
    Exposes the same `record_observation` / `record_trip` / `predict_tt` /
    `get_diagnostics` interface as GlobalNStartPredictor so the existing
    runner code in unified_runner.py works unchanged. `predict_tt` returns
    synthetic (pred_short, pred_long) pair that encodes the decision: the
    chosen path gets T_ref − ε, the other gets T_ref + ε, so the caller's
    "pick lower predicted TT" logic selects the right path.
    """

    BUFFER_SIZE = 199  # matched-N default; one rolling window per path

    # Free-flow TTs per path (network constants; see shared_lib.core.vehicle_filter).
    FF_SHORT: float = 7500.0 / 40.0   # 187.5 s
    FF_LONG: float = 9375.0 / 40.0    # 234.375 s

    def __init__(
        self,
        baseline_tt_short: float = PRIOR_MEAN_SHORT,
        baseline_tt_long: float = PRIOR_MEAN_LONG,
        buffer_size: int | None = None,
    ):
        from collections import deque

        self.baseline_short = baseline_tt_short
        self.baseline_long = baseline_tt_long
        self._buffer_size = buffer_size if buffer_size is not None else self.BUFFER_SIZE

        # Pre-fill with FF pseudo-observations: (N_start=0, TT=FF_path).
        # Pseudo entries flush out within ~1.2 min of real data inflow at
        # operational flow (≈166 obs/min/path).
        self.buffer_short = deque(
            [(0, self.FF_SHORT)] * self._buffer_size,
            maxlen=self._buffer_size,
        )
        self.buffer_long = deque(
            [(0, self.FF_LONG)] * self._buffer_size,
            maxlen=self._buffer_size,
        )

    # ─────────────────────────────────────────────────────────────────
    # Observation API (same shape as GlobalNStartPredictor)
    # ─────────────────────────────────────────────────────────────────
    def record_observation(self, path: str, n_start: int, tt_estimate: float) -> None:
        if path == "short":
            self.buffer_short.append((n_start, tt_estimate))
        elif path == "long":
            self.buffer_long.append((n_start, tt_estimate))
        else:
            raise ValueError(f"path must be 'short' or 'long', got {path!r}")

    def record_trip(
        self, n_start_short: int, n_start_long: int, path: str, actual_tt: float
    ) -> None:
        """Back-compat shim for callers that pass both n_start values."""
        n_start = n_start_short if path == "short" else n_start_long
        self.record_observation(path, n_start, actual_tt)

    # ─────────────────────────────────────────────────────────────────
    # Self-calibrated reference + per-path midpoint
    # ─────────────────────────────────────────────────────────────────
    def _t_ref(self) -> float:
        """Running mean TT across both per-path buffers."""
        # Both buffers have identical length, so a flat mean is unbiased.
        tts = [tt for _, tt in self.buffer_short] + [tt for _, tt in self.buffer_long]
        return sum(tts) / len(tts) if tts else 0.5 * (self.FF_SHORT + self.FF_LONG)

    def _midpoint_and_class_means(self, buf, t_ref: float, ff_tt: float):
        """Return (midpoint, mean_TT_good, mean_TT_bad) for the buffer.

        midpoint = ½(N<_path + N>_path) with single-class fallback.
        mean_TT_good = mean TT over buffer entries with TT ≤ T_ref ("good")
        mean_TT_bad  = mean TT over buffer entries with TT  > T_ref ("bad")
        Empty-class fallback: use ff_tt (the path's free-flow TT) so the
        toll-additive cost comparison stays well-defined during the cold
        start before real observations populate both classes.
        """
        good_ns, good_tts = [], []
        bad_ns, bad_tts = [], []
        for n, tt in buf:
            if tt <= t_ref:
                good_ns.append(n); good_tts.append(tt)
            else:
                bad_ns.append(n); bad_tts.append(tt)
        if good_ns and bad_ns:
            midpoint = 0.5 * (
                sum(good_ns) / len(good_ns) + sum(bad_ns) / len(bad_ns)
            )
        elif good_ns:
            midpoint = sum(good_ns) / len(good_ns)
        elif bad_ns:
            midpoint = sum(bad_ns) / len(bad_ns)
        else:
            midpoint = 0.0  # impossible with pre-fill
        mean_good = sum(good_tts) / len(good_tts) if good_tts else ff_tt
        mean_bad = sum(bad_tts) / len(bad_tts) if bad_tts else ff_tt
        return midpoint, mean_good, mean_bad

    def _midpoint(self, buf, t_ref: float) -> float:
        """½(N<_path + N>_path) — exposed for diagnostics."""
        m, _, _ = self._midpoint_and_class_means(buf, t_ref, self.FF_SHORT)
        return m

    # ─────────────────────────────────────────────────────────────────
    # Decision (exposed as predict_tt for runner compatibility)
    # ─────────────────────────────────────────────────────────────────
    def predict_tt(
        self, n_start_short: int, n_start_long: int
    ) -> tuple[float, float]:
        """Return per-path stratified-mean TT predictions in real seconds.

        For each path, classify the driver's N_start against the per-path
        midpoint (good = below, bad = above). The "predicted TT" is the
        mean TT within the classified class — a step-function predictor
        with 2 levels per path (good-mean, bad-mean).

        Why stratified means instead of synthetic ε:
          - Real TT units → adding toll-as-seconds (toll/VOT*3600) in
            the caller's cost comparison is dimensionally consistent.
          - Toll-flipping behaviour works correctly: low-VOT agents
            pay larger toll-seconds and avoid the tolled path; high-VOT
            agents shrug it off and prefer the faster path on TT alone.
          - The threshold-classification structure of Davis Sec. V is
            preserved — the binary good/bad decision per path still
            comes from the N_start vs midpoint test; only the OUTPUT
            representation is in TT seconds instead of ±ε.
        """
        t_ref = self._t_ref()
        mid_s, good_s, bad_s = self._midpoint_and_class_means(
            self.buffer_short, t_ref, self.FF_SHORT
        )
        mid_l, good_l, bad_l = self._midpoint_and_class_means(
            self.buffer_long, t_ref, self.FF_LONG
        )
        pred_short = good_s if n_start_short < mid_s else bad_s
        pred_long = good_l if n_start_long < mid_l else bad_l
        return (pred_short, pred_long)

    # ─────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────
    def get_diagnostics(self) -> dict:
        t_ref = self._t_ref()
        return {
            "buffer_size_short": len(self.buffer_short),
            "buffer_size_long": len(self.buffer_long),
            "t_ref": round(t_ref, 2),
            "midpoint_short": round(self._midpoint(self.buffer_short, t_ref), 2),
            "midpoint_long": round(self._midpoint(self.buffer_long, t_ref), 2),
        }


class GlobalHabitFixedAtStart:
    """
    Habit-based routing — fixed-at-start.

    Each driver draws a path once at simulation start with probability
    p_short_habit (default 0.55) and never updates.
    """
    def __init__(self, p_short_habit: float = 0.55):
        self.p_short_habit = p_short_habit

    def get_values(self) -> tuple[float, float]:
        # Returns dummy values (FF values)
        return 187.5, 234.375

    def get_diagnostics(self) -> dict:
        return {"p_short_habit": self.p_short_habit}


@dataclass
class TypeBAgent:
    """
    Type B agent with 100% pure EMA-based route choice and Value of Time logic.

    Behavior:
    - Always chooses cost-optimal path based on EMA (no habitual component)
    - Uses Generalized Cost: Time + (Toll / VOT)

    Attributes:
        vid: Vehicle ID
        group_id: Driver group (q1-q5, truck) - Speed Group
        assigned_kmh: Target speed
        assigned_tau: Following time
        vot: Value of Time (USD/hr) - Derived from Income
    """
    vid: str
    group_id: str
    assigned_kmh: float
    assigned_tau: float
    
    # Phase 2: Economic Properties (Assigned in __post_init__)
    vot: float = field(init=False)
    income_gbp: float = field(init=False)
    trip_purpose: str = field(init=False)
    
    # Trip tracking
    departure_time: Optional[float] = None
    path_entry_time: Optional[float] = None  # When entering main path (after junction)
    chosen_path: Optional[str] = None
    choice_type: Optional[str] = None
    ema_short_at_choice: Optional[float] = None
    ema_long_at_choice: Optional[float] = None
    p_short_at_choice: Optional[float] = None

    # Deferred route choice tracking (phase1d common entry)
    route_decided: bool = False
    direction: Optional[str] = None  # "fwd" or "bwd"
    default_route: Optional[str] = None  # Default route assigned at spawn

    # Phase 2 Tracking
    cost_short: Optional[float] = None
    cost_long: Optional[float] = None
    toll_paid: float = 0.0

    # N_start tracking (for Davis anticipation mode - optional)
    n_start_short_at_choice: Optional[int] = None
    n_start_long_at_choice: Optional[int] = None
    
    def __post_init__(self):
        """
        Assign Economic Properties independently of Speed Properties.
        """
        # 1. Draw Income from Phase 2 Log-Normal Distribution
        # Use log-normal mu and sigma from config
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
        self.vot = hourly_income_usd * coeff
        
        # Floor VOT to avoid division by zero or negative
        self.vot = max(0.1, self.vot)

    def choose_route(self, global_beliefs, toll_short_usd: float = 0.0,
                     # NEW: Optional N_start prediction mode (Davis 2009)
                     # Default is None = use Thompson Sampling (realistic)
                     nstart_predictor: Optional['GlobalNStartPredictor'] = None,
                     n_start_short: int = 0,
                     n_start_long: int = 0) -> Literal["short", "long"]:
        """
        Make route choice based on Generalized Cost (Time + Toll).

        Supports three methods (chosen by the runner script):
        1. N_start Anticipation (if nstart_predictor provided): Davis (2009) method
        2. Thompson Sampling (GlobalThompsonSampling): Sample from beliefs, choose lower
        3. EMA (GlobalEMA): Use mean estimates with optional epsilon-greedy
        """
        # Calculate toll penalty in seconds
        monetary_penalty_seconds = (toll_short_usd * 3600.0) / self.vot

        # === N_START ANTICIPATION MODE (Davis 2009) ===
        if nstart_predictor is not None:
            # Store N_start for later learning update
            self.n_start_short_at_choice = n_start_short
            self.n_start_long_at_choice = n_start_long

            # Get predicted travel times based on N_start
            pred_short, pred_long = nstart_predictor.predict_tt(n_start_short, n_start_long)

            # Calculate generalized costs
            cost_short = pred_short + monetary_penalty_seconds
            cost_long = pred_long

            # Store for logging
            self.cost_short = cost_short
            self.cost_long = cost_long
            self.ema_short_at_choice = pred_short  # Use predicted TT for logging
            self.ema_long_at_choice = pred_long

            # Choose path stochastically using logit
            theta = getattr(config, "LOGIT_THETA", 0.05)
            delta = theta * (cost_short - cost_long)
            delta = max(-50.0, min(50.0, delta))
            p_short = 1.0 / (1.0 + math.exp(delta))
            self.p_short_at_choice = p_short

            if random.random() < p_short:
                self.chosen_path = "short"
            else:
                self.chosen_path = "long"
            self.choice_type = "nstart_anticipation"

        # === THOMPSON SAMPLING ===
        elif isinstance(global_beliefs, GlobalThompsonSampling):
            # Get mean beliefs for logging
            mean_short, mean_long = global_beliefs.get_values()
            self.ema_short_at_choice = mean_short
            self.ema_long_at_choice = mean_long

            # Thompson Sampling: sample and choose
            self.chosen_path, self.choice_type = global_beliefs.sample_and_choose(monetary_penalty_seconds)

            # Calculate costs based on means (for logging, not decision)
            self.cost_short = mean_short + monetary_penalty_seconds
            self.cost_long = mean_long
            self.p_short_at_choice = None

        # === EMA ===
        else:
            ema_short, ema_long = global_beliefs.get_values()

            self.cost_short = ema_short + monetary_penalty_seconds
            self.cost_long = ema_long

            # Store for logging
            self.ema_short_at_choice = ema_short
            self.ema_long_at_choice = ema_long

            # Choose path stochastically using logit
            theta = getattr(config, "LOGIT_THETA", 0.05)
            delta = theta * (self.cost_short - self.cost_long)
            delta = max(-50.0, min(50.0, delta))
            p_short = 1.0 / (1.0 + math.exp(delta))
            self.p_short_at_choice = p_short

            self.choice_type = "rational_cost"
            if random.random() < p_short:
                self.chosen_path = "short"
            else:
                self.chosen_path = "long"

        # Record toll if short path chosen
        if self.chosen_path == "short":
            self.toll_paid = toll_short_usd
        else:
            self.toll_paid = 0.0

        return self.chosen_path
    
    def to_log_dict(self, arrival_time: float) -> dict:
        """Generate log entry for this vehicle."""
        travel_time = arrival_time - self.departure_time if self.departure_time else 0
        
        return {
            "vid": self.vid,
            "group_id": self.group_id,
            "assigned_kmh": round(self.assigned_kmh, 2),
            "assigned_tau": round(self.assigned_tau, 3),
            
            # Phase 2 Economic Metrics
            "income_gbp": round(self.income_gbp, 0),
            "trip_purpose": self.trip_purpose,
            "vot_usd_hr": round(self.vot, 2),
            "toll_paid_usd": self.toll_paid,
            "cost_short": round(self.cost_short, 2) if self.cost_short else 0,
            "cost_long": round(self.cost_long, 2) if self.cost_long else 0,
            
            "chosen_path": self.chosen_path,
            "choice_type": self.choice_type,
            "departure_time": round(self.departure_time, 2) if self.departure_time else 0,
            "arrival_time": round(arrival_time, 2),
            "travel_time": round(travel_time, 2),
            "ema_short_at_choice": round(self.ema_short_at_choice, 2) if self.ema_short_at_choice else 0,
            "ema_long_at_choice": round(self.ema_long_at_choice, 2) if self.ema_long_at_choice else 0,
            "p_short_at_choice": round(self.p_short_at_choice, 4) if self.p_short_at_choice is not None else None,
        }


class GlobalMediator:
    """
    Type C Mediated Routing — adaptive centralized controller.

    Learns online from observed trips:
    - Regression TT = a + b * N_start refitted every 60s from recent data
    - IC margin (95th percentile of |EMA error|) updated every 60s
    - SO target recomputed from latest regression coefficients

    Lifecycle:
    1. Warmup (0-60 min): passthrough mode, collects (N_start, TT, EMA_error)
    2. Transition (t=3601s): fits initial model from warmup data
    3. Active (60-120 min): uses learned model, refits every 60s from last 100 trips
    """

    WINDOW_SIZE = 100       # Rolling window for regression and IC margin
    MIN_SAMPLES = 10        # Minimum trips per path before fitting
    REFIT_INTERVAL = 60.0   # Refit every 60 seconds

    def __init__(self, warmup_sec: float = 3600.0,
                 preload: dict | None = None):
        """Initialize the mediator.

        Args:
            warmup_sec: Duration of warmup period (passthrough mode).
            preload: Optional dict with pre-loaded parameters from SO analysis.
                Keys: a_short, b_short, a_long, b_long,
                      ic_margin_short, ic_margin_long, n_short_so.
                When provided, the model starts fitted and won't refit
                during warmup. Only refits during active mode.
        """
        self.warmup_sec = warmup_sec

        # Online data collection: lists of (n_start, actual_tt, ema_error)
        self.history_short = []
        self.history_long = []

        # Current regression coefficients
        self.a_short = 270.0
        self.b_short = 0.01
        self.a_long = 337.5
        self.b_long = 0.01

        # Current SO target
        self.n_short_so = None

        # Current IC margins per path
        self.ic_margin_short = 60.0
        self.ic_margin_long = 60.0

        # Track last refit time
        self.last_refit_time = 0.0
        self.model_fitted = False
        self.refit_count = 0
        self.preloaded = False

        # Runtime counters
        self.signals_issued = 0
        self.signals_short = 0
        self.signals_long = 0
        self.ic_constrained_count = 0

        # Apply pre-loaded parameters if provided
        if preload:
            self.a_short = preload.get("a_short", self.a_short)
            self.b_short = preload.get("b_short", self.b_short)
            self.a_long = preload.get("a_long", self.a_long)
            self.b_long = preload.get("b_long", self.b_long)
            self.ic_margin_short = preload.get("ic_margin_short", self.ic_margin_short)
            self.ic_margin_long = preload.get("ic_margin_long", self.ic_margin_long)
            self.n_short_so = preload.get("n_short_so", self.n_short_so)
            self._n_total_ref = preload.get("n_total_ref", None)
            self.model_fitted = True
            self.preloaded = True

    def record_trip(self, path: str, n_start: int, actual_tt: float,
                    ema_at_choice: float):
        """Record a completed trip for online learning.

        Called on every FWD vehicle arrival (both warmup and active).

        Args:
            path: "short" or "long"
            n_start: N_start on this path at departure time
            actual_tt: Realized travel time (seconds)
            ema_at_choice: EMA prediction for this path at departure
        """
        ema_error = actual_tt - ema_at_choice
        entry = (n_start, actual_tt, ema_error)
        if path == "short":
            self.history_short.append(entry)
        else:
            self.history_long.append(entry)

    def refit(self, sim_time: float, n_total_current: int):
        """Refit regression and IC margins from recent data.

        Called every 60 seconds by the runner. Uses last WINDOW_SIZE trips
        per path (or all available if fewer).

        Args:
            sim_time: Current simulation time
            n_total_current: Current total vehicles in network (for SO calc)
        """
        # Skip refit during warmup if pre-loaded — the warmup data
        # is from EMA oscillation and would corrupt the pre-loaded model.
        # Only refit during active mode when the mediator's own steering
        # produces more stable data.
        if self.preloaded and sim_time <= self.warmup_sec:
            return

        self.last_refit_time = sim_time
        self.refit_count += 1

        # Get recent window
        window_short = self.history_short[-self.WINDOW_SIZE:]
        window_long = self.history_long[-self.WINDOW_SIZE:]

        # --- Refit regression per path ---
        self.a_short, self.b_short = self._fit_regression(window_short, self.a_short, self.b_short)
        self.a_long, self.b_long = self._fit_regression(window_long, self.a_long, self.b_long)

        # --- Recompute SO target ---
        # N_s_so = (a_l - a_s + 2*b_l*N_total) / (2*(b_s + b_l))
        denom = 2.0 * (self.b_short + self.b_long)
        if denom > 0:
            n_so = (self.a_long - self.a_short + 2.0 * self.b_long * n_total_current) / denom
            self.n_short_so = max(0.0, min(n_total_current, n_so))
        else:
            self.n_short_so = n_total_current / 2.0  # Fallback: even split

        # --- Recompute IC margins (only if NOT preloaded or enough active data) ---
        if not self.preloaded:
            self.ic_margin_short = self._compute_ic_margin(window_short)
            self.ic_margin_long = self._compute_ic_margin(window_long)

        self.model_fitted = True

    def _fit_regression(self, window, current_a, current_b):
        """Fit TT = a + b * N_start from recent trips."""
        if len(window) < self.MIN_SAMPLES:
            return current_a, current_b

        n_vals = np.array([w[0] for w in window], dtype=float)
        tt_vals = np.array([w[1] for w in window], dtype=float)

        n_mean = n_vals.mean()
        tt_mean = tt_vals.mean()

        var_n = np.sum((n_vals - n_mean) ** 2)
        if var_n < 1e-6:
            # No variance in N_start — can't fit slope
            return tt_mean, 0.01

        cov_n_tt = np.sum((n_vals - n_mean) * (tt_vals - tt_mean))
        slope = cov_n_tt / var_n
        intercept = tt_mean - slope * n_mean

        # Enforce non-negative slope
        if slope < 0:
            slope = 0.01
            intercept = tt_mean

        return intercept, slope

    def _compute_ic_margin(self, window):
        """Compute IC margin as 95th percentile of |EMA error|."""
        if len(window) < self.MIN_SAMPLES:
            return self.ic_margin_short  # Keep current value

        errors = np.array([abs(w[2]) for w in window])
        margin = float(np.percentile(errors, 95))
        # Must be > 0
        return max(0.01, margin)

    def predict_tt(self, n_short: int, n_long: int) -> tuple:
        """Oracle prediction using current regression coefficients."""
        tt_short = self.a_short + self.b_short * n_short
        tt_long = self.a_long + self.b_long * n_long
        return tt_short, tt_long

    def get_signal(self, n_short: int, n_long: int, flow: int,
                   sim_time: float, ema_short: float, ema_long: float
                   ) -> tuple:
        """Generate routing recommendation for a departing vehicle."""
        self.signals_issued += 1

        # --- Pass-through mode during warmup (only if NOT pre-loaded) ---
        if sim_time <= self.warmup_sec and not self.preloaded:
            signal = "short" if ema_short <= ema_long else "long"
            if signal == "short":
                self.signals_short += 1
            else:
                self.signals_long += 1
            return signal, {
                "mode": "passthrough",
                "pred_short": None, "pred_long": None,
                "gap_physical": 0.0, "ic_margin": 0.0,
                "ic_constrained": False,
            }

        # --- Active mode ---
        if not self.model_fitted or self.n_short_so is None:
            # Model not yet fitted — fall back to passthrough
            signal = "short" if ema_short <= ema_long else "long"
            if signal == "short":
                self.signals_short += 1
            else:
                self.signals_long += 1
            return signal, {
                "mode": "passthrough_fallback",
                "pred_short": None, "pred_long": None,
                "gap_physical": 0.0, "ic_margin": 0.0,
                "ic_constrained": False,
            }

        pred_short, pred_long = self.predict_tt(n_short, n_long)
        n_total = n_short + n_long

        # Proportional SO targeting: use the SO split RATIO (not absolute count)
        # n_short_so was computed at steady-state n_total, so the ratio is stable
        if self.n_short_so is not None and hasattr(self, '_n_total_ref'):
            target_pct = self.n_short_so / max(1.0, self._n_total_ref)
            target_pct = max(0.0, min(1.0, target_pct))
        elif self.n_short_so is not None and n_total > 0:
            # Fallback: use current n_total but cap at reasonable range
            target_pct = min(0.65, max(0.35, self.n_short_so / max(1.0, n_total)))
        else:
            target_pct = 0.5

        # Deterministic split: use signals_issued as counter
        want_short = (self.signals_issued % 100) < int(target_pct * 100)

        if want_short:
            signal = "short"
            ic_margin = self.ic_margin_short
            gap_physical = pred_short - pred_long  # penalty for choosing short over long
            ic_constrained = False
            # IC check: agent must accept short isn't too much worse than long
            if gap_physical > ic_margin:
                signal = "long"  # can't send to short, IC violated
                ic_constrained = True
                self.ic_constrained_count += 1
        else:
            signal = "long"
            ic_margin = self.ic_margin_long
            gap_physical = pred_long - pred_short  # penalty for choosing long over short
            ic_constrained = False
            # IC check: agent must accept long isn't too much worse than short
            if gap_physical > ic_margin:
                signal = "short"  # can't send to long, IC violated
                ic_constrained = True
                self.ic_constrained_count += 1

        if signal == "short":
            self.signals_short += 1
        else:
            self.signals_long += 1

        return signal, {
            "mode": "active",
            "pred_short": round(pred_short, 2),
            "pred_long": round(pred_long, 2),
            "gap_physical": round(gap_physical, 2),
            "ic_margin": round(ic_margin, 2),
            "ic_constrained": ic_constrained,
        }

    def get_ic_margin(self, path: str) -> float:
        """Get current IC margin for a path."""
        return self.ic_margin_short if path == "short" else self.ic_margin_long

    def get_diagnostics(self) -> dict:
        return {
            "signals_issued": self.signals_issued,
            "signals_short": self.signals_short,
            "signals_long": self.signals_long,
            "ic_constrained_count": self.ic_constrained_count,
            "margin_saturation_pct": round(
                100 * self.ic_constrained_count / max(1, self.signals_issued), 2
            ),
            "refit_count": self.refit_count,
            "model_fitted": self.model_fitted,
            "a_short": round(self.a_short, 2),
            "b_short": round(self.b_short, 4),
            "a_long": round(self.a_long, 2),
            "b_long": round(self.b_long, 4),
            "n_short_so": round(self.n_short_so, 1) if self.n_short_so else None,
            "ic_margin_short": round(self.ic_margin_short, 2),
            "ic_margin_long": round(self.ic_margin_long, 2),
            "n_history_short": len(self.history_short),
            "n_history_long": len(self.history_long),
        }


class VehicleLogger:
    """Logs per-vehicle data to CSV."""
    
    FIELDNAMES = [
        "vid", "group_id", "assigned_kmh", "assigned_tau",
        "income_gbp", "trip_purpose", "vot_usd_hr", "toll_paid_usd",
        "cost_short", "cost_long",
        "chosen_path", "choice_type",
        "departure_time", "arrival_time", "travel_time",
        "ema_short_at_choice", "ema_long_at_choice"
    ]
    
    def __init__(self, output_file: str = "phase2_vehicle_log.csv"):
        self.output_file = output_file
        self.records = []
    
    def log(self, record: dict):
        """Add a record to the log."""
        self.records.append(record)
    
    def save(self, filename: str = None):
        """Write all records to CSV."""
        target_file = filename if filename else self.output_file
        with open(target_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Saved {len(self.records)} vehicle records to {target_file}")


class StepLogger:
    """Logs aggregate metrics every simulation step."""
    
    def __init__(self, output_file: str = "phase2_step_log.csv"):
        self.output_file = output_file
        self.records = []
    
    def log(self, sim_time: float, count_short: int, count_long: int, 
            ema_short: float, ema_long: float):
        """Log metrics for current step."""
        total = count_short + count_long
        pct_short = (count_short / total * 100) if total > 0 else 50.0
        tt_diff = abs(ema_short - ema_long)
        
        self.records.append({
            "sim_time": round(sim_time, 1),
            "count_short": count_short,
            "count_long": count_long,
            "pct_short": round(pct_short, 2),
            "ema_short": round(ema_short, 2),
            "ema_long": round(ema_long, 2),
            "tt_diff": round(tt_diff, 2)
        })
    
    def save(self, filename: str = None):
        """Write all records to CSV."""
        target_file = filename if filename else self.output_file
        fieldnames = ["sim_time", "count_short", "count_long", "pct_short", 
                      "ema_short", "ema_long", "tt_diff"]
        with open(target_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"Saved {len(self.records)} step records to {target_file}")
