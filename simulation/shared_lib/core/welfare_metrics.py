"""
Welfare Metrics Module
======================

Implements social welfare calculations based on academic foundations:
- Total Travel Cost (TTC)
- Price of Anarchy (PoA)
- Consumer Surplus
- Toll Revenue
- Social Welfare (B)
- Per-Agent IC Violation Check (Wardrop validation under heterogeneous VOT)

References:
- Ferrari (2002) - Road network toll pricing and social welfare
- Roughgarden & Tardos (2002) - How bad is selfish routing
- Rouwendal & Verhoef (2006) - Basic economic principles of road pricing
- UK HM Treasury Green Book (2022) - Appraisal and evaluation in central government
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
import math


@dataclass
class AgentMetrics:
    """Metrics for a single agent/vehicle."""
    vid: str
    vot_usd_hr: float          # Value of Time ($/hr)
    travel_time_sec: float     # Actual travel time (seconds)
    toll_paid_usd: float       # Toll paid ($)
    chosen_path: str           # "short" or "long"

    # Perceived TT at decision time (EMA or Davis prediction)
    ema_short_at_choice: float = 0.0
    ema_long_at_choice: float = 0.0

    # Departure time (for cohort grouping)
    departure_time_sec: float = 0.0

    # CS pivot v2 (locked 2026-05-26) — stochastic-logit choice columns.
    # For natively logit-eligible agents these come straight from the
    # `vehicles` table. For SO baseline + queued-never-departed agents
    # they are populated by analyses/_shared/cs_backfill.py via the
    # counterfactual-logsum procedure (welfare_measure_dossier.md §1.4, §2.1.1).
    cost_short_at_choice: Optional[float] = None  # generalised cost on short, seconds
    cost_long_at_choice: Optional[float] = None   # generalised cost on long, seconds
    p_short_at_choice: Optional[float] = None     # logit choice probability for short

    @property
    def travel_cost_usd(self) -> float:
        """Time cost in dollars = (time_sec / 3600) * VOT"""
        return (self.travel_time_sec / 3600.0) * self.vot_usd_hr

    @property
    def generalized_cost_usd(self) -> float:
        """Total cost in dollars = time_cost + toll"""
        return self.travel_cost_usd + self.toll_paid_usd

    @property
    def generalized_cost_sec(self) -> float:
        """Total cost in time-equivalent seconds."""
        toll_penalty_sec = (self.toll_paid_usd * 3600.0) / self.vot_usd_hr if self.vot_usd_hr > 0 else 0
        return self.travel_time_sec + toll_penalty_sec

    def consumer_surplus(self, theta: float = 0.05) -> Optional[float]:
        """
        Per-driver consumer surplus via the Williams (1977) logsum theorem.

        CS_i = (VOT_i / (3600 * theta)) * log( exp(-theta * GC_short) + exp(-theta * GC_long) )    [USD]

        See welfare_measure_dossier.md §1.1 for the derivation.
        Returns None if cost_short_at_choice / cost_long_at_choice are
        not populated (the agent is not CS-eligible — e.g. habit /
        fixed-split-blind in a mixed run with no counterfactual backfill).

        Args:
            theta: Logit smoothing scale (1/seconds). Default 0.05 = LOGIT_THETA
                locked by Phase 0 decision 0.1.

        Returns:
            Per-driver CS in USD, or None if the at-choice columns are absent.
        """
        if self.cost_short_at_choice is None or self.cost_long_at_choice is None:
            return None
        if self.vot_usd_hr <= 0:
            return None
        # Log-sum-exp trick for numerical stability:
        v_short = -theta * self.cost_short_at_choice
        v_long = -theta * self.cost_long_at_choice
        m = max(v_short, v_long)
        logsum = m + math.log(math.exp(v_short - m) + math.exp(v_long - m))
        return (self.vot_usd_hr / (3600.0 * theta)) * logsum


@dataclass
class WelfareMetrics:
    """
    Calculates welfare metrics from simulation results.

    Based on Ferrari (2002) social welfare function:
    B = S + λR - (1+λ)K

    Where:
    - S = Driver surplus (time savings)
    - R = Toll revenue
    - K = Road costs (not modeled here)
    - λ = Marginal cost of public funds
    """

    agents: List[AgentMetrics] = field(default_factory=list)
    baseline_ttc_usd: Optional[float] = None  # TTC without intervention (for PoA)

    # -------------------------------------------------------------------------
    # Core Metrics
    # -------------------------------------------------------------------------

    @property
    def total_travel_time_sec(self) -> float:
        """Sum of all travel times in seconds."""
        return sum(a.travel_time_sec for a in self.agents)

    @property
    def total_travel_time_hr(self) -> float:
        """Sum of all travel times in hours."""
        return self.total_travel_time_sec / 3600.0

    @property
    def total_travel_cost_usd(self) -> float:
        """
        Total Travel Cost (TTC) in dollars.
        TTC = Σ (travel_time_i × VOT_i)

        This is the private time cost, excluding tolls.
        """
        return sum(a.travel_cost_usd for a in self.agents)

    @property
    def total_toll_revenue_usd(self) -> float:
        """Total toll revenue collected in dollars."""
        return sum(a.toll_paid_usd for a in self.agents)

    @property
    def total_generalized_cost_usd(self) -> float:
        """
        Total Generalized Cost in dollars.
        TGC = Σ (time_cost_i + toll_i)

        This is what drivers actually experience.
        """
        return sum(a.generalized_cost_usd for a in self.agents)

    # -------------------------------------------------------------------------
    # Price of Anarchy (Academic Definition)
    # -------------------------------------------------------------------------
    #
    # Per Correa & Stier-Moses "Wardrop Equilibria":
    # - PoA is measured in TOTAL TRAVEL TIME (no VOT, no toll)
    # - SO is INDEPENDENT of toll (toll is a transfer payment)
    # - There is ONE SO per flow level, used for all toll values
    #
    # We provide two PoA metrics:
    # 1. PoA_classic: TTT_UE / TTT_SO (unweighted, academic standard)
    # 2. PoA_vot: TTC_UE / TTC_SO (VOT-weighted, economic value)
    # -------------------------------------------------------------------------

    def poa_classic(self, so_total_time_sec: float) -> float:
        """
        Classic Price of Anarchy (academic standard).

        PoA_classic = TTT_UE / TTT_SO

        Where:
        - TTT_UE = Total Travel Time at User Equilibrium (this simulation)
        - TTT_SO = Total Travel Time at System Optimum (from SO simulation)

        Per Correa & Stier-Moses "Wardrop Equilibria":
        - All travelers' time weighted equally
        - SO is toll-independent (one per flow level)

        Args:
            so_total_time_sec: Total travel time from SO simulation (seconds)

        Returns:
            PoA ratio (always >= 1.0)
        """
        if so_total_time_sec <= 0:
            return 1.0
        return self.total_travel_time_sec / so_total_time_sec

    def poa_vot_weighted(self, so_total_cost_usd: float) -> float:
        """
        VOT-Weighted Price of Anarchy (economic value perspective).

        PoA_vot = TTC_UE / TTC_SO

        Where:
        - TTC_UE = Total Travel Cost at User Equilibrium (VOT-weighted)
        - TTC_SO = Total Travel Cost at System Optimum (VOT-weighted)

        This metric reflects economic value:
        - High-VOT travelers' time counts more
        - Shows monetary inefficiency of selfish routing

        Args:
            so_total_cost_usd: Total travel cost from SO simulation ($)

        Returns:
            PoA ratio (always >= 1.0)
        """
        if so_total_cost_usd <= 0:
            return 1.0
        return self.total_travel_cost_usd / so_total_cost_usd

    @property
    def price_of_anarchy(self) -> Optional[float]:
        """
        Legacy PoA using static baseline.

        DEPRECATED: Use poa_classic() or poa_vot_weighted() with SO simulation results.

        Requires baseline_ttc_usd to be set externally.
        """
        if self.baseline_ttc_usd is None or self.baseline_ttc_usd == 0:
            return None
        return self.total_travel_cost_usd / self.baseline_ttc_usd

    @property
    def efficiency_gain_pct(self) -> Optional[float]:
        """
        Percentage improvement over baseline (legacy).

        DEPRECATED: Use poa_classic() or poa_vot_weighted() instead.
        """
        if self.baseline_ttc_usd is None or self.baseline_ttc_usd == 0:
            return None
        return (self.baseline_ttc_usd - self.total_travel_cost_usd) / self.baseline_ttc_usd * 100

    # -------------------------------------------------------------------------
    # Route Statistics
    # -------------------------------------------------------------------------

    @property
    def short_path_count(self) -> int:
        """Number of agents choosing short path."""
        return sum(1 for a in self.agents if a.chosen_path == "short")

    @property
    def long_path_count(self) -> int:
        """Number of agents choosing long path."""
        return sum(1 for a in self.agents if a.chosen_path == "long")

    @property
    def short_path_pct(self) -> float:
        """Percentage of agents on short path."""
        total = len(self.agents)
        return (self.short_path_count / total * 100) if total > 0 else 0

    @property
    def avg_travel_time_short_sec(self) -> float:
        """Average travel time on short path."""
        short_agents = [a for a in self.agents if a.chosen_path == "short"]
        if not short_agents:
            return 0
        return sum(a.travel_time_sec for a in short_agents) / len(short_agents)

    @property
    def avg_travel_time_long_sec(self) -> float:
        """Average travel time on long path."""
        long_agents = [a for a in self.agents if a.chosen_path == "long"]
        if not long_agents:
            return 0
        return sum(a.travel_time_sec for a in long_agents) / len(long_agents)

    @property
    def travel_time_gap_sec(self) -> float:
        """Difference between short and long path travel times."""
        return abs(self.avg_travel_time_short_sec - self.avg_travel_time_long_sec)

    @property
    def wardrop_gap_pct(self) -> float:
        """
        Wardrop equilibrium gap as percentage (TRAVEL TIME BASED).
        Gap < 5% indicates approximate equilibrium.

        NOTE: This is only valid for toll=0. For tolled scenarios,
        use wardrop_gap_gc_pct which uses generalized costs.
        """
        avg_time = (self.avg_travel_time_short_sec + self.avg_travel_time_long_sec) / 2
        if avg_time == 0:
            return 0
        return (self.travel_time_gap_sec / avg_time) * 100

    @property
    def avg_generalized_cost_short_sec(self) -> float:
        """Average generalized cost on short path (in seconds)."""
        short_agents = [a for a in self.agents if a.chosen_path == "short"]
        if not short_agents:
            return 0
        # GC = TT + (toll * 3600 / VOT)
        return sum(a.travel_time_sec + (a.toll_paid_usd * 3600 / a.vot_usd_hr)
                   for a in short_agents) / len(short_agents)

    @property
    def avg_generalized_cost_long_sec(self) -> float:
        """Average generalized cost on long path (in seconds)."""
        long_agents = [a for a in self.agents if a.chosen_path == "long"]
        if not long_agents:
            return 0
        # Long path is toll-free, so GC = TT
        return sum(a.travel_time_sec for a in long_agents) / len(long_agents)

    @property
    def generalized_cost_gap_sec(self) -> float:
        """Difference between short and long path generalized costs."""
        return abs(self.avg_generalized_cost_short_sec - self.avg_generalized_cost_long_sec)

    @property
    def wardrop_gap_gc_pct(self) -> float:
        """
        Wardrop equilibrium gap based on GENERALIZED COST (correct for tolled scenarios).
        With heterogeneous VOT, equilibrium means agents with higher VOT take the
        tolled path and agents with lower VOT take the free path.
        Gap < 2% indicates approximate equilibrium.
        """
        avg_gc = (self.avg_generalized_cost_short_sec + self.avg_generalized_cost_long_sec) / 2
        if avg_gc == 0:
            return 0
        return (self.generalized_cost_gap_sec / avg_gc) * 100

    @property
    def is_equilibrium(self) -> bool:
        """True if Wardrop gap (generalized cost based) < 2%."""
        return self.wardrop_gap_gc_pct < 2.0

    # -------------------------------------------------------------------------
    # Per-Agent IC Violation Check (GAP 2.4)
    # -------------------------------------------------------------------------
    #
    # With heterogeneous VOT, average GC gap across paths is NOT a valid
    # Wardrop test. Instead, check each agent individually:
    #
    # Prospective: Did the agent choose correctly given perceived TT?
    #   GC_chosen_perceived vs GC_alt_perceived (using EMA/Davis at choice time)
    #
    # Retrospective: Was the agent actually on the correct path?
    #   GC_chosen_actual vs GC_alt_actual (actual TT vs cohort avg TT on alt)
    #
    # An agent is "misallocated" if switching paths would lower their GC.
    # -------------------------------------------------------------------------

    def ic_violation_check(self, cohort_window_sec: float = 60.0) -> Dict:
        """
        Per-agent Incentive Compatibility violation check.

        For each agent, computes whether they would reduce their generalized
        cost by switching to the alternative path — both prospectively (using
        perceived TT at decision time) and retrospectively (using actual TT
        with cohort-average TT on the alternative path).

        Args:
            cohort_window_sec: Departure window for grouping agents (default 60s).

        Returns:
            Dict with:
                - n_agents: Total agents checked
                - prospective_violations: Count of agents who chose wrong given beliefs
                - prospective_violation_pct: As percentage
                - retrospective_violations: Count of agents on wrong path ex-post
                - retrospective_violation_pct: As percentage
                - retrospective_checked: Agents with cohort data on alt path
                - per_agent: List of per-agent dicts (for detailed analysis)
        """
        if not self.agents:
            return {
                "n_agents": 0,
                "prospective_violations": 0, "prospective_violation_pct": 0.0,
                "retrospective_violations": 0, "retrospective_violation_pct": 0.0,
                "retrospective_checked": 0,
                "per_agent": [],
            }

        # --- Build departure cohorts for retrospective alt-path actual TT ---
        # Key: (cohort_minute, path) -> list of actual TTs
        cohorts: Dict[tuple, List[float]] = defaultdict(list)
        for a in self.agents:
            cohort_key = int(a.departure_time_sec / cohort_window_sec)
            cohorts[(cohort_key, a.chosen_path)].append(a.travel_time_sec)

        # Pre-compute cohort averages
        cohort_avg: Dict[tuple, float] = {}
        for key, tts in cohorts.items():
            cohort_avg[key] = sum(tts) / len(tts)

        prospective_violations = 0
        retrospective_violations = 0
        retrospective_checked = 0
        per_agent = []

        for a in self.agents:
            vot = a.vot_usd_hr
            if vot <= 0:
                continue

            toll_penalty_sec = (a.toll_paid_usd * 3600.0) / vot
            alt_path = "long" if a.chosen_path == "short" else "short"

            # --- Prospective (perceived at decision time) ---
            if a.chosen_path == "short":
                gc_chosen_perceived = a.ema_short_at_choice + toll_penalty_sec
                gc_alt_perceived = a.ema_long_at_choice
            else:
                gc_chosen_perceived = a.ema_long_at_choice
                # Alt is short (tolled)
                alt_toll_penalty = (a.toll_paid_usd * 3600.0) / vot if a.chosen_path == "long" else 0
                gc_alt_perceived = a.ema_short_at_choice + alt_toll_penalty

            prospective_ok = gc_chosen_perceived <= gc_alt_perceived
            if not prospective_ok:
                prospective_violations += 1

            # --- Retrospective (actual TT + cohort avg on alt path) ---
            cohort_key = int(a.departure_time_sec / cohort_window_sec)
            alt_cohort_key = (cohort_key, alt_path)

            retro_violation = None
            gc_chosen_actual = a.travel_time_sec + toll_penalty_sec
            gc_alt_actual = None

            if alt_cohort_key in cohort_avg:
                retrospective_checked += 1
                alt_tt = cohort_avg[alt_cohort_key]
                if alt_path == "short":
                    # Would pay toll on short
                    gc_alt_actual = alt_tt + toll_penalty_sec
                else:
                    # Long is toll-free
                    gc_alt_actual = alt_tt

                retro_violation = gc_chosen_actual > gc_alt_actual
                if retro_violation:
                    retrospective_violations += 1

            per_agent.append({
                "vid": a.vid,
                "chosen_path": a.chosen_path,
                "vot_usd_hr": round(vot, 2),
                "gc_chosen_perceived": round(gc_chosen_perceived, 2),
                "gc_alt_perceived": round(gc_alt_perceived, 2),
                "prospective_violation": not prospective_ok,
                "gc_chosen_actual": round(gc_chosen_actual, 2),
                "gc_alt_actual": round(gc_alt_actual, 2) if gc_alt_actual is not None else None,
                "retrospective_violation": retro_violation,
            })

        n = len(self.agents)
        return {
            "n_agents": n,
            "prospective_violations": prospective_violations,
            "prospective_violation_pct": round(100.0 * prospective_violations / n, 2),
            "retrospective_violations": retrospective_violations,
            "retrospective_violation_pct": round(
                100.0 * retrospective_violations / retrospective_checked, 2
            ) if retrospective_checked > 0 else 0.0,
            "retrospective_checked": retrospective_checked,
            "per_agent": per_agent,
        }

    # -------------------------------------------------------------------------
    # VOT Statistics
    # -------------------------------------------------------------------------

    @property
    def avg_vot_usd_hr(self) -> float:
        """Average Value of Time across all agents."""
        if not self.agents:
            return 0
        return sum(a.vot_usd_hr for a in self.agents) / len(self.agents)

    @property
    def avg_vot_short_usd_hr(self) -> float:
        """Average VOT of agents choosing short path."""
        short_agents = [a for a in self.agents if a.chosen_path == "short"]
        if not short_agents:
            return 0
        return sum(a.vot_usd_hr for a in short_agents) / len(short_agents)

    @property
    def avg_vot_long_usd_hr(self) -> float:
        """Average VOT of agents choosing long path."""
        long_agents = [a for a in self.agents if a.chosen_path == "long"]
        if not long_agents:
            return 0
        return sum(a.vot_usd_hr for a in long_agents) / len(long_agents)

    # -------------------------------------------------------------------------
    # Social Welfare (Ferrari 2002)
    # -------------------------------------------------------------------------

    def social_welfare_change(
        self,
        baseline_ttc_usd: float,
        road_cost_usd: float = 0,
        marginal_cost_public_funds: float = 0.2
    ) -> float:
        """
        Calculate social welfare change (B) from Ferrari (2002):

        B = S + λR - (1+λ)K

        Args:
            baseline_ttc_usd: TTC before intervention
            road_cost_usd: Annual road costs (K). Default 0 — marginal trips
                on existing infrastructure; no construction/maintenance cost
                attributable to the toll. (UK Green Book, 2022)
            marginal_cost_public_funds: λ (default 0.2; LOCKED 2026-05-26 by
                CS-pivot Phase 0.6 — UK HM Treasury Green Book 2022 MCPF).
                Pre-pivot default was 0.3; the value was reconciled to
                the 2022 Green Book vintage as part of the CS pivot. See
                analyses/_synthesis/welfare_measure_dossier.md §5.3.

        Returns:
            B in dollars (positive = welfare gain)
        """
        S = baseline_ttc_usd - self.total_travel_cost_usd  # Driver surplus
        R = self.total_toll_revenue_usd
        K = road_cost_usd
        lam = marginal_cost_public_funds

        B = S + lam * R - (1 + lam) * K
        return B

    # -------------------------------------------------------------------------
    # CS pivot v2 (LOCKED 2026-05-26) — Williams logsum CS, TCS, W
    #
    # Headline welfare stack under the CS pivot. Per analyses/_synthesis/
    # welfare_measure_dossier.md §1.1–§1.2:
    #   CS_i  =  (VOT_i / (3600·θ)) · ln( e^(-θ·GC_short) + e^(-θ·GC_long) )    [USD]
    #   TCS   =  Σ CS_i  over CS-eligible agents
    #   R     =  τ · N_short_chose
    #   W     =  TCS + (1 + λ) · R                                              with λ = 0.2
    #
    # Absolute TCS / W are interpretable only relative to a baseline (Williams
    # unidentifiable additive constant) — analyses report ΔCS / ΔW, never
    # absolute. The three baselines are computed in analyses/_shared/welfare_v2.py.
    # -------------------------------------------------------------------------

    def total_consumer_surplus(self, theta: float = 0.05) -> float:
        """
        Total Consumer Surplus (TCS) = Σ CS_i over CS-eligible agents.

        Agents whose `cost_short_at_choice` / `cost_long_at_choice` are not
        populated contribute nothing (excluded per methodology.md §0.5; their
        absence is the per-vehicle NULL filter doing its job).

        Args:
            theta: Logit smoothing scale. Default 0.05 (LOGIT_THETA, locked
                by Phase 0 decision 0.1).

        Returns:
            TCS in USD per analysis window. Note: absolute TCS is uninterpretable
            (additive-constant unidentifiability — dossier §1.2); use ΔCS
            against a baseline.
        """
        total = 0.0
        for a in self.agents:
            cs_i = a.consumer_surplus(theta=theta)
            if cs_i is not None:
                total += cs_i
        return total

    def social_welfare(
        self,
        theta: float = 0.05,
        mcpf: float = 0.2,
    ) -> float:
        """
        Social welfare W = TCS + (1 + λ) · R.

        See welfare_measure_dossier.md §1.2 / §5.3 (Small-Verhoef Ch. 4;
        Ferrari 2002). At `τ = 0` we have `R = 0` and therefore `W == TCS`
        (V11 invariant, methodology.md §10).

        Args:
            theta: Logit smoothing scale (LOGIT_THETA = 0.05).
            mcpf: Marginal Cost of Public Funds (MCPF_LAMBDA = 0.2 locked by
                Phase 0 decision 0.6, UK Green Book 2022).

        Returns:
            W in USD per analysis window. Like TCS, only meaningful relative
            to a baseline.
        """
        return self.total_consumer_surplus(theta=theta) + (1.0 + mcpf) * self.total_toll_revenue_usd

    def n_cs_eligible(self) -> int:
        """Count of agents that pass the CS-eligibility filter (have populated
        cost_short_at_choice / cost_long_at_choice). Helper for ``primary.csv``
        column ``n_veh_cs_eligible``.
        """
        return sum(
            1 for a in self.agents
            if a.cost_short_at_choice is not None
            and a.cost_long_at_choice is not None
        )

    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------

    def summary_dict(
        self,
        so_total_time_sec: float = None,
        so_total_cost_usd: float = None,
    ) -> Dict:
        """
        Generate summary dictionary for logging/reporting.

        Args:
            so_total_time_sec: Total travel time from SO simulation (seconds).
                               Required for poa_classic calculation.
            so_total_cost_usd: Total travel cost from SO simulation ($).
                               Required for poa_vot calculation.

        Note: SO values are toll-independent (one per flow level).
              The same SO values should be used for all toll levels at the same flow.
        """
        result = {
            "n_agents": len(self.agents),
            "total_travel_time_sec": round(self.total_travel_time_sec, 1),
            "total_travel_time_hr": round(self.total_travel_time_hr, 2),
            "total_travel_cost_usd": round(self.total_travel_cost_usd, 2),
            "total_toll_revenue_usd": round(self.total_toll_revenue_usd, 2),
            "total_generalized_cost_usd": round(self.total_generalized_cost_usd, 2),
            "short_path_pct": round(self.short_path_pct, 1),
            "avg_tt_short_sec": round(self.avg_travel_time_short_sec, 1),
            "avg_tt_long_sec": round(self.avg_travel_time_long_sec, 1),
            "avg_gc_short_sec": round(self.avg_generalized_cost_short_sec, 1),
            "avg_gc_long_sec": round(self.avg_generalized_cost_long_sec, 1),
            "wardrop_gap_tt_pct": round(self.wardrop_gap_pct, 1),
            "wardrop_gap_gc_pct": round(self.wardrop_gap_gc_pct, 1),
            "is_equilibrium": self.is_equilibrium,
            "avg_vot_usd_hr": round(self.avg_vot_usd_hr, 2),
            "avg_vot_short_usd_hr": round(self.avg_vot_short_usd_hr, 2),
            "avg_vot_long_usd_hr": round(self.avg_vot_long_usd_hr, 2),
        }

        # IC violation check
        ic = self.ic_violation_check()
        result["ic_prospective_violation_pct"] = ic["prospective_violation_pct"]
        result["ic_retrospective_violation_pct"] = ic["retrospective_violation_pct"]

        # Price of Anarchy (requires SO simulation values)
        if so_total_time_sec is not None and so_total_time_sec > 0:
            poa_classic = self.poa_classic(so_total_time_sec)
            result["so_total_time_sec"] = round(so_total_time_sec, 1)
            result["poa_classic"] = round(poa_classic, 3)
            result["efficiency_loss_classic_pct"] = round((poa_classic - 1.0) * 100, 1)
        else:
            result["so_total_time_sec"] = None
            result["poa_classic"] = None
            result["efficiency_loss_classic_pct"] = None

        if so_total_cost_usd is not None and so_total_cost_usd > 0:
            poa_vot = self.poa_vot_weighted(so_total_cost_usd)
            result["so_total_cost_usd"] = round(so_total_cost_usd, 2)
            result["poa_vot"] = round(poa_vot, 3)
            result["efficiency_loss_vot_pct"] = round((poa_vot - 1.0) * 100, 1)
        else:
            result["so_total_cost_usd"] = None
            result["poa_vot"] = None
            result["efficiency_loss_vot_pct"] = None

        return result

    def print_summary(
        self,
        so_total_time_sec: float = None,
        so_total_cost_usd: float = None,
    ):
        """
        Print formatted summary to console.

        Args:
            so_total_time_sec: Total travel time from SO simulation (seconds).
            so_total_cost_usd: Total travel cost from SO simulation ($).
        """
        d = self.summary_dict(so_total_time_sec, so_total_cost_usd)
        print("\n" + "=" * 60)
        print("WELFARE METRICS SUMMARY")
        print("=" * 60)
        print(f"  Agents:              {d['n_agents']}")
        print(f"  Total Travel Time:   {d['total_travel_time_sec']:.1f}s ({d['total_travel_time_hr']:.2f} hr)")
        print(f"  Total Travel Cost:   ${d['total_travel_cost_usd']:.2f}")
        print(f"  Toll Revenue:        ${d['total_toll_revenue_usd']:.2f}")
        print(f"  Generalized Cost:    ${d['total_generalized_cost_usd']:.2f}")
        print("-" * 60)
        print("ROUTE DISTRIBUTION")
        print(f"  Short Path:          {d['short_path_pct']:.1f}%")
        print(f"  Avg TT Short:        {d['avg_tt_short_sec']:.1f}s")
        print(f"  Avg TT Long:         {d['avg_tt_long_sec']:.1f}s")
        print(f"  Avg GC Short:        {d['avg_gc_short_sec']:.1f}s")
        print(f"  Avg GC Long:         {d['avg_gc_long_sec']:.1f}s")
        print(f"  Wardrop Gap (TT):    {d['wardrop_gap_tt_pct']:.1f}%")
        print(f"  Wardrop Gap (GC):    {d['wardrop_gap_gc_pct']:.1f}%")
        print(f"  Equilibrium:         {'YES' if d['is_equilibrium'] else 'NO'}")
        print("-" * 60)
        print("IC VIOLATION CHECK")
        print(f"  Prospective:         {d['ic_prospective_violation_pct']:.1f}% misallocated")
        print(f"  Retrospective:       {d['ic_retrospective_violation_pct']:.1f}% misallocated")
        print("-" * 60)
        print("VALUE OF TIME ANALYSIS")
        print(f"  Avg VOT (all):       ${d['avg_vot_usd_hr']:.2f}/hr")
        print(f"  Avg VOT (short):     ${d['avg_vot_short_usd_hr']:.2f}/hr")
        print(f"  Avg VOT (long):      ${d['avg_vot_long_usd_hr']:.2f}/hr")

        # Price of Anarchy section (requires SO values)
        if d.get('poa_classic') is not None or d.get('poa_vot') is not None:
            print("-" * 60)
            print("PRICE OF ANARCHY")
            if d.get('poa_classic') is not None:
                print(f"  SO Total Time:       {d['so_total_time_sec']:.1f}s")
                print(f"  PoA (Classic):       {d['poa_classic']:.3f}  (TTT_UE / TTT_SO)")
                print(f"  Efficiency Loss:     {d['efficiency_loss_classic_pct']:.1f}%")
            if d.get('poa_vot') is not None:
                print(f"  SO Total Cost:       ${d['so_total_cost_usd']:.2f}")
                print(f"  PoA (VOT-Weighted):  {d['poa_vot']:.3f}  (TTC_UE / TTC_SO)")
                print(f"  Efficiency Loss:     {d['efficiency_loss_vot_pct']:.1f}%")

        print("=" * 60 + "\n")


# -----------------------------------------------------------------------------
# Factory function to create from simulation log
# -----------------------------------------------------------------------------

def from_vehicle_log(records: List[Dict]) -> WelfareMetrics:
    """
    Create WelfareMetrics from vehicle log records.

    Args:
        records: List of dicts with keys:
            - vid, vot_usd_hr, travel_time, toll_paid_usd, chosen_path
            - ema_short_at_choice, ema_long_at_choice (perceived TT)
            - departure_time_sec (for IC cohort grouping)

    Returns:
        WelfareMetrics instance
    """
    agents = []
    for r in records:
        agents.append(AgentMetrics(
            vid=r.get("vid", ""),
            vot_usd_hr=float(r.get("vot_usd_hr", 0)),
            travel_time_sec=float(r.get("travel_time_sec", r.get("travel_time", 0))),
            toll_paid_usd=float(r.get("toll_paid_usd", 0)),
            chosen_path=r.get("chosen_path", "unknown"),
            ema_short_at_choice=float(r.get("ema_short_at_choice", 0)),
            ema_long_at_choice=float(r.get("ema_long_at_choice", 0)),
            departure_time_sec=float(r.get("departure_time_sec", r.get("departure_time", 0))),
        ))
    return WelfareMetrics(agents=agents)


def from_csv(filepath: str) -> WelfareMetrics:
    """Load WelfareMetrics from a vehicle log CSV file."""
    import csv
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        records = list(reader)
    return from_vehicle_log(records)


# -----------------------------------------------------------------------------
# Wardrop Gap Analysis (Actual TT/TC by Departure Time)
# -----------------------------------------------------------------------------

def calculate_wardrop_gap_timeseries(vehicle_records: List[Dict]) -> Dict:
    """
    Calculate Wardrop gap timeseries from per-vehicle actual data.

    Wardrop Equilibrium: At each point in time, vehicles departing simultaneously
    on different paths should experience equal travel times (TT for toll=0) or
    equal total costs (TC for toll>0).

    Args:
        vehicle_records: List of dicts with keys:
            - departure_time_sec: When vehicle departed
            - chosen_path: "short" or "long"
            - actual_tt_sec: Actual travel time
            - actual_tc_sec: Actual total cost (TT + toll penalty)

    Returns:
        Dict with:
            - timeseries: List of {departure_time, gap_tt_pct, gap_tc_pct, ...}
            - gap_tt_min: Minimum TT gap (best convergence)
            - gap_tc_min: Minimum TC gap (best convergence)
            - is_equilibrium_tt: True if gap_tt_min < 2%
            - is_equilibrium_tc: True if gap_tc_min < 2%
    """

    # Group vehicles by departure time (1-second windows)
    departure_groups = defaultdict(lambda: {"short": [], "long": []})
    for v in vehicle_records:
        dep_window = int(float(v.get("departure_time_sec", 0)))
        path = v.get("chosen_path", "unknown")
        if path in ["short", "long"]:
            departure_groups[dep_window][path].append(v)

    # Calculate gap for each departure time
    timeseries = []
    for dep_time in sorted(departure_groups.keys()):
        paths = departure_groups[dep_time]
        if paths["short"] and paths["long"]:
            # Average TT/TC for each path at this departure time
            avg_tt_short = sum(float(v.get("actual_tt_sec", 0)) for v in paths["short"]) / len(paths["short"])
            avg_tt_long = sum(float(v.get("actual_tt_sec", 0)) for v in paths["long"]) / len(paths["long"])
            avg_tc_short = sum(float(v.get("actual_tc_sec", 0)) for v in paths["short"]) / len(paths["short"])
            avg_tc_long = sum(float(v.get("actual_tc_sec", 0)) for v in paths["long"]) / len(paths["long"])

            # Gap at this departure time
            avg_tt = (avg_tt_short + avg_tt_long) / 2
            avg_tc = (avg_tc_short + avg_tc_long) / 2
            gap_tt_pct = abs(avg_tt_short - avg_tt_long) / avg_tt * 100 if avg_tt > 0 else 0
            gap_tc_pct = abs(avg_tc_short - avg_tc_long) / avg_tc * 100 if avg_tc > 0 else 0

            timeseries.append({
                "departure_time_sec": dep_time,
                "gap_tt_pct": round(gap_tt_pct, 2),
                "gap_tc_pct": round(gap_tc_pct, 2),
                "n_short": len(paths["short"]),
                "n_long": len(paths["long"]),
                "avg_tt_short": round(avg_tt_short, 2),
                "avg_tt_long": round(avg_tt_long, 2)
            })

    # Summary: MIN gap (best convergence)
    gap_tt_values = [g["gap_tt_pct"] for g in timeseries]
    gap_tc_values = [g["gap_tc_pct"] for g in timeseries]

    gap_tt_min = min(gap_tt_values) if gap_tt_values else 0
    gap_tc_min = min(gap_tc_values) if gap_tc_values else 0

    return {
        "timeseries": timeseries,
        "gap_tt_min": gap_tt_min,
        "gap_tc_min": gap_tc_min,
        "is_equilibrium_tt": gap_tt_min < 2.0,
        "is_equilibrium_tc": gap_tc_min < 2.0,
        "n_measurements": len(timeseries)
    }
