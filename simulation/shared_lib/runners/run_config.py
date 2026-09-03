"""
RunConfig — immutable simulation configuration.

frozen=True prevents mutation after creation, making it safe
to use as a deduplication key.

Parameters are split into:
    DEDUP KEY    — uniquely identifies a run (matches DB UNIQUE constraint)
    LOCKED       — must match project standards (validated before run)
    DERIVED      — computed from other fields
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True, eq=False)  # eq=False because dict field breaks hash
class RunConfig:
    """Immutable configuration for a single simulation run."""

    # ═══════════════════════════════════════════════════════════════
    # DEDUP KEY (matches runs table UNIQUE constraint)
    # ═══════════════════════════════════════════════════════════════
    flow_rate: int                              # veh/h (must be multiple of 100)
    route_choice_method: str                    # 'ema','davis','mediated','fixed_split','thompson','none'
    agent_mix: str = "100B"                     # '100B','100C','40A_60B', etc.
    ema_alpha: float | None = None              # Required if method='ema'
    logit_theta: float | None = None            # Phase 5+ Stochastic Logit scale parameter (1/seconds)
    toll_short_usd: float = 0.0                 # Toll on short path (Phase 5+)
    pct_short_fixed: float | None = None        # Required if method='fixed_split' (0.0-1.0)
    demand_elasticity: float | None = None      # Phase 6 only
    random_seed: int | None = None              # For reproducibility

    # ═══════════════════════════════════════════════════════════════
    # LOCKED PARAMETERS — DO NOT OVERRIDE
    # Source: CLAUDE.md Simulation Standards + calibration_config.py
    # ═══════════════════════════════════════════════════════════════
    sim_duration_sec: int = 7200                # 120 min — LOCKED
    warmup_sec: int = 3600                      # 60 min — LOCKED
    step_length: float = 0.2                    # — LOCKED (2026-04-16: 0.1→0.2 for perf)
    fwd_ratio: float = 0.78                     # — LOCKED (from calibration)
    bwd_ratio: float = 0.22                     # — LOCKED (from calibration)

    # ═══════════════════════════════════════════════════════════════
    # PHASE-SPECIFIC OPTIONS (not part of dedup key)
    # ═══════════════════════════════════════════════════════════════
    target_path: str | None = None              # 'short' or 'long' — for capacity discovery (all vehicles on one path)
    notes: str = ""                             # Free-form annotation

    # ═══════════════════════════════════════════════════════════════
    # MIXED-METHOD POPULATION (P2; 2026-05-09).
    # If set, each vehicle is independently assigned a routing method
    # at injection per these probabilities. Overrides the per-run
    # `route_choice_method` for the actual choice but the latter is still
    # used for backwards-compat (set to one of the mix keys, or 'mixed').
    # Example: method_mix={"davis": 0.4, "ema": 0.4, "fixed_split_vot": 0.2}
    # ═══════════════════════════════════════════════════════════════
    method_mix: Optional[dict] = None

    # 2026-05-11: matched-N comparison runs — override Davis rolling-buffer size.
    # None = use class default (BUFFER_SIZE=500). Set to 199 to match EMA α=0.01.
    davis_buffer_size: Optional[int] = None

    # 2026-05-13 (P3 sensitivity): when True, the per-minute in-transit
    # sampling block in unified_runner.py is SKIPPED, so EMA and Davis are
    # fed ONLY by Source A (vehicle arrivals). Default False preserves the
    # production behaviour (Source A + Source B, ~15 % / 85 % composition
    # at the operational flow point). See
    # analyses/P3_sensitivity_in_transit/README.md for design.
    disable_intransit_feed: bool = False

    @property
    def sim_steps(self) -> int:
        return int(self.sim_duration_sec / self.step_length)

    @property
    def warmup_steps(self) -> int:
        return int(self.warmup_sec / self.step_length)

    @property
    def fwd_flow(self) -> float:
        return self.flow_rate * self.fwd_ratio

    @property
    def bwd_flow(self) -> float:
        return self.flow_rate * self.bwd_ratio

    def to_db_params(self) -> dict:
        """Convert to dict for DB insertion (runs table columns)."""
        return {
            "flow_rate": self.flow_rate,
            "route_choice_method": self.route_choice_method,
            "agent_mix": self.agent_mix,
            "ema_alpha": self.ema_alpha,
            "logit_theta": self.logit_theta,
            "toll_short_usd": self.toll_short_usd,
            "pct_short_fixed": self.pct_short_fixed,
            "demand_elasticity": self.demand_elasticity,
            "random_seed": self.random_seed,
            "sim_duration_sec": self.sim_duration_sec,
            "warmup_sec": self.warmup_sec,
            "step_length": self.step_length,
            "fwd_ratio": self.fwd_ratio,
            "bwd_ratio": self.bwd_ratio,
            "notes": self.notes,
        }
