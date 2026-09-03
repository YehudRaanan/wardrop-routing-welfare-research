"""
Parallel sweep execution with dedup and interleaved load balancing.

Runs multiple flow rates in parallel using multiprocessing.Pool.
Skips runs that already exist in the DB (dedup).
Uses interleaved worker assignment for load balancing.
"""

import multiprocessing
from typing import Any

from .run_config import RunConfig
from .unified_runner import run_simulation
from ..db.connection import get_fresh_connection, DB_PATH
from ..db.dedup import find_existing_run


def _run_single(args: tuple) -> int:
    """Worker function for multiprocessing.Pool.map().

    Args:
        args: (RunConfig, force, db_path) tuple

    Returns:
        run_id
    """
    config, force, db_path = args
    return run_simulation(config, force=force, db_path=db_path)


def run_sweep(
    configs: list[RunConfig],
    n_workers: int = 6,
    force: bool = False,
    db_path: str | None = None,
) -> list[int]:
    """Run a sweep of configurations with parallel execution + dedup.

    Checks DB for existing runs before launching workers.
    Uses interleaved load balancing (sort by flow desc, round-robin).

    Args:
        configs: List of RunConfig instances
        n_workers: Number of parallel workers (default 6, machine has 12 cores)
        force: If True, skip dedup and re-run everything
        db_path: Override DB path

    Returns:
        List of run_ids (some cached, some newly run)
    """
    path = db_path or str(DB_PATH)

    # ── Dedup check ──
    cached = {}      # config index -> run_id
    to_run = []      # (config, original_index) for configs that need running

    if not force:
        conn = get_fresh_connection(path)
        for i, cfg in enumerate(configs):
            existing = find_existing_run(
                conn,
                flow_rate=cfg.flow_rate,
                route_choice_method=cfg.route_choice_method,
                agent_mix=cfg.agent_mix,
                ema_alpha=cfg.ema_alpha,
                logit_theta=cfg.logit_theta,
                toll_short_usd=cfg.toll_short_usd,
                pct_short_fixed=cfg.pct_short_fixed,
                demand_elasticity=cfg.demand_elasticity,
                random_seed=cfg.random_seed,
                sim_duration_sec=cfg.sim_duration_sec,
                warmup_sec=cfg.warmup_sec,
            )
            if existing is not None:
                cached[i] = existing
                print(f"  CACHED: {cfg.route_choice_method}@{cfg.flow_rate} "
                      f"-> run_id={existing}")
            else:
                to_run.append((cfg, i))
        conn.close()
    else:
        to_run = [(cfg, i) for i, cfg in enumerate(configs)]

    if not to_run:
        print(f"\nAll {len(configs)} runs already in DB. Nothing to do.")
        return [cached.get(i, -1) for i in range(len(configs))]

    print(f"\nRunning {len(to_run)} new simulations "
          f"({len(cached)} cached, {n_workers} workers)")

    # ── Interleaved load balancing ──
    # Sort by flow rate descending (high flows take longer)
    to_run.sort(key=lambda x: x[0].flow_rate, reverse=True)

    # Round-robin into worker buckets
    # This ensures each worker gets a mix of light and heavy flows
    worker_args = [(cfg, force, path) for cfg, _ in to_run]

    # ── Parallel execution ──
    if n_workers > 1 and len(worker_args) > 1:
        with multiprocessing.Pool(n_workers) as pool:
            new_ids = pool.map(_run_single, worker_args)
    else:
        # Sequential for single worker or single config
        new_ids = [_run_single(args) for args in worker_args]

    # ── Merge results ──
    # Map new run_ids back to original indices
    new_id_map = {}
    for (cfg, orig_idx), run_id in zip(to_run, new_ids):
        new_id_map[orig_idx] = run_id

    # Build final result in original order
    result = []
    for i in range(len(configs)):
        if i in cached:
            result.append(cached[i])
        elif i in new_id_map:
            result.append(new_id_map[i])
        else:
            result.append(-1)  # Should not happen

    return result


def build_sweep_configs(
    flow_rates: list[int],
    route_choice_method: str,
    **kwargs: Any,
) -> list[RunConfig]:
    """Build a list of RunConfigs for a sweep.

    Convenience function that creates one RunConfig per flow rate
    with shared parameters.

    Args:
        flow_rates: List of flow rates to sweep
        route_choice_method: Method for all runs
        **kwargs: Additional RunConfig parameters (ema_alpha, agent_mix, etc.)

    Returns:
        List of RunConfig instances
    """
    configs = []
    for flow in flow_rates:
        configs.append(RunConfig(
            flow_rate=flow,
            route_choice_method=route_choice_method,
            **kwargs,
        ))
    return configs
