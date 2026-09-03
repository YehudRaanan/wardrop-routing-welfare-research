"""Tests for shared_lib/validation.py — route and parameter validation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared_lib.validation import (
    validate_routes,
    validate_run_config,
    validate_flow_rates,
    VALID_BODY_EDGES,
    BANNED_ENTRY_EDGES,
)


# ─────────────────────────────────────────────────────────────────────
# Route Validation
# ─────────────────────────────────────────────────────────────────────

class TestValidateRoutes:
    def test_correct_routes_pass(self):
        routes = {
            "short_fwd": ["short_fwd"],
            "long_fwd": ["long_fwd"],
            "short_bwd": ["short_bwd"],
            "long_bwd": ["long_bwd"],
        }
        errors = validate_routes(routes)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_entry_edges_rejected(self):
        routes = {
            "short_fwd": ["entry_short_fwd", "short_fwd"],
            "long_fwd": ["entry_long_fwd", "long_fwd"],
        }
        errors = validate_routes(routes)
        assert len(errors) == 2
        assert "entry_short_fwd" in errors[0]
        assert "BANNED" in errors[0]

    def test_all_entry_edges_caught(self):
        for entry_edge in BANNED_ENTRY_EDGES:
            routes = {"test": [entry_edge]}
            errors = validate_routes(routes)
            assert len(errors) >= 1, f"Entry edge '{entry_edge}' not caught"

    def test_unknown_edge_rejected(self):
        routes = {"test": ["some_random_edge"]}
        errors = validate_routes(routes)
        assert len(errors) == 1
        assert "unknown edge" in errors[0]

    def test_empty_route_rejected(self):
        routes = {"test": []}
        errors = validate_routes(routes)
        assert len(errors) == 1
        assert "empty" in errors[0]

    def test_single_body_edge_valid(self):
        for edge in VALID_BODY_EDGES:
            routes = {"test": [edge]}
            errors = validate_routes(routes)
            assert errors == [], f"Body edge '{edge}' should be valid"


# ─────────────────────────────────────────────────────────────────────
# RunConfig Validation
# ─────────────────────────────────────────────────────────────────────

class TestValidateRunConfig:
    def test_valid_ema_config(self):
        config = {
            "flow_rate": 2000,
            "route_choice_method": "ema",
            "ema_alpha": 0.1,
            "sim_duration_sec": 7200,
            "warmup_sec": 3600,
            "step_length": 0.2,
            "fwd_ratio": 0.78,
            "bwd_ratio": 0.22,
        }
        errors = validate_run_config(config)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_wrong_duration_flagged(self):
        config = {
            "flow_rate": 2000,
            "route_choice_method": "ema",
            "ema_alpha": 0.1,
            "sim_duration_sec": 3600,
        }
        errors = validate_run_config(config)
        assert any("sim_duration_sec" in e for e in errors)

    def test_wrong_warmup_flagged(self):
        config = {
            "flow_rate": 2000,
            "route_choice_method": "ema",
            "ema_alpha": 0.1,
            "warmup_sec": 1800,
        }
        errors = validate_run_config(config)
        assert any("warmup_sec" in e for e in errors)

    def test_ema_without_alpha_flagged(self):
        config = {
            "flow_rate": 2000,
            "route_choice_method": "ema",
        }
        errors = validate_run_config(config)
        assert any("ema_alpha" in e for e in errors)

    def test_fixed_split_without_pct_flagged(self):
        config = {
            "flow_rate": 2000,
            "route_choice_method": "fixed_split",
        }
        errors = validate_run_config(config)
        assert any("pct_short_fixed" in e for e in errors)

    def test_invalid_method_flagged(self):
        config = {
            "flow_rate": 2000,
            "route_choice_method": "invalid_method",
        }
        errors = validate_run_config(config)
        assert any("valid method" in e for e in errors)

    def test_negative_flow_flagged(self):
        config = {
            "flow_rate": -500,
            "route_choice_method": "ema",
            "ema_alpha": 0.1,
        }
        errors = validate_run_config(config)
        assert any("positive" in e for e in errors)

    def test_non_100_flow_flagged(self):
        config = {
            "flow_rate": 1250,
            "route_choice_method": "ema",
            "ema_alpha": 0.1,
        }
        errors = validate_run_config(config)
        assert any("multiple of 100" in e for e in errors)

    def test_defaults_pass(self):
        """Config with only required fields + defaults should pass."""
        config = {
            "flow_rate": 2000,
            "route_choice_method": "davis",
        }
        errors = validate_run_config(config)
        assert errors == [], f"Expected no errors with defaults, got: {errors}"


# ─────────────────────────────────────────────────────────────────────
# Flow Rate Validation
# ─────────────────────────────────────────────────────────────────────

class TestValidateFlowRates:
    def test_valid_500_step(self):
        flows = list(range(500, 6000, 500))
        warnings = validate_flow_rates(flows, "test sweep")
        assert warnings == []

    def test_valid_100_step(self):
        flows = list(range(2000, 4100, 100))
        warnings = validate_flow_rates(flows, "fine sweep")
        assert warnings == []

    def test_inconsistent_steps_warned(self):
        flows = [500, 1000, 1500, 2000, 2100, 2200]
        warnings = validate_flow_rates(flows, "mixed sweep")
        assert any("Inconsistent" in w for w in warnings)

    def test_empty_list_warned(self):
        warnings = validate_flow_rates([], "empty")
        assert len(warnings) == 1

    def test_duplicates_warned(self):
        flows = [500, 1000, 500, 1500]
        warnings = validate_flow_rates(flows, "dup test")
        assert any("Duplicate" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
