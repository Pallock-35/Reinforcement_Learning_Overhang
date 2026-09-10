from __future__ import annotations

import unittest

import numpy as np

from mpc_controller import (
    ControllerState,
    MPCConfig,
    conservative_fallback,
    short_horizon_prediction_error,
    simulate_closed_loop_mpc,
    simulate_fixed_baseline,
    solve_mpc,
    summarize_history,
)
from thermal_model import ThermalModelParams
from thermal_targets import dose_window, temperature_target


class MPCControllerTests(unittest.TestCase):
    def test_mpc_returns_bounded_controls(self) -> None:
        cfg = MPCConfig()
        state = ControllerState(
            T=90.0,
            D=50.0,
            u_prev=np.array([100.0, 215.0]),
        )
        u, info = solve_mpc(
            state,
            [55.0] * cfg.horizon,
            ThermalModelParams(),
            cfg,
        )
        self.assertEqual(len(u), 2)
        self.assertGreaterEqual(u[0], cfg.v_min)
        self.assertLessEqual(u[0], cfg.v_max)
        self.assertGreaterEqual(u[1], cfg.T_nozzle_min)
        self.assertLessEqual(u[1], cfg.T_nozzle_max)
        self.assertIn("success", info)
        self.assertIn("fallback", info)

    def test_mpc_config_layer_scale_defaults(self) -> None:
        cfg = MPCConfig()
        self.assertEqual(cfg.v_min, 50.0)
        self.assertEqual(cfg.v_max, 170.0)
        self.assertEqual(cfg.T_nozzle_min, 190.0)
        self.assertEqual(cfg.T_nozzle_max, 230.0)
        self.assertGreaterEqual(cfg.segment_length_mm, 100.0)

    def test_mpc_beats_fixed_on_window_violations(self) -> None:
        mpc_history = simulate_closed_loop_mpc(n_segments=50, seed=13)
        # Over-cooled fixed slicer settings violate the dose window.
        fixed_history = simulate_fixed_baseline(
            n_segments=50,
            seed=13,
            u_fixed=(50.0, 190.0),
        )
        mpc_metrics = summarize_history(mpc_history)
        fixed_metrics = summarize_history(fixed_history)
        self.assertLess(
            mpc_metrics["thermal_window_violation_rate"],
            fixed_metrics["thermal_window_violation_rate"],
        )
        self.assertLess(
            mpc_metrics["mean_abs_T_error"],
            fixed_metrics["mean_abs_T_error"],
        )

    def test_mpc_beats_nominal_fixed_on_tracking(self) -> None:
        mpc_history = simulate_closed_loop_mpc(n_segments=40, seed=11)
        fixed_history = simulate_fixed_baseline(n_segments=40, seed=11)
        mpc_metrics = summarize_history(mpc_history)
        fixed_metrics = summarize_history(fixed_history)
        self.assertLess(
            mpc_metrics["mean_abs_T_error"],
            fixed_metrics["mean_abs_T_error"],
        )
        self.assertLess(
            mpc_metrics["mean_abs_D_error"],
            fixed_metrics["mean_abs_D_error"],
        )

    def test_fallback_on_empty_alphas_raises(self) -> None:
        cfg = MPCConfig()
        state = ControllerState(T=90.0, D=0.0, u_prev=np.array([100.0, 215.0]))
        with self.assertRaises(ValueError):
            solve_mpc(state, [], ThermalModelParams(), cfg)

    def test_conservative_fallback_in_bounds(self) -> None:
        cfg = MPCConfig()
        u = conservative_fallback(cfg)
        self.assertGreaterEqual(u[0], cfg.v_min)
        self.assertLessEqual(u[0], cfg.v_max)
        self.assertGreaterEqual(u[1], cfg.T_nozzle_min)
        self.assertLessEqual(u[1], cfg.T_nozzle_max)
        self.assertLess(u[1], 0.5 * (cfg.T_nozzle_min + cfg.T_nozzle_max))

    def test_dose_targets_reachable_in_sim(self) -> None:
        history = simulate_closed_loop_mpc(n_segments=30, seed=1)
        D_vals = [row["D"] for row in history[5:]]
        windows = [dose_window(row["alpha_deg"]) for row in history[5:]]
        inside = [
            low <= D <= high
            for D, (low, high) in zip(D_vals, windows)
        ]
        self.assertGreater(np.mean(inside), 0.3)

    def test_short_horizon_prediction_error_finite(self) -> None:
        history = simulate_fixed_baseline(n_segments=20, seed=2)
        err = short_horizon_prediction_error(
            history,
            ThermalModelParams(),
            MPCConfig(),
            horizon=3,
        )
        self.assertTrue(np.isfinite(err))
        self.assertGreaterEqual(err, 0.0)

    def test_temperature_target_decreases_with_angle(self) -> None:
        self.assertGreater(temperature_target(45.0), temperature_target(65.0))


if __name__ == "__main__":
    unittest.main()
