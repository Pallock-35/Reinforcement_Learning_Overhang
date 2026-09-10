from __future__ import annotations

import unittest

import numpy as np

from mpc_controller import MPCConfig
from thermal_model import (
    ThermalModelParams,
    analytical_layer_step,
    analytical_thermal_step,
    fit_thermal_model,
    pack_params,
    unpack_params,
)


class ThermalModelTests(unittest.TestCase):
    def test_analytical_step_cools_toward_ambient(self) -> None:
        model = ThermalModelParams(beta=0.0, tau0=10.0, a_fan=0.0, a_alpha=0.0)
        cfg = MPCConfig()
        T_next, D_next, dt = analytical_thermal_step(
            T=100.0,
            D=0.0,
            u=[50.0, 215.0],
            alpha_deg=45.0,
            model=model,
            cfg=cfg,
        )
        self.assertGreater(dt, 0.0)
        self.assertLess(T_next, 100.0)
        self.assertGreater(T_next, model.T_inf)
        self.assertGreater(D_next, 0.0)

    def test_analytical_step_hand_calc_no_fan_effect(self) -> None:
        model = ThermalModelParams(
            T_inf=25.0,
            T_nozzle=215.0,
            T_g=60.0,
            beta=0.0,
            a_v=0.0,
            tau0=10.0,
            a_fan=0.0,
            a_alpha=0.0,
        )
        cfg = MPCConfig(segment_length_mm=20.0)
        v = 50.0
        dt = cfg.segment_length_mm / v
        T0 = 100.0
        expected = 25.0 + (T0 - 25.0) * np.exp(-dt / 10.0)
        T_next, _, returned_dt = analytical_thermal_step(
            T=T0,
            D=0.0,
            u=[v, 215.0],
            alpha_deg=0.0,
            model=model,
            cfg=cfg,
        )
        self.assertAlmostEqual(returned_dt, dt, places=6)
        self.assertAlmostEqual(T_next, float(expected), places=4)

    def test_nozzle_setpoint_raises_deposit_temperature(self) -> None:
        model = ThermalModelParams(beta=1.0, a_v=0.0, tau0=1e6, a_fan=0.0, a_alpha=0.0)
        cfg = MPCConfig(segment_length_mm=20.0)
        T_cool, _, _ = analytical_thermal_step(
            T=100.0, D=0.0, u=[50.0, 190.0], alpha_deg=0.0, model=model, cfg=cfg
        )
        T_hot, _, _ = analytical_thermal_step(
            T=100.0, D=0.0, u=[50.0, 230.0], alpha_deg=0.0, model=model, cfg=cfg
        )
        self.assertGreater(T_hot, T_cool)

    def test_pack_unpack_roundtrip(self) -> None:
        base = ThermalModelParams()
        theta = pack_params(base)
        restored = unpack_params(theta, base)
        self.assertAlmostEqual(restored.beta, base.beta)
        self.assertAlmostEqual(restored.a_v, base.a_v)
        self.assertAlmostEqual(restored.tau0, base.tau0)

    def test_fit_recovers_synthetic_params(self) -> None:
        true_model = ThermalModelParams(
            beta=0.40,
            a_v=0.12,
            tau0=7.0,
            a_fan=1.8,
            a_alpha=0.35,
        )
        cfg = MPCConfig()
        rows = []
        T = 95.0
        for alpha in (45.0, 50.0, 55.0, 60.0):
            for v, T_nozzle in ((30.0, 200.0), (45.0, 215.0), (55.0, 225.0)):
                T_next, _, _ = analytical_layer_step(
                    T=T,
                    u=[v, T_nozzle],
                    alpha_deg=alpha,
                    model=true_model,
                    cfg=cfg,
                    n_substeps=cfg.n_substeps,
                )
                rows.append(
                    {
                        "T_k": T,
                        "D_k": 0.0,
                        "v": v,
                        "T_nozzle": T_nozzle,
                        "alpha_deg": alpha,
                        "T_next_measured": T_next,
                    }
                )
                T = T_next

        base = ThermalModelParams()
        fitted = fit_thermal_model(rows, base, cfg, n_substeps=cfg.n_substeps)
        self.assertAlmostEqual(fitted.beta, true_model.beta, delta=0.08)
        self.assertAlmostEqual(fitted.a_v, true_model.a_v, delta=0.08)
        self.assertAlmostEqual(fitted.tau0, true_model.tau0, delta=1.5)

    def test_layer_step_holds_heat_better_than_single_long_step(self) -> None:
        model = ThermalModelParams()
        cfg = MPCConfig(segment_length_mm=300.0)
        u = [40.0, 215.0]
        T0 = 120.0
        T_layer, D_layer, _ = analytical_layer_step(T0, u, 45.0, model, cfg, n_substeps=12)
        T_long, D_long, _ = analytical_thermal_step(T0, 0.0, u, 45.0, model, cfg)
        self.assertGreater(T_layer, T_long)
        self.assertGreater(D_layer, D_long)


if __name__ == "__main__":
    unittest.main()
