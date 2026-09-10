"""Gymnasium environment for SAC thermal control (outline §7)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None
    spaces = None

from baseline_controllers import clip_control
from mpc_controller import MPCConfig
from thermal_model import ThermalModelParams, analytical_layer_step
from thermal_targets import (
    dose_target,
    dose_window,
    temperature_bond_limit,
    temperature_sag_limit,
    temperature_target,
)


def thermal_reward(
    T: float,
    D: float,
    u: Sequence[float],
    u_prev: Sequence[float],
    alpha_deg: float,
    cfg: MPCConfig,
    model: ThermalModelParams,
) -> float:
    """Negative thermal cost from outline §7.4 (per-layer dose)."""
    T_ref = temperature_target(alpha_deg)
    D_ref = dose_target(alpha_deg)
    D_low, D_high = dose_window(alpha_deg)
    T_bond = temperature_bond_limit(alpha_deg)
    T_sag = temperature_sag_limit(alpha_deg)

    T_error = T - T_ref
    D_error = D - D_ref
    low_violation = max(D_low - D, 0.0)
    high_violation = max(D - D_high, 0.0)
    cold_violation = max(T_bond - T, 0.0)
    hot_violation = max(T - T_sag, 0.0)

    du = np.asarray(u, dtype=float) - np.asarray(u_prev, dtype=float)

    cost = (
        cfg.w_T * T_error**2
        + cfg.w_D * D_error**2
        + cfg.w_window * (low_violation**2 + high_violation**2)
        + cfg.w_temp_constraint * (cold_violation**2 + hot_violation**2)
        + cfg.w_du * float(np.dot(du, du))
    )
    return -float(cost)


if gym is not None:

    class ThermalControlEnv(gym.Env):
        """Layerwise thermal control on the analytical model."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            *,
            alpha_deg: float = 60.0,
            model: ThermalModelParams | None = None,
            cfg: MPCConfig | None = None,
            max_steps: int = 80,
            param_noise: float = 0.0,
            alpha_jitter: float = 0.0,
            seed: int | None = None,
        ) -> None:
            super().__init__()
            self.alpha_deg = float(alpha_deg)
            self.model = model or ThermalModelParams()
            self.cfg = cfg or MPCConfig()
            self.max_steps = max_steps
            self.param_noise = param_noise
            self.alpha_jitter = alpha_jitter
            self._rng = np.random.default_rng(seed)

            self.action_space = spaces.Box(
                low=np.array([self.cfg.v_min, self.cfg.T_nozzle_min]),
                high=np.array([self.cfg.v_max, self.cfg.T_nozzle_max]),
                dtype=np.float32,
            )
            # s = [T, D, dT/dt, α, v_prev, T_nozzle_prev, layer_index]
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, -50.0, 0.0, 0.0, 150.0, 0.0]),
                high=np.array([250.0, 5000.0, 50.0, 90.0, 200.0, 280.0, 200.0]),
                dtype=np.float32,
            )

            self._T = 90.0
            self._D = 0.0
            self._dT = 0.0
            self._u_prev = np.array([100.0, 215.0], dtype=float)
            self._step_count = 0
            self._episode_alpha = self.alpha_deg

        def _effective_model(self) -> ThermalModelParams:
            if self.param_noise <= 0.0:
                return self.model
            jitter = self._rng.normal(0.0, self.param_noise, size=5)
            theta = np.array(
                [
                    self.model.beta,
                    self.model.a_v,
                    self.model.tau0,
                    self.model.a_fan,
                    self.model.a_alpha,
                ]
            )
            theta = np.clip(
                theta * (1.0 + jitter),
                [0.01, 0.0, 0.5, 0.0, -2.0],
                [0.99, 2.0, 60.0, 10.0, 5.0],
            )
            return ThermalModelParams(
                T_inf=self.model.T_inf,
                T_nozzle=self.model.T_nozzle,
                T_g=self.model.T_g,
                beta=float(theta[0]),
                a_v=float(theta[1]),
                tau0=float(theta[2]),
                a_fan=float(theta[3]),
                a_alpha=float(theta[4]),
            )

        def _obs(self) -> np.ndarray:
            return np.array(
                [
                    self._T,
                    self._D,
                    self._dT,
                    self._episode_alpha,
                    self._u_prev[0],
                    self._u_prev[1],
                    float(self._step_count),
                ],
                dtype=np.float32,
            )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> Tuple[np.ndarray, dict[str, Any]]:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._T = float(self._rng.uniform(80.0, 110.0))
            self._D = 0.0
            self._dT = 0.0
            self._u_prev = np.array([100.0, 215.0], dtype=float)
            self._step_count = 0
            if self.alpha_jitter > 0.0:
                self._episode_alpha = float(
                    np.clip(
                        self.alpha_deg + self._rng.uniform(-self.alpha_jitter, self.alpha_jitter),
                        30.0,
                        80.0,
                    )
                )
            else:
                self._episode_alpha = self.alpha_deg
            return self._obs(), {"alpha_deg": self._episode_alpha}

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            u = clip_control(action, self.cfg)
            model = self._effective_model()
            T_prev = self._T

            # Per-layer dose via substepped deposition model.
            T_next, D_layer, dt = analytical_layer_step(
                self._T,
                u,
                self._episode_alpha,
                model,
                self.cfg,
                n_substeps=self.cfg.n_substeps,
            )
            self._dT = (T_next - T_prev) / max(dt, 1e-9)
            self._T = T_next + float(self._rng.normal(0.0, 0.5))
            self._D = D_layer
            reward = thermal_reward(
                self._T,
                self._D,
                u,
                self._u_prev,
                self._episode_alpha,
                self.cfg,
                model,
            )
            self._u_prev = u.copy()
            self._step_count += 1
            terminated = self._step_count >= self.max_steps
            return self._obs(), reward, terminated, False, {"dt": dt, "D_layer": D_layer}

else:

    class ThermalControlEnv:  # type: ignore[no-redef]
        """Placeholder when gymnasium is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "gymnasium is required for ThermalControlEnv. "
                "Install with: pip install gymnasium"
            )


def make_thermal_env(**kwargs: Any) -> ThermalControlEnv:
    """Factory helper for training scripts."""
    return ThermalControlEnv(**kwargs)
