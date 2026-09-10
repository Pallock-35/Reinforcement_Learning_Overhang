"""Baseline controllers: fixed [v, T_nozzle] (outline §8)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mpc_controller import MPCConfig

DEFAULT_PRINT_SPEED_MMS = 50.0
DEFAULT_NOZZLE_TEMP_C = 215.0


def fixed_control(
    speed_mms: float = DEFAULT_PRINT_SPEED_MMS,
    nozzle_temp_c: float = DEFAULT_NOZZLE_TEMP_C,
    cfg: MPCConfig | None = None,
) -> np.ndarray:
    """Return fixed baseline u = [v_mms, T_nozzle_C]."""
    _ = cfg
    return np.array([float(speed_mms), float(nozzle_temp_c)], dtype=float)


def clip_control(u: Sequence[float], cfg: MPCConfig | None = None) -> np.ndarray:
    """Clip control vector to MPC actuator bounds."""
    cfg = cfg or MPCConfig()
    v, T_nozzle = map(float, u)
    return np.array(
        [
            float(np.clip(v, cfg.v_min, cfg.v_max)),
            float(np.clip(T_nozzle, cfg.T_nozzle_min, cfg.T_nozzle_max)),
        ],
        dtype=float,
    )
