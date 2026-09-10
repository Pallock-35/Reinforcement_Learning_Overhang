"""Thermal dose tracker (outline §3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThermalDoseTracker:
    """
    Accumulate thermal dose D_k = integral max(T - T_g, 0) dt.

    Also tracks dT/dt for RL state features.
    """

    T_g: float = 60.0
    D: float = 0.0
    T_prev: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Clear accumulated dose and temperature history."""
        self.D = 0.0
        self.T_prev = None

    def update(self, T_k: float, dt_k: float) -> tuple[float, float]:
        """
        Update dose with measured temperature and layer interval.

        Returns (D_k, dT_dt).
        """
        if dt_k > 0.0 and self.T_prev is not None:
            dT_dt = (T_k - self.T_prev) / dt_k
        else:
            dT_dt = 0.0

        self.D += dt_k * max(T_k - self.T_g, 0.0)
        self.T_prev = T_k
        return self.D, dT_dt

    def snapshot(self) -> dict[str, float]:
        """Return current dose state."""
        return {"D_k": self.D, "T_g": self.T_g}
