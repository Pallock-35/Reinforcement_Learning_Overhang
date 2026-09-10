#!/usr/bin/env python3
"""Train SAC policy on the analytical thermal model (outline §7.6)."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    _SB3_AVAILABLE = True
except ImportError:
    SAC = None
    DummyVecEnv = None
    _SB3_AVAILABLE = False

from rl_thermal_env import ThermalControlEnv


def train_sac(
    *,
    total_timesteps: int = 50_000,
    alpha_deg: float = 60.0,
    output_path: Path,
    param_noise: float = 0.05,
    alpha_jitter: float = 5.0,
    seed: int = 0,
) -> Path:
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 and gymnasium are required. "
            "Install with: pip install gymnasium stable-baselines3"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _make_env() -> ThermalControlEnv:
        return ThermalControlEnv(
            alpha_deg=alpha_deg,
            param_noise=param_noise,
            alpha_jitter=alpha_jitter,
            seed=seed,
        )

    env = DummyVecEnv([_make_env])
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        gamma=0.99,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(str(output_path))
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SAC thermal controller offline")
    parser.add_argument(
        "--episodes",
        type=int,
        default=50_000,
        help="Training timesteps (default 50000)",
    )
    parser.add_argument("--alpha", type=float, default=60.0, help="Overhang angle (deg)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/sac_thermal"),
        help="Output path without extension (saves .zip)",
    )
    parser.add_argument("--param-noise", type=float, default=0.05)
    parser.add_argument(
        "--alpha-jitter",
        type=float,
        default=5.0,
        help="Randomize episode angle by ±this many degrees (outline §7.6)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not _SB3_AVAILABLE:
        print(
            "stable-baselines3 is not installed. "
            "Install with: pip install gymnasium stable-baselines3"
        )
        return 1

    path = train_sac(
        total_timesteps=args.episodes,
        alpha_deg=args.alpha,
        output_path=args.output,
        param_noise=args.param_noise,
        alpha_jitter=args.alpha_jitter,
        seed=args.seed,
    )
    print(f"Saved SAC policy to {path}.zip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
