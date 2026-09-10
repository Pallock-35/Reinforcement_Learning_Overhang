import numpy as np

from mpc_controller_v3 import (
    ControllerState,
    MPCConfig,
    solve_mpc,
)
from thermal_model import ThermalModelParams


def main() -> None:
    cfg = MPCConfig(
        T_target_c=44.5,
        T_deadband_c=0.75,
    )

    # The empirical predictor no longer uses these thermal-model
    # coefficients, but solve_mpc still accepts this object.
    model = ThermalModelParams()

    previous = np.array(
        [100.0, 210.0],
        dtype=float,
    )

    test_temperatures = [
        38.0,
        40.0,
        42.0,
        43.0,
        44.0,
        44.5,
        45.0,
        46.0,
        48.0,
        50.0,
    ]

    print()
    print("75-degree empirical MPC offline sweep")
    print("-" * 82)
    print(
        f"{'T measured':>10} "
        f"{'Speed':>10} "
        f"{'Nozzle':>10} "
        f"{'T predicted':>12} "
        f"{'Success':>10}"
    )
    print("-" * 82)

    for measured_temperature in test_temperatures:
        state = ControllerState(
            T=measured_temperature,
            u_prev=previous.copy(),
        )

        action, info = solve_mpc(
            state=state,
            alpha_deg=75.0,
            model=model,
            cfg=cfg,
        )

        speed = float(action[0])
        nozzle = float(action[1])
        predicted_temperature = float(
            info["predicted_next_T_c"]
        )

        # Absolute safety checks
        assert cfg.v_min <= speed <= cfg.v_max
        assert (
            cfg.T_nozzle_min
            <= nozzle
            <= cfg.T_nozzle_max
        )

        # Layer-to-layer movement checks
        assert abs(speed - previous[0]) <= (
            cfg.dv_max + 1e-6
        )
        assert abs(nozzle - previous[1]) <= (
            cfg.dT_nozzle_max + 1e-6
        )

        # Direction checks for the fitted empirical model
        if (
            measured_temperature
            < cfg.T_target_c - cfg.T_deadband_c
        ):
            assert speed >= previous[0] - 1e-6
            assert nozzle >= previous[1] - 1e-6

        elif (
            measured_temperature
            > cfg.T_target_c + cfg.T_deadband_c
        ):
            assert speed <= previous[0] + 1e-6
            assert nozzle <= previous[1] + 1e-6

        else:
            assert np.allclose(
                action,
                previous,
                atol=1e-6,
            )

        print(
            f"{measured_temperature:>10.2f} "
            f"{speed:>10.2f} "
            f"{nozzle:>10.2f} "
            f"{predicted_temperature:>12.2f} "
            f"{str(info['success']):>10}"
        )

    print("-" * 82)
    print("PASS: all bounds, rate limits, and direction checks passed.")


if __name__ == "__main__":
    main()