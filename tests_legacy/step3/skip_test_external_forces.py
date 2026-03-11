# tests/step3/test_external_forces.py

import numpy as np

from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _wire_zero_ops(state):
    """
    Wire advection and diffusion operators that return zero everywhere.
    """
    def adv_u(U, V, W):
        return np.zeros_like(U)

    def adv_v(U, V, W):
        return np.zeros_like(V)

    def adv_w(U, V, W):
        return np.zeros_like(W)

    def lap_u(U):
        return np.zeros_like(U)

    def lap_v(V):
        return np.zeros_like(V)

    def lap_w(W):
        return np.zeros_like(W)

    # Step 2 operators live in state.operators
    state.operators["adv_u"] = adv_u
    state.operators["adv_v"] = adv_v
    state.operators["adv_w"] = adv_w

    state.operators["lap_u"] = lap_u
    state.operators["lap_v"] = lap_v
    state.operators["lap_w"] = lap_w


def test_external_forces_modify_velocity():
    """
    External forces must modify predicted velocity.
    """
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)

    # Wire zero advection/diffusion so only forces act
    _wire_zero_ops(state)

    # Apply a simple force in x-direction
    state.config["external_forces"] = {"fx": 1.0, "fy": 0.0, "fz": 0.0}

    fields = {
        "U": state.fields["U"].copy(),
        "V": state.fields["V"].copy(),
        "W": state.fields["W"].copy(),
        "P": state.fields["P"].copy(),
    }

    U0 = fields["U"].copy()

    U_star, V_star, W_star = predict_velocity(state, fields)

    # U* must increase due to fx
    assert np.any(U_star > U0)
