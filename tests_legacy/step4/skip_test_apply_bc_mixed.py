# tests/step4/test_apply_bc_mixed.py

import numpy as np

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_apply_bc_mixed():
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)
    initialize_extended_fields(state)

    state.U_ext[1, :, :] = 4.0
    state.V_ext[:, 1, :] = 2.0

    state.config["boundary_conditions"] = [
        {"type": "no-slip", "variable": "u", "direction": "x", "side": "min"},
        {"type": "inlet",   "variable": "v", "direction": "y", "side": "max", "value": 9.0},
    ]

    apply_boundary_conditions(state)

    assert np.allclose(state.U_ext[0, :, :], -4.0)
    assert np.allclose(state.V_ext[:, -1, :], 9.0)
