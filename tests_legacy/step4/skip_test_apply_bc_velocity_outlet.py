# tests/step4/test_apply_bc_velocity_outlet.py

import numpy as np

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_apply_bc_velocity_outlet():
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)
    initialize_extended_fields(state)

    # Set interior slice
    state.U_ext[1, :, :] = 4.0

    state.config["boundary_conditions"] = [
        {"type": "outlet", "variable": "u", "direction": "x", "side": "min"}
    ]

    apply_boundary_conditions(state)

    assert np.allclose(state.U_ext[0, :, :], state.U_ext[1, :, :])
