# tests/step4/test_apply_bc_pressure_dirichlet.py

import numpy as np

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_apply_bc_pressure_dirichlet():
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)
    initialize_extended_fields(state)

    state.config["boundary_conditions"] = [
        {"type": "pressure_dirichlet", "variable": "p", "direction": "x", "side": "min", "value": 7.0}
    ]

    apply_boundary_conditions(state)

    # Ghost slice at x_min must equal 7
    assert np.allclose(state.P_ext[0, :, :], 7.0)

    # Interior unchanged
    assert (state.P_ext[1, :, :] == 0).all()
