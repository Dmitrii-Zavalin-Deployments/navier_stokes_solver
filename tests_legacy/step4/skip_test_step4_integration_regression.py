# tests/step4/test_step4_integration_regression.py

import numpy as np

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_step4_integration_regression():
    state = make_step3_output_dummy(nx=2, ny=2, nz=2)

    # Known BCs
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 1.0}},
        {"location": "y_max", "type": "pressure", "values": {"p": 4.0}},
    ]

    initialize_extended_fields(state)
    apply_boundary_conditions(state)
    assemble_diagnostics(state)

    # Regression expectations
    assert np.allclose(state.U_ext[0, :, :], 1.0)
    assert np.allclose(state.P_ext[:, -1, :], 4.0)

    # Diagnostics stable
    assert state.step4_diagnostics["post_bc_max_velocity"] >= 1.0
    assert state.step4_diagnostics["post_bc_divergence_norm"] >= 0.0
