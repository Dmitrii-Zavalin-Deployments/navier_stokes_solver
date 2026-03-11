# tests/step4/test_step4_integration_complex_mask.py

import numpy as np

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_step4_integration_complex_mask():
    nx = ny = nz = 3
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # Complex mask: fluid, solid, boundary-fluid
    state.is_fluid = np.array([
        [[1,  1, -1],
         [1,  0,  1],
         [1,  1,  1]],

        [[1,  1,  1],
         [0,  1,  1],
         [1, -1,  1]],

        [[1,  1,  1],
         [1,  1,  1],
         [1,  1,  1]],
    ], dtype=bool)

    # BCs
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 3.0}},
        {"location": "z_max", "type": "outflow"},
    ]

    initialize_extended_fields(state)
    apply_boundary_conditions(state)
    assemble_diagnostics(state)

    # Solids must be zeroed
    assert state.P_ext[1, 2, 1] == 0.0   # solid cell
    assert (state.U_ext[1:3, 2, 1] == 0).all()

    # Boundary-fluid preserved (no special treatment)
    assert state.P_ext[1, 0, 2] == 0.0 or state.P_ext[1, 0, 2] == state.fields["P"][1, 0, 2]

    # No NaNs
    assert not np.isnan(state.P_ext).any()
    assert not np.isnan(state.U_ext).any()
    assert not np.isnan(state.V_ext).any()
    assert not np.isnan(state.W_ext).any()
