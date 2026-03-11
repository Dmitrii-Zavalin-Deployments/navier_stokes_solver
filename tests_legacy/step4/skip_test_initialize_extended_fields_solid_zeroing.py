# tests/step4/test_initialize_extended_fields_solid_zeroing.py

import numpy as np

from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_initialize_extended_fields_solid_zeroing():
    nx = ny = nz = 2
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # Mark one cell as solid
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    state.is_fluid[0, 0, 1] = False

    # Fill fields with nonzero values
    state.fields["P"][:] = 5.0
    state.fields["U"][:] = 7.0
    state.fields["V"][:] = 8.0
    state.fields["W"][:] = 9.0

    initialize_extended_fields(state)

    # Pressure interior at solid cell must be zero
    assert state.P_ext[1, 1, 2] == 0.0

    # U faces touching solid must be zero
    assert (state.U_ext[1:3, 1, 2] == 0).all()
