# tests/step4/test_initialize_extended_fields_copy.py

import numpy as np

from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_initialize_extended_fields_copy():
    nx = ny = nz = 3
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # Fill interior fields with identifiable patterns
    state.fields["P"] = np.random.rand(nx, ny, nz)
    state.fields["U"] = np.random.rand(nx + 1, ny, nz)
    state.fields["V"] = np.random.rand(nx, ny + 1, nz)
    state.fields["W"] = np.random.rand(nx, ny, nz + 1)

    initialize_extended_fields(state)

    # Pressure interior
    assert np.allclose(state.P_ext[1:nx+1, 1:ny+1, 1:nz+1], state.fields["P"])

    # U interior
    assert np.allclose(state.U_ext[1:nx+2, 1:ny+1, 1:nz+1], state.fields["U"])

    # V interior
    assert np.allclose(state.V_ext[:, 1:ny+2, 1:nz+1], state.fields["V"])

    # W interior
    assert np.allclose(state.W_ext[:, :, 1:nz+2], state.fields["W"])
