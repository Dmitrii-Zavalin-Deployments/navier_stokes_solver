# tests/step4/test_initialize_extended_fields_minimal_grid.py

from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_initialize_extended_fields_minimal_grid():
    state = make_step3_output_dummy(nx=1, ny=1, nz=1)

    initialize_extended_fields(state)

    assert state.P_ext.shape == (1 + 2, 1 + 2, 1 + 2)
    assert state.U_ext.shape == (1 + 3, 1 + 2, 1 + 2)
    assert state.V_ext.shape == (1,     1 + 3, 1 + 2)
    assert state.W_ext.shape == (1,     1,     1 + 3)
