# tests/step4/test_initialize_extended_fields_shapes.py

from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_initialize_extended_fields_shapes():
    state = make_step3_output_dummy(nx=4, ny=3, nz=2)

    initialize_extended_fields(state)

    assert state.P_ext.shape == (4 + 2, 3 + 2, 2 + 2)
    assert state.U_ext.shape == (4 + 3, 3 + 2, 2 + 2)
    assert state.V_ext.shape == (4,     3 + 3, 2 + 2)
    assert state.W_ext.shape == (4,     3,     2 + 3)
