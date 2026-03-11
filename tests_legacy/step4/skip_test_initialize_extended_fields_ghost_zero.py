# tests/step4/test_initialize_extended_fields_ghost_zero.py

from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_initialize_extended_fields_ghost_zero():
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)

    initialize_extended_fields(state)

    # All ghost layers must be exactly zero
    P = state.P_ext
    assert (P[0, :, :] == 0).all()
    assert (P[-1, :, :] == 0).all()
    assert (P[:, 0, :] == 0).all()
    assert (P[:, -1, :] == 0).all()
    assert (P[:, :, 0] == 0).all()
    assert (P[:, :, -1] == 0).all()
