# tests/step4/test_diagnostics_max_velocity.py

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_diagnostics_max_velocity():
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)

    initialize_extended_fields(state)

    # Set known velocities
    state.U_ext[1, 1, 1] = 4.0
    state.V_ext[1, 1, 1] = 7.0
    state.W_ext[1, 1, 1] = 2.0

    apply_boundary_conditions(state)
    assemble_diagnostics(state)

    assert state.step4_diagnostics["post_bc_max_velocity"] == 7.0
