# tests/step4/test_diagnostics_bc_violation_count.py

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_diagnostics_bc_violation_count():
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)

    initialize_extended_fields(state)
    apply_boundary_conditions(state)

    # Manually corrupt ghost cells
    state.U_ext[0, :, :] = 999.0

    assemble_diagnostics(state)

    assert state.step4_diagnostics["bc_violation_count"] > 0
