# tests/step4/test_diagnostics_minimal_grid.py

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_diagnostics_minimal_grid():
    state = make_step3_output_dummy(nx=1, ny=1, nz=1)

    initialize_extended_fields(state)
    apply_boundary_conditions(state)

    # Should not raise any errors
    assemble_diagnostics(state)

    assert isinstance(state.step4_diagnostics, dict)
