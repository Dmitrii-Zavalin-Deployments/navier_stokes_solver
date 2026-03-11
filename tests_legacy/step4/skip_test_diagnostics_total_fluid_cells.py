# tests/step4/test_diagnostics_total_fluid_cells.py

import numpy as np

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_diagnostics_total_fluid_cells():
    nx = ny = nz = 3
    state = make_step3_output_dummy(nx=nx, ny=ny, nz=nz)

    # Mark some solids
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    state.is_fluid[0, 0, 0] = False
    state.is_fluid[1, 1, 1] = False

    initialize_extended_fields(state)
    apply_boundary_conditions(state)
    assemble_diagnostics(state)

    expected_fluid_cells = nx * ny * nz - 2
    assert state.step4_diagnostics["total_fluid_cells"] == expected_fluid_cells
