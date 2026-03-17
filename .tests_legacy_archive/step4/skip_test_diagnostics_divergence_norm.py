# tests/step4/test_diagnostics_divergence_norm.py

import numpy as np

from src.step4.apply_boundary_conditions import apply_boundary_conditions
from src.step4.assemble_diagnostics import assemble_diagnostics
from src.step4.initialize_extended_fields import initialize_extended_fields
from tests.helpers.solver_step3_output_dummy import make_step3_output_dummy


def test_diagnostics_divergence_norm():
    state = make_step3_output_dummy(nx=3, ny=3, nz=3)

    initialize_extended_fields(state)

    # Create a synthetic divergence pattern
    # Divergence = Ux + Vy + Wz (finite difference)
    state.U_ext[2, 1, 1] = 1.0
    state.V_ext[1, 2, 1] = 2.0
    state.W_ext[1, 1, 2] = 3.0

    apply_boundary_conditions(state)
    assemble_diagnostics(state)

    # Expected divergence norm = sqrt(1^2 + 2^2 + 3^2)
    expected = np.sqrt(1 + 4 + 9)
    assert np.isclose(state.step4_diagnostics["post_bc_divergence_norm"], expected)
