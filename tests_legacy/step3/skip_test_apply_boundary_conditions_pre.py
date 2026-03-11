# tests/step3/test_apply_boundary_conditions_pre.py

import numpy as np

from src.step3.apply_boundary_conditions_pre import apply_boundary_conditions_pre
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_solid_zeroing():
    """Faces adjacent to internal solid cells must be zeroed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid[1, 1, 1] = False  # Make one internal solid cell

    U = np.ones_like(state.fields["U"])
    V = np.ones_like(state.fields["V"])
    W = np.ones_like(state.fields["W"])
    P = np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_pre(state, {"U": U, "V": V, "W": W, "P": P})

    # Internal faces touching the solid cell should be zero
    assert np.any(fields_out["U"] == 0.0)
    assert np.any(fields_out["V"] == 0.0)
    assert np.any(fields_out["W"] == 0.0)

def test_domain_inflow_pre():
    """Verify that domain inflow from JSON is applied during the PRE step."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 10.5}}
    ]

    fields_in = {k: np.zeros_like(v) for k, v in state.fields.items()}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    # Check normal component at x_min
    assert np.all(fields_out["U"][0, :, :] == 10.5)

def test_bc_handler_invocation():
    """Ensure the custom BC callable hook is executed."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    
    def mock_bc(st, flds):
        flds["V"] += 1.0
        return flds
    
    state.boundary_conditions = mock_bc
    fields_in = {k: np.zeros_like(v) for k, v in state.fields.items()}
    fields_out = apply_boundary_conditions_pre(state, fields_in)

    assert np.all(fields_out["V"] == 1.0)

def test_minimal_grid_no_crash():
    """Ensure stability on 1x1x1 grid."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    fields_in = {k: np.zeros_like(v) for k, v in state.fields.items()}
    fields_out = apply_boundary_conditions_pre(state, fields_in)
    assert fields_out["U"].shape == state.fields["U"].shape