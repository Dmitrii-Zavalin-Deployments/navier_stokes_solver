# tests/step3/test_apply_boundary_conditions_post.py

import numpy as np

from src.step3.apply_boundary_conditions_post import apply_boundary_conditions_post
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_domain_boundary_enforcement_post():
    """Verify that domain enums (like free-slip) are enforced after pressure correction."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)
    state.config["boundary_conditions"] = [{"location": "y_max", "type": "free-slip"}]

    U, V, W, P = np.ones_like(state.fields["U"]), np.ones_like(state.fields["V"]), \
                 np.ones_like(state.fields["W"]), np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)
    assert np.all(fields_out["V"][:, -1, :] == 0.0)

def test_internal_solid_mask_post():
    """Verify internal solids remain blocked after pressure correction."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(False) 
    U, V, W, P = np.ones_like(state.fields["U"]), np.ones_like(state.fields["V"]), \
                 np.ones_like(state.fields["W"]), np.zeros_like(state.fields["P"])

    fields_out = apply_boundary_conditions_post(state, U, V, W, P)
    assert np.all(fields_out["U"] == 0.0)
    assert np.all(fields_out["V"] == 0.0)

def test_bc_handler_called():
    """Custom BC handler must be invoked in the POST step."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.is_fluid.fill(True)
    state.config["boundary_conditions"] = []
    
    # Ensure the handler is a callable detected by getattr
    def mock_handler(st, flds):
        flds["U"] = flds["U"] * 5.0
        return flds
    
    state.boundary_conditions = mock_handler

    U, V, W, P = np.ones_like(state.fields["U"]), np.ones_like(state.fields["V"]), \
                 np.ones_like(state.fields["W"]), np.ones_like(state.fields["P"])
    
    fields_out = apply_boundary_conditions_post(state, U, V, W, P)
    # Use allclose for floating point safety
    assert np.allclose(fields_out["U"], 5.0)

def test_minimal_grid_no_crash():
    """Minimal grid must not crash even with slicing logic."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    U, V, W, P = [np.zeros_like(state.fields[k]) for k in ["U", "V", "W", "P"]]
    fields_out = apply_boundary_conditions_post(state, U, V, W, P)
    assert fields_out["P"].shape == (1, 1, 1)