# tests/step3/test_domain_boundaries.py

import numpy as np

from src.step3.apply_domain_boundaries import apply_domain_boundaries
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_enum_inflow_x_min():
    """Verify 'inflow' at x_min correctly sets the U-velocity component."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    # Target value from JSON
    target_u = 12.5
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": target_u}}
    ]
    
    # Start with zeros
    fields = {k: np.zeros_like(v) for k, v in state.fields.items()}
    
    out = apply_domain_boundaries(state, fields)
    
    # U is staggered in X. Index 0 is the x_min face.
    assert np.all(out["U"][0, :, :] == target_u)
    # Ensure no 'leakage' to the rest of the field
    assert np.all(out["U"][1:, :, :] == 0.0)

def test_enum_no_slip_y_max():
    """Verify 'no-slip' at y_max zeroes the normal V-velocity component."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.config["boundary_conditions"] = [
        {"location": "y_max", "type": "no-slip"}
    ]
    
    # Start with ones
    fields = {k: np.ones_like(v) for k, v in state.fields.items()}
    
    out = apply_domain_boundaries(state, fields)
    
    # V is staggered in Y. Index -1 is the y_max face.
    assert np.all(out["V"][:, -1, :] == 0.0)
    # Ensure tangential components (U, W) are untouched by this specific domain rule
    assert np.all(out["U"] == 1.0)

def test_enum_free_slip_z_min():
    """Verify 'free-slip' (no-penetration) zeroes the normal W-velocity."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.config["boundary_conditions"] = [
        {"location": "z_min", "type": "free-slip"}
    ]
    
    fields = {k: np.ones_like(v) for k, v in state.fields.items()}
    out = apply_domain_boundaries(state, fields)
    
    # W is staggered in Z. Index 0 is the z_min face.
    assert np.all(out["W"][:, :, 0] == 0.0)

def test_enum_outflow_x_max():
    """Verify 'outflow' applies a zero-gradient (Neumann) copy."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.config["boundary_conditions"] = [
        {"location": "x_max", "type": "outflow"}
    ]
    
    fields = {k: np.zeros_like(v) for k, v in state.fields.items()}
    # Set a specific value near the boundary
    fields["U"][-2, :, :] = 7.7 
    
    out = apply_domain_boundaries(state, fields)
    
    # Outflow should copy the value from the inner neighbor (-2) to the face (-1)
    assert np.all(out["U"][-1, :, :] == 7.7)

def test_multiple_boundaries_simultaneously():
    """Verify that multiple enums in the config list don't conflict."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.config["boundary_conditions"] = [
        {"location": "x_min", "type": "inflow", "values": {"u": 1.0}},
        {"location": "x_max", "type": "no-slip"}
    ]
    
    fields = {k: np.zeros_like(v) for k, v in state.fields.items()}
    out = apply_domain_boundaries(state, fields)
    
    assert np.all(out["U"][0, :, :] == 1.0)  # Inflow
    assert np.all(out["U"][-1, :, :] == 0.0) # No-slip