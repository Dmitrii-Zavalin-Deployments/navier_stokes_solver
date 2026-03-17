# tests/step3/test_apply_domain_boundaries.py

import numpy as np

from src.step3.apply_domain_boundaries import apply_domain_boundaries
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_inflow_boundary_mapping():
    """Verify that 'inflow' at x_min sets the correct U slice."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    state.config["boundary_conditions"] = [
        {
            "location": "x_min",
            "type": "inflow",
            "values": {"u": 5.0}
        }
    ]
    
    U = np.zeros_like(state.fields["U"])
    fields = {"U": U, "V": np.zeros_like(state.fields["V"]), "W": np.zeros_like(state.fields["W"]), "P": state.fields["P"]}
    
    out = apply_domain_boundaries(state, fields)
    
    # Check that U at x=0 is 5.0
    assert np.all(out["U"][0, :, :] == 5.0)
    # Check that U elsewhere is still 0.0
    assert np.all(out["U"][1:, :, :] == 0.0)

def test_free_slip_no_penetration():
    """Verify that 'free-slip' zeroes the normal velocity."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.config["boundary_conditions"] = [{"location": "y_max", "type": "free-slip"}]
    
    V = np.ones_like(state.fields["V"])
    fields = {"U": np.zeros_like(state.fields["U"]), "V": V, "W": np.zeros_like(state.fields["W"]), "P": state.fields["P"]}
    
    out = apply_domain_boundaries(state, fields)
    
    # Normal component to y_max is V. V[index=-1] should be 0.
    assert np.all(out["V"][:, -1, :] == 0.0)

def test_minimal_grid_boundaries():
    """Ensure no index errors on 1x1x1 grid."""
    state = make_step2_output_dummy(nx=1, ny=1, nz=1)
    state.config["boundary_conditions"] = [{"location": "z_min", "type": "no-slip"}]
    
    fields = {k: np.ones_like(v) for k, v in state.fields.items()}
    out = apply_domain_boundaries(state, fields)
    
    assert out["W"].shape == state.fields["W"].shape