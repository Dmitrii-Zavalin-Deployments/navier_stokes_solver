# tests/step3/test_staggered_integration.py

import pytest

# FIX: Use absolute import to ensure pytest discovers the mock generator
from tests.step3.test_step3_shape_integrity import inject_staggered_operators

from src.step3.build_ppe_rhs import build_ppe_rhs
from src.step3.predict_velocity import predict_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_full_step3_data_flow():
    """Ensures predict_velocity output is compatible with build_ppe_rhs."""
    state = make_step2_output_dummy(nx=3, ny=3, nz=3)
    state.constants["rho"] = 1.0
    state.config["dt"] = 0.01
    
    # Inject staggered operators (Laplacians, Divergence, etc.)
    state = inject_staggered_operators(state)
    
    # Uses internal state.fields by default
    U_star, V_star, W_star = predict_velocity(state)
    
    try:
        # Verify that RHS can be built from the staggered output
        rhs = build_ppe_rhs(state, U_star, V_star, W_star)
    except ValueError as e:
        pytest.fail(f"Integration failed due to shape mismatch: {e}")
        
    assert rhs.shape == (3, 3, 3), "RHS must be cell-centered (nx, ny, nz)"