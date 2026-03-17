# tests/step3/test_orchestrate_step3.py

import numpy as np
from scipy.sparse import csr_matrix, eye

from src.step3.orchestrate_step3 import orchestrate_step3
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _prepare_mock_state(nx):
    """
    Creates a dummy state with minimal operators to test orchestration flow.
    """
    state = make_step2_output_dummy(nx=nx, ny=nx, nz=nx)
    
    # Mock gradient operators as lambda functions for simplicity
    state.operators["grad_x"] = lambda P: np.zeros_like(state.fields["U"])
    state.operators["grad_y"] = lambda P: np.zeros_like(state.fields["V"])
    state.operators["grad_z"] = lambda P: np.zeros_like(state.fields["W"])
    
    size_p = nx**3
    total_vel = state.fields["U"].size + state.fields["V"].size + state.fields["W"].size
    
    # Mock Sparse Operators
    state.operators["divergence"] = csr_matrix((size_p, total_vel))
    state.ppe["A"] = eye(size_p, format="csr")
    state.ppe["ppe_is_singular"] = False
    
    # Mock Prediction Operators (Advection/Diffusion)
    u_size = state.fields["U"].size
    state.operators["lap_u"] = csr_matrix((u_size, u_size))
    state.operators["advection_u"] = csr_matrix((u_size, u_size))
    # Note: In a real run, V and W operators would be here too

    state.config["solver_settings"] = {
        "ppe_tolerance": 1e-6, 
        "ppe_atol": 1e-12,
        "ppe_max_iter": 10
    }
    
    # Initialize empty history
    state.history = {}
    
    return state

def test_orchestrate_step3_flow():
    """Verify that Step 3 completes and populates fields and history."""
    nx = 3
    state = _prepare_mock_state(nx)
    
    # Execute orchestration
    new_state = orchestrate_step3(state, current_time=0.1, step_index=1)
    
    # Verify fields are updated
    assert "U" in new_state.fields
    assert "P" in new_state.fields
    assert new_state.fields["P"].shape == (nx, nx, nx)
    
    # Verify history tracking
    assert len(new_state.history["times"]) == 1
    assert new_state.history["times"][0] == 0.1
    assert "ppe_status_history" in new_state.history