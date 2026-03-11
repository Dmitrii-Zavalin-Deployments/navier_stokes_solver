# tests/step3/test_build_ppe_rhs.py

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.step3.build_ppe_rhs import build_ppe_rhs
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def _get_total_vel_dof(state):
    """Helper to calculate the total velocity vector length."""
    u_size = state.fields["U"].size
    v_size = state.fields["V"].size
    w_size = state.fields["W"].size
    return u_size + v_size + w_size

def test_uniform_divergence():
    """If every velocity contributes to a divergence of 1, RHS must be scaled."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    
    # CRITICAL FIX: Ensure all cells are treated as fluid so the mask doesn't zero them out
    if state.is_fluid is not None:
        state.is_fluid = np.ones_like(state.fields["P"], dtype=bool)
    
    p_size = state.fields["P"].size
    vel_size = _get_total_vel_dof(state)
    
    # We point to index 0
    target_idx = 0
    
    # Create a 'Sum' matrix: Every row looks at target_idx
    data = np.ones(p_size)
    indices = np.full(p_size, target_idx)
    indptr = np.arange(p_size + 1)
    state.operators["divergence"] = csr_matrix((data, indices, indptr), shape=(p_size, vel_size))

    # Construct U, V, W
    U = np.zeros_like(state.fields["U"])
    U.flat[target_idx] = 1.0
    
    rhs = build_ppe_rhs(state, U, state.fields["V"], state.fields["W"])

    rho = state.constants["rho"]
    dt = state.constants["dt"]
    
    # Now that is_fluid is True, RHS should be (rho/dt) * 1.0 = 10.0
    assert np.allclose(rhs, rho / dt)

def test_solid_zeroing_logic():
    """Verify that the mask actually works by marking one cell as solid."""
    state = make_step2_output_dummy(nx=2, ny=2, nz=2)
    state.is_fluid = np.ones_like(state.fields["P"], dtype=bool)
    
    # Mark the first cell as solid
    state.is_fluid[0, 0, 0] = False
    
    p_size = state.fields["P"].size
    vel_size = _get_total_vel_dof(state)
    state.operators["divergence"] = csr_matrix((np.ones(p_size), np.zeros(p_size), np.arange(p_size+1)), shape=(p_size, vel_size))

    U = np.zeros_like(state.fields["U"])
    U.flat[0] = 1.0
    
    rhs = build_ppe_rhs(state, U, state.fields["V"], state.fields["W"])
    
    # Cell [0,0,0] MUST be 0.0 because it's solid
    assert rhs[0, 0, 0] == 0.0
    # Other cells should be 10.0
    assert rhs[0, 0, 1] == pytest.approx(10.0)