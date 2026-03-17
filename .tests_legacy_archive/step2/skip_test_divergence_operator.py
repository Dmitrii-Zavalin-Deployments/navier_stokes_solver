# tests/step2/test_divergence_operator.py

import numpy as np
from scipy.sparse import issparse

from src.step2.build_divergence_operator import build_divergence_operator
from src.step2.create_fluid_mask import create_fluid_mask
from tests.helpers.solver_step1_output_dummy import (
    make_step1_output_dummy as make_step1_dummy_state,
)


def make_state(nx=4, ny=4, nz=4, dx=1.0):
    state = make_step1_dummy_state(nx=nx, ny=ny, nz=nz)
    state.grid['dx'] = dx
    state.grid['dy'] = dx
    state.grid['dz'] = dx
    state.mask = np.ones((nx, ny, nz), dtype=int)
    create_fluid_mask(state)
    return state

def apply_divergence(state):
    """
    Helper to simulate the action of the sparse matrix on the fields.
    Uses Fortran order ('F') to match the indexing logic in the core operator.
    """
    D = state.operators["divergence"]
    
    # Flatten fields in the specific order [U, V, W] using Fortran order
    v_vec = np.concatenate([
        state.fields["U"].ravel(order='F'),
        state.fields["V"].ravel(order='F'),
        state.fields["W"].ravel(order='F')
    ])
    
    div_flat = D @ v_vec
    
    # Reshape the flat result back to 3D grid using Fortran order
    return div_flat.reshape(
        (state.grid['nx'], state.grid['ny'], state.grid['nz']), 
        order='F'
    )

# ------------------------------------------------------------
# 1. Verification of Structure
# ------------------------------------------------------------
def test_divergence_structure():
    state = make_state()
    build_divergence_operator(state)
    assert "divergence" in state.operators
    assert issparse(state.operators["divergence"])

# ------------------------------------------------------------
# 2. Uniform velocity → divergence = 0
# ------------------------------------------------------------
def test_divergence_uniform_velocity():
    state = make_state()
    state.fields["U"].fill(3.0)
    state.fields["V"].fill(-2.0)
    state.fields["W"].fill(1.5)
    
    build_divergence_operator(state)
    div = apply_divergence(state)
    assert np.allclose(div, 0.0)

# ------------------------------------------------------------
# 3. Linear U field → divergence ≈ dU/dx
# ------------------------------------------------------------
def test_divergence_linear_u_field():
    nx, ny, nz = 4, 4, 4
    dx = 0.5
    state = make_state(nx, ny, nz, dx)
    
    # U[i] = i * dx -> dU/dx = 1.0
    for i in range(nx + 1):
        state.fields["U"][i, :, :] = i * dx
        
    build_divergence_operator(state)
    div = apply_divergence(state)
    
    # Verify the divergence is 1.0 everywhere in the fluid
    assert np.allclose(div, 1.0, atol=1e-6)

# ------------------------------------------------------------
# 4. Minimal grid (1×1×1)
# ------------------------------------------------------------
def test_divergence_minimal_grid():
    state = make_state(nx=1, ny=1, nz=1)
    build_divergence_operator(state)
    div = apply_divergence(state)
    assert div.shape == (1, 1, 1)
    # With uniform or zero velocity, divergence should be zero
    assert np.allclose(div, 0.0)