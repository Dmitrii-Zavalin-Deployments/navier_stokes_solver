# tests/step2/test_gradient_operators.py

import numpy as np
from scipy.sparse import issparse

from src.step2.build_gradient_operators import build_gradient_operators
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

def apply_gradient(state):
    """Multiplies the sparse G matrix by the pressure field."""
    G = state.operators["gradient"]
    p_vec = state.fields["P"].ravel(order='F')
    grad_vec = G @ p_vec
    
    nx, ny, nz = state.grid['nx'], state.grid['ny'], state.grid['nz']
    num_u = (nx + 1) * ny * nz
    num_v = nx * (ny + 1) * nz
    
    # Split the unified vector back into U, V, W components
    gx = grad_vec[:num_u].reshape((nx + 1, ny, nz), order='F')
    gy = grad_vec[num_u:num_u + num_v].reshape((nx, ny + 1, nz), order='F')
    gz = grad_vec[num_u + num_v:].reshape((nx, ny, nz + 1), order='F')
    
    return gx, gy, gz

# ------------------------------------------------------------
# 1. Verification of Structure
# ------------------------------------------------------------
def test_gradient_is_sparse():
    state = make_state()
    build_gradient_operators(state)
    assert issparse(state.operators["gradient"])

# ------------------------------------------------------------
# 2. Linear pressure in X → gradient ≈ 1/dx
# ------------------------------------------------------------
def test_gradient_linear_pressure_x():
    nx, ny, nz = 4, 4, 4
    dx = 0.5
    state = make_state(nx, ny, nz, dx)

    # P[i] = i * dx -> dP/dx = 1.0
    for i in range(nx):
        state.fields["P"][i, :, :] = i * dx

    build_gradient_operators(state)
    gx, gy, gz = apply_gradient(state)

    # Check internal faces (gradient should be 1.0)
    # i=0 and i=nx faces are boundaries where gradient isn't defined here
    assert np.allclose(gx[1:nx, :, :], 1.0, atol=1e-6)
    assert np.allclose(gy, 0.0)
    assert np.allclose(gz, 0.0)

# ------------------------------------------------------------
# 3. Output shapes (staggered)
# ------------------------------------------------------------
def test_gradient_output_shapes():
    nx, ny, nz = 3, 4, 5
    state = make_state(nx, ny, nz)
    build_gradient_operators(state)
    gx, gy, gz = apply_gradient(state)

    assert gx.shape == (nx + 1, ny, nz)
    assert gy.shape == (nx, ny + 1, nz)
    assert gz.shape == (nx, ny, nz + 1)