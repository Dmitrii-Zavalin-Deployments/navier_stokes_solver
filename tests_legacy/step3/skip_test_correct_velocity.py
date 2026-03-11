# tests/step3/test_correct_velocity.py

import numpy as np

from src.step3.correct_velocity import correct_velocity
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def test_correction_math():
    """Verify projection math with correct staggered shapes."""
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.config["dt"] = 1.0
    state.constants["rho"] = 1.0
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    
    # Mock gradients with proper staggered shapes: (N+1, N, N) for X, etc.
    state.operators["grad_x"] = lambda p: np.ones((nx+1, ny, nz))
    state.operators["grad_y"] = lambda p: np.ones((nx, ny+1, nz))
    state.operators["grad_z"] = lambda p: np.ones((nx, ny, nz+1))
    
    U_star = np.full((nx+1, ny, nz), 5.0)
    V_star = np.full((nx, ny+1, nz), 5.0)
    W_star = np.full((nx, ny, nz+1), 5.0)
    
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, np.zeros((nx,ny,nz)))
    
    # Math: 5.0 - (1.0 / 1.0) * 1.0 = 4.0
    assert np.allclose(U_new, 4.0)
    assert np.allclose(V_new, 4.0)
    assert np.allclose(W_new, 4.0)

def test_correction_neighbor_masking():
    """Verify staggered masking at solid boundaries with correct shapes."""
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.config["dt"] = 1.0
    state.constants["rho"] = 1.0
    
    # Cell (1,1,1) is solid
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    state.is_fluid[1, 1, 1] = False
    
    # Mock zero gradients to isolate masking logic
    state.operators["grad_x"] = lambda p: np.zeros((nx+1, ny, nz))
    state.operators["grad_y"] = lambda p: np.zeros((nx, ny+1, nz))
    state.operators["grad_z"] = lambda p: np.zeros((nx, ny, nz+1))
    
    U_star = np.full((nx+1, ny, nz), 5.0)
    V_star = np.full((nx, ny+1, nz), 5.0)
    W_star = np.full((nx, ny, nz+1), 5.0)
    
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, np.zeros((nx,ny,nz)))
    
    # U-Faces adjacent to solid cell (1,1,1) should be 0.0
    # These are at index 1 (face between cell 0 and 1) and index 2 (between 1 and 2)
    assert U_new[1, 1, 1] == 0.0
    assert U_new[2, 1, 1] == 0.0
    
    # Boundary faces (0 and 3) are not touching the internal solid in this setup
    assert U_new[0, 1, 1] == 5.0
    assert U_new[3, 1, 1] == 5.0

def test_correction_zero_gradient():
    """Verify that a zero pressure gradient leaves velocity unchanged in fluid."""
    nx, ny, nz = 3, 3, 3
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    state.config["dt"] = 0.1
    state.constants["rho"] = 1.0
    state.is_fluid = np.ones((nx, ny, nz), dtype=bool)
    
    state.operators["grad_x"] = lambda p: np.zeros((nx+1, ny, nz))
    state.operators["grad_y"] = lambda p: np.zeros((nx, ny+1, nz))
    state.operators["grad_z"] = lambda p: np.zeros((nx, ny, nz+1))
    
    U_star = np.random.rand(nx+1, ny, nz)
    V_star = np.random.rand(nx, ny+1, nz)
    W_star = np.random.rand(nx, ny, nz+1)
    
    U_new, V_new, W_new = correct_velocity(state, U_star, V_star, W_star, np.zeros((nx,ny,nz)))
    
    assert np.allclose(U_new, U_star)
    assert np.allclose(V_new, V_star)
    assert np.allclose(W_new, W_star)