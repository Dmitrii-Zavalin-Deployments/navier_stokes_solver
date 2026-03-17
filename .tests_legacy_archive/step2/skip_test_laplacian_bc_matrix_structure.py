# tests/step2/test_laplacian_bc_matrix_structure.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators


@pytest.fixture
def ppe_state():
    """
    Sets up a 3x3x3 grid where x_max is Pressure and x_min is a Wall.
    This small grid allows for precise index tracking.
    """
    grid = {
        'nx': 3, 'ny': 3, 'nz': 3,
        'dx': 1.0, 'dy': 1.0, 'dz': 1.0
    }
    # Initialize a fluid mask (all cells are fluid for this test)
    is_fluid = np.ones((3, 3, 3), dtype=bool)
    
    # Define BCs: 
    # x_max is Dirichlet (Pressure) -> Matrix row should be identity-like.
    # x_min is Neumann (No-slip) -> Matrix row should be modified stencil.
    boundary_conditions = {
        "x_max": {"type": "pressure", "p": 100.0},
        "x_min": {"type": "no-slip", "p": 0.0}
    }
    
    state = SolverState(grid=grid, boundary_conditions=boundary_conditions)
    state.is_fluid = is_fluid
    return state

def test_pressure_dirichlet_row_identity(ppe_state):
    """
    Theory Check: For Pressure-type BCs, the row in matrix A is replaced 
    with an identity-like row where the diagonal A[i,i] = 1.0.
    """
    # Execute matrix construction - modifies ppe_state.operators
    build_laplacian_operators(ppe_state)
    A = ppe_state.operators["laplacian"]
    
    nx, ny = ppe_state.grid['nx'], ppe_state.grid['ny']
    
    # Target: A cell on the x_max face (i=2, j=1, k=1)
    # Flat Index calculation: i + j*nx + k*nx*ny = 2 + 1*3 + 1*9 = 14
    target_idx = 2 + (1 * nx) + (1 * nx * ny)
    
    # Extract the specific row
    row_data = A.getrow(target_idx).toarray().flatten()
    
    # Assertion 1: Diagonal value must be 1.0
    assert row_data[target_idx] == 1.0, f"Pressure BC at index {target_idx} should have diagonal 1.0"
    
    # Assertion 2: Must be an identity row (exactly 1 non-zero entry)
    non_zero_count = np.count_nonzero(row_data)
    assert non_zero_count == 1, f"Pressure BC row {target_idx} should be identity-like, found {non_zero_count} non-zero entries"

def test_wall_neumann_stencil_modification(ppe_state):
    """
    Theory Check: For Wall-type BCs, the Neumann condition (dp/dn=0) is enforced.
    At x_min (i=0, j=1, k=1), the neighbor at i-1 is missing.
    With dx=dy=dz=1, the center value should be -5.0 instead of -6.0.
    """
    build_laplacian_operators(ppe_state)
    A = ppe_state.operators["laplacian"]
    
    nx, ny = ppe_state.grid['nx'], ppe_state.grid['ny']
    
    # Target: A cell on the x_min face (i=0, j=1, k=1)
    # Flat Index calculation: 0 + 1*3 + 1*9 = 12
    target_idx = 0 + (1 * nx) + (1 * nx * ny)
    
    row_data = A.getrow(target_idx).toarray().flatten()
    
    # In a standard 3D 7-point stencil with unit spacing:
    # Interior center = -6.0.
    # Boundary with 1 missing neighbor = -5.0.
    actual_center = row_data[target_idx]
    assert actual_center == -5.0, f"Neumann wall at {target_idx} should have center -5.0, got {actual_center}"

    # Verify that the neighbor at i+1 (index 13) still exists in the stencil
    assert row_data[target_idx + 1] == 1.0, "Right-side neighbor (i+1) should still have a coefficient of 1.0"