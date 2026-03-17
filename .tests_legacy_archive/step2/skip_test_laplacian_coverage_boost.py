# tests/step2/test_laplacian_coverage_boost.py

import numpy as np

from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators


def test_laplacian_fallback_coverage():
    """
    Targets lines 36-40 in build_laplacian_operators.py.
    Forces the 'if state.boundary_conditions is None' branch to execute.
    """
    # 1. Create a minimal grid
    grid = {'nx': 2, 'ny': 2, 'nz': 2, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    
    # 2. Explicitly set boundary_conditions to None
    state = SolverState(grid=grid, boundary_conditions=None)
    
    # 3. Initialize required fluid mask
    state.is_fluid = np.ones((2, 2, 2), dtype=bool)
    
    # 4. Execute builder
    # If the fallback 'bc_table = {}' works, this will not raise an AttributeError
    build_laplacian_operators(state)
    
    # 5. Verify results
    assert "laplacian" in state.operators
    lap = state.operators["laplacian"].toarray()
    
    # In a 2x2x2 grid with no BCs (all walls/Neumann), 
    # a corner cell has 3 neighbors within the grid.
    # Center value should be -3.0
    assert lap[0, 0] == -3.0
    
def test_laplacian_solid_mask_branch_coverage():
    """
    Ensures the 'if not is_fluid' branch is hit (lines 47-52).
    """
    grid = {'nx': 2, 'ny': 2, 'nz': 2, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    state = SolverState(grid=grid, boundary_conditions={})
    
    # Set one cell as solid
    state.is_fluid = np.ones((2, 2, 2), dtype=bool)
    state.is_fluid[0, 0, 0] = False 
    
    build_laplacian_operators(state)
    
    lap = state.operators["laplacian"].toarray()
    # Solid cell must have 1.0 on diagonal
    assert lap[0, 0] == 1.0