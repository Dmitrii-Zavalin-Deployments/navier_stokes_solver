# tests/step2/test_laplacian_coverage_edge_cases.py

import numpy as np
import pytest

from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators


def test_laplacian_handles_none_boundary_conditions():
    """
    Targets lines 36-40 in build_laplacian_operators.py.
    Ensures the code doesn't crash if state.boundary_conditions is explicitly None.
    """
    grid = {'nx': 2, 'ny': 2, 'nz': 2, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    state = SolverState(grid=grid, boundary_conditions=None) # Explicitly None
    
    # Also initialize is_fluid to avoid crashes in the loop
    state.is_fluid = np.ones((2, 2, 2), dtype=bool)
    
    # This should now trigger the fallback: bc_table = {}
    # If it fails, it will raise an AttributeError on .items()
    try:
        build_laplacian_operators(state)
    except AttributeError:
        pytest.fail("build_laplacian_operators crashed when boundary_conditions was None!")

    assert "laplacian" in state.operators
    # On a 2x2x2 grid with all walls (default), the center value 
    # for a corner cell should be -3.0 (3 neighbors missing)
    lap = state.operators["laplacian"].toarray()
    assert lap[0, 0] == -3.0

def test_laplacian_minimal_fluid_cell():
    """
    Ensures that the stencil logic is fully exercised even in 
    a minimal 1x1x1 configuration.
    """
    grid = {'nx': 1, 'ny': 1, 'nz': 1, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    state = SolverState(grid=grid, boundary_conditions={})
    state.is_fluid = np.ones((1, 1, 1), dtype=bool)
    
    build_laplacian_operators(state)
    
    lap = state.operators["laplacian"].toarray()
    # A single isolated fluid cell has 0 active neighbors
    # Diagonal should be 0.0
    assert lap[0, 0] == 0.0