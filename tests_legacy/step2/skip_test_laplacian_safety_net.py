# tests/step2/test_laplacian_safety_net.py

import numpy as np

from src.solver_state import SolverState
from src.step2.build_laplacian_operators import build_laplacian_operators


def test_laplacian_safety_net_coverage():
    """
    Specifically targets lines 36-40 to reach 100% coverage.
    Forces the 'if state.boundary_conditions is None' branch.
    """
    # 1. Create a minimal grid manually (don't use make_state)
    grid = {'nx': 2, 'ny': 2, 'nz': 2, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0}
    
    # 2. Inject None into boundary_conditions
    state = SolverState(grid=grid, boundary_conditions=None)
    
    # 3. Inject a minimal is_fluid mask
    state.is_fluid = np.ones((2, 2, 2), dtype=bool)
    
    # 4. Call the builder. 
    # This WILL hit lines 36-40.
    build_laplacian_operators(state)
    
    # 5. Verify the operator was still created (fallback to all-wall Neumann)
    assert "laplacian" in state.operators
    assert state.operators["laplacian"].shape == (8, 8)