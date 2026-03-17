# tests/scientific/test_scientific_step4_ghost_manager.py

import numpy as np
import pytest

from src.solver_state import SolverState


class AttributeDict(dict):
    """Allows dot notation for mock diagnostics."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_internal_fields():
    """Sets up a 4x4x4 internal grid with identifiable data."""
    state = SolverState()
    nx, ny, nz = 4, 4, 4
    state.grid._nx, state.grid._ny, state.grid._nz = nx, ny, nz
    
    # Initialize interior fields with unique values
    # We use nx+1 for velocities because they are staggered (faces > cells)
    state.fields.P = np.full((nx, ny, nz), 101.0, order='F')
    state.fields.U = np.full((nx + 1, ny, nz), 1.1, order='F')
    state.fields.V = np.full((nx, ny + 1, nz), 2.2, order='F')
    state.fields.W = np.full((nx, ny, nz + 1), 3.3, order='F')
    
    state.diagnostics = AttributeDict()
    return state

def test_ghost_allocation_shapes(state_internal_fields):
    """
    Scientific check: Verifies Rule 5 compliance.
    Pressure requires +2 (one ghost each side).
    Staggered U requires +3 (one ghost each side + 1 extra face).
    """
    from src.step4.ghost_manager import initialize_ghost_fields
    
    initialize_ghost_fields(state_internal_fields)
    f = state_internal_fields.fields
    
    # Assert N+2 for Pressure
    assert f.P_ext.shape == (6, 6, 6)
    # Assert N+3 in primary axis, N+2 in others for Velocities
    assert f.U_ext.shape == (7, 6, 6)
    assert f.V_ext.shape == (6, 7, 6)
    assert f.W_ext.shape == (6, 6, 7)

def test_ghost_mapping_formula(state_internal_fields):
    """
    Scientific check: Verifies the 'Centered Mapping' formula.
    Interior data must start at index 1 and end at -1.
    """
    from src.step4.ghost_manager import initialize_ghost_fields
    
    # Set a distinct corner value
    state_internal_fields.fields.P[0, 0, 0] = 999.0
    
    initialize_ghost_fields(state_internal_fields)
    P_ext = state_internal_fields.fields.P_ext
    
    # The actual ghost cell (index 0) should still be 0.0
    assert P_ext[0, 0, 0] == 0.0
    # The first interior cell (index 1) should be our 999.0
    assert P_ext[1, 1, 1] == 999.0
    # The last interior cell (index -2) should be 101.0
    assert P_ext[-2, -2, -2] == 101.0
    # The final ghost cell (index -1) should still be 0.0
    assert P_ext[-1, -1, -1] == 0.0

def test_ghost_mismatch_error_handling(state_internal_fields):
    """
    Scientific check: Verifies that a shape mismatch triggers a RuntimeError.
    This prevents silent convergence failure.
    """
    from src.step4.ghost_manager import initialize_ghost_fields
    
    # Corrupt the internal field shape
    state_internal_fields.fields.P = np.zeros((2, 2, 2)) 
    
    with pytest.raises(RuntimeError, match="Ghost Initialization Failed"):
        initialize_ghost_fields(state_internal_fields)

def test_ghost_manager_debug_signals(state_internal_fields, capsys):
    """
    Scientific check: Verifies that all debug strings and memory logs are printed.
    """
    from src.step4.ghost_manager import initialize_ghost_fields
    
    initialize_ghost_fields(state_internal_fields)
    captured = capsys.readouterr().out
    
    assert "DEBUG [Step 4 Ghost]: Initializing extended fields for 4x4x4" in captured
    assert "Allocated P_ext shape: (6, 6, 6)" in captured
    assert "Mapping check - Interior P[0,0,0]" in captured
    assert "Memory allocated for P_ext" in captured

def test_ghost_memory_order_consistency(state_internal_fields):
    """
    Scientific check: Ensures 'F' order (Fortran-contiguous) is preserved.
    Crucial for performance in Laplacian sweeps.
    """
    from src.step4.ghost_manager import initialize_ghost_fields
    
    initialize_ghost_fields(state_internal_fields)
    
    assert np.isfortran(state_internal_fields.fields.P_ext)
    assert np.isfortran(state_internal_fields.fields.U_ext)

def test_staggered_face_alignment(state_internal_fields):
    """
    Scientific check: Verifies that the (N+1)th face of the staggered 
    velocity grid is correctly mapped to the second-to-last index of 
    the (N+3) extended field.
    """
    from src.step4.ghost_manager import initialize_ghost_fields
    
    nx = state_internal_fields.grid.nx
    # Mark the very last face of the interior U-velocity
    # In a 4x4x4 grid, U has 5 faces on the X-axis (index 0 to 4)
    state_internal_fields.fields.U[nx, 0, 0] = 8.88 
    
    initialize_ghost_fields(state_internal_fields)
    U_ext = state_internal_fields.fields.U_ext
    
    # In U_ext (shape 7), the mapping 1:-1 covers indices 1, 2, 3, 4, 5.
    # The value 8.88 should be at index nx+1 (which is index 5)
    assert U_ext[nx + 1, 1, 1] == 8.88
    # The final ghost index (nx + 2, which is index 6) should be 0.0
    assert U_ext[nx + 2, 1, 1] == 0.0