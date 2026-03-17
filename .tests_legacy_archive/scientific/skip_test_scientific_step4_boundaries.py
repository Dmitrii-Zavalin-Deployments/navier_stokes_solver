# tests/scientific/test_scientific_step4_boundaries.py

import numpy as np
import pytest

from src.solver_state import SolverState


class AttributeDict(dict):
    """A dict that allows dot notation access for mocking."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

@pytest.fixture
def state_bc():
    """Fixture to mock a state with extended fields and BC lookup table."""
    state = SolverState()
    
    # 1. Grid Setup (3x3x3 internal -> 5x5x5 extended)
    state.grid._nx, state.grid._ny, state.grid._nz = 3, 3, 3
    shape = (5, 5, 5)
    
    # 2. Field Initialization (Extended)
    # Rule 5 Compliance: We manually populate the _ext slots
    state.fields._U_ext = np.zeros(shape)
    state.fields._V_ext = np.zeros(shape)
    state.fields._W_ext = np.zeros(shape)
    state.fields._P_ext = np.zeros(shape)
    
    # Fill interior with known values to test gradients
    state.fields._P_ext[1:-1, 1:-1, 1:-1] = 10.0
    state.fields._U_ext[1:-1, 1:-1, 1:-1] = 5.0
    
    # 3. BC Lookup Table (SSoT)
    # We test X-Min as 'pressure' and others as 'default'
    state.bc_lookup = {
        "x_min": {"type": "pressure", "p": 15.0},
        "x_max": {"type": "wall"},
        "y_min": {"type": "wall"},
        "y_max": {"type": "wall"},
        "z_min": {"type": "wall"},
        "z_max": {"type": "wall"}
    }
    
    # 4. Diagnostics container
    state.diagnostics = AttributeDict()
    
    return state

def test_boundary_pressure_dirichlet_formula(state_bc):
    """Verifies Dirichlet: P_ghost = 2*P_bc - P_interior."""
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    # Interior P is 10.0. Target BC P is 15.0.
    # Expected Ghost = 2 * 15.0 - 10.0 = 20.0
    fill_ghost_boundaries(state_bc)
    
    # Check X-Min ghost layer (Index 0)
    assert np.all(state_bc.fields.P_ext[0, 1:-1, 1:-1] == 20.0)

def test_boundary_velocity_zero_gradient(state_bc):
    """Verifies Neumann: U_ghost = U_interior."""
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    # Interior U is 5.0. 
    fill_ghost_boundaries(state_bc)
    
    # Check X-Min ghost (Index 0) and X-Max ghost (Index -1)
    assert np.all(state_bc.fields.U_ext[0, :, :] == 5.0)
    assert np.all(state_bc.fields.U_ext[-1, :, :] == 5.0)

def test_boundary_pressure_default_neumann(state_bc):
    """Verifies that non-pressure faces default to Zero-Gradient (P_ghost = P_int)."""
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    # x_max is type "wall" (not "pressure")
    fill_ghost_boundaries(state_bc)
    
    # P_ghost should equal P_interior (10.0)
    assert np.all(state_bc.fields.P_ext[-1, 1:-1, 1:-1] == 10.0)

def test_boundary_missing_config_error(state_bc):
    """Verifies that a missing face in bc_lookup triggers a RuntimeError."""
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    # Delete a required face
    del state_bc.bc_lookup["z_max"]
    
    with pytest.raises(RuntimeError, match="No configuration found for face z_max"):
        fill_ghost_boundaries(state_bc)

def test_boundary_debug_signals(state_bc, capsys):
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    fill_ghost_boundaries(state_bc)
    captured = capsys.readouterr().out
    
    # Using substrings for robustness
    assert "Synchronizing ghost cells" in captured
    assert "Applying pressure to x_min" in captured
    assert "Ghost Pressure Signal" in captured
    
    assert state_bc.diagnostics.bc_verification_passed is True

def test_boundary_staggered_axes(state_bc):
    """Verifies that Y and Z axes are correctly mirrored."""
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    state_bc.fields._V_ext[1:-1, 1:-1, 1:-1] = 7.0
    state_bc.fields._W_ext[1:-1, 1:-1, 1:-1] = 9.0
    
    fill_ghost_boundaries(state_bc)
    
    # V mirror on Y-axis
    assert np.all(state_bc.fields.V_ext[:, 0, :] == 7.0)
    # W mirror on Z-axis
    assert np.all(state_bc.fields.W_ext[:, :, 0] == 9.0)
def test_3d_pressure_mirroring_comprehensive(state_bc):
    """
    Scientific check: Verifies Dirichlet mirroring across all three axes.
    Ensures the 3D expansion of apply_face_bc is working correctly.
    """
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    # Setup 3D interior with a base value
    state_bc.fields._P_ext[1:-1, 1:-1, 1:-1] = 10.0
    
    # Configure pressure BCs on different axes
    state_bc.bc_lookup["y_min"] = {"type": "pressure", "p": 20.0}
    state_bc.bc_lookup["z_min"] = {"type": "pressure", "p": 25.0}
    
    fill_ghost_boundaries(state_bc)
    P = state_bc.fields.P_ext
    
    # X-Axis check (2*15 - 10 = 20)
    assert np.allclose(P[0, 1:-1, 1:-1], 20.0), "X-axis pressure mirroring failed"
    # Y-Axis check (2*20 - 10 = 30)
    assert np.allclose(P[1:-1, 0, 1:-1], 30.0), "Y-axis pressure mirroring failed"
    # Z-Axis check (2*25 - 10 = 40)
    assert np.allclose(P[1:-1, 1:-1, 0], 40.0), "Z-axis pressure mirroring failed"

def test_velocity_staggered_sync_all_axes(state_bc):
    """
    Scientific check: Verifies that U, V, and W ghost layers 
    match their respective interior cells.
    """
    from src.step4.boundary_filler import fill_ghost_boundaries
    
    # Distinct values to ensure we aren't just seeing zeros
    state_bc.fields._U_ext[1:-1, 1:-1, 1:-1] = 1.1
    state_bc.fields._V_ext[1:-1, 1:-1, 1:-1] = 2.2
    state_bc.fields._W_ext[1:-1, 1:-1, 1:-1] = 3.3
    
    fill_ghost_boundaries(state_bc)
    
    assert np.allclose(state_bc.fields.U_ext[0, :, :], 1.1)
    assert np.allclose(state_bc.fields.V_ext[:, 0, :], 2.2)
    assert np.allclose(state_bc.fields.W_ext[:, :, 0], 3.3)

def test_debug_grid_dimensions_logging(state_bc, capsys):
    """
    Verifies the scope fix for nx, ny, nz in the debug print.
    """
    from src.step4.boundary_filler import fill_ghost_boundaries
    fill_ghost_boundaries(state_bc)
    
    captured = capsys.readouterr().out
    # This string specifically confirms the nx, ny, nz defined in local scope
    assert "Synchronizing ghost cells for 3x3x3" in captured
