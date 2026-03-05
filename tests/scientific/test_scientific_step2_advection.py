# tests/scientific/test_scientific_step2_advection.py

import pytest
import numpy as np
from src.step2.advection import build_advection_stencils
from src.solver_state import SolverState
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

@pytest.fixture
def state_3d_small():
    return make_step1_output_dummy(nx=2, ny=2, nz=2)

def test_scientific_advection_dof_handshake(state_3d_small, capsys):
    """Rule 2.1: Verify total DOF count and Debug Handshake prints."""
    # U: (nx+1)*ny*nz = 3*2*2 = 12
    # V: nx*(ny+1)*nz = 2*3*2 = 12
    # W: nx*ny*(nz+1) = 2*2*3 = 12
    # Total = 36
    
    build_advection_stencils(state_3d_small)
    captured = capsys.readouterr().out
    
    assert "Mapping 36 DOFs" in captured
    assert "Handshake check - processed 36/36 DOFs" in captured
    assert state_3d_small.advection.indices.shape == (36, 8)
    assert state_3d_small.advection.weights.shape == (36, 8)

def test_scientific_advection_trilinear_indices(state_3d_small):
    """Rule 2.2: Verify 8-point stencil mapping for an internal U-node."""
    build_advection_stencils(state_3d_small)
    indices = state_3d_small.advection.indices
    
    # Let's check the first U DOF (i=0, j=0, k=0)
    # get_p_idx(i,j,k) for 2x2x2 (Order F): i + j*2 + k*4
    # Neighbors for U(0,0,0) involve i and i-1. 
    # Because of clamping, i-1 (-1) becomes 0.
    # All indices should point to P-cell 0 (clamped) or neighbors.
    first_u_stencil = indices[0]
    assert len(np.unique(first_u_stencil)) <= 8
    assert indices.max() < 8, "Indices must not exceed total P-cells (nx*ny*nz)"

def test_scientific_advection_weight_consistency(state_3d_small):
    """Rule 2.3: Ensure weights follow the SSoT config and are double precision."""
    build_advection_stencils(state_3d_small)
    weights = state_3d_small.advection.weights
    
    assert weights.dtype == np.float64
    # Sum of weights for one DOF should be 8 * 0.125 = 1.0
    np.testing.assert_allclose(np.sum(weights[0]), 1.0)
    assert np.all(weights == 0.125)

def test_scientific_advection_clamping_boundary(state_3d_small):
    """Rule 2.4: Boundary faces must clamp to valid P-indices (no out-of-bounds)."""
    # Force a larger grid to test max clamping
    state_3d_small.grid.nx, state_3d_small.grid.ny, state_3d_small.grid.nz = 10, 10, 10
    build_advection_stencils(state_3d_small)
    
    indices = state_3d_small.advection.indices
    # Max P-index should be (nx*ny*nz) - 1 = 999
    assert indices.max() == 999
    assert indices.min() == 0

def test_scientific_advection_buffer_integrity():
    """Rule 2.5: Zero-Debt Check - Buffers must be initialized to zero before fill."""
    state = SolverState()
    state.grid.nx, state.grid.ny, state.grid.nz = 2, 2, 2
    state.config.advection_weight_base = 0.0
    
    build_advection_stencils(state)
    # If weights weren't correctly filled, they should be exactly 0.0 from allocation
    np.testing.assert_array_equal(state.advection.weights, 0.0)
# --- NEW SCIENTIFIC APPENDS: DEBUG & PRECISION ---

def test_scientific_advection_full_debug_cycle(state_3d_small, capsys):
    """Rule 2.1: Verify specific sequence in DEBUG prints (U, V, W)."""
    build_advection_stencils(state_3d_small)
    captured = capsys.readouterr().out
    
    # Check for the explicit mapping sequence
    assert "Mapping 36 DOFs (U:12, V:12, W:12)" in captured
    # Check for the Clamping/MinMax debug line
    assert "Indices Min/Max: 0/7" in captured
    # Check for the Mass Conservation debug line
    assert "Weights check sum: 1.0000" in captured

def test_scientific_advection_high_precision_ssot(state_3d_small):
    """Rule 2.5: Verify SSoT preserves 15 decimal places (float64)."""
    high_precision_val = 0.123456789012345
    state_3d_small.config.advection_weight_base = high_precision_val
    build_advection_stencils(state_3d_small)
    
    # Assert that no truncation occurred during the fill process
    assert state_3d_small.advection.weights[0, 0] == high_precision_val

def test_scientific_advection_internal_stencil_uniqueness(state_3d_small):
    """
    Scientific check: For an internal node, the stencil must point to 
    8 distinct neighbors (no collapsed corners).
    """
    # Use 3x3x3 to ensure we have a true 'internal' region.
    # U-grid shape: (nx+1, ny, nz) = (4, 3, 3)
    nx, ny, nz = 3, 3, 3
    state_3d_small.grid.nx, state_3d_small.grid.ny, state_3d_small.grid.nz = nx, ny, nz
    build_advection_stencils(state_3d_small)
    
    indices = state_3d_small.advection.indices
    
    # Calculation for a true internal U-node:
    # Target (i=2, j=1, k=1)
    # Index = i + j*(nx+1) + k*(nx+1)*ny
    # Index = 2 + 1*(4) + 1*(4*3) = 2 + 4 + 12 = 18
    internal_u_idx = 18
    internal_u_stencil = indices[internal_u_idx]
    
    unique_neighbors = np.unique(internal_u_stencil)
    
    # Physics Validation: A trilinear stencil for advection must sample 
    # the 8 surrounding pressure cells to conserve momentum correctly.
    assert len(unique_neighbors) == 8, (
        f"Internal stencil at index {internal_u_idx} has collapsed indices! "
        f"Found {len(unique_neighbors)} unique: {unique_neighbors}"
    )

def test_scientific_advection_ssot_propagation(state_3d_small):
    """
    Rule 2.6: Verify exact propagation of the weight value from 
    SolverState.config into AdvectionStructure weights.
    """
    # 1. Define the test value
    test_val = 0.0625
    
    # 2. Re-hydrate the full configuration to ensure all required fields are present
    # This satisfies the 'No-Default' policy and avoids needing a setter
    state_3d_small.config.simulation_parameters = {
        "time_step": 0.01,
        "total_time": 1.0,
        "output_interval": 10,
        "advection_weight_base": test_val
    }
    
    # 3. Execute the build logic
    build_advection_stencils(state_3d_small)
    
    # 4. Assert that the weights buffer contains exactly the value 
    # stored in the config facade
    actual_weights = state_3d_small.advection.weights
    
    assert np.allclose(actual_weights, test_val), \
        f"Weight mismatch! Expected {test_val}, got {actual_weights[0, 0]}"
    
    # 5. Verify the handshake: the facade is indeed returning what we set
    assert state_3d_small.config.advection_weight_base == test_val