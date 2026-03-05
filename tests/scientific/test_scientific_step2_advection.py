import pytest
import numpy as np
from src.step2.advection import build_advection_stencils
from src.solver_state import SolverState

def test_scientific_advection_dof_handshake(state_3d_small, capsys):
    """Rule 2.1: Verify total DOF count and Debug Handshake prints."""
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
    
    first_u_stencil = indices[0]
    assert len(np.unique(first_u_stencil)) <= 8
    assert indices.max() < 8, "Indices must not exceed total P-cells"

def test_scientific_advection_weight_consistency(state_3d_small):
    """Rule 2.3: Ensure weights follow the SSoT config and are double precision."""
    build_advection_stencils(state_3d_small)
    weights = state_3d_small.advection.weights
    
    assert weights.dtype == np.float64
    np.testing.assert_allclose(np.sum(weights[0]), 1.0)
    assert np.all(weights == 0.125)

def test_scientific_advection_clamping_boundary(state_3d_small):
    """Rule 2.4: Boundary faces must clamp to valid P-indices."""
    state_3d_small.grid.nx, state_3d_small.grid.ny, state_3d_small.grid.nz = 10, 10, 10
    build_advection_stencils(state_3d_small)
    
    indices = state_3d_small.advection.indices
    assert indices.max() == 999
    assert indices.min() == 0

def test_scientific_advection_buffer_integrity():
    state = SolverState()
    state.grid.nx, state.grid.ny, state.grid.nz = 2, 2, 2
    # Initialize the internal dict to bypass the read-only property
    state.config._simulation_parameters = {"advection_weight_base": 0.0}
    
    build_advection_stencils(state)
    np.testing.assert_array_equal(state.advection.weights, 0.0)

def test_scientific_advection_full_debug_cycle(state_3d_small, capsys):
    """Rule 2.1: Verify specific sequence in DEBUG prints."""
    build_advection_stencils(state_3d_small)
    captured = capsys.readouterr().out
    
    assert "Mapping 36 DOFs (U:12, V:12, W:12)" in captured
    assert "Indices Min/Max: 0/7" in captured
    assert "Weights check sum: 1.0000" in captured

def test_scientific_advection_high_precision_ssot(state_3d_small):
    high_precision_val = 0.123456789012345
    # Bypass the property setter by accessing the backing dictionary directly
    state_3d_small.config._simulation_parameters["advection_weight_base"] = high_precision_val
    
    build_advection_stencils(state_3d_small)
    assert state_3d_small.advection.weights[0, 0] == high_precision_val

def test_scientific_advection_internal_stencil_uniqueness(state_3d_small):
    """Scientific check: Internal stencil must point to 8 distinct neighbors."""
    nx, ny, nz = 3, 3, 3
    state_3d_small.grid.nx, state_3d_small.grid.ny, state_3d_small.grid.nz = nx, ny, nz
    build_advection_stencils(state_3d_small)
    
    # Internal U-node at target index
    internal_u_idx = 18
    unique_neighbors = np.unique(state_3d_small.advection.indices[internal_u_idx])
    
    assert len(unique_neighbors) == 8, f"Stencil collapsed! Unique neighbors: {unique_neighbors}"

def test_scientific_advection_ssot_propagation(state_3d_small):
    """Rule 2.6: Verify exact weight propagation from Config to AdvectionStructure."""
    test_val = 0.0625
    state_3d_small.config.simulation_parameters["advection_weight_base"] = test_val
    
    build_advection_stencils(state_3d_small)
    
    assert np.allclose(state_3d_small.advection.weights, test_val)
    assert state_3d_small.config.advection_weight_base == test_val