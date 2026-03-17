# tests/step2/test_stencil_assembler.py

import numpy as np
import pytest

from src.step2.stencil_assembler import CellRegistry, assemble_stencil_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


def get_matrix_3d(stencil_list):
    """Helper to maintain SSoT for coordinate-based testing."""
    return {(b.center.i - 1, b.center.j - 1, b.center.k - 1): b for b in stencil_list}

def test_stencil_assembly_logic():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    stencil_list = assemble_stencil_matrix(state)
    
    # 1. Topology Audit: Core Assembly Count
    # Per Section 7, we only assemble blocks for the Core [0, nx-1]
    expected_count = nx * ny * nz
    assert len(stencil_list) == expected_count, f"Assembly count mismatch: {len(stencil_list)} != {expected_count}"
    
    matrix_3d = get_matrix_3d(stencil_list)
    
    # 2. Verify Core Cell (0,0,0) exists
    assert (0, 0, 0) in matrix_3d
    sample_block = matrix_3d[(0, 0, 0)]
    assert sample_block.center.is_ghost is False
    
    # 3. Verify Ghost Neighbor: (0,0,0) should have a Ghost neighbor at (-1, 0, 0)
    # The block's i_minus SHOULD be a ghost cell, but the block itself is Core.
    assert sample_block.i_minus.is_ghost is True
    assert sample_block.i_minus.i == -1

def test_stencil_physics_consistency():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Apply global physics parameters
    state.simulation_parameters.time_step = 0.0123
    state.fluid_properties.density = 999.0
    state.external_forces.force_vector = np.array([0.1, 0.2, 0.3])
    
    stencil_list = assemble_stencil_matrix(state)
    
    for block in stencil_list:
        assert block.dt == 0.0123
        assert block.rho == 999.0
        assert block.f_vals == (0.1, 0.2, 0.3)

def test_stencil_caching_efficiency():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_list = assemble_stencil_matrix(state)
    matrix_3d = get_matrix_3d(stencil_list)
    
    # Identity and pointer integrity
    block = matrix_3d[(0, 0, 0)]
    # Neighbor (1,0,0) is also a core cell
    right_neighbor = matrix_3d[(1, 0, 0)]
    
    assert block.i_plus.index == right_neighbor.center.index
    assert block.i_plus.fields_buffer is right_neighbor.center.fields_buffer

def test_registry_cache_hit():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    registry = CellRegistry(nx, ny, nz)
    
    # Same coordinates must return same object
    cell1 = registry.get_or_create(0, 0, 0, state)
    cell2 = registry.get_or_create(0, 0, 0, state)
    assert cell1 is cell2, "Cache Miss: Registry failed to cache instance"

    # Distinct coordinates must return distinct objects
    cell3 = registry.get_or_create(1, 1, 1, state)
    assert cell1 is not cell3

def test_registry_padding_failure():
    """Verify registry enforces the boundary [-1, nx]."""
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    registry = CellRegistry(nx, ny, nz)
    
    # Test lower padding
    with pytest.raises(IndexError):
        registry.get_or_create(-2, 0, 0, state)
    
    # Test upper padding
    with pytest.raises(IndexError):
        registry.get_or_create(nx + 1, 0, 0, state)