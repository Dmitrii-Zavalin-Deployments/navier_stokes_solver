# tests/step2/test_stencil_assembler.py

import numpy as np
import pytest

from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def get_matrix_3d(stencil_list):
    """Helper to maintain SSoT for coordinate-based testing."""
    return {(b.center.i, b.center.j, b.center.k): b for b in stencil_list}

def test_stencil_assembly_logic():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    stencil_list = assemble_stencil_matrix(state)
    
    # 1. Memory Safety Assertion (The new standard)
    buffer_capacity = state.fields.data.shape[0]
    assert len(stencil_list) <= buffer_capacity, \
        f"POST OVERFLOW: Stencil count ({len(stencil_list)}) > Buffer ({buffer_capacity})"
    
    # 2. Topology Audit
    matrix_3d = get_matrix_3d(stencil_list)
    
    # Verify spatial logic - (0,0,0) is an interior cell in a 4x4x4 grid (with 2 ghost layers)
    sample_block = matrix_3d[(0, 0, 0)]
    assert sample_block.dx == 0.25
    assert sample_block.center.is_ghost is False
    
    # Verify ghost-neighbor connection at the edge of the assembly range (-2)
    # The block at (-2, 0, 0) should be a ghost cell
    ghost_block = matrix_3d[(-1, 0, 0)]
    assert ghost_block.center.is_ghost is True

def test_stencil_physics_consistency():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    state.simulation_parameters.time_step = 0.0123
    state.fluid_properties.density = 999.0
    state.external_forces.force_vector = np.array([0.1, 0.2, 0.3])
    
    stencil_list = assemble_stencil_matrix(state)
    get_matrix_3d(stencil_list)
    
    for block in stencil_list:
        assert block.dt == 0.0123
        assert block.rho == 999.0
        assert block.f_vals == (0.1, 0.2, 0.3)

def test_schema_mismatch_raises_error():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    wrong_fields = np.zeros((state.fields.data.shape[0], 99))
    state.fields.data = wrong_fields
    
    with pytest.raises(RuntimeError, match="Foundation Mismatch"):
        assemble_stencil_matrix(state)

def test_stencil_caching_efficiency():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_list = assemble_stencil_matrix(state)
    matrix_3d = get_matrix_3d(stencil_list)
    
    # 1. Access the blocks (Assuming list is ordered by index, (0,0,0) is index 0)
    block = matrix_3d[(0, 0, 0)]          # Should be (0,0,0)
    right_neighbor = stencil_list[1] # Should be (1,0,0)
    
    # 2. Assert coordinates for the 'block' (0,0,0)
    assert (block.center.i, block.center.j, block.center.k) == (0, 0, 0), \
        f"Expected block to be at (0,0,0), found ({block.center.i}, {block.center.j}, {block.center.k})"
    
    # 3. Assert coordinates for the 'right_neighbor' (1,0,0)
    assert (right_neighbor.center.i, right_neighbor.center.j, right_neighbor.center.k) == (1, 0, 0), \
        f"Expected neighbor to be at (1,0,0), found ({right_neighbor.center.i}, {right_neighbor.center.j}, {right_neighbor.center.k})"
    
    # 4. Assert Logical identity
    # Now we verify that the i_plus neighbor of the first block is indeed the 
    # center cell of the right_neighbor block.
    assert block.i_plus.index == right_neighbor.center.index, \
        f"Index mismatch: {block.i_plus.index} != {right_neighbor.center.index}"
        
    assert block.i_plus.fields_buffer is right_neighbor.center.fields_buffer, \
        "Logical identity failure: fields_buffer pointers do not match"

def test_stencil_matrix_topology():
    nx, ny, nz = 4, 4, 4
    # Note: Ensure make_step2_output_dummy does NOT trigger a strict == assertion
    # if you want to support safety padding.
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_matrix = assemble_stencil_matrix(state)
    matrix_3d = get_matrix_3d(stencil_matrix)

    # Verify wiring: We iterate over the full range assembled
    for (i, j, k), block in matrix_3d.items():
        # Check neighbors exist within the registry range
        if (i + 1) in range(-1, nx + 1):
            assert block.i_plus.index == matrix_3d[(i + 1, j, k)].center.index

def test_registry_cache_hit():
    """
    Verifies that the registry returns the exact same object instance
    for the same (i, j, k) coordinates, proving the cache is working.
    """
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # We assemble the matrix to populate the registry cache
    assemble_stencil_matrix(state)
    
    # Now we manually trigger another retrieval for the same (0,0,0)
    # We must access the registry inside the assembler.
    # To test this without modifying the assembler, we can create a temporary 
    # registry instance or add a small helper. 
    # Here is the direct test of the logic:
    
    from src.step2.stencil_assembler import CellRegistry
    registry = CellRegistry(nx, ny, nz)
    
    cell1 = registry.get_or_create(0, 0, 0, state)
    cell2 = registry.get_or_create(0, 0, 0, state)
    
    # This must be the SAME object instance in memory
    assert cell1 is cell2, "Cache Miss: Registry created a new instance instead of returning cached one"
    assert id(cell1) == id(cell2), "Cache Identity failure"

    # Now test a different coordinate
    cell3 = registry.get_or_create(1, 1, 1, state)
    assert cell1 is not cell3, "Registry error: Different coordinates returned same instance"