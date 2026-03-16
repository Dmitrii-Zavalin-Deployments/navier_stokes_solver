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
    """
    Validates stencil assembly by querying the SolverState as the SSoT.
    Ensures that the wiring (objects) matches the foundation (buffer).
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Assemble the matrix
    # The assembler now operates directly on the state buffer
    stencil_list = assemble_stencil_matrix(state)
    
    # 2. COMPLIANT ACCESS: Extract dimensions from the state's own grid/fields
    # instead of creating a secondary registry.
    buffer_capacity = state.fields.data.shape[0]
    print(f"Buffer capacity: {buffer_capacity}")
    
    # 3. Structural Integrity Assertion (POST Validation)
    # The stencil assembly must reside within the allocated memory foundation.
    # We allow the stencil list to be smaller than the buffer (to accommodate 
    # safety margins/padding cells), but we strictly forbid overflowing it.

    assert len(stencil_list) <= buffer_capacity, \
        f"POST OVERFLOW: Stencil count ({len(stencil_list)}) exceeds allocated Buffer size ({buffer_capacity})"

    # 4. Topology Audit
    # We use a 3D view to verify specific coordinate mappings
    matrix_3d = get_matrix_3d(stencil_list)
    
    # Verify spatial logic at the origin
    sample_block = matrix_3d[(0, 0, 0)]
    assert sample_block.dx == 0.25, f"Expected dx=0.25, got {sample_block.dx}"
    
    # Boundary Analysis: Ensure pointer-graph correctly identifies interior vs ghost
    block_000 = matrix_3d[(0, 0, 0)]
    
    # Validate coordinate logic within the object-graph
    assert block_000.center.i == 0
    assert block_000.center.is_ghost is False
    
    # Validate ghost-neighbor connection (Wiring Audit)
    assert block_000.i_minus.i == -1
    assert block_000.i_minus.is_ghost is True

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
    # Using dummy for Step 2 setup
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_matrix = assemble_stencil_matrix(state)
    matrix_3d = {(b.center.i, b.center.j, b.center.k): b for b in stencil_matrix}

    # Verify that the pointer-based graph is correctly wired
    for (i, j, k), block in matrix_3d.items():
        if i + 1 < nx:
            assert block.i_plus.index == matrix_3d[(i + 1, j, k)].center.index
        if j + 1 < ny:
            assert block.j_plus.index == matrix_3d[(i, j + 1, k)].center.index
        if k + 1 < nz:
            assert block.k_plus.index == matrix_3d[(i, j, k + 1)].center.index
            
        # Verify Flat Index calculation matches buffer index
        nx_buf, ny_buf = nx + 2, ny + 2; offset = 1
        expected_idx = (i + offset) + nx_buf * ((j + offset) + ny_buf * (k + offset))
        assert block.center.index == expected_idx

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