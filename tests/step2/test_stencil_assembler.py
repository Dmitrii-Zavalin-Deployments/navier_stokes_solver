# tests/step2/test_stencil_assembler.py

import numpy as np
import pytest

from src.step2.factory import clear_cell_cache
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


@pytest.fixture(autouse=True)
def reset_factory_cache():
    """Ensure a clean factory state before every test."""
    clear_cell_cache()
    yield

def test_stencil_assembly_logic():
    # Setup: 4x4x4 grid
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Run Assembly
    stencil_list = assemble_stencil_matrix(state)
    
    # 1. Integrity Check: Count (Should be nx * ny * nz = 64)
    assert len(stencil_list) == nx * ny * nz
    
    # 2. Physics Param Verification
    sample_block = stencil_list[0]
    assert sample_block.dx == 0.25
    assert sample_block.dy == 0.25
    assert sample_block.dz == 0.25
    
    # 3. Neighborhood Wiring Verification (Boundary Analysis)
    block_000 = stencil_list[0]
    assert block_000.center.is_ghost is False
    assert block_000.i_minus.is_ghost is True
    assert block_000.j_minus.is_ghost is True
    assert block_000.k_minus.is_ghost is True

def test_stencil_physics_consistency():
    # Test that changing simulation parameters reflects in assembled stencils
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Mutate physics
    state.simulation_parameters.time_step = 0.0123
    state.fluid_properties.density = 999.0
    state.external_forces.force_vector = np.array([0.1, 0.2, 0.3])
    
    stencil_list = assemble_stencil_matrix(state)
    
    for block in stencil_list:
        assert block.dt == 0.0123
        assert block.rho == 999.0
        assert block.f_vals == (0.1, 0.2, 0.3)

def test_schema_mismatch_raises_error():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Intentionally corrupt the fields buffer width
    wrong_fields = np.zeros((state.fields.data.shape[0], 99))
    state.fields.data = wrong_fields
    
    with pytest.raises(RuntimeError, match="Foundation Mismatch"):
        assemble_stencil_matrix(state)

def test_stencil_caching_efficiency():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_list = assemble_stencil_matrix(state)
    
    # Check flyweight pattern: Cell instances must be shared in memory (identity 'is')
    block = stencil_list[0]          # (0,0,0)
    right_neighbor = stencil_list[1] # (1,0,0)
    
    # The cell at (1,0,0) is both the i_plus neighbor of (0,0,0) and the center of (1,0,0)
    assert block.i_plus is right_neighbor.center, f"Identity failure: {id(block.i_plus)} != {id(right_neighbor.center)}"

def test_stencil_matrix_topology():
    """
    Verifies that the assembled stencil matrix maintains correct memory
    identity for neighbor cross-referencing (Flyweight/Topology Check).
    """
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Run the Assembler
    stencil_matrix = assemble_stencil_matrix(state)
    
    # Map the list to a 3D dict for coordinate verification
    matrix_3d = {(b.center.i, b.center.j, b.center.k): b for b in stencil_matrix}

    # Verify Topology Identity: 
    # Check that a cell acts as neighbor to its 6 adjacent blocks AND 
    # that those neighbor references are identical to the 'center' of those blocks.
    for (i, j, k), block in matrix_3d.items():
        # Check i-direction
        if i + 1 < nx:
            assert block.i_plus is matrix_3d[(i + 1, j, k)].center
            assert matrix_3d[(i + 1, j, k)].i_minus is block.center
        
        # Check j-direction
        if j + 1 < ny:
            assert block.j_plus is matrix_3d[(i, j + 1, k)].center
            assert matrix_3d[(i, j + 1, k)].j_minus is block.center
            
        # Check k-direction
        if k + 1 < nz:
            assert block.k_plus is matrix_3d[(i, j, k + 1)].center
            assert matrix_3d[(i, j, k + 1)].k_minus is block.center
            
        # Optional: Verify Memory Index Consistency (SSoT Check)
        assert block.center.index == ( (i + 1) + (nx + 2) * ((j + 1) + (ny + 2) * (k + 1)) ), \
            f"Index mapping failed at ({i}, {j}, {k})"