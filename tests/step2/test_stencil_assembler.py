# tests/step2/test_stencil_assembler.py

import numpy as np
import pytest

from src.step2.stencil_assembler import assemble_stencil_matrix
from src.step2.factory import clear_cell_cache
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

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
    
    # 2. Physics Param Verification (Check the math)
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
    print("\nDEBUG: Flyweight pattern verified - Cell instances are shared.")