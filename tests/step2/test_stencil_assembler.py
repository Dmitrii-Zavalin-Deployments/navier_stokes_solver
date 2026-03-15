# tests/step2/test_stencil_assembler.py

import numpy as np
import pytest

from src.step2.stencil_assembler import assemble_stencil_matrix, registry
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure a clean registry state before every test."""
    registry.clear()
    yield

def test_stencil_assembly_logic():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_list = assemble_stencil_matrix(state)
    
    assert len(stencil_list) == nx * ny * nz
    
    sample_block = stencil_list[0]
    assert sample_block.dx == 0.25
    
    # Verify Boundary Analysis
    block_000 = stencil_list[0]
    assert block_000.center.is_ghost is False
    assert block_000.i_minus.is_ghost is True

def test_stencil_physics_consistency():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
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
    
    wrong_fields = np.zeros((state.fields.data.shape[0], 99))
    state.fields.data = wrong_fields
    
    with pytest.raises(RuntimeError, match="Foundation Mismatch"):
        assemble_stencil_matrix(state)

def test_stencil_caching_efficiency():
    nx, ny, nz = 2, 2, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_list = assemble_stencil_matrix(state)
    
    # Registry ensures topological identity via the same registry instance
    block = stencil_list[0]          # (0,0,0)
    right_neighbor = stencil_list[1] # (1,0,0)
    
    assert block.i_plus is right_neighbor.center, "Identity failure: Registry is not returning the same instance"

def test_stencil_matrix_topology():
    nx, ny, nz = 4, 4, 4
    state = make_step2_output_dummy(nx=nx, ny=ny, nz=nz)
    
    stencil_matrix = assemble_stencil_matrix(state)
    matrix_3d = {(b.center.i, b.center.j, b.center.k): b for b in stencil_matrix}

    for (i, j, k), block in matrix_3d.items():
        if i + 1 < nx:
            assert block.i_plus is matrix_3d[(i + 1, j, k)].center
        if j + 1 < ny:
            assert block.j_plus is matrix_3d[(i, j + 1, k)].center
        if k + 1 < nz:
            assert block.k_plus is matrix_3d[(i, j, k + 1)].center
            
        assert block.center.index == ( (i + 1) + (nx + 2) * ((j + 1) + (ny + 2) * (k + 1)) )