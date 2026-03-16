import pytest
import numpy as np
from src.step2.factory import get_cell
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_factory_topology_zones():
    """
    Exhaustive validation of the 3-Zone Topology: Core, Ghost, and Padding.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Core Zone [0, nx-1]
    core_cell = get_cell(0, 0, 0, state)
    assert core_cell.is_ghost is False
    
    # 2. Ghost Zone [-1, nx] (specifically the boundaries)
    ghost_cell_min = get_cell(-1, 0, 0, state)
    ghost_cell_max = get_cell(nx, ny-1, nz-1, state)
    assert ghost_cell_min.is_ghost is True
    assert ghost_cell_max.is_ghost is True
    
    # 3. Padding Zone (Illegal Territory)
    # Testing boundaries immediately outside the [-1, nx] Ghost Layer
    with pytest.raises(IndexError, match="\[FACTORY\] Out-of-bounds"):
        get_cell(-2, 0, 0, state)
    with pytest.raises(IndexError, match="\[FACTORY\] Out-of-bounds"):
        get_cell(nx + 1, 0, 0, state)

def test_factory_wiring_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Verify Core Wiring
    i, j, k = 2, 2, 2
    cell = get_cell(i, j, k, state)
    
    nx_buf, ny_buf = nx + 2, ny + 2
    expected_index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    assert cell.index == expected_index
    assert cell.is_ghost is False
    assert cell.mask == int(state.mask.mask[i, j, k])

    # Verify Memory Persistence
    cell.vx = 0.999
    assert state.fields.data[cell.index, 0] == 0.999 

def test_exhaustive_field_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    init = state.initial_conditions
    
    # Test across the full valid topological range [-1, nx]
    for k in range(-1, nz + 1):
        for j in range(-1, ny + 1):
            for i in range(-1, nx + 1):
                cell = get_cell(i, j, k, state)
                
                if cell.is_ghost:
                    assert cell.vx == 0.0 and cell.p == 0.0
                    assert cell.mask == 0
                else:
                    assert cell.vx == init.velocity[0]
                    assert cell.p == init.pressure
                    assert cell.mask == int(state.mask.mask[i, j, k])

def test_factory_allocation_behavior():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Ensure Factory returns distinct objects (pure allocator behavior)
    cell1 = get_cell(2, 2, 2, state)
    cell2 = get_cell(2, 2, 2, state)
    assert cell1 is not cell2 

def test_variable_grid_dimension_integrity():
    nx, ny, nz = 8, 4, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    i, j, k = 5, 3, 1
    cell = get_cell(i, j, k, state)
    
    expected_index = (i + 1) + (nx + 2) * ((j + 1) + (ny + 2) * (k + 1))
    assert cell.index == expected_index