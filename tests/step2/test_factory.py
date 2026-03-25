# tests/step2/test_factory.py

import pytest
import numpy as np
from src.step2.factory import get_cell
from src.common.grid_math import get_flat_index
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_factory_topology_zones():
    """
    Exhaustive validation of the 3-Zone Topology: Core, Ghost, and Padding.
    Compliance: Rule 7.1 (Two-Tier Topology validation).
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Core Zone [0, nx-1]
    core_cell = get_cell(0, 0, 0, state)
    assert core_cell.is_ghost is False
    
    # 2. Ghost Zone [-1, nx] inclusive
    ghost_cell_min = get_cell(-1, 0, 0, state)
    ghost_cell_max = get_cell(nx, ny-1, nz-1, state)
    assert ghost_cell_min.is_ghost is True
    assert ghost_cell_max.is_ghost is True
    
    # 3. Padding Zone (Illegal Territory: < -1 or > nx)
    with pytest.raises(IndexError, match=r"\[FACTORY\] Out-of-bounds"):
        get_cell(-2, 0, 0, state)
    with pytest.raises(IndexError, match=r"\[FACTORY\] Out-of-bounds"):
        get_cell(nx + 1, 0, 0, state)

def test_factory_wiring_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Target physical coordinate
    i, j, k = 2, 2, 2
    cell = get_cell(i, j, k, state)
    
    # 1. Verify buffer dimensions: Core(4) + Padding(2) = 6
    assert cell.nx_buf == nx + 2
    assert cell.ny_buf == ny + 2
    
    # 2. Derive expected index using SSoT grid_math
    # Factory logic: index = get_flat_index(i+1, j+1, k+1, nx+2, ny+2)
    expected_index = get_flat_index(i + 1, j + 1, k + 1, nx + 2, ny + 2)

    assert cell.index == expected_index
    assert cell.is_ghost is False
    # Ensure mask is pulled correctly from the 3D mask grid
    assert cell.mask == int(state.mask.mask[i, j, k])

    # 3. Verify Memory Persistence (View-based updates)
    cell.vx = 0.999
    # FieldManager data is flat; check if the flat index in the buffer updated
    assert state.fields.data[cell.index, 0] == 0.999

def test_exhaustive_field_integrity():
    """Verifies that Core cells get ICs and Ghost cells get GHOST_VELOCITY."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    init_v = state.initial_conditions.velocity[0]
    init_p = state.initial_conditions.pressure
    
    for k in range(-1, nz + 1):
        for j in range(-1, ny + 1):
            for i in range(-1, nx + 1):
                cell = get_cell(i, j, k, state)
                
                if cell.is_ghost:
                    # Referencing factory.py GHOST constants
                    assert cell.vx == 0.0 and cell.p == 0.0
                    assert cell.mask == 0
                else:
                    assert cell.vx == init_v
                    assert cell.p == init_p
                    assert cell.mask == int(state.mask.mask[i, j, k])

def test_factory_allocation_behavior():
    """Cells should be transient flyweight instances (not cached by factory)."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    cell1 = get_cell(2, 2, 2, state)
    cell2 = get_cell(2, 2, 2, state)
    assert cell1 is not cell2 

def test_variable_grid_dimension_integrity():
    """Validates indexing on non-cubic grids."""
    nx, ny, nz = 8, 4, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    i, j, k = 5, 3, 1
    cell = get_cell(i, j, k, state)
    
    nx_buf, ny_buf = nx + 2, ny + 2
    expected_index = get_flat_index(i + 1, j + 1, k + 1, nx_buf, ny_buf)
    assert cell.index == expected_index

def test_factory_internal_obstacle_sync():
    """Rule 4 Compliance: SSoT Synchronization between Mask and Cell."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Create a "Wall" obstacle (0) at physical coord
    state.mask.mask[2, 2, 2] = 0 
    
    # 2. Retrieve cell
    obstacle_cell = get_cell(2, 2, 2, state)

    # 3. Verification: Factory must look at [i, j, k] not [i+1, j+1, k+1] for masks
    assert obstacle_cell.mask == 0

def test_factory_mask_alignment_drift_simulation():
    """Proves that physical coords map to the correct mask index."""
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    target_coord = (1, 2, 3)
    state.mask.mask[target_coord] = 0 # Unique Obstacle
    
    cell = get_cell(*target_coord, state)
    assert cell.mask == 0, "Factory failed to align physical coord with mask grid."