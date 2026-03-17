# tests/step2/test_factory.py

import pytest

from src.step2.factory import get_cell, get_flat_index
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
    
    # 2. Ghost Zone [-1, nx] 
    # Note: Valid range is [-1, nx] inclusive for ghosts
    ghost_cell_min = get_cell(-1, 0, 0, state)
    ghost_cell_max = get_cell(nx, ny-1, nz-1, state)
    assert ghost_cell_min.is_ghost is True
    assert ghost_cell_max.is_ghost is True
    
    # 3. Padding Zone (Illegal Territory)
    with pytest.raises(IndexError, match=r"\[FACTORY\] Out-of-bounds"):
        get_cell(-2, 0, 0, state)
    with pytest.raises(IndexError, match=r"\[FACTORY\] Out-of-bounds"):
        get_cell(nx + 1, 0, 0, state)

def test_factory_wiring_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # Target coordinate
    i, j, k = 2, 2, 2
    cell = get_cell(i, j, k, state)
    
    # 1. Assert factory knows its own dimensions (this validates your assumption of 6x6x6)
    assert cell.nx_buf == nx + 2, f"Expected nx_buf=6, got {cell.nx_buf}"
    assert cell.ny_buf == ny + 2, f"Expected ny_buf=6, got {cell.ny_buf}"
    
    # 2. Derive expected values based on validated dimensions
    nx_buf, ny_buf = cell.nx_buf, cell.ny_buf
    i_buf, j_buf, k_buf = i + 1, j + 1, k + 1
    
    # 3. Calculate expected index
    expected_index = i_buf + (nx_buf * j_buf) + (nx_buf * ny_buf * k_buf)

    # 4. Diagnostic print if it fails
    if cell.index != expected_index:
        print(f"\nDEBUG: i={i}, j={j}, k={k}")
        print(f"DEBUG: nx_buf={nx_buf}, ny_buf={ny_buf}")
        print(f"DEBUG: i_buf={i_buf}, j_buf={j_buf}, k_buf={k_buf}")
        print(f"DEBUG: Calculated expected_index={expected_index}")
        print(f"DEBUG: Actual cell.index={cell.index}")

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
    # Range is inclusive for boundaries: [-1, nx]
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
    
    cell1 = get_cell(2, 2, 2, state)
    cell2 = get_cell(2, 2, 2, state)
    assert cell1 is not cell2 

def test_variable_grid_dimension_integrity():
    nx, ny, nz = 8, 4, 2
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    i, j, k = 5, 3, 1
    cell = get_cell(i, j, k, state)
    
    # Updated to reflect explicit buffer shift logic
    nx_buf, ny_buf = nx + 2, ny + 2
    expected_index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    assert cell.index == expected_index

def test_factory_index_calculation_logic():
    # 1. Define only what is actually used
    nx, ny = 4, 4
    nx_buf, ny_buf = nx + 2, ny + 2
    
    # 2. Define the specific coordinate (i, j, k) being tested
    i, j, k = 2, 2, 2
    
    # 3. Define the EXPECTED padded/buffered coordinates
    # The factory should be shifting these internally to account for ghost cells
    i_buf, j_buf, k_buf = i + 1, j + 1, k + 1
    
    # 4. Define the expected result (129 in this case)
    expected_index = i_buf + (nx_buf * j_buf) + (nx_buf * ny_buf * k_buf)
    
    # 5. Perform the check
    # Capture the actual result from the Factory's specific logic
    actual_index = get_flat_index(i_buf, j_buf, k_buf, nx_buf, ny_buf)
    
    # Assertions
    assert i_buf == 3, f"Buffer padding offset error: Expected i_buf=3, got {i_buf}"
    assert actual_index == expected_index, f"Factory indexing error: Expected {expected_index}, got {actual_index}"
    
    print(f"DEBUG: Successfully verified Factory index {actual_index} for coord ({i_buf}, {j_buf}, {k_buf})")

def test_factory_internal_obstacle_sync():
    """
    SRED Compliance Rule 4: SSoT Synchronization.
    Verifies that internal obstacles (non-fluid mask values) in the state
    are correctly mapped to the allocated Cell instances.
    """
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)

    # 1. Create a "Wall" obstacle at physical coordinate (2, 2, 2)
    # 1 = Fluid, 0 = Obstacle/Wall
    state.mask.mask[2, 2, 2] = 0 
    
    # 2. Create a "Fluid" cell at (1, 1, 1)
    state.mask.mask[1, 1, 1] = 1

    # 3. Retrieve cells from factory
    obstacle_cell = get_cell(2, 2, 2, state)
    fluid_cell = get_cell(1, 1, 1, state)

    # 4. Assertions: If the factory math is shifted, the obstacle cell 
    # will wrongly report '1' because it's looking at the wrong part of the mask.
    assert obstacle_cell.mask == 0, f"Factory failed to detect obstacle at (2,2,2)"
    assert fluid_cell.mask == 1, f"Factory wrongly marked fluid at (1,1,1) as obstacle"
    
    # 5. Boundary check: Ensure a ghost cell neighbor of the obstacle 
    # doesn't accidentally inherit the obstacle mask.
    ghost_neighbor = get_cell(-1, 2, 2, state)
    assert ghost_neighbor.mask == 0 # GHOST_MASK is 0 by definition in factory.py