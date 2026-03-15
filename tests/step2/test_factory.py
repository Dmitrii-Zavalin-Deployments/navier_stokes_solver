# tests/step2/test_factory.py

import numpy as np
import pytest

from src.step2.factory import clear_cell_cache, get_cell
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cell_cache()

def test_factory_wiring_integrity():
    # Setup: 4x4x4 grid (6x6x6 buffer)
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Verify Core Cell Wiring via get_cell
    # get_cell(i, j, k) where i,j,k are 0-based indices within grid bounds
    i, j, k = 2, 2, 2
    cell = get_cell(i, j, k, state)
    
    # Buffer index calculation: (i+1) + (nx+2) * ((j+1) + (ny+2) * (k+1))
    # (2+1) + 6 * ((2+1) + 6 * (2+1)) = 3 + 6 * (3 + 18) = 3 + 126 = 129
    expected_index = (i + 1) + 6 * ((j + 1) + 6 * (k + 1))
    assert cell.index == expected_index
    assert cell.is_ghost is False
    assert cell.mask == int(state.mask.mask[i, j, k])

    # 2. Verify Ghost Cell Wiring via get_cell
    # Ghost cell (e.g., outside bounds)
    gi, gj, gk = -1, 2, 2
    ghost_cell = get_cell(gi, gj, gk, state)
    
    # Expected: -1 + 1 = 0
    expected_ghost_index = (gi + 1) + 6 * ((gj + 1) + 6 * (gk + 1))
    assert ghost_cell.index == expected_ghost_index
    assert ghost_cell.is_ghost is True
    assert ghost_cell.mask == 0 
    
    # 3. Verify Memory Persistence
    cell.vx = 0.999
    # The 'vx' property in Cell must write to the underlying buffer
    assert state.fields.data[cell.index, 0] == 0.999 

def test_mask_round_trip_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    test_mask = np.ones((nx, ny, nz), dtype=int)
    test_mask[0, 0, 0] = 0
    state.mask.mask = test_mask
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell = get_cell(i, j, k, state)
                assert cell.mask == test_mask[i, j, k], f"Mask mismatch at ({i}, {j}, {k})"

def test_exhaustive_field_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    init = state.initial_conditions
    
    # Check bounds covering ghosts and core
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

def test_cache_hit_optimization():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # First access - should be a MISS
    cell1 = get_cell(2, 2, 2, state)
    
    # Second access - should be a HIT
    cell2 = get_cell(2, 2, 2, state)
    
    assert cell1 is cell2  # Ensure object identity
    assert id(cell1) == id(cell2)