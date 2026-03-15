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
    
    # 1. Verify Core Cell Wiring
    i, j, k = 2, 2, 2
    cell = get_cell(i, j, k, state)
    
    # Verify index and derived coordinates
    nx_buf, ny_buf = nx + 2, ny + 2
    expected_index = (i + 1) + nx_buf * ((j + 1) + ny_buf * (k + 1))
    
    assert cell.index == expected_index
    assert cell.i == i
    assert cell.j == j
    assert cell.k == k
    assert cell.is_ghost is False
    assert cell.mask == int(state.mask.mask[i, j, k])

    # 2. Verify Ghost Cell Wiring
    gi, gj, gk = -1, 2, 2
    ghost_cell = get_cell(gi, gj, gk, state)
    
    assert ghost_cell.i == gi
    assert ghost_cell.j == gj
    assert ghost_cell.k == gk
    assert ghost_cell.is_ghost is True
    assert ghost_cell.mask == 0 
    
    # 3. Verify Memory Persistence
    cell.vx = 0.999
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
                assert cell.mask == test_mask[i, j, k]
                assert cell.i == i and cell.j == j and cell.k == k

def test_exhaustive_field_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    init = state.initial_conditions
    
    for k in range(-1, nz + 1):
        for j in range(-1, ny + 1):
            for i in range(-1, nx + 1):
                cell = get_cell(i, j, k, state)
                
                # Verify coordinate derivation consistency
                assert cell.i == i and cell.j == j and cell.k == k
                
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
    
    cell1 = get_cell(2, 2, 2, state)
    cell2 = get_cell(2, 2, 2, state)
    
    assert cell1 is cell2
    assert cell1.i == 2 and cell1.j == 2 and cell1.k == 2