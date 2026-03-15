# tests/step2/test_factory.py

import pytest
import numpy as np
from src.step2.factory import build_core_cell, build_ghost_cell
from tests.helpers.solver_step1_output_dummy import make_step1_output_dummy

def test_factory_wiring_integrity():
    # Setup: 4x4x4 grid (6x6x6 buffer)
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Verify Core Cell Wiring
    # Pick an arbitrary interior cell: i=2, j=2, k=2 (1-based buffer index)
    i, j, k = 2, 2, 2
    cell = build_core_cell(i, j, k, state)
    
    # Check index: 2 + 6*(2 + 6*2) = 2 + 6*(14) = 86
    expected_index = 2 + 6 * (2 + 6 * 2)
    assert cell.index == expected_index
    assert cell.is_ghost is False
    # Check mask mapping: should map to physical mask [1, 1, 1]
    assert cell.mask == int(state.mask.mask[i-1, j-1, k-1])

    # 2. Verify Ghost Cell Wiring
    # Pick a ghost cell at the edge: i=0, j=2, k=2
    gi, gj, gk = 0, 2, 2
    ghost_cell = build_ghost_cell(gi, gj, gk, state)
    
    # Check index: 0 + 6*(2 + 6*2) = 84
    expected_ghost_index = gi + 6 * (gj + 6 * gk)
    assert ghost_cell.index == expected_ghost_index
    assert ghost_cell.is_ghost is True
    assert ghost_cell.mask == 0  # Should be isolated
    
    # 3. Verify Memory Persistence
    # Ensure that writing to the cell actually changes the state.fields.data
    cell.vx = 0.999
    assert state.fields.data[cell.index, 0] == 0.999 # Assuming index 0 is VX