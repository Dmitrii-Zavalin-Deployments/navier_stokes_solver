# tests/step2/test_factory.py

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

def test_mask_round_trip_integrity():
    # Setup: 4x4x4 grid
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    
    # 1. Create a custom test mask (different from standard 1s)
    # We'll put a 0 at [0, 0, 0] and a 1 everywhere else
    test_mask = np.ones((nx, ny, nz), dtype=int)
    test_mask[0, 0, 0] = 0
    state.mask.mask = test_mask
    
    # 2. Iterate through all core cells and verify mapping
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                cell = build_core_cell(i, j, k, state)
                
                # Check: Buffer-indexed cell should match source mask index
                expected = test_mask[i-1, j-1, k-1]
                assert cell.mask == expected, f"Mask mismatch at buffer ({i}, {j}, {k})"
    
    print("DEBUG: Topology Round-Trip Integrity Verified.")

def test_exhaustive_field_integrity():
    nx, ny, nz = 4, 4, 4
    state = make_step1_output_dummy(nx=nx, ny=ny, nz=nz)
    init = state.initial_conditions
    
    # Iterate through the entire buffer space (0 to nx+1)
    for k in range(nz + 2):
        for j in range(ny + 2):
            for i in range(nx + 2):
                
                is_ghost = (i == 0 or i == nx + 1 or 
                            j == 0 or j == ny + 1 or 
                            k == 0 or k == nz + 1)
                
                if is_ghost:
                    cell = build_ghost_cell(i, j, k, state)
                    # Verify Ghost Field Init (Rule 5: Explicit Zeroing)
                    assert cell.vx == 0.0 and cell.vy == 0.0 and cell.vz == 0.0
                    assert cell.p == 0.0
                    assert cell.mask == 0
                else:
                    cell = build_core_cell(i, j, k, state)
                    # Verify Core Field Init (Rule 9: Source-of-Truth Injection)
                    # We expect these to match the 'init' values provided to the dummy
                    vx_init, vy_init, vz_init = init.velocity
                    assert cell.vx == vx_init
                    assert cell.vy == vy_init
                    assert cell.vz == vz_init
                    assert cell.p == init.pressure
                    assert cell.mask == int(state.mask.mask[i-1, j-1, k-1])

    print("\nDEBUG: Exhaustive Field Integrity Passed.")