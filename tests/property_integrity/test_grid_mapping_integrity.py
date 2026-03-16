# tests/property_integrity/test_grid_mapping_integrity.py

import numpy as np

from src.common.grid_math import get_coords_from_index, get_flat_index
from src.step1.helpers import generate_3d_masks


class MockGrid:
    def __init__(self, nx, ny, nz):
        self.nx, self.ny, self.nz = nx, ny, nz

def test_mask_mapping_integrity_roundtrip():
    """
    Verifies that interior mask data maintains absolute identity
    when transitioned from a flat list to 3D interior, and finally
    when mapped into a buffered foundation (halo) array.
    """
    # 1. Arrange: Create a 3x3x3 interior grid
    nx, ny, nz = 3, 3, 3
    # Interior: 1=Fluid, -1=Boundary, 0=Empty
    flat_mask = [1, 0, -1] * 9  # 27 elements
    grid = MockGrid(nx, ny, nz)

    # 2. Act: Generate 3D interior mask
    mask_3d, _, _ = generate_3d_masks(flat_mask, grid)
    
    # Verify interior mapping at random coordinates
    test_coords = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    for (i, j, k) in test_coords:
        expected_val = flat_mask[i + nx * (j + ny * k)]
        assert mask_3d[i, j, k] == expected_val, f"Interior mismatch at {i,j,k}"

    # 3. Act: Map to Buffered Foundation (Halo Offset 1)
    # The Foundation includes ghosts, so dimensions become (nx+2, ny+2, nz+2)
    # The offset=1 maps interior (0,0,0) to buffered (1,1,1)
    buf_nx, buf_ny = nx + 2, ny + 2
    
    # Simulate a foundation buffer (monolithic)
    foundation_buffer = np.zeros((nx+2) * (ny+2) * (nz+2))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate flat index in the buffered system
                buf_idx = get_flat_index(i, j, k, buf_nx, buf_ny, offset=1)
                foundation_buffer[buf_idx] = mask_3d[i, j, k]

    # 4. Assert: Verify the "Round Trip" identity
    for (i, j, k) in test_coords:
        # Find index in buffered array
        idx_in_buffer = get_flat_index(i, j, k, buf_nx, buf_ny, offset=1)
        
        # Verify value matches the original mask_3d
        assert foundation_buffer[idx_in_buffer] == mask_3d[i, j, k], \
            f"Drift detected: Buffer value {foundation_buffer[idx_in_buffer]} != Original {mask_3d[i, j, k]}"
            
        # Verify inverse math returns the original interior coordinate
        # Note: We pass offset=1 because the buffer has ghosts
        coords_back = get_coords_from_index(idx_in_buffer, buf_nx, buf_ny, offset=1)
        assert coords_back == (i, j, k), f"Coordinate drift at {i,j,k}"

    print("\n✅ Integrity Check: Mapping passed all round-trip validations.")