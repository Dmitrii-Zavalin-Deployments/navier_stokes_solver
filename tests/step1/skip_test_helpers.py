# tests/step1/test_helpers.py

import numpy as np
import pytest

from src.common.solver_input import GridInput
from src.step1.helpers import generate_3d_masks


def create_test_grid(nx, ny, nz):
    grid = GridInput()
    
    # 1. Set the grid resolution
    grid.nx = nx
    grid.ny = ny
    grid.nz = nz
    
    # 2. Set the physical bounds (this automatically defines dx, dy, dz)
    # Assuming a domain of [0, 1] for each axis for test purposes
    grid.x_min, grid.x_max = 0.0, float(nx)
    grid.y_min, grid.y_max = 0.0, float(ny)
    grid.z_min, grid.z_max = 0.0, float(nz)
    
    return grid

def test_generate_3d_masks_basic():
    """Verify that a simple 2x2x2 mask maps correctly."""
    grid = create_test_grid(2, 2, 2)
    # Flat: [1, 0, -1, 1, 0, -1, 1, 0]
    mask_data = [1, 0, -1, 1, 0, -1, 1, 0]
    mask_3d, is_fluid, is_boundary = generate_3d_masks(mask_data, grid)
    
    assert mask_3d[0, 0, 0] == 1
    assert mask_3d[1, 1, 0] == 1
    assert mask_3d[0, 1, 0] == -1
    assert mask_3d[1, 0, 1] == -1
    assert np.sum(is_fluid) == 3

def test_generate_3d_masks_empty():
    """Verify behavior with an empty mask list."""
    grid = create_test_grid(1, 1, 1)
    with pytest.raises(ValueError):
        generate_3d_masks([], grid)

def test_generate_3d_masks_overflow():
    """Verify that providing too much data raises a ValueError."""
    grid = create_test_grid(1, 1, 1)
    # Only 1 cell expected, providing 2
    with pytest.raises(ValueError):
        generate_3d_masks([1, 0], grid)

def test_mapping_integrity_sequence():
    """Verify that indices map to the correct 3D coordinates sequentially."""
    nx, ny, nz = 3, 2, 2
    grid = create_test_grid(nx, ny, nz)
    
    # Create data where value == index
    mask_data = list(range(nx * ny * nz))
    mask_3d, _, _ = generate_3d_masks(mask_data, grid)
    
    # Check specific corners
    assert mask_3d[0, 0, 0] == 0
    assert mask_3d[nx-1, ny-1, nz-1] == (nx * ny * nz) - 1

def test_different_dimensions():
    """Test non-cubic grid dimensions."""
    grid = create_test_grid(4, 2, 1)
    mask_data = [1, 2, 3, 4, 5, 6, 7, 8]
    mask_3d, _, _ = generate_3d_masks(mask_data, grid)
    
    # 4x2x1 grid, last element index 7 should be at (3, 1, 0)
    assert mask_3d[3, 1, 0] == 8