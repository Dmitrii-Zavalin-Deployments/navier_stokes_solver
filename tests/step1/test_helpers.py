# tests/step1/test_helpers.py

import numpy as np
import pytest

from src.common.solver_input import GridInput
from src.step1.helpers import generate_3d_masks


def test_generate_3d_masks_basic():
    """Verify that a simple 2x2x2 mask maps correctly."""
    grid = GridInput(nx=2, ny=2, nz=2, dx=1.0, dy=1.0, dz=1.0)
    # Flat: [1, 0, -1, 1, 0, -1, 1, 0]
    # Indices: 0 1  2  3  4  5  6  7
    mask_data = [1, 0, -1, 1, 0, -1, 1, 0]
    mask_3d, is_fluid, is_boundary = generate_3d_masks(mask_data, grid)
    
    # Expected:
    # (0,0,0) -> 1, (1,0,0) -> 0, (0,1,0) -> -1, (1,1,0) -> 1
    # (0,0,1) -> 0, (1,0,1) -> -1, (0,1,1) -> 1, (1,1,1) -> 0
    assert mask_3d[0, 0, 0] == 1
    assert mask_3d[1, 1, 0] == 1
    assert mask_3d[0, 1, 0] == -1
    assert mask_3d[1, 0, 1] == -1
    assert np.sum(is_fluid) == 3

def test_generate_3d_masks_empty():
    """Verify behavior with an empty mask list."""
    grid = GridInput(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(ValueError):
        generate_3d_masks([], grid)

def test_generate_3d_masks_overflow():
    """Verify that providing too much data raises a ValueError."""
    grid = GridInput(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    # Only 1 cell expected, providing 2
    with pytest.raises(ValueError):
        generate_3d_masks([1, 0], grid)

def test_mapping_integrity_sequence():
    """Verify that indices map to the correct 3D coordinates sequentially."""
    nx, ny, nz = 3, 2, 2
    grid = GridInput(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)
    
    # Create data where value == index
    mask_data = list(range(nx * ny * nz))
    mask_3d, _, _ = generate_3d_masks(mask_data, grid)
    
    # Check specific corners
    assert mask_3d[0, 0, 0] == 0
    assert mask_3d[nx-1, ny-1, nz-1] == (nx * ny * nz) - 1

def test_different_dimensions():
    """Test non-cubic grid dimensions."""
    grid = GridInput(nx=4, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
    mask_data = [1, 2, 3, 4, 5, 6, 7, 8]
    mask_3d, _, _ = generate_3d_masks(mask_data, grid)
    
    # 4x2x1 grid, last element index 7 should be at (3, 1, 0)
    assert mask_3d[3, 1, 0] == 8