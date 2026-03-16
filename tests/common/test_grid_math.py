# tests/common/test_grid_math.py

import pytest

from src.common.grid_math import get_coords_from_index, get_flat_index

# Test configuration
NX, NY, NZ = 4, 4, 4

def test_manual_mapping():
    """Verify specific known values for 3D to 1D and vice-versa."""
    # (i, j, k) = (2, 2, 2) in a 4x4x4 grid
    # index = 2 + (4 * 2) + (4 * 4 * 2) = 2 + 8 + 32 = 42
    assert get_flat_index(2, 2, 2, NX, NY) == 42
    assert get_coords_from_index(42, NX, NY) == (2, 2, 2)

def test_edge_cases():
    """Test boundaries of the 3D space."""
    # Origin
    assert get_flat_index(0, 0, 0, NX, NY) == 0
    assert get_coords_from_index(0, NX, NY) == (0, 0, 0)
    
    # Last element
    max_idx = (NX * NY * NZ) - 1
    assert get_flat_index(NX-1, NY-1, NZ-1, NX, NY) == max_idx
    assert get_coords_from_index(max_idx, NX, NY) == (NX-1, NY-1, NZ-1)

@pytest.mark.parametrize("i, j, k", [
    (0, 0, 0), (3, 3, 3), (1, 2, 3), (0, 3, 2), (2, 0, 1)
])
def test_round_trip_3d_to_1d_to_3d(i, j, k):
    """
    Set 3D -> get 1D -> get 3D back.
    Verify if the final 3D is same as the initial 3D.
    """
    flat = get_flat_index(i, j, k, NX, NY)
    coords = get_coords_from_index(flat, NX, NY)
    assert coords == (i, j, k)

@pytest.mark.parametrize("index", range(0, NX * NY * NZ, 5))
def test_round_trip_1d_to_3d_to_1d(index):
    """
    Set 1D -> get 3D -> get 1D back.
    Verify if the final 1D is same as the initial input.
    """
    coords = get_coords_from_index(index, NX, NY)
    flat = get_flat_index(*coords, NX, NY)
    assert flat == index

def test_exhaustive_range():
    """Verify that every single index in the volume maps correctly."""
    total_cells = NX * NY * NZ
    for idx in range(total_cells):
        i, j, k = get_coords_from_index(idx, NX, NY)
        assert get_flat_index(i, j, k, NX, NY) == idx