import numpy as np

from src.step3.core.grid_utils import get_interior_slices


def test_get_interior_slices_structure():
    """Verify the function returns the correct slice tuple for 3D grids."""
    slices = get_interior_slices()
    
    assert isinstance(slices, tuple)
    assert len(slices) == 3
    for s in slices:
        assert isinstance(s, slice)
        assert s.start == 1
        assert s.stop == -1

def test_get_interior_slices_application():
    """Verify that the slices correctly mask the boundary of a test array."""
    # Create a 4x4x4 array
    data = np.zeros((4, 4, 4))
    
    # Apply slices
    interior_indices = get_interior_slices()
    data[interior_indices] = 1.0
    
    # In a 4x4x4 array, the interior (excluding 1st and last) is a 2x2x2 block
    # Check that the interior is 1.0 and boundaries are 0.0
    assert np.sum(data) == 8.0  # 2*2*2 = 8
    assert data[0, :, :].sum() == 0.0  # Boundary layer
    assert data[-1, :, :].sum() == 0.0 # Boundary layer
    assert data[1:3, 1:3, 1:3].sum() == 8.0 # Confirmed interior